from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
try:
    from gradio import Progress as _GradioProgress
except Exception:
    _GradioProgress = None

# Resolve from this file, never from the caller's current working directory.
ROOT = Path(__file__).resolve().parent
QWEN_REPO = ROOT / "qwen3-tts"
INPUT = ROOT / "input_audio"
DATASET = ROOT / "dataset"
CLEANED = DATASET / "cleaned"
SEGMENTS = DATASET / "segments"
REJECTED = DATASET / "rejected"
REFERENCES = DATASET / "references"
REPORTS = DATASET / "reports"
TRAINING = ROOT / "training"
GENERATED = ROOT / "generated"
LOGS = ROOT / "logs"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
CONFIG = ROOT / "config"
REVIEW_FILE = DATASET / "review.json"
VOICES = ROOT / "voices"
VOICE_STATE = CONFIG / "voice_state.json"
MODEL_17B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_06B = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
ASR_MODEL = os.environ.get("VOICE_ASR_MODEL", "small")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".webm", ".mp4"}
SEED = 20260826
TRAIN_PROCESS: subprocess.Popen[str] | None = None
TRAIN_OWNER: str | None = None
ASR_INSTANCE: Any = None
ASR_LOCK = threading.Lock()
TTS_CACHE: dict[str, Any] = {}

for p in (INPUT, DATASET, CLEANED, SEGMENTS, REJECTED, REFERENCES, REPORTS,
          TRAINING / "runs", GENERATED, LOGS, CACHE, MODELS, CONFIG):
    p.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE / "transformers"))
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

logging.basicConfig(
    filename=str(LOGS / "app.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)
LOGGER = logging.getLogger("ia_de_voz")


class ProfileError(ValueError):
    """A profile or speaker-artifact boundary could not be validated."""


def validate_voice_id(voice_id: str) -> str:
    if not isinstance(voice_id, str) or not re.fullmatch(r"[a-z0-9]{32}", voice_id):
        raise ProfileError("voice_id inválido")
    return voice_id


def _is_reparse(path: Path) -> bool:
    try:
        attrs = os.lstat(path).st_file_attributes
        return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except (AttributeError, OSError):
        return path.is_symlink()


def _inside(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def _assert_no_reparse_chain(path: Path) -> None:
    current = Path(path)
    while True:
        if os.path.lexists(current) and _is_reparse(current):
            raise ProfileError(f"reparse point rejeitado: {current}")
        if current.parent == current:
            return
        current = current.parent


def _check_no_reparse(root: Path, target: Path) -> None:
    _assert_no_reparse_chain(target)
    current = target
    while True:
        if current.exists() and _is_reparse(current):
            raise ProfileError(f"reparse point rejeitado: {current}")
        if current == root:
            return
        if current.parent == current:
            raise ProfileError("caminho fora do perfil")
        current = current.parent


def _relative_profile_path(ctx: "ProfileContext", value: str, *, must_exist: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProfileError("caminho de perfil inválido")
    _assert_no_reparse_chain(ctx.root)
    win = PureWindowsPath(value)
    if win.is_absolute() or win.drive or value.startswith(("\\", "/")):
        raise ProfileError("caminho absoluto não permitido para artefato de voz")
    parts = [p for p in win.parts if p not in ("", ".")]
    if any(p == ".." or ":" in p for p in parts):
        raise ProfileError("travessia de caminho rejeitada")
    target = (ctx.root.joinpath(*parts)).resolve(strict=False)
    root = ctx.root.resolve(strict=True)
    if not _inside(root, target) or target == root:
        raise ProfileError("caminho fora do perfil")
    _check_no_reparse(root, target)
    if must_exist and not target.is_file():
        raise ProfileError(f"artefato ausente: {value}")
    return target


@dataclass(frozen=True)
class ProfileContext:
    voice_id: str
    root: Path

    def __post_init__(self) -> None:
        validate_voice_id(self.voice_id)
        if self.root.name != self.voice_id:
            raise ProfileError("raiz do perfil não corresponde ao voice_id")

    @property
    def profile_json(self) -> Path: return self.root / "profile.json"
    @property
    def input_audio(self) -> Path: return self.root / "input_audio"
    @property
    def dataset(self) -> Path: return self.root / "dataset"
    @property
    def cleaned(self) -> Path: return self.dataset / "cleaned"
    @property
    def segments(self) -> Path: return self.dataset / "segments"
    @property
    def rejected(self) -> Path: return self.dataset / "rejected"
    @property
    def references(self) -> Path: return self.dataset / "references"
    @property
    def reports(self) -> Path: return self.dataset / "reports"
    @property
    def review_file(self) -> Path: return self.dataset / "review.json"
    @property
    def training(self) -> Path: return self.root / "training"
    @property
    def generated(self) -> Path: return self.root / "generated"

    def path(self, relative: str, *, must_exist: bool = False) -> Path:
        return _relative_profile_path(self, relative, must_exist=must_exist)


PROFILE_DIRS = (
    "input_audio", "dataset/original", "dataset/converted", "dataset/cleaned",
    "dataset/clips", "dataset/segments", "dataset/accepted", "dataset/rejected",
    "dataset/references", "dataset/metadata", "dataset/reports", "dataset/raw",
    "training/runs", "training/checkpoints", "evaluation", "generated",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    _assert_no_reparse_chain(path)
    _assert_no_reparse_chain(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_chain(path.parent)
    _assert_no_reparse_chain(path)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _assert_no_reparse_chain(temp)
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _assert_no_reparse_chain(temp)
        _assert_no_reparse_chain(path)
        os.replace(temp, path)
    finally:
        try:
            if temp.exists() and not _is_reparse(temp):
                temp.unlink()
        except OSError:
            pass


def _manifest(root: Path) -> tuple[list[tuple[str, int, str]], str]:
    records: list[tuple[str, int, str]] = []
    _assert_no_reparse_chain(root)
    root = root.resolve(strict=True)
    if not root.is_dir() or _is_reparse(root):
        raise ProfileError(f"raiz de manifesto inválida: {root}")

    def walk(current: Path) -> None:
        _check_no_reparse(root, current)
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                if _is_reparse(path):
                    raise ProfileError(f"reparse point rejeitado: {path}")
                if entry.is_dir(follow_symlinks=False):
                    walk(path)
                elif entry.is_file(follow_symlinks=False):
                    rel = path.relative_to(root).as_posix()
                    records.append((rel, path.stat().st_size, sha256_file(path)))

    walk(root)
    records.sort()
    text = "\n".join(f"{a}|{b}|{c}" for a, b, c in records)
    return records, hashlib.sha256(text.encode()).hexdigest()


class ProfileStore:
    def __init__(self, root: Path = VOICES, state_file: Path = VOICE_STATE, legacy_root: Path = ROOT):
        self.root = Path(root).resolve()
        self.state_file = Path(state_file).resolve()
        self.legacy_root = Path(legacy_root).resolve()

    def _context(self, voice_id: str) -> ProfileContext:
        validate_voice_id(voice_id)
        _assert_no_reparse_chain(self.root)
        expected = self.root.resolve(strict=True)
        if not expected.is_dir() or _is_reparse(expected):
            raise ProfileError("raiz de perfis inválida")
        raw_root = self.root / voice_id
        _assert_no_reparse_chain(raw_root)
        if not raw_root.exists() or _is_reparse(raw_root):
            raise ProfileError("raiz de perfil inválida")
        root = raw_root.resolve(strict=True)
        if not _inside(expected, root) or root == expected or _is_reparse(root):
            raise ProfileError("raiz de perfil inválida")
        ctx = ProfileContext(voice_id, root)
        _check_no_reparse(ctx.root, ctx.profile_json)
        if not ctx.profile_json.is_file():
            raise ProfileError("profile.json ausente")
        data = json.loads(ctx.profile_json.read_text(encoding="utf-8"))
        if data.get("voice_id") != voice_id or data.get("schema_version") != 1:
            raise ProfileError("profile.json inválido")
        return ctx

    def list_profiles(self) -> list[tuple[str, str]]:
        _assert_no_reparse_chain(self.root)
        if not self.root.is_dir():
            return []
        found: list[tuple[str, str]] = []
        for path in sorted(self.root.iterdir(), key=lambda p: p.name):
            _check_no_reparse(self.root.resolve(strict=True), path)
            if not path.is_dir() or not re.fullmatch(r"[a-z0-9]{32}", path.name):
                continue
            try:
                ctx = self._context(path.name)
                data = json.loads(ctx.profile_json.read_text(encoding="utf-8"))
                found.append((str(data.get("display_name") or path.name), path.name))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return found

    def get(self, voice_id: str | None = None) -> ProfileContext:
        selected = voice_id or self.active_id()
        if not selected:
            raise ProfileError("nenhuma voz ativa")
        return self._context(selected)

    def active_id(self) -> str | None:
        try:
            value = self._read_state()
            voice_id = value.get("active_voice_id")
            if voice_id:
                self._context(voice_id)
                return voice_id
        except (OSError, ValueError, json.JSONDecodeError, ProfileError):
            pass
        return None

    def _read_state(self) -> dict[str, Any]:
        _assert_no_reparse_chain(self.state_file)
        if not self.state_file.is_file():
            raise ProfileError("estado de voz ausente")
        value = json.loads(self.state_file.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ProfileError("estado de voz inválido")
        return value

    def select(self, voice_id: str) -> ProfileContext:
        ctx = self._context(voice_id)
        _atomic_json(self.state_file, {"schema_version": 1, "active_voice_id": voice_id})
        try:
            persisted = self._read_state()
            if persisted.get("active_voice_id") != voice_id:
                raise ProfileError("voz ativa não confirmada no disco")
            return self._context(voice_id)
        except ProfileError:
            raise
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ProfileError("voz ativa não pôde ser confirmada no disco") from exc

    def create(self, display_name: str, base_model: str = MODEL_17B) -> ProfileContext:
        name = str(display_name or "").strip()
        if not name or len(name) > 120 or any(ord(c) < 32 for c in name):
            raise ProfileError("nome de voz inválido")
        _assert_no_reparse_chain(self.root.parent)
        if self.root.exists() and (not self.root.is_dir() or _is_reparse(self.root)):
            raise ProfileError("raiz de perfis inválida")
        self.root.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_chain(self.root)
        voice_id = uuid4().hex
        profile_root = self.root / voice_id
        profile_root.mkdir()
        _assert_no_reparse_chain(profile_root)
        for rel in PROFILE_DIRS:
            directory = profile_root / rel
            _assert_no_reparse_chain(directory)
            directory.mkdir(parents=True, exist_ok=True)
            _assert_no_reparse_chain(directory)
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        _atomic_json(profile_root / "profile.json", {
            "schema_version": 1, "voice_id": voice_id, "display_name": name,
            "created_at": now, "updated_at": now, "base_model": base_model,
            "active_reference": None, "active_dataset": {},
            "training_method": "base_zero_shot", "best_checkpoint_or_adapter": None,
            "notes": None,
        })
        ctx = self._context(voice_id)
        if not self.active_id():
            self.select(voice_id)
        return ctx

    def rename(self, voice_id: str, display_name: str) -> ProfileContext:
        ctx = self._context(voice_id)
        name = str(display_name or "").strip()
        if not name or len(name) > 120 or any(ord(c) < 32 for c in name):
            raise ProfileError("nome de voz inválido")
        data = json.loads(ctx.profile_json.read_text(encoding="utf-8"))
        data["display_name"] = name
        data["updated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        _atomic_json(ctx.profile_json, data)
        try:
            persisted = self._context(voice_id)
            saved = json.loads(persisted.profile_json.read_text(encoding="utf-8"))
            if saved.get("display_name") != name:
                raise ProfileError("nome da voz não confirmado no disco")
            return persisted
        except ProfileError:
            raise
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ProfileError("nome da voz não pôde ser confirmado no disco") from exc

    def delete(self, voice_id: str, confirmation: str) -> None:
        ctx = self._context(voice_id)
        if not isinstance(confirmation, str) or confirmation.strip().casefold() != "delete":
            raise ProfileError("confirmação de exclusão inválida")
        was_active = self.active_id() == voice_id
        _assert_no_reparse_chain(ctx.root)
        _manifest(ctx.root)
        shutil.rmtree(ctx.root)
        if was_active:
            remaining = self.list_profiles()
            if remaining:
                self.select(remaining[0][1])
            else:
                _atomic_json(self.state_file, {"schema_version": 1, "active_voice_id": None})

    def _copy_tree(self, source: Path, destination: Path) -> None:
        if not source.exists():
            return
        source = source.resolve(strict=True)
        if not source.is_dir() or _is_reparse(source):
            raise ProfileError(f"origem de migração inválida: {source}")
        _assert_no_reparse_chain(source)
        _assert_no_reparse_chain(destination)

        def copy_dir(current: Path, target_dir: Path) -> None:
            _assert_no_reparse_chain(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            _assert_no_reparse_chain(target_dir)
            with os.scandir(current) as entries:
                for entry in entries:
                    source_path = Path(entry.path)
                    target = target_dir / entry.name
                    if _is_reparse(source_path):
                        raise ProfileError(f"reparse point rejeitado: {source_path}")
                    _assert_no_reparse_chain(target.parent)
                    if entry.is_dir(follow_symlinks=False):
                        copy_dir(source_path, target)
                    elif entry.is_file(follow_symlinks=False):
                        if target.exists() and _is_reparse(target):
                            raise ProfileError(f"reparse point rejeitado: {target}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, target)

        copy_dir(source, destination)

    _PATH_FIELDS = {"audio", "file", "source", "train_audio", "original", "cleaned", "segment", "ref_audio", "checkpoint"}

    def _rewrite_legacy_path(self, value: str, destination_root: Path, field: str | None = None, *, allow_missing: bool = False) -> str:
        if not isinstance(value, str) or not value:
            raise ProfileError("referência legada inválida")
        windows = PureWindowsPath(value)
        absolute = Path(value).is_absolute() or windows.is_absolute() or bool(windows.drive) or value.startswith(("\\", "/"))
        # A review row's `file` is normally a display-only basename.
        if field == "file" and not absolute and not any(sep in value for sep in ("/", "\\")):
            return value
        if windows.drive and not windows.is_absolute():
            raise ProfileError(f"referência legada absoluta inválida: {value}")
        legacy = self.legacy_root.resolve(strict=True)
        raw = Path(value) if absolute else legacy / value
        candidates = [raw]
        if not absolute and not any(sep in value for sep in ("/", "\\")):
            candidates.extend(legacy / name / value for name in ("input_audio", "dataset", "training", "generated"))
        candidate = next((p for p in candidates if p.exists()), raw).resolve(strict=False)
        if not _inside(legacy, candidate):
            raise ProfileError(f"referência legada fora da origem: {value}")
        _check_no_reparse(legacy, candidate)
        if not candidate.exists() and not allow_missing:
            raise ProfileError(f"referência legada ausente: {value}")
        rel = candidate.relative_to(legacy).as_posix()
        target = destination_root / rel
        if not _inside(destination_root.resolve(strict=True), target.resolve(strict=False)) or (not target.exists() and not allow_missing):
            raise ProfileError(f"referência legada ausente: {rel}")
        _check_no_reparse(destination_root.resolve(strict=True), target)
        return rel

    def _rewrite_metadata_value(self, value: Any, field: str | None, profile_root: Path, *, allow_missing: bool = False) -> Any:
        if isinstance(value, dict):
            return {key: self._rewrite_metadata_value(item, key, profile_root, allow_missing=allow_missing) for key, item in value.items()}
        if isinstance(value, list):
            return [self._rewrite_metadata_value(item, field, profile_root, allow_missing=allow_missing) for item in value]
        if not isinstance(value, str):
            return value
        windows = PureWindowsPath(value)
        if ".." in windows.parts:
            raise ProfileError(f"travessia de caminho rejeitada: {value}")
        absolute = Path(value).is_absolute() or windows.is_absolute() or bool(windows.drive) or value.startswith(("\\", "/"))
        pathlike = field in self._PATH_FIELDS or absolute
        if not pathlike:
            return value
        return self._rewrite_legacy_path(value, profile_root, field, allow_missing=allow_missing)

    def _rewrite_metadata(self, profile_root: Path) -> None:
        _assert_no_reparse_chain(profile_root)
        fixture_root = profile_root / "dataset" / "test_fixtures"
        for path in profile_root.rglob("*"):
            if _is_reparse(path):
                raise ProfileError(f"reparse point rejeitado: {path}")
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            if path.name == "profile.json":
                continue
            if path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                rows = [self._rewrite_metadata_value(row, None, profile_root, allow_missing=_inside(fixture_root, path)) for row in rows]
                _assert_no_reparse_chain(path.parent)
                path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                data = self._rewrite_metadata_value(data, None, profile_root, allow_missing=_inside(fixture_root, path))
                _assert_no_reparse_chain(path.parent)
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _legacy_manifest(self) -> tuple[list[tuple[str, int, str]], str]:
        records: list[tuple[str, int, str]] = []
        for name in ("input_audio", "dataset", "training", "generated"):
            source = self.legacy_root / name
            if not source.is_dir():
                continue
            for rel, size, digest in _manifest(source)[0]:
                records.append((f"{name}/{rel}", size, digest))
        text = "\n".join(f"{a}|{b}|{c}" for a, b, c in records)
        return records, hashlib.sha256(text.encode()).hexdigest()

    def migrate_legacy(self) -> tuple[ProfileContext, str]:
        source_dirs = ("input_audio", "dataset", "training", "generated")
        legacy = self.legacy_root.resolve(strict=True)
        if not legacy.is_dir() or _is_reparse(legacy):
            raise ProfileError("raiz legada inválida")
        _assert_no_reparse_chain(legacy)
        _assert_no_reparse_chain(self.root.parent)
        if self.root.exists() and (not self.root.is_dir() or _is_reparse(self.root)):
            raise ProfileError("destino de migração inválido")
        self.root.mkdir(parents=True, exist_ok=True)
        destination_root = self.root.resolve(strict=True)
        if not destination_root.is_dir() or _is_reparse(destination_root):
            raise ProfileError("destino de migração inválido")
        _assert_no_reparse_chain(destination_root)
        source_manifest, source_digest = self._legacy_manifest()
        for _, voice_id in self.list_profiles():
            try:
                data = json.loads((self.root / voice_id / "profile.json").read_text(encoding="utf-8"))
                migration = data.get("migration", {})
                if migration.get("source") == "legacy-global" and migration.get("source_manifest") == source_digest:
                    return self._context(voice_id), "Migração já concluída; manifesto idêntico."
                if migration.get("source") == "legacy-global":
                    raise ProfileError("migração legada existente com manifesto diferente")
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        voice_id = uuid4().hex
        stage = self.root / f".migration-{voice_id}"
        final = self.root / voice_id
        try:
            if stage.exists() or final.exists():
                raise ProfileError("destino de migração já existe")
            _assert_no_reparse_chain(stage.parent)
            stage.mkdir()
            _assert_no_reparse_chain(stage)
            for rel in PROFILE_DIRS:
                (stage / rel).mkdir(parents=True, exist_ok=True)
            for name in source_dirs:
                self._copy_tree(self.legacy_root / name, stage / name)
            self._rewrite_metadata(stage)
            current_manifest, current_digest = self._legacy_manifest()
            if current_digest != source_digest or current_manifest != source_manifest:
                raise ProfileError("origem legada mudou durante a migração")
            copied_manifest, _ = _manifest(stage)
            copied_by_rel = {rel: (size, digest) for rel, size, digest in copied_manifest}
            for rel, size, digest in source_manifest:
                target = stage / rel
                if not target.is_file() or (Path(rel).suffix.lower() not in {".json", ".jsonl"} and target.stat().st_size != size):
                    raise ProfileError(f"artefato não preservado: {rel}")
                if Path(rel).suffix.lower() not in {".json", ".jsonl"} and copied_by_rel.get(rel, (None, None))[1] != digest:
                    raise ProfileError(f"hash não preservado: {rel}")
            now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            _atomic_json(stage / "profile.json", {
                "schema_version": 1, "voice_id": voice_id, "display_name": "Existing Voice",
                "created_at": now, "updated_at": now, "base_model": MODEL_17B,
                "active_reference": "dataset/references/ref.wav" if (stage / "dataset/references/ref.wav").is_file() else None,
                "active_dataset": {
                    "train": "dataset/train_raw.jsonl" if (stage / "dataset/train_raw.jsonl").is_file() else None,
                    "validation": "dataset/validation_raw.jsonl" if (stage / "dataset/validation_raw.jsonl").is_file() else None,
                    "test": "dataset/test_raw.jsonl" if (stage / "dataset/test_raw.jsonl").is_file() else None,
                    "codes": "dataset/train_with_codes.jsonl" if (stage / "dataset/train_with_codes.jsonl").is_file() else None,
                },
                "training_method": "legacy_unknown", "best_checkpoint_or_adapter": None,
                "notes": "Identidade do falante não inferida.",
                "migration": {"source": "legacy-global", "source_manifest": source_digest, "source_file_count": len(source_manifest), "completed_at": now},
            })
            stage.replace(final)
            ctx = self._context(voice_id)
            self.select(voice_id)
            return ctx, f"Migração concluída sem alterar a origem ({len(source_manifest)} arquivos verificados)."
        except Exception:
            if stage.exists() and not _is_reparse(stage):
                shutil.rmtree(stage)
            raise


PROFILE_STORE = ProfileStore()


def profile_context(voice_id: str | None = None) -> ProfileContext:
    return PROFILE_STORE.get(voice_id)


def callback_context(voice_id: str | None) -> ProfileContext:
    ctx = profile_context(voice_id)
    if voice_id is not None and PROFILE_STORE.active_id() != ctx.voice_id:
        raise ProfileError("callback obsoleto: a voz ativa mudou")
    return ctx


def profile_data(ctx: ProfileContext) -> dict[str, Any]:
    _check_no_reparse(ctx.root, ctx.profile_json)
    return json.loads(ctx.profile_json.read_text(encoding="utf-8"))


def save_profile_data(ctx: ProfileContext, data: dict[str, Any]) -> None:
    data["updated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    _atomic_json(ctx.profile_json, data)
    persisted = profile_data(ctx)
    if persisted != data:
        raise ProfileError("perfil não confirmado no disco")


def profile_path(ctx: ProfileContext, value: str | Path, *, must_exist: bool = False) -> Path:
    raw = str(value)
    _assert_no_reparse_chain(ctx.root)
    windows = PureWindowsPath(raw)
    if not Path(raw).is_absolute() and not windows.is_absolute() and not windows.drive:
        return ctx.path(raw, must_exist=must_exist)
    path = Path(value)
    _assert_no_reparse_chain(path)
    resolved = path.resolve(strict=False)
    root = ctx.root.resolve(strict=True)
    if not _inside(root, resolved) or resolved == root:
        raise ProfileError("artefato fora do perfil")
    _check_no_reparse(root, resolved)
    if must_exist and not resolved.is_file():
        raise ProfileError(f"artefato ausente: {value}")
    return resolved


def profile_relative(ctx: ProfileContext, path: Path) -> str:
    path = profile_path(ctx, path)
    resolved = path.resolve(strict=False)
    root = ctx.root.resolve(strict=True)
    if not _inside(root, resolved) or resolved == root:
        raise ProfileError("artefato fora do perfil")
    _check_no_reparse(root, resolved)
    return resolved.relative_to(root).as_posix()


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or ""


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("FFmpeg não encontrado. Rode INSTALAR.bat novamente.")
    return subprocess.run([exe, *args], capture_output=True, text=True, check=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(QWEN_REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "não disponível"


def system_info() -> dict[str, str]:
    info: dict[str, str] = {
        "Windows": platform.platform(),
        "CPU": platform.processor() or "não disponível",
        "Python": sys.version.split()[0],
        "FFmpeg": ffmpeg_exe() or "ausente",
        "qwen-tts commit": git_commit(),
        "ASR": ASR_MODEL,
        "modelo": MODEL_17B,
        "fine-tuning": "SFT oficial; low-VRAM dinâmico (FP16, batch 1, acumulação)",
    }
    try:
        import psutil
        info["RAM"] = f"{psutil.virtual_memory().total / 2**30:.1f} GB"
        info["Espaço livre C"] = f"{shutil.disk_usage(ROOT).free / 2**30:.1f} GB"
    except Exception:
        pass
    try:
        import torch
        info["PyTorch"] = torch.__version__
        info["CUDA"] = str(torch.version.cuda or "CPU")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["GPU"] = props.name
            info["VRAM"] = f"{props.total_memory / 2**30:.1f} GB"
            info["Precision"] = "FP16 (GTX/compute < 8.0)" if props.major < 8 else "BF16/FP16"
        else:
            info["GPU"] = "CUDA indisponível"
            info["VRAM"] = "0"
            info["Precision"] = "FP32 CPU"
    except Exception as exc:
        info["PyTorch"] = f"erro: {exc}"
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
        info["Driver"] = smi.splitlines()[0] if smi else "não disponível"
    except Exception:
        info["Driver"] = "não disponível"
    return info


def diagnose_system() -> str:
    info = system_info()
    report = CONFIG / "environment_report.txt"
    report.write_text("\n".join(f"{k}: {v}" for k, v in info.items()) + "\n", encoding="utf-8")
    lines = ["### Diagnóstico local", ""]
    lines += [f"- **{k}:** {v}" for k, v in info.items()]
    lines += ["", "Privacidade: processamento local; nenhuma API de voz externa é usada."]
    if info.get("GPU", "").startswith("NVIDIA GeForce GTX 1060"):
        lines += ["", "⚠️ GTX 1060 detectada: BF16 e FlashAttention 2 ficam desativados. O SFT 1.7B só inicia após pre-flight de VRAM."]
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audio_files(voice_id: str | None = None) -> list[Path]:
    input_root = profile_context(voice_id).input_audio
    _check_no_reparse(input_root.parent, input_root)
    files = []
    for path in input_root.rglob("*"):
        _check_no_reparse(input_root, path)
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            if path.suffix.lower() != ".mp4" or has_audio_stream(path):
                files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def has_audio_stream(path: Path) -> bool:
    if path.suffix.lower() != ".mp4":
        return True
    exe = ffmpeg_exe()
    if not exe:
        return False
    probe = subprocess.run([exe, "-v", "error", "-i", str(path), "-map", "0:a:0", "-frames:a", "1", "-f", "null", "-"], capture_output=True, text=True)
    return probe.returncode == 0


def audio_duration(path: Path) -> float:
    try:
        import soundfile as sf
        return float(len(sf.SoundFile(path)) / sf.info(path).samplerate)
    except Exception:
        try:
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            try:
                probe = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
                match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
                return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3)) if match else 0.0
            except Exception:
                return 0.0


def audio_inventory(voice_id: str | None = None) -> str:
    try:
        ctx = profile_context(voice_id)
    except ProfileError as exc:
        return f"Nenhuma voz ativa: {exc}"
    files = audio_files(ctx.voice_id)
    total_seconds = sum(audio_duration(p) for p in files)
    for path in files:
        _check_no_reparse(ctx.input_audio, path)
    total_bytes = sum(p.stat().st_size for p in files)
    rows = [[html.escape(p.name), f"{audio_duration(p):.1f}s", f"{p.stat().st_size / 1024**2:.2f} MB"] for p in files]
    table = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    name = html.escape(profile_presentation_label(ctx.voice_id, str(profile_data(ctx).get("display_name") or ctx.voice_id)))
    monitored = html.escape(str(ctx.input_audio))
    if not files:
        return f"**Voz ativa:** {name}\n\n**Pasta monitorada:** `{monitored}`\n\nNenhum áudio encontrado nesta pasta."
    return f"**Voz ativa:** {name}\n\n**Pasta monitorada:** `{monitored}`\n\n**{len(files)} arquivos · {total_seconds/3600:.2f} h · {total_bytes/1024**2:.2f} MB**\n\n| Arquivo | Duração | Tamanho |\n|---|---:|---:|\n{table}"


def _upload_source(value: str | Path) -> Path:
    raw = str(value or "")
    if not raw or "\x00" in raw:
        raise ProfileError("arquivo de upload inválido")
    if ".." in PureWindowsPath(raw).parts:
        raise ProfileError("travessia de caminho rejeitada no upload")
    source = Path(raw)
    _assert_no_reparse_chain(source)
    if not source.is_file():
        raise ProfileError("arquivo de upload ausente ou inválido")
    source = source.resolve(strict=True)
    _assert_no_reparse_chain(source)
    if not source.is_file():
        raise ProfileError("arquivo de upload inválido")
    return source


def _upload_values(files: Any) -> list[Any]:
    if files is None or files == "":
        return []
    if isinstance(files, (str, Path, os.PathLike)):
        return [files]
    if isinstance(files, (list, tuple)):
        return list(files)
    raise ProfileError("formato de upload inválido")


def _upload_target(ctx: ProfileContext, filename: str, digest: str) -> tuple[Path, bool]:
    if not filename or filename in {".", ".."} or any(c in filename for c in ("/", "\\", ":", "\x00")):
        raise ProfileError("nome de arquivo inválido")
    if len(PureWindowsPath(filename).parts) != 1:
        raise ProfileError("caminho de upload rejeitado")
    candidates = [filename, f"{Path(filename).stem}__{digest[:12]}{Path(filename).suffix}", f"{Path(filename).stem}__{digest}{Path(filename).suffix}"]
    candidates.extend(f"{Path(filename).stem}__{digest}-{n}{Path(filename).suffix}" for n in range(2, 1001))
    for candidate in candidates:
        target = _relative_profile_path(ctx, f"input_audio/{candidate}")
        if not target.exists():
            return target, False
        if target.is_file() and sha256_file(target) == digest:
            return target, True
    raise ProfileError("não foi possível reservar um nome seguro para o upload")


def upload_audio(files: Any, voice_id: str | None = None) -> tuple[str, str]:
    """Copy Gradio uploads into the active profile without changing source files."""
    try:
        ctx = callback_context(voice_id)
        _check_no_reparse(ctx.root, ctx.input_audio)
        ctx.input_audio.mkdir(parents=True, exist_ok=True)
        _check_no_reparse(ctx.root, ctx.input_audio)
        values = _upload_values(files)
        if not values:
            return "Nenhum arquivo selecionado.", audio_inventory(ctx.voice_id)
        results: list[str] = []
        for value in values:
            try:
                callback_context(ctx.voice_id)
                source = _upload_source(value)
                if source.suffix.lower() not in AUDIO_EXTENSIONS:
                    raise ProfileError(f"extensão não suportada: {source.suffix or '(sem extensão)'}")
                if source.suffix.lower() == ".mp4" and not has_audio_stream(source):
                    raise ProfileError("MP4 rejeitado: não há uma faixa de áudio utilizável")
                digest = sha256_file(source)
                for _ in range(1001):
                    target, already_present = _upload_target(ctx, source.name, digest)
                    if already_present:
                        results.append(f"já existia: {target.name}")
                        break
                    try:
                        with source.open("rb") as source_stream, target.open("xb") as target_stream:
                            shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
                    except FileExistsError:
                        continue
                    _check_no_reparse(ctx.root, target)
                    if sha256_file(target) != digest:
                        try:
                            target.unlink()
                        except OSError:
                            pass
                        raise ProfileError(f"hash não conferido após copiar {source.name}")
                    results.append(f"copiado: {target.name}")
                    break
                else:
                    raise ProfileError("não foi possível reservar um nome seguro para o upload")
            except (OSError, ProfileError, ValueError) as exc:
                results.append(f"rejeitado ({Path(str(value)).name or 'arquivo'}): {exc}")
        callback_context(ctx.voice_id)
        name = profile_presentation_label(ctx.voice_id, str(profile_data(ctx).get("display_name") or ctx.voice_id))
        status = f"Upload para a voz {name} concluído. " + "; ".join(results)
        return status, audio_inventory(ctx.voice_id)
    except (OSError, ProfileError, ValueError, json.JSONDecodeError) as exc:
        selected = PROFILE_STORE.active_id()
        return f"Upload recusado: {exc}", audio_inventory(selected)


def voice_overview() -> str:
    active = PROFILE_STORE.active_id()
    rows = []
    for display_name, voice_id in profile_choices():
        try:
            ctx = profile_context(voice_id)
            files = audio_files(voice_id)
            duration = sum(audio_duration(path) for path in files)
            size = sum(path.stat().st_size for path in files)
            marker = "✅ Ativa" if voice_id == active else "—"
            rows.append(f"| {html.escape(display_name)} | {marker} | {len(files)} | {duration / 3600:.2f} h | {size / 1024**2:.2f} MB | `{html.escape(str(ctx.input_audio))}` |")
        except (OSError, ProfileError, ValueError, json.JSONDecodeError) as exc:
            rows.append(f"| {html.escape(display_name)} | ⚠️ indisponível | — | — | — | {html.escape(str(exc))} |")
    if not rows:
        return "Nenhuma voz criada ainda. Crie uma voz para começar."
    return "| Voz | Estado | Áudios | Duração total | Tamanho total | Pasta de áudios |\n|---|---|---:|---:|---:|---|\n" + "\n".join(rows)


def open_audio_folder(voice_id: str | None = None) -> str:
    ctx = callback_context(voice_id)
    _check_no_reparse(ctx.root, ctx.input_audio)
    ctx.input_audio.mkdir(parents=True, exist_ok=True)
    _check_no_reparse(ctx.root, ctx.input_audio)
    try:
        os.startfile(str(ctx.input_audio))
    except Exception:
        subprocess.Popen(["explorer.exe", str(ctx.input_audio)])
    return f"Explorer aberto em `{ctx.input_audio}`"


def normalise_audio(source: Path, dest: Path) -> None:
    _assert_no_reparse_chain(source)
    _assert_no_reparse_chain(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_chain(dest)
    if source.suffix.lower() == ".wav":
        try:
            import soundfile as sf
            data, sr = sf.read(source, always_2d=False)
            if sr == 16000:
                import numpy as np
                if getattr(data, "ndim", 1) > 1:
                    data = np.mean(data, axis=1)
                sf.write(dest, data, 16000, subtype="PCM_16")
                return
        except Exception:
            pass
    run_ffmpeg(["-y", "-i", str(source), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dest)])


def make_training_audio(source: Path, key: str, ctx: ProfileContext | None = None) -> Path:
    """Qwen's official TTSDataset expects the reference mel at 24 kHz."""
    profile = ctx or profile_context()
    dest = profile.dataset / "raw" / f"{key}.wav"
    _assert_no_reparse_chain(source)
    _check_no_reparse(profile.root, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _check_no_reparse(profile.root, dest)
    try:
        import soundfile as sf
        if dest.exists() and sf.info(dest).samplerate == 24000:
            return dest
    except Exception:
        pass
    run_ffmpeg(["-y", "-i", str(source), "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(dest)])
    return dest


def metrics(path: Path) -> dict[str, float]:
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=1)
    if len(data) == 0:
        raise ValueError("áudio vazio")
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data)) + 1e-12))
    frame = max(1, int(sr * 0.03))
    vals = np.array([np.sqrt(np.mean(np.square(data[i:i+frame])) + 1e-12) for i in range(0, len(data), frame)])
    floor = float(np.percentile(vals, 10))
    snr = 20 * math.log10(max(rms, 1e-6) / max(floor, 1e-6))
    silence = float(np.mean(vals < max(0.008, rms * 0.12)))
    clipping = float(np.mean(np.abs(data) >= 0.999))
    return {"duration": len(data) / sr, "rms_db": 20 * math.log10(max(rms, 1e-6)), "noise_db": 20 * math.log10(max(floor, 1e-6)), "snr_db": snr, "silence": silence, "clipping": clipping, "sample_rate": sr}


def split_segments(wav: Path, stem: str, ctx: ProfileContext | None = None) -> list[Path]:
    profile = ctx or profile_context()
    segments_root = profile.segments
    _assert_no_reparse_chain(wav)
    _check_no_reparse(profile.root, segments_root)
    segments_root.mkdir(parents=True, exist_ok=True)
    _check_no_reparse(profile.root, segments_root)
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(wav, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=1)
    frame = max(1, int(sr * 0.03))
    hop = max(1, int(sr * 0.01))
    rms = np.array([np.sqrt(np.mean(np.square(data[i:i+frame])) + 1e-12) for i in range(0, max(1, len(data)-frame), hop)])
    threshold = max(0.008, float(np.percentile(rms, 35)) * 1.6)
    active = rms > threshold
    ranges: list[tuple[int, int]] = []
    start = None
    gap = 0
    for i, on in enumerate(active):
        if on and start is None:
            start = i
            gap = 0
        elif not on and start is not None:
            gap += 1
            if gap >= 25:
                ranges.append((start * hop, (i - gap + 1) * hop + frame))
                start = None
                gap = 0
    if start is not None:
        ranges.append((start * hop, len(data)))
    if not ranges:
        ranges = [(0, len(data))]
    # Keep natural utterances but cap long recordings to 20 seconds.
    output: list[Path] = []
    index = 0
    for a, b in ranges:
        a = max(0, a - int(0.12 * sr)); b = min(len(data), b + int(0.12 * sr))
        while b - a > 20 * sr:
            cut = a + 20 * sr
            output.append(segments_root / f"{stem}_{index:04d}.wav")
            _check_no_reparse(profile.root, output[-1])
            sf.write(output[-1], data[a:cut], sr, subtype="PCM_16")
            index += 1; a = cut
        if b - a >= int(0.8 * sr):
            output.append(segments_root / f"{stem}_{index:04d}.wav")
            _check_no_reparse(profile.root, output[-1])
            sf.write(output[-1], data[a:b], sr, subtype="PCM_16")
            index += 1
    return output


def load_asr() -> Any:
    global ASR_INSTANCE
    with ASR_LOCK:
        if ASR_INSTANCE is not None:
            return ASR_INSTANCE
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # CTranslate2 cannot execute int8_float16 on Pascal (e.g. GTX 1060).
        if device == "cuda":
            compute = "int8_float16" if torch.cuda.get_device_capability(0)[0] >= 7 else "int8"
        else:
            compute = "int8"
        LOGGER.info("carregando ASR %s (%s/%s)", ASR_MODEL, device, compute)
        try:
            ASR_INSTANCE = WhisperModel(ASR_MODEL, device=device, compute_type=compute, download_root=str(CACHE / "whisper"))
        except ValueError:
            # Last safe fallback keeps the pipeline usable when a backend rejects a dtype.
            LOGGER.exception("backend ASR rejeitou %s; usando CPU int8", compute)
            ASR_INSTANCE = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8", download_root=str(CACHE / "whisper"))
        return ASR_INSTANCE


def transcribe(path: Path) -> tuple[str, float]:
    try:
        model = load_asr()
        segments, _ = model.transcribe(str(path), language="pt", vad_filter=True, beam_size=5)
        parts = list(segments)
        text = " ".join(s.text.strip() for s in parts).strip()
        confs = [max(0.0, min(1.0, (float(s.avg_logprob) + 2.5) / 2.5)) * (1.0 - float(s.no_speech_prob)) for s in parts]
        return text, (sum(confs) / len(confs) if confs else 0.0)
    except Exception as exc:
        LOGGER.exception("ASR falhou para %s", path)
        return "", 0.0


def classify(m: dict[str, float], text: str, confidence: float) -> tuple[str, str]:
    if not text:
        return "REVISAR", "ASR sem transcrição; revisar manualmente"
    if m["duration"] < 0.8:
        return "REJEITADO", "curto demais"
    if m["clipping"] > 0.02:
        return "REJEITADO", "clipping detectado"
    if m["rms_db"] < -45:
        return "REJEITADO", "nível muito baixo"
    if confidence >= 0.82 and m["snr_db"] >= 18 and m["clipping"] < 0.001:
        return "EXCELENTE", "voz clara, ASR confiante"
    if confidence >= 0.60 and m["snr_db"] >= 10:
        return "BOM", "aceitável; revisar se necessário"
    return "REVISAR", "ruído ou confiança intermediária"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    _assert_no_reparse_chain(path)
    _assert_no_reparse_chain(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"{SEED}:{r['audio']}".encode()).hexdigest())
    if len(ordered) < 4:
        return ordered, [], []
    test_n = max(1, round(len(ordered) * 0.1)); val_n = max(1, round(len(ordered) * 0.1))
    return ordered[: -(val_n + test_n)], ordered[-(val_n + test_n): -test_n], ordered[-test_n:]


def _report(ctx: ProfileContext, rows: list[dict[str, Any]], ref: Path | None) -> Path:
    counts = {k: sum(1 for r in rows if r["quality"] == k) for k in ("EXCELENTE", "BOM", "REVISAR", "REJEITADO")}
    useful = sum(r["duration"] for r in rows if r["quality"] in ("EXCELENTE", "BOM"))
    rejected = sum(r["duration"] for r in rows if r["quality"] == "REJEITADO")
    durations = sorted(float(r.get("duration", 0)) for r in rows)
    distribution = f"min {durations[0]:.1f}s · mediana {durations[len(durations)//2]:.1f}s · máx {durations[-1]:.1f}s" if durations else "sem clips"
    body = "".join(f"<tr><td>{html.escape(str(r['file']))}</td><td>{r['duration']:.1f}</td><td>{r['quality']}</td><td>{html.escape(r.get('reason',''))}</td><td>{r.get('confidence',0):.2f}</td><td>{r.get('snr_db',0):.1f}</td><td>{r.get('clipping',0):.4f}</td></tr>" for r in rows)
    out = ctx.reports / "dataset_report.html"
    _check_no_reparse(ctx.root, out)
    out.write_text(f"<!doctype html><meta charset='utf-8'><title>Relatório IA DE VOZ</title><h1>Relatório do dataset</h1><p>Clips: {len(rows)} · útil: {useful/3600:.2f} h · rejeitado: {rejected/3600:.2f} h<br>Duração: {distribution}</p><p>EXCELENTE {counts['EXCELENTE']} · BOM {counts['BOM']} · REVISAR {counts['REVISAR']} · REJEITADO {counts['REJEITADO']}<br>Referência: {html.escape(str(ref) if ref else 'não selecionada')}</p><table border='1' cellpadding='4'><tr><th>Arquivo</th><th>s</th><th>qualidade</th><th>motivo</th><th>conf.</th><th>SNR</th><th>clipping</th></tr>{body}</table>", encoding="utf-8")
    return out


def prepare_dataset(voice_id: str | None = None, progress=_GradioProgress(track_tqdm=False) if _GradioProgress else None) -> tuple[str, list[list[Any]], str | None]:
    ctx = callback_context(voice_id)
    files = audio_files(ctx.voice_id)
    if not files:
        return f"Nenhum áudio em `{ctx.input_audio}`.", [], None
    cached: dict[str, dict[str, Any]] = {}
    _check_no_reparse(ctx.root, ctx.review_file)
    if ctx.review_file.exists():
        try:
            cached = {r.get("segment_hash", r.get("audio", "")): r for r in json.loads(ctx.review_file.read_text(encoding="utf-8"))}
        except Exception:
            cached = {}
    rows: list[dict[str, Any]] = []
    for n, source in enumerate(files, 1):
        if progress:
            progress((n - 1) / max(1, len(files)), desc=f"Inspeção/conversão: {source.name}")
        _check_no_reparse(ctx.root, source)
        digest = sha256_file(source)
        clean = ctx.cleaned / f"{digest}.wav"
        try:
            _check_no_reparse(ctx.root, clean)
            if not clean.exists():
                normalise_audio(source, clean)
            segs = split_segments(clean, digest[:12], ctx)
        except Exception as exc:
            LOGGER.exception("falha de áudio %s", source)
            rows.append({"file": source.name, "audio": str(source), "duration": 0, "text": "", "quality": "REJEITADO", "reason": str(exc), "confidence": 0, "snr_db": 0, "clipping": 0})
            continue
        for seg in segs:
            sh = sha256_file(seg)
            if sh in cached:
                row = dict(cached[sh]); row["audio"] = str(seg); row["file"] = source.name
                rows.append(row); continue
            try:
                m = metrics(seg)
                text, conf = transcribe(seg)
                quality, reason = classify(m, text, conf)
                row = {"file": source.name, "audio": str(seg), "segment_hash": sh, "duration": round(m["duration"], 3), "text": text, "quality": quality, "reason": reason, "confidence": round(conf, 4), "snr_db": round(m["snr_db"], 2), "clipping": round(m["clipping"], 6), "noise_db": round(m["noise_db"], 2)}
            except Exception as exc:
                row = {"file": source.name, "audio": str(seg), "segment_hash": sh, "duration": 0, "text": "", "quality": "REJEITADO", "reason": str(exc), "confidence": 0, "snr_db": 0, "clipping": 0}
            rows.append(row)
    callback_context(ctx.voice_id)
    _check_no_reparse(ctx.root, ctx.review_file)
    ctx.review_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    accepted = [r for r in rows if r["quality"] in ("EXCELENTE", "BOM") and r.get("text")]
    for r in accepted:
        try:
            r["train_audio"] = str(make_training_audio(profile_path(ctx, r["audio"], must_exist=True), r["segment_hash"], ctx))
        except Exception as exc:
            r["quality"] = "REVISAR"; r["reason"] = f"conversão 24 kHz falhou: {exc}"
    accepted = [r for r in rows if r["quality"] in ("EXCELENTE", "BOM") and r.get("text") and r.get("train_audio")]
    ref: Path | None = None
    if accepted:
        chosen = sorted(accepted, key=lambda r: (r["quality"] == "EXCELENTE", r.get("snr_db", 0), r.get("confidence", 0), r.get("duration", 0)), reverse=True)[0]
        ref = ctx.references / "ref.wav"
        _check_no_reparse(ctx.root, ctx.references)
        _check_no_reparse(ctx.root, ref)
        shutil.copy2(chosen["train_audio"], ref)
        _check_no_reparse(ctx.root, ctx.references / "ref.txt")
        (ctx.references / "ref.txt").write_text(chosen["text"], encoding="utf-8")
    train, validation, test = _split_records(accepted)
    to_json = lambda r: {"audio": profile_relative(ctx, Path(r.get("train_audio", r["audio"]))), "text": r["text"], "ref_audio": profile_relative(ctx, ref) if ref else ""}
    callback_context(ctx.voice_id)
    _write_jsonl(ctx.dataset / "train_raw.jsonl", [to_json(r) for r in train])
    _write_jsonl(ctx.dataset / "validation_raw.jsonl", [to_json(r) for r in validation])
    _write_jsonl(ctx.dataset / "test_raw.jsonl", [to_json(r) for r in test])
    callback_context(ctx.voice_id)
    _check_no_reparse(ctx.root, ctx.rejected)
    for r in rows:
        if r["quality"] == "REJEITADO":
            try:
                _check_no_reparse(ctx.root, r["audio"])
                target = ctx.rejected / Path(r["audio"]).name
                _check_no_reparse(ctx.root, target)
                shutil.copy2(r["audio"], target)
            except Exception: pass
    callback_context(ctx.voice_id)
    report = _report(ctx, rows, ref)
    token_status = "tokenizer pendente"
    if train and ref:
        token_status = prepare_codes(ctx.voice_id)
    rejected_duration = sum(r["duration"] for r in rows if r["quality"] == "REJEITADO")
    stats = f"**{len(rows)} clips** · aceitos: **{len(accepted)}** · rejeitados: **{sum(r['quality']=='REJEITADO' for r in rows)}** · duração útil: **{sum(r['duration'] for r in accepted)/3600:.2f} h** · rejeitada: **{rejected_duration/3600:.2f} h**\n\nReferência: `{ref or 'não selecionada'}`\n\nJSONL: `train_raw.jsonl`, `validation_raw.jsonl`, `test_raw.jsonl`\n\n{token_status}\n\nRelatório: `{report}`"
    table = [[r.get("file", ""), r.get("duration", 0), r.get("text", ""), r.get("quality", ""), r.get("reason", ""), r.get("confidence", 0), r.get("snr_db", 0), r.get("clipping", 0)] for r in rows]
    return stats, table, profile_relative(ctx, ref) if ref else None


def validate_jsonl(path: Path, ctx: ProfileContext | None = None) -> tuple[bool, str]:
    ctx = ctx or profile_context()
    try:
        path = ctx.path(profile_relative(ctx, path), must_exist=True)
    except ProfileError as exc:
        return False, str(exc)
    _check_no_reparse(ctx.root, path)
    if not path.exists(): return False, f"ausente: {path}"
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try: row = json.loads(line)
        except Exception as exc: return False, f"linha {i} inválida: {exc}"
        for key in ("audio", "text", "ref_audio"):
            if not row.get(key): return False, f"linha {i} sem {key}"
        try:
            ctx.path(str(row["audio"]), must_exist=True)
            ctx.path(str(row["ref_audio"]), must_exist=True)
        except ProfileError as exc:
            return False, f"linha {i}: {exc}"
    return True, "ok"


def prepare_codes(voice_id: str | None = None) -> str:
    ctx = callback_context(voice_id)
    raw = ctx.dataset / "train_raw.jsonl"
    ok, msg = validate_jsonl(raw, ctx)
    if not ok: return f"Tokenizer não executado: {msg}"
    out = ctx.dataset / "train_with_codes.jsonl"
    _check_no_reparse(ctx.root, out)
    if out.exists() and out.stat().st_mtime >= raw.stat().st_mtime:
        try:
            _check_no_reparse(ctx.root, out)
            rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
            if rows and all(row.get("audio_codes") for row in rows):
                return f"Tokenizer já preparado: `{out}`"
        except Exception:
            pass
    token_input = ctx.dataset / "metadata" / ".tokenizer_input.jsonl"
    token_output = ctx.dataset / "metadata" / ".tokenizer_output.jsonl"
    _check_no_reparse(ctx.root, raw)
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    absolute_rows = []
    for row in rows:
        copy = dict(row)
        copy["audio"] = str(ctx.path(row["audio"], must_exist=True))
        copy["ref_audio"] = str(ctx.path(row["ref_audio"], must_exist=True))
        absolute_rows.append(copy)
    _write_jsonl(token_input, absolute_rows)
    cmd = [sys.executable, str(QWEN_REPO / "finetuning" / "prepare_data.py"), "--device", "cuda:0" if _cuda() else "cpu", "--tokenizer_model_path", TOKENIZER_MODEL, "--input_jsonl", str(token_input), "--output_jsonl", str(token_output)]
    try:
        LOGGER.info("tokenizer: %s", cmd)
        subprocess.run(cmd, cwd=str(QWEN_REPO / "finetuning"), check=True, capture_output=True, text=True)
        _check_no_reparse(ctx.root, token_output)
        coded = [json.loads(line) for line in token_output.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in coded:
            row["audio"] = profile_relative(ctx, Path(row["audio"]))
            row["ref_audio"] = profile_relative(ctx, Path(row["ref_audio"]))
        callback_context(ctx.voice_id)
        _write_jsonl(out, coded)
        ok, msg = validate_jsonl(out, ctx)
        if ok:
            try:
                _check_no_reparse(ctx.root, out)
                lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
                ok = bool(lines) and all(row.get("audio_codes") for row in lines)
                msg = "ok" if ok else "audio_codes ausente"
            except Exception as exc:
                ok, msg = False, str(exc)
        return f"Tokenizer concluído: `{out}`" if ok else f"Tokenizer gerou JSONL inválido: {msg}"
    except subprocess.CalledProcessError as exc:
        LOGGER.error("tokenizer falhou: %s\n%s", exc, exc.stderr)
        return f"Tokenizer falhou; veja `{LOGS / 'app.log'}`. Treinamento bloqueado."
    finally:
        for temp in (token_input, token_output):
            try: temp.unlink()
            except FileNotFoundError: pass


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_tts(model_id: str = MODEL_17B) -> Any:
    key = str(Path(model_id).resolve()) if Path(model_id).exists() else model_id
    if key in TTS_CACHE: return TTS_CACHE[key]
    from qwen_tts import Qwen3TTSModel
    import torch
    local_name = model_id.replace("/", "--")
    local = MODELS / local_name
    # The 0.6B checkpoint triggers a CUDA device-side assert on Pascal cards;
    # keep it available as a safe CPU fallback instead of crashing the GUI.
    pascal = _cuda() and torch.cuda.get_device_capability(0)[0] < 7
    use_cuda = _cuda() and not ("0.6B" in str(model_id) and pascal)
    kwargs: dict[str, Any] = {"device_map": "cuda:0" if use_cuda else "cpu", "torch_dtype": torch.float16 if use_cuda else torch.float32, "attn_implementation": "eager"}
    path = str(local) if (local / "config.json").exists() else model_id
    tts = Qwen3TTSModel.from_pretrained(path, **kwargs)
    TTS_CACHE[key] = tts
    return tts


def ensure_model(model_id: str = MODEL_17B) -> str:
    local = MODELS / model_id.replace("/", "--")
    if (local / "config.json").exists() and any(local.glob("model*.safetensors")):
        return str(local)
    required = 8 * 2**30 if "1.7B" in model_id else 4 * 2**30
    free = shutil.disk_usage(ROOT).free
    if free < required:
        raise RuntimeError(f"Espaço livre insuficiente para {model_id}: {free/2**30:.1f} GB disponíveis; requer pelo menos {required/2**30:.0f} GB.")
    from huggingface_hub import snapshot_download
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id, local_dir=str(local), cache_dir=str(CACHE / "huggingface"), local_dir_use_symlinks=False)
    return str(local)


def checkpoint_choices(voice_id: str | None = None) -> list[str]:
    try:
        ctx = profile_context(voice_id)
        _check_no_reparse(ctx.root, ctx.training)
        choices = []
        for path in sorted(ctx.training.rglob("checkpoint-*")):
            _check_no_reparse(ctx.root, path)
            if path.is_dir():
                choices.append(profile_relative(ctx, path))
        return choices
    except (ProfileError, OSError):
        return []


def _checkpoint_for(ctx: ProfileContext, choice: str) -> tuple[str, Path | None]:
    if not choice or choice.startswith("Base zero-shot"):
        return (MODEL_06B if "0.6" in choice else MODEL_17B), None
    checkpoint = ctx.path(choice)
    if not checkpoint.is_dir() or not _inside(ctx.training.resolve(strict=True), checkpoint):
        raise ProfileError("checkpoint fora do treinamento da voz ativa")
    _check_no_reparse(ctx.root, checkpoint)
    return str(checkpoint), checkpoint


def generate_voice(text: str, choice: str, voice_id: str | None = None) -> str:
    ctx = callback_context(voice_id)
    if not text or not text.strip(): raise ValueError("Digite um texto em Português.")
    ref = ctx.path("dataset/references/ref.wav", must_exist=True)
    model_id, checkpoint = _checkpoint_for(ctx, choice)
    model_path = str(checkpoint) if checkpoint else ensure_model(model_id)
    tts = load_tts(model_path)
    ref_text_path = ctx.references / "ref.txt"
    _check_no_reparse(ctx.root, ref_text_path)
    ref_text = ref_text_path.read_text(encoding="utf-8") if ref_text_path.exists() else None
    if checkpoint:
        wavs, sr = tts.generate_custom_voice(text=text, speaker=f"speaker_{ctx.voice_id[:8]}", language="Portuguese")
    else:
        wavs, sr = tts.generate_voice_clone(text=text, language="Portuguese", ref_audio=str(ref), ref_text=ref_text, x_vector_only_mode=not bool(ref_text), non_streaming_mode=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "base" if not checkpoint else checkpoint.name
    out = ctx.generated / f"{stamp}_{ctx.voice_id}_{re.sub(r'[^a-zA-Z0-9_-]+', '_', label)}.wav"
    _check_no_reparse(ctx.root, ctx.generated)
    callback_context(ctx.voice_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    callback_context(ctx.voice_id)
    _check_no_reparse(ctx.root, out)
    import soundfile as sf
    sf.write(out, wavs[0], sr)
    try:
        score = score_generated(out, text, ref)
        score.update({"voice_id": ctx.voice_id, "model": model_id, "checkpoint": profile_relative(ctx, checkpoint) if checkpoint else None, "timestamp": stamp})
        callback_context(ctx.voice_id)
        _check_no_reparse(ctx.root, out.with_suffix(".json"))
        out.with_suffix(".json").write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        LOGGER.exception("avaliação da geração falhou")
    return str(out)


BASELINE_PHRASES = [
    "Olá, esta é uma frase curta em Português Brasileiro.",
    "Hoje vamos comparar clareza, timbre e naturalidade da voz treinada.",
    "A pergunta é simples: você consegue ouvir cada palavra com nitidez?",
    "Quatrocentos e vinte e três reais foram registrados no relatório de avaliação.",
]


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _speaker_similarity(ref: Path, generated: Path) -> float | None:
    try:
        import librosa
        import numpy as np
        ry, rs = librosa.load(ref, sr=16000, mono=True)
        gy, gs = librosa.load(generated, sr=16000, mono=True)
        rf = librosa.feature.mfcc(y=ry, sr=rs, n_mfcc=20).mean(axis=1)
        gf = librosa.feature.mfcc(y=gy, sr=gs, n_mfcc=20).mean(axis=1)
        return float(np.dot(rf, gf) / (np.linalg.norm(rf) * np.linalg.norm(gf) + 1e-8))
    except Exception:
        return None


def score_generated(path: Path, requested: str, ref: Path) -> dict[str, Any]:
    heard, confidence = transcribe(path)
    ref_words = requested.lower().split(); heard_words = heard.lower().split()
    wer = _edit_distance(ref_words, heard_words) / max(1, len(ref_words))
    ref_chars = list("".join(ref_words)); heard_chars = list("".join(heard_words))
    cer = _edit_distance(ref_chars, heard_chars) / max(1, len(ref_chars))
    return {"requested_text": requested, "asr_text": heard, "asr_confidence": round(confidence, 4), "wer": round(wer, 4), "cer": round(cer, 4), "speaker_similarity_proxy": _speaker_similarity(ref, path), "warning": "similaridade é apenas apoio; não é verdade absoluta"}


def evaluate_checkpoints(run_dir: Path, voice_id: str | None = None) -> str:
    ctx = callback_context(voice_id)
    run_dir = ctx.path(profile_relative(ctx, run_dir))
    if not run_dir.is_dir() or not _inside(ctx.training.resolve(strict=True), run_dir):
        raise ProfileError("run fora do treinamento da voz ativa")
    _check_no_reparse(ctx.root, run_dir)
    checkpoints = []
    for checkpoint in sorted(run_dir.glob("checkpoint-epoch-*")):
        _check_no_reparse(ctx.root, checkpoint)
        if checkpoint.is_dir():
            checkpoints.append(checkpoint)
    if not checkpoints:
        return "Nenhum checkpoint encontrado para avaliação."
    scores: list[dict[str, Any]] = []
    loss_value = None
    log_path = run_dir / "train.log"
    _check_no_reparse(ctx.root, log_path)
    if log_path.exists():
        losses = re.findall(r"loss=([0-9]+(?:\.[0-9]+)?)", log_path.read_text(encoding="utf-8", errors="ignore"))
        if losses:
            loss_value = float(losses[-1])
    for checkpoint in checkpoints:
        vals = []
        for phrase in BASELINE_PHRASES:
            try:
                out = Path(generate_voice(phrase, profile_relative(ctx, checkpoint), ctx.voice_id))
                _check_no_reparse(ctx.root, out.with_suffix(".json"))
                vals.append(json.loads(out.with_suffix(".json").read_text(encoding="utf-8")))
            except Exception as exc:
                LOGGER.exception("checkpoint %s falhou", checkpoint)
                vals.append({"wer": 1.0, "cer": 1.0, "error": str(exc)})
        valid = [v for v in vals if "wer" in v]
        score = {"checkpoint": str(checkpoint), "wer": sum(v["wer"] for v in valid) / max(1, len(valid)), "cer": sum(v["cer"] for v in valid) / max(1, len(valid)), "speaker_similarity_proxy": sum((v.get("speaker_similarity_proxy") or 0) for v in valid) / max(1, len(valid)), "loss": loss_value}
        loss_component = 1 / (1 + loss_value) if loss_value is not None else 0.0
        score["score"] = round((1 - min(1, score["wer"])) * 0.5 + max(0, score["speaker_similarity_proxy"]) * 0.3 + loss_component * 0.2, 4)
        scores.append(score)
    scores.sort(key=lambda x: x["score"], reverse=True)
    callback_context(ctx.voice_id)
    _check_no_reparse(ctx.root, run_dir / "checkpoint_ranking.json")
    (run_dir / "checkpoint_ranking.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"Ranking salvo em `{run_dir / 'checkpoint_ranking.json'}`; melhor: `{scores[0]['checkpoint']}`."


def generate_baseline(voice_id: str | None = None) -> str:
    """Generate the fixed zero-shot comparison set once per reference."""
    ctx = callback_context(voice_id)
    ref = ctx.references / "ref.wav"
    _check_no_reparse(ctx.root, ctx.references)
    _check_no_reparse(ctx.root, ref)
    if not ref.exists():
        return f"Baseline pendente: ainda não há `{ctx.references / 'ref.wav'}`."
    out_dir = ctx.generated / "baseline"
    _check_no_reparse(ctx.root, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_no_reparse(ctx.root, out_dir)
    existing = []
    for path in sorted(out_dir.glob("*.wav")):
        _check_no_reparse(ctx.root, path)
        existing.append(path)
    if len(existing) >= len(BASELINE_PHRASES):
        return f"Baseline já disponível em `{out_dir}`."
    tts = load_tts(ensure_model(MODEL_17B))
    ref_text_path = ctx.references / "ref.txt"
    _check_no_reparse(ctx.root, ref_text_path)
    ref_text = ref_text_path.read_text(encoding="utf-8") if ref_text_path.exists() else None
    import soundfile as sf
    for i, phrase in enumerate(BASELINE_PHRASES, 1):
        wavs, sr = tts.generate_voice_clone(text=phrase, language="Portuguese", ref_audio=str(ref), ref_text=ref_text, x_vector_only_mode=not bool(ref_text), non_streaming_mode=True)
        output = out_dir / f"phrase_{i:02d}.wav"
        callback_context(ctx.voice_id)
        _check_no_reparse(ctx.root, output)
        callback_context(ctx.voice_id)
        sf.write(output, wavs[0], sr)
    return f"Baseline zero-shot gerado em `{out_dir}` com {len(BASELINE_PHRASES)} frases fixas."


def start_training(voice_id: str, model_choice: str, method: str, batch: int, grad_acc: int, lr: float, epochs: int, progress=None) -> str:
    global TRAIN_PROCESS, TRAIN_OWNER
    ctx = callback_context(voice_id)
    if TRAIN_PROCESS and TRAIN_PROCESS.poll() is None:
        return f"Treinamento já em execução para a voz `{TRAIN_OWNER}`."
    if "LoRA" in (method or ""):
        return "LoRA/PEFT permanece experimental e bloqueado: o repositório oficial ainda não fornece um caminho validado para este modelo. Use SFT oficial em GPU compatível."
    train_json = ctx.dataset / "train_with_codes.jsonl"
    ok, msg = validate_jsonl(train_json, ctx)
    if not ok: return f"Treinamento bloqueado: {msg}. Clique PREPARAR DATASET primeiro."
    if "0.6" in model_choice:
        return "Treinamento 0.6B bloqueado: o SFT oficial atual assume embeddings de dimensão igual e falha no 0.6B. A opção 0.6B permanece disponível para inferência CPU; use o SFT 1.7B em GPU compatível."
    if not _cuda(): return "Treinamento não iniciado: CUDA não disponível; não vou fingir um SFT."
    baseline_note = generate_baseline(ctx.voice_id)
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        required = 7 * 2**30 if "1.7" in model_choice else 5 * 2**30
        if total < required:
            model_label = "1.7B" if "1.7" in model_choice else "0.6B"
            return f"{baseline_note}\n\nTreinamento não iniciado: VRAM total {total/2**30:.1f} GB; o perfil {model_label} requer pelo menos {required/2**30:.0f} GB para o pre-flight conservador."
    except Exception as exc:
        return f"Pre-flight de VRAM falhou: {exc}"
    _check_no_reparse(ctx.root, ctx.training / "runs")
    run_dir = ctx.training / "runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _check_no_reparse(ctx.root, run_dir)
    callback_context(ctx.voice_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    _check_no_reparse(ctx.root, run_dir)
    model_id = MODEL_06B if "0.6" in model_choice else MODEL_17B
    model_path = ensure_model(model_id)
    callback_context(ctx.voice_id)
    _atomic_json(run_dir / "run.json", {"schema_version": 1, "voice_id": ctx.voice_id, "base_model": model_id, "method": method, "train_jsonl": profile_relative(ctx, train_json), "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z"})
    cmd = [sys.executable, str(ROOT / "scripts" / "train_runner.py"), "--init_model_path", model_path, "--output_model_path", str(run_dir), "--train_jsonl", str(train_json), "--voice_id", ctx.voice_id, "--profile_root", str(ctx.root), "--batch_size", str(max(1, int(batch))), "--gradient_accumulation", str(max(1, int(grad_acc))), "--lr", str(lr), "--num_epochs", str(max(1, int(epochs))), "--speaker_name", f"speaker_{ctx.voice_id[:8]}"]
    log = run_dir / "train.log"
    callback_context(ctx.voice_id)
    _check_no_reparse(ctx.root, log)
    LOGGER.info("treino: %s", cmd)
    with log.open("w", encoding="utf-8") as f:
        TRAIN_OWNER = ctx.voice_id
        TRAIN_PROCESS = subprocess.Popen(cmd, cwd=str(QWEN_REPO / "finetuning"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        started = time.time()
        if TRAIN_PROCESS.stdout:
            for line in TRAIN_PROCESS.stdout:
                f.write(line); f.flush()
                if progress:
                    progress(min(0.99, (time.time() - started) / max(1, int(epochs) * 600)), desc=line.strip()[:120])
        code = TRAIN_PROCESS.wait()
    TRAIN_PROCESS = None
    TRAIN_OWNER = None
    ranking = evaluate_checkpoints(run_dir, ctx.voice_id) if code == 0 else "Ranking não executado porque o treino falhou."
    return f"Treinamento {'concluído' if code == 0 else 'falhou'} (código {code}). Run: `{run_dir}`. Log: `{log}`\n\n{ranking}"


def stop_training(voice_id: str | None = None) -> str:
    global TRAIN_PROCESS
    if TRAIN_PROCESS and TRAIN_PROCESS.poll() is None:
        if voice_id and TRAIN_OWNER != voice_id:
            return "Treinamento pertence a outra voz; parada recusada."
        TRAIN_PROCESS.terminate(); return "Solicitada parada segura; checkpoint anterior permanece intacto."
    return "Nenhum treinamento em execução."


def review_rows(voice_id: str | None = None) -> list[list[Any]]:
    try:
        ctx = profile_context(voice_id)
        review_file = ctx.review_file
        _check_no_reparse(ctx.root, review_file)
    except ProfileError:
        return []
    if not review_file.exists(): return []
    try:
        rows = json.loads(review_file.read_text(encoding="utf-8"))
        return [[r.get("file", ""), r.get("duration", 0), r.get("text", ""), r.get("quality", ""), r.get("reason", ""), r.get("confidence", 0), r.get("snr_db", 0), r.get("clipping", 0)] for r in rows]
    except Exception: return []


def rebuild_dataset(voice_id: str, table: list[list[Any]]) -> tuple[str, list[list[Any]]]:
    ctx = callback_context(voice_id)
    _check_no_reparse(ctx.root, ctx.review_file)
    if not ctx.review_file.exists(): return "Nada para reconstruir.", []
    original = {r.get("audio"): r for r in json.loads(ctx.review_file.read_text(encoding="utf-8"))}
    for row in table or []:
        if len(row) < 4: continue
        matches = [r for r in original.values() if r.get("file") == row[0] and abs(float(r.get("duration", 0)) - float(row[1] or 0)) < 0.01]
        if matches:
            matches[0]["text"] = str(row[2] or "").strip(); matches[0]["quality"] = str(row[3] or "REVISAR").upper(); matches[0]["reason"] = str(row[4] or "edição manual")
    callback_context(ctx.voice_id)
    _check_no_reparse(ctx.root, ctx.review_file)
    ctx.review_file.write_text(json.dumps(list(original.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    # Re-run only the cheap deterministic export; ASR/audio work stays cached.
    rows = list(original.values()); accepted = [r for r in rows if r["quality"] in ("EXCELENTE", "BOM") and r.get("text")]
    _check_no_reparse(ctx.root, ctx.references)
    ref = ctx.references / "ref.wav" if (ctx.references / "ref.wav").exists() else None
    if ref:
        _check_no_reparse(ctx.root, ref)
    for r in accepted:
        if not r.get("train_audio"):
            try: r["train_audio"] = str(make_training_audio(profile_path(ctx, r["audio"], must_exist=True), r["segment_hash"], ctx))
            except Exception: continue
    accepted = [r for r in accepted if r.get("train_audio")]
    train, val, test = _split_records(accepted)
    to_json = lambda r: {"audio": profile_relative(ctx, Path(r.get("train_audio", r["audio"]))), "text": r["text"], "ref_audio": profile_relative(ctx, ref) if ref else ""}
    callback_context(ctx.voice_id)
    _write_jsonl(ctx.dataset / "train_raw.jsonl", [to_json(r) for r in train]); _write_jsonl(ctx.dataset / "validation_raw.jsonl", [to_json(r) for r in val]); _write_jsonl(ctx.dataset / "test_raw.jsonl", [to_json(r) for r in test])
    status = prepare_codes(ctx.voice_id) if accepted and ref else "Tokenizer bloqueado: sem referência ou clips aceitos."
    return f"Dataset reconstruído: {len(accepted)} aceitos. {status}", review_rows(ctx.voice_id)


def profile_choices() -> list[tuple[str, str]]:
    labels = profile_presentation_labels()
    return [(labels.get(voice_id, display_name), voice_id) for display_name, voice_id in PROFILE_STORE.list_profiles()]


def profile_presentation_labels() -> dict[str, str]:
    """Build unique UI labels without changing persisted profile names."""
    entries: list[tuple[str, str, bool]] = []
    for display_name, voice_id in PROFILE_STORE.list_profiles():
        try:
            data = profile_data(profile_context(voice_id))
            migrated = data.get("migration", {}).get("source") == "legacy-global"
        except (OSError, ProfileError, ValueError, json.JSONDecodeError):
            migrated = False
        base = str(display_name or voice_id)
        if migrated:
            base += " (migrada)"
        entries.append((voice_id, base, migrated))
    counts: dict[str, int] = {}
    for _, base, _ in entries:
        counts[base] = counts.get(base, 0) + 1
    return {voice_id: base if counts[base] == 1 else f"{base} · {voice_id[:6]}" for voice_id, base, _ in entries}


def profile_presentation_label(voice_id: str, fallback: str | None = None) -> str:
    return profile_presentation_labels().get(voice_id, fallback or voice_id)


def active_voice_label(voice_id: str | None = None) -> str:
    try:
        ctx = profile_context(voice_id)
        name = html.escape(profile_presentation_label(ctx.voice_id, str(profile_data(ctx).get("display_name") or ctx.voice_id)))
        return f"## Voz ativa: {name}\n<details><summary>Detalhes técnicos</summary><code>voice_id: {ctx.voice_id}</code></details>"
    except ProfileError:
        return "## Voz ativa: nenhuma voz selecionada"


def active_voice_help(voice_id: str | None = None) -> str:
    try:
        ctx = profile_context(voice_id)
        name = html.escape(profile_presentation_label(ctx.voice_id, str(profile_data(ctx).get("display_name") or ctx.voice_id)))
        return f"A próxima etapa é abrir **ÁUDIOS** e enviar arquivos somente para **{name}**. Para outra voz, troque primeiro o seletor **Voz ativa**. **Existing Voice (migrada)** é a voz legada copiada automaticamente; os demais nomes são perfis separados."
    except ProfileError:
        return "Crie ou selecione uma voz para começar."


def refresh_profile(voice_id: str | None) -> tuple[str | None, str, str, str, list[list[Any]], Any, str]:
    try:
        ctx = profile_context(voice_id)
        selected = ctx.voice_id
    except ProfileError:
        selected = None
    import gradio as gr
    return selected, active_voice_label(selected), active_voice_help(selected), audio_inventory(selected), review_rows(selected), gr.Dropdown(choices=checkpoint_choices(selected), value=None), voice_overview()


def select_voice(voice_id: str) -> tuple[str | None, str, str, str, list[list[Any]], Any, str, str]:
    try:
        ctx = PROFILE_STORE.select(voice_id)
        selected, label, help_text, audio, review, checkpoints, overview = refresh_profile(ctx.voice_id)
        name = profile_presentation_label(ctx.voice_id, str(profile_data(ctx).get("display_name") or ctx.voice_id))
        return selected, label, help_text, audio, review, checkpoints, overview, f"Voz ativa: {name}. Salvo no disco; permanece após reiniciar."
    except (ProfileError, OSError, json.JSONDecodeError) as exc:
        selected, label, help_text, audio, review, checkpoints, overview = refresh_profile(None)
        return selected, label, help_text, audio, review, checkpoints, overview, f"Seleção recusada: {exc}"


def create_voice(display_name: str) -> tuple[str, Any, str | None, str, str, str, list[list[Any]], Any, str]:
    import gradio as gr
    try:
        ctx = PROFILE_STORE.create(display_name)
        PROFILE_STORE.select(ctx.voice_id)
        selected, label, help_text, audio, review, checkpoints, overview = refresh_profile(ctx.voice_id)
        name = profile_presentation_label(ctx.voice_id, display_name)
        return f"Voz criada: {name}. Salvo no disco; permanece após reiniciar.", gr.Dropdown(choices=profile_choices(), value=ctx.voice_id), selected, label, help_text, audio, review, checkpoints, overview
    except (ProfileError, OSError) as exc:
        selected, label, help_text, audio, review, checkpoints, overview = refresh_profile(PROFILE_STORE.active_id())
        return f"Criação recusada: {exc}", gr.Dropdown(choices=profile_choices(), value=selected), selected, label, help_text, audio, review, checkpoints, overview


def rename_voice(voice_id: str | None, display_name: str) -> tuple[str, Any, str | None, str, str, str, list[list[Any]], Any, str]:
    import gradio as gr
    try:
        callback_context(voice_id)
        ctx = PROFILE_STORE.rename(voice_id or "", display_name)
        selected, label, help_text, audio, review, checkpoints, overview = refresh_profile(ctx.voice_id)
        name = profile_presentation_label(ctx.voice_id, display_name)
        return f"Voz renomeada: {name}. Salvo no disco; permanece após reiniciar.", gr.Dropdown(choices=profile_choices(), value=ctx.voice_id), selected, label, help_text, audio, review, checkpoints, overview
    except (ProfileError, OSError, json.JSONDecodeError) as exc:
        selected, label, help_text, audio, review, checkpoints, overview = refresh_profile(PROFILE_STORE.active_id())
        return f"Renomeação recusada: {exc}", gr.Dropdown(choices=profile_choices(), value=selected), selected, label, help_text, audio, review, checkpoints, overview


def delete_voice(voice_id: str | None, confirmation: str) -> tuple[str, Any, str | None, str, str, str, list[list[Any]], Any, str]:
    import gradio as gr
    try:
        callback_context(voice_id)
        PROFILE_STORE.delete(voice_id or "", confirmation)
        selected = PROFILE_STORE.active_id()
        _, label, help_text, audio, review, checkpoints, overview = refresh_profile(selected)
        return "Voz excluída após confirmação.", gr.Dropdown(choices=profile_choices(), value=selected), selected, label, help_text, audio, review, checkpoints, overview
    except (ProfileError, OSError, json.JSONDecodeError) as exc:
        selected, label, help_text, audio, review, checkpoints, overview = refresh_profile(PROFILE_STORE.active_id())
        return f"Exclusão recusada: {exc}", gr.Dropdown(choices=profile_choices(), value=selected), selected, label, help_text, audio, review, checkpoints, overview


def migrate_legacy_voice() -> tuple[str, Any, str | None, str, str, str, list[list[Any]], Any, str]:
    import gradio as gr
    try:
        ctx, status = PROFILE_STORE.migrate_legacy()
        selected, label, help_text, audio, review, checkpoints, overview = refresh_profile(ctx.voice_id)
        return status, gr.Dropdown(choices=profile_choices(), value=ctx.voice_id), selected, label, help_text, audio, review, checkpoints, overview
    except (ProfileError, OSError, json.JSONDecodeError) as exc:
        selected = PROFILE_STORE.active_id()
        selected, label, help_text, audio, review, checkpoints, overview = refresh_profile(selected)
        return f"Migração recusada: {exc}", gr.Dropdown(choices=profile_choices(), value=selected), selected, label, help_text, audio, review, checkpoints, overview


def build_ui() -> Any:
    import gradio as gr
    initial_voice = PROFILE_STORE.active_id()
    with gr.Blocks(title="IA DE VOZ — Qwen3-TTS", analytics_enabled=False) as demo:
        active_state = gr.State(initial_voice)
        active_label = gr.Markdown(active_voice_label(initial_voice))
        active_help = gr.Markdown(active_voice_help(initial_voice))
        voice_selector = gr.Dropdown(choices=profile_choices(), value=initial_voice, label="Voz ativa — troque antes de enviar áudio", type="value")
        gr.Markdown("### Primeiro: escolha a voz acima\nTudo o que você enviar, preparar ou gerar ficará ligado à voz selecionada.")
        gr.Markdown("# IA DE VOZ\nEstação local para dataset, fine-tuning e geração PT-BR. **Os áudios permanecem neste computador.** Use somente voz autorizada.")
        with gr.Tab("SISTEMA"):
            system_box = gr.Markdown(diagnose_system())
            gr.Button("DIAGNOSTICAR SISTEMA", variant="primary").click(diagnose_system, outputs=system_box)
        with gr.Tab("VOZES"):
            gr.Markdown("### Visão geral das vozes\nCada voz possui áudio, dataset, referência, treinamento e geração isolados. **Existing Voice (migrada)** é a voz legada copiada automaticamente; os demais nomes são perfis separados. Confira aqui as contagens por voz. Alterações de nome, criação e voz ativa são salvas automaticamente no disco e permanecem após reiniciar o app.")
            voice_overview_box = gr.Markdown(voice_overview())
            voice_status = gr.Markdown()
            new_voice_name = gr.Textbox(label="Nome de exibição")
            create_button = gr.Button("CRIAR VOZ", variant="primary")
            rename_name = gr.Textbox(label="Novo nome da voz ativa")
            rename_button = gr.Button("RENOMEAR VOZ")
            delete_confirmation = gr.Textbox(label="Confirmação: digite DELETE", placeholder="DELETE")
            delete_button = gr.Button("EXCLUIR VOZ")
            open_voice_button = gr.Button("ABRIR PASTA DE ÁUDIO DA VOZ")
            migrate_button = gr.Button("MIGRAR VOZ EXISTENTE (CÓPIA NÃO DESTRUTIVA)")
        with gr.Tab("ÁUDIOS"):
            gr.Markdown("### Adicionar áudio à voz selecionada\nA área abaixo apenas escolhe os arquivos. Clique no botão para confirmar e adicionar áudios somente à voz ativa. Para outra voz, troque primeiro o seletor no topo; depois confira a contagem e a pasta na aba **VOZES**.")
            upload_input = gr.File(label="Adicionar áudios à voz ativa (arraste ou selecione)", file_count="multiple", file_types=sorted(AUDIO_EXTENSIONS), type="filepath")
            add_audio_button = gr.Button("ADICIONAR ÁUDIOS À VOZ ATIVA", variant="primary")
            upload_status = gr.Markdown()
            audio_box = gr.Markdown(audio_inventory(initial_voice))
            with gr.Row():
                refresh_audio_button = gr.Button("ATUALIZAR")
                refresh_audio_button.click(audio_inventory, inputs=active_state, outputs=audio_box).then(voice_overview, outputs=voice_overview_box)
                open_folder_status = gr.Markdown()
                gr.Button("ABRIR PASTA DE ÁUDIOS").click(open_audio_folder, inputs=active_state, outputs=open_folder_status)
            add_audio_button.click(upload_audio, inputs=[upload_input, active_state], outputs=[upload_status, audio_box]).then(voice_overview, outputs=voice_overview_box)
        with gr.Tab("PREPARAR DATASET"):
            gr.Markdown("Fluxo: Inspeção → conversão → VAD → limpeza → transcrição → filtragem → referência → JSONL → tokenizer")
            prep_status = gr.Markdown()
            prep_table = gr.Dataframe(headers=["arquivo", "duração", "transcrição", "qualidade", "motivo", "confidence", "ruído/SNR", "clipping"], interactive=False)
            ref_player = gr.Audio(label="Referência selecionada", type="filepath")
            gr.Button("PREPARAR DATASET", variant="primary").click(prepare_dataset, inputs=active_state, outputs=[prep_status, prep_table, ref_player])
        with gr.Tab("REVISÃO"):
            review_table = gr.Dataframe(value=review_rows(initial_voice), headers=["arquivo", "duração", "transcrição", "qualidade", "motivo", "confidence", "ruído/SNR", "clipping"], interactive=True)
            review_status = gr.Markdown()
            gr.Button("RECONSTRUIR DATASET", variant="primary").click(rebuild_dataset, inputs=[active_state, review_table], outputs=[review_status, review_table])
            with gr.Row():
                review_path = gr.Textbox(label="Caminho relativo do clip para ouvir", placeholder="dataset/segments/clip.wav")
                original_player = gr.Audio(label="Original/segmento", type="filepath")
                cleaned_player = gr.Audio(label="Cleaned", type="filepath")
            def _load_review_audio(path, voice_id):
                try:
                    ctx = callback_context(voice_id)
                    p = ctx.path(str(path or ""), must_exist=True)
                    clean = ctx.cleaned / p.name
                    _check_no_reparse(ctx.root, clean)
                    return str(p), str(clean) if clean.is_file() else str(p)
                except ProfileError:
                    return None, None
            gr.Button("OUVIR CLIP").click(_load_review_audio, inputs=[review_path, active_state], outputs=[original_player, cleaned_player])
            gr.Markdown("Edite transcrição/classificação na tabela e reconstrua. Os rejeitados ficam no dataset da voz ativa.")
        with gr.Tab("TREINAMENTO"):
            model_choice = gr.Dropdown(["Qwen3-TTS 1.7B Base (recomendado)", "Qwen3-TTS 0.6B Base"], value="Qwen3-TTS 1.7B Base (recomendado)", label="Modelo")
            method = gr.Dropdown(["Automático recomendado", "Full SFT oficial", "LoRA / low-VRAM experimental"], value="Automático recomendado", label="Método")
            with gr.Row():
                batch = gr.Number(value=1, label="Batch size", precision=0)
                grad = gr.Number(value=4, label="Gradient accumulation", precision=0)
                lr = gr.Number(value=2e-6, label="Learning rate")
                epochs = gr.Number(value=3, label="Epochs", precision=0)
            train_status = gr.Markdown()
            with gr.Row():
                gr.Button("INICIAR TREINAMENTO", variant="primary").click(start_training, inputs=[active_state, model_choice, method, batch, grad, lr, epochs], outputs=train_status)
                gr.Button("PARAR TREINAMENTO").click(stop_training, inputs=active_state, outputs=train_status)
            gr.Markdown("O pre-flight interrompe sem iniciar quando a VRAM não comporta o perfil. Cada execução usa uma pasta `training\\runs\\AAAA-MM-DD_HH-MM-SS`.")
        with gr.Tab("TESTAR VOZ"):
            text = gr.Textbox(value="Olá. Esta é uma frase de teste em Português Brasileiro.", label="Texto", lines=3)
            gen_model = gr.Dropdown(["Base zero-shot (1.7B)", "Base zero-shot (0.6B)"], value="Base zero-shot (1.7B)", label="Modelo/checkpoint")
            checkpoint_path = gr.Dropdown(choices=checkpoint_choices(initial_voice), label="Checkpoint do perfil ativo", allow_custom_value=False)
            audio_out = gr.Audio(label="Saída", type="filepath")
            gen_status = gr.Markdown()
            def _generate(t, c, cp, voice_id):
                try: return generate_voice(t, cp or c, voice_id), "Geração concluída."
                except Exception as exc: LOGGER.exception("inferência falhou"); return None, f"Geração não executada: {exc}"
            gr.Button("GERAR", variant="primary").click(_generate, inputs=[text, gen_model, checkpoint_path, active_state], outputs=[audio_out, gen_status])
        profile_outputs = [active_state, active_label, active_help, audio_box, review_table, checkpoint_path, voice_overview_box]
        voice_selector.change(select_voice, inputs=voice_selector, outputs=profile_outputs + [voice_status])
        create_button.click(create_voice, inputs=new_voice_name, outputs=[voice_status, voice_selector] + profile_outputs)
        rename_button.click(rename_voice, inputs=[active_state, rename_name], outputs=[voice_status, voice_selector] + profile_outputs)
        delete_button.click(delete_voice, inputs=[active_state, delete_confirmation], outputs=[voice_status, voice_selector] + profile_outputs)
        migrate_button.click(migrate_legacy_voice, outputs=[voice_status, voice_selector] + profile_outputs)
        open_voice_button.click(open_audio_folder, inputs=active_state, outputs=voice_status)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
