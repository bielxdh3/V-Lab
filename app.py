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
from pathlib import Path
from typing import Any

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
MODEL_17B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_06B = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
ASR_MODEL = os.environ.get("VOICE_ASR_MODEL", "small")
SEED = 20260826
TRAIN_PROCESS: subprocess.Popen[str] | None = None
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


def audio_files() -> list[Path]:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".webm", ".mp4"}
    return sorted((p for p in INPUT.rglob("*") if p.is_file() and p.suffix.lower() in exts and (p.suffix.lower() != ".mp4" or has_audio_stream(p))), key=lambda p: str(p).lower())


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


def audio_inventory() -> str:
    files = audio_files()
    total_seconds = sum(audio_duration(p) for p in files)
    total_bytes = sum(p.stat().st_size for p in files)
    rows = [[p.name, f"{audio_duration(p):.1f}s", f"{p.stat().st_size / 1024**2:.2f} MB"] for p in files]
    table = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    monitored = str(INPUT)
    if not files:
        return f"**Pasta monitorada:** `{monitored}`\n\nNenhum áudio encontrado em {monitored}"
    return f"**Pasta monitorada:** `{monitored}`\n\n**{len(files)} arquivos · {total_seconds/3600:.2f} h · {total_bytes/1024**2:.2f} MB**\n\n| Arquivo | Duração | Tamanho |\n|---|---:|---:|\n{table}"


def open_audio_folder() -> str:
    try:
        os.startfile(str(INPUT))
    except Exception:
        subprocess.Popen(["explorer.exe", str(INPUT)])
    return f"Explorer aberto em `{INPUT}`"


def normalise_audio(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
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


def make_training_audio(source: Path, key: str) -> Path:
    """Qwen's official TTSDataset expects the reference mel at 24 kHz."""
    dest = DATASET / "raw" / f"{key}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
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


def split_segments(wav: Path, stem: str) -> list[Path]:
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
            output.append(SEGMENTS / f"{stem}_{index:04d}.wav")
            sf.write(output[-1], data[a:cut], sr, subtype="PCM_16")
            index += 1; a = cut
        if b - a >= int(0.8 * sr):
            output.append(SEGMENTS / f"{stem}_{index:04d}.wav")
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
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"{SEED}:{r['audio']}".encode()).hexdigest())
    if len(ordered) < 4:
        return ordered, [], []
    test_n = max(1, round(len(ordered) * 0.1)); val_n = max(1, round(len(ordered) * 0.1))
    return ordered[: -(val_n + test_n)], ordered[-(val_n + test_n): -test_n], ordered[-test_n:]


def _report(rows: list[dict[str, Any]], ref: Path | None) -> Path:
    counts = {k: sum(1 for r in rows if r["quality"] == k) for k in ("EXCELENTE", "BOM", "REVISAR", "REJEITADO")}
    useful = sum(r["duration"] for r in rows if r["quality"] in ("EXCELENTE", "BOM"))
    rejected = sum(r["duration"] for r in rows if r["quality"] == "REJEITADO")
    durations = sorted(float(r.get("duration", 0)) for r in rows)
    distribution = f"min {durations[0]:.1f}s · mediana {durations[len(durations)//2]:.1f}s · máx {durations[-1]:.1f}s" if durations else "sem clips"
    body = "".join(f"<tr><td>{html.escape(str(r['file']))}</td><td>{r['duration']:.1f}</td><td>{r['quality']}</td><td>{html.escape(r.get('reason',''))}</td><td>{r.get('confidence',0):.2f}</td><td>{r.get('snr_db',0):.1f}</td><td>{r.get('clipping',0):.4f}</td></tr>" for r in rows)
    out = REPORTS / "dataset_report.html"
    out.write_text(f"<!doctype html><meta charset='utf-8'><title>Relatório IA DE VOZ</title><h1>Relatório do dataset</h1><p>Clips: {len(rows)} · útil: {useful/3600:.2f} h · rejeitado: {rejected/3600:.2f} h<br>Duração: {distribution}</p><p>EXCELENTE {counts['EXCELENTE']} · BOM {counts['BOM']} · REVISAR {counts['REVISAR']} · REJEITADO {counts['REJEITADO']}<br>Referência: {html.escape(str(ref) if ref else 'não selecionada')}</p><table border='1' cellpadding='4'><tr><th>Arquivo</th><th>s</th><th>qualidade</th><th>motivo</th><th>conf.</th><th>SNR</th><th>clipping</th></tr>{body}</table>", encoding="utf-8")
    return out


def prepare_dataset(progress=_GradioProgress(track_tqdm=False) if _GradioProgress else None) -> tuple[str, list[list[Any]], str | None]:
    files = audio_files()
    if not files:
        return "Nenhum áudio em `C:\\IA DE VOZ\\input_audio`.", [], None
    cached: dict[str, dict[str, Any]] = {}
    if REVIEW_FILE.exists():
        try:
            cached = {r.get("segment_hash", r.get("audio", "")): r for r in json.loads(REVIEW_FILE.read_text(encoding="utf-8"))}
        except Exception:
            cached = {}
    rows: list[dict[str, Any]] = []
    for n, source in enumerate(files, 1):
        if progress:
            progress((n - 1) / max(1, len(files)), desc=f"Inspeção/conversão: {source.name}")
        digest = sha256_file(source)
        clean = CLEANED / f"{digest}.wav"
        try:
            if not clean.exists():
                normalise_audio(source, clean)
            segs = split_segments(clean, digest[:12])
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
    REVIEW_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    accepted = [r for r in rows if r["quality"] in ("EXCELENTE", "BOM") and r.get("text")]
    for r in accepted:
        try:
            r["train_audio"] = str(make_training_audio(Path(r["audio"]), r["segment_hash"]))
        except Exception as exc:
            r["quality"] = "REVISAR"; r["reason"] = f"conversão 24 kHz falhou: {exc}"
    accepted = [r for r in rows if r["quality"] in ("EXCELENTE", "BOM") and r.get("text") and r.get("train_audio")]
    ref: Path | None = None
    if accepted:
        chosen = sorted(accepted, key=lambda r: (r["quality"] == "EXCELENTE", r.get("snr_db", 0), r.get("confidence", 0), r.get("duration", 0)), reverse=True)[0]
        ref = REFERENCES / "ref.wav"
        shutil.copy2(chosen["train_audio"], ref)
        (REFERENCES / "ref.txt").write_text(chosen["text"], encoding="utf-8")
    train, validation, test = _split_records(accepted)
    to_json = lambda r: {"audio": str(Path(r.get("train_audio", r["audio"]))).replace("\\", "/"), "text": r["text"], "ref_audio": str(ref).replace("\\", "/") if ref else ""}
    _write_jsonl(DATASET / "train_raw.jsonl", [to_json(r) for r in train])
    _write_jsonl(DATASET / "validation_raw.jsonl", [to_json(r) for r in validation])
    _write_jsonl(DATASET / "test_raw.jsonl", [to_json(r) for r in test])
    for r in rows:
        if r["quality"] == "REJEITADO":
            try: shutil.copy2(r["audio"], REJECTED / Path(r["audio"]).name)
            except Exception: pass
    report = _report(rows, ref)
    token_status = "tokenizer pendente"
    if train and ref:
        token_status = prepare_codes()
    rejected_duration = sum(r["duration"] for r in rows if r["quality"] == "REJEITADO")
    stats = f"**{len(rows)} clips** · aceitos: **{len(accepted)}** · rejeitados: **{sum(r['quality']=='REJEITADO' for r in rows)}** · duração útil: **{sum(r['duration'] for r in accepted)/3600:.2f} h** · rejeitada: **{rejected_duration/3600:.2f} h**\n\nReferência: `{ref or 'não selecionada'}`\n\nJSONL: `train_raw.jsonl`, `validation_raw.jsonl`, `test_raw.jsonl`\n\n{token_status}\n\nRelatório: `{report}`"
    table = [[r.get("file", ""), r.get("duration", 0), r.get("text", ""), r.get("quality", ""), r.get("reason", ""), r.get("confidence", 0), r.get("snr_db", 0), r.get("clipping", 0)] for r in rows]
    return stats, table, str(ref) if ref else None


def validate_jsonl(path: Path) -> tuple[bool, str]:
    if not path.exists(): return False, f"ausente: {path}"
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try: row = json.loads(line)
        except Exception as exc: return False, f"linha {i} inválida: {exc}"
        for key in ("audio", "text", "ref_audio"):
            if not row.get(key): return False, f"linha {i} sem {key}"
        if not Path(row["audio"]).exists() or not Path(row["ref_audio"]).exists(): return False, f"linha {i} referencia arquivo ausente"
    return True, "ok"


def prepare_codes() -> str:
    ok, msg = validate_jsonl(DATASET / "train_raw.jsonl")
    if not ok: return f"Tokenizer não executado: {msg}"
    out = DATASET / "train_with_codes.jsonl"
    if out.exists() and out.stat().st_mtime >= (DATASET / "train_raw.jsonl").stat().st_mtime:
        try:
            rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
            if rows and all(row.get("audio_codes") for row in rows):
                return f"Tokenizer já preparado: `{out}`"
        except Exception:
            pass
    cmd = [sys.executable, str(QWEN_REPO / "finetuning" / "prepare_data.py"), "--device", "cuda:0" if _cuda() else "cpu", "--tokenizer_model_path", TOKENIZER_MODEL, "--input_jsonl", str(DATASET / "train_raw.jsonl"), "--output_jsonl", str(out)]
    try:
        LOGGER.info("tokenizer: %s", cmd)
        subprocess.run(cmd, cwd=str(QWEN_REPO / "finetuning"), check=True, capture_output=True, text=True)
        ok, msg = validate_jsonl(out)
        if ok:
            try:
                lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
                ok = bool(lines) and all(row.get("audio_codes") for row in lines)
                msg = "ok" if ok else "audio_codes ausente"
            except Exception as exc:
                ok, msg = False, str(exc)
        return f"Tokenizer concluído: `{out}`" if ok else f"Tokenizer gerou JSONL inválido: {msg}"
    except subprocess.CalledProcessError as exc:
        LOGGER.error("tokenizer falhou: %s\n%s", exc, exc.stderr)
        return f"Tokenizer falhou; veja `{LOGS / 'app.log'}`. Treinamento bloqueado."


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_tts(model_id: str = MODEL_17B) -> Any:
    key = model_id
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


def generate_voice(text: str, choice: str) -> str:
    if not text or not text.strip(): raise ValueError("Digite um texto em Português.")
    ref = REFERENCES / "ref.wav"
    if not ref.exists(): raise ValueError("Prepare o dataset e selecione uma referência primeiro.")
    model_id = MODEL_06B if "0.6" in choice else MODEL_17B
    model_path = choice if choice and Path(choice).exists() else ensure_model(model_id)
    tts = load_tts(model_path)
    ref_text = (REFERENCES / "ref.txt").read_text(encoding="utf-8") if (REFERENCES / "ref.txt").exists() else None
    if "checkpoint" in model_path.lower():
        wavs, sr = tts.generate_custom_voice(text=text, speaker="speaker_1", language="Portuguese")
    else:
        wavs, sr = tts.generate_voice_clone(text=text, language="Portuguese", ref_audio=str(ref), ref_text=ref_text, x_vector_only_mode=not bool(ref_text), non_streaming_mode=True)
    out = GENERATED / f"{datetime.now():%Y%m%d_%H%M%S}.wav"
    import soundfile as sf
    sf.write(out, wavs[0], sr)
    try:
        score = score_generated(out, text, ref)
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


def evaluate_checkpoints(run_dir: Path) -> str:
    checkpoints = sorted(run_dir.glob("checkpoint-epoch-*"))
    if not checkpoints:
        return "Nenhum checkpoint encontrado para avaliação."
    scores: list[dict[str, Any]] = []
    loss_value = None
    log_path = run_dir / "train.log"
    if log_path.exists():
        losses = re.findall(r"loss=([0-9]+(?:\.[0-9]+)?)", log_path.read_text(encoding="utf-8", errors="ignore"))
        if losses:
            loss_value = float(losses[-1])
    for checkpoint in checkpoints:
        vals = []
        for phrase in BASELINE_PHRASES:
            try:
                out = Path(generate_voice(phrase, str(checkpoint)))
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
    (run_dir / "checkpoint_ranking.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"Ranking salvo em `{run_dir / 'checkpoint_ranking.json'}`; melhor: `{scores[0]['checkpoint']}`."


def generate_baseline() -> str:
    """Generate the fixed zero-shot comparison set once per reference."""
    ref = REFERENCES / "ref.wav"
    if not ref.exists():
        return "Baseline pendente: ainda não há `dataset\\references\\ref.wav`."
    out_dir = GENERATED / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.wav"))
    if len(existing) >= len(BASELINE_PHRASES):
        return f"Baseline já disponível em `{out_dir}`."
    tts = load_tts(ensure_model(MODEL_17B))
    ref_text = (REFERENCES / "ref.txt").read_text(encoding="utf-8") if (REFERENCES / "ref.txt").exists() else None
    import soundfile as sf
    for i, phrase in enumerate(BASELINE_PHRASES, 1):
        wavs, sr = tts.generate_voice_clone(text=phrase, language="Portuguese", ref_audio=str(ref), ref_text=ref_text, x_vector_only_mode=not bool(ref_text), non_streaming_mode=True)
        sf.write(out_dir / f"phrase_{i:02d}.wav", wavs[0], sr)
    return f"Baseline zero-shot gerado em `{out_dir}` com {len(BASELINE_PHRASES)} frases fixas."


def start_training(model_choice: str, method: str, batch: int, grad_acc: int, lr: float, epochs: int, progress=None) -> str:
    if "LoRA" in (method or ""):
        return "LoRA/PEFT permanece experimental e bloqueado: o repositório oficial ainda não fornece um caminho validado para este modelo. Use SFT oficial em GPU compatível."
    train_json = DATASET / "train_with_codes.jsonl"
    ok, msg = validate_jsonl(train_json)
    if not ok: return f"Treinamento bloqueado: {msg}. Clique PREPARAR DATASET primeiro."
    if "0.6" in model_choice:
        return "Treinamento 0.6B bloqueado: o SFT oficial atual assume embeddings de dimensão igual e falha no 0.6B. A opção 0.6B permanece disponível para inferência CPU; use o SFT 1.7B em GPU compatível."
    if not _cuda(): return "Treinamento não iniciado: CUDA não disponível; não vou fingir um SFT."
    baseline_note = generate_baseline()
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        required = 7 * 2**30 if "1.7" in model_choice else 5 * 2**30
        if total < required:
            model_label = "1.7B" if "1.7" in model_choice else "0.6B"
            return f"{baseline_note}\n\nTreinamento não iniciado: VRAM total {total/2**30:.1f} GB; o perfil {model_label} requer pelo menos {required/2**30:.0f} GB para o pre-flight conservador."
    except Exception as exc:
        return f"Pre-flight de VRAM falhou: {exc}"
    run_dir = TRAINING / "runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    model_id = MODEL_06B if "0.6" in model_choice else MODEL_17B
    model_path = ensure_model(model_id)
    cmd = [sys.executable, str(ROOT / "scripts" / "train_runner.py"), "--init_model_path", model_path, "--output_model_path", str(run_dir), "--train_jsonl", str(train_json), "--batch_size", str(max(1, int(batch))), "--gradient_accumulation", str(max(1, int(grad_acc))), "--lr", str(lr), "--num_epochs", str(max(1, int(epochs))), "--speaker_name", "speaker_1"]
    global TRAIN_PROCESS
    log = run_dir / "train.log"
    LOGGER.info("treino: %s", cmd)
    with log.open("w", encoding="utf-8") as f:
        TRAIN_PROCESS = subprocess.Popen(cmd, cwd=str(QWEN_REPO / "finetuning"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        started = time.time()
        if TRAIN_PROCESS.stdout:
            for line in TRAIN_PROCESS.stdout:
                f.write(line); f.flush()
                if progress:
                    progress(min(0.99, (time.time() - started) / max(1, int(epochs) * 600)), desc=line.strip()[:120])
        code = TRAIN_PROCESS.wait()
    TRAIN_PROCESS = None
    ranking = evaluate_checkpoints(run_dir) if code == 0 else "Ranking não executado porque o treino falhou."
    return f"Treinamento {'concluído' if code == 0 else 'falhou'} (código {code}). Run: `{run_dir}`. Log: `{log}`\n\n{ranking}"


def stop_training() -> str:
    global TRAIN_PROCESS
    if TRAIN_PROCESS and TRAIN_PROCESS.poll() is None:
        TRAIN_PROCESS.terminate(); return "Solicitada parada segura; checkpoint anterior permanece intacto."
    return "Nenhum treinamento em execução."


def review_rows() -> list[list[Any]]:
    if not REVIEW_FILE.exists(): return []
    try:
        rows = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
        return [[r.get("file", ""), r.get("duration", 0), r.get("text", ""), r.get("quality", ""), r.get("reason", ""), r.get("confidence", 0), r.get("snr_db", 0), r.get("clipping", 0)] for r in rows]
    except Exception: return []


def rebuild_dataset(table: list[list[Any]]) -> tuple[str, list[list[Any]]]:
    if not REVIEW_FILE.exists(): return "Nada para reconstruir.", []
    original = {r.get("audio"): r for r in json.loads(REVIEW_FILE.read_text(encoding="utf-8"))}
    for row in table or []:
        if len(row) < 4: continue
        matches = [r for r in original.values() if r.get("file") == row[0] and abs(float(r.get("duration", 0)) - float(row[1] or 0)) < 0.01]
        if matches:
            matches[0]["text"] = str(row[2] or "").strip(); matches[0]["quality"] = str(row[3] or "REVISAR").upper(); matches[0]["reason"] = str(row[4] or "edição manual")
    REVIEW_FILE.write_text(json.dumps(list(original.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    # Re-run only the cheap deterministic export; ASR/audio work stays cached.
    rows = list(original.values()); accepted = [r for r in rows if r["quality"] in ("EXCELENTE", "BOM") and r.get("text")]
    ref = REFERENCES / "ref.wav" if (REFERENCES / "ref.wav").exists() else None
    for r in accepted:
        if not r.get("train_audio"):
            try: r["train_audio"] = str(make_training_audio(Path(r["audio"]), r["segment_hash"]))
            except Exception: continue
    accepted = [r for r in accepted if r.get("train_audio")]
    train, val, test = _split_records(accepted)
    to_json = lambda r: {"audio": str(Path(r.get("train_audio", r["audio"]))).replace("\\", "/"), "text": r["text"], "ref_audio": str(ref).replace("\\", "/") if ref else ""}
    _write_jsonl(DATASET / "train_raw.jsonl", [to_json(r) for r in train]); _write_jsonl(DATASET / "validation_raw.jsonl", [to_json(r) for r in val]); _write_jsonl(DATASET / "test_raw.jsonl", [to_json(r) for r in test])
    status = prepare_codes() if accepted and ref else "Tokenizer bloqueado: sem referência ou clips aceitos."
    return f"Dataset reconstruído: {len(accepted)} aceitos. {status}", review_rows()


def build_ui() -> Any:
    import gradio as gr
    with gr.Blocks(title="IA DE VOZ — Qwen3-TTS", analytics_enabled=False) as demo:
        gr.Markdown("# IA DE VOZ\nEstação local para dataset, fine-tuning e geração PT-BR. **Os áudios permanecem neste computador.** Use somente voz autorizada.")
        with gr.Tab("SISTEMA"):
            system_box = gr.Markdown(diagnose_system())
            gr.Button("DIAGNOSTICAR SISTEMA", variant="primary").click(diagnose_system, outputs=system_box)
        with gr.Tab("ÁUDIOS"):
            audio_box = gr.Markdown(audio_inventory())
            with gr.Row():
                gr.Button("ATUALIZAR").click(audio_inventory, outputs=audio_box)
                open_folder_status = gr.Markdown()
                gr.Button("ABRIR PASTA DE ÁUDIOS").click(open_audio_folder, outputs=open_folder_status)
        with gr.Tab("PREPARAR DATASET"):
            gr.Markdown("Fluxo: Inspeção → conversão → VAD → limpeza → transcrição → filtragem → referência → JSONL → tokenizer")
            prep_status = gr.Markdown()
            prep_table = gr.Dataframe(headers=["arquivo", "duração", "transcrição", "qualidade", "motivo", "confidence", "ruído/SNR", "clipping"], interactive=False)
            ref_player = gr.Audio(label="Referência selecionada", type="filepath")
            gr.Button("PREPARAR DATASET", variant="primary").click(prepare_dataset, outputs=[prep_status, prep_table, ref_player])
        with gr.Tab("REVISÃO"):
            review_table = gr.Dataframe(value=review_rows(), headers=["arquivo", "duração", "transcrição", "qualidade", "motivo", "confidence", "ruído/SNR", "clipping"], interactive=True)
            review_status = gr.Markdown()
            gr.Button("RECONSTRUIR DATASET", variant="primary").click(rebuild_dataset, inputs=review_table, outputs=[review_status, review_table])
            with gr.Row():
                review_path = gr.Textbox(label="Caminho do clip para ouvir", placeholder="C:\\IA DE VOZ\\dataset\\segments\\clip.wav")
                original_player = gr.Audio(label="Original/segmento", type="filepath")
                cleaned_player = gr.Audio(label="Cleaned", type="filepath")
            def _load_review_audio(path):
                p = Path(path or "")
                if not p.exists(): return None, None
                clean = CLEANED / p.name
                return str(p), str(clean) if clean.exists() else str(p)
            gr.Button("OUVIR CLIP").click(_load_review_audio, inputs=review_path, outputs=[original_player, cleaned_player])
            gr.Markdown("Edite transcrição/classificação na tabela e reconstrua. Os rejeitados ficam em `dataset\\rejected`.")
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
                gr.Button("INICIAR TREINAMENTO", variant="primary").click(start_training, inputs=[model_choice, method, batch, grad, lr, epochs], outputs=train_status)
                gr.Button("PARAR TREINAMENTO").click(stop_training, outputs=train_status)
            gr.Markdown("O pre-flight interrompe sem iniciar quando a VRAM não comporta o perfil. Cada execução usa uma pasta `training\\runs\\AAAA-MM-DD_HH-MM-SS`.")
        with gr.Tab("TESTAR VOZ"):
            text = gr.Textbox(value="Olá. Esta é uma frase de teste em Português Brasileiro.", label="Texto", lines=3)
            gen_model = gr.Dropdown(["Base zero-shot (1.7B)", "Base zero-shot (0.6B)"], value="Base zero-shot (1.7B)", label="Modelo/checkpoint")
            checkpoint_path = gr.Textbox(label="Checkpoint opcional", placeholder="C:\\IA DE VOZ\\training\\runs\\...\\checkpoint-epoch-1")
            audio_out = gr.Audio(label="Saída", type="filepath")
            gen_status = gr.Markdown()
            def _generate(t, c, cp):
                try: return generate_voice(t, cp.strip() if cp and Path(cp.strip()).exists() else c), "Geração concluída."
                except Exception as exc: LOGGER.exception("inferência falhou"); return None, f"Geração não executada: {exc}"
            gr.Button("GERAR", variant="primary").click(_generate, inputs=[text, gen_model, checkpoint_path], outputs=[audio_out, gen_status])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
