import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class Phase2ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = app.ProfileStore(root / "voices", root / "state.json", root / "legacy")
        self.previous_store = app.PROFILE_STORE
        app.PROFILE_STORE = self.store

    def tearDown(self):
        app.PROFILE_STORE = self.previous_store
        self.temp.cleanup()

    def test_profile_lifecycle_persists_and_renames_without_moving_data(self):
        first = self.store.create("Biel")
        second = self.store.create("Friend")
        marker = first.input_audio / "kept.wav"
        marker.write_bytes(b"profile-a")
        self.store.select(second.voice_id)
        self.store.select(first.voice_id)
        self.store.rename(first.voice_id, "Biel renomeado")

        reopened = app.ProfileStore(self.store.root, self.store.state_file, self.store.legacy_root)
        self.assertEqual(reopened.active_id(), first.voice_id)
        self.assertEqual(reopened.get(first.voice_id).input_audio / "kept.wav", marker)
        self.assertEqual(dict((voice_id, name) for name, voice_id in reopened.list_profiles())[first.voice_id], "Biel renomeado")
        for confirmation in ("wrong", second.voice_id, f"DELETE {second.voice_id}", "DELETE VOICE"):
            with self.subTest(confirmation=confirmation), self.assertRaises(app.ProfileError):
                reopened.delete(second.voice_id, confirmation)
        reopened.delete(second.voice_id, "  DeLeTe  ")

    def test_profile_paths_reject_traversal_absolute_unc_and_drive(self):
        ctx = self.store.create("Safe")
        for value in ("../outside.wav", "dataset/../../outside.wav", "/outside.wav", r"\\server\share\x.wav", r"C:\outside.wav", "C:relative.wav"):
            with self.subTest(value=value), self.assertRaises(app.ProfileError):
                ctx.path(value)
        other = self.store.create("Other")
        with self.assertRaises(app.ProfileError):
            app.profile_relative(ctx, other.input_audio / "x.wav")

    def test_profile_callbacks_verify_disk_writes_and_report_failures(self):
        first = self.store.create("Antes")
        second = self.store.create("Outra")
        self.store.select(first.voice_id)

        renamed = app.rename_voice(first.voice_id, "Depois")
        self.assertIn("Salvo no disco", renamed[0])
        selected = app.select_voice(second.voice_id)
        self.assertIn("Salvo no disco", selected[-1])

        with patch("app._atomic_json"):
            failed_rename = app.rename_voice(second.voice_id, "Não salvo")
            failed_select = app.select_voice(first.voice_id)
        self.assertIn("Renomeação recusada", failed_rename[0])
        self.assertIn("Seleção recusada", failed_select[-1])

    def test_migration_rewrites_all_known_path_fields_and_is_idempotent(self):
        legacy = self.store.legacy_root
        original = legacy / "input_audio" / "speaker.wav"
        original.parent.mkdir(parents=True)
        original.write_bytes(b"original")
        checkpoint = legacy / "training" / "runs" / "run-1" / "checkpoint-epoch-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        fields = {field: str(original) for field in ("audio", "source", "train_audio", "original", "cleaned", "segment", "ref_audio")}
        fields["checkpoint"] = str(checkpoint)
        review = legacy / "dataset" / "review.json"
        review.parent.mkdir(parents=True)
        review.write_text(json.dumps([{"file": "speaker.wav", **fields}]), encoding="utf-8")
        jsonl = legacy / "dataset" / "train_raw.jsonl"
        jsonl.write_text(json.dumps(fields) + "\n", encoding="utf-8")
        before = original.read_bytes()

        migrated, status = self.store.migrate_legacy()
        self.assertIn("Migração concluída", status)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(migrated.input_audio.joinpath("speaker.wav").read_bytes(), before)
        migrated_review = json.loads((migrated.dataset / "review.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(migrated_review["file"], "speaker.wav")
        for field in fields:
            self.assertFalse(Path(migrated_review[field]).is_absolute())
            self.assertNotIn(str(legacy), migrated_review[field])
        migrated_jsonl = json.loads((migrated.dataset / "train_raw.jsonl").read_text(encoding="utf-8"))
        for field in fields:
            self.assertFalse(Path(migrated_jsonl[field]).is_absolute())
            self.assertNotIn(str(legacy), migrated_jsonl[field])
        same, second_status = self.store.migrate_legacy()
        self.assertEqual(same.voice_id, migrated.voice_id)
        self.assertIn("idêntico", second_status)
        self.assertEqual(len(self.store.list_profiles()), 1)

    def test_jsonl_cross_profile_and_stale_callback_are_rejected(self):
        first = self.store.create("A")
        second = self.store.create("B")
        first_audio = first.dataset / "cleaned" / "a.wav"
        first_ref = first.dataset / "references" / "ref.wav"
        second_audio = second.dataset / "cleaned" / "b.wav"
        first_audio.write_bytes(b"a")
        first_ref.write_bytes(b"ref")
        second_audio.write_bytes(b"b")
        raw = first.dataset / "train.jsonl"
        raw.write_text(json.dumps({"audio": str(second_audio), "text": "x", "ref_audio": str(first_ref)}) + "\n", encoding="utf-8")
        self.store.select(first.voice_id)
        ok, message = app.validate_jsonl(raw, first)
        self.assertFalse(ok)
        self.assertIn("absoluto", message)
        self.store.select(second.voice_id)
        with self.assertRaises(app.ProfileError):
            app.callback_context(first.voice_id)

    def test_profile_scoped_audio_checkpoints_and_run_metadata(self):
        first = self.store.create("A")
        second = self.store.create("B")
        (first.input_audio / "a.wav").write_bytes(b"a")
        (second.input_audio / "b.wav").write_bytes(b"b")
        first_checkpoint = first.training / "runs" / "a" / "checkpoint-epoch-1"
        second_checkpoint = second.training / "runs" / "b" / "checkpoint-epoch-1"
        first_checkpoint.mkdir(parents=True)
        second_checkpoint.mkdir(parents=True)
        self.store.select(first.voice_id)
        self.assertEqual([p.name for p in app.audio_files()], ["a.wav"])
        self.assertEqual(app.checkpoint_choices(), [app.profile_relative(first, first_checkpoint)])
        app._atomic_json(first.training / "runs" / "a" / "run.json", {"voice_id": first.voice_id})
        self.assertEqual(json.loads((first.training / "runs" / "a" / "run.json").read_text(encoding="utf-8"))["voice_id"], first.voice_id)
        with self.assertRaises(app.ProfileError):
            app._checkpoint_for(first, app.profile_relative(second, second_checkpoint))
        self.store.select(second.voice_id)
        self.assertEqual([p.name for p in app.audio_files()], ["b.wav"])
        self.assertEqual(app.checkpoint_choices(), [app.profile_relative(second, second_checkpoint)])


if __name__ == "__main__":
    unittest.main()
