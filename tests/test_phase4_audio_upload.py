import tempfile
import unittest
from pathlib import Path

import app


class AudioUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = app.ProfileStore(root / "voices", root / "state.json", root / "legacy")
        self.previous_store = app.PROFILE_STORE
        app.PROFILE_STORE = self.store

    def tearDown(self):
        app.PROFILE_STORE = self.previous_store
        self.temp.cleanup()

    def test_upload_copies_to_selected_profile_and_keeps_source(self):
        voice = self.store.create("Pessoa A")
        source = Path(self.temp.name) / "drop" / "fala.wav"
        source.parent.mkdir()
        source.write_bytes(b"audio-a")
        before = source.read_bytes()

        status, inventory = app.upload_audio([source], voice.voice_id)

        target = voice.input_audio / source.name
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(source.read_bytes(), before)
        self.assertIn("copiado: fala.wav", status)
        self.assertIn("1 arquivos", inventory)

    def test_upload_uses_deterministic_suffix_for_different_duplicate(self):
        voice = self.store.create("Pessoa A")
        first = Path(self.temp.name) / "first.wav"
        second = Path(self.temp.name) / "second.wav"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        first_named = Path(self.temp.name) / "same.wav"
        first_named.write_bytes(first.read_bytes())

        app.upload_audio(first_named, voice.voice_id)
        first_named.write_bytes(second.read_bytes())
        status, _ = app.upload_audio(first_named, voice.voice_id)

        digest = app.sha256_file(second)
        target = voice.input_audio / f"same__{digest[:12]}.wav"
        self.assertTrue(target.is_file())
        self.assertEqual((voice.input_audio / "same.wav").read_bytes(), b"first")
        self.assertEqual(target.read_bytes(), b"second")
        self.assertIn(target.name, status)

    def test_upload_rejects_extension_mp4_without_audio_and_unsafe_destination(self):
        voice = self.store.create("Pessoa A")
        unsupported = Path(self.temp.name) / "bad.txt"
        unsupported.write_bytes(b"not audio")
        video = Path(self.temp.name) / "silent.mp4"
        video.write_bytes(b"mp4")
        original_probe = app.has_audio_stream
        app.has_audio_stream = lambda path: False
        try:
            status, _ = app.upload_audio([unsupported, video], voice.voice_id)
        finally:
            app.has_audio_stream = original_probe

        self.assertIn("extensão não suportada", status)
        self.assertIn("MP4 rejeitado", status)
        with self.assertRaises(app.ProfileError):
            app._upload_target(voice, r"..\outside.wav", "a" * 64)
        self.assertEqual(app.audio_files(voice.voice_id), [])

    def test_overview_and_active_copy_are_per_voice(self):
        first = self.store.create("Pessoa A")
        second = self.store.create("Pessoa B")
        source = Path(self.temp.name) / "voice.wav"
        source.write_bytes(b"a")
        self.store.select(first.voice_id)
        app.upload_audio(source, first.voice_id)

        overview = app.voice_overview()
        self.assertIn("Pessoa A", overview)
        self.assertIn("Pessoa B", overview)
        self.assertIn("✅ Ativa", overview)
        self.assertEqual(len(app.audio_files(first.voice_id)), 1)
        self.assertEqual(app.audio_files(second.voice_id), [])
        self.assertIn("voice_id:", app.active_voice_label(first.voice_id))
        self.assertIn("troque primeiro", app.active_voice_help(first.voice_id))

    def test_profile_labels_mark_migration_and_disambiguate_only_duplicates(self):
        migrated_a = self.store.create("Existing Voice")
        migrated_b = self.store.create("Existing Voice")
        separate = self.store.create("Existing Voice")
        for ctx in (migrated_a, migrated_b):
            data = app.profile_data(ctx)
            data["migration"] = {"source": "legacy-global"}
            app.save_profile_data(ctx, data)

        labels = dict((voice_id, label) for label, voice_id in app.profile_choices())
        self.store.select(migrated_a.voice_id)
        source = Path(self.temp.name) / "label-check.wav"
        source.write_bytes(b"audio")
        status, inventory = app.upload_audio(source, migrated_a.voice_id)

        self.assertEqual(labels[separate.voice_id], "Existing Voice")
        self.assertEqual(labels[migrated_a.voice_id], f"Existing Voice (migrada) · {migrated_a.voice_id[:6]}")
        self.assertEqual(labels[migrated_b.voice_id], f"Existing Voice (migrada) · {migrated_b.voice_id[:6]}")
        self.assertIn(labels[migrated_a.voice_id], status)
        self.assertIn(labels[migrated_a.voice_id], inventory)

    def test_upload_is_wired_to_confirmation_button_not_file_drop(self):
        self.store.create("Pessoa A")
        demo = app.build_ui()
        components = demo.config["components"]
        file_id = next(c["id"] for c in components if c["type"] == "file")
        button_id = next(c["id"] for c in components if c["type"] == "button" and c["props"].get("value") == "ADICIONAR ÁUDIOS À VOZ ATIVA")
        upload_events = [d for d in demo.config["dependencies"] if d.get("api_name") == "upload_audio"]

        self.assertEqual(len(upload_events), 1)
        self.assertEqual(upload_events[0]["targets"], [(button_id, "click")])
        self.assertNotEqual(upload_events[0]["targets"], [(file_id, "upload")])

    def test_stale_upload_callback_cannot_route_into_new_active_voice(self):
        first = self.store.create("Pessoa A")
        second = self.store.create("Pessoa B")
        source = Path(self.temp.name) / "stale.wav"
        source.write_bytes(b"stale")
        self.store.select(first.voice_id)
        self.store.select(second.voice_id)

        status, inventory = app.upload_audio(source, first.voice_id)

        self.assertIn("callback obsoleto", status)
        self.assertIn("Pessoa B", inventory)
        self.assertEqual(app.audio_files(first.voice_id), [])
        self.assertEqual(app.audio_files(second.voice_id), [])


if __name__ == "__main__":
    unittest.main()
