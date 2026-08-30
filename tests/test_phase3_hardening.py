import os
import tempfile
import unittest
from pathlib import Path

import app


class Phase3HardeningTests(unittest.TestCase):
    def test_audio_discovery_rejects_reparse_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            store = app.ProfileStore(root / "voices", root / "state.json", root / "legacy")
            previous_store = app.PROFILE_STORE
            app.PROFILE_STORE = store
            try:
                ctx = store.create("Reparse")
                outside = root / "outside"
                outside.mkdir()
                (outside / "voice.wav").write_bytes(b"not real audio")
                link = ctx.input_audio / "linked"
                try:
                    os.symlink(outside, link, target_is_directory=True)
                except (NotImplementedError, OSError):
                    marker = ctx.input_audio / "synthetic-reparse-marker"
                    marker.touch()
                    original = app._is_reparse
                    app._is_reparse = lambda path: path == marker
                    try:
                        with self.assertRaises(app.ProfileError):
                            app._assert_no_reparse_chain(marker)
                    finally:
                        app._is_reparse = original
                else:
                    with self.assertRaises(app.ProfileError):
                        app.audio_files(ctx.voice_id)
            finally:
                app.PROFILE_STORE = previous_store


if __name__ == "__main__":
    unittest.main()
