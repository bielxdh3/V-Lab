from __future__ import annotations

import json
import math
import sys
import wave
from pathlib import Path

ROOT = Path(r"C:\IA DE VOZ")
sys.path.insert(0, str(ROOT))
import app


def make_fixture() -> Path:
    out = ROOT / "dataset" / "test_fixtures" / "smoke_ptbr.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000; seconds = 2.0
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        for i in range(int(sr * seconds)):
            sample = int(0.12 * 32767 * math.sin(2 * math.pi * 220 * i / sr))
            w.writeframes(sample.to_bytes(2, "little", signed=True))
    return out


def main() -> None:
    fixture = make_fixture()
    converted = ROOT / "dataset" / "test_fixtures" / "smoke_converted.wav"
    app.normalise_audio(fixture, converted)
    parts = app.split_segments(converted, "smoke")
    assert parts and all(p.exists() for p in parts)
    m = app.metrics(parts[0]); assert m["sample_rate"] == 16000 and m["duration"] > 0
    print(json.dumps({"cuda": app._cuda(), "ffmpeg": app.ffmpeg_exe(), "segments": len(parts), "metrics": m}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
