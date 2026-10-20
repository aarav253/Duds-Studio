"""Local / free text-to-speech for narration audio.

Primary: edge-tts (free neural voices, needs internet). Falls back to fully
offline Windows SAPI (pyttsx3) if edge-tts fails for any reason.

synthesize_with_word_timings() also writes a <name>_words.json sidecar with
edge-tts's real per-word timestamps, so captions follow actual speech pauses.
The SAPI fallback has no per-word data (captions then use a proportional estimate).

    python src/tts.py "Some text to speak" narration_audio/test.wav
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

# BrianNeural over ChristopherNeural: Christopher reads as the generic "AI
# narrator" cliche; Brian's casual register fits first-person content better.
DEFAULT_VOICE = "en-US-BrianNeural"


def words_sidecar_path(out_path):
    p = Path(out_path)
    return p.with_name(p.stem + "_words.json")


def _to_wav(src_mp3, out_wav):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src_mp3), "-ar", "44100",
                    "-ac", "2", str(out_wav)], check=True)


def _rate_string(wpm):
    """Map a words-per-minute setting (SAPI-style) to an edge-tts rate offset."""
    if not wpm:
        return "+0%"
    pct = int(round((wpm / 160.0 - 1.0) * 100))
    return f"{pct:+d}%"


async def _edge_tts(text, mp3_path, voice, rate):
    import edge_tts
    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    words = []
    with open(mp3_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7        # 100ns units -> seconds
                dur = chunk["duration"] / 1e7
                words.append({"word": chunk["text"], "start": round(start, 3),
                              "end": round(start + dur, 3)})
    return words


def _sapi(text, out_wav, rate_wpm):
    import pyttsx3
    engine = pyttsx3.init()
    if rate_wpm:
        engine.setProperty("rate", int(rate_wpm))
    engine.save_to_file(text, str(out_wav))
    engine.runAndWait()


def synthesize_with_word_timings(text, out_path, voice=DEFAULT_VOICE, rate_wpm=None):
    """Render text to a .wav. Returns (out_path, engine_name)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    side = words_sidecar_path(out)
    tmp_mp3 = out.with_suffix(".edge.mp3")
    try:
        words = asyncio.run(_edge_tts(text, tmp_mp3, voice, _rate_string(rate_wpm)))
        if tmp_mp3.stat().st_size == 0:
            raise RuntimeError("edge-tts returned no audio")
        _to_wav(tmp_mp3, out)
        side.write_text(json.dumps(words, indent=1), encoding="utf-8")
        return out, "edge-tts"
    except Exception as e:  # noqa: BLE001 - any failure means fall back
        print(f"  [tts] edge-tts failed ({e}); falling back to offline SAPI")
        if side.exists():
            side.unlink()  # a stale sidecar would mis-time captions
        _sapi(text, out, rate_wpm)
        return out, "sapi"
    finally:
        if tmp_mp3.exists():
            tmp_mp3.unlink()


def synthesize(text, out_path, voice=DEFAULT_VOICE, rate_wpm=None):
    return synthesize_with_word_timings(text, out_path, voice, rate_wpm)[0]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('usage: python src/tts.py "text" narration_audio/out.wav')
        sys.exit(1)
    path, engine = synthesize_with_word_timings(sys.argv[1], sys.argv[2])
    print(f"{path} ({engine})")
