"""Download a "<character> voice lines for edits" video and split it into one
file per line.

  1. audio via yt-dlp (src/fetch.py)
  2. if the video has background music, the voice is isolated with Demucs
     (htdemucs_ft); clean voiceline videos are left untouched
  3. faster-whisper finds each line (VAD + timestamps) and transcribes it;
     a second pass translates it to English for the file name
  4. lines/<NN>_<english_meaning>.wav + lines/index.json
     ({file, start, end, duration, text, meaning, source})

    python src/voicelines.py "<Character folder>" <url> [<url> ...]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import fetch  # noqa: E402

_WHISPER = None


def whisper():
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        _WHISPER = WhisperModel("small", device="cpu", compute_type="int8")
    return _WHISPER


def music_share(y, sr):
    """fraction of the energy that is NOT voice (0 = clean voicelines)."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    m = get_model("htdemucs_ft"); m.eval()
    x = torch.tensor(y.T.astype(np.float32))[None]
    sd = x.mean(1).std() + 1e-8
    with torch.no_grad():
        o = apply_model(m, x / sd, device="cpu", progress=False, overlap=0.25)[0] * sd
    voc = o[m.sources.index("vocals")].numpy().T
    rest = y - voc
    return float(np.sqrt((rest ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-9)), voc


def slug(s, n=48):
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:n].rstrip("_") or "line"


def process(char_dir, urls):
    char_dir = ROOT / char_dir
    vdir = char_dir / "voicelines"
    lines_dir = vdir / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for ui, url in enumerate(urls, 1):
        r = fetch(url, mode="audio", to=str(vdir.relative_to(ROOT)), name=f"source_{ui}")
        src = ROOT / r["wav"]
        y, sr = sf.read(str(src), always_2d=True)
        share, voc = music_share(y, sr)
        use = y
        if share > 0.35:                                     # music under the lines -> isolate the voice
            use = voc
            sf.write(str(src.with_name(src.stem + "_voice.wav")), voc, sr)
        print(f"[{char_dir.name}] source {ui}: {r['title'][:60]} | {len(y)/sr:.0f}s | non-voice share {share:.2f}"
              f" -> {'voice isolated' if use is voc else 'used as-is'}", flush=True)
        tmp = src.with_name(src.stem + "_use.wav")
        sf.write(str(tmp), use, sr)
        w = whisper()
        segs, info = w.transcribe(str(tmp), vad_filter=True, vad_parameters={"min_silence_duration_ms": 400})
        segs = [s for s in segs if s.end - s.start >= 0.35]
        for s in segs:
            a, b = max(0.0, s.start - 0.06), min(len(use) / sr, s.end + 0.2)
            clip = use[int(a * sr):int(b * sr)].copy()
            # trim trailing silence (whisper segments can run long)
            env = np.abs(clip).max(1)
            loud = np.nonzero(env > 0.02 * max(env.max(), 1e-6))[0]
            if len(loud):
                clip = clip[:min(len(clip), loud[-1] + int(0.15 * sr))]
            if len(clip) < int(0.3 * sr) or np.abs(clip).max() < 0.01:
                continue
            f = min(int(0.01 * sr), len(clip) // 4)
            clip[:f] *= np.linspace(0, 1, f)[:, None]
            clip[-f:] *= np.linspace(1, 0, f)[:, None]
            # English meaning for the file name
            tr, _ = w.transcribe(_to16k(clip, sr), task="translate",
                                 language=info.language if info.language_probability > 0.5 else None)
            meaning = " ".join(t.text.strip() for t in tr).strip() or s.text.strip()
            k = len(index) + 1
            name = f"{k:02d}_{slug(meaning)}.wav"
            sf.write(str(lines_dir / name), clip, sr)
            index.append({"file": name, "source": ui, "start": round(a, 2), "end": round(a + len(clip) / sr, 2),
                          "duration": round(len(clip) / sr, 2), "text": s.text.strip(), "meaning": meaning,
                          "language": info.language})
        tmp.unlink()
    (lines_dir / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    return index


def _to16k(clip, sr):
    from math import gcd
    from scipy.signal import resample_poly                 # (librosa/numba is blocked by Windows App Control)
    g = gcd(16000, sr)
    return resample_poly(clip.mean(1), 16000 // g, sr // g).astype(np.float32)


if __name__ == "__main__":
    idx = process(sys.argv[1], sys.argv[2:])
    for e in idx:
        print(f"  {e['file']:52s} {e['duration']:4.1f}s  {e['meaning'][:70]}")
