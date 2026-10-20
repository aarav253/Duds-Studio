"""Remove spoken character lines from a reference edit's audio while keeping
the song: separate vocals (htdemucs_ft), find speech with faster-whisper on the
vocal stem, and subtract the vocal stem only inside those ranges (40 ms ramps).
Writes music/template_<n>_song.wav (+ _song.json with the ranges) so the
engines pick it up like a matched clean song.

    python src/strip_voices.py <n> [<n> ...]
"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
ROOT = Path(__file__).resolve().parent.parent

def strip(n):
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from faster_whisper import WhisperModel
    src = ROOT / f"music/template_{n}_ref.wav"
    y, sr = sf.read(str(src), always_2d=True)
    m = get_model("htdemucs_ft"); m.eval()
    x = torch.tensor(y.T.astype(np.float32))[None]; sd = x.mean(1).std() + 1e-8
    with torch.no_grad():
        o = apply_model(m, x / sd, device="cpu", progress=False, overlap=0.5)[0] * sd
    voc = o[m.sources.index("vocals")].numpy().T
    vp = ROOT / f"music/template_{n}_vocals_ft.wav"; sf.write(str(vp), voc, sr)
    w = WhisperModel("small", device="cpu", compute_type="int8")
    segs, info = w.transcribe(str(vp), vad_filter=True, vad_parameters={"min_silence_duration_ms": 300}, word_timestamps=True)
    rng = []
    for s in segs:
        if s.no_speech_prob > 0.7 or s.avg_logprob < -1.3:
            continue
        rng.append((max(0.0, s.start - 0.1), min(len(y) / sr, s.end + 0.2)))
    gate = np.zeros(len(y), np.float32); k = int(0.04 * sr)
    for a, b in rng:
        ia, ib = int(a * sr), int(b * sr); gate[ia:ib] = 1
        gate[max(0, ia - k):ia] = np.maximum(gate[max(0, ia - k):ia], np.linspace(0, 1, ia - max(0, ia - k)))
        gate[ib:ib + k] = np.maximum(gate[ib:ib + k], np.linspace(1, 0, len(gate[ib:ib + k])))
    out = y - voc * gate[:, None]
    sf.write(str(ROOT / f"music/template_{n}_song.wav"), out.astype(np.float32), sr, subtype="PCM_24")
    (ROOT / f"music/template_{n}_song.json").write_text(json.dumps({"ok": True, "method": "voices stripped from reference audio",
        "speech_ranges": [[round(a, 2), round(b, 2)] for a, b in rng], "language": info.language}, indent=1), encoding="utf-8")
    tot = sum(b - a for a, b in rng)
    print(f"T{n}: {len(rng)} spoken stretches ({tot:.1f}s, lang {info.language}) removed: {[(round(a,1), round(b,1)) for a, b in rng]}", flush=True)

if __name__ == "__main__":
    for n in sys.argv[1:]:
        strip(int(n))
