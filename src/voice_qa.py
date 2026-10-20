"""Score every voiceline: duration, speech confidence (faster-whisper no-speech
probability + avg log-prob) and pitch (share of voiced frames above 400 Hz -
a meow/squeal, not a male speaking voice). Writes qa fields into each
character's lines/index.json and prints suspicious lines."""
import json, sys
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
ROOT = Path(__file__).resolve().parent.parent
from faster_whisper import WhisperModel
W = WhisperModel("small", device="cpu", compute_type="int8")

def f0_track(x, sr=16000, fr=1024, hop=256):
    out = []
    for i in range(0, len(x) - fr, hop):
        s = x[i:i + fr] * np.hanning(fr)
        if np.sqrt((s ** 2).mean()) < 0.01: continue
        ac = np.correlate(s, s, "full")[fr - 1:]
        lo, hi = sr // 1000, sr // 70
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.35 * ac[0]: out.append(sr / k)
    return np.array(out)

for idx in sorted((ROOT / "panels/Blue Lock").glob("*/voicelines/lines/index.json")):
    rows = json.loads(idx.read_text(encoding="utf-8"))
    for r in rows:
        x, sr = sf.read(str(idx.parent / r["file"]), always_2d=True); x = x.mean(1)
        g = gcd(16000, sr); x16 = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        f0 = f0_track(x16)
        hi = float((f0 > 400).mean()) if len(f0) else 1.0
        segs, info = W.transcribe(x16, language=r.get("language") if r.get("language") in ("ja", "en") else None)
        segs = list(segs)
        nsp = max([s.no_speech_prob for s in segs], default=1.0); lp = min([s.avg_logprob for s in segs], default=-9)
        bad = hi > 0.35 or nsp > 0.6 or lp < -1.2 or not segs or "thank you for watching" in r["meaning"].lower() or r["meaning"].strip().lower() in ("line", "eugh", "ah")
        r.update(qa={"high_pitch": round(hi, 2), "no_speech": round(nsp, 2), "logprob": round(lp, 2), "median_f0": round(float(np.median(f0)) if len(f0) else 0), "ok": not bad})
        if bad: print(f"  BAD {idx.parent.parent.parent.name:16s} {r['file']:50s} hiF0 {hi:.2f} nospeech {nsp:.2f} lp {lp:.2f} f0 {r['qa']['median_f0']}")
    idx.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    ok = [r for r in rows if r["qa"]["ok"]]
    print(f"{idx.parent.parent.parent.name:16s} {len(ok)}/{len(rows)} clean; short (<=2.5s): {[r['file'][:28] for r in ok if r['duration'] <= 2.5]}", flush=True)
