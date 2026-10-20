"""Write music/template_<n>_song.wav: the matched clean song, played at the
editor's speed, starting where the edit starts, level-matched to the reference.

    python src/song_prep.py <n> [<n> ...]      (uses music/template_<n>_song.json)
"""
import json, sys
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
ROOT = Path(__file__).resolve().parent.parent

def prep(n, extra=4.0):
    info = json.loads((ROOT / f"music/template_{n}_song.json").read_text(encoding="utf-8"))
    b = info["best"]
    y, sr = sf.read(str(ROOT / b["file"]), always_2d=True)
    if sr != 44100:
        g = gcd(44100, sr); y = resample_poly(y, 44100 // g, sr // g, axis=0); sr = 44100
    r = b["rate"]
    if r != 1.0:
        up, down = 1000, int(round(1000 * r)); g = gcd(up, down)
        y = resample_poly(y, up // g, down // g, axis=0)
    ref, _ = sf.read(str(ROOT / f"music/template_{n}_ref.wav"), always_2d=True)
    o = int(round(b["offset_played"] * sr))
    pad = max(0, -o)
    seg = y[max(0, o):max(0, o) + len(ref) + int(extra * sr)]
    if pad:
        seg = np.concatenate([np.zeros((pad, seg.shape[1])), seg])
    n0 = min(len(ref), len(seg)) - int(0.5 * sr)
    g = np.sqrt((ref[:n0] ** 2).mean()) / (np.sqrt((seg[:n0] ** 2).mean()) + 1e-9)
    out = ROOT / f"music/template_{n}_song.wav"
    sf.write(str(out), (seg * g).astype(np.float32), sr, subtype="PCM_24")
    print(f"T{n}: {b['title'][:60]} | rate {r} | start {b['offset_played']:.2f}s | gain {g:.2f} -> {out.name}")

if __name__ == "__main__":
    for n in sys.argv[1:]:
        prep(int(n))
