"""Repair a clipped / crushed audio rip: every run of samples stuck near the ceiling is redrawn with a cubic
spline through the good samples around it (so the peaks get their rounded tops back), then the file is
scaled down so the restored peaks fit with headroom.  Used for the distorted Rin voicelines and template 35.

    python src/declip.py <in.wav> <out.wav> [thr=0.97]
"""
import sys

import numpy as np
import soundfile as sf
from scipy.interpolate import CubicSpline


def declip(x, thr=0.97, ctx=8):
    x = x.copy()
    lim = thr * np.abs(x).max()
    bad = np.abs(x) >= lim
    idx = np.flatnonzero(bad)
    if not len(idx):
        return x, 0
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for r in runs:
        a, b = r[0], r[-1]
        L = np.arange(max(0, a - ctx), a); R = np.arange(b + 1, min(len(x), b + 1 + ctx))
        k = np.concatenate([L, R])
        if len(k) < 4:
            continue
        x[a:b + 1] = CubicSpline(k, x[k])(np.arange(a, b + 1))
    return x, len(runs)


def main(src, dst, thr=0.97):
    y, sr = sf.read(src, always_2d=True)
    out = np.zeros_like(y); n = 0
    for c in range(y.shape[1]):
        out[:, c], k = declip(y[:, c], thr); n += k
    out *= 0.89 / np.abs(out).max()                       # -1 dBFS
    sf.write(dst, out, sr)
    print(f"{n} clipped runs repaired -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.97)
