"""Find the exact song (upload, speed, start offset) used in a reference edit.

For each candidate audio file: onset-envelope cross-correlation over playback
rates 0.80-1.25 (slowed / sped-up uploads), then sample-accurate refinement on
the raw waveform at a few points to confirm (no drift = same recording).

    python src/song_match.py <ref.wav> <cand1.wav> [<cand2.wav> ...]
match(ref, cand) -> {"corr", "rate", "offset", "confirm"}  (offset in the candidate,
                    seconds, at which ref t=0 starts; rate = candidate speed factor)
"""
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly, stft

SR = 11025


def load(p, sr=SR):
    x, s = sf.read(str(p), always_2d=True)
    x = x.mean(1)
    if s != sr:
        g = gcd(sr, s); x = resample_poly(x, sr // g, s // g)
    return x.astype(np.float32)


def env(x, hop=128):
    f, t, Z = stft(x, SR, nperseg=1024, noverlap=1024 - hop)
    S = np.log1p(np.abs(Z) * 50)
    o = np.maximum(0, np.diff(S, axis=1)).sum(0)
    o = (o - o.mean()) / (o.std() + 1e-9)
    return o


def stretch(x, rate):
    """play x at `rate` (resample: changes speed and pitch like slowed/sped uploads)."""
    up, down = 1000, int(round(1000 * rate))
    g = gcd(up, down)
    return resample_poly(x, up // g, down // g).astype(np.float32)


def match(ref_p, cand_p, rates=None):
    ref, cand = load(ref_p), load(cand_p)
    er = env(ref)
    best = (-1, 1.0, 0)
    rates = rates or list(np.round(np.arange(0.80, 1.251, 0.01), 3))
    for r in rates:
        c = stretch(cand, r) if r != 1.0 else cand          # candidate played at speed r
        ec = env(c)
        if len(ec) <= len(er):
            continue
        cc = np.correlate(ec, er, mode="valid") / len(er)
        k = int(np.argmax(cc))
        if cc[k] > best[0]:
            best = (float(cc[k]), r, k)
    corr, r, k = best
    c = stretch(cand, r)
    off = k * 128 / SR
    # sample-accurate confirmation at 3 points
    conf = []
    for t0 in np.linspace(1.0, max(1.5, len(ref) / SR - 2.0), 3):
        a = ref[int(t0 * SR):int((t0 + 1.0) * SR)]
        lo = int((t0 + off - 0.05) * SR)
        b = c[max(0, lo):max(0, lo) + len(a) + int(0.1 * SR)]
        if len(b) <= len(a):
            continue
        x = correlate(b, a, mode="valid"); j = int(np.argmax(x))
        conf.append(float(x[j] / np.sqrt(np.dot(a, a) * np.dot(b[j:j + len(a)], b[j:j + len(a)]) + 1e-9)))
        if t0 == np.linspace(1.0, max(1.5, len(ref) / SR - 2.0), 3)[0]:
            off = (max(0, lo) + j) / SR - t0
    return {"corr": round(corr, 3), "rate": r, "offset": round(off * r, 4), "offset_played": round(off, 4),
            "confirm": [round(v, 3) for v in conf]}


if __name__ == "__main__":
    for c in sys.argv[2:]:
        print(c, match(sys.argv[1], c), flush=True)
