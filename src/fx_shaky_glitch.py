"""Shaky Glitch - a replica of CapCut's "Shaky glitch" video effect, measured from the
'Meant To Be' reference (template 24), where it is applied over the whole edit at
intensity 10 %. `intensity` scales everything; 0.10 reproduces the reference.
Four parts that always run together:

  0. constant hand-held jitter, 2-4 px at 1080 (bigger jolt right after a cut)
  1. a blurred, slightly warped band (~22 % of the height) that rolls from top
     to bottom in ~0.55 s, about every 2.4 s
  2. thin horizontal line tears (2-7 px rows shoved sideways) on ~1/3 of frames
  3. tiny one-frame blocks (below)

    from fx_shaky_glitch import shaky_glitch
    frame = shaky_glitch(frame, k, t, fps=60, intensity=0.10, cut=False)
    # or in a seg_edit manifest:  "look": {"shaky_glitch": 0.10}

Tiny blocks:

Measured on the reference's held shots (1080 px frames, 30 fps):
  - small axis-aligned rectangles that differ from both neighbouring frames, so
    each lives 1 frame (sometimes 2)
  - ~2-5 per frame, on almost every frame
  - width ~24-60 px (median 37, i.e. 2-6 % of the frame), height ~24-33 px,
    width/height ~0.9-1.9; now and then one is ~120 x 75
  - contents: (a) a blurred / smeared copy of what is under it, (b) a patch copied
    from a few pixels to ~10 % of the frame away, (c) a flat block of the local
    average colour; often two touch, giving an L shape

    from fx_shaky_glitch import blocks
    frame = blocks(frame, k, fps=60, rate=3.0)      # frame: HxWx3 uint8/float, k: frame index
"""
import cv2
import numpy as np

DEFAULT = {"band_period": 2.4, "band_travel": 0.55, "band_h": 0.22, "tears": 0.35, "blocks": 3.0}


def _mblur(img, n):
    k = np.zeros((n, n), np.float32); k[n // 2, :] = 1.0 / n
    return cv2.filter2D(img, -1, k)


def shaky_glitch(img, k, t, fps=60, intensity=0.10, cut=False, **opt):
    """the full effect at a CapCut-style intensity (0-1; reference = 0.10). Deterministic per frame."""
    s_ = intensity / 0.10
    o = dict(DEFAULT, **opt)
    o["band_h"] *= min(2.5, s_ ** 0.5); o["tears"] = min(1.0, o["tears"] * s_); o["blocks"] *= s_
    f = img.astype(np.float32, copy=True)
    H, W = f.shape[:2]
    px = 3.0 * s_ * (2.2 if cut else 1.0) * W / 1080          # 0. jitter, new offset every reference frame
    if px > 0.2:
        jr = np.random.default_rng(k // max(1, int(round(fps / 30))) + 17)
        dx, dy = jr.uniform(-px, px, 2); z = 1.0 + 2.5 * px / W
        M = np.float32([[z, 0, (1 - z) * W / 2 + dx], [0, z, (1 - z) * H / 2 + dy]])
        f = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ph = (t % o["band_period"]) / o["band_travel"]
    if ph < 1.0 and o["band_h"] > 0:                                             # 1. rolling blur band
        hh = int(o["band_h"] * H); y0 = int(ph * (H + hh)) - hh
        a0, a1 = max(0, y0), min(H, y0 + hh)
        if a1 - a0 > 4:
            band = _mblur(np.clip(f[a0:a1], 0, 255), max(3, int(0.03 * W)))
            shift = (np.sin((np.arange(a0, a1) - y0) / hh * np.pi) * 0.012 * W).astype(int)
            for sh in np.unique(shift):
                m = shift == sh
                band[m] = np.roll(band[m], sh, 1)
            w = np.sin(np.linspace(0, np.pi, a1 - a0))[:, None, None]
            f[a0:a1] = f[a0:a1] * (1 - w) + band * w
    rng = np.random.default_rng(k // max(1, int(round(fps / 30))) + 911)
    if rng.random() < o["tears"]:                           # 2. line tears
        for _ in range(rng.integers(1, 4)):
            y = int(rng.uniform(0, H - 8)); h = max(2, int(rng.uniform(2, 7) * H / 1350))
            f[y:y + h] = np.roll(f[y:y + h], int(rng.uniform(-0.03, 0.03) * W), 1)
    if o["blocks"]:                                          # 3. tiny blocks
        f = blocks(f, k, fps, rate=o["blocks"])
    return f


def blocks(img, k, fps=60, rate=3.0, size=(0.022, 0.058), big_chance=0.06, seed=4242, life=1):
    """returns a copy of `img` with this frame's glitch blocks. Deterministic per (k, seed).
    rate = mean blocks per reference frame (30 fps); at 60 fps a block is held `life` reference frames."""
    H, W = img.shape[:2]
    step = max(1, int(round(fps / 30)) * life)               # hold each block pattern for one reference frame
    rng = np.random.default_rng(seed * 100003 + k // step)
    out = img.astype(np.float32, copy=True)
    n = rng.poisson(rate)
    for _ in range(n):
        if rng.random() < big_chance:
            w = int(rng.uniform(0.09, 0.12) * W); h = int(w * rng.uniform(0.5, 0.7))
        else:
            w = int(rng.uniform(*size) * W); h = max(4, int(w / rng.uniform(0.9, 1.9)))
        x, y = int(rng.uniform(0, W - w)), int(rng.uniform(0, H - h))
        _paint(out, rng, x, y, w, h, W, H)
        if rng.random() < 0.3:                               # a second block touching the first -> L shape
            w2, h2 = int(w * rng.uniform(0.5, 1.2)), int(h * rng.uniform(0.5, 1.2))
            x2 = int(np.clip(x + (w if rng.random() < 0.5 else -w2), 0, W - w2)); y2 = int(np.clip(y + rng.uniform(-h2, h), 0, H - h2))
            _paint(out, rng, x2, y2, w2, h2, W, H)
    return out


def _paint(out, rng, x, y, w, h, W, H):
    kind = rng.choice(["blur", "shift", "flat"], p=[0.45, 0.4, 0.15])
    reg = out[y:y + h, x:x + w]
    if kind == "blur":                                       # smeared copy of itself
        horiz = rng.random() < 0.6
        kk = max(3, int((w if horiz else h) * rng.uniform(0.4, 0.9))) | 1
        out[y:y + h, x:x + w] = cv2.blur(reg, (kk, 1) if horiz else (1, kk))
    elif kind == "shift":                                    # patch pulled from nearby
        dx, dy = int(rng.uniform(-0.1, 0.1) * W), int(rng.uniform(-0.04, 0.04) * H)
        sx, sy = int(np.clip(x + dx, 0, W - w)), int(np.clip(y + dy, 0, H - h))
        src = out[sy:sy + h, sx:sx + w].copy()
        if rng.random() < 0.5:
            src = cv2.blur(src, (max(3, w // 3) | 1, 1))
        out[y:y + h, x:x + w] = src
    else:                                                    # flat local colour
        out[y:y + h, x:x + w] = reg.reshape(-1, reg.shape[2]).mean(0)


if __name__ == "__main__":                                   # demo: python src/fx_shaky_glitch.py <image> <out.png>
    import sys
    im = cv2.imread(sys.argv[1])
    tiles = [np.clip(shaky_glitch(im, k * 2, 0.1 + k * 0.08), 0, 255).astype(np.uint8) for k in range(4)]
    cv2.imwrite(sys.argv[2], np.hstack([cv2.resize(t, (360, int(360 * im.shape[0] / im.shape[1]))) for t in tiles]))
