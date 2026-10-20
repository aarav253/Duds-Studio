"""Make 'dramatic' versions of calm manga panels so they suit hype edits.

The character (found with the anime cutout model) is left exactly as drawn;
only the space around him changes - that's where action panels differ from
portraits:
  ink       dark gradient + screentone dots on the empty paper (ink density)
  lines     manga focus/concentration lines bursting from behind the character
  splatter  ink splashes around the silhouette
  tilt      dutch-angle rotation with a tighter crop

Originals are never modified; results go to <folder>/prepped/<name>_drama.png

    python src/panel_prep.py "panels/Sae Itoshi"               # all images
    python src/panel_prep.py "panels/Sae Itoshi" --tilt 0      # no rotation
"""
import argparse
import hashlib
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _mask(path):
    import wekisava_edit
    return wekisava_edit.cutout_mask(path)


def screentone(h, w, pitch, angle_deg=45):
    """Classic round-dot screentone field (values 0..1, 1 = ink dot centre)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle_deg)
    u = (xx * math.cos(a) + yy * math.sin(a)) / pitch
    v = (-xx * math.sin(a) + yy * math.cos(a)) / pitch
    d = np.hypot(u - np.round(u), v - np.round(v))           # 0 at dot centre, ~0.7 max
    return d


def prep(path, out_dir, tilt=6.0, seed=None, ink=0.55, lines=True, splatter=True):
    path = Path(path)
    rng = np.random.default_rng(seed if seed is not None else int(hashlib.md5(path.name.encode()).hexdigest()[:8], 16))
    img = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("L")).astype(np.float32) / 255
    h, w = img.shape
    m = _mask(path)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    m = np.clip(cv2.GaussianBlur(m, (0, 0), max(1.0, w / 600)), 0, 1)
    bg = 1 - m
    ys, xs = np.nonzero(m > 0.5)
    cx, cy = (xs.mean(), ys.mean()) if len(xs) else (w / 2, h / 2)
    scale = max(w, h)

    out = img.copy()
    # 1) ink density: gradient that darkens away from the character + screentone
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - cx, yy - cy) / scale
    grad = np.clip((r - 0.15) * 1.6, 0, 1)                    # 0 near him -> 1 at the far edges
    corner = rng.choice([-1, 1])                              # one side darker (asymmetry)
    grad = np.clip(grad + corner * (xx / w - 0.5) * 0.35, 0, 1)
    tone = screentone(h, w, pitch=max(5, scale / 140))
    dot_r = 0.12 + 0.55 * grad                                # dots grow with darkness
    dots = (tone < dot_r).astype(np.float32)
    dots = cv2.GaussianBlur(dots, (0, 0), 0.6)
    shade = np.clip(dots * 0.85 + grad * 0.35, 0, 1) * ink
    out = out * (1 - shade * bg)

    # 2) focus lines from behind the character (black wedges, not over him)
    if lines:
        lay = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(lay)
        R = scale * 1.5
        inner = scale * rng.uniform(0.22, 0.32)
        for _ in range(int(rng.integers(90, 140))):
            a = rng.uniform(0, 2 * math.pi)
            wdt = rng.uniform(0.002, 0.009)
            r0 = inner * rng.uniform(0.9, 1.5)
            p0 = (cx + r0 * math.cos(a), cy + r0 * math.sin(a))
            p1 = (cx + R * math.cos(a - wdt), cy + R * math.sin(a - wdt))
            p2 = (cx + R * math.cos(a + wdt), cy + R * math.sin(a + wdt))
            d.polygon([p0, p1, p2], fill=255)
        ln = np.asarray(lay).astype(np.float32) / 255
        out = out * (1 - ln * bg * 0.85)

    # 3) ink splatter around the silhouette
    if splatter:
        edge = np.clip(cv2.GaussianBlur(m, (0, 0), scale / 60) - m, 0, 1)
        eys, exs = np.nonzero(edge > 0.15)
        lay = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(lay)
        for _ in range(int(rng.integers(40, 80))):
            if not len(exs):
                break
            i = rng.integers(len(exs))
            x, y = exs[i], eys[i]
            rr = rng.uniform(0.002, 0.012) * scale
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=255)
            for _ in range(int(rng.integers(0, 4))):             # droplets flung outward
                a = math.atan2(y - cy, x - cx) + rng.uniform(-0.5, 0.5)
                dd = rng.uniform(1.5, 5) * rr
                r2 = rr * rng.uniform(0.2, 0.5)
                px, py = x + dd * math.cos(a), y + dd * math.sin(a)
                d.ellipse([px - r2, py - r2, px + r2, py + r2], fill=255)
        sp = np.asarray(lay).astype(np.float32) / 255
        out = out * (1 - sp * bg)

    # 4) dutch angle with a tighter crop around the character
    if tilt:
        ang = tilt * rng.choice([-1, 1])
        th = math.radians(abs(ang))
        cover = math.cos(th) + max(h / w, w / h) * math.sin(th)
        # rotate about the image centre: re-centring on the character would
        # expose the borders (mirrored ghosts via BORDER_REFLECT)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, cover * 1.06)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{path.stem}_drama.png"
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(dst)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tilt", type=float, default=6.0)
    ap.add_argument("--ink", type=float, default=0.55)
    a = ap.parse_args()
    folder = Path(a.folder) if Path(a.folder).is_absolute() else ROOT / a.folder
    outd = folder / "prepped"
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            print(prep(p, outd, tilt=a.tilt, ink=a.ink).relative_to(ROOT))
