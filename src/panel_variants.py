"""Framing variants of cleaned panels, to match reference edits that cut
between tight close-ups and mid shots (Xavier / Valtrez style).

For every image in <folder>/clean/ writes to <folder>/variants/:
  <stem>_full.png   the cleaned panel
  <stem>_mid.png    head-to-waist crop (4:5) centred on the character
  <stem>_face.png   face close-up crop (4:5)
Crops are only made when the character is small enough in the panel for
the crop to add something, and never need more than 2.6x upscaling.
index.json records each variant's kind and cutout quality ("mask" =
share of the frame the cutout model marks as character; <0.12 = unreliable).

    python src/panel_variants.py "panels/Micheal Kaiser"
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_clean import upscale  # noqa: E402
from wekisava_edit import ROOT, cutout_mask  # noqa: E402

ASPECT = 1350 / 1080
# panels where the top of the cutout is not the head (upside-down dives etc.)
SKIP_CROPS = {"20260327_072251314_iOS"}


def crop_around(g, cx, cy, ch):
    h, w = g.shape
    ch = min(ch, h, w * ASPECT)
    cw = ch / ASPECT
    x0 = int(np.clip(cx - cw / 2, 0, w - cw))
    y0 = int(np.clip(cy - ch * 0.40, 0, h - ch))          # head sits ~40% down
    return g[y0:y0 + int(ch), x0:x0 + int(cw)]


def variants(path, out_dir):
    g = np.asarray(Image.open(path).convert("L"))
    h, w = g.shape
    m = cv2.resize(cutout_mask(path), (w, h)) > 0.5
    cov = float(m.mean())
    out = [("full", g)]
    ys, xs = np.nonzero(m)
    if cov > 0.03 and path.stem not in SKIP_CROPS:
        top, bot = np.percentile(ys, 1), np.percentile(ys, 99)
        bh = bot - top
        sel = ys <= top + 0.30 * bh
        hx, hy = xs[sel].mean(), ys[sel].mean()          # head
        for kind, frac in (("mid", 0.62), ("face", 0.30)):
            ch = frac * bh * (1.9 if cov > 0.45 else 1.0)  # portraits: the "character" is already mostly face
            if ch > 0.85 * min(h, w * ASPECT) or ch / ASPECT < 1440 / 2.6 / ASPECT * 0.9:
                continue
            # the top of the cutout is hair: drop to the eye line (portraits: the
            # face is a bigger share of the cutout, so further down)
            dy = {"mid": 0.18, "face": 0.20 if cov > 0.45 else 0.07}[kind] * bh
            out.append((kind, crop_around(g, hx, hy + dy, ch)))
    res = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, im in out:
        dst = out_dir / f"{path.stem}_{kind}.png"
        Image.fromarray(upscale(im)).save(dst)
        res.append({"path": str(dst.relative_to(ROOT)).replace("\\", "/"), "kind": kind, "source": path.name,
                    "mask": round(float(cutout_mask(dst).mean()) if kind != "full" else cov, 3)})
    return res


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    folder = folder if folder.is_absolute() else ROOT / folder
    idx = []
    for p in sorted((folder / "clean").glob("*.png")):
        r = variants(p, folder / "variants")
        idx += r
        print(p.name, [(v["kind"], v["mask"]) for v in r])
    (folder / "variants" / "index.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")
