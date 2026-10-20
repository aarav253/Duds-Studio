"""Clean raw manga panels into edit-ready images.

Per folder (originals are never modified):
  - skips byte-identical duplicates ("x 1.jpg" copies from AirDrop/OneDrive)
  - erases text boxes / captions / SFX listed in <folder>/clean.json
    (ink pixels inside each box are inpainted from the surrounding paper)
  - optional crop per image, or skip an image entirely
  - de-JPEG (light non-local-means), auto levels (true black / true white),
    gentle local contrast so washed-out scans pop
  - upscales so the short side is >= 1440 px (the 1080x1356 edit frame plus
    room for zooms) with Lanczos + unsharp mask to keep ink lines crisp

Writes <folder>/clean/<stem>.png

    python src/panel_clean.py "panels/Micheal Kaiser"

clean.json:
  {"<file name>": {"skip": "reason"} |
                  {"erase": [[x0,y0,x1,y1], ...], "crop": [x0,y0,x1,y1]}}
  boxes are in original-image pixels.
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MIN_SHORT = 1440


def unique_images(folder):
    seen, out = set(), []
    # originals before their " 1" / " (1)" copies so the kept name is the plain one
    for p in sorted(folder.iterdir(), key=lambda q: (q.stem.endswith((" 1", " (1)")), q.name)):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(p)
    return sorted(out)


def erase_text(g, boxes):
    """Inpaint dark text strokes inside each box (keeps the paper/lines around)."""
    mask = np.zeros(g.shape, np.uint8)
    for x0, y0, x1, y1 in boxes:
        roi = g[y0:y1, x0:x1]
        ink = (roi < 150).astype(np.uint8) * 255
        mask[y0:y1, x0:x1] = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    return cv2.inpaint(g, mask, 5, cv2.INPAINT_TELEA) if mask.any() else g


def tone(g):
    """Auto levels to true black/white + mild CLAHE (manga is ink on paper)."""
    lo, hi = np.percentile(g, [0.8, 99.2])
    if hi - lo < 40:
        return g
    f = np.clip((g.astype(np.float32) - lo) / (hi - lo), 0, 1)
    g = (f * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(6, 6))
    return cv2.addWeighted(g, 0.6, clahe.apply(g), 0.4, 0)


def upscale(g):
    h, w = g.shape
    s = MIN_SHORT / min(h, w)
    if s <= 1:
        return g
    up = cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.2 * s ** 0.5)
    return cv2.addWeighted(up, 1.6, blur, -0.6, 0)          # unsharp: crisp ink edges


def clean(path, cfg, out_dir):
    g = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("L"))
    g = cv2.fastNlMeansDenoising(g, None, h=6, templateWindowSize=7, searchWindowSize=21)
    if cfg.get("erase"):
        g = erase_text(g, cfg["erase"])
    if cfg.get("crop"):
        x0, y0, x1, y1 = cfg["crop"]
        g = g[y0:y1, x0:x1]
    g = upscale(tone(g))
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{path.stem}.png"
    Image.fromarray(g).save(dst)
    return dst, g.shape


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    folder = folder if folder.is_absolute() else ROOT / folder
    cfgf = folder / "clean.json"
    cfg = json.loads(cfgf.read_text(encoding="utf-8-sig")) if cfgf.exists() else {}
    for p in unique_images(folder):
        c = cfg.get(p.name, {})
        if c.get("skip"):
            print(f"skip  {p.name}: {c['skip']}")
            continue
        dst, (h, w) = clean(p, c, folder / "clean")
        print(f"clean {p.name} -> {dst.relative_to(ROOT)} ({w}x{h})")
