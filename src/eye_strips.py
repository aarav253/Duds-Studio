"""Harvest eye strips (wide crops across both eyes) for a character from the
full-width source videos of its clip folders and from its images/ folder.

    python src/eye_strips.py "<Character folder>" [aspect=3.2]

Writes panels/Blue Lock/<Character>/eyes/<name>.jpg plus eyes/index.json
[{file, colour (mean saturation), sharp, span}] best first. A strip needs two
detected eyes, eyes at least 18% of the frame wide apart, and is cropped to the
strip aspect (1080 x 337 by default) around them.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BL = ROOT / "panels/Blue Lock"


def strip(rgb, aspect, min_span=0.18):
    from imgutils.detect import detect_eyes
    H, W = rgb.shape[:2]
    eyes = [e for e in detect_eyes(Image.fromarray(rgb)) if e[2] > 0.45]
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: -e[2])[:2]
    xs = [v for e in eyes for v in (e[0][0], e[0][2])]
    ys = [v for e in eyes for v in (e[0][1], e[0][3])]
    span = max(xs) - min(xs)
    if span < min_span * W:
        return None
    if abs((eyes[0][0][1] + eyes[0][0][3]) - (eyes[1][0][1] + eyes[1][0][3])) / 2 > 0.6 * span:
        return None                                        # not side by side (tilted / two faces)
    (a0, _, a1, _), (b0, _, b1, _) = eyes[0][0], eyes[1][0]
    wa, wb = a1 - a0, b1 - b0
    gap = abs((a0 + a1) - (b0 + b1)) / 2
    if max(wa, wb) / max(1, min(wa, wb)) > 1.7 or gap > 3.0 * (wa + wb) / 2:
        return None                                        # one eye each of two different characters
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    w = min(W, span * 2.0); h = w / aspect
    if h > H:
        h = H; w = h * aspect
    x0 = int(np.clip(cx - w / 2, 0, W - w)); y0 = int(np.clip(cy - h * 0.5, 0, H - h))
    crop = rgb[y0:y0 + int(h), x0:x0 + int(w)]
    if crop.shape[0] < 80:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return crop, {"colour": round(float(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[..., 1].mean()), 1),
                  "sharp": round(float(cv2.Laplacian(cv2.resize(g, (540, int(540 / aspect))), cv2.CV_32F).var()), 1),
                  "span": round(span / W, 3)}


def harvest(char, aspect=3.2):
    d = BL / char
    out = d / "eyes"; out.mkdir(exist_ok=True)
    rows = []
    for sj in sorted(d.glob("clips*/scenes.json")):
        S = json.loads(sj.read_text(encoding="utf-8-sig"))
        cap = cv2.VideoCapture(str(ROOT / S["source"]))
        for sc in S["scenes"][1:]:
            for k in (sc["start"] + (sc["end"] - sc["start"]) // 3, sc["start"] + 2 * (sc["end"] - sc["start"]) // 3):
                cap.set(1, k); ok, f = cap.read()
                if not ok:
                    continue
                r = strip(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), aspect)
                if r:
                    name = f"{sj.parent.name}_{sc['i']:02d}_{k}.jpg"
                    Image.fromarray(r[0]).save(out / name, quality=95)
                    rows.append(dict(file=str((out / name).relative_to(ROOT)).replace("\\", "/"), **r[1]))
                    break
    for p in sorted((d / "images").glob("*.jpg")):
        r = strip(np.asarray(Image.open(p).convert("RGB")), aspect, 0.22)
        if r:
            name = f"img_{p.stem}.jpg"
            Image.fromarray(r[0]).save(out / name, quality=95)
            rows.append(dict(file=str((out / name).relative_to(ROOT)).replace("\\", "/"), **r[1]))
    rows.sort(key=lambda r: -(r["sharp"] ** 0.5 * (1 + r["span"])))
    (out / "index.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(char, len(rows), "strips,", sum(r["colour"] >= 30 for r in rows), "colour", flush=True)
    return rows


if __name__ == "__main__":
    harvest(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 3.2)
