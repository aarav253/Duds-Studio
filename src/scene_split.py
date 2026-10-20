"""Split a source video (e.g. a manga-animation clip compilation) into scenes,
so each scene can be used like a panel in a template edit.

A cut = a frame whose colour histogram and pixels both jump far above the
local level of change (robust to fast motion inside a scene, which changes
pixels but keeps the histogram). Very short scenes (< min_len frames) are
merged into the previous one; long fades are cut at their midpoint.

    python src/scene_split.py "panels/Micheal Kaiser/<video>.mp4" [clips_dir_name]

Writes <video folder>/clips/scenes.json  [{i, start, end, t0, t1}] and
clips/scenes.png (first/middle/last frame of every scene).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def scan(path, w=160):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    hists, smalls = [], []
    while True:
        ok, b = cap.read()
        if not ok:
            break
        s = cv2.resize(b, (w, int(w * b.shape[0] / b.shape[1])), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256]).ravel()
        hists.append(h / max(h.sum(), 1))
        smalls.append(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32))
    return fps, np.array(hists), smalls


def detect(hists, smalls, min_len=12, k=4.0):
    n = len(hists)
    hd = np.zeros(n)
    pd = np.zeros(n)
    for i in range(1, n):
        hd[i] = 0.5 * np.abs(hists[i] - hists[i - 1]).sum()
        pd[i] = np.abs(smalls[i] - smalls[i - 1]).mean()
    cuts = [0]
    for i in range(1, n):
        lo, hi = max(1, i - 15), min(n, i + 16)
        loc_h = np.median(hd[lo:hi]) + 1e-3
        loc_p = np.median(pd[lo:hi]) + 1e-3
        if hd[i] > max(0.25, k * loc_h) and pd[i] > max(12, 2.0 * loc_p) and hd[i] >= hd[max(0, i - 1):i + 2].max():
            if i - cuts[-1] >= min_len:
                cuts.append(i)
    cuts.append(n)
    return [(cuts[j], cuts[j + 1] - 1) for j in range(len(cuts) - 1)]


def sheet(path, scenes, out, w=200):
    cap = cv2.VideoCapture(str(path))
    want = {}
    for i, (a, b) in enumerate(scenes):
        for tag, f in (("a", a + 2), ("m", (a + b) // 2), ("b", b - 2)):
            want.setdefault(max(a, min(b, f)), []).append((i, tag))
    got = {}
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            got[k] = cv2.resize(fr, (w, int(w * fr.shape[0] / fr.shape[1])))
        k += 1
    rows = []
    for i, (a, b) in enumerate(scenes):
        tiles = []
        for f in (max(a, min(b, a + 2)), (a + b) // 2, max(a, min(b, b - 2))):
            t = got[f].copy()
            tiles.append(t)
        r = np.hstack(tiles)
        cv2.putText(r, f"#{i} {a}-{b} ({b - a + 1}f)", (4, 16), 0, 0.5, (0, 255, 255), 1)
        rows.append(r)
    cols = 3
    blank = np.zeros_like(rows[0])
    grid = [np.hstack(rows[j:j + cols] + [blank] * (cols - len(rows[j:j + cols]))) for j in range(0, len(rows), cols)]
    cv2.imwrite(str(out), np.vstack(grid))


if __name__ == "__main__":
    src = Path(sys.argv[1])
    src = src if src.is_absolute() else ROOT / src
    fps, hists, smalls = scan(src)
    scenes = detect(hists, smalls)
    # one clips folder per video (a character folder can hold several videos)
    vids = sorted(q for q in src.parent.iterdir() if q.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv"))
    outd = src.parent / (sys.argv[2] if len(sys.argv) > 2 else "clips" if len(vids) == 1 else f"clips_{vids.index(src) + 1}")
    outd.mkdir(exist_ok=True)
    data = {"source": str(src.relative_to(ROOT)).replace("\\", "/"), "fps": fps,
            "scenes": [{"i": i, "start": a, "end": b, "t0": round(a / fps, 3), "t1": round((b + 1) / fps, 3)}
                       for i, (a, b) in enumerate(scenes)]}
    (outd / "scenes.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    sheet(src, scenes, outd / "scenes.png")
    print(len(scenes), "scenes; lengths:", [b - a + 1 for a, b in scenes])
