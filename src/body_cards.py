"""Full-body character cards on a white background (the 'strongest characters' parade look).

    python src/body_cards.py "<Character folder>" [...]
Scans every scene of the character's source videos for one clean, whole figure
(src/cutout.py scoring, figure tall and touching at most the bottom edge), cuts the
best few out and writes panels/Blue Lock/<char>/body/body_<n>.png (transparent) +
body/index.json. The edit composites them on white.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cutout import frame, mask, cut  # noqa: E402


def body_score(src, k):
    rgb = frame(src, k)
    if rgb is None:
        return 0
    m = mask(rgb) > 0.5
    H, W = m.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n < 2:
        return 0
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = st[i]
    if a / max(1, m.sum()) < 0.85:                       # one figure, not a crowd
        return 0
    if y <= 2 or x <= 2 or x + w >= W - 2:               # cut at top / sides -> not whole
        return 0
    tall = h / H
    if tall < 0.55 or h / max(1, w) < 1.3:               # standing / full body: tall and narrow
        return 0
    return tall * (1.0 if y + h < H - 2 else 0.8)


def run(char, keep=4):
    d = ROOT / "panels/Blue Lock" / char
    if not d.is_dir():
        print(char, "- no such character folder", flush=True); return
    out = d / "body"; out.mkdir(exist_ok=True)
    cands = []
    for sj in sorted(d.glob("clips*/scenes.json")):
        S = json.loads(sj.read_text(encoding="utf-8-sig"))
        for sc in S["scenes"][1:]:
            for k in (sc["start"] + (sc["end"] - sc["start"]) // 4, (sc["start"] + sc["end"]) // 2,
                      sc["end"] - (sc["end"] - sc["start"]) // 4):
                s = body_score(S["source"], k)
                if s:
                    cands.append((s, S["source"], k))
    cands.sort(key=lambda c: -c[0])
    rows, used = [], []
    for s, src, k in cands:
        if any(src == u[0] and abs(k - u[1]) < 90 for u in used):
            continue
        p = out / f"body_{len(rows)}.png"
        cut(src, k, str(p.relative_to(ROOT)))
        rows.append({"file": str(p.relative_to(ROOT)).replace("\\", "/"), "score": round(s, 3), "src": src, "frame": k})
        used.append((src, k))
        if len(rows) >= keep:
            break
    (out / "index.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(char, len(cands), "candidates ->", len(rows), "cards", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run(c)
