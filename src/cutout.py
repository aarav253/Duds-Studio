"""Cut a character out of a clip frame (rembg isnet-anime) -> transparent PNG.

    python src/cutout.py <clip.mp4> <frame> <out.png>
    python src/cutout.py --scan <clip.mp4> [<clip.mp4> ...]    # score frames: one clean, whole figure?
    python src/cutout.py --scan-src <clips/scenes.json> [...]  # same on the full-width source (whole figures)

The mask is cleaned to its largest component (stray background bits dropped),
edges softened by a pixel, and the PNG cropped to the figure.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
_SESSION = None


def _p(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def mask(rgb):
    global _SESSION
    from rembg import new_session, remove
    if _SESSION is None:
        _SESSION = new_session("isnet-anime")
    m = np.asarray(remove(Image.fromarray(rgb), session=_SESSION, only_mask=True)).astype(np.float32) / 255
    b = (m > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    if n > 1:
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        keep = cv2.dilate((lab == i).astype(np.uint8), np.ones((9, 9), np.uint8))
        m = m * keep
    return cv2.GaussianBlur(m, (0, 0), 1.0)


def frame(clip, k):
    c = cv2.VideoCapture(str(_p(clip))); c.set(1, int(k)); ok, f = c.read(); c.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None


def cut(clip, k, out):
    rgb = frame(clip, k)
    m = mask(rgb)
    ys, xs = np.nonzero(m > 0.05)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([rgb, (m * 255).astype(np.uint8)])[y0:y1, x0:x1]
    _p(out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(_p(out))
    return out


def score(clip, k):
    """how usable frame k is as a cutout: one solid figure, not touching the frame on 3+ sides."""
    rgb = frame(clip, k)
    if rgb is None:
        return 0, {}
    m = mask(rgb) > 0.5
    H, W = m.shape
    area = m.mean()
    sides = int(m[:3].any()) + int(m[-3:].any()) + int(m[:, :3].any()) + int(m[:, -3:].any())
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    frag = 1 - (st[1:, cv2.CC_STAT_AREA].max() / max(1, m.sum())) if n > 1 else 1
    s = (0.08 < area < 0.6) * (1 - frag) * (1 if sides <= 1 else 0.5 if sides == 2 else 0.1)
    return round(float(s), 3), {"area": round(float(area), 2), "sides": sides}


def scan_src(sj):
    """score the middle frame of every scene on the full-width source video; best first."""
    import json
    S = json.loads(_p(sj).read_text(encoding="utf-8-sig"))
    out = []
    for sc in S["scenes"][1:]:
        k = (sc["start"] + sc["end"]) // 2
        s, info = score(S["source"], k)
        if s > 0:
            out.append((s * (0.5 + info["area"]), S["source"], k, info))
    return sorted(out, key=lambda x: -x[0])


if __name__ == "__main__":
    if sys.argv[1] == "--scan-src":
        for sj in sys.argv[2:]:
            for r in scan_src(sj)[:8]:
                print(f"{r[0]:.3f} {r[1]} frame {r[2]} {r[3]}", flush=True)
    elif sys.argv[1] == "--scan":
        for clip in sys.argv[2:]:
            n = int(cv2.VideoCapture(str(_p(clip))).get(7))
            best = max(((score(clip, k), k) for k in (n // 4, n // 2, 3 * n // 4)), key=lambda x: x[0][0])
            print(f"{best[0][0]:.3f} {clip} frame {best[1]} {best[0][1]}", flush=True)
    else:
        print(cut(sys.argv[1], sys.argv[2], sys.argv[3]))
