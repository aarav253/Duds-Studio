"""Trim baked-in letterbox bars from scene clips (sources that were letterboxed
leave black bands inside the 1080x1350 crops). Each clip with bars is cropped
to its picture and cover-fitted back to its size; originals go to
<clips>/_with_bars/.

    python src/trim_bars.py "panels/<Character>/clips"
"""
import shutil
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sins_edit import Clip  # noqa: E402  (bar detection lives there)


def _blur_fill(b, y0, y1, x0, x1, W, H):
    """Keep the whole picture at its original scale and replace the bars with
    a blurred, darkened stretch of it (no extra zoom)."""
    pic = b[y0:y1, x0:x1]
    small = cv2.resize(pic, (max(1, W // 10), max(1, H // 10)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) * 0.55
    out = bg.copy()
    f = 14
    m = np.zeros((H, W), np.float32)
    m[y0:y1, x0:x1] = 1
    m = cv2.GaussianBlur(m, (0, 0), f / 2)
    m[y0 + f:y1 - f, x0 + f:x1 - f] = 1
    full = np.zeros_like(bg)
    full[y0:y1, x0:x1] = pic
    out = bg * (1 - m[..., None]) + full * m[..., None]
    return out.astype(np.uint8)


def trim(path, mode="cover"):
    bak = path.parent / "_with_bars" / path.name
    if bak.exists():                                             # always start from the original
        shutil.copy2(bak, path)
    c = Clip(path, 1080, 1350)
    y0, y1, x0, x1 = c.box or (0, 0, 0, 0)
    c.cap.release()                                             # Windows can't replace an open file
    cap = cv2.VideoCapture(str(path))
    W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    cap.release()
    if not c.box or (y0 == 0 and y1 == H and x0 == 0 and x1 == W):
        return False
    pad = 2                                                     # eat the soft bar edge
    y0, y1, x0, x1 = y0 + (pad if y0 else 0), y1 - (pad if y1 < H else 0), x0 + (pad if x0 else 0), x1 - (pad if x1 < W else 0)
    bak = path.parent / "_with_bars"
    bak.mkdir(exist_ok=True)
    src = bak / path.name
    if not src.exists():
        shutil.copy2(path, src)
    cap = cv2.VideoCapture(str(src))
    tmp = path.with_suffix(".tmp.mp4")
    wr = imageio_ffmpeg.write_frames(str(tmp), (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                     output_params=["-crf", "15", "-preset", "medium"], macro_block_size=2)
    wr.send(None)
    while True:
        ok, b = cap.read()
        if not ok:
            break
        if mode == "blur":
            wr.send(np.ascontiguousarray(cv2.cvtColor(_blur_fill(b, y0, y1, x0, x1, W, H), cv2.COLOR_BGR2RGB)))
            continue
        b = b[y0:y1, x0:x1]
        h, w = b.shape[:2]
        s = max(W / w, H / h)
        nw, nh = max(W, round(w * s)), max(H, round(h * s))
        b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_CUBIC)
        x, y = (nw - W) // 2, (nh - H) // 2
        wr.send(np.ascontiguousarray(cv2.cvtColor(b[y:y + H, x:x + W], cv2.COLOR_BGR2RGB)))
    wr.close()
    cap.release()
    tmp.replace(path)
    return 1.0 if mode == "blur" else round(s, 2)


if __name__ == "__main__":
    mode = "blur" if "--blur" in sys.argv else "cover"   # --blur: no zoom, bars become a blurred fill
    only = None
    if "--only" in sys.argv:
        only = {int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}
    for p in sorted(Path(sys.argv[1]).glob("scene_*.mp4")):
        if only is not None and int(p.stem[-2:]) not in only:
            continue
        r = trim(p, mode)
        print(p.name, f"trimmed, zoom {r}x" if r else "clean")
