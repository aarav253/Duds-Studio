"""Face-focused vertical crops for every scene from scene_split.py.

For each scene: every `step` frames the anime cutout model finds the
character; the head is the centroid of the top 30% of the largest cutout
component, and the head width sets the crop size (face fills the upper-middle
of a 4:5 frame). The per-frame path is median-filtered and smoothed so the
crop glides instead of jittering, then each scene is exported as its own
1080x1350 clip.

Manual fixes go in clips/crops_override.json:
  {"<scene i>": {"cx": 0.42, "cy": 0.35, "h": 0.8}}   (fractions of the source
  frame: face centre, crop height) - a fixed crop for that scene.
  {"<scene i>": {"skip": true}}                        - don't export.

    python src/clip_crops.py "panels/Micheal Kaiser/clips/scenes.json"
"""
import json
import os
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT_W, OUT_H = 1080, 1350
_SESSION = None


def mask_of(bgr, w=640):
    global _SESSION
    from rembg import new_session, remove
    if _SESSION is None:
        _SESSION = new_session("isnet-anime")
    h = int(w * bgr.shape[0] / bgr.shape[1])
    im = Image.fromarray(cv2.cvtColor(cv2.resize(bgr, (w, h)), cv2.COLOR_BGR2RGB))
    return np.asarray(remove(im, session=_SESSION, only_mask=True)).astype(np.float32) / 255


def head(m):
    """(cx, cy, head_w) as fractions of the frame, or None."""
    mm = (m > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mm)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 0.01 * mm.size:
        return None
    ys, xs = np.nonzero(lab == i)
    top, bot = np.percentile(ys, 1), np.percentile(ys, 99)
    sel = ys <= top + 0.30 * (bot - top)
    H, W = mm.shape
    hx, hy = xs[sel].mean(), ys[sel].mean()
    hw = np.percentile(xs[sel], 95) - np.percentile(xs[sel], 5)
    # the top band is hair: the face sits a little below its centroid
    return hx / W, (hy + 0.25 * hw) / H, hw / W


def smooth(v, n):
    v = np.asarray(v, float)
    if len(v) < 3:
        return v
    from scipy.signal import medfilt
    v = medfilt(v, 5 if len(v) >= 5 else 3)
    pad = np.pad(v, (n, n), mode="edge")
    return np.convolve(pad, np.ones(2 * n + 1) / (2 * n + 1), "valid")


def plan(cap, a, b, step=int(os.environ.get("CROP_STEP", 6))):
    """Per-frame (cx, cy, crop_h) for frames a..b (fractions of the source)."""
    pts = []
    for f in range(a, b + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        h = head(mask_of(fr))
        if h:
            pts.append((f, *h))
    if not pts:
        return None
    f, cx, cy, hw = map(np.array, zip(*pts))
    src_aspect = cap.get(3) / cap.get(4)
    # crop height (fraction of source height): face ~ 3.4 head-widths tall frame
    ch = np.clip(hw * src_aspect * 3.4, 0.5, 1.0)
    cx, cy, ch = smooth(cx, 2), smooth(cy, 2), smooth(ch, 3)
    ks = np.arange(a, b + 1)
    return np.interp(ks, f, cx), np.interp(ks, f, cy), np.interp(ks, f, ch)


def crop_box(cx, cy, ch, W, H):
    hpx = ch * H
    wpx = hpx * OUT_W / OUT_H
    x0 = np.clip(cx * W - wpx / 2, 0, W - wpx)
    y0 = np.clip(cy * H - 0.38 * hpx, 0, H - hpx)        # face ~38% down the crop
    return x0, y0, wpx, hpx


def export(cap, a, b, path_cx, path_cy, path_ch, out):
    W, H = cap.get(3), cap.get(4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wr = imageio_ffmpeg.write_frames(str(out), (OUT_W, OUT_H), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                     output_params=["-crf", "14", "-preset", "medium"], macro_block_size=2)
    wr.send(None)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    for j in range(b - a + 1):
        ok, fr = cap.read()
        if not ok:
            break
        x0, y0, w, h = crop_box(path_cx[j], path_cy[j], path_ch[j], W, H)
        M = np.float32([[OUT_W / w, 0, -x0 * OUT_W / w], [0, OUT_H / h, -y0 * OUT_H / h]])
        im = cv2.warpAffine(fr, M, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        wr.send(np.ascontiguousarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    wr.close()


if __name__ == "__main__":
    sj = Path(sys.argv[1])
    sj = sj if sj.is_absolute() else ROOT / sj
    data = json.loads(sj.read_text(encoding="utf-8-sig"))
    only = {int(x) for x in sys.argv[2].split(",")} if len(sys.argv) > 2 else None
    ov_path = sj.parent / "crops_override.json"
    ov = json.loads(ov_path.read_text(encoding="utf-8-sig")) if ov_path.exists() else {}
    cap = cv2.VideoCapture(str(ROOT / data["source"]))
    plans = {}
    pp = sj.parent / "crops.json"
    if pp.exists():
        plans = json.loads(pp.read_text(encoding="utf-8-sig"))
    for s in data["scenes"]:
        i, a, b = s["i"], s["start"], s["end"]
        if only is not None and i not in only:
            continue
        o = ov.get(str(i), {})
        if o.get("skip"):
            print(f"scene {i}: skipped")
            continue
        n = b - a + 1
        if "cx" in o:
            p = (np.full(n, o["cx"]), np.full(n, o["cy"]), np.full(n, o.get("h", 1.0)))
        else:
            p = plan(cap, a, b)
            if p is None:
                p = (np.full(n, 0.5), np.full(n, 0.4), np.full(n, 1.0))
        out = sj.parent / f"scene_{i:02d}.mp4"
        export(cap, a, b, *p, out)
        plans[str(i)] = {"clip": str(out.relative_to(ROOT)).replace("\\", "/"), "frames": n,
                         "cx": [round(float(v), 4) for v in p[0][::6]], "cy": [round(float(v), 4) for v in p[1][::6]],
                         "h": [round(float(v), 4) for v in p[2][::6]]}
        pp.write_text(json.dumps(plans, indent=1), encoding="utf-8")
        print(f"scene {i}: {n}f crop h {np.mean(p[2]):.2f} centre ({np.mean(p[0]):.2f},{np.mean(p[1]):.2f}) -> {out.name}")
