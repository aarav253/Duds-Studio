"""Per-layer motion analysis of a video's shots with SAM 2 + RAFT.

For every shot in a template timeline (or explicit frame ranges):
  1. SAM 2 (video predictor, tiny) tracks the character through the shot,
     seeded with the anime cutout mask on the shot's first usable frame.
  2. RAFT (torchvision, small) gives dense optical flow between consecutive
     frames.
  3. For each layer (character = SAM 2 mask, background = the rest) a
     similarity transform (scale, translation) is fitted to the flow inside
     that layer -> per-frame motion of each layer on its own.

Output JSON per shot:
  {start, end, look, char: {cov, tx, ty, scale}, bg: {tx, ty, scale},
   rel: character motion relative to background, mask_area}
where tx/ty are px per frame at 1080 wide and scale is the per-frame zoom.

    python src/layer_analysis.py <video> <template.json> <out.json> [--looks red,yellow,duo]
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SAM_CKPT = ROOT / "tools" / "models" / "sam2.1_hiera_tiny.pt"
SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
AW, AH = 272, 336                                   # analysis size (RAFT needs /8)

_raft = None
_sam = None


def raft():
    global _raft
    if _raft is None:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        _raft = raft_small(weights=Raft_Small_Weights.DEFAULT).eval()
    return _raft


def sam():
    global _sam
    if _sam is None:
        from sam2.build_sam import build_sam2_video_predictor
        _sam = build_sam2_video_predictor(SAM_CFG, str(SAM_CKPT), device="cpu")
    return _sam


def read_frames(video, a, b):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    out = []
    for _ in range(a, b + 1):
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


@torch.no_grad()
def flows(frames):
    """RAFT flow between consecutive frames at analysis size (px at AW)."""
    m = raft()
    res = []
    t = [torch.from_numpy(cv2.cvtColor(cv2.resize(f, (AW, AH)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1
         for f in frames]
    for i in range(1, len(t)):
        fl = m(t[i - 1][None], t[i][None], num_flow_updates=8)[-1][0].numpy()
        res.append(fl)                                  # (2, AH, AW)
    return res


@torch.no_grad()
def track(frames, seed_mask, seed_idx=0):
    """SAM 2 masks for every frame, seeded with a mask on frame seed_idx and
    propagated forwards and backwards."""
    pred = sam()
    tmp = Path(tempfile.mkdtemp(prefix="sam2_"))
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp / f"{i:05d}.jpg"), cv2.resize(f, (AW * 2, AH * 2)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        st = pred.init_state(video_path=str(tmp), offload_video_to_cpu=True, offload_state_to_cpu=True)
        if seed_mask is None:
            # no clean cutout to seed with: let SAM 2 segment from a click on
            # the character (frame centre, upper half) - SAM's own segmentation
            W2, H2 = AW * 2, AH * 2
            pts = np.array([[W2 * .5, H2 * .3], [W2 * .5, H2 * .75],                     # character (positive)
                            [W2 * .06, H2 * .12], [W2 * .94, H2 * .12], [W2 * .06, H2 * .9], [W2 * .94, H2 * .9]],  # background corners
                           np.float32)
            pred.add_new_points_or_box(st, frame_idx=seed_idx, obj_id=1, points=pts,
                                       labels=np.array([1, 1, 0, 0, 0, 0], np.int32))
        else:
            pred.add_new_mask(st, frame_idx=seed_idx, obj_id=1, mask=cv2.resize(seed_mask.astype(np.uint8), (AW * 2, AH * 2)) > 0)
        out = [None] * len(frames)
        for rev in (False, True):
            for fi, _, logits in pred.propagate_in_video(st, start_frame_idx=seed_idx, reverse=rev):
                out[fi] = cv2.resize((logits[0, 0] > 0).cpu().numpy().astype(np.uint8), (AW, AH)) > 0
        pred.reset_state(st)
        return [o if o is not None else np.zeros((AH, AW), bool) for o in out]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fit(flow, mask, step=4):
    """Least-squares similarity (scale s, tx, ty) of the flow inside mask."""
    ys, xs = np.nonzero(mask[::step, ::step])
    if len(ys) < 30:
        return None
    ys, xs = ys * step, xs * step
    u, v = flow[0][ys, xs], flow[1][ys, xs]
    cx, cy = AW / 2, AH / 2
    X, Y = xs - cx, ys - cy
    # x' = s*X + tx, y' = s*Y + ty  (rotation ignored)
    A = np.concatenate([np.stack([X, np.ones_like(X), np.zeros_like(X)], 1),
                        np.stack([Y, np.zeros_like(Y), np.ones_like(Y)], 1)])
    rhs = np.concatenate([X + u, Y + v])
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    k = 1080 / AW
    return float(sol[0]), float(sol[1] * k), float(sol[2] * k)


def analyse_shot(video, a, b):
    import sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    import clip_crops as cc
    frames = read_frames(video, a, b)
    if len(frames) < 4:
        return None
    # seed on the sharpest frame (transition frames at the start are blurred)
    sharp = [cv2.Laplacian(cv2.cvtColor(cv2.resize(f, (AW, AH)), cv2.COLOR_BGR2GRAY), cv2.CV_32F).var() for f in frames]
    order = np.argsort(sharp)[::-1]
    seed, seed_i = None, 0
    for i in order[:6]:
        mk = cv2.resize(cc.mask_of(frames[i], 400), (AW, AH)) > 0.5
        if 0.03 < mk.mean() < 0.95:
            seed, seed_i = mk, int(i)
            break
    if seed is None:
        seed_i = int(order[0])                       # SAM 2 click prompt on the sharpest frame
    masks = track(frames, seed, seed_i)
    fl = flows(frames)
    rows = {"char": [], "bg": [], "rel": [], "cov": []}
    for i, f in enumerate(fl):
        m = masks[i] & masks[i + 1]
        cm = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        bm = cv2.erode((~(masks[i] | masks[i + 1])).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        c, g = fit(f, cm), fit(f, bm)
        rows["char"].append(c)
        rows["bg"].append(g)
        rows["rel"].append([c[1] - g[1], c[2] - g[2]] if c and g else None)
        rows["cov"].append(float(masks[i].mean()))

    def ser(lst):
        return {"scale": [round(x[0], 5) if x else None for x in lst], "tx": [round(x[1], 2) if x else None for x in lst],
                "ty": [round(x[2], 2) if x else None for x in lst]}
    return {"start": a, "end": b, "char": ser(rows["char"]), "bg": ser(rows["bg"]),
            "rel": [[round(v, 2) for v in r] if r else None for r in rows["rel"]],
            "cov": [round(v, 3) for v in rows["cov"]]}


def summary(r):
    def st(layer):
        sc = [math.log(s) * 100 for s in r[layer]["scale"] if s]
        tx = [t for t in r[layer]["tx"] if t is not None]
        ty = [t for t in r[layer]["ty"] if t is not None]
        if not tx:
            return None
        sp = np.hypot(tx, ty)
        jit = np.std(np.diff(tx)) + np.std(np.diff(ty)) if len(tx) > 2 else 0
        return dict(zoom_pct_per_f=round(float(np.mean(sc)), 3), speed=round(float(np.median(sp)), 2),
                    jitter=round(float(jit), 2), drift_x=round(float(np.sum(tx)), 1), drift_y=round(float(np.sum(ty)), 1))
    rel = [x for x in r["rel"] if x]
    return {"char": st("char"), "bg": st("bg"), "cov": round(float(np.mean(r["cov"])), 3),
            "parallax": round(float(np.median(np.hypot(*np.array(rel).T))), 2) if rel else None}


if __name__ == "__main__":
    video, tpl, out = sys.argv[1], sys.argv[2], sys.argv[3]
    looks = None
    if "--looks" in sys.argv:
        looks = set(sys.argv[sys.argv.index("--looks") + 1].split(","))
    t = json.loads(Path(tpl).read_text(encoding="utf-8-sig"))
    res = []
    for seg in t["timeline"]:
        if seg["type"] != "shot" or (looks and seg.get("look") not in looks):
            continue
        r = analyse_shot(video, seg["start"], seg["end"])
        if r is None:
            continue
        r["look"] = seg.get("look")
        if "skip" not in r:
            r["summary"] = summary(r)
        res.append(r)
        print(seg["start"], seg.get("look"), r.get("skip") or r["summary"], flush=True)
        Path(out).write_text(json.dumps(res), encoding="utf-8")
