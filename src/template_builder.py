"""Build a reusable edit template from an analyzed reference edit.

    python src/template_builder.py references/<video>.mp4 <template_name>

Needs the reference analysis (run reference_analyzer first; this script runs
it automatically if references/template_N/analysis/report.json is missing).

Extracts, frame-exact:
  - intro: fade-in window, the "card pop" start (inset card of the intro image
    flying in one beat before the drop), caption band position
  - every post-drop cut: the frame the halftone dissolve starts (d), from
    which the whole 9-frame impact sequence and the landing (d+9) follow
  - every shot's kind (in / out / card) and measured zoom amount
  - the end card hard cut and fade-to-dark window
  - the soundtrack, copied to music/template_<name>.m4a

Writes templates/<name>.json. Reference files are only analyzed for timing and
style; the template never contains any of their imagery.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

LAND_OFFSET = 9          # frames from dither start to the sharp landing frame (measured, both refs)


def _load_small_gray(video, width=540):
    cap = cv2.VideoCapture(str(video))
    out = []
    while True:
        ok, b = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        h = int(g.shape[0] * width / g.shape[1])
        out.append(cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA))
    return out


_ORB = cv2.ORB_create(3000)
_BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _sim(a, b):
    k1, d1 = _ORB.detectAndCompute(a, None)
    k2, d2 = _ORB.detectAndCompute(b, None)
    if d1 is None or d2 is None:
        return None
    m = _BF.match(d1, d2)
    if len(m) < 20:
        return None
    p1 = np.float32([k1[x.queryIdx].pt for x in m])
    p2 = np.float32([k2[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None or inl.sum() < 15:
        return None
    s = math.hypot(M[0, 0], M[1, 0])
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    A, t = M[:, :2], M[:, 2]
    try:
        fp = np.linalg.solve(np.eye(2) - A, t)
    except np.linalg.LinAlgError:
        fp = np.array([np.nan, np.nan])
    h, w = a.shape
    return {"scale": s, "rot": rot, "fixed": (fp[0] / w, fp[1] / h), "inliers": int(inl.sum())}


def _dominant_motion_ratio(a, b):
    """Fraction of feature matches explained by ONE similarity transform.
    Single-layer zoom shots: ~0.67-0.83. Card shots (card layer + background
    moving independently): ~0.19-0.38 (measured on both references)."""
    k1, d1 = _ORB.detectAndCompute(a, None)
    k2, d2 = _ORB.detectAndCompute(b, None)
    if d1 is None or d2 is None:
        return None
    m = _BF.match(d1, d2)
    if len(m) < 40:
        return None
    p1 = np.float32([k1[x.queryIdx].pt for x in m])
    p2 = np.float32([k2[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if M is None:
        return None
    return float(inl.sum()) / len(m)


def track_card_pop(video, card_start, drop, thresh=18):
    """Per-frame box of the intro card pop (the only thing changing on the
    static intro): [{f, cx, top, bot, w}] with f = frames after card_start.
    Used by the renderer to follow the reference's exact path."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    c0, d0 = int(round(card_start * fps)), int(round(drop * fps))
    frames, k = {}, 0
    while k <= d0:
        ok, b = cap.read()
        if not ok:
            break
        if k >= c0 - 30:
            frames[k] = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        k += 1
    cap.release()
    ref = np.median([frames[i] for i in range(c0 - 30, c0 - 10) if i in frames], 0)
    H, W = ref.shape
    track = []
    for k in range(c0 - 3, d0):
        if k not in frames:
            continue
        m = (np.abs(frames[k] - ref) > thresh).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, a = st[i]
        if a < 400 or w < 0.2 * W:        # ignore specks / the first sliver
            continue
        track.append({"f": k - c0, "cx": round((x + w / 2) / W, 4), "top": round(y / H, 4),
                      "bot": round((y + h) / H, 4), "w": round(w / W, 4)})
    return track


def build(video, name, analysis_dir=None):
    video = Path(video)
    if analysis_dir:
        adir = Path(analysis_dir)
    elif video.stem == "reference":                     # numbered layout: references/template_N/
        adir = video.parent / "analysis"
    else:
        adir = ROOT / "references" / f"{name}_analysis"
    rep_path = adir / "report.json"
    if not rep_path.exists():
        import reference_analyzer
        reference_analyzer.analyze(video, out_dir=adir)
    rep = json.loads(rep_path.read_text(encoding="utf-8-sig"))
    fps = rep["fps"]
    fr = rep["frames"]
    n = len(fr)
    luma = np.array([f["luma"] for f in fr])
    sharp = np.array([f["sharp"] for f in fr])
    t = lambda k: round(k / fps, 4)

    # ---------------- end card: last hard jump to a bright frame ----------------
    # end card = a near-flat bright gradient with small text: brightness ~228,
    # contrast (luma std) ~42, while the shot before it has clearly more
    # contrast (measured identically on all three Valtrez references). A
    # brightness jump alone also fires on a final shot that starts bright.
    lstd = np.array([f["luma_std"] for f in fr])
    end_k = None
    for k in range(int(n * 0.6), n - 2):
        # ...and it HOLDS: a one-frame white impact flash (followed by dark
        # frames) has the same numbers (template 7)
        held = k + 3 < n and min(luma[k + 1:k + 4]) > 190
        if 215 < luma[k] < 240 and lstd[k] < 46 and lstd[k - 1] > lstd[k] + 8 and lstd[k + 1] < 50 and held:
            end_k = k
            break
    if end_k is None:
        for k in range(n - 2, int(n * 0.6), -1):
            if luma[k] > 215 and luma[k - 1] < luma[k] - 40:
                end_k = k
                break
    # fade: from end card until luma drops near the final dark level
    dark_level = float(np.median(luma[-8:]))
    fade_start_k = end_k + 1
    while fade_start_k < n - 1 and luma[fade_start_k + 1] > luma[end_k] - 8:
        fade_start_k += 1
    fade_end_k = fade_start_k
    while fade_end_k < n - 1 and luma[fade_end_k] > dark_level + 6:
        fade_end_k += 1

    # ---------------- post-drop cut starts (dither start frames) ----------------
    events = [e for e in rep["events"] if e["cut_t"] > 5 and (end_k is None or e["start"] < end_k - 3)]
    starts = []
    for e in events:
        a = e["start"]
        # the halftone dot screen is extremely high-frequency (Laplacian var
        # ~6-9k vs a few hundred for the shots): d = first frame whose sharpness
        # jumps >3x over the previous frame
        d = next((k for k in range(max(1, a - 4), min(n, a + 6))
                  if sharp[k] > 3 * sharp[k - 1] and sharp[k] > 2500), a)
        starts.append(d)
    starts = sorted(set(starts))
    drop_k = starts[0]

    # ---------------- intro timings ----------------
    first_dark = float(np.median(luma[:4]))
    fade_in_start = next(k for k in range(n) if luma[k] > first_dark + 6)
    intro_level = float(np.median(luma[fade_in_start + 20:fade_in_start + 80]))
    fade_in_end = next(k for k in range(fade_in_start, n) if luma[k] > intro_level - 4)
    # card pop: first strong change in the late intro (after 70% of the hold)
    pdiff = np.array([f["pix_diff"] for f in fr])
    lo = fade_in_end + int(0.7 * (drop_k - fade_in_end))
    base = np.median(pdiff[fade_in_end + 10:lo])
    card_k = next((k for k in range(lo, drop_k) if pdiff[k] > base + 4), None)

    # ---------------- shot kinds + zoom (measure between landing and last clean frame) ----------------
    print("measuring shots ...")
    small = _load_small_gray(video)
    shot_ends = starts[1:] + [end_k]
    shots = []
    for d, nxt in zip(starts, shot_ends):
        land, last = d + LAND_OFFSET, nxt - 1
        # card test on a short pair well inside the shot (sharp, little change)
        a_, b_ = min(d + 17, last - 4), min(d + 23, last - 1)
        ratio = _dominant_motion_ratio(small[a_], small[b_]) if b_ > a_ else None
        m = _sim(small[land], small[last]) if last > land + 3 else None
        if ratio is not None and ratio < 0.5:
            kind = "card"
        elif m is not None and m["inliers"] >= 15:
            kind = "in" if m["scale"] >= 1.0 else "out"
        else:
            kind = "card"
        shot = {"kind": kind, "land_frames": last - land, "single_motion_ratio": round(ratio, 2) if ratio else None,
                "confident": ratio is not None and (ratio < 0.45 or (ratio > 0.55 and m is not None))}
        if kind != "card" and m:
            shot["zoom"] = round(m["scale"], 3)
            shot["anchor"] = [round(float(v), 3) for v in m["fixed"]]
        shots.append(shot)

    # Low-confidence shots (heavy motion blur starved the tracker): these edits
    # repeat a fixed shot cycle, so vote on the kind by position in the cycle
    # using only confidently measured shots.
    for period in (4, 3, 2):
        votes = {}
        for i, s in enumerate(shots):
            if s["confident"]:
                votes.setdefault(i % period, []).append(s["kind"])
        consistent = all(len(set(v)) == 1 for v in votes.values()) and len(votes) == period
        if consistent:
            for i, s in enumerate(shots):
                if not s["confident"]:
                    guess = votes[i % period][0]
                    if guess != s["kind"]:
                        print(f"  shot {i}: low-confidence '{s['kind']}' -> '{guess}' (cycle of {period})")
                        s["kind"] = guess
            break

    # typical zoom per kind (template uses the median so every slot is consistent)
    def med(kind, key):
        v = [s[key] for s in shots if s["kind"] == kind and key in s]
        return v
    zin = med("in", "zoom")
    zout = med("out", "zoom")
    ain = np.median(np.array(med("in", "anchor")), 0) if zin else [0.47, 0.23]
    aout = np.median(np.array(med("out", "anchor")), 0) if zout else [0.5, 0.57]

    # ---------------- soundtrack ----------------
    music_rel = f"music/template_{name}.m4a"
    (ROOT / "music").mkdir(exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-vn",
                    "-c:a", "aac", "-b:a", "256k", str(ROOT / music_rel)], check=True)

    h_ref, w_ref = rep["height"], rep["width"]
    tpl = {
        "name": name,
        "reference": str(video.relative_to(ROOT)).replace("\\", "/") if video.is_absolute() and ROOT in video.parents
        else str(video),
        "reference_fps": round(fps, 3),
        "size": [w_ref, h_ref],
        "fps": 60,
        "duration": t(n),
        "music": music_rel,
        "intro": {
            "fade_start": t(fade_in_start), "fade_end": t(fade_in_end),
            "card_start": t(card_k) if card_k else t(drop_k) - 0.5,
        },
        "cuts": [t(d) for d in starts],
        "shots": [{"kind": s["kind"]} for s in shots],
        "end": {"card": t(end_k), "fade_start": t(fade_start_k), "fade_end": t(fade_end_k)},
        "zoom": {
            "in": round(float(np.median(zin)), 3) if zin else 1.65,
            "out": round(float(np.median(zout)), 3) if zout else 0.6,
            "in_anchor": [round(float(v), 3) for v in ain],
            "out_anchor": [round(float(v), 3) for v in aout],
        },
        "measured_shots": shots,
    }
    out = ROOT / "templates" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        # keep hand-calibrated render style (tuned against the reference)
        old = json.loads(out.read_text(encoding="utf-8-sig"))
        if "style" in old:
            tpl["style"] = old["style"]
        # hand-verified shot kinds (checked frame by frame) survive rebuilds
        for idx, kind in old.get("kind_overrides", {}).items():
            if int(idx) < len(tpl["shots"]):
                tpl["shots"][int(idx)]["kind"] = kind
        if old.get("kind_overrides"):
            tpl["kind_overrides"] = old["kind_overrides"]
    tpl["intro"]["pop_track"] = track_card_pop(video, tpl["intro"]["card_start"], tpl["cuts"][0])
    out.write_text(json.dumps(tpl, indent=2), encoding="utf-8")
    return tpl, out, starts


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    tpl, out, starts = build(sys.argv[1], sys.argv[2])
    print(json.dumps({k: v for k, v in tpl.items() if k != "measured_shots"}, indent=1))
    print("dither start frames:", starts)
    print("->", out)
