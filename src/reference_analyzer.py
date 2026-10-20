"""Frame-by-frame analysis of a reference edit (for studying an editing STYLE -
reference files are never used as footage).

    python src/reference_analyzer.py references/some_edit.mp4

Writes references/template_N/analysis/ (or <name>_analysis/ next to other videos):
  report.json      every metric, machine-readable
  summary.txt      human-readable: events, shot table, beat alignment stats
  timeline.png     brightness / change / zoom curves with beats, downbeats,
                   drum hits and the drop overlaid
  overview.png     every Nth frame
  event_XX.png     each cut/transition: the frames around it, labeled with
                   frame number, time and metrics

Per frame: luma mean/std, sharpness (Laplacian variance), histogram change vs
previous frame, and camera motion vs previous frame (scale / rotation /
translation from ORB feature matching + a RANSAC similarity fit).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

THUMB_W = 216
MOTION_W = 360


def _font(size=13):
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", size)
    except OSError:
        return ImageFont.load_default()


# --------------------------------------------------------------------------
# Per-frame metrics
# --------------------------------------------------------------------------

_ORB = cv2.ORB_create(nfeatures=800)
_BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _motion(prev_gray, gray):
    """Similarity transform prev->cur. Returns (scale, rot_deg, tx, ty, inlier_frac) or None."""
    k1, d1 = _ORB.detectAndCompute(prev_gray, None)
    k2, d2 = _ORB.detectAndCompute(gray, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None
    matches = _BF.match(d1, d2)
    if len(matches) < 12:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl is None:
        return None
    frac = float(inl.sum()) / len(matches)
    if inl.sum() < 10 or frac < 0.25:
        return None
    # Static overlays (caption bars, watermarks) are feature-rich and can win
    # RANSAC with an identity fit while the artwork underneath is zooming.
    # If the best fit is ~identity, refit on the remaining matches and prefer
    # a real second motion when it has enough support.
    near_identity = abs(math.hypot(M[0, 0], M[1, 0]) - 1) < 1e-3 and abs(M[0, 2]) < 0.3 and abs(M[1, 2]) < 0.3
    rest = ~inl.ravel().astype(bool)
    if near_identity and rest.sum() >= 15:
        M2, inl2 = cv2.estimateAffinePartial2D(p1[rest], p2[rest], method=cv2.RANSAC,
                                               ransacReprojThreshold=2.0)
        if M2 is not None and inl2 is not None and inl2.sum() >= max(10, 0.1 * len(matches)):
            if abs(math.hypot(M2[0, 0], M2[1, 0]) - 1) > 2e-4:
                M, frac = M2, float(inl2.sum()) / len(matches)
    s = math.hypot(M[0, 0], M[1, 0])
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    h, w = gray.shape
    # translation of the frame center, as a fraction of frame width
    cx, cy = w / 2, h / 2
    ncx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    ncy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    return s, rot, (ncx - cx) / w, (ncy - cy) / w, frac


def scan(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"can't open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, thumbs = [], []
    prev_hist = prev_small = None
    ring = []   # last STRIDE+1 small frames: slow zooms move <0.1px per frame, so
    STRIDE = 5  # also measure motion across 5-frame gaps (5x the displacement)
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (MOTION_W, int(H * MOTION_W / W)), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([small], [0], None, [32], [0, 256]).ravel()
        hist /= hist.sum() + 1e-9
        sat = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())
        f = {
            "i": i, "t": round(t, 4),
            "luma": round(float(gray.mean()), 2), "luma_std": round(float(gray.std()), 2),
            "sat": round(sat, 1),
            "sharp": round(float(cv2.Laplacian(small, cv2.CV_64F).var()), 1),
            "hist_diff": round(float(0.5 * np.abs(hist - prev_hist).sum()), 4) if prev_hist is not None else 0.0,
            "pix_diff": round(float(np.abs(small.astype(np.int16) - prev_small).mean()), 2)
            if prev_small is not None else 0.0,
            "motion": None,
        }
        if prev_small is not None and f["hist_diff"] < 0.5:
            m = _motion(prev_small.astype(np.uint8), small)
            if m:
                f["motion"] = {"scale": round(m[0], 5), "rot": round(m[1], 3), "tx": round(m[2], 5),
                               "ty": round(m[3], 5), "inliers": round(m[4], 2)}
        ring.append(small)
        ring = ring[-(STRIDE + 1):]
        f["motion5"] = None
        if i % STRIDE == 0 and len(ring) == STRIDE + 1:
            m5 = _motion(ring[0], small)
            if m5:
                f["motion5"] = {"scale": round(m5[0], 5), "rot": round(m5[1], 3)}
        frames.append(f)
        th = Image.fromarray(rgb)
        th.thumbnail((THUMB_W, THUMB_W * 2))
        thumbs.append(th)
        prev_hist, prev_small = hist, small.astype(np.int16)
        i += 1
    cap.release()
    return {"fps": fps, "width": W, "height": H, "n_frames": len(frames)}, frames, thumbs


# --------------------------------------------------------------------------
# Events: cuts, flashes, transitions
# --------------------------------------------------------------------------

def _robust_thresh(x, k=6.0, floor=0.0):
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return max(floor, med + k * 1.4826 * mad)


MERGE_GAP = 4  # (at 30fps; scaled to the real fps) flagged frames this close belong to one transition (lead-in, flash, resolve tail)


def find_events(frames, fps=30.0, merge_gap=None):
    """Frames whose content changes abruptly (cuts) or that are flash frames.
    Flagged frames within `merge_gap` frames are grouped into one transition
    event (a dissolve 3 frames before a flash + a blur tail after it is ONE cut)."""
    k = fps / 30.0
    merge_gap = merge_gap or max(1, round(MERGE_GAP * k))
    max_span = round(12 * k)
    hd = np.array([f["hist_diff"] for f in frames])
    pdiff = np.array([f["pix_diff"] for f in frames])
    luma = np.array([f["luma"] for f in frames])
    std = np.array([f["luma_std"] for f in frames])
    th_h = _robust_thresh(hd, 6, 0.12)
    th_p = _robust_thresh(pdiff, 6, 18)
    flagged = []
    for i, f in enumerate(frames):
        flash = std[i] < 12 and (luma[i] < 30 or luma[i] > 225)
        f["flash"] = "black" if flash and luma[i] < 30 else "white" if flash else None
        # a big pixel change that the motion tracker explains (zoom/shake of the
        # same image) is camera motion, not a cut
        tracked = f["motion"] is not None and f["motion"]["inliers"] >= 0.5
        if i and (hd[i] > th_h or (pdiff[i] > th_p and not tracked) or f["flash"]):
            flagged.append(i)
    # 1) contiguous runs, 2) merge neighbouring runs into one transition only if
    # the result stays short and has at most one flash (no chaining across cuts)
    runs = []
    for i in flagged:
        if runs and i - runs[-1][-1] <= 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    events = []
    for r in runs:
        if events:
            prev = events[-1]["flagged"]
            n_flash = sum(1 for k in prev + r if frames[k]["flash"] and
                          (k == 0 or not frames[k - 1]["flash"]))
            if r[0] - prev[-1] <= merge_gap and r[-1] - prev[0] <= max_span and n_flash <= 1:
                events[-1]["flagged"] += r
                events[-1]["end"] = r[-1]
                continue
        events.append({"start": r[0], "end": r[-1], "flagged": list(r)})
    for e in events:
        span = frames[e["start"]:e["end"] + 1]
        e["t"] = frames[e["start"]]["t"]
        e["frames"] = e["end"] - e["start"] + 1
        e["flash_frames"] = [f["i"] for f in span if f["flash"]]
        e["flash_color"] = span[[f["i"] for f in span].index(e["flash_frames"][0])]["flash"] \
            if e["flash_frames"] else None
        # the "cut point": first flash frame if any, else biggest change in the span
        e["cut_frame"] = e["flash_frames"][0] if e["flash_frames"] else \
            max(span, key=lambda f: f["hist_diff"] + f["pix_diff"] / 100)["i"]
        e["cut_t"] = frames[e["cut_frame"]]["t"]
        after = frames[e["end"] + 1:e["end"] + 6]
        sharp_after = [f["sharp"] for f in after]
        e["resolve"] = "blur" if len(sharp_after) >= 3 and sharp_after[0] < 0.6 * max(sharp_after) else "sharp"
        e["kind"] = (f"flash_{e['flash_color']}" if e["flash_color"] else
                     "hard_cut" if len(e["flagged"]) == 1 else "multi_frame")
        # frames of visible transition treatment before/after the cut point
        e["lead_frames"] = e["cut_frame"] - e["start"]
        e["tail_frames"] = e["end"] - e["cut_frame"]
    return events, {"hist_thresh": round(th_h, 4), "pix_thresh": round(th_p, 2)}


# --------------------------------------------------------------------------
# Shots: camera motion between events
# --------------------------------------------------------------------------

def shots_from(frames, events, fps):
    bounds = [0] + [e["end"] + 1 for e in events] + [len(frames)]
    shots = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        seg = frames[a:b]
        scales, rots, txs, tys = [], [], [], []
        for f in seg[1:]:
            m = f["motion"]
            scales.append(m["scale"] if m else 1.0)
            rots.append(m["rot"] if m else 0.0)
            txs.append(m["tx"] if m else 0.0)
            tys.append(m["ty"] if m else 0.0)
        cum = np.cumprod([1.0] + scales)
        tracked = sum(1 for f in seg[1:] if f["motion"])
        # slow-zoom estimate from 5-frame-stride samples fully inside this shot
        s5 = [f["motion5"]["scale"] for f in frames[a + 5:b] if f.get("motion5")]
        zoom5 = float(np.prod(s5)) if s5 else None
        cover5 = len(s5) * 5 / max(1, b - a)
        jitter = float(np.std(np.diff(txs)) + np.std(np.diff(tys))) if len(txs) > 2 else 0.0
        shots.append({
            "start_frame": a, "end_frame": b - 1, "t": seg[0]["t"], "dur": round((b - a) / fps, 3),
            "frames": b - a, "tracked_frac": round(tracked / max(1, len(seg) - 1), 2),
            "zoom_total": round(float(cum[-1]), 4),
            "zoom_per_sec": round(float(cum[-1] ** (fps / max(1, len(scales))) - 1), 4) if scales else 0.0,
            "zoom_curve": [round(float(c), 4) for c in cum],
            # more accurate for slow drifts; covers `zoom_stride5_coverage` of the shot
            "zoom_total_stride5": round(zoom5, 4) if zoom5 else None,
            "zoom_stride5_coverage": round(cover5, 2),
            "rot_total_deg": round(float(np.sum(rots)), 2),
            "pan_total": [round(float(np.sum(txs)), 4), round(float(np.sum(tys)), 4)],
            "shake": round(jitter, 5),
            "mean_luma": round(float(np.mean([f["luma"] for f in seg])), 1),
            "mean_sat": round(float(np.mean([f["sat"] for f in seg])), 1),
        })
    return shots


# --------------------------------------------------------------------------
# Accurate zoom curves: ECC alignment against each shot's first frame
# --------------------------------------------------------------------------

ECC_W = 480


def _ecc_scale(ref, img, init_scale=1.0):
    """Zoom of `img` relative to `ref` (>1 = zoomed in), measured over a long
    baseline (shot's first frame -> this frame) with feature matching, which
    only looks at inked detail (flat white manga areas match at ANY scale, so
    whole-image intensity methods can't measure zoom on this material).
    Static overlays are handled by _motion's two-model refit."""
    m = _motion(ref, img)
    return m[0] if m else None


def refine_zoom(video, shots, fps=30.0, max_samples=40):
    """Second decode pass: for each shot, measure zoom of sampled frames
    relative to the shot's first frame (long baseline -> slow drifts that
    move <0.1px per frame become measurable)."""
    want = {}
    for si, s in enumerate(shots):
        a, b = s["start_frame"], s["end_frame"]
        n = b - a
        step = max(1, int(math.ceil(n / max_samples)))
        idx = list(range(a, b + 1, step))
        if idx[-1] != b:
            idx.append(b)
        for k in idx:
            want.setdefault(k, []).append(si)
        s["_idx"] = idx
    cap = cv2.VideoCapture(str(video))
    imgs = {}
    i = 0
    last = max(want) if want else -1
    while i <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in want:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            h = int(g.shape[0] * ECC_W / g.shape[1])
            imgs[i] = cv2.resize(g, (ECC_W, h), interpolation=cv2.INTER_AREA)
        i += 1
    cap.release()
    for s in shots:
        idx = [k for k in s.pop("_idx") if k in imgs]
        if len(idx) < 2:
            s["zoom_ecc"] = None
            continue
        ref = imgs[idx[0]]
        blur = lambda a: cv2.GaussianBlur(a, (9, 9), 0).astype(np.float32)
        s["first_last_diff"] = round(float(np.abs(blur(ref) - blur(imgs[idx[-1]])).mean()), 1)
        curve = [(0, 1.0)]
        chain = s["zoom_curve"]
        for k in idx[1:]:
            off = k - s["start_frame"]
            init = chain[min(off, len(chain) - 1)] if chain else 1.0
            sc = _ecc_scale(ref, imgs[k], init)
            if sc is not None and 0.2 < sc < 20:
                curve.append((off, round(sc, 5)))
        s["zoom_ecc"] = curve
        s["zoom_total"] = curve[-1][1] if len(curve) > 1 else s["zoom_total"]
        dur = max(1e-6, (curve[-1][0]) / fps) if len(curve) > 1 else 1
        s["zoom_per_sec"] = round(float(curve[-1][1] ** (1 / dur) - 1), 4) if len(curve) > 1 else 0.0


# --------------------------------------------------------------------------
# Audio alignment
# --------------------------------------------------------------------------

def extract_audio(video, out_wav):
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
                        "-vn", "-ac", "2", "-ar", "44100", str(out_wav)], capture_output=True)
    return r.returncode == 0 and Path(out_wav).exists() and Path(out_wav).stat().st_size > 1000


def align(events, music, fps):
    def nearest(t, arr):
        if not arr:
            return None
        n = min(arr, key=lambda x: abs(x - t))
        return round((t - n) * fps, 2)  # offset in frames (+ = cut after the grid point)

    beats, downs = music["beats"], music["downbeats"]
    onsets = [o["t"] for o in music["drum_onsets"]]
    strong = [o["t"] for o in music["drum_onsets"] if o["strength"] >= 0.25]
    for e in events:
        t = e["cut_t"]
        e["beat_off"] = nearest(t, beats)
        e["downbeat_off"] = nearest(t, downs)
        e["onset_off"] = nearest(t, onsets)
        e["strong_onset_off"] = nearest(t, strong)
        if beats:
            k = int(np.searchsorted(beats, t))
            prev_b = beats[max(0, k - 1)]
            nxt_b = beats[min(len(beats) - 1, k)]
            ibi = (nxt_b - prev_b) or 1
            e["beat_phase"] = round(((t - prev_b) / ibi) % 1.0, 2)

    def pct(key, tol):
        vals = [e[key] for e in events if e.get(key) is not None]
        return round(100 * sum(abs(v) <= tol for v in vals) / len(vals), 1) if vals else None

    return {f"{k}_within_{tol}f": pct(k, tol) for k in ("beat_off", "downbeat_off", "onset_off", "strong_onset_off")
            for tol in (1, 2)}


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

def _label(im, lines, color=(255, 60, 60)):
    im = im.copy()
    d = ImageDraw.Draw(im)
    f = _font(12)
    y = 2
    for ln in lines:
        w = d.textlength(ln, font=f)
        d.rectangle([0, y, w + 6, y + 15], fill=(0, 0, 0))
        d.text((3, y + 1), ln, font=f, fill=color if y == 2 else (255, 255, 255))
        y += 16
    return im


def _grid(tiles, cols):
    w, h = tiles[0].size
    rows = -(-len(tiles) // cols)
    g = Image.new("RGB", (cols * w, rows * h), (40, 40, 40))
    for k, t in enumerate(tiles):
        g.paste(t, ((k % cols) * w, (k // cols) * h))
    return g


def save_event_sheets(out, events, frames, thumbs, pad=3, max_events=60):
    for n, e in enumerate(events[:max_events]):
        a, b = max(0, e["start"] - pad), min(len(frames) - 1, e["end"] + pad + 3)
        tiles = []
        for i in range(a, b + 1):
            f = frames[i]
            m = f["motion"]
            tag = "FLASH" if f["flash"] else ("CUT" if i == e["cut_frame"] else "")
            tiles.append(_label(thumbs[i], [
                f"#{i} {f['t']:.3f}s {tag}",
                f"L{f['luma']:.0f} sh{f['sharp']:.0f}",
                f"zoom{(m['scale'] - 1) * 100:+.2f}%" if m else "zoom ?",
            ]))
        _grid(tiles, min(8, len(tiles))).save(out / f"event_{n:02d}.png")


def save_overview(out, frames, thumbs, n=36):
    step = max(1, len(frames) // n)
    tiles = [_label(thumbs[i], [f"#{i} {frames[i]['t']:.2f}s"]) for i in range(0, len(frames), step)]
    _grid(tiles, 9).save(out / "overview.png")


def save_timeline(out, frames, events, shots, music, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([f["t"] for f in frames])
    fig, ax = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
    ax[0].plot(t, [f["luma"] for f in frames], lw=0.8)
    ax[0].set_ylabel("luma")
    ax[1].plot(t, [f["hist_diff"] for f in frames], lw=0.8, color="tab:orange")
    ax[1].set_ylabel("change")
    ax[2].plot(t, [f["sharp"] for f in frames], lw=0.8, color="tab:green")
    ax[2].set_ylabel("sharpness")
    for s in shots:
        if s.get("zoom_ecc") and len(s["zoom_ecc"]) > 1:
            offs, sc = zip(*s["zoom_ecc"])
            ts = frames[s["start_frame"]]["t"] + np.array(offs) / fps
            ax[3].plot(ts, (np.array(sc) - 1) * 100, lw=1.2, marker=".", ms=2)
        else:
            ts = np.array([frames[s["start_frame"]]["t"] + k / fps for k in range(len(s["zoom_curve"]))])
            ax[3].plot(ts, (np.array(s["zoom_curve"]) - 1) * 100, lw=1.0, ls=":")
    ax[3].set_ylabel("zoom % (per shot)")
    for a in ax:
        for e in events:
            a.axvline(e["cut_t"], color="red", lw=0.6, alpha=0.6)
        if music:
            for b in music["beats"]:
                a.axvline(b, color="gray", lw=0.4, alpha=0.4, ls=":")
            for b in music["downbeats"]:
                a.axvline(b, color="black", lw=0.6, alpha=0.5, ls="--")
            if music.get("drop"):
                a.axvline(music["drop"], color="magenta", lw=2, alpha=0.7)
    if music:
        on = [o["t"] for o in music["drum_onsets"]]
        st = [o["strength"] for o in music["drum_onsets"]]
        ax[1].scatter(on, np.array(st) * max(0.2, max(f["hist_diff"] for f in frames)), s=6, color="purple",
                      label="drum hits")
        ax[1].legend(loc="upper right")
    ax[3].set_xlabel("seconds  (red = detected cuts, dotted = beats, dashed = downbeats, magenta = drop)")
    fig.tight_layout()
    fig.savefig(out / "timeline.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def analyze(video, out_dir=None, use_neural=True):
    video = Path(video)
    # numbered layout: references/template_N/reference.mp4 -> references/template_N/analysis/
    out = Path(out_dir) if out_dir else video.parent / ("analysis" if video.stem == "reference" else f"{video.stem}_analysis")
    out.mkdir(parents=True, exist_ok=True)
    print(f"scanning {video.name} ...")
    meta, frames, thumbs = scan(video)
    fps = meta["fps"]
    events, thresholds = find_events(frames, fps)
    shots = shots_from(frames, events, fps)
    print("measuring zoom curves ...")
    refine_zoom(video, shots, fps)

    music, align_stats = None, None
    wav = out / "audio.wav"
    if extract_audio(video, wav):
        import music_analysis
        print("analyzing soundtrack ...")
        music = music_analysis.analyze(wav, use_neural=use_neural, verbose=True)
        align_stats = align(events, music, fps)
        wav.unlink()

    report = {"video": video.name, **meta, "duration": round(meta["n_frames"] / fps, 3),
              "thresholds": thresholds, "events": events, "shots": shots,
              "music": music, "alignment": align_stats, "frames": frames}
    (out / "report.json").write_text(json.dumps(report), encoding="utf-8")

    lines = [f"{video.name}: {meta['width']}x{meta['height']} @ {fps:.3f}fps, {meta['n_frames']} frames, "
             f"{report['duration']}s"]
    if music:
        lines.append(f"music: {music['method']} tempo={music['tempo']} bpm, drop={music['drop']}s, "
                     f"{len(music['beats'])} beats / {len(music['downbeats'])} downbeats / "
                     f"{len(music['drum_onsets'])} drum hits")
        lines.append(f"alignment (% of cuts within N frames): {align_stats}")
    gaps = np.diff([e["cut_t"] for e in events]) if len(events) > 1 else []
    if len(gaps):
        lines.append(f"cuts: {len(events)}  gap median {np.median(gaps):.3f}s  min {np.min(gaps):.3f}s  "
                     f"max {np.max(gaps):.3f}s")
    kinds = {}
    for e in events:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    lines.append(f"event kinds: {kinds}")
    lines.append("")
    lines.append(" #  frame   time    kind          lead/tail flashF resolve  beatOff dbOff onsetOff phase")
    for n, e in enumerate(events):
        lines.append(f"{n:2d} {e['cut_frame']:5d} {e['cut_t']:7.3f}  {e['kind']:12s} "
                     f"{e['lead_frames']:3d}/{e['tail_frames']:<3d}   "
                     f"{len(e['flash_frames']):5d}  {e['resolve']:6s} {str(e.get('beat_off')):>7} "
                     f"{str(e.get('downbeat_off')):>5} {str(e.get('onset_off')):>8} {str(e.get('beat_phase')):>5}")
    lines.append("")
    lines.append("(zoom from feature tracking; '?' = first/last frames differ a lot but no zoom was "
                 "measured - inspect visually, a static overlay may be hiding the motion)")
    lines.append(" shot  start    dur  frames tracked zoomTot  zoom5   zoom/s   rot    pan(x,y)          shake   luma  sat")
    for n, s in enumerate(shots):
        z5 = f"{s['zoom_total_stride5']:.4f}" if s["zoom_total_stride5"] else "   -  "
        if abs(s["zoom_total"] - 1) < 0.005 and s.get("first_last_diff", 0) > 8:
            z5 += "?"
        lines.append(f"{n:4d} {s['t']:7.3f} {s['dur']:6.3f} {s['frames']:6d} {s['tracked_frac']:6.2f} "
                     f"{s['zoom_total']:7.4f} {z5:>7} {s['zoom_per_sec']:+7.4f} {s['rot_total_deg']:+6.2f} "
                     f"{str(s['pan_total']):18s} {s['shake']:.4f} {s['mean_luma']:5.1f} {s['mean_sat']:5.1f}")
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("rendering sheets ...")
    save_overview(out, frames, thumbs)
    save_event_sheets(out, events, frames, thumbs)
    save_timeline(out, frames, events, shots, music, fps)
    print("\n".join(lines[:6]))
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for v in sys.argv[1:]:
        analyze(v)
