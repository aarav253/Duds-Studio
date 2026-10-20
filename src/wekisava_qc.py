"""Quality check for wekisava-engine renders, frame by frame vs the reference.

    python src/wekisava_qc.py edits/<manifest>.json

Checks: format + audio; every segment boundary lands on the reference frame
(detected cuts); per-segment brightness / darkness / colour (cyan share)
vs the same segment in the reference; strobe black frames exact; smear
frames really smeared; smoothness (no single-frame jumps inside held shots);
outro timing. Writes output/qc/<id>/qc_report.md + segments.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_check import PASS, WARN, FAIL, Report, audio_of  # noqa: E402


def scan(path, w=270):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)))
    out = []
    prev = None
    while True:
        ok, b = cap.read()
        if not ok:
            break
        s = cv2.resize(b, (w, int(w * b.shape[0] / b.shape[1])), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
        cyan = float(((hsv[..., 0] > 80) & (hsv[..., 0] < 100) & (hsv[..., 1] > 80)).mean())
        red = float((((hsv[..., 0] < 8) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 60)).mean())
        yel = float(((hsv[..., 0] > 18) & (hsv[..., 0] < 38) & (hsv[..., 1] > 90) & (hsv[..., 2] > 80)).mean())
        gx = np.abs(np.diff(g, axis=1)).mean()
        gy = np.abs(np.diff(g, axis=0)).mean()
        out.append({"luma": float(g.mean()), "dark": float((g < 16).mean()), "cyan": cyan, "red": red, "yel": yel,
                    "smear": float(gy / (gx + 1e-6)), "pd": float(np.abs(g - prev).mean()) if prev is not None else 0.0,
                    "thumb": s})
        prev = g
    return fps, size, out


def run(manifest_path, render_path=None, log=print):
    mp = Path(manifest_path)
    mp = mp if mp.is_absolute() else ROOT / mp
    m = json.loads(mp.read_text(encoding="utf-8-sig"))
    t = json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig"))
    render = Path(render_path) if render_path else ROOT / "output" / f"{m['id']}.mp4"
    outdir = ROOT / "output" / "qc" / m["id"]
    outdir.mkdir(parents=True, exist_ok=True)
    R = Report()
    log("  [qc] scanning render + reference ...")
    fq, size_q, q = scan(render)
    fr, _, r = scan(ROOT / t["reference"])

    W, H = t.get("frame_size", t["size"])
    R.add("format", "resolution", PASS if tuple(size_q) == (W, H) else FAIL, f"{size_q[0]}x{size_q[1]} (template {W}x{H})")
    R.add("format", "frame count", PASS if abs(len(q) - t["frames"]) <= 1 else FAIL, f"{len(q)} (template {t['frames']})")
    try:
        aq, am = audio_of(render), audio_of(ROOT / t["music"])
        n = min(len(aq), len(am), 22050 * 6)
        a, b = aq[22050:22050 + n // 2], am[22050:22050 + n // 2]
        xc = np.correlate(a - a.mean(), b - b.mean(), "full")
        lag = (np.argmax(xc) - (len(b) - 1)) / 22050
        R.add("format", "audio aligned", PASS if abs(lag) < 0.012 else FAIL, f"offset {lag * 1000:+.1f} ms")
    except Exception as e:  # noqa: BLE001
        R.add("format", "audio", FAIL, str(e))

    n = min(len(q), len(r))
    rows = []
    section_luma = {}
    for seg in t["timeline"]:
        s, e = seg["start"], min(seg["end"], n - 1)
        if e < s:
            continue
        typ = seg["type"]
        name = f"{typ} {s}-{e}" + (f" [{seg.get('slot')}]" if seg.get("slot") else "")
        Q, Rf = q[s:e + 1], r[s:e + 1]
        ql = np.mean([f["luma"] for f in Q])
        rl = np.mean([f["luma"] for f in Rf])
        qd, rd = np.mean([f["dark"] for f in Q]), np.mean([f["dark"] for f in Rf])
        qc_, rc = np.mean([f["cyan"] for f in Q]), np.mean([f["cyan"] for f in Rf])
        issues = []
        if typ == "black":
            if ql > 20:
                issues.append(f"not black (luma {ql:.0f})")
        elif typ == "white":
            if max(f["luma"] for f in Q) < 180:
                issues.append(f"flash not bright (peak {max(f['luma'] for f in Q):.0f})")
        elif typ == "smear":
            sm = np.median([f["smear"] for f in Q])
            if sm < 1.6:
                issues.append(f"not smeared (horizontal ratio {sm:.2f}, ref {np.median([f['smear'] for f in Rf]):.2f})")
            pk_q, pk_r = int(np.argmax([f["luma"] for f in Q])), int(np.argmax([f["luma"] for f in Rf]))
            if abs(pk_q - pk_r) > 2:
                issues.append(f"flare peak on frame +{pk_q} (ref +{pk_r})")
        elif typ == "shot":
            # per-shot brightness depends on the user's panel vs the reference
            # panel in that slot; brightness is checked per SECTION below
            sec = {"bw": "intro (B&W)", "cyan": "drop (cyan)", "cyan_strip": "drop (cyan)", "eyes": "eyes (dark)",
                   "intro": "intro (duotone)", "duo": "drop (duotone)", "lines": "build-up (lines)",
                   "red": "drop (red)", "yellow": "drop (yellow)", "bright": "outro (bright)"}.get(seg["look"], seg["look"])
            for acc, key in (("red", "red"), ("yellow", "yel")):
                if seg.get("look") == acc:
                    qa, ra = np.mean([f[key] for f in Q]), np.mean([f[key] for f in Rf])
                    if ra > 0.03 and qa < 0.35 * ra:
                        issues.append(f"not {acc} enough ({qa:.2f} vs ref {ra:.2f})")
            section_luma.setdefault(sec, ([], []))
            section_luma[sec][0].append(ql)
            section_luma[sec][1].append(rl)
            if seg.get("look") in ("cyan", "cyan_strip", "eyes") and rc > 0.1 and qc_ < 0.4 * rc:
                issues.append(f"not cyan enough ({qc_:.2f} vs ref {rc:.2f})")
            if seg.get("look") == "bw" and qc_ > 0.05:
                issues.append(f"B&W section has colour ({qc_:.2f})")
            bf = seg.get("black_frames", [])
            if bf:
                miss = [k for k in bf if k < n and q[k]["luma"] > 20]
                extra = [k for k in range(s, e + 1) if k not in bf and q[k]["luma"] < 15]
                if miss or extra:
                    issues.append(f"strobe wrong: missing black {miss[:6]}, unexpected black {extra[:6]}")
            # smoothness: a held shot must not jump; compare with the reference's own motion level
            inner = [f["pd"] for f in Q[2:]] if len(Q) > 3 else []
            if inner and not bf:
                jump = max(inner)
                ref_jump = max([f["pd"] for f in Rf[2:]]) if len(Rf) > 3 else jump
                if jump > max(12, 2.2 * ref_jump):
                    k = s + 2 + int(np.argmax(inner))
                    issues.append(f"jerky: frame {k} changes {jump:.1f} (ref max {ref_jump:.1f})")
        status = PASS if not issues else WARN if len(issues) == 1 else FAIL
        R.add("segments", name, status, "; ".join(issues) or
              f"luma {ql:.0f} (ref {rl:.0f}), dark {qd * 100:.0f}% (ref {rd * 100:.0f}%), cyan {qc_:.2f} (ref {rc:.2f})")
        rows.append((s, e))

    for sec, (ours, ref) in section_luma.items():
        a, b = float(np.mean(ours)), float(np.mean(ref))
        tol = 0.2 * b if "intro" not in sec else 0.4 * b     # intro: whites are kept white on purpose
        R.add("grade", f"{sec} average brightness", PASS if abs(a - b) <= tol else WARN, f"{a:.0f} vs ref {b:.0f}")

    # cut alignment: reference cuts (big frame changes) must be cuts in the render too
    # (skip the first 30 frames: the reference intro is a continuous whip-pan,
    # every frame differs, which is motion not cuts)
    ref_cuts = [k for k in range(30, n) if r[k]["pd"] > 35]
    miss = [k for k in ref_cuts if max(q[j]["pd"] for j in range(max(1, k - 1), min(n, k + 2))) < 12]
    R.add("timing", "every reference cut is a cut in the render", PASS if len(miss) <= 2 else WARN,
          f"{len(ref_cuts) - len(miss)}/{len(ref_cuts)} matched" + (f", unmatched frames {miss[:10]}" if miss else ""))

    # sheet: reference vs render at every segment midpoint
    mids = [(s + e) // 2 for s, e in rows]
    tw = 90
    cols = []
    for k in mids:
        a = cv2.resize(r[k]["thumb"], (tw, int(tw * 1.255)))
        b = cv2.resize(q[k]["thumb"], (tw, int(tw * 1.255)))
        cv2.putText(a, str(k), (2, 10), 0, 0.3, (0, 0, 255), 1)
        cols.append(np.vstack([a, b]))
    per = 27
    sheet = np.vstack([np.hstack(cols[i:i + per] + [np.zeros_like(cols[0])] * (per - len(cols[i:i + per])))
                       for i in range(0, len(cols), per)])
    cv2.imwrite(str(outdir / "segments.png"), sheet)

    c = R.counts()
    lines = [f"# QC report: {m['id']} ({t.get('engine', 'wekisava')} engine)", "", f"**Overall: {R.overall()}** "
             f"({c[PASS]} pass, {c[WARN]} warn, {c[FAIL]} fail)", "", "| section | check | status | detail |", "|---|---|---|---|"]
    lines += [f"| {i['section']} | {i['check']} | {i['status']} | {i['detail']} |" for i in R.items]
    (outdir / "qc_report.md").write_text("\n".join(lines), encoding="utf-8")
    s = {"overall": R.overall(), "counts": c, "items": R.items, "report": str(outdir.relative_to(ROOT)).replace("\\", "/")}
    log(f"  [qc] {R.overall()}: {c[PASS]} pass, {c[WARN]} warn, {c[FAIL]} fail -> {outdir}")
    return s


if __name__ == "__main__":
    s = run(sys.argv[1])
    for i in s["items"]:
        if i["status"] != PASS:
            print(f"    [{i['section']}] {i['check']}: {i['detail']}")
