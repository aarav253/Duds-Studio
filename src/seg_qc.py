"""Frame-by-frame quality check of a seg_edit render (the current template-replication engine) against the
reference edit it replicates.  quality_check.py does the same job for the older template_edit renders.

    python src/seg_qc.py <id>          (output/<id>.mp4 must exist)

The reference is references/template_<N>/ (reference.mp4 + cuts.json), where N comes from the manifest's music
file (music/template_<N>_*.wav), which is how every seg manifest is tied to its template.

Checks (each PASS / WARN / FAIL):
  format   60 fps, resolution = manifest size, duration = body + outro
  timing   every reference cut in the body has a render cut within 2 frames (merged fast cuts are expected,
           so a few misses only WARN)
  sync     the render's audio is aligned with the reference audio (onset-envelope cross-correlation lag)
  blur     shots much softer than the rest of the edit (median Laplacian sharpness per shot)
  frozen   the picture not changing for 0.5 s or more inside the body
  black    black frames inside the body (outside the fade-in and the outro) where the reference isn't dark too

Writes output/qc/<id>/qc_report.json, qc_report.md and frames.png (reference vs render at 8 points), and
stores {overall, counts, report, issues} under the id in review_queue.json (the dashboard shows it).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, stft

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_queue as rq  # noqa: E402

FF = imageio_ffmpeg.get_ffmpeg_exe()


def scan(path):
    """per frame: mean luma, mean abs difference to the previous frame, Laplacian sharpness."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    luma, diff, sharp, prev = [], [], [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (216, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        luma.append(g.mean())
        diff.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        prev = g
    return fps, (W, H), np.array(luma), np.array(diff), np.array(sharp)


def audio(path, sr=22050):
    tmp = ROOT / "output" / f".qc_{Path(path).stem}.wav"
    subprocess.run([FF, "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(sr), str(tmp)], check=True)
    _, a = wavfile.read(str(tmp)); tmp.unlink()
    return a.astype(np.float32) / 32768


def onset_env(a, sr=22050, hop=256):
    _, _, Z = stft(a, sr, nperseg=1024, noverlap=1024 - hop)
    S = np.log1p(np.abs(Z))
    flux = np.maximum(0, np.diff(S, axis=1)).sum(0)
    return (flux - flux.mean()) / (flux.std() + 1e-9), sr / hop


def run(key, log=print):
    m = json.loads((ROOT / "edits" / f"{key}.json").read_text(encoding="utf-8-sig"))
    render = ROOT / "output" / f"{key}.mp4"
    if not render.exists():
        raise FileNotFoundError(f"{render} - build it first")
    out_dir = ROOT / "output" / "qc" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    tn = re.search(r"template_(\d+)", json.dumps(m.get("music", "")))
    ref_dir = ROOT / "references" / f"template_{tn.group(1)}" if tn else None
    ref = ref_dir / "reference.mp4" if ref_dir else None
    items = []

    def add(section, check, status, detail):
        items.append({"section": section, "check": check, "status": status, "detail": detail})

    body = float(m["body_end"])
    log(f"[qc] scanning {render.name}")
    fps, size, luma, diff, sharp = scan(render)
    n = len(luma); dur = n / fps
    o = m.get("outro", {"card": 0.033, "fade": 0.551, "dark": 0.2})
    want = body + o["card"] + o["fade"] + o["dark"]
    add("format", "frame rate", "PASS" if abs(fps - m.get("fps", 60)) < 0.01 else "FAIL", f"{fps:.2f} fps")
    add("format", "resolution", "PASS" if list(size) == list(m.get("size", [1080, 1350])) else "FAIL", f"{size[0]}x{size[1]}")
    add("format", "duration", "PASS" if abs(dur - want) < 0.1 else "WARN", f"{dur:.2f}s (expected {want:.2f}s)")

    kb = int(body * fps)
    thr = max(12.0, 4 * float(np.median(diff[1:kb])))
    rcuts = [i / fps for i in range(1, kb) if diff[i] > thr and diff[i] >= diff[i - 1] and (i + 1 >= n or diff[i] >= diff[i + 1])]

    if ref and ref.exists():
        cuts = json.loads((ref_dir / "cuts.json").read_text())
        refc = [c for c in cuts if 0.1 < c < body - 0.05]
        tol = 2 / fps + 0.5 / 25                                   # 2 render frames + half a reference frame (refs are 25-60 fps)
        hit = [c for c in refc if rcuts and min(abs(c - r) for r in rcuts) <= tol]
        frac = len(hit) / max(1, len(refc))
        miss = [round(c, 2) for c in refc if c not in hit]
        add("timing", "reference cuts matched", "PASS" if frac >= 0.8 else "WARN" if frac >= 0.5 else "FAIL",
            f"{len(hit)}/{len(refc)} within {tol * 1000:.0f} ms" + (f"; missed at {miss[:10]}" if miss else ""))
        log("[qc] audio sync")
        ea, sr_e = onset_env(audio(render)); eb, _ = onset_env(audio(ref))
        L = min(len(ea), len(eb), int(body * sr_e))
        if L > sr_e * 2:
            c = correlate(ea[:L], eb[:L], "full")
            mid = L - 1; w = int(0.3 * sr_e)
            seg = c[mid - w:mid + w + 1]; i = int(np.argmax(seg)) - w
            lag_ms = i / sr_e * 1000; peak = float(seg.max() / L)
            st = "PASS" if abs(lag_ms) <= 25 else "WARN" if abs(lag_ms) <= 60 else "FAIL"
            if peak < 0.25:
                st = "WARN"
            add("sync", "audio vs reference", st, f"lag {lag_ms:+.0f} ms, onset correlation {peak:.2f}"
                + (" (low: different mix or song version)" if peak < 0.25 else ""))
        rfps, _, rl, _, rs = scan(ref)
        ref_sharp = float(np.median(rs[:int(body * rfps)])) if len(rs) else 0
    else:
        add("timing", "reference", "WARN", "no reference video found for this manifest (music file not template_<N>)")
        ref_sharp = 0
        rfps, rl = None, None

    bounds = [0.0] + rcuts + [body]
    soft = []
    med = float(np.median(sharp[:kb]))
    for a, b in zip(bounds, bounds[1:]):
        k0, k1 = int(a * fps) + 3, int(b * fps) - 1                 # skip the first frames (transition blur is intended)
        if k1 - k0 < 6:
            continue
        s = float(np.median(sharp[k0:k1]))
        if s < 0.2 * med:
            soft.append(f"{a:.2f}-{b:.2f}s")
    nshots = max(1, len(bounds) - 1)
    add("blur", "soft shots", "PASS" if not soft else "WARN" if len(soft) <= 0.25 * nshots else "FAIL",
        (f"{len(soft)} shot(s) far softer than the edit's median: {soft[:8]}" if soft else "no unusually soft shots")
        + (f"; render/reference sharpness {med / ref_sharp:.2f}" if ref_sharp else ""))

    still = diff[:kb] < 0.25
    runs, k = [], 0
    while k < kb:
        if still[k]:
            j = k
            while j < kb and still[j]:
                j += 1
            if j - k >= int(0.5 * fps):
                runs.append((k / fps, j / fps))
            k = j
        else:
            k += 1
    longest = max([b - a for a, b in runs], default=0)
    add("frozen", "frozen picture", "PASS" if not runs else "WARN" if longest < 2 else "FAIL",
        f"{len(runs)} still stretch(es) >= 0.5 s" + (f", longest {longest:.2f}s at {runs[0][0]:.2f}s" if runs else ""))

    def ref_dark(i):                                               # black where the reference is black too = intended
        if rl is None:
            return False
        j = int(i / fps * rfps)
        return j < len(rl) and rl[j] < 25
    # a black frame is dark AND blank (a dark title card with text has detail, so it isn't one)
    blk = [i for i in range(int(0.4 * fps), kb) if luma[i] < 8 and sharp[i] < 0.05 * med and not ref_dark(i)]
    add("black", "black frames", "PASS" if not blk else "WARN" if len(blk) < int(0.25 * fps) else "FAIL",
        f"{len(blk)} black frame(s) inside the body" + (f", first at {blk[0] / fps:.2f}s" if blk else ""))

    # frames.png: reference (top) vs render (bottom) at 8 points of the body
    ts = np.linspace(0.5, body - 0.3, 8)
    def grab(path, t):
        cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
        f = f if ok else np.zeros((270, 216, 3), np.uint8)
        f = cv2.resize(f, (216, 270)); cv2.putText(f, f"{t:.1f}s", (4, 16), 0, 0.5, (0, 255, 255), 1)
        return f
    rows = [np.hstack([grab(render, t) for t in ts])]
    if ref and ref.exists():
        rows.insert(0, np.hstack([grab(ref, t) for t in ts]))
    cv2.imwrite(str(out_dir / "frames.png"), np.vstack(rows))

    counts = {s: sum(i["status"] == s for i in items) for s in ("PASS", "WARN", "FAIL")}
    overall = "FAIL" if counts["FAIL"] else "WARN" if counts["WARN"] else "PASS"
    rep = {"id": key, "overall": overall, "counts": counts, "reference": str(ref.relative_to(ROOT)) if ref else None,
           "render_cuts": len(rcuts), "items": items}
    (out_dir / "qc_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = [f"# QC {key}: {overall}", "", f"Reference: {rep['reference']}", "",
          f"PASS {counts['PASS']} / WARN {counts['WARN']} / FAIL {counts['FAIL']}", "",
          "| Section | Check | Result | Detail |", "|---|---|---|---|"]
    md += [f"| {i['section']} | {i['check']} | {i['status']} | {i['detail']} |" for i in items]
    md += ["", "![frames](frames.png)"]
    (out_dir / "qc_report.md").write_text("\n".join(md), encoding="utf-8")
    rq.update(key, qc={"overall": overall, "counts": counts, "report": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
                       "issues": [f"[{i['section']}] {i['check']}: {i['detail']}" for i in items if i["status"] != "PASS"]})
    log(f"[qc] {key}: {overall} {counts}")
    return rep


if __name__ == "__main__":
    r = run(sys.argv[1])
    for i in r["items"]:
        print(f"{i['status']:4}  {i['section']:7} {i['check']}: {i['detail']}")
