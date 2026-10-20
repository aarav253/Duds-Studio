"""Frame-by-frame quality check of a template_edit render against the
reference edit its template was built from.

    python src/quality_check.py edits/<manifest>.json        (render must exist)

Runs automatically after every template_edit build (batch_build). Writes
output/qc/<id>/:
  qc_report.md / qc_report.json   every check with PASS / WARN / FAIL
  cuts_XX-YY.png                  reference vs render, frame by frame, 4 cuts per sheet
                                  (d-2 .. d+11 around each cut, aligned on the cut)
  intro_outro.png                 reference vs render at the intro / card pop / end card
  curves.png                      brightness + sharpness over the whole edit, both videos

Checks
  format     resolution, fps, duration, audio present + aligned with the template song
  timing     each cut's dither start + landing frame vs the template (0 frame tolerance)
  sync       landing time vs the reference's landing time and vs the kick near it
  transition per-frame luma / sharpness of the 9-frame impact vs the reference's
             per-frame mean +- spread (across all its cuts)
  shots      kind (in/out/card), measured zoom vs reference, landing sharpness
             (soft = low-res image), source-image upscale factor
  intro/end  fade-in window, card pop start, caption band present, end card cut + fade
  global     frozen frames, unexpected black frames
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
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from template_builder import LAND_OFFSET, _dominant_motion_ratio, _sim  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan(path, keep_small=True):
    """Per-frame luma / sharpness (360px, same as reference_analyzer) / pixel
    change, plus 540px grays for zoom measurement."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    luma, sharp, pdiff, small = [], [], [], []
    prev = None
    while True:
        ok, b = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        s360 = cv2.resize(g, (360, int(g.shape[0] * 360 / g.shape[1])), interpolation=cv2.INTER_AREA)
        luma.append(float(g.mean()))
        sharp.append(float(cv2.Laplacian(s360, cv2.CV_64F).var()))
        pdiff.append(float(np.abs(s360.astype(np.int16) - prev).mean()) if prev is not None else 0.0)
        prev = s360.astype(np.int16)
        if keep_small:
            small.append(cv2.resize(g, (540, int(g.shape[0] * 540 / g.shape[1])), interpolation=cv2.INTER_AREA))
    cap.release()
    return {"fps": fps, "W": W, "H": H, "n": len(luma), "luma": np.array(luma), "sharp": np.array(sharp),
            "pdiff": np.array(pdiff), "small": small}


def grab_frames(path, idx, width=150):
    need = set(i for i in idx if i >= 0)
    cap = cv2.VideoCapture(str(path))
    got, k = {}, 0
    last = max(need) if need else -1
    while k <= last:
        ok, b = cap.read()
        if not ok:
            break
        if k in need:
            im = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            h = int(im.shape[0] * width / im.shape[1])
            got[k] = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
        k += 1
    return got


def audio_of(path, sr=22050):
    tmp = Path(path).with_suffix(".qc.wav")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-vn",
                    "-ac", "1", "-ar", str(sr), str(tmp)], check=True)
    rate, a = wavfile.read(tmp)
    tmp.unlink()
    return a.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lattice_score(gray540, pitch_px=17.6 * 540 / 1080):
    """How strongly a frame shows the halftone's square dot lattice: FFT energy
    at the lattice frequency vs the surrounding band. Content-independent
    (works on soft video frames and on crisp line art alike)."""
    g = gray540.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 6)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h, w = F.shape
    yy, xx = np.mgrid[0:h, 0:w]
    fy, fx = (yy - h // 2) / h, (xx - w // 2) / w
    fr = np.hypot(fx, fy)
    f0 = 1 / pitch_px
    ring = np.abs(fr - f0) < 0.006
    band = (np.abs(fr - f0) < 0.05) & ~ring
    return float(F[ring].max() / (np.median(F[band]) + 1e-6))


def moire_score(gray540):
    """Strongest isolated periodic peak in mid frequencies (vs median) - a
    blurred frame should have none; interference patterns show as spikes."""
    g = gray540.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 8)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h, w = F.shape
    yy, xx = np.mgrid[0:h, 0:w]
    fy, fx = (yy - h // 2) / h, (xx - w // 2) / w
    fr = np.hypot(fy, fx)
    # skip the axis bands: long straight edges (collars, panel borders) put
    # energy there; interference moire is an off-axis criss-cross pattern
    band = (fr > 0.035) & (fr < 0.2) & (np.abs(fx) > 0.012) & (np.abs(fy) > 0.012)
    # prominence: a periodic pattern makes a sharp isolated spike that stands
    # far above its immediate spectral neighbourhood; noise/blur doesn't
    Fl = np.log1p(F)
    local = cv2.GaussianBlur(Fl, (0, 0), 4)
    prom = Fl - local
    return float(np.exp(prom[band].max()))


def cell_autocorr(gray540, pitch=17.6 * 540 / 1080):
    """Correlation between horizontally neighbouring halftone cells. A real
    (darkened) dot image: ~0.9+. Randomly dropped dots (TV static): ~0.1-0.4."""
    gh, gw = int(gray540.shape[0] / pitch), int(gray540.shape[1] / pitch)
    c = cv2.resize(gray540.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    if c.std() < 2:
        return None          # essentially black - nothing to judge
    return float(np.corrcoef(c[:, :-1].ravel(), c[:, 1:].ravel())[0, 1])


def dither_start(small, guess, lo=-4, hi=6, thresh=40.0):
    """First frame showing the halftone lattice near the expected cut."""
    for k in range(max(1, guess + lo), min(len(small), guess + hi)):
        if lattice_score(small[k]) > thresh:
            return k
    return None


def profile(luma, sharp, d):
    L = luma[d - 2:d + 14] / max(1e-6, luma[d + 12:d + 18].mean())
    S = sharp[d - 2:d + 14] / max(1e-6, sharp[d + 9:d + 12].max())
    return L, S


def kick_envelope(a, sr=22050, hop=128):
    import librosa
    S = np.abs(librosa.stft(a, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[freqs < 150].sum(0)
    k = np.maximum(0, np.diff(low, prepend=low[0]))
    return k / (k.max() + 1e-9), hop / sr


class Report:
    def __init__(self):
        self.items = []

    def add(self, section, name, status, detail):
        self.items.append({"section": section, "check": name, "status": status, "detail": detail})

    def counts(self):
        c = {PASS: 0, WARN: 0, FAIL: 0}
        for it in self.items:
            c[it["status"]] += 1
        return c

    def overall(self):
        c = self.counts()
        return FAIL if c[FAIL] else WARN if c[WARN] else PASS


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def run(manifest_path, render_path=None, out_dir=None, sheets=True, log=print):
    mp = Path(manifest_path)
    mp = mp if mp.is_absolute() else ROOT / mp
    m = json.loads(mp.read_text(encoding="utf-8-sig"))
    t = json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig"))
    render = Path(render_path) if render_path else ROOT / "output" / f"{m['id']}.mp4"
    out = Path(out_dir) if out_dir else ROOT / "output" / "qc" / m["id"]
    out.mkdir(parents=True, exist_ok=True)
    ref_video = ROOT / t["reference"] if not Path(t["reference"]).is_absolute() else Path(t["reference"])
    R = Report()

    log("  [qc] scanning render ...")
    q = scan(render)
    log("  [qc] scanning reference ...")
    r = scan(ref_video)
    fq, fr = q["fps"], r["fps"]
    Dq = [int(round(c * fq)) for c in t["cuts"]]      # expected render cut frames
    Dr = [int(round(c * fr)) for c in t["cuts"]]
    Eq, Er = int(round(t["end"]["card"] * fq)), int(round(t["end"]["card"] * fr))

    # ---------------- format ----------------
    W, H = t["size"]
    R.add("format", "resolution", PASS if (q["W"], q["H"]) == (W, H) else FAIL,
          f"{q['W']}x{q['H']} (template {W}x{H})")
    R.add("format", "fps", PASS if abs(fq - t.get("fps", 60)) < 0.01 else FAIL, f"{fq:.3f}")
    dur = q["n"] / fq
    R.add("format", "duration", PASS if abs(dur - t["duration"]) < 0.05 else FAIL,
          f"{dur:.3f}s (template {t['duration']:.3f}s)")
    try:
        aq = audio_of(render)
        music = audio_of(ROOT / t["music"])
        n = min(len(aq), len(music), 22050 * 8)
        seg_q, seg_m = aq[22050 * 2:22050 * 2 + n // 2], music[22050 * 2:22050 * 2 + n // 2]
        xc = np.correlate(seg_q - seg_q.mean(), seg_m - seg_m.mean(), "full")
        lag = (np.argmax(xc) - (len(seg_m) - 1)) / 22050
        corr = float(xc.max() / (np.linalg.norm(seg_q - seg_q.mean()) * np.linalg.norm(seg_m - seg_m.mean()) + 1e-9))
        R.add("format", "audio = template song, aligned",
              PASS if abs(lag) < 0.012 and corr > 0.9 else FAIL, f"offset {lag * 1000:+.1f} ms, correlation {corr:.3f}")
        R.add("format", "audio length", PASS if abs(len(aq) / 22050 - dur) < 0.1 else WARN,
              f"{len(aq) / 22050:.3f}s")
        peak = float(np.abs(aq).max())
        R.add("format", "audio clipping", PASS if peak < 0.999 else WARN, f"peak {peak:.3f}")
    except Exception as e:  # noqa: BLE001
        aq = None
        R.add("format", "audio", FAIL, f"could not read audio: {e}")

    # ---------------- timing: every cut ----------------
    found = []
    for i, d in enumerate(Dq):
        k = dither_start(q["small"], d)
        found.append(k)
        if k is None:
            R.add("timing", f"cut {i + 1:02d} dither start", FAIL, f"expected frame {d}, no halftone detected")
        else:
            R.add("timing", f"cut {i + 1:02d} dither start", PASS if k == d else FAIL,
                  f"frame {k} (expected {d}, {k - d:+d}f) at {k / fq:.3f}s")

    # ---------------- sync: landings vs reference + kicks ----------------
    ref_land = []
    for d in Dr:
        k = dither_start(r["small"], d)
        ref_land.append(((k if k is not None else d) + LAND_OFFSET) / fr)
    offs = []
    for i, (k, d) in enumerate(zip(found, Dq)):
        land_t = ((k if k is not None else d) + LAND_OFFSET) / fq
        off_ms = (land_t - ref_land[i]) * 1000
        offs.append(off_ms)
        R.add("sync", f"cut {i + 1:02d} landing vs reference", PASS if abs(off_ms) <= 17 else FAIL,
              f"{land_t:.3f}s vs ref {ref_land[i]:.3f}s ({off_ms:+.1f} ms, 1 frame = 16.7 ms)")
    if aq is not None:
        env, hop_t = kick_envelope(aq)
        tt = np.arange(len(env)) * hop_t
        ko = []
        for k, d in zip(found, Dq):
            land_t = ((k if k is not None else d) + LAND_OFFSET) / fq
            m_ = (tt > land_t - 0.2) & (tt < land_t + 0.2)
            if m_.any():
                ko.append((tt[m_][np.argmax(env[m_])] - land_t) * fq)
        R.add("sync", "kick near landing (median offset)", PASS,
              f"{np.median(ko):+.1f} frames (reference edit has the same offsets - timing is copied frame-exact)")

    # ---------------- transitions: per-frame profile ----------------
    ref_L, ref_S = [], []
    for d in Dr[1:-1]:
        k = dither_start(r["small"], d) or d
        L, S = profile(r["luma"], r["sharp"], k)
        ref_L.append(L)
        ref_S.append(S)
    ref_L, ref_S = np.array(ref_L), np.array(ref_S)
    mL, sL = ref_L.mean(0), ref_L.std(0)
    mS = np.median(ref_S, 0)
    labels = {0: "halftone", 1: "halftone", 2: "dark flash", 3: "white burst", 4: "dark dip", 5: "dark dip",
              6: "resolve", 7: "resolve", 8: "resolve", 9: "LANDING", 10: "post-land", 11: "post-land"}
    # Reference ranges that do NOT depend on the panels' content:
    #   halftone lattice strength on d+0/d+1, absolute luma of flash + dips,
    #   moire peaks on the blurred frames
    ref_lat, ref_abs, ref_moire = [], [], []
    for d in Dr[1:-1]:
        k = dither_start(r["small"], d) or d
        ref_lat += [lattice_score(r["small"][k]), lattice_score(r["small"][k + 1])]
        ref_abs.append([r["luma"][k + 2], r["luma"][k + 4], r["luma"][k + 5]])
        ref_moire.append(max(moire_score(r["small"][k + j]) for j in range(3, 9)))
    ref_abs = np.array(ref_abs)
    # dark-flash dot agreement, calibrated per template from its own reference
    # (dense flashes score 0.90-0.99; sparse speck flashes score lower)
    ref_ac = [a for a in (cell_autocorr(r["small"][(dither_start(r["small"], d) or d) + 2]) for d in Dr[1:-1])
              if a is not None]
    flash_ac_min = float(np.percentile(ref_ac, 10)) - 0.25 if len(ref_ac) >= 3 else None
    log(f"  [qc] reference flash dot agreement: {np.round(ref_ac, 2).tolist()}")
    lat_min = float(np.min(ref_lat)) * 0.5
    moire_max = float(np.percentile(ref_moire, 90)) * 1.6
    per_frame_rows = []
    worst = []
    for i, (k, d) in enumerate(zip(found, Dq)):
        if i == 0 or i == len(Dq) - 1:
            continue          # the drop (intro card) and last cut have different neighbours
        k = k or d
        L, S = profile(q["luma"], q["sharp"], k)
        bad = []
        # d+0/d+1: the halftone lattice must be there, at the reference pitch
        for j in (0, 1):
            ls = lattice_score(q["small"][k + j])
            if ls < lat_min:
                bad.append(f"d+{j} halftone: lattice {ls:.0f} (ref >= {lat_min:.0f}) - dots missing or wrong pitch")
        # luma shape (relative) on the frames where it is content-independent enough
        for j in (0, 1, 3, 6, 7, 8):
            idx = j + 2
            tol = max(0.15, 2.5 * sL[idx])
            if abs(L[idx] - mL[idx]) > tol:
                bad.append(f"d+{j} {labels[j]}: luma {L[idx]:.2f} (ref {mL[idx]:.2f}+-{sL[idx]:.2f})")
        # absolute darkness of the flash and the dips
        for col, j in enumerate((2, 4, 5)):
            lo, hi = ref_abs[:, col].min(), ref_abs[:, col].max()
            v = q["luma"][k + j]
            if v > hi + 15 or v < lo - 15:
                bad.append(f"d+{j} {labels[j]}: brightness {v:.0f} (ref {lo:.0f}-{hi:.0f})")
        # structure: burst/dips heavily blurred, resolve sharpening, sharp landing
        land = q["sharp"][k + 9]
        if max(q["sharp"][k + 3:k + 6]) > 0.25 * land:
            bad.append("d+3..d+5 not blurred enough")
        res = q["sharp"][k + 6:k + 10]
        if not (res[0] <= res[1] * 1.15 and res[1] <= res[2] * 1.15 and res[2] < res[3]):
            bad.append(f"resolve not sharpening frame by frame ({', '.join(f'{v:.0f}' for v in res)})")
        if land < 1.4 * q["sharp"][k + 8]:
            bad.append(f"landing not a sharp hit ({land:.0f} vs d+8 {q['sharp'][k + 8]:.0f})")
        # the dark flash must still follow the picture (a darkened dot image),
        # not look like TV static: neighbouring dots should agree
        # (measured: reference 0.90-0.99, random-dot static 0.10-0.37)
        ac = cell_autocorr(q["small"][k + 2])
        if ac is not None and flash_ac_min is not None and ac < flash_ac_min:
            bad.append(f"d+2 dark flash looks like random static, not the picture (dot agreement {ac:.2f}, "
                       f"this template's reference >= {flash_ac_min + 0.25:.2f})")
        # moire / interference on the blurred frames
        mo = max(moire_score(q["small"][k + j]) for j in range(3, 9))
        if mo > moire_max:
            bad.append(f"moire/interference pattern on blurred frames (peak {mo:.0f}, ref <= {moire_max:.0f})")
        per_frame_rows.append((i, L, S))
        worst.append(len(bad))
        R.add("transition", f"cut {i + 1:02d} 9-frame impact", PASS if not bad else WARN if len(bad) <= 2 else FAIL,
              "matches reference on every frame" if not bad else "; ".join(bad))
    if per_frame_rows:
        avgL = np.mean([row[1] for row in per_frame_rows], 0)
        avgS = np.median([row[2] for row in per_frame_rows], 0)
        table = ["| frame | ref luma | ours | ref sharp | ours |", "|---|---|---|---|---|"]
        for j in range(-2, 14):
            idx = j + 2
            table.append(f"| d{j:+d} {labels.get(j, '')} | {mL[idx]:.2f} | {avgL[idx]:.2f} | {mS[idx]:.2f} | {avgS[idx]:.2f} |")
    else:
        table = []

    # ---------------- shots: kind, zoom, sharpness, source resolution ----------------
    kinds = [s["kind"] for s in t["shots"]]
    zin, zout = t["zoom"]["in"], t["zoom"]["out"]
    ref_land_sharp = []
    for d in Dr:
        k = dither_start(r["small"], d) or d
        ref_land_sharp.append(r["sharp"][k + LAND_OFFSET])
    ref_land_med = float(np.median(ref_land_sharp))
    shots = m.get("shots", [])
    for i, kind in enumerate(kinds):
        k = found[i] if found[i] is not None else Dq[i]
        land = k + LAND_OFFSET
        nxt = (found[i + 1] if i + 1 < len(found) and found[i + 1] else (Dq[i + 1] if i + 1 < len(Dq) else Eq))
        last = nxt - 1
        a_, b_ = min(k + 17, last - 4), min(k + 23, last - 1)
        ratio = _dominant_motion_ratio(q["small"][a_], q["small"][b_]) if b_ > a_ else None
        measured_kind = "card" if ratio is not None and ratio < 0.5 else None
        zoom_txt = ""
        status = PASS
        if kind != "card":
            fit = _sim(q["small"][land], q["small"][last])
            if fit is None:
                status, zoom_txt = WARN, "zoom not measurable (too few features)"
            else:
                measured_kind = "in" if fit["scale"] >= 1 else "out"
                sh = t["shots"][i]
                target = 1.0 if kind == "hold" else sh.get("zoom") or (zin if kind == "in" else zout)
                st = dict(t.get("style", {}), **m.get("style_overrides", {}))
                if st.get("zoom_max") and kind != "hold":            # clip edits: zoom capped on purpose
                    cap = st["zoom_max"]
                    target = min(target, cap) if target >= 1 else max(target, 1 / cap)
                if kind == "hold":
                    measured_kind = "hold" if abs(fit["scale"] - 1) < 0.06 else measured_kind
                err = fit["scale"] / target - 1
                zoom_txt = f"zoom {fit['scale']:.3f} (reference {target:.3f}, {err * 100:+.0f}%)"
                if abs(err) > 0.12:
                    status = WARN
        if measured_kind and measured_kind != kind:
            status = FAIL
        land_ratio = q["sharp"][land] / max(1e-6, ref_land_med)
        soft = land_ratio < 0.35
        if soft and status == PASS:
            status = WARN
        src = shots[i] if i < len(shots) else None
        src_path = src if isinstance(src, str) else (src or {}).get("image")
        up = ""
        if src_path:
            try:
                p = Path(src_path)
                p = p if p.is_absolute() else ROOT / p
                w, h = Image.open(p).size
                need = max(W / w, H / h) * (zin if kind != "out" else 1 / zout)
                up = f", source {w}x{h} -> peak upscale {need:.1f}x"
                # checked by eye on real renders: line art at 2.9x still looks
                # clean; softness becomes visible from ~3.3x
                if need > 3.2 and status == PASS:
                    status = WARN
            except OSError:
                up = ", source image unreadable"
        R.add("shots", f"shot {i + 1:02d} ({kind})", status,
              f"{zoom_txt}{'; ' if zoom_txt else ''}detected {measured_kind or '?'} "
              f"(single-motion ratio {ratio if ratio is None else round(ratio, 2)}); landing sharpness "
              f"{land_ratio:.2f}x reference{' - SOFT (low-res image?)' if soft else ''}{up}")

    # duplicate neighbours
    flat = [s if isinstance(s, str) else s.get("image") for s in shots]
    for i in range(1, len(flat)):
        if flat[i] and flat[i] == flat[i - 1]:
            R.add("shots", f"shots {i}/{i + 1} same image back to back", WARN, flat[i])

    # ---------------- intro / end ----------------
    fi0 = int(round(t["intro"]["fade_start"] * fq))
    fi1 = int(round(t["intro"]["fade_end"] * fq))
    dark = q["luma"][:max(1, fi0 - 1)].mean()
    # smoothstep fade: brightness barely moves on its first frames, so accept a
    # few frames of lag before it passes the detection threshold
    rise = next((k for k in range(len(q["luma"])) if q["luma"][k] > dark + 4), None)
    R.add("intro", "fade-in start", PASS if rise is not None and -1 <= rise - fi0 <= 4 else WARN,
          f"frame {rise} (template {fi0})")
    card_k = int(round(t["intro"]["card_start"] * fq))
    hold = slice(fi1 + 1, card_k + 1)          # the intro panel is meant to be static here
    y0, y1 = [int(v * H) for v in (0.559, 0.707)]
    cap = cv2.VideoCapture(str(render))
    cap.set(cv2.CAP_PROP_POS_FRAMES, (hold.start + hold.stop) // 2)
    ok, fb = cap.read()
    cap.release()
    if ok:
        g = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        band = g[y0 + 8:y1 - 8].mean()
        above = g[max(0, y0 - 120):y0 - 10].mean()
        below = g[y1 + 10:y1 + 120].mean()
        R.add("intro", "caption band present", PASS if band < 0.75 * min(above, below) + 10 else WARN,
              f"band luma {band:.0f} vs surroundings {above:.0f}/{below:.0f}")
    # the card starts tiny and blurred, so look at the pop's first few frames
    # and require nothing to change in the frames before it
    jump = q["pdiff"][card_k:card_k + 6].max()
    base = np.median(q["pdiff"][hold.start:card_k - 1])
    early = q["pdiff"][card_k - 6:card_k - 1].max()
    R.add("intro", "card pop starts on time",
          PASS if jump > base + 1.0 and early < base + 0.5 else WARN,
          f"change {jump:.1f} in frames {card_k}-{card_k + 5}, before it {early:.1f} (hold {base:.1f})")
    frozen = [k for k in range(fi1 + 1, len(q["pdiff"])) if not (hold.start <= k < hold.stop)
              and k < Eq and q["pdiff"][k] < 0.15]
    R.add("global", "frozen frames (outside intro hold)", PASS if not frozen else WARN,
          f"{len(frozen)} frames" + (f": {frozen[:12]}" if frozen else ""))
    blk = [k for k in range(fi1 + 1, Eq) if q["luma"][k] < 12 and (k - Dq[0]) % 1 == 0
           and not any(0 <= k - d <= 5 for d in Dq)]
    R.add("global", "unexpected black frames", PASS if not blk else WARN, f"{len(blk)}" + (f": {blk[:12]}" if blk else ""))
    # a hard cut = a big picture change on exactly that frame (brightness alone
    # can barely move when the last shot is already a white panel)
    base_e = np.median(q["pdiff"][Eq - 8:Eq - 1])
    R.add("end", "end card hard cut", PASS if q["luma"][Eq] > 200 and q["pdiff"][Eq] > base_e + 5 else WARN,
          f"frame {Eq}: luma {q['luma'][Eq - 1]:.0f} -> {q['luma'][Eq]:.0f}, change {q['pdiff'][Eq]:.1f} "
          f"(shot motion {base_e:.1f})")
    fe = int(round(t["end"]["fade_end"] * fq))
    R.add("end", "fade to dark", PASS if q["luma"][min(fe + 3, q["n"] - 1)] < 40 else WARN,
          f"luma at fade end {q['luma'][min(fe + 3, q['n'] - 1)]:.0f}")

    # ---------------- sheets ----------------
    if sheets:
        log("  [qc] building frame-by-frame sheets ...")
        tw = 150
        offsets = list(range(-2, 12))
        want_r, want_q = set(), set()
        for i in range(len(Dq)):
            dr = dither_start(r["small"], Dr[i]) or Dr[i]
            dq = found[i] or Dq[i]
            want_r |= {dr + o for o in offsets}
            want_q |= {dq + o for o in offsets}
        extra_t = [0.1, t["intro"]["fade_end"], 5.0, t["intro"]["card_start"] + 0.05,
                   (t["intro"]["card_start"] + t["cuts"][0]) / 2, t["cuts"][0] - 0.02,
                   t["end"]["card"] + 0.02, (t["end"]["fade_start"] + t["end"]["fade_end"]) / 2, t["end"]["fade_end"]]
        want_r |= {int(round(x * fr)) for x in extra_t}
        want_q |= {int(round(x * fq)) for x in extra_t}
        GR = grab_frames(ref_video, sorted(want_r), tw)
        GQ = grab_frames(render, sorted(want_q), tw)
        font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 11)

        def tile(im, lab, col):
            p = Image.fromarray(im) if im is not None else Image.new("RGB", (tw, int(tw * H / W)), (60, 0, 0))
            d = ImageDraw.Draw(p)
            d.rectangle([0, 0, d.textlength(lab, font=font) + 6, 13], fill=col)
            d.text((3, 1), lab, font=font, fill="white")
            return p

        th = int(tw * H / W)
        names = {-2: "pre", -1: "pre", **labels}
        for start in range(0, len(Dq), 4):
            block = range(start, min(start + 4, len(Dq)))
            sheet = Image.new("RGB", (tw * len(offsets), th * 2 * len(block) + 18 * len(block)), (25, 25, 25))
            dd = ImageDraw.Draw(sheet)
            y = 0
            for i in block:
                dr = dither_start(r["small"], Dr[i]) or Dr[i]
                dq = found[i] or Dq[i]
                dd.text((4, y + 3), f"CUT {i + 1}  ref frame {dr} / render frame {dq}  ({t['cuts'][i]:.3f}s)  "
                        f"shot kind: {kinds[i]}", font=font, fill=(255, 220, 90))
                y += 18
                for c, o in enumerate(offsets):
                    sheet.paste(tile(GR.get(dr + o), f"REF d{o:+d} {names.get(o, '')}", (170, 0, 0)), (c * tw, y))
                    sheet.paste(tile(GQ.get(dq + o), f"OURS d{o:+d}", (0, 90, 170)), (c * tw, y + th))
                y += 2 * th
            sheet.save(out / f"cuts_{start + 1:02d}-{block[-1] + 1:02d}.png")
        io = Image.new("RGB", (tw * len(extra_t), th * 2), (25, 25, 25))
        for c, x in enumerate(extra_t):
            io.paste(tile(GR.get(int(round(x * fr))), f"REF {x:.2f}s", (170, 0, 0)), (c * tw, 0))
            io.paste(tile(GQ.get(int(round(x * fq))), f"OURS {x:.2f}s", (0, 90, 170)), (c * tw, th))
        io.save(out / "intro_outro.png")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
        ax[0].plot(np.arange(r["n"]) / fr, r["luma"], lw=0.8, label="reference", color="tab:red")
        ax[0].plot(np.arange(q["n"]) / fq, q["luma"], lw=0.8, label="render", color="tab:blue", alpha=0.8)
        ax[0].set_ylabel("brightness")
        ax[0].legend(loc="upper left")
        ax[1].semilogy(np.arange(r["n"]) / fr, r["sharp"] + 1, lw=0.8, color="tab:red")
        ax[1].semilogy(np.arange(q["n"]) / fq, q["sharp"] + 1, lw=0.8, color="tab:blue", alpha=0.8)
        ax[1].set_ylabel("sharpness (log)")
        for a in ax:
            for c in t["cuts"]:
                a.axvline(c, color="gray", lw=0.5, ls=":")
        ax[1].set_xlabel("seconds (dotted = template cuts)")
        fig.tight_layout()
        fig.savefig(out / "curves.png", dpi=100)
        plt.close(fig)

    # ---------------- write report ----------------
    c = R.counts()
    lines = [f"# QC report: {m['id']}", "",
             f"Template **{t['name']}** vs reference `{Path(t['reference']).name}`", "",
             f"**Overall: {R.overall()}** ({c[PASS]} pass, {c[WARN]} warn, {c[FAIL]} fail)", ""]
    if table:
        lines += ["## Average transition profile (all cuts)", ""] + table + [""]
    sec = None
    for it in R.items:
        if it["section"] != sec:
            sec = it["section"]
            lines += ["", f"## {sec}", "", "| check | status | detail |", "|---|---|---|"]
        lines.append(f"| {it['check']} | {it['status']} | {it['detail']} |")
    (out / "qc_report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"overall": R.overall(), "counts": c, "items": R.items, "report": str(out.relative_to(ROOT)).replace("\\", "/")}
    (out / "qc_report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"  [qc] {R.overall()}: {c[PASS]} pass, {c[WARN]} warn, {c[FAIL]} fail -> {out}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for mpath in sys.argv[1:]:
        s = run(mpath)
        issues = [f"[{it['section']}] {it['check']}: {it['detail']}" for it in s["items"] if it["status"] != PASS]
        for line in issues:
            print("   ", line)
        # keep the dashboard's QC badge in sync with manual runs
        import review_queue as rq
        mid = json.loads(Path(mpath).read_text(encoding="utf-8-sig"))["id"]
        if rq.get(mid):
            rq.update(mid, qc={"overall": s["overall"], "counts": s["counts"], "report": s["report"],
                               "issues": issues})
