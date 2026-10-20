"""Section edit engine for "character roster" references (template 10, the
Seven Deadly Sins edit): the video is a run of sections, one per character,
each with a fixed three-line title block (name / sin / nickname), a colour
grade and a beat-cut run of clips.

Everything timing/look related comes from the template (measured off the
reference, frames in reference 30fps units):
  sections[].cuts/end   beat cuts           sections[].grade  3-point gradient map
  sections[].zoom       push-in per second   sections[].text   line boxes, colours
  white_in / white_out  white dips           subtitles         word-by-word captions
The manifest picks the clips per section and may rename the text lines, and
describes the audio: the reference's instrumental + voice stems with voicelines
muted, moved or replaced.

    python src/sins_edit.py edits/<id>.json [--preview]
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SUB_FONT = r"C:\Windows\Fonts\bahnschrift.ttf"


def _p(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- clips
class Clip:
    """Frames of a scene clip, letterbox bars trimmed, cover-fitted to W x H.
    Fractional frame positions blend the two neighbours (smooth slow-mo)."""

    def __init__(self, path, W, H):
        self.W, self.H, self.pos, self.cache = W, H, -1, {}
        if Path(path).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):   # a still panel: one frame, no bar trim
            self.img = cv2.imread(str(_p(path)))
            self.n, self.box, self.cap, self.still = 1, None, None, True
            return
        self.img, self.still = None, False
        self.cap = cv2.VideoCapture(str(_p(path)))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.box = self._bars()

    def _bars(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.n // 2)
        ok, b = self.cap.read()
        self.pos = -2
        if not ok:
            return None
        g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape

        def run(v):
            k = 0
            while k < len(v) and v[k] < 14:
                k += 1
            return k
        rows, cols = g.mean(1), g.mean(0)
        t, btm, l, r = run(rows), run(rows[::-1]), run(cols), run(cols[::-1])
        ok_v = 0.015 * h < t + btm < 0.35 * h
        ok_h = 0.015 * w < l + r < 0.35 * w
        return (t if ok_v else 0, h - (btm if ok_v else 0), l if ok_h else 0, w - (r if ok_h else 0))

    def _raw(self, i):
        i = int(min(max(i, 0), self.n - 1))
        if i in self.cache:
            return self.cache[i]
        if self.still:
            ok, b = self.img is not None, self.img
        else:
            if i != self.pos + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, b = self.cap.read()
        self.pos = i
        if not ok:
            b = np.zeros((self.H, self.W, 3), np.uint8)
        elif self.box:
            y0, y1, x0, x1 = self.box
            b = b[y0:y1, x0:x1]
        h, w = b.shape[:2]
        s = max(self.W / w, self.H / h)
        nw, nh = max(self.W, round(w * s)), max(self.H, round(h * s))
        b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        x, y = (nw - self.W) // 2, (nh - self.H) // 2
        if self.still:                                 # tall art: keep the head (upper part) in frame
            y = int((nh - self.H) * 0.3)
        f = cv2.cvtColor(b[y:y + self.H, x:x + self.W], cv2.COLOR_BGR2RGB)
        if len(self.cache) > 6:
            self.cache.pop(next(iter(self.cache)))
        self.cache[i] = f
        return f

    def frame(self, t):
        a = math.floor(t)
        u = t - a
        if u < 0.08 or a + 1 >= self.n:
            return self._raw(a)
        if u > 0.92:
            return self._raw(a + 1)
        return cv2.addWeighted(self._raw(a), 1 - u, self._raw(a + 1), u, 0)


# ---------------------------------------------------------------- grade
def grade_luts(g, src_p):
    """(levels LUT, gradient-map LUT 256x3, amount). Levels move the clip's
    own p10/p50/p90 luminance toward the reference section's; the gradient
    map recolours by luminance with the section's shadow/mid/highlight."""
    tp = g.get("p", [10, 60, 220])
    xs = [0, src_p[0], src_p[1], src_p[2], 255]
    ys = [0, tp[0], tp[1], tp[2], 255]
    xs = np.maximum.accumulate(np.array(xs, np.float32) + np.arange(5) * 1e-3)
    lv = np.interp(np.arange(256), xs, ys)
    lv = np.clip(0.4 * np.arange(256) + 0.6 * lv, 0, 255).astype(np.uint8)  # don't over-edit
    amt = g.get("amount", 0.0)
    if amt <= 0:
        return None, None, 0.0
    lo, mid, hi = (np.array(g[k], np.float32) for k in ("lo", "mid", "hi"))
    ks = [0, tp[0], tp[1], tp[2], 255]
    cols = [lo * 0.35, lo, mid, hi, np.maximum(hi, 245)]
    gm = np.stack([np.interp(np.arange(256), ks, [c[ch] for c in cols]) for ch in range(3)], 1)
    return lv, gm.astype(np.float32), amt


def apply_grade(img, luts):
    lv, gm, amt = luts
    if gm is None:
        return img
    img = cv2.LUT(img, lv)
    y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mapped = gm[y]
    return np.clip(img.astype(np.float32) * (1 - amt) + mapped * amt, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- text
def _font(path, size, variation=None):
    f = ImageFont.truetype(str(_p(path)) if not Path(path).is_absolute() else path, size)
    if variation:
        try:
            f.set_variation_by_name(variation)
        except Exception:  # noqa: BLE001 - static font
            pass
    return f


def _draw_tracked(d, x, y, text, f, track, fill):
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill, anchor="ls")
        x += d.textlength(ch, font=f) + track


def _tracked_width(d, text, f, track):
    return sum(d.textlength(ch, font=f) for ch in text) + track * (len(text) - 1)


class TextBlock:
    """One section's title block, rendered once as a premultiplied layer
    (alpha, rgb*alpha) per line so lines can fade separately."""

    def __init__(self, W, H, y_off, fonts, lines, overrides):
        self.layers = []
        for ln in lines:
            role = ln["role"]
            text = overrides.get(ln.get("key", ""), ln["text"]) if overrides else ln["text"]
            fp = {"name": fonts["name"], "sin": fonts["sin"], "title": fonts["title"]}[role]
            var = "Black" if role == "sin" else None
            x0, y0, x1, y1 = ln["box"]
            cap_h = (y1 - y0) - 5                       # measured box includes the drop shadow
            probe = Image.new("L", (10, 10))
            d = ImageDraw.Draw(probe)
            size = 40
            for _ in range(3):                          # size so cap height matches the reference
                f = _font(fp, size, var)
                bb = f.getbbox("HE", anchor="ls")
                size = max(8, int(round(size * cap_h / max(1, -bb[1]))))
            f = _font(fp, size, var)
            # the reference fonts are wider than ours: keep a natural letter
            # gap and stretch the line horizontally to the measured width
            # (the same stretch is used for replacement text)
            track = {"sin": 0.06, "name": 0.12, "title": 0.02}[role] * size
            bold = max(1, round(size * 0.035)) if role == "name" else 0   # Michroma has no bold weight
            nat = _tracked_width(d, ln["text"], f, track) + 2 * bold
            sx = min(2.0, max(0.8, (x1 - x0 - 4) / nat))
            w_nat = _tracked_width(d, text, f, track) + 2 * bold
            sx = min(sx, 0.92 * W / w_nat)
            pad = bold + 4
            loc = Image.new("L", (int(w_nat) + 2 * pad, H), 0)
            base = y1 - 5 + y_off
            dl = ImageDraw.Draw(loc)
            x = pad
            for ch in text:
                dl.text((x, base), ch, font=f, fill=255, anchor="ls", stroke_width=bold, stroke_fill=255)
                x += dl.textlength(ch, font=f) + track
            loc = loc.resize((max(1, int(loc.width * sx)), H), Image.LANCZOS)
            cx = W / 2                                  # the reference centres every line; measured boxes lean with the shadow
            a = Image.new("L", (W, H), 0)
            a.paste(loc, (int(round(cx - loc.width / 2)), 0))
            A = np.asarray(a).astype(np.float32) / 255
            sh = np.asarray(a.filter(ImageFilter.GaussianBlur(1.6))).astype(np.float32) / 255
            sh = np.roll(sh, (4, 3), (0, 1))
            col = np.array(ln["color"], np.float32)
            glow = None
            if role in ("sin", "title"):
                glow = np.asarray(a.filter(ImageFilter.GaussianBlur(9))).astype(np.float32) / 255 * 0.45
            self.layers.append({"A": A, "sh": sh * 0.78, "col": col, "glow": glow, "role": role})

    def composite(self, img, alphas):
        out = img
        for L, al in zip(self.layers, alphas):
            if al <= 0.003:
                continue
            out = out * (1 - L["sh"][..., None] * al)
            if L["glow"] is not None:
                g = L["glow"][..., None] * al
                out = out * (1 - g) + L["col"] * g
            A = L["A"][..., None] * al
            out = out * (1 - A) + L["col"] * A
        return out


class Subtitles:
    def __init__(self, W, H, y_off, subs, box):
        x0, y0, x1, y1 = box
        self.items = []
        cache = {}
        for s in subs:
            if s["text"] not in cache:
                f = _font(SUB_FONT, int((y1 - y0) * 1.05), "Bold Condensed")
                a = Image.new("L", (W, H), 0)
                ImageDraw.Draw(a).text((W / 2, (y0 + y1) / 2 + y_off), s["text"], font=f, fill=255, anchor="mm")
                A = np.asarray(a).astype(np.float32) / 255
                glow = np.asarray(a.filter(ImageFilter.GaussianBlur(7))).astype(np.float32) / 255 * 0.55
                sh = np.roll(np.asarray(a.filter(ImageFilter.GaussianBlur(2))).astype(np.float32) / 255, (3, 2), (0, 1)) * 0.6
                cache[s["text"]] = (A, glow, sh)
            self.items.append((s["a"], s["b"], cache[s["text"]]))

    def composite(self, img, r):
        for a, b, (A, glow, sh) in self.items:
            if a <= r < b:
                al = min(1.0, (b - r) / 5.0)            # pop in, fade out over the last ~5 frames
                img = img * (1 - sh[..., None] * al)
                img = img * (1 - glow[..., None] * al) + 255 * glow[..., None] * al
                img = img * (1 - A[..., None] * al) + 252 * A[..., None] * al
        return img


# ---------------------------------------------------------------- audio
def _load_audio(path, sr=44100):
    p = _p(path)
    if p.suffix.lower() != ".wav":
        tmp = ROOT / "output" / f"_tmp_{p.stem[:20]}.wav"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(p), "-ac", "2", "-ar", str(sr), str(tmp)], check=True)
        y, s = sf.read(str(tmp), always_2d=True)
        tmp.unlink()
    else:
        y, s = sf.read(str(p), always_2d=True)
    assert s == sr, f"{p} is {s} Hz"
    if y.shape[1] == 1:
        y = np.repeat(y, 2, 1)
    return y.astype(np.float32)



def clean_loops(mix, sr, beat, phase, per_loop=8, hb_db=5.0, low_db=5.0, protect=(), passes=2):
    """Remove one-off sound effects (whooshes, hits) and ducked spots from a
    loop-based song: every beat is compared with the same beat of every other
    loop; a beat whose high band is far louder than usual (an effect) or
    whose level is far below usual (music ducked under a voiceline) is
    replaced by the most typical copy of that beat from another loop.
    Beats overlapping `protect` ranges (seconds) are left alone."""
    import librosa
    n = len(mix)
    starts = np.arange(phase, n / sr - beat, beat)
    if phase > 0.02:
        starts = np.concatenate([[0.0], starts])  # partial first beat stays untouched below
    feats = []
    for s0 in starts:
        a, b = int(s0 * sr), int((s0 + beat) * sr)
        x = mix[a:b].mean(1)
        S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512)) ** 2
        f = librosa.fft_frequencies(sr=sr, n_fft=2048)
        hb = 10 * np.log10(S[f > 3000].sum(0).mean() + 1e-9)
        lv = 10 * np.log10((x ** 2).mean() + 1e-9)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=sr, n_mels=40)).mean(1)
        feats.append((hb, lv, mel))
    ph = np.array([round((s0 - phase) / beat) % per_loop for s0 in starts])
    hb = np.array([f[0] for f in feats]); lv = np.array([f[1] for f in feats]); mel = np.stack([f[2] for f in feats])
    full = np.array([abs((s1 - s0) - beat) < 1e-6 for s0, s1 in zip(starts, list(starts[1:]) + [starts[-1] + beat])])
    full[0] = starts[0] >= phase - 1e-6
    bad = np.zeros(len(starts), bool)
    for p in range(per_loop):
        idx = np.where((ph == p) & full)[0]
        if len(idx) < 3:
            continue
        mh, ml = np.median(hb[idx]), np.median(lv[idx])
        bad[idx] = (hb[idx] - mh > hb_db) | (ml - lv[idx] > low_db)
    for a0, a1 in protect:
        bad &= ~((starts < a1) & (starts + beat > a0))
    out = mix.copy()
    xf = int(0.012 * sr)
    replaced = []
    for j in np.where(bad)[0]:
        p = ph[j]
        cand = [i for i in np.where((ph == p) & full & ~bad)[0]]
        if not cand:
            continue
        med = np.median(mel[cand], 0)
        src = min(cand, key=lambda i: np.abs(mel[i] - med).mean())
        shift = int(round((starts[src] - starts[j]) * sr))
        a, b = int(starts[j] * sr) - xf, int((starts[j] + beat) * sr) - xf   # cut just before the transients
        a, b = max(0, a), min(n, b)
        if a + shift < 0 or b + shift + xf > n:
            continue
        seg = mix[a + shift:b + shift + xf].copy()
        ramp = np.linspace(0, 1, xf)[:, None]
        cur = out[a:b + xf]
        seg[:xf] = cur[:xf] * (1 - ramp) + seg[:xf] * ramp
        seg[-xf:] = seg[-xf:] * (1 - ramp) + cur[-xf:] * ramp
        out[a:b + xf] = seg
        replaced.append(round(float(starts[j]), 2))
    print(f"  [audio] replaced {len(replaced)} beats: {replaced}")
    return out


def level(mix, sr, beat, max_gain_db=6.0, smooth=2.0, protect_head=0.0):
    """Gentle gain riding to a constant loudness (beat-length windows so the
    kick doesn't pump it), limited to +-max_gain_db."""
    from scipy.ndimage import uniform_filter1d
    w = int(beat * sr)
    e = uniform_filter1d((mix ** 2).mean(1), w)
    rms = np.sqrt(e + 1e-10)
    target = np.median(rms[int(protect_head * sr):])
    g = np.clip(target / rms, 10 ** (-max_gain_db / 20), 10 ** (max_gain_db / 20))
    g = np.exp(uniform_filter1d(np.log(g), int(smooth * sr)))
    return mix * g[:, None]


def mix_audio(t, plan, out, dur):
    """The reference's own mix, with voicelines (or sound effects) removed by
    subtracting the separated voice stem only inside the "mute" ranges, and
    voicelines added back where wanted. Everywhere else the audio is the
    untouched original - rebuilding from stems everywhere adds separation
    artefacts (harsh high-end blocks) to the whole song."""
    sr = 44100
    base = _load_audio(t["audio"][plan.get("base", "mix")])   # "song" = the clean track, "mix" = the reference audio
    voc = _load_audio(t["audio"]["vocals"])
    n = int(dur * sr)
    mix, voc0 = base[:n].copy(), voc[:n]
    if len(mix) < n:
        mix = np.concatenate([mix, np.zeros((n - len(mix), 2), np.float32)])
    rm = np.zeros(len(voc0), np.float32)
    ramp = int(0.03 * sr)
    for a, b in plan.get("mute", []):
        ia, ib = int(a * sr), min(len(rm), int(b * sr))
        rm[ia:ib] = 1
        k = ia - max(0, ia - ramp)
        rm[ia - k:ia] = np.maximum(rm[ia - k:ia], np.linspace(0, 1, k))
        k = len(rm[ib:ib + ramp])
        rm[ib:ib + k] = np.maximum(rm[ib:ib + k], np.linspace(1, 0, k))
    mix -= voc0[:len(mix)] * rm[:len(mix), None]
    # the voice stem also carries some of the music under the line, so the
    # subtraction leaves holes; fill them from the same point of a voice-free
    # loop (the beat loops every `loop` seconds) wherever the residual falls
    # below that loop's level
    loop = t["audio"].get("loop")
    fills = plan.get("fill_from", [])
    strength = plan.get("fill_strength", [1.0] * len(fills))
    for (a, b), donor, fs in zip(plan.get("mute", []), fills, strength):
        if donor is None or not loop:
            continue
        ia, ib = max(0, int((a - 0.1) * sr)), min(len(mix), int((b + 0.1) * sr))
        k = round((a - donor) / loop)
        shift = int(round(k * loop * sr))
        d = base[ia - shift:ib - shift]
        if len(d) != ib - ia:
            continue
        import librosa
        w = rm[ia:ib]
        for ch in range(2):
            R = librosa.stft(mix[ia:ib, ch], n_fft=2048, hop_length=512)
            D = librosa.stft(d[:, ch], n_fft=2048, hop_length=512)
            # the donor loop may be a louder part of the song: per-band gain
            # from the frames of this range where nobody is talking
            V = np.abs(librosa.stft(voc0[ia:ib, ch], n_fft=2048, hop_length=512)).sum(0)
            quiet = V < 0.1 * V.max()
            if quiet.sum() >= 8:
                g = np.median(np.abs(R[:, quiet]), 1) / (np.median(np.abs(D[:, quiet]), 1) + 1e-6)
                D = D * np.clip(g, 0.3, 1.2)[:, None]
            ww = np.interp(np.arange(R.shape[1]) * 512, np.arange(len(w)), w)[None]
            fill = np.clip((np.abs(D) - np.abs(R)) / (np.abs(D) + 1e-6), 0, 1) * ww * fs
            mix[ia:ib, ch] = librosa.istft(R + D * fill, hop_length=512, length=ib - ia)
    cl = plan.get("clean_loops")
    if cl and plan.get("level"):                 # level first so quiet stretches don't hide effects
        mix = level(mix, sr, plan["level"]["beat"], plan["level"].get("max_gain_db", 6.0))
    if cl:
        for _ in range(cl.get("passes", 2)):     # second pass catches beats that hid behind a louder one
            mix = clean_loops(mix, sr, cl["beat"], cl["phase"], hb_db=cl.get("hb_db", 5.0),
                              low_db=cl.get("low_db", 5.0), protect=cl.get("protect", []))
    if plan.get("level"):
        mix = level(mix, sr, plan["level"]["beat"], plan["level"].get("max_gain_db", 6.0))
    # voicelines go on their own track so the music can be ducked under them
    vt = np.zeros_like(mix)
    srcs = {"vocals": voc}
    for ins in plan.get("inserts", []):
        if "file" in ins:
            p = _p(ins["file"])
            if not p.exists() or p.stat().st_size == 0:
                print(f"  [audio] skipping missing/empty {p.name}")
                continue
            seg = _load_audio(p)
        else:
            key = ins.get("src", "vocals")
            if key not in srcs:
                srcs[key] = _load_audio(t["audio"][key])
            seg = srcs[key][int(ins["from"] * sr):int(ins["to"] * sr)].copy()
        if "match_rms" in ins:                          # level-match to the voiceline it replaces
            a, b = ins["match_rms"]
            ref = voc[int(a * sr):int(b * sr)]
            r_ref = np.sqrt(np.mean(ref[np.abs(ref).max(1) > 0.02] ** 2) + 1e-9)
            r_seg = np.sqrt(np.mean(seg[np.abs(seg).max(1) > 0.02] ** 2) + 1e-9)
            seg *= r_ref / r_seg
        seg *= ins.get("gain", 1.0)
        f = min(ramp, len(seg) // 4)
        seg[:f] *= np.linspace(0, 1, f)[:, None]
        seg[-f:] *= np.linspace(1, 0, f)[:, None]
        i = int(ins["at"] * sr)
        j = min(len(mix), i + len(seg))
        vt[i:j] += seg[:j - i]
    duck = plan.get("duck_db", 0.0)
    if duck and np.abs(vt).max() > 0:
        from scipy.ndimage import maximum_filter1d as _mf, uniform_filter1d as _uf
        act = (_uf(np.abs(vt).max(1), int(0.03 * sr)) > 0.01).astype(np.float32)
        act = _mf(act, int(0.25 * sr))                  # hold through short gaps between words
        act = _uf(act, int(0.12 * sr))                  # ~120 ms attack / release
        mix *= (1 - act * (1 - 10 ** (-duck / 20)))[:, None]
    mix += vt
    # look-ahead peak limiter: keep the edit's peaks at the reference's level
    # (inserted lines stacked on the beat can exceed it); untouched audio
    # below the ceiling passes through unchanged
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    ceil = plan.get("ceiling", float(np.abs(base).max()) + 0.03)
    env = maximum_filter1d(np.abs(mix).max(1), int(0.012 * sr))
    g = np.minimum(1.0, ceil / np.maximum(env, 1e-6))
    g = np.minimum(g, uniform_filter1d(g, int(0.02 * sr)))
    mix *= g[:, None]
    sf.write(str(out), mix, sr)
    return out


# ---------------------------------------------------------------- edit
class SinsEdit:
    def __init__(self, m, preview=False):
        self.m = m
        self.t = json.loads(_p(m["template"]).read_text(encoding="utf-8"))
        t = self.t
        W0, H0 = m.get("size", t["size"])           # output frame (e.g. 1080x1920 for full-screen Shorts)
        self.k = 0.5 if preview else 1.0
        self.W, self.H = int(W0 * self.k) // 2 * 2, int(H0 * self.k) // 2 * 2
        # "cover": clips fill the frame (crops the sides of 4:5 clips in 9:16);
        # "blur":  the whole clip sits centred on a blurred copy of itself
        self.fill = m.get("fill", "cover")
        cw, ch = t["size"]
        self.CH = int(ch * self.k * self.W / (cw * self.k)) // 2 * 2 if self.fill == "blur" else self.H
        self.fps = t["fps"]
        self.rf = t["ref_fps"]
        self.n = int(t["length"] * self.fps / self.rf)
        y_off = (H0 - t["ref_size"][1]) / 2
        fonts = t["fonts"]
        msec = m["sections"]
        self.secs = []
        for s in t["sections"]:
            ms = msec.get(s["id"], {})
            clips = ms.get("clips", [])
            shots = []
            bounds = s["cuts"] + [s["end"]]
            for i in range(len(s["cuts"])):
                c = clips[i % len(clips)] if clips else None
                if isinstance(c, str):
                    c = {"clip": c}
                shots.append({"a": bounds[i], "b": bounds[i + 1], "spec": c, "clip": None, "luts": None})
            tb = None
            if s.get("text"):
                cols = ms.get("colors", [])               # manifest may recolour lines (null = keep)
                lines = [dict(ln, key=f"{s['id']}.{j}", **({"color": cols[j]} if j < len(cols) and cols[j] else {}))
                         for j, ln in enumerate(s["text"])]
                ov = {f"{s['id']}.{j}": v for j, v in enumerate(ms.get("text", [])) if v}
                tb = self._scaled_text(lines, ov, fonts, y_off)
            subs = Subtitles(self.W, self.H, y_off * self.k, [dict(x) for x in s["subtitles"]],
                             [v * self.k for v in s["subtitle_box"]]) if s.get("subtitles") else None
            self.secs.append({"s": s, "shots": shots, "text": tb, "subs": subs})
        self.wm = self._wm(m.get("watermark", "DUDS"))

    def _scaled_text(self, lines, ov, fonts, y_off):
        k = self.k
        if k != 1:
            lines = [dict(ln, box=[v * k for v in ln["box"]]) for ln in lines]
        return TextBlock(self.W, self.H, y_off * k, fonts, lines, ov)

    def _wm(self, text):
        if not text:
            return None
        lay = Image.new("L", (self.W, self.H), 0)
        f = ImageFont.truetype(r"C:\Windows\Fonts\bahnschrift.ttf", int(self.H * 0.016))
        ImageDraw.Draw(lay).text((self.W / 2, self.H * 0.962), " ".join(text), font=f, fill=150, anchor="mm")
        return np.asarray(lay).astype(np.float32)[..., None] / 255

    def _shot_img(self, sec, sh, r):
        spec = sh["spec"]
        if spec is None:
            return np.zeros((self.H, self.W, 3), np.uint8)
        if sh["clip"] is None:
            sh["clip"] = Clip(spec["clip"], self.W, self.CH)
        c = sh["clip"]
        start = spec.get("start", 6)
        need = (sh["b"] - sh["a"]) * self.fps / self.rf
        speed = spec.get("speed") or min(1.0, max(0.35, (c.n - start - 2) / max(1, need)))
        img = c.frame(start + (r - sh["a"]) * self.fps / self.rf * speed)
        if sh["luts"] is None:
            g = cv2.cvtColor(c.frame(start + need * speed * 0.4), cv2.COLOR_RGB2GRAY)
            sh["luts"] = grade_luts(sec["s"].get("grade", {}), np.percentile(g, [10, 50, 90]))
        img = apply_grade(img, sh["luts"])
        z = sec["s"].get("zoom", 0.0) + spec.get("zoom", 0.0)
        if z:
            s = math.exp(z * (r - sh["a"]) / self.rf)
            M = cv2.getRotationMatrix2D((self.W / 2, self.CH / 2), 0, s)
            img = cv2.warpAffine(img, M, (self.W, self.CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if self.CH != self.H:
            img = self._blur_fill(img)
        return img

    def _blur_fill(self, img):
        """Centre the full clip on a blurred, darkened, enlarged copy of itself."""
        small = cv2.resize(img, (self.W // 8, self.CH // 8), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        s = self.H / self.CH
        bg = cv2.resize(small, (int(self.W * s), self.H), interpolation=cv2.INTER_LINEAR)
        x = (bg.shape[1] - self.W) // 2
        bg = bg[:, x:x + self.W].astype(np.float32) * 0.5
        y0 = (self.H - self.CH) // 2
        if not hasattr(self, "_feather"):
            f = max(4, self.CH // 40)
            ramp = np.ones(self.CH, np.float32)
            ramp[:f] = np.linspace(0, 1, f)
            ramp[-f:] = np.linspace(1, 0, f)
            self._feather = ramp[:, None, None]
        bg[y0:y0 + self.CH] = bg[y0:y0 + self.CH] * (1 - self._feather) + img.astype(np.float32) * self._feather
        return bg.astype(np.uint8)

    def frame(self, kf):
        r = kf * self.rf / self.fps                      # time in reference frames
        idx = max(i for i, sec in enumerate(self.secs) if sec["s"]["cuts"][0] <= r)
        sec = self.secs[idx]
        s = sec["s"]
        sh = [x for x in sec["shots"] if x["a"] <= r][-1]
        img = self._shot_img(sec, sh, r).astype(np.float32)
        # white dips: fade out of this section / in from white at its start
        white, text_white = 0.0, False
        wi = s.get("white_in", 0)
        if wi and r < s["cuts"][0] + wi:
            white = 1 - (r - s["cuts"][0]) / wi
            text_white = True
        nxt = self.secs[idx + 1]["s"] if idx + 1 < len(self.secs) else None
        wo = s.get("white_out", 0)
        if wo and r > s["end"] - wo:
            white = max(white, (r - (s["end"] - wo)) / wo)
        if white > 0:
            w = min(1.0, white) ** 0.8
            img = img * (1 - w) + 250 * w
        if sec["text"] is not None:
            al = []
            for L in sec["text"].layers:
                if text_white and L["role"] not in ("sin", "title"):
                    al.append(max(0.0, 1 - white * 1.6))   # only the sin word on the white frame
                else:
                    al.append(1.0)
            img = sec["text"].composite(img, al)
        if sec["subs"] is not None:
            img = sec["subs"].composite(img, r)
        if self.wm is not None:
            img = img * (1 - self.wm * 0.7) + 235 * self.wm * 0.7
        return np.clip(img, 0, 255).astype(np.uint8)

    def render(self, out):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".video.mp4")
        wav = out.with_suffix(".mix.wav")
        mix_audio(self.t, self.m.get("audio_plan", {}), wav, self.n / self.fps)
        wr = imageio_ffmpeg.write_frames(str(tmp), (self.W, self.H), fps=self.fps, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "16", "-preset", "medium"], macro_block_size=2)
        wr.send(None)
        for k in range(self.n):
            wr.send(np.ascontiguousarray(self.frame(k)))
            if k % 300 == 0:
                print(f"  frame {k}/{self.n}", flush=True)
        wr.close()
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(wav), "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-t", f"{self.n / self.fps:.3f}", str(out)], check=True)
        tmp.unlink()
        wav.unlink()
        return out


if __name__ == "__main__":
    p = Path(sys.argv[1])
    m = json.loads(_p(p).read_text(encoding="utf-8-sig"))
    pv = "--preview" in sys.argv
    out = ROOT / "output" / f"{m['id']}{'_preview' if pv else ''}.mp4"
    print(SinsEdit(m, preview=pv).render(out))
