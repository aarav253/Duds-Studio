"""General segment engine: an edit is a list of timed segments, each with its
own layout, grade, transition effects and captions, over a song, with
voicelines and the standard DUDS outro. Used for templates 14-21, 23, the
Barou-vs-Shidou versus edit and the freestyles.

segment = {a, b,                         # seconds
  layout: single | video | split3 | versus | card,
  clip / clips / src (raw video) / t0,   # material
  start, speed, focus [x,y], zoom [z0, z1], pan [dx, dy],
  grade: {sat, bright, contrast, tint [r,g,b], tint_amt, bw, glow},
  fx: [radial_in, invert_in, white_in, glitch_in, rgb_in, shake, zoom_punch, fade_in, end_blur, dark_out,
       mblur (+mblur, mblur_angle), flicker (+flicker_frames, flicker_dark), static (+static_tint)],
  tiles: [{clip, rect, at, start, still, from, dur, fy (focus band), pop}], stars (starfield bg),
  bg [r,g,b] + text (card), captions: [{text, y, size, a, b (relative s), font, color, glow, band, sweep, fade_in}]}

manifest = {id, size, fps, music: {file, offset, gain_db}, body_end, segments,
            captions (absolute), voicelines [{file, at, gain_db}], duck_db, lufs, outro, watermark}

    python src/seg_edit.py edits/<id>.json [--preview]
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sins_edit import Clip                                   # noqa: E402
from template_edit import end_card, Film, STYLE              # noqa: E402
from focus_edit import zoom, radial_blur, caption_layer, over, ease   # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CAP_DEFAULT = {"font": "C:/Windows/Fonts/bahnschrift.ttf", "variation": "Bold Condensed", "size": 0.05, "y": 0.5,
               "color": [250, 250, 250], "stroke": 0.003, "stroke_alpha": 0.75}


def _p(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class Video:
    """raw footage (any size/fps) cover-fitted, sampled by seconds."""

    def __init__(self, path, W, H, t0=0.0):
        self.cap = cv2.VideoCapture(str(_p(path)))
        self.fps = self.cap.get(5) or 30
        self.n = int(self.cap.get(7))
        self.W, self.H, self.t0, self.pos, self.last = W, H, t0, -1, None

    def at(self, t):
        k = int(min(max((self.t0 + t) * self.fps, 0), self.n - 1))
        if k != self.pos:
            if k != self.pos + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, b = self.cap.read()
            self.pos = k
            if ok:
                h, w = b.shape[:2]
                s = max(self.W / w, self.H / h)
                b = cv2.resize(b, (max(self.W, round(w * s)), max(self.H, round(h * s))), interpolation=cv2.INTER_AREA)
                y, x = (b.shape[0] - self.H) // 2, (b.shape[1] - self.W) // 2
                self.last = cv2.cvtColor(b[y:y + self.H, x:x + self.W], cv2.COLOR_BGR2RGB)
        return self.last if self.last is not None else np.zeros((self.H, self.W, 3), np.uint8)


def grade(img, g):
    if not g:
        return img.astype(np.float32)
    f = img.astype(np.float32)
    gray = f.mean(2, keepdims=True)
    if g.get("bw"):
        f = np.repeat(gray, 3, 2)
    elif "sat" in g:
        f = gray + (f - gray) * g["sat"]
    if g.get("contrast"):
        f = (f - 128) * g["contrast"] + 128
    if g.get("tint") is not None:
        a = g.get("tint_amt", 0.6)
        lum = np.clip(f.mean(2, keepdims=True) / 255, 0, 1)
        col = np.array(g["tint"], np.float32)
        duo = np.where(lum < 0.6, lum / 0.6 * col, col + (255 - col) * (lum - 0.6) / 0.4)
        f = f * (1 - a) + duo * a
    if g.get("dots"):                                          # halftone screen: bright areas -> big dots, dark -> small
        H_, W_ = f.shape[:2]; p = max(4, int(g.get("dot_p", 0.009) * W_))
        yy, xx = np.mgrid[0:H_, 0:W_]
        d = np.hypot((xx % p) - p / 2, (yy % p) - p / 2)[..., None] / (p * 0.72)
        lum = np.clip(f.mean(2, keepdims=True) / 255, 0, 1)
        m_ = np.clip((np.sqrt(lum) - d) * 4 + 0.5, 0, 1)
        f = f * (1 - g["dots"]) + f * (0.35 + 0.95 * m_) * g["dots"]
    if g.get("bright"):
        f = f * g["bright"]
    if g.get("glow"):
        hl = cv2.GaussianBlur(np.clip(f - 150, 0, 255), (0, 0), 0.02 * f.shape[1])
        f = f + hl * g["glow"]
    return f


def glitch(img, rng, n=5, amt=0.07, red=True):
    img = img.copy()
    H, W = img.shape[:2]
    for _ in range(n):
        y = int(rng.uniform(0.05, 0.95) * H); h = max(2, int(rng.uniform(0.01, 0.05) * H))
        img[y:y + h] = np.roll(img[y:y + h], int(rng.uniform(-amt, amt) * W), 1)
        if red:
            img[y:y + h, :, 0] = np.minimum(255, img[y:y + h, :, 0] + 70)
    return img


def _load_cutout(path):
    """RGBA png -> (rgb uint8, alpha float 0-1)."""
    im = np.asarray(Image.open(_p(path)).convert("RGBA"))
    return np.ascontiguousarray(im[..., :3]), im[..., 3].astype(np.float32) / 255


def mblur(img, n, angle=0):
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1.0
    if angle:
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle, 1.0), (n, n))
    return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))


def static(img, rng, tint):
    """TV-static band noise over a washed-out frame (glitch transition)."""
    H, W = img.shape[:2]
    rows = rng.uniform(0, 1, (H // 3 + 1, 1)).repeat(3, 0)[:H]
    noise = (rows * 0.6 + rng.uniform(0, 0.4, (H, W))) * 255
    col = np.array(tint, np.float32) / 255
    base = img.astype(np.float32) * 0.25 + 150
    out = base * 0.35 + noise[..., None] * col * 0.65 + (255 - noise[..., None]) * 0.15
    return np.clip(out, 0, 255).astype(np.uint8)


def manga_ink(img):
    """the frame as a black-and-white manga page: inked edges, flat whites, screentone in the mid-greys."""
    g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g = cv2.bilateralFilter(g, 7, 40, 7)
    edges = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6)
    H, W = g.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dots = ((np.sin(xx * 0.9) * np.sin(yy * 0.9) + 1) * 127.5)            # halftone screen
    tone = np.where(g > 170, 255, np.where(g < 70, 0, np.where(g > dots * 0.55 + 60, 255, 0))).astype(np.uint8)
    ink = np.minimum(tone, edges)
    return np.repeat(ink[..., None], 3, 2).astype(np.float32)


def rgb_split(img, px):
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], px, 1)
    out[..., 2] = np.roll(img[..., 2], -px, 1)
    return out


class SegEdit:
    def __init__(self, m, preview=False):
        self.m = m
        W0, H0 = m.get("size", [1080, 1350])
        k = 0.5 if preview else 1.0
        self.W, self.H = int(W0 * k) // 2 * 2, int(H0 * k) // 2 * 2
        self.fps = m.get("fps", 60)
        self.body = m["body_end"]
        o = m.get("outro", {"card": 0.033, "fade": 0.551, "dark": 0.2})
        self.outro = o
        self.n = int(round((self.body + o["card"] + o["fade"] + o["dark"]) * self.fps))
        self.segs = []
        for s in m["segments"]:
            s = dict(s)
            lay = s.get("layout", "single")
            if lay == "single":
                s["_c"] = Clip(s["clip"], self.W, self.H)
            elif lay == "video":
                s["_v"] = Video(s["src"], self.W, self.H, s.get("t0", 0.0))
            elif lay == "split3":
                s["_cs"] = [Clip(c, *self._rect_size(r)) for c, r in zip(s["clips"], self._split_rects())]
            elif lay == "versus":
                if s.get("fy"):                                   # full-frame clips, half-height window around each face
                    s["_cs"] = [Clip(c, self.W, self.H) for c in s["clips"]]
                else:
                    s["_cs"] = [Clip(c, *self._rect_size(r)) for c, r in zip(s["clips"], self._versus_rects())]
            elif lay == "card" and s.get("clip"):
                s["_c"] = Clip(s["clip"], self.W, self.H)
            elif lay == "tiles":
                s["_tiles"] = []
                s["_under"] = [(u, Clip(u["clip"], self.W, self.H)) for u in s.get("under", [])]
                for tl in s["tiles"]:
                    x0, y0, x1, y1 = [int(v * (self.W if i % 2 == 0 else self.H)) for i, v in enumerate(tl["rect"])]
                    if tl.get("half"):                            # one half of a full-frame image
                        c = Clip(tl["clip"], self.W, self.H)
                    elif tl.get("wide"):                          # 16:9 source at tile width, band cut around fy
                        c = Clip(tl["clip"], x1 - x0, max(int((x1 - x0) * 9 / 16) // 2 * 2, y1 - y0))
                    elif tl.get("fy") is not None:                # strip: whole frame at tile width, band cut around fy
                        c = Clip(tl["clip"], x1 - x0, int((x1 - x0) * self.H / self.W))
                    else:
                        c = Clip(tl["clip"], x1 - x0, y1 - y0)
                    s["_tiles"].append((tl, (x0, y0, x1, y1), c))
            if s.get("cutout"):
                s["_cut"] = _load_cutout(s["cutout"]["png"])
            s["_caps"] = [(dict(CAP_DEFAULT, **c), caption_layer(self.W, self.H, dict(CAP_DEFAULT, **c))) for c in s.get("captions", [])]
            self.segs.append(s)
        self.gcaps = [(dict(CAP_DEFAULT, **c), caption_layer(self.W, self.H, dict(CAP_DEFAULT, **c))) for c in m.get("captions", [])]
        self.endcard = end_card(m.get("end_text", "DUDS"), self.W, self.H, dict(STYLE)).astype(np.float32)
        self.dark = np.zeros((self.H, self.W, 3), np.float32) + np.array(STYLE["dark_bg"], np.float32)
        self.film = Film(self.W, self.H)
        lay = Image.new("L", (self.W, self.H), 0)
        ImageDraw.Draw(lay).text((self.W / 2, self.H * 0.962), " ".join(m.get("watermark", "DUDS")),
                                 font=ImageFont.truetype(r"C:\Windows\Fonts\bahnschrift.ttf", int(self.H * 0.016)), fill=150, anchor="mm")
        self.wm = np.asarray(lay).astype(np.float32)[..., None] / 255

    # ---------- layouts
    def _split_rects(self):
        W, H = self.W, self.H
        g = max(2, int(0.006 * W)); top = int(0.55 * H)
        return [(0, 0, W, top - g), (0, top + g, W // 2 - g, H), (W // 2 + g, top + g, W, H)]

    def _versus_rects(self):
        W, H = self.W, self.H
        g = max(2, int(0.004 * H)); mid = H // 2
        return [(0, 0, W, mid - g), (0, mid + g, W, H)]

    @staticmethod
    def _rect_size(r):
        return r[2] - r[0], r[3] - r[1]

    @staticmethod
    def _tile_frame(tl, c, k, h):
        fr = c.frame(k)
        if tl.get("half"):
            w = fr.shape[1] // 2
            fr = fr[:, :w] if tl["half"] == "L" else fr[:, w:]
            return fr
        if tl.get("fy") is not None:
            yy = int(np.clip(tl["fy"] * fr.shape[0] - h / 2, 0, fr.shape[0] - h))
            fr = fr[yy:yy + h]
        return fr

    def _look(self, f, lk, k, s, t):
        """reference-style overall look: highlight bloom, soft diffusion, a touch more contrast / colour,
        and a constant hand-held jitter (new offset every `shake_every` frames, bigger right on a cut)."""
        f = np.clip(f, 0, 255)
        if lk.get("diffuse") and not s.get("no_blur"):
            f = f * (1 - lk["diffuse"]) + cv2.GaussianBlur(f, (0, 0), 0.0025 * self.W) * lk["diffuse"]
        if lk.get("bloom"):
            hl = np.clip(f - lk.get("bloom_thr", 150), 0, 255)
            f = f + cv2.GaussianBlur(hl, (0, 0), 0.018 * self.W) * lk["bloom"]
        if lk.get("sat", 1.0) != 1.0:
            g = f.mean(2, keepdims=True); f = g + (f - g) * lk["sat"]
        if lk.get("contrast", 1.0) != 1.0:
            f = (f - 128) * lk["contrast"] + 128
        if lk.get("shaky_glitch"):                            # CapCut "Shaky glitch" replica (src/fx_shaky_glitch.py)
            from fx_shaky_glitch import shaky_glitch
            f = shaky_glitch(f, k, t, self.fps, intensity=lk["shaky_glitch"], cut=t - s["a"] < 0.1,
                             **({"band_h": 0.0} if s.get("no_blur") else {}))
        return f

    def _draw_cutout(self, f, s, t, u):
        """character cutout (RGBA) centred at (x, y), height h of the frame, slow zoom + drift + pop-in."""
        cu = s["cutout"]; rgb, al = s["_cut"]
        z0, z1 = cu.get("zoom", [1.0, 1.06])
        rel = t - s["a"] - cu.get("at", 0.0)
        if rel < 0:
            return f
        pop = min(1.0, rel / cu.get("pop", 0.15))
        h = int(cu.get("h", 0.62) * self.H * (z0 + (z1 - z0) * u) * (0.85 + 0.15 * ease(pop)))
        w = int(h * rgb.shape[1] / rgb.shape[0])
        mw = cu.get("max_w", 0.92) * self.W                       # wide poses (a kick) are fitted by width instead
        if w > mw:
            h, w = int(h * mw / w), int(mw)
        if w < 4 or h < 4:
            return f
        im = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        a = cv2.resize(al, (w, h), interpolation=cv2.INTER_AREA)[..., None] * min(1.0, rel / 0.08)
        if cu.get("grade"):
            im = grade(np.clip(im, 0, 255).astype(np.uint8), cu["grade"])
        dx, dy = cu.get("drift", [0.0, -0.02])
        cx = int((cu.get("x", 0.5) + dx * u) * self.W); cy = int((cu.get("y", 0.5) + dy * u) * self.H)
        if cu.get("enter"):                                     # slide into frame from a side, smeared while moving
            pe = min(1.0, rel / cu.get("enter_dur", 0.14))
            if pe < 1.0:
                ex, ey = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}[cu["enter"]]
                cx += int(ex * (1 - ease(pe)) * (self.W * 0.5 + w / 2)); cy += int(ey * (1 - ease(pe)) * (self.H * 0.5 + h / 2))
                n_ = max(3, int(0.12 * self.W * (1 - pe)))
                im = mblur(np.clip(im, 0, 255).astype(np.uint8), n_, 90 if ey else 0).astype(np.float32)
                a = mblur(a, n_, 90 if ey else 0)[..., None] if a.ndim == 3 else mblur(a, n_, 90 if ey else 0)
        x0, y0 = cx - w // 2, cy - h // 2
        ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(self.W, x0 + w), min(self.H, y0 + h)
        if ax1 <= ax0 or ay1 <= ay0:
            return f
        sub_i = im[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]; sub_a = a[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
        if cu.get("glow"):                                     # soft light rim so the cutout separates from the panels
            g = cv2.GaussianBlur(sub_a, (0, 0), 0.012 * self.W)[..., None] * cu["glow"]
            f[ay0:ay1, ax0:ax1] = f[ay0:ay1, ax0:ax1] * (1 - g) + 255 * g
        f[ay0:ay1, ax0:ax1] = f[ay0:ay1, ax0:ax1] * (1 - sub_a) + sub_i * sub_a
        return f

    def _stars(self, seed=7):
        if getattr(self, "_star_bg", None) is None:
            rng = np.random.default_rng(seed)
            bg = np.zeros((self.H, self.W), np.float32)
            n = int(self.W * self.H / 2500)
            ys, xs = rng.integers(0, self.H, n), rng.integers(0, self.W, n)
            bg[ys, xs] = rng.uniform(80, 255, n)
            bg = cv2.GaussianBlur(bg, (0, 0), 0.8) * 2.2
            self._star_bg = np.repeat(np.clip(bg, 0, 255)[..., None], 3, 2)
        return self._star_bg

    def _clip_t(self, s, clip, t, spec=None):
        spec = spec or s
        start = spec.get("start", 4)
        need = max(1e-3, (s["b"] - s["a"]) * 60)
        speed = spec.get("speed") or min(1.0, max(0.45, (clip.n - start - 2) / need))
        return clip.frame(start + (t - s["a"]) * 60 * speed)

    def _material(self, s, t, u):
        lay = s.get("layout", "single")
        z0, z1 = s.get("zoom", [1.0, 1.04])
        zz = z0 + (z1 - z0) * (np.sin(np.pi * u) if s.get("zoom_bounce") else ease(u) if s.get("zoom_ease", "out") == "out" else u)
        pan = s.get("pan", [0, 0])
        foc = tuple(s.get("focus", (0.5, 0.45)))
        if lay == "single":
            return zoom(self._clip_t(s, s["_c"], t), zz, foc, pan[0] * u, pan[1] * u)
        if lay == "video":
            fr = zoom(s["_v"].at(t - s["a"]), zz, foc, pan[0] * u, pan[1] * u)
            for r in s.get("blur_rects", []):
                x0, y0, x1, y1 = int(r[0] * self.W), int(r[1] * self.H), int(r[2] * self.W), int(r[3] * self.H)
                fr = fr.copy()
                fr[y0:y1, x0:x1] = cv2.GaussianBlur(fr[y0:y1, x0:x1], (0, 0), 0.02 * self.W)
            return fr
        if lay in ("split3", "versus"):
            rects = self._split_rects() if lay == "split3" else self._versus_rects()
            img = np.zeros((self.H, self.W, 3), np.uint8) + np.array(s.get("gap_color", [240, 240, 240]), np.uint8)
            for i, ((x0, y0, x1, y1), c) in enumerate(zip(rects, s["_cs"])):
                fr = self._clip_t(s, c, t)
                if lay == "versus" and s.get("fy"):
                    h = y1 - y0
                    fy = s["fy"][i]
                    z = s.get("half_zoom", 1.0)
                    if z != 1.0:
                        fr = zoom(fr, z, (0.5, fy))
                    yy = int(np.clip(fy * self.H - h / 2, 0, self.H - h))
                    fr = fr[yy:yy + h]
                if lay == "versus":                          # the two halves drift apart slightly
                    fr = zoom(fr, 1.0 + 0.05 * u, (0.5, 0.45), (0.02 if i == 0 else -0.02) * u, 0)
                img[y0:y1, x0:x1] = fr
            return img
        if lay == "tiles":
            img = np.zeros((self.H, self.W, 3), np.float32) + np.array(s.get("bg", [20, 20, 20]), np.float32)
            if s.get("stars"):
                img = img + self._stars()
            rel = t - s["a"]
            act = [(u, c) for u, c in s["_under"] if u["a"] <= rel < u["b"]]
            if act:
                u, c = act[-1]
                fr = c.frame(u.get("start", 6) + (rel - u["a"]) * 60 * 0.6)
                fr = zoom(fr, 1.08 - 0.08 * ((rel - u["a"]) / max(1e-3, u["b"] - u["a"])), (0.5, 0.45))
                if u.get("mblur"):
                    fr = mblur(fr, max(3, int(u["mblur"] * self.W)), u.get("angle", 0))
                img = grade(fr, u.get("grade"))
            for tl, (x0, y0, x1, y1), c in s["_tiles"]:
                if rel < tl["at"]:
                    continue
                if tl.get("from"):                                # slide in from a side and lock into its quadrant
                    p = ease(min(1.0, (rel - tl["at"]) / tl.get("dur", 0.16)))
                    k_ = tl.get("start", 6) + ((rel - tl["at"]) * 60 * 0.6 if tl.get("play") else 0)   # play: keeps moving
                    fr = self._tile_frame(tl, c, k_, y1 - y0).astype(np.float32)
                    fr = grade(fr.astype(np.uint8), tl.get("grade") or s.get("tile_grade"))
                    w, h = x1 - x0, y1 - y0
                    dx, dy = {"top": (0, -1), "bottom": (0, 1), "left": (-1, 0), "right": (1, 0)}[tl["from"]]
                    ox = x0 + int(dx * (1 - p) * (self.W - x0 if dx > 0 else x1)); oy = y0 + int(dy * (1 - p) * (self.H - y0 if dy > 0 else y1))
                    if p < 1.0 and tl.get("smear", True):
                        fr = mblur(np.clip(fr, 0, 255).astype(np.uint8), max(3, int(tl.get("smear_amt", 0.08) * self.W * (1 - p))),
                                   90 if dy else 0).astype(np.float32)
                    imp = tl.get("impact")
                    if imp and p >= 1.0:                          # landing impact: arrives zoomed + smeared, settles sharp
                        q = (rel - tl["at"] - tl.get("dur", 0.16)) / imp
                        if q < 1.0:
                            e_ = (1 - ease(max(0.0, q)))
                            z = 1.0 + tl.get("impact_zoom", 0.14) * e_
                            fr = zoom(np.clip(fr, 0, 255).astype(np.uint8), z, (0.5, 0.5),
                                      np.sin(rel * 90) * 0.012 * e_, np.cos(rel * 77) * 0.012 * e_).astype(np.float32)
                            sm = int(tl.get("impact_blur", 0.05) * self.W * e_)
                            if sm >= 3:
                                fr = mblur(np.clip(fr, 0, 255).astype(np.uint8), sm, 90 if dy else 0).astype(np.float32)
                    ax0, ay0, ax1, ay1 = max(0, ox), max(0, oy), min(self.W, ox + w), min(self.H, oy + h)
                    if ax1 > ax0 and ay1 > ay0:
                        img[ay0:ay1, ax0:ax1] = fr[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox]
                    continue
                p = min(1.0, (rel - tl["at"]) / 0.12)            # pop: scale 0.82 -> 1 in 0.12 s, white flash
                k_ = tl.get("start", 6) if tl.get("still") else tl.get("start", 6) + (rel - tl["at"]) * 60 * 0.6
                fr = self._tile_frame(tl, c, k_, y1 - y0).astype(np.float32)
                fr = grade(fr.astype(np.uint8), tl.get("grade") or s.get("tile_grade"))
                sc = (0.82 + 0.18 * ease(p)) if tl.get("pop", True) else 1.0
                w, h = x1 - x0, y1 - y0
                fw, fh = max(2, int(w * sc)), max(2, int(h * sc))
                fr = cv2.resize(fr, (fw, fh))
                if tl.get("pop", True):
                    fr = fr + (255 - fr) * 0.7 * (1 - p)
                ox, oy = x0 + (w - fw) // 2, y0 + (h - fh) // 2
                img[oy:oy + fh, ox:ox + fw] = fr
            return np.clip(img, 0, 255).astype(np.uint8)
        if lay == "card":
            bg = np.zeros((self.H, self.W, 3), np.float32) + np.array(s.get("bg", [10, 10, 14]), np.float32)
            if s.get("stars"):
                bg = bg + self._stars()
            if s.get("_c") is not None:
                fr = zoom(self._clip_t(s, s["_c"], t), zz, foc).astype(np.float32)
                a = s.get("clip_alpha", 0.35)
                bg = bg * (1 - a) + fr * a
            return np.clip(bg, 0, 255).astype(np.uint8)
        raise ValueError(lay)

    def frame(self, k):
        t = k / self.fps
        if t >= self.body:
            return self.outro_frame(k, t)
        s = next((x for x in self.segs if x["a"] <= t < x["b"]), self.segs[-1])
        dur = s["b"] - s["a"]
        u = min(1.0, max(0.0, (t - s["a"]) / dur))
        j = int(round((t - s["a"]) * self.fps))
        fx = s.get("fx", [])
        rng = np.random.default_rng(k)
        img = self._material(s, t, u)
        pu = s.get("push")                                       # smooth push: the next panel slides in, the last one out
        if pu and t - s["a"] < pu.get("dur", 0.18):
            i_ = self.segs.index(s)
            if i_ > 0 and self.segs[i_ - 1].get("layout") != "card":
                prev = self.segs[i_ - 1]
                q = ease((t - s["a"]) / pu.get("dur", 0.18))
                pimg = self._material(prev, t, 1.0)
                dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[pu.get("dir", "left")]
                ox, oy = int(dx * self.W * (1 - q)), int(dy * self.H * (1 - q))
                px_, py_ = ox - dx * self.W, oy - dy * self.H
                M1 = np.float32([[1, 0, ox], [0, 1, oy]]); M0 = np.float32([[1, 0, px_], [0, 1, py_]])
                cur = cv2.warpAffine(img, M1, (self.W, self.H)); old_ = cv2.warpAffine(pimg, M0, (self.W, self.H))
                mask = cv2.warpAffine(np.ones((self.H, self.W), np.uint8), M1, (self.W, self.H))[..., None]
                img = np.where(mask > 0, cur, old_)
                n_ = int(0.06 * self.W * np.sin(np.pi * q))
                if n_ >= 3:
                    img = mblur(img, n_, 90 if dy else 0)
        if s.get("track"):                                       # slow push-in over the whole shot (any layout)
            z0, z1 = s["track"]
            img = zoom(img, z0 + (z1 - z0) * ease(u), tuple(s.get("track_focus", (0.5, 0.5))))
        if s.get("steps"):                                       # zoom that jumps on beats: [[t, total scale, ramp s], ...]
            zc = 1.0
            for st_t, st_z, st_d in s["steps"]:
                if t >= st_t:
                    zc_prev = zc; zc = st_z
                    if t < st_t + st_d:
                        zc = zc_prev + (st_z - zc_prev) * ease((t - st_t) / st_d)
            if zc != 1.0:
                img = zoom(img, zc, tuple(s.get("track_focus", (0.5, 0.5))))
        for h in s.get("hits", []):                              # impact at an absolute time: zoom punch + hard shake
            q = t - h
            if 0 <= q < 0.2:
                e_ = (1 - q / 0.2) ** 2
                img = zoom(img, 1 + s.get("hit_zoom", 0.12) * e_, (0.5, 0.5), rng.uniform(-0.025, 0.025) * e_, rng.uniform(-0.025, 0.025) * e_)
                if q < 0.07:
                    img = mblur(img, max(3, int(0.05 * self.W * e_)), 90)
        mr = s.get("manga_reveal")                              # manga panel -> the coloured version of the same panel
        if mr:
            at = s["a"] + mr.get("at", 0.5) * dur; wipe = mr.get("wipe", 0.12)
            if t < at + wipe:
                ink = manga_ink(img)
                if t < at:
                    img = ink
                else:                                              # diagonal wipe with a white seam, smeared
                    q = ease((t - at) / wipe)
                    H_, W_ = img.shape[:2]
                    yy, xx = np.mgrid[0:H_, 0:W_]
                    d = (xx / W_ + yy / H_) / 2
                    m_ = np.clip((q * 1.3 - d) / 0.06, 0, 1)[..., None]
                    seam = np.exp(-((d - q * 1.3 + 0.03) / 0.015) ** 2)[..., None]
                    img = ink * (1 - m_) + img.astype(np.float32) * m_
                    img = np.clip(img + 255 * seam * 0.9, 0, 255)
                img = img.astype(np.uint8)
        if "radial_in" in fx and j < 6:
            img = radial_blur(img, 0.18 * (1 - j / 6) ** 1.5)
        if "glitch_in" in fx and j < 8:
            img = glitch(img, rng, 6, 0.08, s.get("glitch_red", True))
        if "rgb_in" in fx and j < 6:
            img = rgb_split(img, int(0.012 * self.W * (1 - j / 6)) + 1)
        if "shake" in fx:
            a_ = s.get("shake_amt", 0.006)
            img = zoom(img, 1.0 + 5 * a_, (0.5, 0.5), rng.uniform(-a_, a_), rng.uniform(-a_, a_))
        if "mblur" in fx:                                      # smeared fast-motion look, strongest on the cut
            amt = s.get("mblur", 0.05) * (0.45 + 0.55 * (1 - u) ** 2)
            img = mblur(img, max(3, int(amt * self.W)), s.get("mblur_angle", 0))
        if "static" in fx:
            img = static(img, rng, s.get("static_tint", [120, 220, 235]))
        if "zoom_punch" in fx and j < 8:
            img = zoom(img, 1 + 0.08 * (1 - j / 8) ** 2, (0.5, 0.45))
        f = grade(img, s.get("grade"))
        if "invert_in" in fx and j <= 1:
            g = 255 - cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            f = np.repeat((cv2.GaussianBlur(g, (0, 0), 2) * 0.85 + 60)[..., None], 3, 2)
        if "white_in" in fx and j < 5:
            f = f + (255 - f) * 0.8 * (1 - j / 5)
        for fa, fb, amt in s.get("flashes", []):                 # absolute-time white flashes inside a shot
            if fa <= t < fb:
                f = f + (255 - f) * amt
        if "fade_in" in fx:
            f = f * ease(min(1.0, (t - s["a"]) / s.get("fade_time", 0.3)))
        if "end_blur" in fx and u > 0.35:
            b = ease((u - 0.35) / 0.65)
            bl = cv2.GaussianBlur(np.clip(f, 0, 255).astype(np.uint8), (0, 0), 1 + 18 * b).astype(np.float32)
            f = bl + (235 - bl) * 0.55 * b
        if "flicker" in fx and (j // max(1, int(s.get("flicker_frames", 4)))) % 2 == 1:
            f = f * s.get("flicker_dark", 0.12)
        if s.get("dark_last") and t > s["b"] - s["dark_last"]:
            f = f * (1 - 0.92 * ease((t - (s["b"] - s["dark_last"])) / s["dark_last"]))
        if "dark_out" in fx and u > 0.6:
            f = f * (1 - 0.85 * ease((u - 0.6) / 0.4))
        if s.get("layout") == "versus" and s.get("divider", True):
            mid = self.H // 2; th = max(2, int(0.004 * self.H))
            f[mid - th:mid + th] = np.array(s.get("divider_color", [235, 40, 60]), np.float32)
        if s.get("_cut") is not None:
            f = self._draw_cutout(f, s, t, u)
        for c, layer in s["_caps"]:
            a = s["a"] + c.get("a", 0.0); b = s["a"] + c.get("b", dur)
            if a <= t < b:
                al = min(1.0, (t - a) / c["fade_in"]) if c.get("fade_in") else 1.0
                xm = None
                if c.get("sweep"):
                    p = ease((t - a) / c["sweep"]); xs = np.linspace(0, 1, self.W)
                    xm = np.clip((p * 1.15 - xs) / 0.15, 0, 1).astype(np.float32)
                lay_ = layer
                if c.get("slide"):                                 # enter from the left/right, smeared while moving
                    p = min(1.0, (t - a) / c["slide"][1])
                    if p < 1.0:
                        dx = int((1 - ease(p)) * self.W * (-1 if c["slide"][0] == "left" else 1))
                        lay_ = np.roll(layer, dx, 1)
                        lay_ = mblur(lay_, max(3, int(0.12 * self.W * (1 - p))))
                if c.get("blur_out") and t > b - c["blur_out"]:
                    q = (t - (b - c["blur_out"])) / c["blur_out"]
                    lay_ = mblur(lay_, max(3, int(0.15 * self.W * q))); al = al * (1 - 0.7 * q)
                f = over(f, lay_, al, xm)
        for c, layer in self.gcaps:
            if c["a"] <= t < c["b"]:
                al = min(1.0, (t - c["a"]) / c["fade_in"]) if c.get("fade_in") else 1.0
                f = over(f, layer, al)
        lk = self.m.get("look")
        if lk and s.get("layout") != "video":                  # whole-edit filter (the reference's video part already has it)
            f = self._look(f, lk, k, s, t)
        fi = self.m.get("fade_in", 0)
        if fi and t < fi:
            f = f * ease(t / fi)
        f = f * (1 - self.wm * 0.7) + 235 * self.wm * 0.7
        return np.clip(f, 0, 255).astype(np.uint8)

    def outro_frame(self, k, t):
        o = self.outro
        u = t - self.body
        if u < o["card"]:
            f, dark = self.endcard, False
        else:
            a = ease(min(1.0, (u - o["card"]) / o["fade"]))
            f, dark = self.endcard * (1 - a) + self.dark * a, a > 0.4
        return self.film.apply(np.clip(f, 0, 255).astype(np.uint8), k, 0.6 if dark else 0.22)

    # ---------- audio
    def mix(self, out):
        sr = 44100
        mu = self.m["music"]
        y, s0 = sf.read(str(_p(mu["file"])), always_2d=True)
        if s0 != sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, s0); y = resample_poly(y, sr // g, s0 // g, axis=0)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, 1)
        n = int(self.n / self.fps * sr)
        o = int(mu.get("offset", 0) * sr)
        music = np.zeros((n, 2), np.float32)
        seg = y[o:o + n]
        music[:len(seg)] = seg * 10 ** (mu.get("gain_db", 0) / 20)
        for part in self.m.get("music_parts", []):             # extra pieces (e.g. intro clip's own audio)
            yy, s1 = sf.read(str(_p(part["file"])), always_2d=True)
            a, b = int(part["at"] * sr), int(part["to"] * sr)
            src = yy[int(part.get("from", 0) * s1):int(part.get("from", 0) * s1) + (b - a)]
            if s1 != sr:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, s1); src = resample_poly(src, sr // g, s1 // g, axis=0)[:b - a]
            if src.shape[1] == 1:
                src = np.repeat(src, 2, 1)
            music[a:a + len(src)] = src * 10 ** (part.get("gain_db", 0) / 20)
            if part.get("silence_music_until"):
                pass
        from voice_mix import add_voices
        mix = add_voices(music, self.m.get("voicelines", []), sr, self.m.get("duck_db", 4.0), self.m.get("voice_over_music", 1.4))
        from scipy.ndimage import maximum_filter1d as mf, uniform_filter1d as uf
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        body = int(self.body * sr)
        for _ in range(3):
            mix = pyln.normalize.loudness(mix, meter.integrated_loudness(mix[:body]), self.m.get("lufs", -14.0))
            env = mf(np.abs(mix).max(1), int(0.01 * sr))
            g = np.minimum(1.0, 0.94 / np.maximum(env, 1e-6))
            mix = mix * np.minimum(g, uf(g, int(0.02 * sr)))[:, None]
        a = int((self.body - self.m.get("audio_fade", 1.5) * 0.6) * sr)
        mix[a:] *= (np.cos(np.linspace(0, np.pi / 2, n - a)) ** 1.5)[:, None]
        sf.write(str(out), mix.astype(np.float32), sr)
        return meter.integrated_loudness(mix[:body])

    def render(self, out):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp, wav = out.with_suffix(".video.mp4"), out.with_suffix(".mix.wav")
        lufs = self.mix(wav)
        wr = imageio_ffmpeg.write_frames(str(tmp), (self.W, self.H), fps=self.fps, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "15", "-preset", "medium"], macro_block_size=2)
        wr.send(None)
        for k in range(self.n):
            wr.send(np.ascontiguousarray(self.frame(k)))
        wr.close()
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(wav), "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", str(out)], check=True)
        tmp.unlink(); wav.unlink()
        print(f"  {out.name}: {self.n / self.fps:.1f}s, loudness {lufs:.1f} LUFS", flush=True)
        return out


    def render_frames(self, out, k0, k1):
        """video only, frames k0..k1-1 (one chunk of a parallel render, see src/fast_render.py)."""
        wr = imageio_ffmpeg.write_frames(str(out), (self.W, self.H), fps=self.fps, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "15", "-preset", "medium"], macro_block_size=2)
        wr.send(None)
        for k in range(k0, min(k1, self.n)):
            wr.send(np.ascontiguousarray(self.frame(k)))
        wr.close()
        return out


if __name__ == "__main__":
    m = json.loads(_p(sys.argv[1]).read_text(encoding="utf-8-sig"))
    pv = "--preview" in sys.argv
    if "--frames" in sys.argv:                                   # --frames k0 k1 out.mp4
        i = sys.argv.index("--frames")
        print(SegEdit(m, pv).render_frames(Path(sys.argv[i + 3]), int(sys.argv[i + 1]), int(sys.argv[i + 2])))
    else:
        print(SegEdit(m, pv).render(ROOT / "output" / f"{m['id']}{'_preview' if pv else ''}.mp4"))
