"""Renderer for Wekisava-style templates ("engine": "wekisava").

Different grammar from the Valtrez engine (template_edit.py):
  - a fixed beat grid of HELD shots joined by HORIZONTAL SMEAR transitions
    (rows tear sideways, blur, flare to a cyan/white peak, re-form)
  - a black-and-white intro (with a black strobe) that drops into a CYAN
    DUOTONE section: the character is cut out, lit in cyan with an edge glow
    and procedural lightning over a crushed, near-black background
  - a dark section of eye-glow panels that fade to black between bursts
  - a glitch logo outro

The template holds an explicit, frame-exact timeline (measured on the
reference) - see templates/video_edit_template_4.json. Frames are indices at
the template fps (= reference fps), so every event lands on the reference frame.

    python src/wekisava_edit.py edits/<manifest>.json [--preview]
"""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
CUT_CACHE = ROOT / ".cache" / "cutouts"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def ease_out(x, p=2.0):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** p


def ease_in(x, p=2.0):
    x = min(max(x, 0.0), 1.0)
    return x ** p


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def smooth_noise(t, seed, freqs=(0.9, 1.7, 3.1)):
    """Continuous pseudo-random signal in ~[-1, 1] (sum of sines): used for
    camera shake so it is fluid, never jumping between random positions."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * math.pi, len(freqs))
    return sum(math.sin(2 * math.pi * f * t + p) for f, p in zip(freqs, ph)) / len(freqs)


def load_rgb(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    return np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))


def luma(img):
    return img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


class Placed:
    """A source image cover-fitted to the frame, with the mapping from source
    fractions to frame pixels (so eye positions etc. follow the crop)."""

    def __init__(self, src, W, H, focus=(0.5, 0.40), overscan=1.0, at=(0.5, 0.5)):
        h, w = src.shape[:2]
        s = max(W / w, H / h) * overscan
        nw, nh = int(math.ceil(w * s)), int(math.ceil(h * s))
        self.img_full = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        # `focus` (source fraction) lands at `at` (frame fraction)
        x0 = min(max(focus[0] * nw - at[0] * W, 0), nw - W)
        y0 = min(max(focus[1] * nh - at[1] * H, 0), nh - H)
        self.x0, self.y0, self.s, self.W, self.H = int(round(x0)), int(round(y0)), s, W, H
        self.img = self.img_full[self.y0:self.y0 + H, self.x0:self.x0 + W].copy()
        self.src_wh = (w, h)

    def map(self, u, v):
        w, h = self.src_wh
        return u * w * self.s - self.x0, v * h * self.s - self.y0

    def crop(self, arr_full):
        return arr_full[self.y0:self.y0 + self.H, self.x0:self.x0 + self.W]


def warp(img, scale=1.0, anchor=(0.5, 0.5), rot=0.0, dx=0.0, dy=0.0, border=cv2.BORDER_REFLECT):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((anchor[0] * W, anchor[1] * H), rot, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border)


# ---------------------------------------------------------------------------
# Cutouts (anime segmentation), cached on disk
# ---------------------------------------------------------------------------

_SESSION = None


def cutout_mask(path):
    """Soft foreground mask (0..1, source resolution) of the character."""
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    CUT_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(p.read_bytes()).hexdigest()[:16]
    cached = CUT_CACHE / f"{key}.png"
    if cached.exists():
        return np.asarray(Image.open(cached)).astype(np.float32) / 255
    global _SESSION
    from rembg import new_session, remove
    if _SESSION is None:
        _SESSION = new_session("isnet-anime")
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    m = remove(im, session=_SESSION, only_mask=True)
    m.save(cached)
    return np.asarray(m).astype(np.float32) / 255


def subject_point(path):
    """(x, y) source fractions of the character's head/upper body: centroid
    of the cutout mask inside the top 40% of its bounding box."""
    m = cutout_mask(path) > 0.5
    ys, xs = np.nonzero(m)
    if len(ys) < 0.02 * m.size:
        return 0.5, 0.40
    top, bot = np.percentile(ys, 1), np.percentile(ys, 99)
    sel = ys <= top + 0.40 * (bot - top)
    h, w = m.shape
    return float(xs[sel].mean() / w), float(ys[sel].mean() / h)


# ---------------------------------------------------------------------------
# Grades (measured colour curves: luma -> RGB, see template "grades")
# ---------------------------------------------------------------------------

def make_lut(curve):
    """curve: [[luma, R, G, B], ...] -> (256, 3) uint8 LUT."""
    c = np.array(curve, np.float32)
    xs = np.concatenate([[0], c[:, 0], [255]])
    lut = np.zeros((256, 3), np.float32)
    for ch in range(3):
        ys = np.concatenate([[c[0, ch + 1] * 0.6], c[:, ch + 1], [c[-1, ch + 1]]])
        lut[:, ch] = np.interp(np.arange(256), xs, ys)
    return lut.clip(0, 255).astype(np.uint8)


def apply_lut(gray_u8, lut):
    return lut[gray_u8]


# ---------------------------------------------------------------------------
# Procedural overlays
# ---------------------------------------------------------------------------

def bolt_points(p0, p1, rng, depth=6, disp=0.22):
    pts = [np.array(p0, float), np.array(p1, float)]
    for d in range(depth):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            mid = (a + b) / 2
            v = b - a
            n = np.array([-v[1], v[0]])
            mid = mid + n * rng.uniform(-disp, disp) / (1.4 ** d)
            new += [mid, b]
        pts = new
    return pts


def lightning_layer(W, H, rng, anchors, n=4, length=0.35):
    """RGBA-ish float layers (glow, core) of branching cyan bolts."""
    glow = Image.new("L", (W, H), 0)
    core = Image.new("L", (W, H), 0)
    dg, dc = ImageDraw.Draw(glow), ImageDraw.Draw(core)
    for _ in range(n):
        if len(anchors):
            ax, ay = anchors[rng.integers(len(anchors))]
        else:
            ax, ay = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H
        ang = rng.uniform(0, 2 * math.pi)
        L = length * min(W, H) * rng.uniform(0.5, 1.2)
        p1 = (ax + math.cos(ang) * L, ay + math.sin(ang) * L)
        pts = [tuple(p) for p in bolt_points((ax, ay), p1, rng)]
        dg.line(pts, fill=255, width=int(0.018 * W))
        dc.line(pts, fill=255, width=max(1, int(0.004 * W)))
        # one branch
        j = rng.integers(len(pts) // 3, 2 * len(pts) // 3)
        b1 = (pts[j][0] + rng.uniform(-0.15, 0.15) * W, pts[j][1] + rng.uniform(-0.15, 0.15) * H)
        bp = [tuple(p) for p in bolt_points(pts[j], b1, rng, depth=4)]
        dg.line(bp, fill=180, width=int(0.012 * W))
        dc.line(bp, fill=200, width=max(1, int(0.003 * W)))
    g = cv2.GaussianBlur(np.asarray(glow).astype(np.float32) / 255, (0, 0), 0.012 * W)
    c = cv2.GaussianBlur(np.asarray(core).astype(np.float32) / 255, (0, 0), 0.7)
    return g, c


def speed_lines_layer(W, H, rng, n=18):
    lay = np.zeros((H, W), np.float32)
    for _ in range(n):
        y = int(rng.uniform(0, H))
        x0 = int(rng.uniform(-0.3, 0.8) * W)
        L = int(rng.uniform(0.2, 0.7) * W)
        t = max(1, int(rng.uniform(1, 4) * H / 1350))
        cv2.line(lay, (x0, y), (x0 + L, y), float(rng.uniform(0.4, 1.0)), t)
    return cv2.GaussianBlur(lay, (0, 0), 1.2)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class WekisavaEdit:
    def __init__(self, manifest, preview=False):
        self.m = manifest
        self.t = json.loads((ROOT / manifest["template"]).read_text(encoding="utf-8-sig"))
        self.st = self.t["style"]
        W, H = self.t["size"]
        sc = 0.5 if preview else 1.0
        self.W, self.H = int(W * sc) // 2 * 2, int(H * sc) // 2 * 2
        self.fps = self.t["fps"]
        self.n = self.t["frames"]
        self.luts = {k: make_lut(v) for k, v in self.t["grades"].items()}
        self.timeline = self.t["timeline"]
        self.slots = manifest["slots"]
        self.eyes = {k: v for k, v in manifest.get("eyes", {}).items()}
        self.focus = manifest.get("focus", {})
        self._cache = {}
        self._auto_center = {s["slot"] for s in self.timeline if s.get("look") == "bw" and "slot" in s}
        self.dark = np.array(self.st["black_rgb"], np.float32)
        self.wm = self._watermark(manifest.get("watermark", "DUDS"))
        self.logo_text = manifest.get("end_text", "DUDS")
        missing = sorted({s["slot"] for s in self.timeline if "slot" in s} - set(self.slots))
        if missing:
            raise ValueError(f"manifest is missing slots: {missing}")

    # ---- per-slot assets ----
    def placed(self, slot, overscan=1.0):
        key = ("placed", slot, overscan)
        if key not in self._cache:
            path = self.slots[slot]
            src = load_rgb(path)
            if slot in self.focus:
                pl = Placed(src, self.W, self.H, self.focus[slot], overscan)
            elif slot in self._auto_center:
                # intro panels: the character's head/upper body at frame centre
                pl = Placed(src, self.W, self.H, subject_point(path), overscan, at=(0.5, 0.40))
            else:
                pl = Placed(src, self.W, self.H, (0.5, 0.40), overscan)
            self._cache[key] = pl
        return self._cache[key]

    def mask(self, slot, overscan=1.0):
        key = ("mask", slot, overscan)
        if key not in self._cache:
            pl = self.placed(slot, overscan)
            m = cutout_mask(self.slots[slot])
            mf = cv2.resize(m, (pl.img_full.shape[1], pl.img_full.shape[0]), interpolation=cv2.INTER_LINEAR)
            self._cache[key] = pl.crop(mf).copy()
        return self._cache[key]

    def eye_points(self, slot, overscan=1.0):
        path = self.slots[slot]
        pts = self.eyes.get(path) or self.eyes.get(slot) or []
        pl = self.placed(slot, overscan)
        return [pl.map(u, v) for u, v in pts]

    # ---- looks ----
    def look_bw(self, slot, u, seg):
        pl = self.placed(slot, seg.get("overscan", 1.08))
        pan = seg.get("pan_x")
        if pan:
            # camera slides across a wider panel: content moves right by
            # `px` (at 1080 wide) with an exponential ease-out (measured on
            # the reference: 22 px/frame falling to ~2 px/frame)
            delay = pan.get("delay", 0)
            j = max(0.0, u * (seg["end"] - seg["start"]) - delay)
            L = seg["end"] - seg["start"] - delay
            tau = pan.get("tau", 18.0)
            p = (1 - math.exp(-j / tau)) / (1 - math.exp(-L / tau))
            shift = pan["px"] * self.W / 1080
            full = pl.img_full
            room = full.shape[1] - self.W
            mdx, mdy = self._drift_mean(seg)
            x_end = min(max(pl.x0 + mdx, 0), room)          # land centred on the character
            shift = min(shift, room - x_end)               # never run off the image
            x = x_end + shift * (1 - p)
            y = min(max(pl.y0 + mdy, 0), full.shape[0] - self.H)
            M = np.float32([[1, 0, -x], [0, 1, -y]])
            img = cv2.warpAffine(full, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            g = luma(img)
            # motion blur matching the pan speed (px moved this frame)
            v = shift * math.exp(-j / tau) / tau / (1 - math.exp(-L / tau)) if j > 0 else 0.0
            kw = int(v * 0.8) | 1
            if kw > 2:
                g = cv2.blur(g, (kw, 1))
        else:
            mdx, mdy = self._drift_mean(seg)
            full = pl.img_full
            x = min(max(pl.x0 + mdx, 0), full.shape[1] - self.W)
            y = min(max(pl.y0 + mdy, 0), full.shape[0] - self.H)
            g = luma(cv2.warpAffine(full, np.float32([[1, 0, -x], [0, 1, -y]]), (self.W, self.H),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        # crisp ink on near-white paper (reference tones: p5 14, p50 125, p95 252):
        # full-range S-curve, not a grey cast
        x = np.clip(g / 255, 0, 1)
        x = np.clip((x - 0.5) * self.st["bw_contrast"] + 0.5, 0, 1)
        x = x * x * (3 - 2 * x) * 0.55 + x * 0.45                 # gentle S
        # vignette (edges fall off ~25%) + vertical light-leak stripes
        W, H = self.W, self.H
        if ("bw_vig",) not in self._cache:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.hypot((xx / W - 0.5) * 1.2, (yy / H - 0.5))
            self._cache[("bw_vig",)] = np.clip(1.08 - r * 0.55, 0.62, 1.0)
        x = x * self._cache[("bw_vig",)]
        rng = np.random.default_rng(seg["start"])
        for _ in range(2):
            cx = rng.uniform(0.08, 0.92) * W
            wdt = rng.uniform(0.02, 0.06) * W
            xs = np.arange(W, dtype=np.float32)
            stripe = np.exp(-((xs - cx) / wdt) ** 2)[None, :] * rng.uniform(0.10, 0.22)
            x = x + stripe * (1 - x)
        g = (np.clip(x, 0, 1) * 255).astype(np.float32)
        rgb = np.repeat(g[..., None], 3, -1)
        # match the reference intro's tone distribution (keeps crisp whites at 0.7)
        return self.match_section(rgb, "bw", "bw", strength=self.st.get("bw_match", 0.7))

    def _flame_noise(self, k, seed, scale=0.06):
        """Low-frequency noise stretched vertically, scrolling upward with
        time: thresholded, it gives licking-flame blobs for the aura."""
        key = ("flamebase", seed)
        if key not in self._cache:
            rng = np.random.default_rng(seed)
            h, w = int(self.H * scale * 3) + 8, int(self.W * scale) + 8
            self._cache[key] = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.6)
        base = self._cache[key]
        off = (k * 0.35) % (base.shape[0] / 3)                # upward drift
        rows = (np.arange(int(self.H * scale)) + off).astype(int) % base.shape[0]
        small = base[rows]
        n = cv2.resize(small, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        return (n - n.mean()) / (n.std() + 1e-6)

    def look_cyan_aura(self, slot, seg, k):
        """Wekisava drop look (measured on the reference): the character is
        DARK (near-silhouette, deep teal-black) and EMITS cyan - flame-shaped
        wisps hugging its outline + cyan highlights - over a darkened grey
        manga background that keeps its detail. Composited additively."""
        ov = seg.get("overscan", 1.1)
        pl = self.placed(slot, ov)
        g = luma(pl.img) / 255
        # flatten the printed screentone first: amplified by the grade it
        # turned into busy speckle instead of smooth light
        g = cv2.bilateralFilter(g.astype(np.float32), 0, 0.08, self.st.get("tone_smooth", 2.5) * self.W / 1080)
        m = self.mask(slot, ov)
        bg_slot = seg.get("bg")
        pool = self.m.get("bg_pool", [])
        if bg_slot:
            bgl = luma(self.placed(bg_slot, ov).img) / 255
        elif pool:
            # the reference puts the character over OTHER busy manga panels
            idx = [s for s in self.timeline if s.get("look") in ("cyan", "cyan_strip")].index(seg) % len(pool)
            key = ("bgpool", idx)
            if key not in self._cache:
                self._cache[key] = luma(Placed(load_rgb(pool[idx]), self.W, self.H, (0.5, 0.45), 1.15).img) / 255
            bgl = self._cache[key]
        else:
            bgl = g
        # background: grey manga art, darkened + contrasty (detail kept), vignetted
        bgv = np.clip((bgl - 0.5) * 1.25 + 0.5, 0, 1) ** 1.35 * self.st["aura_bg_level"]
        if ("aura_vig",) not in self._cache:
            yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
            r = np.hypot((xx / self.W - 0.5) * 1.1, yy / self.H - 0.45)
            self._cache[("aura_vig",)] = np.clip(1.15 - r * 1.3, 0.2, 1.0)
        bgv = bgv * self._cache[("aura_vig",)]
        bg = np.stack([bgv * 0.92, bgv * 0.98, bgv * 1.06], -1)
        # ---- directional, pose-aware light ----
        # the light comes from ONE side: toward the open space the character
        # faces / leaves free (opposite the silhouette's offset from centre),
        # from slightly above. Overridable per shot with seg["light_deg"].
        L, dirw, keyw = self._light_field(slot, seg, m)
        # character: dark teal-black, ink lines still readable, key-lit on the lit side
        fv = np.clip(g, 0, 1) ** 1.6 * self.st["aura_fg_level"] * (0.55 + 0.9 * keyw)
        fg = np.stack([fv * 0.35, fv * 0.95, fv * 1.0], -1)
        m3 = m[..., None]
        out = fg * m3 + bg * (1 - m3)
        cyan = np.array(self.st["aura_rgb"], np.float32) / 255
        # aura: band around the outline (mostly outside), broken into flames
        s = self.W / 1080
        outer = cv2.GaussianBlur(m, (0, 0), self.st.get("aura_spread", 38) * s)
        inner = cv2.GaussianBlur(m, (0, 0), 6 * s)
        # flames live OUTSIDE the silhouette (hugging it); inside only a thin rim
        outside = np.clip(outer - m, 0, 1) * (1 - m)
        band = np.clip(outside * 3.0, 0, 1)
        edge_in = np.clip(m - cv2.GaussianBlur(m, (0, 0), 4 * s), 0, 1) * 1.6
        n = self._flame_noise(k, seg["start"])
        flames = np.clip((band * 1.35 + n * 0.55 - 0.78) * 2.4, 0, 1) * (1 - m * 0.9)
        flames = np.maximum(flames, np.clip(edge_in * (0.5 + 0.35 * n), 0, 1))
        flames = flames * dirw                     # flames + rim only on the lit side
        glow = cv2.GaussianBlur(flames, (0, 0), self.st.get("aura_glow", 12) * s)
        a = np.clip(flames * 0.85 + glow * 0.45, 0, 1.3)[..., None] * self.st["aura_level"]
        out = out + cyan * a
        # wide bloom: large soft pool of light, strongest on the lit side
        bloom = cv2.GaussianBlur(np.clip(a[..., 0], 0, 1), (0, 0), self.st.get("bloom_wide", 70) * s)
        out = out + cyan * bloom[..., None] * self.st.get("bloom_wide_level", 0.55)
        # background light spill from the light side (the reference shots
        # are lit from one side, with the far side falling into black)
        out = out * (0.68 + 0.32 * L[..., None]) + cyan * (L ** 2 * self.st.get("spill", 0.10))[..., None]
        # eyes glow only on close-ups (where the eyes are big) - fewer global effects
        eyes = self.eye_points(slot, ov)
        if eyes and (len(eyes) < 2 or abs(eyes[0][0] - eyes[1][0]) > self.st.get("eye_glow_min_sep", 0.14) * self.W):
            yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
            er = self.st["eye_radius"] * self.W * 0.8
            eg = np.zeros((self.H, self.W), np.float32)
            for ex, ey in eyes:
                d2 = ((xx - ex) ** 2 + (yy - ey) ** 2) / er ** 2
                eg = np.maximum(eg, np.exp(-d2 * 3.0) + 0.3 * np.exp(-d2 * 0.4))
            out = out + cyan * eg[..., None] + (eg ** 3)[..., None] * 0.5
        return out, m

    def _light_field(self, slot, seg, m):
        """(L, dirw, keyw): L = frame-wide light falloff from the light side,
        dirw = weight of rim/flames by direction around the silhouette,
        keyw = key light across the character itself. Cached per shot."""
        key = ("light", seg["start"])
        if key in self._cache:
            return self._cache[key]
        H, W = self.H, self.W
        ys, xs = np.nonzero(m > 0.5)
        if len(xs):
            cx, cy = xs.mean() / W, ys.mean() / H
        else:
            cx, cy = 0.5, 0.5
        if "light_deg" in seg:
            ang = math.radians(seg["light_deg"])
        else:
            # light from the side with more free space, 35 deg above horizontal
            side = 1.0 if cx < 0.5 else -1.0
            ang = math.atan2(-0.7, side)          # screen coords: y down, so -0.7 = up
        lx, ly = math.cos(ang), math.sin(ang)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # frame falloff: 1 on the light side -> ~0 on the far side
        proj = ((xx / W - 0.5) * lx + (yy / H - 0.5) * ly) * 1.6
        L = np.clip(0.5 + proj, 0, 1) ** 1.2
        # direction of each pixel from the character centre
        dx, dy = xx / W - cx, yy / H - cy
        norm = np.sqrt(dx * dx + dy * dy) + 1e-6
        cosang = (dx * lx + dy * ly) / norm
        dirw = 0.15 + 0.85 * np.clip(cosang, 0, 1) ** 1.3
        keyw = np.clip(0.5 + (dx * lx + dy * ly) * 2.2, 0, 1)
        self._cache[key] = (L.astype(np.float32), dirw.astype(np.float32), keyw.astype(np.float32))
        return self._cache[key]

    def look_cyan(self, slot, u, seg, k):
        """Cut-out character lit cyan over a crushed dark background."""
        ov = seg.get("overscan", 1.1)
        pl = self.placed(slot, ov)
        g = luma(pl.img) / 255
        # flatten the printed screentone first: amplified by the grade it
        # turned into busy speckle instead of smooth light
        g = cv2.bilateralFilter(g.astype(np.float32), 0, 0.08, self.st.get("tone_smooth", 2.5) * self.W / 1080)
        m = self.mask(slot, ov)
        bg_slot = seg.get("bg")
        if bg_slot:
            bgl = luma(self.placed(bg_slot, ov).img) / 255
        else:
            bgl = g
        bgl = cv2.GaussianBlur(bgl, (0, 0), self.st["bg_blur"] * self.W / 1080)
        bg = (bgl ** self.st["bg_gamma"]) * self.st["bg_level"]
        # character: paper -> bright cyan, ink stays dark; a little extra contrast
        fg = np.clip((g ** self.st["fg_gamma"]) * self.st["fg_level"], 0, 1)
        # key light from above on the face; the body falls into shadow (the
        # reference characters are lit from the head down, not evenly)
        yy0 = np.linspace(0, 1, self.H, dtype=np.float32)[:, None]
        fy = self.focus.get(slot, (0.5, 0.40))[1] if isinstance(self.focus.get(slot), (list, tuple)) else 0.40
        fg *= np.clip(1.15 - np.clip(yy0 - (fy - 0.05), 0, 1) * self.st["body_falloff"], 0.25, 1.0)
        # rim light: cyan glow along the cutout edge, outside the character
        edge = cv2.GaussianBlur(m, (0, 0), self.st["rim_blur"] * self.W / 1080)
        rim = np.clip((edge - m) * 2.2, 0, 1) * self.st["rim_level"]
        out = m * fg + (1 - m) * bg + rim
        # soft cyan glow bloom from the bright parts of the character (the
        # reference drop frames glow; a flat cutout looks pasted on)
        bloom = cv2.GaussianBlur(np.clip(fg * m - 0.45, 0, 1), (0, 0), self.st.get("bloom_blur", 24) * self.W / 1080)
        out = out + bloom * self.st.get("bloom_level", 0.9)
        # vignette (the reference frames fall to near-black at the edges)
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        r = np.hypot((xx / self.W - 0.5) * 1.1, (yy / self.H - 0.45))
        out *= np.clip(1.25 - r * self.st["vignette"], 0.15, 1.0)
        return out, m

    def finish_cyan(self, out, lut="cyan"):
        u8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return apply_lut(u8, self.luts[lut]).astype(np.float32)

    # ---- motion ----
    def motion(self, seg, k):
        s0, s1 = seg["start"], seg["end"]
        u = (k - s0) / max(1, s1 - s0)
        mo = seg.get("motion", {})
        typ = mo.get("type", "shake")
        scale, anchor, dx, dy, rot = 1.0, mo.get("anchor", (0.5, 0.45)), 0.0, 0.0, 0.0
        if typ == "curve":
            # the reference's measured camera path for this shot (zoom +
            # drift), normalised so the smallest scale is 1.0
            c = mo["curve"]
            j = min(max(k - s0, 0), len(c) - 1)
            scale = c[j] / min(c)
            if "tx" in mo and not mo.get("no_drift"):
                lim = self.st.get("drift_lim", 0.12)
                dx = float(np.clip(mo["tx"][j] - mo["tx"][0], -lim * 1080, lim * 1080)) * self.W / 1080
                dy = float(np.clip(mo["ty"][j] - mo["ty"][0], -lim * 1350, lim * 1350)) * self.H / 1356
            # ONE constant overscan per shot so drift + shake never expose an
            # edge (a per-frame boost changed the zoom curve every frame)
            scale *= self._cover(seg)
        elif typ == "zoom_in":
            scale = mo["to"] ** ease_in(u, mo.get("power", 1.0))
        elif typ == "zoom_in_accel":
            scale = mo["to"] ** (u ** mo.get("power", 2.2))
        elif typ == "zoom_out":
            scale = mo["from"] ** (1 - ease_out(u, mo.get("power", 1.6)))
        elif typ == "pan":
            dx = mo.get("dx", 0.0) * self.W * ease_out(u, 1.3)
            dy = mo.get("dy", 0.0) * self.H * ease_out(u, 1.3)
            scale = mo.get("zoom", 1.0) ** u
        elif typ == "slide_in":
            sf = mo.get("frames", 6)
            p = ease_out(min(1.0, (k - s0) / sf), 3.0)
            dx = (1 - p) * mo.get("from_dx", 0.9) * self.W
            scale = mo.get("zoom", 1.0) ** max(0.0, (k - s0 - sf) / max(1, s1 - s0 - sf))
        # handheld shake measured on the reference (per section): a per-frame
        # AR(1) jitter in x/y/rotation/zoom - it changes direction almost
        # every frame, which is what makes the reference camera feel alive
        sh = self.st.get("shake_stats", {}).get(seg.get("look", "cyan").replace("cyan_strip", "cyan"))
        env, punch = self._hit_env(s0, k)
        # zoom punch on each drum hit inside the shot: fast rise, settles back to 0
        scale *= math.exp(punch * self.st.get("hit_punch", 0.0))
        if sh:
            jit = self._shake_seq(seg)[min(max(k - s0, 0), s1 - s0)]
            g = mo.get("shake_gain", 1.0) * self.st.get("shake_gain", 1.0)
            # beat-synced shake: calm between hits, spikes on them
            if self.st.get("hit_shake"):
                lo, hi = self.st["hit_shake"]
                g *= lo + hi * env
            dx += jit[0] * sh["x"] * g * self.W / 1080
            dy += jit[1] * sh["y"] * g * self.H / 1356
            rot += jit[2] * sh["rot"] * g
            scale *= math.exp(jit[3] * sh["zoom"] * g)
        return scale, anchor, rot, dx, dy

    def _drift_mean(self, seg):
        """Average camera drift (px) over the second half of a shot, so the
        crop can pre-compensate it and the character sits centred on average."""
        n = seg["end"] - seg["start"]
        d = [self.motion(seg, seg["start"] + j)[3:5] for j in range(n // 2, n + 1)]
        mo = seg.get("motion", {})
        sc = (mo["curve"][-1] / min(mo["curve"])) if mo.get("type") == "curve" else 1.0
        return -float(np.mean([a for a, _ in d])) / sc, -float(np.mean([b for _, b in d])) / sc

    def _hit_env(self, s0, k, tau=6.0):
        """(shake envelope, zoom punch) at frame k from the song's drum hits
        (template 'hits' = [frame, strength]). Hits from 2 frames before the
        shot on count; each decays with time constant tau frames."""
        env = punch = 0.0
        for f, s in self.t.get("hits", []):
            d = k - f
            if d < 0 or f < s0 - 2 or d > 5 * tau:
                continue
            env += s * math.exp(-d / tau)
            punch += s * (1 - math.exp(-(d + 1) / 1.3)) * math.exp(-d / (tau + 1))
        return min(env, 1.5), punch

    def _cover(self, seg):
        """Constant per-shot overscan: the largest drift+shake offset in the
        shot, relative to the zoom available at that frame."""
        key = ("cover", seg["start"])
        if key in self._cache:
            return self._cache[key]
        mo = seg.get("motion", {})
        need = 1.0
        if mo.get("type") == "curve":
            c = mo["curve"]
            sh = self.st.get("shake_stats", {}).get(seg.get("look", "cyan").replace("cyan_strip", "cyan"), {})
            g = mo.get("shake_gain", 1.0) * self.st.get("shake_gain", 1.0)
            if self.st.get("hit_shake"):
                g *= self.st["hit_shake"][0] + self.st["hit_shake"][1] * 1.5    # worst case on a hit
            seq = self._shake_seq(seg)
            for j in range(len(c)):
                dx = dy = 0.0
                if "tx" in mo and not mo.get("no_drift"):
                    lim = self.st.get("drift_lim", 0.12)
                    dx = abs(np.clip(mo["tx"][j] - mo["tx"][0], -lim * 1080, lim * 1080)) / 1080
                    dy = abs(np.clip(mo["ty"][j] - mo["ty"][0], -lim * 1350, lim * 1350)) / 1356
                if sh and j < len(seq):
                    dx += abs(seq[j][0]) * sh["x"] * g / 1080
                    dy += abs(seq[j][1]) * sh["y"] * g / 1356
                s = c[j] / min(c)
                need = max(need, (1 + 2.05 * max(dx, dy)) / s)
        self._cache[key] = need
        return need

    def _shake_seq(self, seg, rho=0.6):
        """Unit-variance AR(1) sequences (x, y, rot, zoom) for a shot, seeded
        per shot so re-renders are identical. rho=0.6 reproduces the
        reference's frame-to-frame velocity correlation (~ -0.2)."""
        key = ("shake", seg["start"])
        if key not in self._cache:
            n = seg["end"] - seg["start"] + 1
            rng = np.random.default_rng(seg["start"] * 13 + 5)
            e = rng.standard_normal((n, 4))
            x = np.zeros((n, 4))
            x[0] = e[0]
            for i in range(1, n):
                x[i] = rho * x[i - 1] + math.sqrt(1 - rho ** 2) * e[i]
            self._cache[key] = x
        return self._cache[key]

    # ---- frame builders ----
    def shot_frame(self, seg, k, apply_motion=True):
        look = seg["look"]
        slot = seg["slot"]
        rng = np.random.default_rng(k * 7 + 3)
        u = (k - seg["start"]) / max(1, seg["end"] - seg["start"])
        if look == "bw":
            img = self.look_bw(slot, u, seg).astype(np.float32)
            if apply_motion:
                img = warp(img, *self._mo(seg, k))
            return img
        if look in ("cyan", "cyan_strip") and self.st.get("aura_mode", False):
            rgb, m = self.look_cyan_aura(slot, seg, k)
            sc, an, rot, dx, dy = self._mo(seg, k) if apply_motion else (1, (0.5, 0.5), 0, 0, 0)
            rgb = warp(rgb, sc, an, rot, dx, dy)
            m2 = warp(m, sc, an, rot, dx, dy)
            fx = seg.get("fx", [])
            cyan = np.array(self.st["aura_rgb"], np.float32) / 255
            if "lightning" in fx and (k // 2) % 5 != 4:
                ys, xs = np.nonzero(cv2.Canny((m2 > 0.5).astype(np.uint8) * 255, 50, 150)[::8, ::8])
                rng = np.random.default_rng((seg["start"], k // 2))
                g_, c_ = lightning_layer(self.W, self.H, rng, list(zip(xs * 8, ys * 8)), n=seg.get("bolts", 4))
                col = cyan if rng.random() > 0.25 else np.array([1.0, 0.45, 0.75], np.float32)   # occasional pink bolt
                rgb = rgb + (g_ * 0.5)[..., None] * col + (c_ * 0.95)[..., None] * (0.6 + 0.4 * col)
            if "pillars" in fx:
                rgb = rgb + self._pillars(seg, k)[..., None] * cyan
            if "speedlines" in fx:
                rgb = rgb + (speed_lines_layer(self.W, self.H, np.random.default_rng((seg["start"], k // 2))) * 0.6)[..., None] * cyan
            img = np.clip(rgb, 0, 1.2) * 255
            if look == "cyan_strip":
                img = self._strip(img, seg, k, fg=m2 if self.st.get("strip_behind") else None)
            return self.match_section(img, "cyan", seg.get("lut", "cyan"), strength=1.0)
        if look in ("cyan", "cyan_strip"):
            out, m = self.look_cyan(slot, u, seg, k)
            sc, an, rot, dx, dy = self._mo(seg, k) if apply_motion else (1, (0.5, 0.5), 0, 0, 0)
            out = warp(out, sc, an, rot, dx, dy)
            m2 = warp(m, sc, an, rot, dx, dy)
            fx = seg.get("fx", [])
            if "lightning" in fx and (k // 2) % 5 != 4:        # bolts re-strike every 2 frames, brief gaps
                ys, xs = np.nonzero(cv2.Canny((m2 > 0.5).astype(np.uint8) * 255, 50, 150)[::8, ::8])
                anchors = list(zip(xs * 8, ys * 8))
                g, c = lightning_layer(self.W, self.H, np.random.default_rng((seg["start"], k // 2)),
                                       anchors, n=seg.get("bolts", 4))
                out = out + g * 0.55 + c * 0.9
            if "speedlines" in fx:
                out = out + speed_lines_layer(self.W, self.H, np.random.default_rng((seg["start"], k // 2))) * 0.5
            img = self.finish_cyan(out, seg.get("lut", "cyan"))
            if look == "cyan_strip":
                img = self._strip(img, seg, k, fg=m2 if self.st.get("strip_behind") else None)
            return img
        if look == "eyes":
            return self.eyes_frame(seg, k, apply_motion)
        raise ValueError(look)

    def match_section(self, rgb255, section, lut, strength=0.85):
        """Force the reference's look: histogram-match this frame's brightness
        to the reference section's distribution, then colour every brightness
        level with the reference's measured curve (blue-black shadows, teal
        mids, saturated cyan highlights - never neutral grey or blown white)."""
        tgt = self.t.get("luma_target", {}).get(section)
        L = np.clip(rgb255 @ np.array([0.299, 0.587, 0.114], np.float32), 0, 255)
        if tgt is not None:
            tgt = np.asarray(tgt, np.float32)
            hist = np.bincount(L.astype(np.uint8).ravel(), minlength=256).astype(np.float32)
            cdf = np.cumsum(hist) / hist.sum()
            mapping = np.interp(cdf, tgt, np.arange(256, dtype=np.float32))
            L = L * (1 - strength) + mapping[L.astype(np.uint8)] * strength
        return apply_lut(np.clip(L, 0, 255).astype(np.uint8), self.luts[lut]).astype(np.float32)

    def _pillars(self, seg, k):
        """Vertical cyan light streaks behind the character. Each set of
        streaks crossfades into the next over 6 frames (a hard re-roll
        every few frames read as a jerk in QC)."""
        period = 6
        i, f = divmod(k, period)
        a = self._pillar_set(seg, i)
        b = self._pillar_set(seg, i + 1)
        t = f / period
        t = t * t * (3 - 2 * t)
        return a * (1 - t) + b * t

    def _pillar_set(self, seg, i):
        key = ("pillars", seg["start"], i)
        if key in self._cache:
            return self._cache[key]
        rng = np.random.default_rng((seg["start"], i))
        lay = np.zeros((self.H, self.W), np.float32)
        for _ in range(int(rng.integers(5, 9))):
            x = int(rng.uniform(0.02, 0.98) * self.W)
            w = max(2, int(rng.uniform(4, 22) * self.W / 1080))
            y0 = int(rng.uniform(0, 0.4) * self.H)
            lay[y0:, max(0, x - w // 2):x + w // 2] = rng.uniform(0.5, 1.0)
        lay = cv2.GaussianBlur(lay, (0, 0), 3 * self.W / 1080)
        out = lay * 0.85 + cv2.GaussianBlur(lay, (0, 0), 20 * self.W / 1080) * 0.6
        self._cache[key] = out
        return out

    def _mo(self, seg, k):
        sc, an, rot, dx, dy = self.motion(seg, k)
        return sc, an, rot, dx, dy

    def _strip_pos(self, st, d):
        """Horizontal offset (fraction of W, 0 = in place) of a face strip d
        frames after its shot starts. Reference: the strip SLAMS in - it is
        fully across within ~3 frames (accelerating, so the stop is hard),
        overshoots a little, snaps back, then only creeps."""
        T = st.get("slam_frames", 3.0)
        side = st.get("from_dx", -1.0)
        if d < T:
            p = (d / T) ** 1.6                                  # accelerate into the stop
            return side * (1 - p)
        e = d - T
        over = -side * st.get("overshoot", 0.035) * math.exp(-e / 2.2) * math.cos(e * 1.1)
        creep = -side * st.get("creep", 0.03) * (1 - math.exp(-e / 40))
        return over + creep

    def _strip(self, img, seg, k, fg=None):
        """A horizontal band showing another panel (the face strip) slammed
        in behind the character (fg = character mask drawn back on top)."""
        st = seg["strip"]
        pl = self.placed(st["slot"], 1.0)
        H, W = self.H, self.W
        d = k - seg["start"]
        y0, y1 = int(st["y"][0] * H), int(st["y"][1] * H)
        band_h = y1 - y0
        src = luma(pl.img)
        # the strip must show the EYES: centre the band on the eye line
        eyes = self.eye_points(st["slot"], 1.0)
        ex = W / 2
        if eyes:
            cy = int(np.mean([p[1] for p in eyes]))
            ex = float(np.mean([p[0] for p in eyes]))
        else:
            cy = int(st.get("src_y", 0.35) * H)
        cy = min(max(cy, band_h // 2), H - band_h // 2)
        band = src[cy - band_h // 2:cy - band_h // 2 + band_h].astype(np.float32)
        if band.shape[0] < band_h:
            return img
        g = ((band / 255) ** 1.8 * 0.5).clip(0, 1)
        col = self.finish_cyan(g)
        if eyes:                                                # glowing eyes, like the reference's strip faces
            yy, xx = np.mgrid[0:band_h, 0:W].astype(np.float32)
            er = self.st["eye_radius"] * W * 0.8
            eg = np.zeros((band_h, W), np.float32)
            for px, py in eyes:
                d2 = ((xx - px) ** 2 + (yy - (py - (cy - band_h // 2))) ** 2) / er ** 2
                eg = np.maximum(eg, np.exp(-d2 * 3.0) + 0.3 * np.exp(-d2 * 0.4))
            col = col + np.array(self.st["eye_rgb"], np.float32) * eg[..., None] + 255 * (eg ** 3)[..., None] * 0.4
        col = np.clip(col, 0, 255).astype(np.float32)
        col[:2] = 230
        col[-2:] = 230
        # position: eyes centred, plus the slam offset (sub-pixel, no wrap-around)
        pos = self._strip_pos(st, d)
        x = (W / 2 - ex) + pos * W
        v = abs(pos - self._strip_pos(st, d - 1)) * W if d > 0 else 0.0
        # impact: band flashes and swells slightly, whole frame jolts
        T = st.get("slam_frames", 3.0)
        hit = math.exp(-(d - T) / 2.0) if d >= T else 0.0
        sy = 1 + 0.10 * hit
        M = np.float32([[1, 0, x], [0, sy, (1 - sy) * band_h / 2]])
        alpha = cv2.warpAffine(np.ones((band_h, W), np.float32), M, (W, band_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        col = cv2.warpAffine(col, M, (W, band_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if v > 2:                                               # motion blur = distance moved this frame
            kw = int(v * 0.9) | 1
            col = cv2.blur(col, (kw, 1))
            alpha = cv2.blur(alpha, (kw, 1))
        col = col * (1 + 0.35 * hit) + 22 * hit
        out = img.copy()
        a = alpha[..., None]
        out[y0:y1] = out[y0:y1] * (1 - a) + col * a
        if fg is not None:                                      # character stays IN FRONT of the strip
            f = np.clip(fg[y0:y1], 0, 1)[..., None]
            out[y0:y1] = out[y0:y1] * (1 - f) + img[y0:y1] * f
        if hit > 0.05:
            jolt = st.get("jolt", 9.0) * self.H / 1356 * hit * math.cos((d - T) * 1.9)
            out = cv2.warpAffine(out, np.float32([[1, 0, 0], [0, 1, jolt]]), (W, H),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out

    def eyes_frame(self, seg, k, apply_motion=True):
        slot = seg["slot"]
        ov = seg.get("overscan", 1.15)
        pl = self.placed(slot, ov)
        g = luma(pl.img) / 255
        g = cv2.bilateralFilter(g.astype(np.float32), 0, 0.08, self.st.get("tone_smooth", 2.5) * self.W / 1080)
        base = (g ** self.st["eyes_gamma"]) * self.st["eyes_level"]
        eyes = self.eye_points(slot, ov)
        glow = np.zeros_like(base)
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        er = self.st["eye_radius"] * self.W
        for ex, ey in eyes:
            d2 = ((xx - ex) ** 2 + (yy - ey) ** 2) / er ** 2
            glow = np.maximum(glow, np.exp(-d2 * 3.0) + 0.35 * np.exp(-d2 * 0.35))
        glow = glow + cv2.GaussianBlur(glow, (0, 0), self.st.get("bloom_wide", 70) * self.W / 1080) * 0.8   # wide soft pool
        # fade: panel dims toward black at the segment end, the eyes linger
        u = (k - seg["start"]) / max(1, seg["end"] - seg["start"])
        fade = seg.get("fade", [0.0, 1.0])
        amt = 1 - smoothstep((u - fade[0]) / max(1e-3, fade[1] - fade[0])) * seg.get("fade_to", 0.25)
        u8 = (np.clip(base, 0, 1) * 255).astype(np.uint8)
        img = apply_lut(u8, self.luts[seg.get("lut", "cyan_dark")]).astype(np.float32)
        eyecol = np.array(self.st["eye_rgb"], np.float32)
        gl = np.clip(glow, 0, 1)[..., None]
        img = img * (1 - gl * 0.85) + eyecol * gl + 255 * (gl ** 3) * 0.5
        # reference look for this section (tones + colour), THEN the fade -
        # matching after the fade would undo it
        img = self.match_section(img, "eyes", seg.get("lut", "cyan_dark"))
        img = img * amt + (eyecol * gl + 255 * (gl ** 3) * 0.5) * (1 - amt) * 0.9   # eyes linger in the dark
        if apply_motion:
            img = warp(img, *self._mo(seg, k))
        return img

    # ---- transitions ----
    def smear_frame(self, seg, k):
        """Horizontal tear/smear: rows shift sideways in bands and blur, the
        picture flares to a peak, then the next shot re-forms."""
        s0, s1 = seg["start"], seg["end"]
        L = s1 - s0 + 1
        j = k - s0
        peak = seg.get("peak", int(round(L * 0.36)))
        prev_seg, next_seg = self.neighbours(seg)
        # both shots stay IN MOTION through the transition (a frozen frame
        # torn apart reads as a stall): the outgoing one keeps pushing in,
        # the incoming one settles from a slight overscale
        if prev_seg:
            bf = set(prev_seg.get("black_frames", []))
            last = next((f for f in range(prev_seg["end"], prev_seg["start"] - 1, -1) if f not in bf), prev_seg["end"])
            a_img = warp(self.render_seg(prev_seg, last, apply_motion=True), 1 + 0.012 * (j + 1))
        else:
            a_img = self.black()
        if next_seg:
            b_img = warp(self.render_seg(next_seg, next_seg["start"], apply_motion=True),
                         1 + 0.06 * (1 - ease_out((j + 1) / L, 2.0)))
        else:
            b_img = self.black()
        # the tear rises to the peak and resolves after it (continuous in j)
        strength = ease_in((j + 1) / (peak + 1), 1.4) if j < peak else 1 - ease_out((j - peak) / max(1, L - peak), 1.6)
        if next_seg is None or prev_seg is None:
            strength = max(strength, 0.3)
        mix = smoothstep((j + 0.5 - peak + 1.5) / 3.0)          # swap shots across ~3 frames around the peak
        src = a_img * (1 - mix) + b_img * mix
        H, W = self.H, self.W
        out = src.copy()
        # row-band displacement: ONE band pattern per transition that drifts
        # smoothly frame to frame (re-randomising it every frame was jitter)
        key = ("smearbands", s0)
        if key not in self._cache:
            rng = np.random.default_rng(s0)
            prof = np.zeros(H * 2, np.float32)
            y = 0
            while y < len(prof):
                bh = int(rng.uniform(4, 36) * H / 1350)
                prof[y:y + bh] = rng.normal(0, 1)
                y += bh
            self._cache[key] = (prof, float(rng.choice([-1, 1])))
        prof, direction = self._cache[key]
        drift = int(j * self.st.get("smear_crawl", 3) * H / 1350)                                # bands crawl vertically
        rows = prof[drift:drift + H]
        slide = direction * strength * 0.06 * W                      # the whole picture slides one way
        for y in range(0, H, 2):
            sh = int(rows[y] * 0.16 * W * strength + slide)
            out[y:y + 2] = np.roll(out[y:y + 2], sh, axis=1)
        # horizontal blur, then contrast stretch (the reference smears are
        # hard bright/dark streaks, not a flat wash)
        kw = int(1 + strength * self.st["smear_blur"] * W) | 1
        out = cv2.blur(out, (kw, 1))
        c = 1 + strength * self.st.get("smear_contrast", 0.8)
        out = (out - 110) * c + 110 + strength * self.st.get("smear_lift", 40)
        # flare
        flare = strength ** 2 * seg.get("flare", self.st["smear_flare"])
        tint = np.array(seg.get("tint_rgb", self.st["smear_tint"]), np.float32)
        out = out * (1 - flare) + tint * flare
        if seg.get("white_peak") and j == peak:
            out = out * 0.12 + 245 * 0.88
        # match the reference's per-frame brightness of this transition
        prof_l = seg.get("lum_profile")
        if prof_l and j < len(prof_l):
            cur = float((out @ np.array([0.299, 0.587, 0.114], np.float32)).mean())
            target = prof_l[j]
            if target > cur:     # lift toward white (keeps streak contrast)
                a = min(0.95, (target - cur) / max(1.0, 250 - cur))
                out = out * (1 - a) + 250 * a
            else:
                out = out * (target / max(cur, 1.0))
        return out

    def neighbours(self, seg):
        i = self.timeline.index(seg)
        prev_seg = next((s for s in reversed(self.timeline[:i]) if s["type"] == "shot"), None)
        next_seg = next((s for s in self.timeline[i + 1:] if s["type"] == "shot"), None)
        if prev_seg is not None and prev_seg["end"] < seg["start"] - 2:
            prev_seg = None            # a black gap sits in between
        if next_seg is not None and next_seg["start"] > seg["end"] + 2:
            next_seg = None
        return prev_seg, next_seg

    def black(self):
        return np.ones((self.H, self.W, 3), np.float32) * self.dark

    def white_frame(self, seg, k):
        """Radial white flash (bright centre, soft grey edges), then fades
        into the next shot over the last frames."""
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        r = np.hypot((xx / self.W - 0.5), (yy / self.H - 0.5) * 0.8)
        white = (255 - np.clip(r - 0.25, 0, 1) * 160)[..., None] * np.ones(3, np.float32)
        fade_in = seg.get("fade_into_next", 0)
        j = k - seg["start"]
        L = seg["end"] - seg["start"] + 1
        if fade_in and j >= L - fade_in:
            _, nxt = self.neighbours(seg)
            if nxt:
                a = (j - (L - fade_in) + 1) / (fade_in + 1)
                return white * (1 - a) + self.render_seg(nxt, nxt["start"]) * a
        return white

    def strobe_black(self, seg, k):
        return k in set(seg.get("black_frames", []))

    # ---- outro ----
    def logo_frame(self, seg, k):
        H, W = self.H, self.W
        j = k - seg["start"]
        pattern = seg["pattern"]             # per-frame brightness 0..1 of the logo (ref p99)
        b = pattern[j] if j < len(pattern) else 0.0
        img = self.black()
        flash = seg.get("flash", [])
        fl = flash[j] if j < len(flash) else 0.0
        if fl > 0.05:                        # full-frame radial white flash (ref 1066-1072)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.hypot(xx / W - 0.5, (yy / H - 0.5) * 0.8)
            white = (255 - np.clip(r - 0.2, 0, 1) * 200)[..., None] * np.ones(3, np.float32)
            img = img * (1 - fl) + white * fl
        if b <= 0.02:
            return img
        f = ImageFont.truetype(r"C:\Windows\Fonts\bahnschrift.ttf", int(H * 0.115))
        try:
            f.set_variation_by_name("Bold Condensed")
        except (OSError, ValueError):
            pass
        lay = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(lay)
        big_until = seg.get("big_mark_until", 0)
        if j < big_until:
            # first beat: a huge jagged initial letter tearing across the frame
            fb = ImageFont.truetype(r"C:\Windows\Fonts\ariblk.ttf", int(H * 0.62))
            d.text((W / 2, H * 0.45), self.logo_text[:1], font=fb, fill=255, anchor="mm")
            d.text((W / 2, H * 0.72), self.logo_text, font=f, fill=200, anchor="mm")
        else:
            d.text((W / 2, H * 0.52), self.logo_text, font=f, fill=255, anchor="mm")
        a = np.asarray(lay).astype(np.float32) / 255
        rng = np.random.default_rng(k)
        if j < big_until:
            # jagged tear: diagonal lightning-shaped cuts through the big letter
            for _ in range(3):
                x0 = rng.uniform(0.2, 0.8) * W
                pts = [(x0 + rng.uniform(-0.12, 0.12) * W, H * t) for t in np.linspace(0.1, 0.8, 7)]
                cut = Image.new("L", (W, H), 0)
                ImageDraw.Draw(cut).line(pts, fill=255, width=int(rng.uniform(0.02, 0.05) * W))
                a = a * (1 - np.asarray(cut).astype(np.float32) / 255)
        if seg.get("glitch", True) and b < 0.95:
            y = 0
            while y < H:
                bh = int(rng.uniform(3, 20))
                a[y:y + bh] = np.roll(a[y:y + bh], int(rng.normal(0, 0.03) * W), axis=1)
                y += bh
        glow = cv2.GaussianBlur(a, (0, 0), 6)
        lum = (a * 250 + glow * 60) * b
        # chromatic fringe: red channel offset (the reference logo bleeds red)
        red = np.roll(lum, int(0.006 * W), axis=1)
        img = img + np.stack([np.maximum(lum, red * 1.1), lum * 0.96, lum * 0.96], -1)
        if rng.random() < 0.5:                  # thin red glitch line like the reference
            yl = int(H * rng.uniform(0.45, 0.6))
            img[yl:yl + 2, :, 0] += 150 * b
        return img

    # ---- dispatch ----
    def seg_at(self, k):
        for s in self.timeline:
            if s["start"] <= k <= s["end"]:
                return s
        return None

    def render_seg(self, seg, k, apply_motion=True):
        key = ("frame", id(seg), k, apply_motion)
        if key in self._cache:
            return self._cache[key]
        typ = seg["type"]
        if typ == "shot":
            if self.strobe_black(seg, k):
                img = self.black()
            else:
                img = self.shot_frame(seg, k, apply_motion)
        elif typ == "smear":
            img = self.smear_frame(seg, k)
        elif typ == "white":
            img = self.white_frame(seg, k)
        elif typ == "logo":
            img = self.logo_frame(seg, k)
        else:
            img = self.black()
        if len([c for c in self._cache if c[0] == "frame"]) > 12:
            for c in [c for c in self._cache if c[0] == "frame"][:6]:
                del self._cache[c]
        self._cache[key] = img
        return img

    def _watermark(self, text):
        if not text:
            return None
        W, H = self.W, self.H
        f = ImageFont.truetype(r"C:\Windows\Fonts\bahnschrift.ttf", int(H * 0.018))
        lay = Image.new("L", (W, H), 0)
        ImageDraw.Draw(lay).text((W / 2, H * 0.957), " ".join(text), font=f, fill=210, anchor="mm")
        return np.asarray(lay).astype(np.float32)[..., None] / 255

    def frame(self, k):
        seg = self.seg_at(k)
        img = self.render_seg(seg, k) if seg else self.black()
        if seg is not None and seg["type"] == "shot" and not self.strobe_black(seg, k):
            # exposure flicker (measured: the reference's brightness pulses
            # frame to frame; a rock-steady exposure reads as static)
            fl = self._shake_seq(seg, rho=0.3)[min(k - seg["start"], seg["end"] - seg["start"])][3]
            img = img * math.exp(fl * self.st.get("exposure_flicker", 0.0))
        if self.wm is not None and seg is not None and seg["type"] != "logo" and img.mean() > 12:
            img = img * (1 - self.wm * 0.85) + 225 * self.wm * 0.85
        # film grain (fine, subtle)
        rng = np.random.default_rng(k)
        img = img + rng.normal(0, self.st["grain"], (self.H // 2, self.W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)
        return np.clip(img, 0, 255).astype(np.uint8)

    def render(self, out_path, log=print):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".video.mp4")
        wr = imageio_ffmpeg.write_frames(str(tmp), (self.W, self.H), fps=self.fps, codec="libx264",
                                         pix_fmt_out="yuv420p", output_params=["-crf", "16", "-preset", "medium"],
                                         macro_block_size=2)
        wr.send(None)
        for k in range(self.n):
            wr.send(np.ascontiguousarray(self.frame(k)))
            if k % 120 == 0:
                log(f"  frame {k}/{self.n}")
        wr.close()
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp),
                        "-i", str(ROOT / self.t["music"]), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "256k", "-t", f"{self.n / self.fps:.3f}", str(out_path)], check=True)
        tmp.unlink()
        return out_path


def render_manifest(manifest_path, preview=False, out=None):
    p = Path(manifest_path)
    m = json.loads((p if p.is_absolute() else ROOT / p).read_text(encoding="utf-8-sig"))
    out = out or ROOT / "output" / f"{m['id']}.mp4"
    return WekisavaEdit(m, preview).render(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(render_manifest(sys.argv[1], preview="--preview" in sys.argv))











