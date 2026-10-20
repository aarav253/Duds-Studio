"""Frame-exact renderer for edit templates (templates/*.json, built from a
reference edit by template_builder.py).

A manifest supplies only the content - images + caption - and the template
supplies everything else: music, every cut frame, shot kinds, zoom curves,
transition sequence, intro/outro, grade and film texture.

    python src/template_edit.py edits/<manifest>.json [--preview]

Manifest:
{
  "id": "my_edit", "type": "template_edit",
  "template": "templates/valtrez_kaiser.json",
  "caption": "POV : you ascended so hard that even your old friends don't recognise you.",
  "intro_image": "panels/Kaiser/intro.png",
  "shots": ["panels/Kaiser/01.png", "panels/Kaiser/02.png", "panels/Kaiser/03.png",
            {"image": "panels/Kaiser/page.png", "card": "panels/Kaiser/04.png"}, ...],
  "watermark": "DUDS", "end_text": "DUDS",
  "focus": {"3": [0.5, 0.3]}          // optional per-shot zoom focus override
}

Why a custom frame loop instead of MoviePy layers: the reference's impact
sequence is 9 individually different frames at 60fps (two halftone frames, a
dark halftone flash, a white burst, two dark dips, three resolving frames,
then a sharp landing that hits the beat). Rendering frame-by-frame makes every
one of those land on the exact frame the template says.

All timing constants below were measured on the reference edits (see
references/template_N/analysis and the notes in each template).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
FONTS = Path(r"C:\Windows\Fonts")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Measured style constants (Valtrez preset). A template's "style" block can
# override any of these.
# ---------------------------------------------------------------------------
STYLE = {
    "dark_bg": [14, 17, 23],
    # transition, frames relative to the dither start d (measured luma ratios:
    # d0 .84/.79, d1 .82/.72, d2 .26/.60, d3 1.16/1.18, d4-5 .47/.89, d6-8 ~1.0/1.12)
    "halftone_cell": 22,                  # (legacy screen, unused)
    "halftone_pitch": 17.6,               # px at 1080 wide - measured by FFT, identical on every ref halftone frame
    "flash_keep": 0.22,                   # fraction of dots surviving on the d+2 dark flash
    "dip_cap": 0.17,                      # max mean brightness of the d+4/d+5 dips (0-1, before grade)
    "halftone_gain": [1.0, 0.97, 0.55],   # brightness of the dot screen on d0, d1, d2
    "flash_gamma": 4.0,                   # d2 dark flash: higher = only the brightest dots survive
    "strobe": [1.12, 0.5, 0.52, 1.04, 1.02, 1.01],   # exposure of the incoming shot on d3..d8
    "strobe_blur": [0.42, 0.34, 0.27, 0.16, 0.085, 0.055],  # radial zoom-blur spread on d3..d8
    "land_offset": 9,
    # per-shot zoom speed curves (normalized log-scale vs normalized time), from
    # both references (IN: fast start, long slow middle, faster finish)
    "in_curve": [[0, 0], [0.04, 0.08], [0.23, 0.365], [0.42, 0.505], [0.615, 0.61], [0.81, 0.74], [1, 1]],
    "out_curve": [[0, 0], [0.04, 0.075], [0.23, 0.335], [0.42, 0.56], [0.615, 0.69], [0.81, 0.795], [1, 1]],
    "motion_blur": 0.12,                   # radial blur per unit of per-frame log-zoom speed
    "motion_blur_tau": 3.0,                # frames: how fast the post-landing blur decays
                                           # (Kaiser ~0.12; Rin's heavier blur is set in its template)
    # card shots (tilted inset card over a slowly zooming background)
    "card_width": [0.44, 0.76],
    "card_growth": [[0, 0], [0.3, 0.22], [0.6, 0.47], [0.8, 0.70], [1, 1]],
    "card_center": [[0.46, 0.55], [0.50, 0.51]],
    "card_tilt_deg": 55,                   # rotateX at landing, eases to 0 by 35% of the shot
    "card_bg_zoom": 1.35,
    "card_bg_lift": 0.12,
    # intro card pop (one beat before the drop)
    "pop_width": [0.12, 0.62],
    # caption band (fractions of height) + text
    "caption_band": [0.559, 0.707],
    "caption_font": "GOTHICB.TTF",
    "caption_cap_px": 44,                  # font size for capitals at 1080 wide
    "caption_small_ratio": 0.78,           # lowercase drawn as small capitals
    "caption_tracking": 0.3,               # em
    "caption_wrap": 0.84,                  # max line width (fraction of W)
    "caption_color": [228, 230, 235],
    # grade
    "contrast": 1.12, "brightness": 8, "black_floor": 11, "shadow_blue": 6.0,
    "watermark_y": 0.933, "watermark_size": 34,
    "end_font": "bahnschrift.ttf", "end_font_variation": "Bold Condensed",
}


def _font(name, size, variation=None):
    for p in (ROOT / "fonts" / name, FONTS / name):
        if p.exists():
            f = ImageFont.truetype(str(p), int(size))
            if variation:
                try:
                    f.set_variation_by_name(variation)
                except (OSError, ValueError):
                    pass
            return f
    return ImageFont.truetype(str(FONTS / "arialbd.ttf"), int(size))


def _isotonic(y):
    """Best non-decreasing fit to y (pool-adjacent-violators)."""
    vals, wts, lens = [], [], []
    for v in y:
        vals.append(float(v)); wts.append(1.0); lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2], lens[-2] = w, lens[-2] + lens[-1]
            vals.pop(); wts.pop(); lens.pop()
    return np.repeat(vals, lens)


def ease_out_cubic(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_rgb(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return np.asarray(Image.open(p).convert("RGB"))


class ClipSource:
    """A video clip used in place of a still panel: frame j of the clip,
    cover-fitted to the edit size. Reads sequentially (cheap) and seeks only
    when asked for an earlier/later frame out of order."""

    def __init__(self, path, W, H, start=0):
        p = Path(path)
        self.path = p if p.is_absolute() else ROOT / p
        self.W, self.H, self.start = W, H, start
        cap = cv2.VideoCapture(str(self.path))
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.cap, self.pos, self.last = None, -1, {}

    def frame(self, j):
        j = int(min(max(j + self.start, 0), self.n - 1))
        if j in self.last:
            return self.last[j]
        if self.cap is None or j != self.pos + 1:
            if self.cap is None:
                self.cap = cv2.VideoCapture(str(self.path))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        ok, b = self.cap.read()
        if not ok:
            return self.last.get(max(self.last) if self.last else None, np.zeros((self.H, self.W, 3), np.uint8))
        self.pos = j
        img = cover(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), self.W, self.H, (0.5, 0.5))
        if len(self.last) > 4:
            self.last.pop(min(self.last))
        self.last[j] = img
        return img


def content_center(arr):
    g = cv2.cvtColor(cv2.resize(arr, (256, int(256 * arr.shape[0] / arr.shape[1]))), cv2.COLOR_RGB2GRAY)
    g = g.astype(np.float32) / 255
    ink = (1 - g) ** 1.5 + 2 * np.hypot(*np.gradient(g))
    ys, xs = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    tot = ink.sum() + 1e-6
    return 0.75 * (ink * xs).sum() / tot / g.shape[1] + 0.125, 0.75 * (ink * ys).sum() / tot / g.shape[0] + 0.125


def cover(arr, W, H, center=None):
    h, w = arr.shape[:2]
    s = max(W / w, H / h)
    nw, nh = max(W, int(math.ceil(w * s))), max(H, int(math.ceil(h * s)))
    img = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if center:
        cx, cy = center
    else:
        cx, cy = content_center(arr)
        # manga portraits: the face is usually in the upper part while the ink
        # mass (hair + jersey + chains) pulls the centroid down - bias upward
        # so a tall image cropped to 4:5 doesn't cut off the top of the head
        cy = 0.5 * cy + 0.5 * 0.36
    x0 = int(round(min(max(cx * nw - W / 2, 0), nw - W)))
    y0 = int(round(min(max(cy * nh - H / 2, 0), nh - H)))
    return img[y0:y0 + H, x0:x0 + W].copy()


def zoom_about(img, s, anchor):
    """Scale image by s about anchor (fractions), output same size, reflect edges."""
    H, W = img.shape[:2]
    ax, ay = anchor[0] * W, anchor[1] * H
    M = np.float32([[s, 0, ax - s * ax], [0, s, ay - s * ay]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def radial_blur(img, spread, anchor, samples=10, scale_down=2, _force_half=False):
    """Zoom blur: average of copies scaled 1..(1+spread) about anchor."""
    if _force_half:
        return _radial_blur_half(img, spread, anchor, samples, scale_down)
    if spread <= 0.002:
        return img
    H, W = img.shape[:2]
    lo, hi = 0.03, 0.06
    if spread < hi:
        # light motion blur at full resolution (downscaling would soften the
        # whole frame far more than the blur itself). Between lo and hi it
        # crossfades into the half-res path below: a hard switch between the
        # two made the blur visibly pop as it faded through the threshold.
        n = int(min(max(3, spread * max(W, H) * 0.6), 24))
        acc = np.zeros(img.shape, np.float32)
        for i in range(n):
            acc += zoom_about(img, 1 + spread * i / (n - 1), anchor)
        full = acc / n
        if spread <= lo:
            return full.clip(0, 255).astype(np.uint8)
        a = smoothstep((spread - lo) / (hi - lo))
        half = radial_blur(img, spread, anchor, samples, scale_down, _force_half=True).astype(np.float32)
        return (full * (1 - a) + half * a).clip(0, 255).astype(np.uint8)
    return _radial_blur_half(img, spread, anchor, samples, scale_down)


def _radial_blur_half(img, spread, anchor, samples, scale_down):
    H, W = img.shape[:2]
    small = cv2.resize(img, (W // scale_down, H // scale_down), interpolation=cv2.INTER_AREA).astype(np.float32)
    # pre-blur + enough copies so the step between copies is < ~1.5 px at the
    # frame edge: fewer, farther-apart copies of fine screentone interfere
    # into a criss-cross moire (seen on real manga panels in QC)
    small = cv2.GaussianBlur(small, (0, 0), 1.2 + spread * 6)
    edge = spread * max(small.shape[:2]) / 2
    samples = int(min(64, max(samples, edge / 1.5)))
    acc = np.zeros_like(small)
    for i in range(samples):
        acc += zoom_about(small, 1 + spread * i / (samples - 1), anchor)
    out = acc / samples
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR).clip(0, 255).astype(np.uint8)


def halftone_dots(img, pitch, gain=1.0, keep=1.0, floor=0.09, seed=0, blob=3.0):
    """The reference's dot screen, measured at full res: the picture is
    REPLACED by dots (no line art survives). Square lattice, pitch 17.6 px at
    1080 wide (identical on every halftone frame of both references). Each dot
    takes its cell's average brightness; bright dots grow until they touch.
    The gaps show a dark, blurred version of the image (gray, not black).
    keep<1 drops dots at random (the near-black 'dark flash' frame)."""
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gh, gw = max(1, int(round(H / pitch))), max(1, int(round(W / pitch)))
    cells = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
    cells = np.clip((cells - 0.5) * 1.4 + 0.55, 0, 1) * gain
    if keep < 1.0:
        # dark flash: keep dots in COHERENT PATCHES, chosen by a smooth field
        # (blurred brightness + slow blobby variation). Per-dot selection -
        # random, or "brightest" on flat white line art where every dot ties -
        # reads as TV static; QC measures this as neighbouring-dot agreement.
        rng = np.random.default_rng(seed)
        noise = cv2.GaussianBlur(rng.random(cells.shape).astype(np.float32), (0, 0), blob)
        noise = (noise - noise.mean()) / (noise.std() + 1e-6)
        field = cv2.GaussianBlur(cells, (0, 0), min(1.5, blob)) + 0.25 * noise
        thr = np.quantile(field, 1 - keep)
        cells = cells * (field >= thr) * 0.85
    py, px = H / gh, W / gw
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    iy = np.minimum((yy / py).astype(np.int32), gh - 1)
    ix = np.minimum((xx / px).astype(np.int32), gw - 1)
    b = cells[iy, ix]
    dy = yy - (iy + 0.5) * py
    dx = xx - (ix + 0.5) * px
    dist = np.sqrt(dx * dx + dy * dy)
    R = (0.5 * min(px, py)) * (0.35 + 0.72 * np.sqrt(b)) * (b > 0.03)
    a = np.clip(R - dist + 0.7, 0, 1)
    dot = np.clip(b * 1.25, 0, 1) * 255
    base = cv2.GaussianBlur(g, (0, 0), pitch * 0.8) * 255 * floor * (gain if keep >= 1 else 0.25)
    v = base * (1 - a) + dot * a
    return np.stack([v, v, v], -1).clip(0, 255).astype(np.uint8)


def halftone_white(img, cell, gain, gamma=1.0):
    """The reference's dot screen: the image seen through a grid of round
    holes on black. Hole radius grows with local brightness (so white areas
    read as big touching dots and ink stays crisp black inside them).
    gamma>1 + gain<1 gives the dark 'flash' version where only the brightest
    dots survive."""
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    ch, cw = -(-H // cell), -(-W // cell)
    gp = np.pad(g, ((0, ch * cell - H), (0, cw * cell - W)), mode="edge")
    blocks = gp.reshape(ch, cell, cw, cell).mean(axis=(1, 3))
    b = np.clip(blocks, 0, 1) ** gamma * gain
    R = cell * (0.21 + 0.29 * np.sqrt(b)) * (b > 0.04)
    yy, xx = np.mgrid[0:cell, 0:cell].astype(np.float32) + 0.5 - cell / 2
    dist = np.tile(np.hypot(yy, xx), (ch, cw))
    Rf = np.repeat(np.repeat(R, cell, 0), cell, 1)
    soft = max(1.0, cell * 0.22)                           # soft hole edge (reference dots aren't razor sharp)
    a = np.clip((Rf - dist) / soft + 0.5, 0, 1)[:H, :W]
    if gamma <= 1.5:
        a = 0.12 + 0.88 * a                                # gaps aren't pure black: the image shows faintly
                                                           # (the dark flash frame keeps pure-black gaps)
    lum = np.clip((g - 0.5) * 1.35 + 0.55, 0, 1) ** gamma * gain   # contrasty image inside the holes
    v = (a * lum * 255).clip(0, 255).astype(np.uint8)
    return np.stack([v, v, v], -1)


def glitch_bands(img, rng, n=(1, 3), strength=1.0):
    out = img.copy()
    H, W = img.shape[:2]
    for _ in range(rng.integers(n[0], n[1] + 1)):
        h = int(rng.integers(8, 60) * H / 1350)
        y = int(rng.integers(0, H - h))
        dx = int(rng.integers(-80, 80) * strength * W / 1080)
        band = np.roll(out[y:y + h], dx, axis=1).astype(np.float32)
        band *= rng.uniform(0.6, 1.3)
        out[y:y + h] = band.clip(0, 255).astype(np.uint8)
    return out


def expose(img, e, dip_cap=1.0):
    """e>1 lifts toward white (burst). e<1 is the dark dip: a crushing gamma
    curve (mids/shadows go near-black, highlights survive as bright streaks),
    which is what the reference's dip frames look like - not a flat dim."""
    f = img.astype(np.float32) / 255
    if e >= 1:
        k = min(1.0, (e - 1) * 3.2)
        f = f + (1 - f) * k
    else:
        # high-contrast dip: stretch around the frame mean (black with bright
        # streaks, like the reference) while landing on the measured brightness
        # dark base (under-exposed, crushed) + the image's ink lines coming
        # through as bright streaks - the reference dips are dark frames with
        # light streak detail, not a dimmed copy of a white panel
        mean = float(f.mean())
        g = f.mean(2, keepdims=True)
        lines = np.clip(cv2.Laplacian(cv2.GaussianBlur(g[..., 0], (0, 0), 2.0), cv2.CV_32F) * 6, 0, 1)[..., None]
        base = (f * 0.55) ** 2.2
        f = base + 0.8 * lines * (1 - base)
        # the reference dips are dark in ABSOLUTE terms (luma ~25-60 Kaiser,
        # ~40-85 Rin) whatever the panel; a ratio alone leaves white line-art gray
        target = min(e * mean, dip_cap)
        f = f * (target / max(1e-3, float(f.mean())))
    return (f * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Card (tilted inset) rendering
# ---------------------------------------------------------------------------

def make_card(img, card_w_px, aspect, border, shadow=True):
    """Return RGBA card: image contain-fit into a portrait card with light border."""
    cw = max(8, int(card_w_px))
    ch = max(8, int(cw / aspect))
    inner = cover(img, max(2, cw - 2 * border), max(2, ch - 2 * border), center=(0.5, 0.45))
    card = np.zeros((ch, cw, 4), np.uint8)
    card[..., :3] = (214, 216, 220)
    card[..., 3] = 255
    card[border:ch - border, border:cw - border, :3] = inner
    return card


def warp_card(canvas, card, center, tilt_deg, rot_deg, blur=0.0, shadow=True, width_px=None):
    """Composite `card` (RGBA) onto canvas with a 3D X-tilt (top edge toward the
    viewer, like the reference) plus a small Z rotation, and a soft shadow.
    width_px: on-screen width (float) - geometry is sub-pixel, so growth is
    smooth instead of stepping 1px at a time."""
    H, W = canvas.shape[:2]
    if width_px and width_px < 0.75 * card.shape[1]:
        # big downscale in warpPerspective aliases screentone -> pre-shrink (area filter)
        tw = max(8, int(width_px * 1.3))
        card = cv2.resize(card, (tw, max(8, int(card.shape[0] * tw / card.shape[1]))), interpolation=cv2.INTER_AREA)
    ch, cw = card.shape[:2]
    gw = width_px or cw                  # geometric width on screen
    gh = gw * ch / cw
    t = math.radians(tilt_deg)
    # perspective: top edge wider, card squashed vertically by cos(t)
    squash = max(0.08, math.cos(t))
    widen = 1 + 0.35 * math.sin(t)
    hh = gh * squash / 2
    # top edge tilted AWAY from the viewer -> narrower (as in the reference)
    top_hw, bot_hw = gw / 2 / widen, gw / 2 * widen ** 0.35
    pts = np.float32([[-top_hw, -hh], [top_hw, -hh], [bot_hw, hh], [-bot_hw, hh]])
    r = math.radians(rot_deg)
    Rm = np.float32([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    pts = pts @ Rm.T + np.float32([center[0] * W, center[1] * H])
    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    M = cv2.getPerspectiveTransform(src, pts)
    warped = cv2.warpPerspective(card, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    alpha = warped[..., 3:4].astype(np.float32) / 255
    rgb = warped[..., :3].astype(np.float32)
    if blur > 0.5:
        k = int(blur) * 2 + 1
        rgb = cv2.GaussianBlur(rgb * alpha, (k, k), 0)
        a2 = cv2.GaussianBlur(alpha[..., 0], (k, k), 0)[..., None]
        rgb = rgb / np.maximum(a2, 1e-3)
        alpha = a2
    out = canvas.astype(np.float32)
    if shadow:
        sh = cv2.GaussianBlur(alpha[..., 0], (0, 0), 10 * W / 1080)
        sh = np.roll(np.roll(sh, int(8 * W / 1080), 0), int(5 * W / 1080), 1)[..., None]
        out = out * (1 - 0.45 * sh)
    out = out * (1 - alpha) + rgb * alpha
    return out.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Text layers (prebuilt RGBA)
# ---------------------------------------------------------------------------

def caption_layer(text, W, H, st):
    k = W / 1080
    y0, y1 = int(st["caption_band"][0] * H), int(st["caption_band"][1] * H)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # band: dark translucent with a vertical gradient and thin light edges
    band = np.zeros((y1 - y0, W, 4), np.uint8)
    grad = np.linspace(0.84, 0.74, y1 - y0)[:, None]
    band[..., 0], band[..., 1], band[..., 2] = 22, 25, 31
    band[..., 3] = (grad * 255).astype(np.uint8)
    layer.paste(Image.fromarray(band), (0, y0))
    for yy in (y0, y1 - 2):
        d.rectangle([0, yy, W, yy + max(1, int(2 * k))], fill=(175, 180, 190, 150))
    if not text:
        return np.asarray(layer)

    def layout(cap_px):
        big = _font(st["caption_font"], cap_px * k)
        small = _font(st["caption_font"], cap_px * k * st["caption_small_ratio"])
        track = st["caption_tracking"] * cap_px * k * 0.5
        glyphs = lambda word: [(c.upper(), small if c.islower() else big) for c in word]
        width = lambda word: sum(f.getlength(c) + track for c, f in glyphs(word)) - track
        space = big.getlength(" ") + track * 2
        maxw = W * st["caption_wrap"]
        lines, cur, cw = [], [], 0
        for word in text.split():
            ww = width(word)
            if cur and cw + space + ww > maxw:
                lines.append(cur)
                cur, cw = [], 0
            cw += (space if cur else 0) + ww
            cur.append(word)
        if cur:
            lines.append(cur)
        return glyphs, width, space, track, lines, cap_px * k * 1.42

    # shrink long captions until every line fits inside the band (the
    # reference's 4-line caption uses visibly smaller text than its 3-line one)
    cap = st["caption_cap_px"]
    while True:
        glyphs, width, space, track, lines, lh = layout(cap)
        if lh * len(lines) <= (y1 - y0) * 0.88 or cap <= 22:
            break
        cap -= 1
    total = lh * len(lines)
    cy = (y0 + y1) / 2 - total / 2 + lh * 0.72
    for words in lines:
        lw = sum(width(w) for w in words) + space * (len(words) - 1)
        x = (W - lw) / 2
        for wi, word in enumerate(words):
            for c, f in glyphs(word):
                d.text((x + 1, cy + 1), c, font=f, fill=(0, 0, 0, 120), anchor="ls")
                d.text((x, cy), c, font=f, fill=tuple(st["caption_color"]) + (255,), anchor="ls")
                x += f.getlength(c) + track
            x += space - track
        cy += lh
    return np.asarray(layer)


def watermark_layer(text, W, H, st):
    k = W / 1080
    f = _font("ariblk.ttf", st["watermark_size"] * k * 0.62)
    track = 4 * k
    tw = int(sum(f.getlength(c) + track for c in text) + 40 * k)
    th = int(st["watermark_size"] * k * 1.4)
    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for off, col in ((3, (35, 35, 38, 230)), (0, (205, 206, 210, 235))):
        x = 20 * k
        for c in text:
            d.text((x + off * k, th * 0.15 + off * k), c, font=f, fill=col)
            x += f.getlength(c) + track
    img = img.transform(img.size, Image.AFFINE, (1, 0.28, -0.14 * th, 0, 1, 0), resample=Image.BICUBIC)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.paste(img, (int(W / 2 - tw / 2), int(st["watermark_y"] * H - th / 2)), img)
    return np.asarray(layer)


def end_card(text, W, H, st):
    g = np.linspace(252, 236, int(H * 0.55))
    g = np.concatenate([g, np.linspace(236, 176, H - len(g))])
    bg = np.repeat(g[:, None], W, 1)
    img = Image.fromarray(np.stack([bg, bg, bg], -1).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f = _font(st["end_font"], H * 0.092, st["end_font_variation"])
    d.text((W / 2, H * 0.5), text, font=f, fill=(58, 58, 60), anchor="mm")
    return np.asarray(img)


def over(base, rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (base.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a).astype(np.uint8)


# ---------------------------------------------------------------------------
# Grade + film texture
# ---------------------------------------------------------------------------

def grade(img, st):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = ((g - 128) * st["contrast"] + 128 + st["brightness"]).clip(0, 255)
    f = st["black_floor"]
    base = f + g2 * (255 - f) / 255
    blue = base + st["shadow_blue"] * (1 - g2 / 255) ** 0.7
    mono = np.stack([base, base, blue], -1)
    col = st.get("color", 0.0)
    if col <= 0:
        return mono.clip(0, 255).astype(np.uint8)
    # colour version: same tone curve (contrast/floor/blue shadows) applied as
    # a luma-ratio to the original colours, saturation scaled by `color`
    rgb = img.astype(np.float32) * (base / np.maximum(g, 1))[..., None]
    rgb[..., 2] += st["shadow_blue"] * (1 - g2 / 255) ** 0.7
    L = rgb.mean(-1, keepdims=True)
    rgb = L + (rgb - L) * st.get("saturation", 1.15)
    return (mono * (1 - col) + rgb * col).clip(0, 255).astype(np.uint8)


class Film:
    """Vertical scratches, dust and the occasional fiber, deterministic per frame."""

    def __init__(self, W, H, seed=7):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        self.scratches = []   # [x, life, alpha, width]

    def layer(self, k):
        W, H = self.W, self.H
        rng = np.random.default_rng(k * 7919 + 13)
        if rng.random() < 0.35:
            self.scratches.append([rng.uniform(0.05, 0.95) * W, int(rng.integers(2, 8)),
                                   rng.uniform(0.25, 0.7), rng.choice([1, 1, 2])])
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        keep = []
        for s in self.scratches:
            x = s[0] + rng.normal(0, 1.5)
            y0 = rng.uniform(-0.2, 0.3) * H
            y1 = y0 + rng.uniform(0.5, 1.2) * H
            d.line([(x, y0), (x, y1)], fill=int(255 * s[2]), width=int(s[3]))
            s[1] -= 1
            if s[1] > 0:
                keep.append(s)
        self.scratches = keep
        for _ in range(int(rng.integers(4, 14))):
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            r = rng.uniform(0.8, 2.6)
            d.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.uniform(120, 255)))
        if rng.random() < 0.04:
            x, y = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H
            pts = [(x + 60 * math.sin(i / 3 + rng.random()), y + i * 6) for i in range(int(rng.integers(8, 20)))]
            d.line(pts, fill=200, width=1)
        return np.asarray(img).astype(np.float32) / 255

    def apply(self, img, k, strength=0.55):
        a = self.layer(k)[..., None] * strength
        f = img.astype(np.float32)
        return (f + (255 - f) * a).clip(0, 255).astype(np.uint8)   # screen-like


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class TemplateEdit:
    def __init__(self, manifest, preview=False):
        self.m = manifest
        tpath = ROOT / manifest["template"]
        self.t = json.loads(tpath.read_text(encoding="utf-8-sig"))
        self.st = dict(STYLE)
        self.st.update(self.t.get("style", {}))
        self.st.update(manifest.get("style_overrides", {}))     # per-edit tweaks (e.g. the colour version)
        W, H = self.t["size"]
        self.scale = 0.5 if preview else 1.0
        self.W, self.H = int(W * self.scale) // 2 * 2, int(H * self.scale) // 2 * 2
        self.fps = self.t.get("fps", 60)
        self.nframes = int(round(self.t["duration"] * self.fps))
        fr = lambda s: int(round(s * self.fps))
        self.D = [fr(c) for c in self.t["cuts"]]
        self.E = fr(self.t["end"]["card"])
        self.E_fade = (fr(self.t["end"]["fade_start"]), fr(self.t["end"]["fade_end"]))
        self.fade_in = (fr(self.t["intro"]["fade_start"]), fr(self.t["intro"]["fade_end"]))
        self.C = fr(self.t["intro"]["card_start"])
        self.kinds = [s["kind"] for s in self.t["shots"]]
        n_slots = len(self.kinds)
        shots = manifest.get("shots", [])
        if len(shots) != n_slots:
            raise ValueError(f"template '{self.t['name']}' needs exactly {n_slots} shot images, got {len(shots)}. "
                             f"Slot kinds: {self.kinds} ('card' slots take an image + optional 'card' image)")
        self.in_curve = PchipInterpolator(*zip(*self.st["in_curve"]))
        self.out_curve = PchipInterpolator(*zip(*self.st["out_curve"]))
        self.card_growth = PchipInterpolator(*zip(*self.st["card_growth"]))
        W, H = self.W, self.H
        focus = {int(k): v for k, v in manifest.get("focus", {}).items()}
        self.intro = cover(load_rgb(manifest["intro_image"]), W, H)
        self.intro_src = load_rgb(manifest["intro_image"])
        self.shots = []
        for i, (kind, s) in enumerate(zip(self.kinds, shots)):
            if isinstance(s, str):
                s = {"clip": s} if s.lower().endswith((".mp4", ".mov", ".webm")) else {"image": s}
            if s.get("clip"):
                clip = ClipSource(s["clip"], W, H, s.get("clip_start", 0))
                src = clip.frame(0)
                entry = {"img": src, "clip": clip}
            else:
                src = load_rgb(s["image"])
                entry = {"img": cover(src, W, H, focus.get(i))}
            if kind == "card":
                card_src = load_rgb(s["card"]) if s.get("card") else src
                h, w = card_src.shape[:2]
                entry["card_src"] = card_src
                entry["card_aspect"] = float(np.clip(w / h, 0.62, 1.0))
                bg = cover(src, W, H, focus.get(i))
                lift = self.st["card_bg_lift"]
                bg = cv2.GaussianBlur(bg, (0, 0), 1.2 * W / 1080)
                entry["img"] = (bg.astype(np.float32) * (1 - lift) + 255 * lift).astype(np.uint8)
            self.shots.append(entry)
        # template 7 (measured): every card that drops in mid-drop shows the
        # NEXT shot's image, framed exactly as that shot lands - the card
        # becomes the next shot
        if self.st.get("card_from_next"):
            for i, kind in enumerate(self.kinds[:-1]):
                if kind == "card":
                    nxt = self.shots[i + 1]
                    self.shots[i]["card_src"] = nxt.get("card_src", nxt["img"]) if self.kinds[i + 1] == "card" else nxt["img"]
                    self.shots[i]["card_aspect"] = W / H
        self.caption = caption_layer(manifest.get("caption", ""), W, H, self.st)
        self.wm = watermark_layer(manifest.get("watermark", "DUDS"), W, H, self.st) if manifest.get("watermark", "DUDS") else None
        self.endcard = end_card(manifest.get("end_text", "DUDS"), W, H, self.st)
        self.dark = np.zeros((H, W, 3), np.uint8)
        self.dark[:] = self.st["dark_bg"]
        self.film = Film(W, H)
        zin, zout = self.t["zoom"]["in"], self.t["zoom"]["out"]
        self.ln_in, self.ln_out = math.log(zin), math.log(1 / zout)
        if self.st.get("zoom_max"):
            # clips are already framed close: cap the template's zoom depth
            # (the curve's shape/timing stays the same, only the amount shrinks)
            cap = math.log(self.st["zoom_max"])
            self.ln_in, self.ln_out = min(self.ln_in, cap), min(self.ln_out, cap)
        self.anchor_in, self.anchor_out = self.t["zoom"]["in_anchor"], self.t["zoom"]["out_anchor"]
        self.card_cache = {}
        # what the pre-drop pop card shows: the intro image (templates 1-3) or
        # the first shot after the drop, framed exactly as that shot lands, so
        # the card "becomes" the next shot (template 7, style pop_from=first_shot)
        if manifest.get("pop_image"):
            self.pop_img = cover(load_rgb(manifest["pop_image"]), W, H)
        elif self.st.get("pop_from") == "first_shot":
            s0 = self.shots[0]
            self.pop_img = cover(s0["card_src"], W, H) if self.kinds[0] == "card" else s0["img"]
        else:
            self.pop_img = self.intro
        self.pop_track = [r for r in self.t["intro"].get("pop_track", []) if r["w"] >= 0.3]
        self._pop_card = None
        self._pop_fit = None

    # ---- cutout popups (template 9: a character cutout rises / slides in over
    # the previous panel across several shots) ----
    def _cutout(self, c):
        key = ("cutout", c["src"])
        if key not in self.card_cache:
            from wekisava_edit import cutout_mask
            e = self.shots[c["src"]]
            if "clip" in e:
                img = e["clip"].frame(c.get("src_frame", 0))
                tmp = ROOT / ".cache" / f"cutout_src_{abs(hash(str(e['clip'].path)))}_{c.get('src_frame', 0)}.png"
                tmp.parent.mkdir(exist_ok=True)
                Image.fromarray(img).save(tmp)
                path = tmp
            else:
                img, path = e["img"], None
            m = cutout_mask(path) if path else np.ones(img.shape[:2], np.float32)
            m = cv2.resize(m, (img.shape[1], img.shape[0]))
            # solid silhouette: close gaps and fill holes (light skin / white
            # shirt read as background to the cutout model -> see-through holes)
            mb = (m > 0.4).astype(np.uint8)
            mb = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            ff = mb.copy()
            cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 1)
            mb = mb | (1 - ff)
            n, lab, st, _ = cv2.connectedComponentsWithStats(mb)
            if n > 1:                                          # keep the character, drop specks
                big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                mb = (lab == big).astype(np.uint8)
            m = np.maximum(m * mb, mb.astype(np.float32) * (m > 0.15))
            ys, xs = np.nonzero(m > 0.5)
            if len(ys) == 0:
                ys, xs = np.array([0, img.shape[0] - 1]), np.array([0, img.shape[1] - 1])
            box = (xs.min(), ys.min(), xs.max(), ys.max())
            self.card_cache[key] = (img.astype(np.float32), m.astype(np.float32), box)
        return self.card_cache[key]

    def _cutout_offset(self, c, k):
        """(dx, dy) in frame fractions. One continuous ease-out from off-screen
        to exactly 0 at the handoff cut (the cutout then sits where the
        character is in its own shot, so the cut only swaps the background)."""
        hand = self.D[c["src"]]
        u = min(max((k - c["start"]) / max(1, hand - c["start"]), 0.0), 1.0)
        e = 1 - (1 - u) ** c.get("ease", 3.0)
        d = (1 - e) * c.get("dist", 0.8)
        return {"bottom": (0.0, d), "left": (-d, 0.0), "right": (d, 0.0), "top": (0.0, -d)}[c.get("from", "bottom")]

    def _with_cutout(self, k, base):
        for c in self.t.get("cutouts", []):
            if not (c["start"] <= k <= c["end"]):
                continue
            img, m, _ = self._cutout(c)
            W, H = self.W, self.H
            img = cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img
            m = cv2.resize(m, (W, H)) if m.shape[:2] != (H, W) else m
            m = cv2.GaussianBlur(m, (0, 0), 1.2)                 # clean, feathered edge
            dx, dy = self._cutout_offset(c, k)
            M = np.float32([[1, 0, dx * W], [0, 1, dy * H]])
            ci = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            cm = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            pdx, pdy = self._cutout_offset(c, k - 1)
            v = max(abs(dx - pdx) * W, abs(dy - pdy) * H)          # px moved this frame
            if v > 2:
                n = int(min(31, v * 0.9)) | 1
                kern = np.zeros((n, n), np.float32)
                if abs(dx - pdx) > abs(dy - pdy):
                    kern[n // 2, :] = 1.0 / n
                else:
                    kern[:, n // 2] = 1.0 / n
                ci, cm = cv2.filter2D(ci, -1, kern), cv2.filter2D(cm, -1, kern)
            b = base.astype(np.float32)
            base = (b * (1 - cm[..., None]) + ci * cm[..., None]).clip(0, 255).astype(np.uint8)
        return base

    def _cutout_bg(self, i, k):
        for c in self.t.get("cutouts", []):
            if c["start"] <= k <= c["end"] and i != c["bg"] and i in c.get("covers", []):
                return c["bg"]
        return None

    def shot_img(self, i, k):
        if self.t.get("cutouts"):
            b = self._cutout_bg(i, k)
            img = self._shot_img_raw(b if b is not None else i, k if b is None else k)
            return self._with_cutout(k, img)
        return self._shot_img_raw(i, k)

    def _shot_img_raw(self, i, k):
        e = self.shots[i]
        if "clip" not in e:
            return e["img"]
        img = e["clip"].frame(k - self.D[i] if self.t.get("cutouts") is None else k - self.D[i])
        if self.kinds[i] == "card":
            lift = self.st["card_bg_lift"]
            img = cv2.GaussianBlur(img, (0, 0), 1.2 * self.W / 1080)
            img = (img.astype(np.float32) * (1 - lift) + 255 * lift).astype(np.uint8)
        return img

    # ---- shot geometry ----
    def shot_land(self, i):
        sh = self.t["shots"][i]
        return sh.get("land", 3 if sh.get("impact") == "quick" else self.st["land_offset"])

    def shot_bounds(self, i):
        land = self.D[i] + self.shot_land(i)
        last = (self.D[i + 1] if i + 1 < len(self.D) else self.E) - 1
        return land, last

    def shot_scale(self, i, k):
        """(scale, anchor, per-frame log speed) for shot i at frame k."""
        land, last = self.shot_bounds(i)
        n = max(1, last - land)
        u = (k - land) / n
        kind = self.kinds[i]
        if kind == "hold":
            # quick-cut shots: the panel just sits (measured scale 1.00)
            return 1.0, [0.5, 0.45], 0.0
        if kind == "out":
            curve, lnz, anchor = self.out_curve, self.ln_out, self.anchor_out
        else:
            curve, lnz, anchor = self.in_curve, self.ln_in, self.anchor_in
        z = self.t["shots"][i].get("zoom")
        if z:                                           # measured per-shot zoom amount
            lnz = abs(math.log(z))
            if self.st.get("zoom_max"):
                lnz = min(lnz, math.log(self.st["zoom_max"]))
        if kind == "card":
            lnz, anchor = math.log(self.st["card_bg_zoom"]), [0.5, 0.35]
        f = float(curve(min(max(u, 0), 1)))
        # beyond [0,1]: continue with the end slopes (pre-landing frames are the
        # heavily blurred burst; after the last clean frame come the halftone frames)
        if u < 0:
            f = u * float(curve.derivative()(0))
        elif u > 1:
            f = 1 + (u - 1) * float(curve.derivative()(1))
        speed = abs(lnz * float(curve.derivative()(min(max(u, 0), 1)))) / n
        if kind == "out":
            s = math.exp(lnz * (1 - f))           # starts zoomed in, ends at 1.0
        else:
            s = math.exp(lnz * f)                 # starts at 1.0, zooms in
        return max(1.0, s), anchor, speed

    def shot_frame(self, i, k, motion_blur=True):
        e = self.shots[i]
        s, anchor, speed = self.shot_scale(i, k)
        img = zoom_about(self.shot_img(i, k), s, anchor)
        land, last = self.shot_bounds(i)
        if motion_blur and k > land:
            # measured: blur only right after the landing, decaying within a few
            # frames - frames before the next cut are sharp even though the zoom
            # is accelerating there
            decay = math.exp(-(k - land - 1) / self.st["motion_blur_tau"])
            img = radial_blur(img, self.st["motion_blur"] * speed * decay, anchor, samples=6)
        if self.kinds[i] == "card":
            img = self.draw_shot_card(i, k, img)
        return img

    def card_params(self, i, k):
        """(width frac, cx, cy, tilt deg, rot deg, blur px) of card shot i at
        frame k. Every term is continuous across the landing: the transition
        frames before it (u < 0) blend smoothly into it instead of switching
        tilt/blur on the landing frame (that switch read as a jerk)."""
        land, last = self.shot_bounds(i)
        u = (k - land) / max(1, last - land)
        uc = min(max(u, 0), 1)
        pre = smoothstep(min(1.0, -u / 0.2)) if u < 0 else 0.0     # 0 at landing -> 1 further before
        w0, w1 = self.st["card_width"]
        g = float(self.card_growth(uc)) - 0.25 * pre
        wfrac = w0 * (w1 / w0) ** g
        (cx0, cy0), (cx1, cy1) = self.st["card_center"]
        cx, cy = cx0 + (cx1 - cx0) * uc, cy0 + (cy1 - cy0) * uc + 0.04 * pre
        # tilt only ever decreases (no back-and-forth); no rotation - the
        # reference card stays perfectly upright the whole shot
        tilt = self.st["card_tilt_deg"] * (1 - ease_out_cubic(min(1.0, uc / 0.45))) + 15 * pre
        rot = 0.0
        blur = 6 * self.W / 1080 * pre
        return wfrac, cx, cy, tilt, rot, blur

    def draw_shot_card(self, i, k, img):
        e = self.shots[i]
        wfrac, cx, cy, tilt, rot, blur = self.card_params(i, k)
        key = ("card", i)
        if key not in self.card_cache:          # build once at full size, draw at fractional size
            self.card_cache[key] = make_card(e["card_src"], self.st["card_width"][1] * self.W * 1.1,
                                             e["card_aspect"], max(2, int(4 * self.W / 1080)))
        card = self.card_cache[key]
        return warp_card(img, card, (cx, cy), tilt, rot, blur, width_px=wfrac * self.W)

    # ---- intro ----
    def intro_frame(self, k):
        img = over(self.intro, self.caption)
        if k >= self.C:
            img = self.draw_pop_card(k, img)
        return img

    def _pop_state(self, k):
        """Card box at frame k from the reference's measured path (smoothed)."""
        tr = self.pop_track
        f = np.array([r["f"] for r in tr], float)
        f_now = (k - self.C) * self.t.get("reference_fps", 60) / self.fps
        if f_now < f[0]:
            return None

        if self._pop_fit is None:
            self._pop_fit = self._fit_pop(tr)
        fits = self._pop_fit
        fg = fits["f"]

        def val(key):
            p, v = fits[key], fits[key + "_vel"]
            if f_now <= fg[-1]:
                return float(p(f_now))
            return float(p(fg[-1]) + v * (f_now - fg[-1]))      # continue at the final speed
        return fits["cx"], val("top"), val("bot"), val("w")

    @staticmethod
    def _fit_pop(tr):
        """Robust, smooth, monotone path from the measured boxes: drop the last
        sample (the transition is starting there) and single-frame detection
        glitches, smooth, enforce the motion's direction (the card only rises;
        after settling it only grows), then PCHIP (no overshoot)."""
        from scipy.interpolate import PchipInterpolator, UnivariateSpline
        from scipy.ndimage import median_filter
        tr = tr[:-1]
        f = np.array([r["f"] for r in tr], float)
        grid = np.arange(f[0], f[-1] + 1)
        out = {"f": grid, "cx": float(np.median([r["cx"] for r in tr]))}
        # Model the card with three parameters that each move ONE way only,
        # instead of fitting its top and bottom edges separately (two
        # independent fits made its height/tilt breathe = visible wobble):
        #   w    width           only grows
        #   flat tilt flattening only increases (0 = edge-on, 1 = facing camera)
        #   cy   centre height   only rises
        aspect = 1080 / 1350
        raw = {"w": np.array([r["w"] for r in tr]),
               "cy": np.array([(r["top"] + r["bot"]) / 2 for r in tr])}
        hgt = np.array([r["bot"] - r["top"] for r in tr])
        raw["flat"] = np.clip(hgt * 1350 / (raw["w"] * 1080 / aspect), 0.05, 1.0)
        fits = {}
        for key, v in raw.items():
            med = median_filter(v, size=5, mode="nearest")
            keep = np.abs(v - med) < 0.03
            g = np.interp(grid, f[keep], v[keep])
            g = -_isotonic(-g) if key == "cy" else _isotonic(g)
            g = np.convolve(np.pad(g, 4, mode="edge"), np.hanning(9) / np.hanning(9).sum(), mode="valid")
            fits[key] = g
        fits["flat"] = np.clip(fits["flat"], 0.05, 1.0)
        full_h = fits["w"] * 1080 / aspect / 1350
        h = full_h * fits["flat"]
        comp = {"top": fits["cy"] - h / 2, "bot": fits["cy"] + h / 2, "w": fits["w"]}
        for key, g in comp.items():
            out[key] = PchipInterpolator(grid, g)
            out[key + "_vel"] = float(np.mean(np.diff(g[-4:])))
        return out

    def draw_pop_card(self, k, img):
        """Intro image pops out as a card one beat before the drop, following
        the reference's measured path exactly: it rises off the caption bar
        while its tilt flattens, holds centered, then rushes toward the camera.
        No shake - the reference card stays centered on every frame."""
        if not getattr(self, "pop_track", None):
            return self._draw_pop_card_legacy(k, img)
        s = self._pop_state(k)
        if s is None:
            return img
        cx, top, bot, w = s
        W, H = self.W, self.H
        aspect = W / H                                   # the card IS the framed intro image
        full_h = w * W / aspect
        proj_h = max(2.0, (bot - top) * H)
        flat = float(np.clip(proj_h / full_h, 0.05, 1.0))   # 1 = facing camera, small = tilted back
        top_hw = w * W / 2
        bot_hw = top_hw * (0.72 + 0.28 * flat)           # far (bottom) edge narrower while tilted
        x = cx * W
        quad = np.float32([[x - top_hw, top * H], [x + top_hw, top * H],
                           [x + bot_hw, bot * H], [x - bot_hw, bot * H]])
        if self._pop_card is None:
            self._pop_card = make_card(self.pop_img, W * 0.8, aspect, max(2, int(4 * W / 1080)))
        card = self._pop_card
        ch, cw = card.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), quad)
        warped = cv2.warpPerspective(card, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        alpha = warped[..., 3:4].astype(np.float32) / 255
        rgb = warped[..., :3].astype(np.float32)
        # motion blur only while the card moves fast (the first rise frames)
        prev = self._pop_state(k - 1)
        speed = abs(top - prev[1]) * H if prev else 0.0
        if speed > 6:
            kk = int(min(31, speed * 0.6)) | 1
            rgb = cv2.blur(rgb * alpha, (1, kk))
            a2 = cv2.blur(alpha[..., 0], (1, kk))[..., None]
            rgb, alpha = rgb / np.maximum(a2, 1e-3), a2
        out = img.astype(np.float32)
        sh = cv2.GaussianBlur(alpha[..., 0], (0, 0), 10 * W / 1080)
        sh = np.roll(np.roll(sh, int(8 * W / 1080), 0), int(5 * W / 1080), 1)[..., None]
        out = out * (1 - 0.45 * sh)
        out = out * (1 - alpha) + rgb * alpha
        return out.clip(0, 255).astype(np.uint8)

    def _draw_pop_card_legacy(self, k, img):
        """Fallback for templates without a measured pop_track."""
        span = max(1, self.D[0] + 2 - self.C)
        p = (k - self.C) / span
        w0, w1 = self.st["pop_width"]
        wfrac = w0 + (w1 - w0) * (0.55 * min(1, p / 0.25) + 0.45 * max(0, (p - 0.25) / 0.75) ** 1.6)
        rng = np.random.default_rng(k // 3 + 101)           # shake: new offset every 3 frames
        amp = 0.07 * min(1, p / 0.3)
        center = (0.47 + rng.uniform(-amp, amp), 0.6 - 0.1 * min(1, p / 0.35) + rng.uniform(-amp, amp) * 0.6)
        tilt = 70 * (1 - smoothstep(p / 0.4)) + 4
        rot = rng.uniform(-3, 3) * min(1, p / 0.3)
        blur = 10 * self.W / 1080 * max(0, 1 - p / 0.3)
        h, w = self.pop_img.shape[:2]
        card = make_card(self.pop_img, wfrac * self.W, float(np.clip(w / h, 0.62, 1.0)),
                         max(2, int(4 * self.W / 1080)))
        return warp_card(img, card, center, tilt, rot, blur)

    # ---- transitions ----
    def outgoing_frame(self, i, k):
        """What the edit was showing before cut i (intro for the drop)."""
        return self.intro_frame(k) if i == 0 else self.shot_frame(i - 1, k, motion_blur=False)

    def transition_frame(self, i, k):
        d = self.D[i]
        j = k - d
        if self.t["shots"][i].get("impact") == "quick":
            rng = np.random.default_rng(k * 31 + 5)
            if j == 0:
                pitch = self.st["halftone_pitch"] * self.W / 1080
                return glitch_bands(halftone_dots(self.outgoing_frame(i, k), pitch, self.st["halftone_gain"][0]), rng, (0, 2), 0.6)
            s, anchor, _ = self.shot_scale(i, k)
            base = zoom_about(self.shot_img(i, k), s, anchor)
            img = radial_blur(base, self.st["strobe_blur"][0], anchor, samples=12, scale_down=3)
            return expose(img, self.st["strobe"][0] if j == 1 else self.st["strobe"][1], self.st["dip_cap"])
        cell = max(4, int(self.st["halftone_cell"] * self.W / 1080))
        rng = np.random.default_rng(k * 31 + 5)
        if j <= 2:
            out = self.outgoing_frame(i, k)
            pitch = self.st["halftone_pitch"] * self.W / 1080
            if j < 2:
                img = halftone_dots(out, pitch, self.st["halftone_gain"][j])
                img = glitch_bands(img, rng, (0, 2), 0.6)
            else:
                # dark flash: near-black with only a few surviving dots
                img = halftone_dots(out, pitch, self.st["halftone_gain"][2], keep=self.st["flash_keep"], seed=k,
                                    blob=self.st.get("flash_blob", 3.0))
                img = np.maximum(img, self.dark)
            return img
        land, _ = self.shot_bounds(i)
        e = self.shots[i]
        s, anchor, _ = self.shot_scale(i, k)
        base = zoom_about(self.shot_img(i, k), s, anchor)
        if self.kinds[i] == "card":
            base = self.draw_shot_card(i, k, base)
        idx = j - 3
        img = radial_blur(base, self.st["strobe_blur"][idx], anchor, samples=12, scale_down=3)
        img = expose(img, self.st["strobe"][idx], self.st["dip_cap"])
        if idx in (1, 2):
            img = glitch_bands(img, rng, (1, 3), 1.0)
        return img

    # ---- outro ----
    def end_frame(self, k):
        f0, f1 = self.E_fade
        if k < f0:
            return self.endcard.copy()
        a = smoothstep((k - f0) / max(1, f1 - f0))
        return (self.endcard.astype(np.float32) * (1 - a) + self.dark.astype(np.float32) * a).astype(np.uint8)

    # ---- main ----
    def frame(self, k):
        fi0, fi1 = self.fade_in
        is_dark_scene = False
        if k >= self.E:
            img = self.end_frame(k)
            wm = True
            is_dark_scene = k >= self.E_fade[0]
        elif k < self.D[0]:
            if k < fi0:
                img = self.dark.copy()
                is_dark_scene = True
            else:
                img = self.intro_frame(k)
                if k < fi1:
                    a = smoothstep((k - fi0) / max(1, fi1 - fi0))
                    img = (self.dark.astype(np.float32) * (1 - a) + img.astype(np.float32) * a).astype(np.uint8)
                    is_dark_scene = a < 0.6
            wm = False
        else:
            i = max(j for j, d in enumerate(self.D) if d <= k)
            # cuts hidden behind a rising cutout are plain cuts: the impact
            # strobe would flash over the cutout while it comes up
            hidden = any((i in c.get("covers", []) and self.D[i] <= c["end"]
                          and self.D[i] + self.shot_land(i) > c["start"]) or
                         (i == c.get("bg") and self.D[i] <= c["start"] < self.D[i] + self.shot_land(i))
                         for c in self.t.get("cutouts", []))
            if k < self.D[i] + self.shot_land(i) and not hidden:
                img = self.transition_frame(i, k)
            else:
                img = self.shot_frame(i, k)
            wm = True
        img = grade(img, self.st)
        if wm and self.wm is not None:
            img = over(img, self.wm)
        img = self.film.apply(img, k, (0.6 if is_dark_scene else 0.22) * self.st.get("film_gain", 1.0))
        return img

    def render(self, out_path, log=print):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".video.mp4")
        writer = imageio_ffmpeg.write_frames(str(tmp), (self.W, self.H), fps=self.fps, codec="libx264",
                                             quality=None, bitrate=None, pix_fmt_out="yuv420p",
                                             output_params=["-crf", "16", "-preset", "medium"],
                                             macro_block_size=2)
        writer.send(None)
        for k in range(self.nframes):
            writer.send(np.ascontiguousarray(self.frame(k)))
            if k % 120 == 0:
                log(f"  frame {k}/{self.nframes}")
        writer.close()
        music = ROOT / self.m.get("music_file", self.t["music"])   # exact clean song when matched
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp),
                        "-i", str(music), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "256k", "-t", f"{self.nframes / self.fps:.3f}", str(out_path)], check=True)
        tmp.unlink()
        return out_path


def list_templates():
    out = {}
    for p in sorted((ROOT / "templates").glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8-sig"))
        out[p.stem] = {"file": f"templates/{p.name}", "duration": t["duration"], "size": t["size"],
                       "slots": [s["kind"] for s in t["shots"]], "music": t["music"]}
    return out


def new_manifest(template_name, edit_id, image_folder=None):
    """Write edits/<edit_id>.json pre-filled for a template. If image_folder has
    images, they're assigned in name order: first = intro, rest = shots."""
    imgs = []
    if image_folder:
        folder = Path(image_folder)
        folder = folder if folder.is_absolute() else ROOT / folder
        imgs = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in folder.iterdir()
                      if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    m = manifest_skeleton(template_name, edit_id, imgs)
    out = ROOT / "edits" / f"{edit_id}.json"
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return out, len(imgs)


def manifest_skeleton(template_name, edit_id, imgs=()):
    t = json.loads((ROOT / "templates" / f"{template_name}.json").read_text(encoding="utf-8-sig"))
    kinds = [s["kind"] for s in t["shots"]]
    imgs = list(imgs)
    intro = imgs[0] if imgs else "panels/FOLDER/intro.png"
    shots = []
    for i, kind in enumerate(kinds):
        path = imgs[1 + i] if len(imgs) > 1 + i else f"panels/FOLDER/{i + 1:02d}_{kind}.png"
        shots.append({"image": path, "card": path} if kind == "card" else path)
    m = {
        "id": edit_id, "type": "template_edit", "template": f"templates/{template_name}.json",
        "title": "", "description": "", "tags": ["shorts", "manga", "mangaedit"],
        "caption": "POV : write your caption here",
        "intro_image": intro,
        "_slots": f"{len(kinds)} shots: " + ", ".join(f"{i + 1}={k}" for i, k in enumerate(kinds)) +
                  ". 'in' = zooms in toward the upper-middle (put faces there), 'out' = starts close and pulls "
                  "back, 'card' = 'image' is the background and 'card' is the tilted inset card.",
        "shots": shots,
        "watermark": "DUDS", "end_text": "DUDS",
    }
    return m


def render_manifest(manifest_path, preview=False, out=None):
    p = Path(manifest_path)
    m = json.loads((p if p.is_absolute() else ROOT / p).read_text(encoding="utf-8-sig"))
    out = out or ROOT / "output" / f"{m['id']}.mp4"
    return TemplateEdit(m, preview).render(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        print("Other commands:\n  python src/template_edit.py --list\n"
              "  python src/template_edit.py --new <template> <edit_id> [image_folder]")
        sys.exit(1)
    if args[0] == "--list":
        for name, info in list_templates().items():
            print(f"{name}: {info['duration']:.1f}s {info['size'][0]}x{info['size'][1]}  "
                  f"{len(info['slots'])} shots {info['slots']}")
    elif args[0] == "--new":
        path, n = new_manifest(args[1], args[2], args[3] if len(args) > 3 else None)
        print(f"wrote {path} ({n} images assigned) - edit the caption/paths, then build it")
    else:
        print(render_manifest(args[0], preview="--preview" in args))
