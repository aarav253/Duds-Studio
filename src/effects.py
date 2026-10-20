"""VFX / transition library used by assemble.py.

Every public function returns (or transforms into) a MoviePy clip, or a plain
HxWx3 uint8 frame where noted, so results compose directly into assemble.py's
`layers` list.

Overlay-style effects (flash, speed lines, border, rim/glow/aura, grain) are
built with `_rgba_videoclip`, because a VideoClip does NOT auto-split a
4-channel make_frame into RGB + mask the way ImageClip does for a static array
(feeding RGBA frames straight in crashes the compositor with a broadcast-shape
error).
"""
import hashlib
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from moviepy.editor import ImageClip, VideoClip

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _rgba_videoclip(make_rgba_frame, duration):
    """Wrap an RGBA make_frame into an RGB VideoClip with a proper mask clip."""
    cache = {}

    def get(t):
        if t not in cache:
            cache.clear()  # rgb + mask ask for the same t back to back; keep one frame
            cache[t] = make_rgba_frame(t)
        return cache[t]

    rgb = VideoClip(lambda t: get(t)[..., :3], duration=duration)
    mask = VideoClip(lambda t: get(t)[..., 3].astype(np.float64) / 255.0,
                     ismask=True, duration=duration)
    return rgb.set_mask(mask)


def _frames_clip(frames, fps):
    """A clip that plays a precomputed list of frames at `fps`."""
    n = len(frames)

    def make(t):
        return frames[min(int(t * fps + 1e-6), n - 1)]

    return VideoClip(make, duration=n / fps)


def _to_rgb(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def cover_resize_array(arr, W, H, center=(0.5, 0.5)):
    """Crop-to-fill `arr` to exactly WxH, keeping the point `center`
    (fractions of the source) as close to frame-center as the edges allow."""
    arr = _to_rgb(arr)
    h, w = arr.shape[:2]
    scale = max(W / w, H / h)
    nw, nh = max(W, int(math.ceil(w * scale))), max(H, int(math.ceil(h * scale)))
    img = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)
    cx, cy = center[0] * nw, center[1] * nh
    x0 = int(round(min(max(cx - W / 2, 0), nw - W)))
    y0 = int(round(min(max(cy - H / 2, 0), nh - H)))
    return np.asarray(img)[y0:y0 + H, x0:x0 + W].copy()


def _ensure_size(arr, W, H):
    arr = _to_rgb(arr)
    if arr.shape[0] != H or arr.shape[1] != W:
        arr = cover_resize_array(arr, W, H)
    return arr


def _seed_from_frames(*frames):
    """Deterministic seed from corner pixels, so re-rendering the same
    manifest reproduces the same 'random' choices."""
    h = hashlib.md5()
    for f in frames:
        for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1), (f.shape[0] // 2, f.shape[1] // 2)):
            h.update(np.ascontiguousarray(f[y, x]).tobytes())
    return int(h.hexdigest()[:8], 16)


def _motion_blur(frame, strength, angle_deg):
    """Directional (linear) motion blur of `strength` px at `angle_deg`."""
    k = int(round(strength))
    if k < 2:
        return frame
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    s = kernel.sum()
    if s <= 0:
        return frame
    kernel /= s
    return cv2.filter2D(frame, -1, kernel, borderType=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------
# Halftone dissolve internals
# --------------------------------------------------------------------------

_DISSOLVE_LEVELS = [0.40, 0.80, 1.00]   # halftone coverage per dissolve keyframe
_BLUR_LEVELS = [1.00, 0.50, 0.15]       # fraction of blur_strength per resolve keyframe


def _ramp_levels(n, table):
    """Resample a keyframe table to n steps (exact table when n == len(table))."""
    if n <= 0:
        return []
    if n == len(table):
        return list(table)
    if n == 1:
        return [table[-1]]
    return list(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(table)), table))


def _ordered_dither(frame, cell=10):
    """Tone-based circular halftone: each cell gets ONE dot whose radius is
    proportional to sqrt(local darkness) - print screentone, not a Bayer grid
    (a uniform threshold grid reads as TV static, not print shading)."""
    frame = _to_rgb(frame)
    H, W = frame.shape[:2]
    gray = frame.astype(np.float32).mean(axis=2) / 255.0
    ch, cw = -(-H // cell), -(-W // cell)
    g = np.pad(gray, ((0, ch * cell - H), (0, cw * cell - W)), mode="edge")
    blocks = g.reshape(ch, cell, cw, cell).mean(axis=(1, 3))
    radius = (cell / math.sqrt(2)) * np.sqrt(np.clip(1.0 - blocks, 0, 1))
    yy, xx = np.mgrid[0:cell, 0:cell].astype(np.float32) + 0.5 - cell / 2
    dist = np.tile(np.hypot(yy, xx), (ch, cw))
    R = np.repeat(np.repeat(radius, cell, axis=0), cell, axis=1)
    ink = (dist < R)[:H, :W]
    out = np.where(ink, 18, 245).astype(np.uint8)
    return np.stack([out] * 3, axis=-1)


def _growing_dither_frame(frame, halftone, level, cell=10, seed=0):
    """Replace a `level` fraction of cells with their halftone version, in a
    randomly scattered order (dots spreading across the image, not a wipe)."""
    if level >= 1.0:
        return halftone
    if level <= 0.0:
        return frame
    H, W = frame.shape[:2]
    ch, cw = -(-H // cell), -(-W // cell)
    ranks = np.random.default_rng(seed).permutation(ch * cw).reshape(ch, cw)
    cell_on = ranks < level * ch * cw
    m = np.repeat(np.repeat(cell_on, cell, axis=0), cell, axis=1)[:H, :W]
    return np.where(m[..., None], halftone, frame)


# --------------------------------------------------------------------------
# Core transitions
# --------------------------------------------------------------------------

def _dissolve_flash_frames(frame_a, W, H, dither_frames, flash_frames, dither_cell, seed,
                           white_flash_chance):
    rng = np.random.default_rng(seed)
    halftone_a = _ordered_dither(frame_a, dither_cell)
    frames = []
    for lvl in _ramp_levels(dither_frames, _DISSOLVE_LEVELS):
        frames.append(_growing_dither_frame(frame_a, halftone_a, lvl, dither_cell, seed))
    flash_val = 255 if rng.random() < white_flash_chance else 0
    flash = np.full((H, W, 3), flash_val, np.uint8)
    frames.extend([flash] * flash_frames)
    return frames, rng


def staged_cut_transition(frame_a, frame_b, W, H, fps=30, dither_frames=3, flash_frames=1,
                          blur_frames=3, dither_cell=10, blur_angle=None, blur_strength=40,
                          white_flash_chance=0.0):
    """THE core cut: still outgoing panel dissolves into halftone dots
    (40%->80%->100%) -> one solid flash frame -> incoming panel under
    directional blur resolving 100%->50%->15%.

    white_flash_chance defaults to 0: on mostly-white manga art a white flash
    is invisible (measured ~241/255 between ~200/255 panels) and reads as a
    soft dissolve with no flash.

    Placement: start at cut_time - (dither_frames + flash_frames) / fps, so the
    flash lands on the cut and the blur tail overlaps blur_frames/fps into the
    new panel's hold (hiding its zoom starting cold)."""
    a = _ensure_size(frame_a, W, H)
    b = _ensure_size(frame_b, W, H)
    seed = _seed_from_frames(a, b)
    frames, rng = _dissolve_flash_frames(a, W, H, dither_frames, flash_frames, dither_cell,
                                         seed, white_flash_chance)
    angle = blur_angle if blur_angle is not None else float(rng.uniform(0, 180))
    for lvl in _ramp_levels(blur_frames, _BLUR_LEVELS):
        frames.append(_motion_blur(b, blur_strength * lvl, angle))
    return _frames_clip(frames, fps)


def growing_inset_cut_transition(frame_a, frame_b, W, H, fps=30, dither_frames=3,
                                 flash_frames=1, grow_frames=10, dither_cell=10,
                                 start_frac=0.18, end_overscan=1.35, anchor_x_frac=0.55,
                                 anchor_y_frac=0.72, white_flash_chance=0.0):
    """Same dissolve -> flash lead-in, but the resolve is the incoming panel
    GROWING from a small inset around an off-center anchor, accelerating
    (ease_in_cubic, like something falling toward the camera).

    The growth scale runs start_frac -> end_overscan; the rendered image scale
    is capped at 1.0, i.e. once the curve passes 1.0 the panel fully covers
    the frame at exactly the scale its own hold starts at (no jump on the cut).

    Placement DIFFERS from staged_cut_transition on purpose: this clip ends
    EXACTLY at the cut (start = cut_time - (dither+flash+grow)/fps). Overlapping
    forward would hide the new panel's overshoot/settle bounce entirely."""
    a = _ensure_size(frame_a, W, H)
    b = _ensure_size(frame_b, W, H)
    seed = _seed_from_frames(a, b)
    frames, _ = _dissolve_flash_frames(a, W, H, dither_frames, flash_frames, dither_cell,
                                       seed, white_flash_chance)
    ax, ay = anchor_x_frac * W, anchor_y_frac * H
    img_b = Image.fromarray(b)
    for k in range(grow_frames):
        x = (k + 1) / grow_frames
        s = start_frac + (end_overscan - start_frac) * ease_in_cubic(x)
        s = min(s, 1.0)
        if s >= 1.0 - 1e-6:
            frames.append(b)
            continue
        nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
        small = img_b.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        # scale about the anchor: the anchor point stays fixed on screen
        canvas.paste(small, (int(round(ax - ax * s)), int(round(ay - ay * s))))
        frames.append(np.asarray(canvas))
    return _frames_clip(frames, fps)


# --------------------------------------------------------------------------
# Easing
# --------------------------------------------------------------------------

def ease_out_back(x, overshoot=1.70158):
    """Standard easeOutBack. WARNING: extremely front-loaded - sampled at only
    3-4 frames ~96% of the motion lands on frame 1 and the rest reads as a
    stepped near-freeze. Fine for 8+ frames; avoid for short punches."""
    x = min(max(x, 0.0), 1.0)
    c3 = overshoot + 1
    return 1 + c3 * (x - 1) ** 3 + overshoot * (x - 1) ** 2


def ease_out_cubic(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_in_cubic(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 3


def ease_overshoot_settle(x, peak_frac=0.7, overshoot_mult=1.45):
    """Rise past 1.0 to overshoot_mult (peaking at x=peak_frac), then settle
    back to exactly 1.0 at x=1. Both phases use ease_out_cubic so each stays
    smooth over a handful of sampled frames. Multiply a (target-1) delta by it."""
    x = min(max(x, 0.0), 1.0)
    if x <= peak_frac:
        return overshoot_mult * ease_out_cubic(x / peak_frac)
    u = (x - peak_frac) / max(1e-9, 1 - peak_frac)
    return overshoot_mult + (1 - overshoot_mult) * ease_out_cubic(u)


def smoothstep_fade(f):
    """1 at f=0 -> 0 at f=1, smoothstep shaped (used for the outro fade)."""
    f = min(max(f, 0.0), 1.0)
    return 1 - (3 * f * f - 2 * f ** 3)


# --------------------------------------------------------------------------
# Zoom
# --------------------------------------------------------------------------

def zoom_frame(frame, s, center=(0.5, 0.5)):
    """Zoom one frame by float scale s>=1 with no integer quantization."""
    if s <= 1.0 + 1e-6:
        return frame
    H, W = frame.shape[:2]
    cw, ch = W / s, H / s
    x0 = min(max(center[0] * W - cw / 2, 0.0), W - cw)
    y0 = min(max(center[1] * H - ch / 2, 0.0), H - ch)
    img = Image.fromarray(frame).resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))
    return np.asarray(img)


def subpixel_zoom(clip, scale_fn, center=(0.5, 0.5)):
    """Replacement for vfx.resize-based zooms. vfx.resize int()-truncates the
    target size every frame, so slow zooms staircase (size changes only every
    3-4 frames, then jumps 1px). Here we crop a continuously floating
    sub-rectangle via PIL's float `box`, output size never changes.
    Scales below 1.0 are clamped to 1.0."""
    return clip.fl(lambda gf, t: zoom_frame(gf(t), max(1.0, scale_fn(t)), center))


# --------------------------------------------------------------------------
# General-purpose effects library (1-14)
# --------------------------------------------------------------------------

# 1
def flash_strobe(W, H, duration, color=(255, 255, 255), peak_opacity=0.9, fade=True):
    """Full-frame colored flash overlay (fades out over `duration` if fade)."""
    base = np.zeros((H, W, 4), np.uint8)
    base[..., :3] = color

    def make(t):
        a = peak_opacity * ((1 - t / duration) if fade else 1.0)
        base[..., 3] = int(255 * max(0.0, a))
        return base

    return _rgba_videoclip(make, duration)


# 2
def camera_shake(clip, intensity_px=18, rotation_deg=1.2, frequency_hz=15, decay=3.0, seed=7):
    """Randomized translate+rotate with exponential decay. A NEW random offset
    is drawn only every 1/frequency_hz seconds (per-frame redraws look like
    noise, not a shake). Deterministic via a fixed seed."""
    W, H = clip.size
    zoom = 1.0 + 2.2 * intensity_px / min(W, H)  # overscan to hide borders

    def fl(gf, t):
        fr = gf(t)
        k = int(t * frequency_hz)
        rng = np.random.default_rng((seed, k))
        amp = math.exp(-decay * t)
        dx, dy = rng.uniform(-1, 1, 2) * intensity_px * amp
        rot = rng.uniform(-1, 1) * rotation_deg * amp
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, zoom)
        M[0, 2] += dx
        M[1, 2] += dy
        return cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    return clip.fl(fl)


# 3
def zoom_punch(clip, amount=0.06, punch_duration=0.2):
    """Quick decelerating punch-in that then holds (ease_out_cubic)."""
    return subpixel_zoom(clip, lambda t: 1 + amount * ease_out_cubic(t / punch_duration))


# 4
def glitch_effect(clip, intensity=0.5, seed=11, fps=30):
    """Horizontal slice displacement + RGB channel offset, re-rolled per frame."""
    def fl(gf, t):
        fr = gf(t).copy()
        H, W = fr.shape[:2]
        rng = np.random.default_rng((seed, int(t * fps)))
        for _ in range(int(3 + 10 * intensity)):
            y = rng.integers(0, H)
            h = rng.integers(4, max(5, int(H * 0.06 * intensity) + 5))
            fr[y:y + h] = np.roll(fr[y:y + h], int(rng.integers(-40, 40) * intensity), axis=1)
        shift = int(8 * intensity)
        fr[..., 0] = np.roll(fr[..., 0], shift, axis=1)
        fr[..., 2] = np.roll(fr[..., 2], -shift, axis=1)
        return fr

    return clip.fl(fl)


def rgb_split_frame(frame, shift_px):
    if shift_px == 0:
        return frame
    out = frame.copy()
    out[..., 0] = np.roll(frame[..., 0], shift_px, axis=1)
    out[..., 2] = np.roll(frame[..., 2], -shift_px, axis=1)
    return out


# 5
def chromatic_aberration(clip, shift_px=6):
    return clip.fl_image(lambda f: rgb_split_frame(f, shift_px))


# 6
def _jagged_panel_polygon(W, H, progress, seed=5, jitter=0.03, n=7):
    """A torn / jittered manga-panel border polygon, grown from center by `progress`."""
    rng = np.random.default_rng(seed)
    cx, cy = W / 2, H / 2
    hw, hh = W / 2 * progress * 1.15, H / 2 * progress * 1.15
    pts = []
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for i in range(4):
        (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % 4]
        for j in range(n):
            f = j / n
            x = x0 + (x1 - x0) * f + rng.uniform(-jitter, jitter)
            y = y0 + (y1 - y0) * f + rng.uniform(-jitter, jitter)
            pts.append((cx + x * hw, cy + y * hh))
    return pts


def panel_shape_reveal(clip, reveal_duration=0.4, seed=5):
    """Reveal a clip through an animated torn-panel-shaped mask."""
    W, H = clip.size

    def mask_frame(t):
        p = ease_out_cubic(t / reveal_duration)
        if p >= 1.0:
            return np.ones((H, W))
        img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(img).polygon(_jagged_panel_polygon(W, H, p, seed), fill=255)
        return np.asarray(img).astype(np.float64) / 255.0

    mask = VideoClip(mask_frame, ismask=True, duration=clip.duration)
    return clip.set_mask(mask)


# 7
def speed_ramp(clip, keyframes):
    """Time-remap via piecewise-linear (time, speed) keyframes, integrated so
    output-time -> source-time stays smooth across a slow-mo dip."""
    ts = np.array([k[0] for k in keyframes], float)
    sp = np.array([k[1] for k in keyframes], float)
    grid = np.linspace(0, clip.duration * 4, 4000)
    speed = np.interp(grid, ts, sp)
    src = np.concatenate([[0], np.cumsum((speed[1:] + speed[:-1]) / 2 * np.diff(grid))])
    out_dur = float(np.interp(clip.duration, src, grid))
    new = clip.fl_time(lambda t: min(float(np.interp(t, grid, src)), clip.duration - 1e-3),
                       keep_duration=False)
    return new.set_duration(out_dur)


# 8
def freeze_frame(clip, t_freeze, hold):
    from moviepy.editor import concatenate_videoclips
    still = ImageClip(clip.get_frame(t_freeze)).set_duration(hold)
    parts = [clip.subclip(0, t_freeze), still]
    if t_freeze < clip.duration:
        parts.append(clip.subclip(t_freeze))
    return concatenate_videoclips(parts)


# 9
def speed_lines(W, H, duration, n_lines=70, center_frac=(0.5, 0.5), inner_frac=0.32,
                color=(255, 255, 255), redraw_every_frames=2, fps=30, seed=21):
    """Radial speed lines. Drawn twice: a wider black stroke pass then the
    color pass, so they stay visible on white manga backgrounds too."""
    cx, cy = center_frac[0] * W, center_frac[1] * H
    r_out = math.hypot(W, H)
    r_in = inner_frac * min(W, H)
    cache = {}

    def make(t):
        k = int(t * fps) // redraw_every_frames
        if k in cache:
            return cache[k]
        rng = np.random.default_rng((seed, k))
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        lines = []
        for _ in range(n_lines):
            a = rng.uniform(0, 2 * math.pi)
            r0 = r_in * rng.uniform(0.85, 1.4)
            half = rng.uniform(0.004, 0.012)
            lines.append((a, r0, half))
        for pass_color, widen in (((0, 0, 0, 255), 2.2), (color + (235,), 1.0)):
            for a, r0, half in lines:
                p0 = (cx + r0 * math.cos(a), cy + r0 * math.sin(a))
                p1 = (cx + r_out * math.cos(a - half * widen), cy + r_out * math.sin(a - half * widen))
                p2 = (cx + r_out * math.cos(a + half * widen), cy + r_out * math.sin(a + half * widen))
                d.polygon([p0, p1, p2], fill=pass_color)
        cache.clear()
        cache[k] = np.asarray(img)
        return cache[k]

    return _rgba_videoclip(make, duration)


# 10
def panel_border(W, H, duration, thickness=18, color=(255, 255, 255), inner_line=4):
    arr = np.zeros((H, W, 4), np.uint8)
    arr[:thickness, :] = arr[-thickness:, :] = color + (255,)
    arr[:, :thickness] = arr[:, -thickness:] = color + (255,)
    t2 = thickness + inner_line
    arr[thickness:t2, thickness:-thickness] = (0, 0, 0, 255)
    arr[-t2:-thickness, thickness:-thickness] = (0, 0, 0, 255)
    arr[thickness:-thickness, thickness:t2] = (0, 0, 0, 255)
    arr[thickness:-thickness, -t2:-thickness] = (0, 0, 0, 255)
    return ImageClip(arr).set_duration(duration)


# 11
def ink_line_overlay(clip, asset_path=None, strength=0.8, canny=(80, 160)):
    """Multiply-blend ink lines on top: Canny edges (no assets) or an overlay image."""
    W, H = clip.size
    asset = None
    if asset_path:
        asset = cover_resize_array(np.asarray(Image.open(asset_path).convert("RGB")), W, H)
        asset = asset.astype(np.float32) / 255.0

    def fl(frame):
        f = frame.astype(np.float32) / 255.0
        if asset is not None:
            mult = asset
        else:
            edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), *canny)
            mult = (1.0 - edges.astype(np.float32) / 255.0)[..., None]
        out = f * (1 - strength + strength * mult)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    return clip.fl_image(fl)


# 12
def sketch_transition(clip, hold=0.25, fade=0.3):
    """A brief pencil-sketch hold of the first frame, cross-fading into the
    normal clip - 'the art resolving into itself'."""
    from moviepy.editor import CompositeVideoClip
    first = clip.get_frame(0)
    gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    inv = 255 - cv2.GaussianBlur(255 - gray, (21, 21), 0)
    sketch = cv2.divide(gray, np.maximum(inv, 1), scale=256)
    sketch = np.stack([np.clip(sketch, 0, 255).astype(np.uint8)] * 3, axis=-1)
    sk = ImageClip(sketch).set_duration(hold + fade).crossfadeout(fade)
    return CompositeVideoClip([clip, sk], size=clip.size).set_duration(clip.duration)


# 13 (see directional_smear_drip below)

# 14
def shake_zoom_combo(clip, amount=0.06, punch_duration=0.18, intensity_px=16):
    return camera_shake(zoom_punch(clip, amount, punch_duration), intensity_px=intensity_px)


# --------------------------------------------------------------------------
# Grade / grain
# --------------------------------------------------------------------------

def grade_frame(frame, contrast=1.35, black_crush=0.12, extra_contrast=0.0):
    """Desaturate -> high contrast -> crushed blacks: the global B&W manga look."""
    f = frame.astype(np.float32) / 255.0
    g = f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114
    g = (g - 0.5) * (contrast + extra_contrast) + 0.5
    g = (g - black_crush) / (1.0 - black_crush)
    g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def film_grain_frame(frame, t, opacity=0.10, fps=30, seed=99):
    """Add animated grain to a frame (new noise pattern each frame index)."""
    H, W = frame.shape[:2]
    rng = np.random.default_rng((seed, int(t * fps)))
    small = rng.standard_normal((H // 2 + 1, W // 2 + 1)).astype(np.float32)
    noise = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    out = frame.astype(np.float32) + (opacity * 128.0) * noise[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def film_grain_layer(clip, opacity=0.10, fps=30, seed=99):
    """Continuous animated grain applied over an entire (already composited) clip."""
    return clip.fl(lambda gf, t: film_grain_frame(gf(t), t, opacity, fps, seed))


# --------------------------------------------------------------------------
# Shot-scale helpers
# --------------------------------------------------------------------------

def inset_blur_reveal_frame(source, W, H, inset_frac=0.40, bg_scale=1.4, blur_radius=20,
                            bg_lift=0.45):
    """Small sharp crop over an enlarged, blurred duplicate of the same image.
    bg_lift blends the blurred bg toward white AFTER blurring: blur preserves
    local average, so dark regions (hair, sleeves) otherwise stay a ~20/255
    blob that reads as a black splotch. Not used by any active type."""
    src = _to_rgb(source)
    bg = cover_resize_array(src, int(W * bg_scale), int(H * bg_scale))
    ox, oy = (bg.shape[1] - W) // 2, (bg.shape[0] - H) // 2
    bg = bg[oy:oy + H, ox:ox + W]
    bg = np.asarray(Image.fromarray(bg).filter(ImageFilter.GaussianBlur(blur_radius)))
    bg = (bg.astype(np.float32) * (1 - bg_lift) + 255 * bg_lift).astype(np.uint8)
    iw, ih = int(W * inset_frac * 1.6), int(H * inset_frac)
    inset = cover_resize_array(src, iw, ih)
    out = bg.copy()
    x0, y0 = (W - iw) // 2, (H - ih) // 2
    out[y0:y0 + ih, x0:x0 + iw] = inset
    return out


def canted_overscan(angle_deg, W, H):
    """Exact minimum overscan so a WxH frame rotated by angle has no empty
    corners: every rotated canvas corner must stay inside the scaled source."""
    th = math.radians(abs(angle_deg))
    c, s = math.cos(th), math.sin(th)
    return max(c + (H / W) * s, c + (W / H) * s)


def canted_still(source, W, H, angle_deg=5.0, overscan=None):
    """A still held at a slight tilt. Overscan is computed from the angle and
    aspect ratio, not guessed (a fixed 1.14 exposed black corners)."""
    base = cover_resize_array(source, W, H)
    s = overscan if overscan is not None else canted_overscan(angle_deg, W, H) * 1.002
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, s)
    return cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# 13
def directional_smear_drip(frame, strength=60, drip_amount=40, direction_down=True, n_taps=10):
    """Directional streak blur with a downward 'drip' bias. Divides by the
    ACTUAL number of contributing taps: small strengths round every offset to
    0, and dividing by a fixed n_taps+1 then produced near-black output."""
    f = frame.astype(np.float32)
    acc = f.copy()
    count = 1
    sign = 1 if direction_down else -1
    for i in range(1, n_taps + 1):
        frac = i / n_taps
        dx = int(round(strength * frac))
        dy = int(round(drip_amount * frac * frac)) * sign
        if dx == 0 and dy == 0:
            continue
        acc += np.roll(np.roll(f, dx, axis=1), dy, axis=0)
        count += 1
    return np.clip(acc / count, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Accent overlays (small positioned patches, not full-frame, for speed)
# --------------------------------------------------------------------------

def rim_light_flare_layer(W, H, duration, x_frac=0.88, width_frac=0.09, fps=30):
    """One bright soft vertical streak near the right edge, CONSTANT intensity
    (no fade-in). Because it's full-bright from its first frame, callers must
    trim its window back from adjacent transition lead-ins."""
    pw = max(4, int(W * width_frac * 3))
    xs = np.arange(pw) - pw / 2
    sigma = W * width_frac / 2.5
    horiz = np.exp(-(xs / sigma) ** 2)
    ys = np.linspace(0, 1, H)
    vert = np.clip(np.minimum(ys, 1 - ys) / 0.12, 0, 1)
    alpha = (np.outer(vert, horiz) * 0.85 * 255).astype(np.uint8)
    arr = np.zeros((H, pw, 4), np.uint8)
    arr[..., :3] = 255
    arr[..., 3] = alpha
    x0 = int(W * x_frac - pw / 2)
    return ImageClip(arr).set_duration(duration).set_position((x0, 0))


def glow_burst_layer(W, H, duration, pos_frac=(0.7, 0.32), diameter_frac=0.09,
                     attack_frames=3, fps=30):
    """Small bright circular glint; ramps up over attack_frames then holds."""
    d = max(8, int(W * diameter_frac))
    size = d * 3
    yy, xx = np.mgrid[0:size, 0:size] - size / 2
    r = np.hypot(xx, yy) / (d / 2)
    core = np.clip(1 - r, 0, 1) ** 0.6
    halo = np.exp(-(r / 1.4) ** 2) * 0.55
    base_alpha = np.clip(np.maximum(core, halo), 0, 1)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = 255

    def make(t):
        ramp = min(1.0, (t * fps + 1) / max(1, attack_frames))
        rgba[..., 3] = (base_alpha * ramp * 255).astype(np.uint8)
        return rgba

    x0 = int(W * pos_frac[0] - size / 2)
    y0 = int(H * pos_frac[1] - size / 2)
    return _rgba_videoclip(make, duration).set_position((x0, y0))


def aura_swirl_layer(W, H, duration, fps=30, seed=3, n_blobs=4, max_alpha=0.32):
    """Dark translucent smoke swirl in the right third, blobs orbiting.
    Blob alpha combined with np.maximum, NOT summed: summing 5-7 overlapping
    translucent blobs saturated past 1.0 and clipped to opaque black."""
    zw, zh = W // 3, H
    ds = 4  # compute at 1/4 res then upscale
    sw, sh = zw // ds, zh // ds
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    rng = np.random.default_rng(seed)
    blobs = []
    for _ in range(n_blobs):
        blobs.append(dict(
            cx=rng.uniform(0.3, 0.8) * sw, cy=rng.uniform(0.15, 0.85) * sh,
            orbit=rng.uniform(0.05, 0.15) * sw, speed=rng.uniform(0.4, 1.0) * rng.choice([-1, 1]),
            phase=rng.uniform(0, 2 * math.pi), r=rng.uniform(0.18, 0.28) * min(sw, sh * 0.5),
            a=rng.uniform(0.6, 1.0) * max_alpha))

    def make(t):
        alpha = np.zeros((sh, sw), np.float32)
        for b in blobs:
            ang = b["phase"] + b["speed"] * t * 2
            cx = b["cx"] + b["orbit"] * math.cos(ang)
            cy = b["cy"] + b["orbit"] * math.sin(ang) - t * 3.0
            d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (b["r"] ** 2)
            alpha = np.maximum(alpha, b["a"] * np.exp(-d2))
        # fade toward the zone's left edge so it doesn't end in a hard line
        alpha *= np.clip(xx / (sw * 0.35), 0, 1)
        a = cv2.resize(alpha, (zw, zh), interpolation=cv2.INTER_LINEAR)
        rgba = np.zeros((zh, zw, 4), np.uint8)
        rgba[..., :3] = (18, 16, 22)
        rgba[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return rgba

    return _rgba_videoclip(make, duration).set_position((W - zw, 0))


# --------------------------------------------------------------------------
# Legacy transitions - NOT called by any active assemble function.
# Kept for reuse; don't wire more than one transition style into a type's
# default cut loop without a deliberate, cut-index-gated reason.
# --------------------------------------------------------------------------

def photo_insert_composite(bg_frame, photo, W, H, inset_frac=0.55, border=10):
    out = _ensure_size(bg_frame, W, H).copy()
    iw = int(W * inset_frac)
    ih = int(iw * 0.75)
    inset = cover_resize_array(photo, iw, ih)
    x0, y0 = (W - iw) // 2, (H - ih) // 2
    out[y0 - border:y0 + ih + border, x0 - border:x0 + iw + border] = 255
    out[y0:y0 + ih, x0:x0 + iw] = inset
    return out


def manga_wakeup_transition(frame_a, frame_b, W, H, fps=30, frames=8):
    """Eyelid 'blink': black bars close over A then open on B."""
    a, b = _ensure_size(frame_a, W, H), _ensure_size(frame_b, W, H)
    out = []
    half = frames // 2
    for i in range(frames):
        src = a if i < half else b
        p = (i + 1) / half if i < half else 1 - (i - half + 1) / (frames - half)
        f = src.copy()
        bar = int(H / 2 * ease_out_cubic(p))
        f[:bar] = 0
        f[H - bar:] = 0
        out.append(f)
    return _frames_clip(out, fps)


def whip_smear_transition(frame_a, frame_b, W, H, fps=30, frames=6, strength=120):
    a, b = _ensure_size(frame_a, W, H), _ensure_size(frame_b, W, H)
    out = []
    for i in range(frames):
        p = i / max(1, frames - 1)
        src = a if p < 0.5 else b
        s = strength * (1 - abs(p - 0.5) * 2)
        out.append(directional_smear_drip(src, s, s * 0.3))
    return _frames_clip(out, fps)


def halftone_bridge_transition(frame_a, frame_b, W, H, fps=30, frames=6, cell=10):
    a, b = _ensure_size(frame_a, W, H), _ensure_size(frame_b, W, H)
    ha, hb = _ordered_dither(a, cell), _ordered_dither(b, cell)
    half = frames // 2
    out = [_growing_dither_frame(a, ha, (i + 1) / half, cell, 1) for i in range(half)]
    out += [_growing_dither_frame(b, hb, 1 - (i + 1) / (frames - half), cell, 2)
            for i in range(frames - half)]
    return _frames_clip(out, fps)
