"""Procedurally generated visuals (no downloaded media).

- generate_theme_bg: gradient + emoji-icon still, the fallback for the top strip
  of split-layout videos when there's no Pexels key / results.
- generate_satisfying_bg: bouncing-balls "satisfying" loop for the bottom half.

    python src/visuals.py satisfying clips/satisfying_bg.mp4 [seconds]
    python src/visuals.py theme space out.png
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_EMOJI = r"C:\Windows\Fonts\seguiemj.ttf"

THEMES = {
    "space":   {"colors": [(10, 12, 40), (70, 30, 120)],  "icons": ["🚀", "🪐", "⭐", "🌌"]},
    "ocean":   {"colors": [(0, 60, 110), (0, 160, 180)],  "icons": ["🌊", "🐋", "🐙", "🐠"]},
    "history": {"colors": [(80, 50, 20), (190, 140, 80)], "icons": ["🏛️", "📜", "⚔️", "👑"]},
    "science": {"colors": [(10, 50, 60), (40, 170, 140)], "icons": ["🔬", "🧪", "🧬", "⚛️"]},
    "animals": {"colors": [(30, 80, 30), (150, 190, 80)], "icons": ["🦁", "🐘", "🦊", "🦉"]},
    "food":    {"colors": [(140, 40, 30), (240, 150, 60)], "icons": ["🍕", "🍣", "🍩", "🌮"]},
    "tech":    {"colors": [(15, 15, 30), (30, 90, 200)],  "icons": ["💻", "🤖", "📱", "⚡"]},
    "mystery": {"colors": [(5, 5, 10), (60, 20, 40)],     "icons": ["🔍", "👁️", "🕯️", "❓"]},
    "default": {"colors": [(25, 25, 35), (90, 60, 140)],  "icons": ["💡", "✨", "🔥", "🤯"]},
}


def _gradient(W, H, c0, c1):
    t = np.linspace(0, 1, H)[:, None, None]
    diag = np.linspace(0, 0.25, W)[None, :, None]
    t = np.clip(t * 0.8 + diag, 0, 1)
    g = np.array(c0)[None, None] * (1 - t) + np.array(c1)[None, None] * t
    return g.astype(np.uint8)


def _emoji_font(size):
    try:
        return ImageFont.truetype(FONT_EMOJI, size)
    except OSError:
        return None


def generate_theme_bg(theme, W, H, seed=None):
    """Gradient + large centered emoji + faint scattered icons. Returns HxWx3 uint8."""
    spec = THEMES.get(theme, THEMES["default"])
    rng = np.random.default_rng(seed if seed is not None else abs(hash(theme)) % (2 ** 32))
    img = Image.fromarray(_gradient(W, H, *spec["colors"])).convert("RGBA")
    big = _emoji_font(int(min(W, H) * 0.45))
    small = _emoji_font(int(min(W, H) * 0.12))
    if big is not None:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for _ in range(10):
            icon = spec["icons"][rng.integers(len(spec["icons"]))]
            d.text((rng.uniform(0, W), rng.uniform(0, H)), icon, font=small, anchor="mm",
                   embedded_color=True)
        layer.putalpha(Image.eval(layer.getchannel("A"), lambda a: int(a * 0.25)))
        img = Image.alpha_composite(img, layer)
        d = ImageDraw.Draw(img)
        d.text((W / 2, H / 2), spec["icons"][0], font=big, anchor="mm", embedded_color=True)
    return np.asarray(img.convert("RGB"))


def generate_satisfying_bg(out_path, duration=60, W=1080, H=1280, fps=30, n_balls=9, seed=4):
    """Bouncing balls with soft trails, rendered to an mp4 (no audio)."""
    from moviepy.editor import VideoClip
    rng = np.random.default_rng(seed)
    r = rng.uniform(40, 90, n_balls)
    pos = np.stack([rng.uniform(r, W - r), rng.uniform(r, H - r)], axis=1)
    vel = rng.uniform(-1, 1, (n_balls, 2))
    vel = vel / np.linalg.norm(vel, axis=1, keepdims=True) * rng.uniform(250, 480, (n_balls, 1))
    hues = rng.uniform(0, 1, n_balls)
    colors = [tuple(int(255 * c) for c in _hsv(h, 0.75, 1.0)) for h in hues]
    state = {"t": 0.0, "pos": pos.copy(), "vel": vel.copy(), "trail": None}
    base = _gradient(W, H, (12, 10, 24), (30, 20, 50)).astype(np.float32)

    def step(dt):
        p, v = state["pos"], state["vel"]
        p += v * dt
        for i in range(n_balls):
            for ax, lim in ((0, W), (1, H)):
                if p[i, ax] < r[i]:
                    p[i, ax] = r[i]
                    v[i, ax] = abs(v[i, ax])
                elif p[i, ax] > lim - r[i]:
                    p[i, ax] = lim - r[i]
                    v[i, ax] = -abs(v[i, ax])

    def make(t):
        # advance simulation deterministically in fixed steps up to t
        while state["t"] + 1e-9 < t:
            step(1.0 / fps)
            state["t"] += 1.0 / fps
        img = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(img)
        for i in range(n_balls):
            x, y = state["pos"][i]
            d.ellipse([x - r[i], y - r[i], x + r[i], y + r[i]], fill=colors[i])
        cur = np.asarray(img).astype(np.float32)
        trail = state["trail"]
        trail = cur if trail is None else np.maximum(cur, trail * 0.86)
        state["trail"] = trail
        glow = np.asarray(Image.fromarray(trail.astype(np.uint8)).filter(ImageFilter.GaussianBlur(18)))
        out = np.clip(base + trail * 0.9 + glow.astype(np.float32) * 0.6, 0, 255)
        return out.astype(np.uint8)

    clip = VideoClip(make, duration=duration)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(out_path), fps=fps, codec="libx264", audio=False, preset="medium",
                         ffmpeg_params=["-pix_fmt", "yuv420p"], logger=None)
    return str(out_path)


def _hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "satisfying":
        dur = float(sys.argv[3]) if len(sys.argv) > 3 else 60
        print(generate_satisfying_bg(sys.argv[2], duration=dur))
    elif len(sys.argv) >= 4 and sys.argv[1] == "theme":
        Image.fromarray(generate_theme_bg(sys.argv[2], 1080, 640)).save(sys.argv[3])
        print(sys.argv[3])
    else:
        print(__doc__)
