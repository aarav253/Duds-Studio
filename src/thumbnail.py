"""Generate a custom 1280x720 YouTube thumbnail for a manifest.

Background priority: a frame from the rendered video (or source clip / first
panel), else a Pexels topic photo, else a procedural theme background.

    python src/thumbnail.py edits/my_manifest.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import effects as fx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THUMB_DIR = ROOT / "output" / "thumbnails"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
TW, TH = 1280, 720


def _font(size):
    try:
        return ImageFont.truetype(FONT_BOLD, size)
    except OSError:
        return ImageFont.load_default()


def _video_frame(path, t_frac=0.35):
    from moviepy.editor import VideoFileClip
    with VideoFileClip(str(path)) as c:
        return c.get_frame(c.duration * t_frac)


def _background(m):
    out = ROOT / "output" / f"{m['id']}.mp4"
    try:
        if out.exists():
            return _video_frame(out)
        if m.get("source_clip") and (ROOT / m["source_clip"]).exists():
            return _video_frame(ROOT / m["source_clip"])
        if m.get("panels"):
            return np.asarray(Image.open(ROOT / m["panels"][0]).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        print(f"  [thumbnail] couldn't grab a frame: {e}")
    import images
    import visuals
    q = (m.get("segments") or [{}])[0].get("query") or m.get("theme") or m.get("title", "")
    p = images.fetch_topic_image(q) if q else None
    if p:
        return np.asarray(Image.open(p).convert("RGB"))
    return visuals.generate_theme_bg(m.get("theme", "default"), TW, TH)


def make_thumbnail(manifest, text=None):
    m = manifest if isinstance(manifest, dict) else json.loads(Path(manifest).read_text(encoding="utf-8-sig"))
    bg = fx.cover_resize_array(_background(m), TW, TH)
    img = Image.fromarray(bg)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Brightness(img).enhance(0.8)
    text = (text or m.get("thumbnail_text") or m.get("title", "")).upper()
    words = text.split()
    if len(words) > 5:
        text = " ".join(words[:5])
    d = ImageDraw.Draw(img)
    size = 120
    while size > 50:
        f = _font(size)
        lines, cur = [], ""
        for w in text.split():
            trial = (cur + " " + w).strip()
            if cur and f.getlength(trial) > TW * 0.9:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
        if len(lines) <= 3:
            break
        size -= 10
    lh = int(size * 1.1)
    y = TH - lh * len(lines) - 50
    for i, line in enumerate(lines):
        color = "#ffe066" if i == len(lines) - 1 else "white"
        d.text((TW / 2, y), line, font=f, fill=color, anchor="mt", stroke_width=8, stroke_fill="black")
        y += lh
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMB_DIR / f"{m['id']}.jpg"
    img.save(out, quality=90)
    return out


if __name__ == "__main__":
    print(make_thumbnail(sys.argv[1]))
