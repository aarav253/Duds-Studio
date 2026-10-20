"""The renderer: one JSON manifest -> one finished vertical .mp4.

Usage:
    python src/assemble.py edits/my_manifest.json

Content types: edit, meme, narration (+ layout "split"), manga_edit, manga_hype_edit.
"""
import functools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# MoviePy 1.0.3's resize fx calls Image.ANTIALIAS, which modern Pillow removed.
# Patch the constant back in. Do NOT downgrade Pillow instead: a Pillow old
# enough to still have ANTIALIAS ships a FreeType build that segfaults on
# multi-word text rendering.
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

from moviepy.editor import (AudioFileClip, ColorClip, CompositeAudioClip,  # noqa: E402
                            CompositeVideoClip, ImageClip, VideoClip, VideoFileClip,
                            afx, concatenate_videoclips, vfx)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import effects as fx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
FONT_EMOJI = r"C:\Windows\Fonts\seguiemj.ttf"


# ==========================================================================
# Config / paths
# ==========================================================================

def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p):
    """Manifest paths are relative to the project root."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_manifest(path):
    with open(resolve(path), "r", encoding="utf-8-sig") as f:
        return json.load(f)


def words_sidecar_path(audio_path):
    p = Path(audio_path)
    return p.with_name(p.stem + "_words.json")


# ==========================================================================
# Text rendering
# ==========================================================================

@functools.lru_cache(maxsize=64)
def get_font(size, path=FONT_BOLD):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        try:
            return ImageFont.load_default(size)
        except TypeError:
            return ImageFont.load_default()


def _wrap_lines(text, font, max_w, stroke=0):
    if max_w is None:
        return text.split("\n")
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            trial = (cur + " " + word).strip()
            if cur and font.getlength(trial) + 2 * stroke > max_w:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
    return lines


def render_text_rgba(text, fontsize, color="white", stroke_width=0, stroke_color="black",
                     max_width=None, bg=None, pad=(30, 18), box_size=None, radius=0,
                     font_path=FONT_BOLD, line_spacing=1.12):
    """Render (wrapped, centered) text to an RGBA numpy array.
    bg: optional RGBA tuple for a backdrop box; box_size forces (w, h)."""
    font = get_font(int(fontsize), font_path)
    inner_w = None
    if max_width:
        inner_w = max_width - 2 * pad[0]
    elif box_size:
        inner_w = box_size[0] - 2 * pad[0]
    lines = _wrap_lines(text, font, inner_w, stroke_width)
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * line_spacing)
    widths = [font.getlength(l) + 2 * stroke_width for l in lines]
    text_w = int(max(widths) if widths else 0)
    text_h = line_h * len(lines) + 2 * stroke_width
    w = box_size[0] if box_size else text_w + 2 * pad[0]
    h = box_size[1] if box_size else text_h + 2 * pad[1]
    img = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if bg is not None:
        if radius:
            d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=tuple(bg))
        else:
            d.rectangle([0, 0, w, h], fill=tuple(bg))
    y = (h - text_h) / 2 + stroke_width
    for line, lw in zip(lines, widths):
        x = (w - lw) / 2 + stroke_width
        d.text((x, y), line, font=font, fill=color, stroke_width=stroke_width,
               stroke_fill=stroke_color)
        y += line_h
    return np.asarray(img)


def text_image_clip(text, duration, **kw):
    return ImageClip(render_text_rgba(text, **kw)).set_duration(duration)


# ==========================================================================
# Framing helpers
# ==========================================================================

def fit_vertical(clip, W, H):
    """Crop-to-fill a video clip to WxH (center crop)."""
    cw, ch = clip.size
    scale = max(W / cw, H / ch)
    if abs(scale - 1.0) > 1e-3:
        clip = clip.resize(scale)
    nw, nh = clip.size
    return clip.crop(x_center=nw / 2, y_center=nh / 2, width=W, height=H)


def load_image_array(path):
    return np.asarray(Image.open(resolve(path)).convert("RGB"))


def content_center(arr):
    """Weighted centroid of 'ink' (dark / edge) content as (x_frac, y_frac)."""
    small = Image.fromarray(arr).convert("L")
    small.thumbnail((256, 256))
    g = np.asarray(small).astype(np.float32) / 255.0
    ink = np.clip(1.0 - g, 0, 1) ** 1.5
    gy, gx = np.gradient(g)
    ink = ink + 2.0 * np.hypot(gx, gy)
    total = ink.sum()
    if total < 1e-6:
        return 0.5, 0.5
    ys, xs = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    cx = float((ink * xs).sum() / total) / g.shape[1]
    cy = float((ink * ys).sum() / total) / g.shape[0]
    # pull 25% toward geometric center so a stray dark corner can't dominate
    return 0.75 * cx + 0.125, 0.75 * cy + 0.125


def content_aware_crop_array(arr, W, H):
    """Crop-to-fill centered on the image's actual ink content."""
    return fx.cover_resize_array(arr, W, H, center=content_center(arr))


def fit_letterbox(arr, W, H, darken=0.4, blur=30):
    """Fit the whole image inside WxH over a blurred, darkened copy of itself."""
    bg = fx.cover_resize_array(arr, W, H)
    bg = np.asarray(Image.fromarray(bg).filter(ImageFilter.GaussianBlur(blur)))
    bg = (bg.astype(np.float32) * darken).astype(np.uint8)
    h, w = arr.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    fg = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.LANCZOS))
    out = bg.copy()
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = fg
    return out


def fit_panel(arr, W, H, mode="crop"):
    return fit_letterbox(arr, W, H) if mode == "letterbox" else content_aware_crop_array(arr, W, H)


# ==========================================================================
# Branding
# ==========================================================================

def _cta_layer(W, H, D, cfg):
    b = cfg["branding"]
    dur = min(b["cta_duration_seconds"], D)
    cta = text_image_clip(b["cta_text"], dur, fontsize=int(W * 0.065), color="#ffe066",
                          stroke_width=5, stroke_color="black", max_width=int(W * 0.9))
    return cta.set_start(D - dur).set_position(("center", int(H * 0.80))).crossfadein(0.25)


def _brand_layers(W, H, D, cfg, avoid_band=None):
    """edit / meme: full-screen centered intro text + CTA near the end.
    avoid_band=(top, bottom): move the intro above a caption occupying that band."""
    b = cfg["branding"]
    intro_d = min(b["intro_duration_seconds"], D)
    arr = render_text_rgba(b["intro_text"], int(W * 0.16), color="white", stroke_width=8,
                           stroke_color="black")
    y = (H - arr.shape[0]) // 2
    if avoid_band and y < avoid_band[1] and y + arr.shape[0] > avoid_band[0]:
        y = max(0, avoid_band[0] - arr.shape[0] - int(H * 0.04))
    intro = ImageClip(arr).set_duration(intro_d).set_start(0).set_position(("center", y))
    intro = intro.crossfadeout(min(0.3, intro_d / 2))
    return [intro, _cta_layer(W, H, D, cfg)]


def _brand_layers_narration(W, H, D, cfg, cta=True, corner="top_left"):
    """narration / manga_edit: small persistent corner watermark (+ CTA)."""
    b = cfg["branding"]
    fs = cfg["content_types"]["narration"].get("watermark_fontsize", 34)
    wm_arr = render_text_rgba(b["intro_text"], fs, color=(255, 255, 255, 190),
                              stroke_width=2, stroke_color=(0, 0, 0, 160), pad=(4, 2))
    wm = ImageClip(wm_arr).set_duration(D)
    margin = int(W * 0.04)
    if corner == "top_right":
        pos = (W - wm_arr.shape[1] - margin, margin)
    else:
        pos = (margin, margin)
    layers = [wm.set_position(pos)]
    if cta:
        layers.append(_cta_layer(W, H, D, cfg))
    return layers


# ==========================================================================
# Audio helpers
# ==========================================================================

def fit_audio(clip, D):
    """Loop if shorter than D, trim if longer. A plain subclip(0, min(...))
    silently truncates a short track and leaves the tail of the video silent."""
    if clip.duration < D:
        return afx.audio_loop(clip, duration=D)
    return clip.subclip(0, D)


def load_music(path, D, volume):
    if not path:
        return None
    p = resolve(path)
    if not p.exists():
        print(f"  [warn] music not found: {p} - rendering without it")
        return None
    return fit_audio(AudioFileClip(str(p)), D).volumex(volume)


def mix_audio(tracks, D):
    tracks = [t for t in tracks if t is not None]
    if not tracks:
        return None
    return CompositeAudioClip(tracks).set_duration(D)


def _apply_fade(audio, factor_fn):
    def fl(gf, t):
        a = gf(t)
        f = factor_fn(np.asarray(t, dtype=np.float64))
        if np.ndim(t) == 0:
            return a * float(f)
        return a * np.asarray(f).reshape(-1, 1)
    return audio.fl(fl, keep_duration=True)


# ==========================================================================
# Beat / onset / drop analysis
# ==========================================================================

@functools.lru_cache(maxsize=8)
def _load_audio_mono(path):
    import librosa
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    return y, sr


@functools.lru_cache(maxsize=8)
def _music(music_path):
    """Neural analysis (beat_this + Demucs drum stem), cached on disk.
    music_analysis falls back to librosa by itself if the models fail."""
    import music_analysis
    return music_analysis.analyze(music_path, verbose=True)


@functools.lru_cache(maxsize=8)
def _detect_beats(music_path):
    """Beat grid from beat_this."""
    return tuple(_music(music_path)["beats"])


@functools.lru_cache(maxsize=8)
def _detect_downbeats(music_path):
    return tuple(_music(music_path)["downbeats"])


@functools.lru_cache(maxsize=8)
def _detect_onsets(music_path, min_strength=0.25):
    """Real drum hits (onsets on the Demucs drum stem) - swing naturally between
    beats. Preferred for cut placement: a steady grid reads as a fixed interval,
    not 'on the music', and full-mix onsets also fire on vocals/synths."""
    import music_analysis
    return tuple(music_analysis.strong_onsets(_music(music_path), min_strength))


@functools.lru_cache(maxsize=8)
def _detect_drop(music_path, min_start=1.5, window=0.75):
    """Drop from drums+bass energy, snapped to a downbeat."""
    d = _music(music_path).get("drop")
    if d is not None and d >= min_start:
        return float(d)
    return _detect_drop_librosa(music_path, min_start, window)


@functools.lru_cache(maxsize=8)
def _detect_drop_librosa(music_path, min_start=1.5, window=0.75):
    """Biggest sustained energy jump: mean RMS over ~0.75s AFTER vs BEFORE
    each point (not a single-frame lookback, so one early transient can't win).
    Returns None if nothing clearly drop-like."""
    import librosa
    from scipy.ndimage import uniform_filter1d
    y, sr = _load_audio_mono(music_path)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms = uniform_filter1d(rms, size=max(1, int(0.25 * sr / hop)))
    w = max(1, int(window * sr / hop))
    if len(rms) < 2 * w + 2:
        return None
    csum = np.concatenate([[0], np.cumsum(rms)])
    idx = np.arange(w, len(rms) - w)
    before = (csum[idx] - csum[idx - w]) / w
    after = (csum[idx + w] - csum[idx]) / w
    score = after - before
    times = librosa.frames_to_time(idx, sr=sr, hop_length=hop)
    score[times < min_start] = -np.inf
    best = int(np.argmax(score))
    if not np.isfinite(score[best]) or score[best] < 0.2 * float(rms.max()):
        return None
    return float(times[best])


def _tile_times(times, track_dur, D):
    out = list(times)
    k = 1
    while track_dur > 0 and track_dur * k < D:
        out += [t + track_dur * k for t in times]
        k += 1
    return sorted(t for t in out if t < D)


def snap(t, fps):
    """Snap a timestamp to the frame grid. A cut at e.g. 9.98s (frame 299.4)
    otherwise lands mid-frame and which frame carries the flash depends on
    rounding - snapping makes every transition frame-exact and auditable."""
    return round(t * fps) / fps


def _min_gap_filter(times, gap, after=0.0):
    out = []
    last = after
    for t in sorted(times):
        if t - last >= gap:
            out.append(t)
            last = t
    return out


# ==========================================================================
# Shared zoom curve builder (manga types)
# ==========================================================================

def make_scale_fn(punch_start, punch_dur, amount, peak_frac, overshoot_mult,
                  drift_amp, drift_period):
    """Three continuous phases:
    1) hold at 1.0 while the incoming transition still occludes the layer,
    2) overshoot-and-settle punch to 1+amount,
    3) bounded sine 'breathing' for the REST of the hold, however long.
    Never asymptotes to a constant (a flat settle reads as a frozen frame on
    long holds) and never creeps linearly (compounds into a runaway zoom)."""
    target = 1.0 + amount

    def scale(t):
        if t < punch_start:
            return 1.0
        u = t - punch_start
        if u < punch_dur:
            return 1.0 + amount * fx.ease_overshoot_settle(u / punch_dur, peak_frac, overshoot_mult)
        return target + drift_amp * math.sin(2 * math.pi * (u - punch_dur) / drift_period)

    return scale


# ==========================================================================
# Type: edit
# ==========================================================================

def _source_subclip(m, cfg, W, H):
    src = resolve(m["source_clip"])
    if not src.exists():
        raise FileNotFoundError(f"source_clip not found: {src}")
    clip = VideoFileClip(str(src))
    start = float(m.get("start", 0))
    end = float(m.get("end", clip.duration))
    end = min(end, clip.duration, start + cfg["output"]["max_duration_seconds"])
    return clip.subclip(start, end)


def _assemble_edit(m, cfg, W, H):
    clip = fit_vertical(_source_subclip(m, cfg, W, H), W, H)
    D = clip.duration
    vol = m.get("original_clip_volume", cfg["audio"]["original_clip_volume"])
    audio = mix_audio([clip.audio.volumex(vol) if clip.audio else None,
                       load_music(m.get("music"), D, m.get("music_volume", cfg["audio"]["music_volume"]))], D)
    comp = CompositeVideoClip([clip] + _brand_layers(W, H, D, cfg), size=(W, H)).set_duration(D)
    return comp.set_audio(audio)


# ==========================================================================
# Type: meme
# ==========================================================================

def _assemble_meme(m, cfg, W, H):
    mc = cfg["content_types"]["meme"]
    clip = fit_vertical(_source_subclip(m, cfg, W, H), W, H)
    D = clip.duration
    caption = m.get("caption") or m.get("meme_caption") or m.get("title", "")
    arr = render_text_rgba(caption, mc["caption_fontsize"], color="black", max_width=int(W * 0.9),
                           bg=(255, 255, 255, 240), pad=(36, 24), radius=24)
    y = int(H * mc["caption_position_frac"] - arr.shape[0] / 2)
    y = min(max(y, 0), H - arr.shape[0])
    cap = ImageClip(arr).set_duration(D).set_position(("center", y))
    vol = m.get("original_clip_volume", cfg["audio"]["original_clip_volume"])
    audio = mix_audio([clip.audio.volumex(vol) if clip.audio else None,
                       load_music(m.get("music"), D, m.get("music_volume", mc["music_volume"]))], D)
    brand = _brand_layers(W, H, D, cfg, avoid_band=(y, y + arr.shape[0]))
    comp = CompositeVideoClip([clip, cap] + brand, size=(W, H)).set_duration(D)
    return comp.set_audio(audio)


# ==========================================================================
# Type: narration (captions shared with split layout)
# ==========================================================================

def _norm_word(w):
    return re.sub(r"[^\w']", "", w.lower())


def _align_words(words, data, lookahead=4):
    """Greedy-align script words to TTS word-boundary tokens. Tokens can differ
    from the script ("1,000" is spoken as two words), so match normalized text
    within a small look-ahead and interpolate any script word left unmatched."""
    toks = [_norm_word(d["word"]) for d in data]
    times = [None] * len(words)
    j = 0
    for i, w in enumerate(words):
        nw = _norm_word(w)
        for k in range(j, min(j + lookahead, len(toks))):
            if toks[k] and (toks[k] == nw or nw.startswith(toks[k]) or toks[k].startswith(nw)):
                times[i] = (float(data[k]["start"]), float(data[k]["end"]))
                j = k + 1
                break
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        k = len(data)
        return [(float(data[min(k - 1, i * k // len(words))]["start"]),
                 float(data[min(k - 1, i * k // len(words))]["end"])) for i in range(len(words))]
    for i in range(len(words)):
        if times[i] is None:
            prev = max((q for q in known if q < i), default=None)
            nxt = min((q for q in known if q > i), default=None)
            if prev is None:
                times[i] = (times[nxt][0], times[nxt][0])
            elif nxt is None:
                times[i] = (times[prev][1], times[prev][1] + 0.2)
            else:
                f = (i - prev) / (nxt - prev)
                s = times[prev][1] + f * (times[nxt][0] - times[prev][1])
                times[i] = (s, s)
    return times


def word_timings(words, audio_path, audio_dur):
    """(start, end) per script word. Uses edge-tts's real WordBoundary sidecar
    if present; otherwise a proportional word-length estimate."""
    side = words_sidecar_path(audio_path) if audio_path else None
    if side and side.exists():
        try:
            data = json.loads(side.read_text(encoding="utf-8-sig"))
            if data:
                return _align_words(words, data)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    weights = np.array([len(_norm_word(w)) + 2 for w in words], float)
    usable = audio_dur * 0.97
    edges = np.concatenate([[0], np.cumsum(weights / weights.sum() * usable)])
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(words))]


def caption_layers(script, timings, D, W, cfg, region_top, region_h, style="phrase"):
    """Word-synced captions placed inside the vertical region [top, top+h)."""
    nc = cfg["content_types"]["narration"]
    words = script.split()
    lead = nc["caption_lead_seconds"]
    hook_n = nc["hook_words"]
    layers = []
    if style == "word_by_word":
        chunks = [[i] for i in range(len(words))]
    else:
        per = nc["words_per_chunk"]
        chunks = [list(range(i, min(i + per, len(words)))) for i in range(0, len(words), per)]
    starts = [max(0.0, timings[c[0]][0] - lead) for c in chunks]
    for ci, c in enumerate(chunks):
        t0 = starts[ci]
        t1 = starts[ci + 1] if ci + 1 < len(chunks) else min(D, timings[c[-1]][1] + 0.6)
        if t1 <= t0:
            continue
        text = " ".join(words[i] for i in c)
        is_hook = c[0] < hook_n
        if style == "word_by_word":
            fs = int(nc["hook_fontsize"] * 1.7 if is_hook else nc["caption_fontsize"] * 1.8)
            arr = render_text_rgba(text.upper(), fs, color=nc["hook_color"] if is_hook else "white",
                                   stroke_width=max(6, fs // 11), stroke_color="black",
                                   max_width=int(W * 0.94))
        else:
            fs = nc["hook_fontsize"] if is_hook else nc["caption_fontsize"]
            arr = render_text_rgba(text, fs, color=nc["hook_color"] if is_hook else "white",
                                   stroke_width=4, stroke_color="black", max_width=int(W * 0.88),
                                   bg=(0, 0, 0, nc["caption_backdrop_alpha"]), radius=22)
        frac = 0.5 if style == "word_by_word" else nc["caption_position_frac"]
        y = int(region_top + region_h * frac - arr.shape[0] / 2)
        y = min(max(y, region_top), region_top + region_h - arr.shape[0])
        clip = ImageClip(arr).set_duration(t1 - t0).set_start(t0).set_position(("center", y))
        if not is_hook and style != "word_by_word":
            clip = clip.crossfadein(min(0.08, (t1 - t0) / 3))
        layers.append(clip)
    return layers


def pop_keyword_layers(script, keywords, timings, D, W, region_top, region_h):
    norm = [_norm_word(w) for w in script.split()]
    layers = []
    for kw in keywords or []:
        toks = [_norm_word(w) for w in kw.split() if _norm_word(w)]
        hit = None
        for i in range(len(norm) - len(toks) + 1):
            if norm[i:i + len(toks)] == toks:
                hit = i
                break
        if hit is None:
            print(f"  [warn] pop_keyword not found verbatim in script, skipped: {kw!r}")
            continue
        t0 = timings[hit][0]
        dur = min(0.9, max(0.1, D - t0))
        arr = render_text_rgba(kw.upper(), int(W * 0.13), color="#ffe066", stroke_width=10,
                               stroke_color="black", max_width=int(W * 0.95))
        y = int(region_top + region_h * 0.32 - arr.shape[0] / 2)
        clip = ImageClip(arr).set_duration(dur).set_start(t0)
        clip = clip.resize(lambda t: 0.6 + 0.4 * fx.ease_out_back(min(1.0, t / 0.18)))
        clip = clip.set_position(lambda t, a=arr, y=y: (
            (W - a.shape[1] * (0.6 + 0.4 * fx.ease_out_back(min(1.0, t / 0.18)))) / 2, y))
        layers.append(clip.crossfadeout(0.15))
    return layers


def _narration_audio(m, cfg):
    path = resolve(m["narration_audio"])
    if not path.exists():
        raise FileNotFoundError(f"narration_audio not found: {path} (generate it with tts.synthesize)")
    return path, AudioFileClip(str(path))


def _assemble_narration(m, cfg, W, H):
    if m.get("layout") == "split":
        return _assemble_narration_split(m, cfg, W, H)
    nc = cfg["content_types"]["narration"]
    audio_path, voice = _narration_audio(m, cfg)
    D = min(cfg["output"]["max_duration_seconds"], voice.duration + 0.4)
    src = _source_subclip(m, cfg, W, H)
    if src.duration < D:
        # freeze on the last frame rather than letting the container outrun the picture
        src = vfx.freeze(src, t="end", total_duration=D, padding_end=1.0 / cfg["output"]["fps"])
    src = fit_vertical(src.subclip(0, D), W, H)
    script = m["script"]
    words = script.split()
    timings = word_timings(words, audio_path, voice.duration)
    layers = [src]
    layers += caption_layers(script, timings, D, W, cfg, 0, H, m.get("caption_style", "phrase"))
    layers += pop_keyword_layers(script, m.get("pop_keywords"), timings, D, W, 0, H)
    layers += _brand_layers_narration(W, H, D, cfg)
    music = None
    if m.get("music") and (cfg["music_selection"]["include_in_narration"] or m.get("force_music")):
        music = load_music(m["music"], D, nc["background_music_volume"])
    audio = mix_audio([
        src.audio.volumex(nc["original_clip_volume"]) if src.audio else None,
        voice.volumex(nc["voiceover_volume"]),
        music], D)
    return CompositeVideoClip(layers, size=(W, H)).set_duration(D).set_audio(audio)


# ---------- split layout (facts / story) ----------

def render_fake_text_hook(hook, W, H):
    """iMessage-style screenshot frames; one frame per revealed message count."""
    msgs = hook.get("messages", [])
    contact = hook.get("contact", "Unknown")
    frames = []
    font = get_font(int(W * 0.045))
    head_font = get_font(int(W * 0.05))
    for shown in range(1, len(msgs) + 1):
        img = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, int(H * 0.12)], fill=(246, 246, 248))
        d.ellipse([W / 2 - 50, int(H * 0.02), W / 2 + 50, int(H * 0.02) + 100], fill=(160, 165, 175))
        d.text((W / 2, int(H * 0.02) + 50), contact[:1].upper(), font=head_font, fill="white", anchor="mm")
        d.text((W / 2, int(H * 0.095)), contact, font=get_font(int(W * 0.035)), fill="black", anchor="mm")
        y = int(H * 0.16)
        for msg in msgs[:shown]:
            mine = msg.get("from") == "me"
            lines = _wrap_lines(msg.get("text", ""), font, int(W * 0.62))
            lh = int(sum(font.getmetrics()) * 1.15)
            bw = int(max(font.getlength(l) for l in lines)) + 56
            bh = lh * len(lines) + 36
            x0 = W - bw - 40 if mine else 40
            d.rounded_rectangle([x0, y, x0 + bw, y + bh], radius=36,
                                fill=(10, 132, 255) if mine else (233, 233, 235))
            ty = y + 18
            for l in lines:
                d.text((x0 + 28, ty), l, font=font, fill="white" if mine else "black")
                ty += lh
            y += bh + 22
        frames.append(np.asarray(img))
    return frames


def _fake_text_hook_clip(hook, W, H):
    frames = render_fake_text_hook(hook, W, H)
    if not frames:
        return None
    dur = float(hook.get("duration", 2.5))
    per = dur / len(frames)
    return concatenate_videoclips([ImageClip(f).set_duration(per) for f in frames])


def _top_strip_clip(m, D, W, top_h):
    import images
    import visuals
    segments = m.get("segments") or [{"query": m.get("theme") or m.get("title", "facts"),
                                      "start_frac": 0.0, "end_frac": 1.0}]
    theme = m.get("theme", "default")
    slices = []
    for seg in segments:
        s0, s1 = seg["start_frac"] * D, seg["end_frac"] * D
        n = max(1, int(math.ceil((s1 - s0) / 7.0)))
        paths = images.fetch_topic_images(seg["query"], n) or []
        step = (s1 - s0) / n
        for k in range(n):
            if paths:
                arr = fx.cover_resize_array(load_image_array(paths[k % len(paths)]), W, top_h)
            else:
                arr = visuals.generate_theme_bg(theme, W, top_h)
            slices.append((s0 + k * step, step, arr))
    layers = []
    for i, (t0, dur, arr) in enumerate(slices):
        xfade = 0.4 if i > 0 else 0.0
        c = ImageClip(arr).set_duration(dur + xfade)
        c = fx.subpixel_zoom(c, lambda t, d=dur: 1.0 + 0.06 * t / max(d, 0.1))
        c = c.set_start(max(0.0, t0 - xfade))
        if xfade:
            c = c.crossfadein(xfade)
        layers.append(c)
    return CompositeVideoClip(layers, size=(W, top_h)).set_duration(D)


def _assemble_narration_split(m, cfg, W, H):
    nc = cfg["content_types"]["narration"]
    audio_path, voice = _narration_audio(m, cfg)
    D = min(cfg["output"]["max_duration_seconds"], voice.duration + 0.4)
    top_h = int(H * m.get("top_frac", 0.33)) // 2 * 2
    bot_h = H - top_h
    gp = resolve(m["gameplay_source"])
    if not gp.exists():
        raise FileNotFoundError(f"gameplay_source not found: {gp} "
                                f"(make one with: python src/visuals.py satisfying <out.mp4>)")
    game = VideoFileClip(str(gp))
    g0 = float(m.get("gameplay_start", 0))
    game = game.subclip(g0)
    if game.duration < D:
        game = game.fx(vfx.loop, duration=D)
    game = fit_vertical(game.subclip(0, D), W, bot_h).without_audio().set_position((0, top_h))
    top = _top_strip_clip(m, D, W, top_h).set_position((0, 0))
    script = m["script"]
    words = script.split()
    timings = word_timings(words, audio_path, voice.duration)
    layers = [ColorClip((W, H), (0, 0, 0)).set_duration(D), top, game]
    # captions overlay the top strip only, never the gameplay footage
    layers += caption_layers(script, timings, D, W, cfg, 0, top_h, m.get("caption_style", "phrase"))
    layers += pop_keyword_layers(script, m.get("pop_keywords"), timings, D, W, 0, top_h)
    layers += _brand_layers_narration(W, H, D, cfg, corner="top_right")
    if m.get("fake_text_hook"):
        hook = _fake_text_hook_clip(m["fake_text_hook"], W, H)
        if hook is not None:
            layers.append(hook.set_start(0))
    music = None
    if m.get("music") and (cfg["music_selection"]["include_in_narration"] or m.get("force_music")):
        music = load_music(m["music"], D, nc["background_music_volume"])
    audio = mix_audio([voice.volumex(nc["voiceover_volume"]), music], D)
    return CompositeVideoClip(layers, size=(W, H)).set_duration(D).set_audio(audio)


# ==========================================================================
# Type: manga_edit
# ==========================================================================

def _ring_layer(W, H, duration, color, center=(0.5, 0.5)):
    """Expanding-ring 'shockwave' accent."""
    rgb = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    cx, cy = center[0] * W, center[1] * H
    rmax = math.hypot(W, H) * 0.6

    def make(t):
        p = fx.ease_out_cubic(t / duration)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        r = rmax * p
        a = int(220 * (1 - p))
        wdt = max(2, int(28 * (1 - p)))
        ImageDraw.Draw(img).ellipse([cx - r, cy - r, cx + r, cy + r], outline=rgb + (a,), width=wdt)
        return np.asarray(img)

    return fx._rgba_videoclip(make, duration)


def _apply_impact_fx(clip, impacts, fps, window=0.15, seed=13):
    """RGB-split + jitter burst at each (time, strength)."""
    W, H = clip.size
    impacts = sorted(impacts)

    def fl(gf, t):
        fr = gf(t)
        for ti, strength in impacts:
            dt = t - ti
            if 0 <= dt < window:
                k = 1 - dt / window
                fr = fx.rgb_split_frame(fr, int(round(10 * strength * k)))
                rng = np.random.default_rng((seed, int(t * fps)))
                dx, dy = (rng.uniform(-1, 1, 2) * 14 * strength * k).astype(int)
                fr = np.roll(np.roll(fr, dy, axis=0), dx, axis=1)
                break
        return fr

    return clip.fl(fl)


def _caption_bar(text, W, fontsize, box_h):
    """Full-width, edge-to-edge opaque black bar (not a rounded pill)."""
    return render_text_rgba(text, fontsize, color="white", box_size=(W, box_h), bg=(0, 0, 0, 255),
                            pad=(int(W * 0.06), 20))


def _assemble_manga_edit(m, cfg, W, H):
    fps = cfg["output"]["fps"]
    music_path = resolve(m["music"])
    track = AudioFileClip(str(music_path))
    D = float(m.get("duration_seconds") or min(track.duration, cfg["output"]["max_duration_seconds"]))
    panels = [load_image_array(p) for p in m["panels"]]
    if not panels:
        raise ValueError("manga_edit needs at least one panel")
    fit_mode = m.get("panel_fit", "crop")
    frames = [fit_panel(p, W, H, fit_mode) for p in panels]

    dither_f = int(m.get("dither_frames", 3))
    flash_f = int(m.get("flash_frames", 1))
    blur_f = int(m.get("blur_frames", 3))
    lead = (dither_f + flash_f) / fps
    gap = float(m.get("min_cut_gap_seconds", 0.35))

    beats = list(_detect_beats(music_path))
    beats = _tile_times(beats, track.duration, D)
    if m.get("cut_on_onsets", True):
        cands = _tile_times(_detect_onsets(music_path), track.duration, D)
    else:
        cands = beats[::max(1, int(m.get("beat_subdivision", 1)))]
    drop_t = _detect_drop(music_path) if m.get("auto_drop", True) else None
    first_cut_after = drop_t if drop_t is not None else 0.0
    cands = [t for t in cands if t >= first_cut_after + gap and t < D - 0.3]
    cut_times = [0.0]
    if drop_t is not None:
        # hold ONE static panel through the build-up, start cutting AT the drop
        cut_times.append(drop_t)
    cut_times += _min_gap_filter(cands, gap, after=cut_times[-1])
    cut_times = sorted({snap(t, fps) for t in cut_times if t >= lead or t == 0.0})
    if drop_t is not None:
        drop_t = snap(drop_t, fps)

    zoom = m.get("zoom", True)
    punch_f = int(m.get("punch_frames", 4))
    punch_amt = float(m.get("punch_zoom_amount", 0.04))
    layers = [ColorClip((W, H), (0, 0, 0)).set_duration(D)]
    for i, t0 in enumerate(cut_times):
        t1 = cut_times[i + 1] if i + 1 < len(cut_times) else D
        frame = frames[i % len(frames)]
        hold = ImageClip(frame).set_duration(t1 - t0)
        if i == 0:
            hold = fx.subpixel_zoom(hold, lambda t, d=t1 - t0: 1.0 + 0.04 * t / max(d, 0.1))
        elif zoom:
            hold = fx.subpixel_zoom(hold, make_scale_fn(blur_f / fps, punch_f / fps, punch_amt,
                                                        0.6, 1.3, 0.012, 0.6))
        layers.append(hold.set_start(t0))
    for i in range(1, len(cut_times)):
        a = frames[(i - 1) % len(frames)]
        b = frames[i % len(frames)]
        tr = fx.staged_cut_transition(a, b, W, H, fps, dither_f, flash_f, blur_f)
        layers.append(tr.set_start(cut_times[i] - lead))

    impacts = []
    flash_color = m.get("flash_color", "#00e5ff")
    rgb = tuple(int(flash_color.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4))
    flash_times = [beats[b] for b in m.get("flash_beats", []) if b < len(beats)]
    if drop_t is not None and m.get("drop_flash"):
        flash_times.append(drop_t)
    for ft in flash_times:
        layers.append(fx.flash_strobe(W, H, 4 / fps, rgb, 0.75).set_start(ft))
        layers.append(_ring_layer(W, H, 0.4, flash_color).set_start(ft))
        impacts.append((ft, 1.0))
    if drop_t is not None:
        impacts.append((drop_t, 2.0))

    for ov in m.get("title_overlays", []):
        b0 = int(ov.get("at_beat", 0))
        if b0 >= len(beats):
            continue
        t0 = beats[b0]
        b1 = b0 + int(ov.get("hold_beats", 2))
        t1 = beats[b1] if b1 < len(beats) else min(D, t0 + 1.5)
        if ov.get("style", "caption") == "caption":
            arr = _caption_bar(ov["text"], W, int(W * 0.07), int(H * 0.09))
            clip = ImageClip(arr).set_position(("center", int(H * 0.72)))
        else:
            arr = render_text_rgba(ov["text"].upper(), int(W * 0.1), color="#ffe066",
                                   stroke_width=8, max_width=int(W * 0.9))
            arr = np.asarray(Image.fromarray(arr).rotate(7, expand=True, resample=Image.BICUBIC))
            clip = ImageClip(arr).set_position("center")
        layers.append(clip.set_duration(t1 - t0).set_start(t0))

    if m.get("intro_caption"):
        end = (cut_times[1] - lead) if len(cut_times) > 1 else D
        if end > 0.5:
            arr = _caption_bar(m["intro_caption"], W, int(m.get("intro_caption_fontsize", 88)),
                               int(m.get("intro_caption_box_height", 340)))
            layers.append(ImageClip(arr).set_duration(end - 0.4).set_start(0.4)
                          .set_position(("center", int(H * 0.5 - arr.shape[0] / 2))))
    layers += _brand_layers_narration(W, H, D, cfg, cta=False)

    comp = CompositeVideoClip(layers, size=(W, H)).set_duration(D)
    if impacts:
        comp = _apply_impact_fx(comp, impacts, fps)
    music = fit_audio(track, D).volumex(m.get("music_volume", 1.0)).audio_fadeout(0.5)
    return comp.set_audio(music)


# ==========================================================================
# Type: manga_hype_edit
# ==========================================================================

HYPE_DEFAULTS = dict(
    dither_frames=3, flash_frames=1, blur_frames=3, punch_frames=10, punch_zoom_amount=0.07,
    punch_overshoot_peak_frac=0.6, punch_overshoot_mult=1.45, panel_drift_amplitude=0.025,
    panel_drift_period_seconds=0.5, grow_reveal_frames=10, min_cut_gap_seconds=0.42,
    outro_fade_frames=22, intro_zoom_amount=0.04,
)
SHOT_STYLES = ["medium", "tight", "wide", "canted"]
LAST_TIMELINE = {}
HYPE_PANEL_COUNT = 13


def styled_frame(arr, style, W, H, cant_sign=1):
    """One panel at one shot scale - varied cut-to-cut so a reused panel
    doesn't read as a literal repeat. (No 'inset' style: a static sharp-over-
    blurred composite + punch zoom mimicked a soft cross-dissolve.)"""
    if style == "tight":
        med = content_aware_crop_array(arr, W, H)
        return fx.zoom_frame(med, 1 / 0.72)
    if style == "wide":
        return fx.cover_resize_array(arr, W, H, (0.5, 0.5))
    if style == "canted":
        return fx.canted_still(arr, W, H, angle_deg=5.0 * cant_sign)
    return content_aware_crop_array(arr, W, H)


def _assign_styles(n_cuts, n_panels, seed=17):
    """Shuffle styles per round, never repeating the previous cut's style,
    and avoid showing a reused panel at the style it had last time."""
    rng = np.random.default_rng(seed)
    styles = ["medium"]
    last_for_panel = {0: "medium"}
    pool = []
    for i in range(1, n_cuts):
        if not pool:
            pool = list(rng.permutation(SHOT_STYLES))
        avoid = {styles[-1], last_for_panel.get(i % n_panels)}
        pick = next((s for s in pool if s not in avoid), None)
        if pick is None:
            pick = next(s for s in SHOT_STYLES if s not in avoid)
        else:
            pool.remove(pick)
        styles.append(pick)
        last_for_panel[i % n_panels] = pick
    return styles


def _assemble_manga_hype_edit(m, cfg, W, H):
    fps = cfg["output"]["fps"]
    p = dict(HYPE_DEFAULTS)
    p.update({k: m[k] for k in HYPE_DEFAULTS if k in m})
    dither_f, flash_f, blur_f = int(p["dither_frames"]), int(p["flash_frames"]), int(p["blur_frames"])
    lead = (dither_f + flash_f) / fps

    if len(m.get("panels", [])) != HYPE_PANEL_COUNT:
        raise ValueError(f"manga_hype_edit needs exactly {HYPE_PANEL_COUNT} panels, "
                         f"got {len(m.get('panels', []))}")
    panels = [load_image_array(x) for x in m["panels"]]

    music_path = resolve(m["music"])
    track = AudioFileClip(str(music_path))
    intro_dur = snap(float(m.get("intro_duration_seconds", 3.0)), fps)

    # ---- cut timeline ----
    explicit = m.get("post_intro_cut_times")
    if explicit is None:
        onsets = _tile_times(_detect_onsets(music_path), track.duration,
                             float(m.get("duration_seconds") or cfg["output"]["max_duration_seconds"]))
        post = _min_gap_filter([t for t in onsets if t >= intro_dur], p["min_cut_gap_seconds"],
                               after=intro_dur - p["min_cut_gap_seconds"])
    else:
        post = sorted(float(t) for t in explicit)
    post = sorted({snap(t, fps) for t in post})
    D = float(m.get("duration_seconds") or ((post[-1] + 1.0) if post else intro_dur + 1.0))
    outro_n = int(p["outro_fade_frames"])
    outro_start = D - outro_n / fps
    post = [t for t in post if intro_dur <= t < outro_start]
    cut_times = [0.0] + post
    n = len(cut_times)
    if D > cfg["output"]["max_duration_seconds"]:
        print(f"  [warn] duration {D:.2f}s exceeds the Shorts ceiling")

    styles = _assign_styles(n, HYPE_PANEL_COUNT)
    grow_targets = set(range(1, n, 4))
    impact_targets = set(range(3, n, 6))
    frames = []
    for i in range(n):
        style = styles[i]
        frames.append(styled_frame(panels[i % HYPE_PANEL_COUNT], style, W, H,
                                   cant_sign=1 if (i // 2) % 2 == 0 else -1))

    # ---- transitions: decide placement first (holds need to know punch start) ----
    trans = {}   # i -> (kind, start_time, grow_frames)
    for i in range(1, n):
        prev_end_of_blur = cut_times[i - 1] + (blur_f / fps if i - 1 >= 1 else 0.0)
        if i in grow_targets:
            gf = int(p["grow_reveal_frames"])
            # shrink growth if the previous hold is too short to fit the lead-in
            avail = int((cut_times[i] - prev_end_of_blur) * fps) - dither_f - flash_f - 1
            gf = min(gf, avail)
            if gf >= 4:
                trans[i] = ("grow", cut_times[i] - (dither_f + flash_f + gf) / fps, gf)
                continue
        trans[i] = ("staged", cut_times[i] - lead, 0)

    scale_fns = []
    for i in range(n):
        if i == 0:
            amt, d = p["intro_zoom_amount"], cut_times[1] if n > 1 else D
            scale_fns.append(lambda t, a=amt, d=d: 1.0 + a * min(1.0, t / d))
            continue
        punch_start = 0.0 if trans[i][0] == "grow" else blur_f / fps
        scale_fns.append(make_scale_fn(punch_start, p["punch_frames"] / fps, p["punch_zoom_amount"],
                                       p["punch_overshoot_peak_frac"], p["punch_overshoot_mult"],
                                       p["panel_drift_amplitude"], p["panel_drift_period_seconds"]))

    layers = [ColorClip((W, H), (0, 0, 0)).set_duration(D)]
    for i in range(n):
        t0 = cut_times[i]
        t1 = cut_times[i + 1] if i + 1 < n else D
        hold = fx.subpixel_zoom(ImageClip(frames[i]).set_duration(t1 - t0), scale_fns[i])
        if i in impact_targets:
            hold = fx.camera_shake(hold, intensity_px=14, rotation_deg=0.8, decay=5.0, seed=i)
        layers.append(hold.set_start(t0))

    for i in range(1, n):
        kind, start, gf = trans[i]
        prev_local = start - cut_times[i - 1]
        a = fx.zoom_frame(frames[i - 1], max(1.0, scale_fns[i - 1](prev_local)))
        b = frames[i]
        if kind == "grow":
            tr = fx.growing_inset_cut_transition(a, b, W, H, fps, dither_f, flash_f, gf)
        else:
            tr = fx.staged_cut_transition(a, b, W, H, fps, dither_f, flash_f, blur_f,
                                          white_flash_chance=0.0)
        layers.append(tr.set_start(start))

    # ---- impact accents land exactly on the cut ----
    for i in sorted(impact_targets):
        t = cut_times[i]
        layers.append(fx.flash_strobe(W, H, 2 / fps, (255, 255, 255), 0.35).set_start(t))
        layers.append(fx.speed_lines(W, H, 8 / fps, fps=fps, seed=i).set_start(t))

    # ---- rim / glow / aura accents: every 4th cut, medium/wide only ----
    accent_makers = [
        lambda d: fx.rim_light_flare_layer(W, H, d, fps=fps),
        lambda d: fx.glow_burst_layer(W, H, d, fps=fps),
        lambda d: fx.aura_swirl_layer(W, H, d, fps=fps),
    ]
    k = 0
    for i in range(4, n, 4):
        if styles[i] not in ("medium", "wide"):
            continue
        start = cut_times[i]
        end = trans[i + 1][1] if i + 1 < n else outro_start   # trimmed to the next lead-in
        if end - start > 2 / fps:
            layers.append(accent_makers[k % 3](end - start).set_start(start))
            k += 1

    # ---- intro caption bar ----
    if m.get("intro_caption"):
        end = trans[1][1] if n > 1 else D   # stop before the first cut's lead-in, not at intro_dur
        if end > 0.5:
            arr = _caption_bar(m["intro_caption"], W, int(m.get("intro_caption_fontsize", 80)),
                               int(m.get("intro_caption_box_height", 300)))
            y = int(H * m.get("intro_caption_y_frac", 0.5) - arr.shape[0] / 2)
            layers.append(ImageClip(arr).set_duration(end - 0.4).set_start(0.4).set_position(("center", y)))

    if m.get("watermark"):
        layers += _brand_layers_narration(W, H, D, cfg, cta=False)

    # exposed for programmatic per-cut audits (every flash frame, accent windows)
    LAST_TIMELINE.clear()
    LAST_TIMELINE.update(fps=fps, duration=D, cut_times=cut_times, styles=styles,
                         transitions={i: {"kind": k, "start": s, "grow_frames": g}
                                      for i, (k, s, g) in trans.items()},
                         dither_frames=dither_f, flash_frames=flash_f, blur_frames=blur_f,
                         outro_start=outro_start)

    comp = CompositeVideoClip(layers, size=(W, H)).set_duration(D)

    def fade_factor(t):
        f = np.clip((np.asarray(t, dtype=np.float64) - outro_start) / (outro_n / fps), 0.0, 1.0)
        return 1.0 - (3 * f ** 2 - 2 * f ** 3)

    def post_fx(gf, t):
        fr = fx.grade_frame(gf(t), contrast=1.35, black_crush=0.14)
        fr = fx.film_grain_frame(fr, t, opacity=0.10, fps=fps)
        if t >= outro_start:
            fr = (fr.astype(np.float32) * float(fade_factor(t))).astype(np.uint8)
        return fr

    final = comp.fl(post_fx)
    music = _apply_fade(fit_audio(track, D).volumex(m.get("music_volume", 1.0)), fade_factor)
    return final.set_audio(music.set_duration(D))


# ==========================================================================
# Entry points
# ==========================================================================

ASSEMBLERS = {
    "edit": _assemble_edit,
    "meme": _assemble_meme,
    "narration": _assemble_narration,
    "manga_edit": _assemble_manga_edit,
    "manga_hype_edit": _assemble_manga_hype_edit,
}


def output_path_for(manifest):
    return OUTPUT_DIR / f"{manifest['id']}.mp4"


def assemble(manifest_path, logger="bar", preset="medium"):
    """Render one manifest. Returns the output path."""
    cfg = load_config()
    m = load_manifest(manifest_path)
    kind = m.get("type", "edit")
    if kind == "template_edit" and json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig")).get("engine") == "rise":
        import rise_edit
        print(f"Rendering {m['id']} (rise engine: {m.get('template')})")
        return rise_edit.render_manifest(manifest_path, preview=(preset == "ultrafast"), out=output_path_for(m))
    if kind == "template_edit" and json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig")).get("engine") == "xavier":
        import xavier_edit
        print(f"Rendering {m['id']} (xavier engine: {m.get('template')})")
        return xavier_edit.render_manifest(manifest_path, preview=(preset == "ultrafast"), out=output_path_for(m))
    if kind == "template_edit" and json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig")).get("engine") == "wekisava":
        import wekisava_edit
        print(f"Rendering {m['id']} (wekisava engine: {m.get('template')})")
        return wekisava_edit.render_manifest(manifest_path, preview=(preset == "ultrafast"), out=output_path_for(m))
    if kind == "template_edit":
        # frame-exact template renderer (own 60fps frame loop, own audio mux)
        import template_edit
        print(f"Rendering {m['id']} (template_edit: {m.get('template')})")
        return template_edit.render_manifest(manifest_path, preview=(preset == "ultrafast"),
                                             out=output_path_for(m))
    if kind not in ASSEMBLERS:
        raise ValueError(f"unknown manifest type: {kind}")
    W, H = cfg["output"]["width"], cfg["output"]["height"]
    if kind == "manga_hype_edit":   # the ONE type that honors a manifest-level aspect override
        W, H = int(m.get("width", W)), int(m.get("height", H))
    fps = cfg["output"]["fps"]
    print(f"Rendering {m['id']} ({kind}, {W}x{H})")
    video = ASSEMBLERS[kind](m, cfg, W, H)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = output_path_for(m)
    video.write_videofile(str(out), fps=fps, codec="libx264", audio_codec="aac", preset=preset,
                          threads=4, ffmpeg_params=["-pix_fmt", "yuv420p"],
                          temp_audiofile=str(OUTPUT_DIR / f"{m['id']}_TEMP_audio.m4a"),
                          logger=logger)
    video.close()
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/assemble.py edits/<manifest>.json")
        sys.exit(1)
    print(assemble(sys.argv[1]))
