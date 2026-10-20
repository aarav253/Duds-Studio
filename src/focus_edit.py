"""Intro-then-focus edits (templates 12 and 13).

A captioned intro shows several characters (single shots, or a 3-panel manga
split), optionally teases the focus character with a growing picture-in-picture,
then the drop switches to the focus character for the rest of the edit:
  fx "invert"  - each cut opens with an inverted/white bloom flash, tinted grade
  fx "radial"  - each cut opens with a radial zoom-blur, natural colour
Ends with the DUDS outro used by the template edits. Timings live in the
template (snapped to the song); clips come from the manifest.

    python src/focus_edit.py edits/<id>.json [--preview]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sins_edit import Clip                        # noqa: E402  bar trimming + frame blending
from template_edit import end_card, Film, STYLE   # noqa: E402  the standard DUDS outro

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _p(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def zoom(img, s, focus=(0.5, 0.45), dx=0.0, dy=0.0):
    H, W = img.shape[:2]
    cx, cy = focus[0] * W, focus[1] * H
    M = np.float32([[s, 0, (1 - s) * cx + dx * W], [0, s, (1 - s) * cy + dy * H]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def radial_blur(img, amount, n=8):
    if amount <= 0.002:
        return img
    acc = np.zeros(img.shape, np.float32)
    for i in range(n):
        acc += zoom(img, 1 + amount * i / (n - 1), (0.5, 0.5)).astype(np.float32)
    return (acc / n).astype(np.uint8)


# ---------------------------------------------------------------- text
def _font(path, size, var=None):
    f = ImageFont.truetype(str(_p(path)) if not str(path).startswith("C:") else path, int(size))
    if var:
        try:
            f.set_variation_by_name(var)
        except Exception:  # noqa: BLE001
            pass
    return f


def caption_layer(W, H, c):
    """premultiplied RGBA float layer for one caption style."""
    lines = c["text"].split("\n")
    f = _font(c["font"], c["size"] * H, c.get("variation"))
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    while max(probe.textlength(ln, font=f) for ln in lines) > c.get("max_w", 0.9) * W:
        f = _font(c["font"], f.size * 0.95, c.get("variation"))
    lh = f.size * c.get("line_gap", 1.05)
    y0 = c["y"] * H - lh * (len(lines) - 1) / 2
    fill, stroke = Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)
    sw = max(1, int(c.get("stroke", 0.004) * H))
    for i, ln in enumerate(lines):
        ImageDraw.Draw(fill).text((c.get("x", 0.5) * W, y0 + i * lh), ln, font=f, fill=255, anchor="mm")
        ImageDraw.Draw(stroke).text((c.get("x", 0.5) * W, y0 + i * lh), ln, font=f, fill=255, anchor="mm", stroke_width=sw, stroke_fill=255)
    A = np.asarray(fill).astype(np.float32) / 255
    S = np.asarray(stroke).astype(np.float32) / 255
    col = np.array(c.get("color", [250, 250, 250]), np.float32)
    out = np.zeros((H, W, 4), np.float32)
    if c.get("glow"):                                         # coloured glow (T13 bars / red title)
        g = np.asarray(Image.fromarray((S * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(c.get("glow_r", 0.012) * H))
                       ).astype(np.float32) / 255 * c["glow"]
        gc = np.array(c.get("glow_color", c.get("color")), np.float32)
        out[..., :3] += gc * g[..., None]; out[..., 3] = np.maximum(out[..., 3], g)
    if c.get("band"):                                         # glowing horizontal band behind the text
        b = c["band"]
        band = np.zeros((H, W), np.float32)
        yc, hh = int(c["y"] * H), int(b["h"] * H)
        band[max(0, yc - hh // 2):max(0, yc + hh // 2)] = 1
        band = cv2.GaussianBlur(band, (0, 0), b.get("blur", 0.01) * H) * b.get("alpha", 0.8)
        bc = np.array(b["color"], np.float32)
        out[..., :3] = out[..., :3] * (1 - band[..., None]) + bc * band[..., None]
        out[..., 3] = np.maximum(out[..., 3], band)
    sc = np.array(c.get("stroke_color", [20, 20, 20]), np.float32)
    so = S * c.get("stroke_alpha", 0.85)
    out[..., :3] = out[..., :3] * (1 - so[..., None]) + sc * so[..., None]; out[..., 3] = np.maximum(out[..., 3], so)
    out[..., :3] = out[..., :3] * (1 - A[..., None]) + col * A[..., None]; out[..., 3] = np.maximum(out[..., 3], A)
    return out


def over(img, layer, alpha=1.0, xmask=None):
    a = layer[..., 3:4] * alpha
    if xmask is not None:
        a = a * xmask[None, :, None]
    return img.astype(np.float32) * (1 - a) + layer[..., :3] * a


# ---------------------------------------------------------------- edit
class FocusEdit:
    def __init__(self, m, preview=False):
        self.m = m
        self.t = json.loads(_p(m["template"]).read_text(encoding="utf-8"))
        W0, H0 = self.t["size"]
        k = 0.5 if preview else 1.0
        self.W, self.H = int(W0 * k) // 2 * 2, int(H0 * k) // 2 * 2
        self.fps = self.t["fps"]
        self.body = self.t["body_end"]
        o = self.t["outro"]
        self.n = int(round((self.body + o["card"] + o["fade"] + o["dark"]) * self.fps))
        segs = self.t["segments"]
        assert len(m["clips"]) == len(segs), f"manifest needs {len(segs)} clips, has {len(m['clips'])}"
        self.segs = []
        for s, c in zip(segs, m["clips"]):
            if s["kind"] == "split":
                clips = [Clip(x["clip"], *self._panel_size(i)) for i, x in enumerate(c)]
                self.segs.append({**s, "spec": c, "clips": clips})
            else:
                c = c if isinstance(c, dict) else {"clip": c}
                self.segs.append({**s, "spec": c, "clip": Clip(c["clip"], self.W, self.H)})
        self.caps = []
        for c in self.t.get("captions", []):
            c = dict(c)
            c["text"] = m.get("captions", {}).get(c["id"], c["text"])
            c["a"], c["b"] = m.get("caption_times", {}).get(c["id"], (c["a"], c["b"]))
            self.caps.append((c, caption_layer(self.W, self.H, c)))
        pp = self.t.get("popup")
        if pp:
            pc = Clip(m["popup"]["clip"], self.W, self.H)
            self.pop = pc.frame(m["popup"].get("frame", pc.n // 2))
        self.endcard = end_card(m.get("end_text", "DUDS"), self.W, self.H, dict(STYLE)).astype(np.float32)
        self.dark = np.zeros((self.H, self.W, 3), np.float32) + np.array(STYLE["dark_bg"], np.float32)
        self.film = Film(self.W, self.H)
        lay = Image.new("L", (self.W, self.H), 0)
        ImageDraw.Draw(lay).text((self.W / 2, self.H * 0.962), " ".join(m.get("watermark", "DUDS")),
                                 font=ImageFont.truetype(r"C:\Windows\Fonts\bahnschrift.ttf", int(self.H * 0.016)), fill=150, anchor="mm")
        self.wm = np.asarray(lay).astype(np.float32)[..., None] / 255

    # split layout: top panel full width, two panels below
    def _panel_rects(self):
        W, H = self.W, self.H
        g = max(2, int(0.006 * W))
        top = int(0.55 * H)
        return [(0, 0, W, top - g), (0, top + g, W // 2 - g, H), (W // 2 + g, top + g, W, H)]

    def _panel_size(self, i):
        x0, y0, x1, y1 = self._panel_rects()[i]
        return x1 - x0, y1 - y0

    def _clip_frame(self, seg, clip, spec, t):
        start = spec.get("start", 4)
        need = (seg["b"] - seg["a"]) * 60
        speed = spec.get("speed") or min(1.0, max(0.45, (clip.n - start - 2) / max(1, need)))
        return clip.frame(start + (t - seg["a"]) * 60 * speed)

    # ---------------- grades / fx
    def tint(self, img, col, amt, glow):
        f = img.astype(np.float32)
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None] / 255
        g = np.clip((g - 0.5) * 1.25 + 0.5, 0, 1)                       # a bit more contrast
        col = np.array(col, np.float32)
        duo = np.where(g < 0.6, g / 0.6 * col, col + (255 - col) * (g - 0.6) / 0.4)   # dark -> tint -> white
        f = f * (1 - amt) + duo * amt
        if glow:
            hl = cv2.GaussianBlur(np.clip(f - 150, 0, 255), (0, 0), 0.02 * self.W)
            f = f + hl * glow
        return f

    def frame(self, k):
        t = k / self.fps
        T = self.t
        if t >= self.body:
            return self.outro_frame(k, t)
        seg = next(s for s in self.segs if s["a"] <= t < s["b"])
        u = (t - seg["a"]) / (seg["b"] - seg["a"])
        j = int(round((t - seg["a"]) * self.fps))                      # frames since the cut
        kind = seg["kind"]
        if kind == "split":
            img = np.zeros((self.H, self.W, 3), np.float32) + 245
            for (x0, y0, x1, y1), c, sp in zip(self._panel_rects(), seg["clips"], seg["spec"]):
                fr = self._clip_frame(seg, c, sp, t)
                fr = zoom(fr, 1.0 + 0.04 * u, tuple(sp.get("focus", (0.5, 0.45))))
                img[y0:y1, x0:x1] = fr
            img = self.tint(img.astype(np.uint8), T.get("split_tint", [200, 60, 60]), T.get("split_tint_amt", 0.0), 0)
            if j < T.get("split_glitch_frames", 0):                    # red glitch bands on arrival
                rng = np.random.default_rng(k)
                for _ in range(4):
                    y = int(rng.uniform(0.2, 0.8) * self.H); h = int(rng.uniform(0.01, 0.04) * self.H)
                    img[y:y + h] = np.roll(img[y:y + h], int(rng.uniform(-0.08, 0.08) * self.W), 1)
                    img[y:y + h, :, 0] = np.minimum(255, img[y:y + h, :, 0] + 90)
        else:
            sp = seg["spec"]
            fr = self._clip_frame(seg, seg["clip"], sp, t)
            z0, z1 = seg.get("zoom", [1.0, 1.04])
            img = zoom(fr, z0 + (z1 - z0) * (ease(u) if kind != "post" else u), tuple(sp.get("focus", (0.5, 0.45))))
            if kind == "intro":
                f = img.astype(np.float32)
                gr = f.mean(2, keepdims=True)
                f = (gr + (f - gr) * T.get("intro_sat", 0.8)) * seg.get("bright", T.get("intro_bright", 0.85))
                img = f
            elif kind == "post":
                fx = T["post_fx"]
                if fx == "invert":
                    img = self.tint(img.astype(np.uint8), self.m.get("tint", T.get("tint")), T.get("tint_amt", 0.75), T.get("glow", 0.35))
                    if j <= 1:                                        # inverted grey flash
                        g = 255 - cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                        g = cv2.GaussianBlur(g, (0, 0), 2) * 0.85 + 60
                        img = np.repeat(g[..., None], 3, 2)
                    elif j == 2:
                        img = img + (255 - img) * 0.35
                    if seg.get("shake"):
                        rng = np.random.default_rng(k)
                        img = zoom(np.clip(img, 0, 255).astype(np.uint8), 1.02, (0.5, 0.5), rng.uniform(-.008, .008), rng.uniform(-.008, .008)).astype(np.float32)
                else:                                                 # radial
                    if j < 5:
                        img = radial_blur(np.clip(img, 0, 255).astype(np.uint8), 0.16 * (1 - j / 5) ** 1.5)
                    img = img.astype(np.float32) * T.get("post_bright", 1.0)
                if seg.get("end_blur"):                               # last shot dissolves into a white blur
                    b = ease((u - 0.35) / 0.65)
                    if b > 0:
                        bl = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), 1 + 18 * b).astype(np.float32)
                        img = bl + (235 - bl) * 0.55 * b
        img = np.asarray(img, np.float32)
        # popup teaser (T12): the focus character grows in a box before the drop
        pp = T.get("popup")
        if pp and pp["a"] <= t < pp["b"]:
            v = ease((t - pp["a"]) / (pp["grow"]))
            w = int(self.W * (pp["w0"] + (pp["w1"] - pp["w0"]) * v))
            h = int(w * self.H / self.W)
            if w > 4:
                card = cv2.resize(self.tint(self.pop, self.m.get("tint", T.get("tint")), T.get("tint_amt", 0.75), T.get("glow", 0.35)).clip(0, 255).astype(np.uint8), (w, h))
                cx = int((pp["x"] + pp.get("drift", 0) * (t - pp["a"])) * self.W); cy = int(pp["y"] * self.H)
                x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
                x1, y1 = min(self.W, x0 + w), min(self.H, y0 + h)
                img[y0:y1, x0:x1] = card[:y1 - y0, :x1 - x0]
        # captions
        for c, layer in self.caps:
            if c["a"] <= t < c["b"]:
                al = 1.0
                if c.get("fade_in"):
                    al = min(1.0, (t - c["a"]) / c["fade_in"])
                xm = None
                if c.get("sweep"):                                    # reveal left -> right
                    p = ease((t - c["a"]) / c["sweep"])
                    xs = np.linspace(0, 1, self.W)
                    xm = np.clip((p * 1.15 - xs) / 0.15, 0, 1).astype(np.float32)
                img = over(img, layer, al, xm)
        fi = T.get("fade_in", 0)
        if fi and t < fi:
            img = img * ease(t / fi)
        img = img * (1 - self.wm * 0.7) + 235 * self.wm * 0.7
        return np.clip(img, 0, 255).astype(np.uint8)

    def outro_frame(self, k, t):
        o = self.t["outro"]
        u = t - self.body
        if u < o["card"]:
            f, dark = self.endcard, False
        else:
            a = ease(min(1.0, (u - o["card"]) / o["fade"]))
            f, dark = self.endcard * (1 - a) + self.dark * a, a > 0.4
        return self.film.apply(np.clip(f, 0, 255).astype(np.uint8), k, 0.6 if dark else 0.22)

    # ---------------- audio
    def mix(self, out):
        sr = 44100
        y, s0 = sf.read(str(_p(self.m.get("music_file", self.t["music"]))), always_2d=True)
        assert s0 == sr
        n = int(self.n / self.fps * sr)
        mix = np.zeros((n, 2), np.float32)
        mix[:min(n, len(y))] = y[:n]
        if self.m.get("voicelines"):
            from voice_mix import add_voices
            mix = add_voices(mix, self.m["voicelines"], sr, self.m.get("duck_db", 4.0), self.m.get("voice_over_music", 1.4))
        import pyloudnorm as pyln
        from scipy.ndimage import maximum_filter1d as mf, uniform_filter1d as uf
        meter = pyln.Meter(sr)
        for _ in range(3):
            mix = pyln.normalize.loudness(mix, meter.integrated_loudness(mix), self.m.get("lufs", -14.0))
            env = mf(np.abs(mix).max(1), int(0.01 * sr))
            g = np.minimum(1.0, 0.94 / np.maximum(env, 1e-6))
            mix = mix * np.minimum(g, uf(g, int(0.02 * sr)))[:, None]
        fo = self.t.get("audio_fade", 1.5)
        a = int((self.body - fo * 0.6) * sr)
        k = n - a
        mix[a:] *= (np.cos(np.linspace(0, np.pi / 2, k)) ** 1.5)[:, None]
        sf.write(str(out), mix, sr)
        return meter.integrated_loudness(mix[:int(self.body * sr)])

    def render(self, out):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp, wav = out.with_suffix(".video.mp4"), out.with_suffix(".mix.wav")
        lufs = self.mix(wav)
        wr = imageio_ffmpeg.write_frames(str(tmp), (self.W, self.H), fps=self.fps, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "15", "-preset", "slow"], macro_block_size=2)
        wr.send(None)
        for k in range(self.n):
            wr.send(np.ascontiguousarray(self.frame(k)))
            if k % 120 == 0:
                print(f"  frame {k}/{self.n}", flush=True)
        wr.close()
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(wav), "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", str(out)], check=True)
        tmp.unlink()
        wav.unlink()
        print(f"  loudness {lufs:.1f} LUFS")
        return out


if __name__ == "__main__":
    m = json.loads(_p(sys.argv[1]).read_text(encoding="utf-8-sig"))
    pv = "--preview" in sys.argv
    print(FocusEdit(m, pv).render(ROOT / "output" / f"{m['id']}{'_preview' if pv else ''}.mp4"))
