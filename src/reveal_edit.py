"""'No perfect ___ ... brutal.' reveal edit (template 11).

Before the drop: other characters in long held shots (action / push-in / pan),
a flash shot, then a build shot that brightens. At the drop that shot goes
near-grey for one beat while the caption flips to the post text, then the
reveal character takes one shot per beat, muted and gently pulling back.
Voicelines are processed with pedalboard (cleanup + compression), the music is
ducked under them, and the final mix is normalised to -14 LUFS (pyloudnorm).

    python src/reveal_edit.py edits/<id>.json [--preview]
manifest: {id, template, shots: [{clip, start?, speed?, focus?: [x, y]}...],
           caption_pre?, caption_post?, voicelines: [{file, at, gain_db?}], duck_db?}
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
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sins_edit import Clip  # noqa: E402  (bar trimming + frame blending)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _p(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def zoom_pan(img, s, dx=0.0, dy=0.0, focus=(0.5, 0.5)):
    H, W = img.shape[:2]
    cx, cy = focus[0] * W, focus[1] * H
    M = np.float32([[s, 0, (1 - s) * cx + dx * W], [0, s, (1 - s) * cy + dy * H]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def saturate(img, k, gain=(1.0, 1.0, 1.0), lift=1.0):
    f = img.astype(np.float32)
    g = f.mean(2, keepdims=True)
    f = g + (f - g) * k
    f = f * np.array(gain, np.float32) * lift
    return f


class Caption:
    def __init__(self, W, H, cfg, text):
        f = ImageFont.truetype(cfg["font"], int(cfg["size"] * W / 1080))
        lay = Image.new("L", (W, H), 0)
        st = Image.new("L", (W, H), 0)
        lines = text.split("\n")
        lh = f.size * cfg.get("line_gap", 1.0)
        y0 = cfg["y"] * H - lh * (len(lines) - 1) / 2
        sw = max(1, int(cfg.get("stroke", 3) * W / 1080))
        for i, ln in enumerate(lines):
            ImageDraw.Draw(lay).text((W / 2, y0 + i * lh), ln, font=f, fill=255, anchor="mm")
            ImageDraw.Draw(st).text((W / 2, y0 + i * lh), ln, font=f, fill=255, anchor="mm", stroke_width=sw, stroke_fill=255)
        self.A = np.asarray(lay).astype(np.float32)[..., None] / 255
        self.S = np.asarray(st).astype(np.float32)[..., None] / 255

    def draw(self, img):
        img = img * (1 - self.S * 0.85)                      # thin dark outline like the reference
        return img * (1 - self.A) + 250 * self.A


class Reveal:
    def __init__(self, m, preview=False):
        self.m = m
        self.t = json.loads(_p(m["template"]).read_text(encoding="utf-8"))
        W0, H0 = m.get("size", self.t["size"])
        k = 0.5 if preview else 1.0
        self.W, self.H = int(W0 * k) // 2 * 2, int(H0 * k) // 2 * 2
        self.fps = self.t["fps"]
        self.n = int(self.t["duration"] * self.fps)
        self.drop = self.t["drop"]
        self.shots = []
        for spec, slot in zip(m["shots"], self.t["shots"]):
            self.shots.append({"slot": slot, "spec": spec, "clip": Clip(spec["clip"], self.W, self.H)})
        c = self.t["caption"]
        self.cap_pre = Caption(self.W, self.H, c, m.get("caption_pre", c["pre"]))
        self.cap_post = Caption(self.W, self.H, c, m.get("caption_post", c["post"]))
        lay = Image.new("L", (self.W, self.H), 0)
        ImageDraw.Draw(lay).text((self.W / 2, self.H * 0.962), " ".join(m.get("watermark", "DUDS")),
                                 font=ImageFont.truetype(r"C:\Windows\Fonts\bahnschrift.ttf", int(self.H * 0.016)),
                                 fill=150, anchor="mm")
        self.wm = np.asarray(lay).astype(np.float32)[..., None] / 255
        # outro: same as the template edits (template_edit.end_card + Film) - card, fade to dark, dark hold
        self.outro = m.get("outro")
        self.body_end = self.t["duration"]
        if self.outro:
            from template_edit import end_card, Film, STYLE
            st = dict(STYLE)
            self.endcard = end_card(self.outro.get("text", "DUDS"), self.W, self.H, st).astype(np.float32)
            self.dark = np.zeros((self.H, self.W, 3), np.float32) + np.array(st["dark_bg"], np.float32)
            self.film = Film(self.W, self.H)
            o = self.outro
            self.n = int(round((self.body_end + o["card"] + o["fade"] + o["dark"]) * self.fps))

    def outro_frame(self, k, t):
        o = self.outro
        u = t - self.body_end
        if u < o["card"]:
            f, dark = self.endcard, False
        else:
            a = min(1.0, (u - o["card"]) / o["fade"])
            a = a * a * (3 - 2 * a)
            f, dark = self.endcard * (1 - a) + self.dark * a, a > 0.4
        img = np.clip(f, 0, 255).astype(np.uint8)
        return self.film.apply(img, k, 0.6 if dark else 0.22)

    def frame(self, k):
        t = k / self.fps
        if self.outro and t >= self.body_end:
            return self.outro_frame(k, t)
        g = self.t["grade"]
        sh = next(s for s in self.shots if s["slot"]["a"] <= t < s["slot"]["b"] or s is self.shots[-1])
        sl, sp, c = sh["slot"], sh["spec"], sh["clip"]
        u = (t - sl["a"]) / (sl["b"] - sl["a"])
        start = sp.get("start", 4)
        need = (sl["b"] - sl["a"]) * 60
        speed = sp.get("speed") or min(1.0, max(0.4, (c.n - start - 2) / need))
        img = c.frame(start + (t - sl["a"]) * 60 * speed)
        z0, z1 = sl.get("zoom", [1, 1])
        if sl.get("motion") == "flash":
            s = z0 + (z1 - z0) * (1 - (1 - u) ** 3)          # quick punch that settles
        else:
            s = z0 + (z1 - z0) * smooth(u) if sl["role"] != "post" else z0 + (z1 - z0) * (1 - (1 - u) ** 2)
        pan = sl.get("pan", [0, 0])
        img = zoom_pan(img, s, pan[0] * u, pan[1] * u, tuple(sp.get("focus", (0.5, 0.45))))
        # grade
        if sl["role"] == "post":
            f = saturate(img, g["post_sat"], g["post_warm"], g["post_lift"])
        elif sl["role"] == "build":
            if t >= self.drop:                               # the drop beat: near-grey, slightly lifted
                f = saturate(img, g["drop_sat"], (1, 1, 1), 1.18)
            else:
                b = smooth((t - sl["bright_from"]) / (sl["bright_to"] - sl["bright_from"]))
                f = img.astype(np.float32)
                f = f + (255 - f) * (sl["bright_gain"] - 1) * 0.55 * b   # wash toward white, keeps detail
        else:
            f = saturate(img, g["pre_sat"])
        if sl.get("motion") == "flash" and u < 0.25:
            f = f + (255 - f) * 0.35 * (1 - u / 0.25)
        cap = self.cap_post if t >= self.drop else self.cap_pre
        f = cap.draw(np.clip(f, 0, 255))
        f = f * (1 - self.wm * 0.7) + 235 * self.wm * 0.7
        return np.clip(f, 0, 255).astype(np.uint8)

    # ---------------- audio ----------------
    def mix(self, out):
        sr = 44100
        y, s0 = sf.read(str(_p(self.m.get("music_file", self.t["music"]))), always_2d=True)
        assert s0 == sr
        n = int(self.n / self.fps * sr)                       # body + outro
        music = np.zeros((n, 2), np.float32)
        music[:min(n, len(y))] = y[:n]
        voice = np.zeros_like(music)
        from pedalboard import Pedalboard, HighpassFilter, Compressor, PeakFilter, Limiter
        board = Pedalboard([HighpassFilter(90), PeakFilter(3200, 2.5, 1.0), Compressor(-18, 3.0, 5, 120), Limiter(-1.0)])
        for v in self.m.get("voicelines", []):
            x, vs = sf.read(str(_p(v["file"])), always_2d=True)
            if vs != sr:
                import librosa
                x = librosa.resample(x.T, orig_sr=vs, target_sr=sr).T
            if x.shape[1] == 1:
                x = np.repeat(x, 2, 1)
            x = board(x.T.astype(np.float32), sr).T * 10 ** (v.get("gain_db", 0) / 20)
            i = int(v["at"] * sr)
            j = min(n, i + len(x))
            voice[i:j] += x[:j - i]
        if np.abs(voice).max() > 0:
            from scipy.ndimage import maximum_filter1d, uniform_filter1d
            act = (uniform_filter1d(np.abs(voice).max(1), int(0.03 * sr)) > 0.01).astype(np.float32)
            act = uniform_filter1d(maximum_filter1d(act, int(0.25 * sr)), int(0.12 * sr))
            duck = 10 ** (-self.m.get("duck_db", 4.0) / 20)
            music *= (1 - act * (1 - duck))[:, None]
            # voice sits ~3 dB over the ducked music
            vr = np.sqrt(np.mean(voice[act > 0.5] ** 2) + 1e-9)
            mr = np.sqrt(np.mean(music[act > 0.5] ** 2) + 1e-9)
            voice *= min(4.0, (mr * self.m.get("voice_over_music", 1.4)) / vr)
        mix = music + voice
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        target = self.m.get("lufs", -14.0)
        from scipy.ndimage import maximum_filter1d as _mf, uniform_filter1d as _uf
        for _ in range(3):                     # (pedalboard's Limiter adds make-up gain, so limit by hand)
            mix = pyln.normalize.loudness(mix, meter.integrated_loudness(mix), target)
            env = _mf(np.abs(mix).max(1), int(0.01 * sr))                # look-ahead peak envelope
            g = np.minimum(1.0, 0.94 / np.maximum(env, 1e-6))
            mix = mix * np.minimum(g, _uf(g, int(0.02 * sr)))[:, None]  # only the peaks that would clip
        fo = self.m.get("fade_out", 0)
        if fo:                                                 # smooth (equal-power) fade at the end
            k = int(fo * sr)
            mix[-k:] *= (np.cos(np.linspace(0, np.pi / 2, k)) ** 1.5)[:, None]
        sf.write(str(out), mix, sr)
        return meter.integrated_loudness(mix)

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
    print(Reveal(m, pv).render(ROOT / "output" / f"{m['id']}{'_preview' if pv else ''}.mp4"))
