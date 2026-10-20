"""Freestyle beat-synced clip edit (no reference template).

Cuts are generated from the music analysis (beat_this + demucs drum onsets):
  intro     one cut every 2 beats, smooth punch-in transitions
  build     the last bar before the drop accelerates: beat, then half-beat cuts
  drop      white flash on the drop downbeat, then a cut on every beat plus
            every strong drum hit in between
  outro     final shot eases out and fades
Every shot is a velocity edit (clip plays fast -> slow -> fast), zoom-punches
and shakes on drum hits, and joins the next shot with a directional whip
(motion-blurred slide) or a punch-in, so the motion never stops.

    python src/freestyle_edit.py edits/<id>.json [--preview]
manifest: {id, music, clips: [...], caption, colour: "yellow", watermark}
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_analysis  # noqa: E402

FPS = 60
W0, H0 = 1080, 1350
FONT = r"C:\Windows\Fonts\impact.ttf"


def ease_out(u, p=3.0):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** p


def ease_in(u, p=2.0):
    u = min(max(u, 0.0), 1.0)
    return u ** p


class Clip:
    def __init__(self, path, W, H):
        self.cap = cv2.VideoCapture(str(ROOT / path))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W, self.H, self.pos, self.cache = W, H, -1, {}

    def frame(self, i):
        i = int(min(max(i, 0), self.n - 1))
        if i in self.cache:
            return self.cache[i]
        if i != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, b = self.cap.read()
        self.pos = i
        if not ok:
            b = np.zeros((self.H, self.W, 3), np.uint8)
        f = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float32)
        if len(self.cache) > 8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[i] = f
        return f


def plan_cuts(a):
    beats, down = a["beats"], a["downbeats"]
    drop = a.get("drop") or down[len(down) // 2]
    drop = min(down, key=lambda d: abs(d - drop))
    dur = a["duration"]
    hits = [(o["t"], o["strength"]) for o in a["drum_onsets"]]
    cuts = [0.0]
    pre = [b for b in beats if b < drop]
    first = down[0] if down else beats[0]
    # intro: every 2 beats from the first downbeat until one bar before the drop
    bar = 4 * (beats[1] - beats[0]) if len(beats) > 1 else 2.0
    build_start = drop - bar
    k = next(i for i, b in enumerate(beats) if b >= first - 0.01)
    while k < len(beats) and beats[k] < build_start - 0.05:
        cuts.append(beats[k])
        k += 2
    # build: every beat, then half beats in the final 2 beats
    bb = [b for b in pre if b >= build_start - 0.05]
    for i, b in enumerate(bb):
        cuts.append(b)
        if i >= len(bb) - 2 and i + 1 < len(bb):
            cuts.append((b + bb[i + 1]) / 2)
        elif i == len(bb) - 1:
            cuts.append((b + drop) / 2)
    # drop: every beat + strong hits between beats
    last_down = max(d for d in down if d < dur - 0.3)
    post = [b for b in beats if drop <= b < last_down]
    cuts += post
    for t, s in hits:
        if drop < t < last_down and s >= 0.62 and min(abs(t - c) for c in cuts) > 0.16:
            cuts.append(t)
    cuts.append(last_down)
    cuts = sorted(set(round(c, 3) for c in cuts))
    cuts = [c for i, c in enumerate(cuts) if i == 0 or c - cuts[i - 1] >= 0.14]
    return cuts, drop, last_down, dur, hits


class Freestyle:
    def __init__(self, m, preview=False):
        self.m = m
        self.W, self.H = (W0 // 2, H0 // 2) if preview else (W0, H0)
        a = music_analysis.analyze(ROOT / m["music"])
        self.cuts, self.drop, self.last, self.dur, hits = plan_cuts(a)
        self.n = int(self.dur * FPS)
        self.F = [int(round(c * FPS)) for c in self.cuts] + [self.n]
        self.hits = [(t * FPS, s) for t, s in hits if s >= 0.45]
        self.downs = [d * FPS for d in a["downbeats"]]
        clips = m["clips"]
        self.shots = []
        for i in range(len(self.F) - 1):
            path = clips[i % len(clips)]
            self.shots.append({"clip": Clip(path, self.W, self.H), "a": self.F[i], "b": self.F[i + 1] - 1,
                               "dir": 1 if i % 2 else -1, "off": (i * 37) % 25})
        self.tint = np.array(m.get("tint", [255, 205, 60]), np.float32)
        self.cap = self._caption(m.get("caption", ""))
        self.wm = self._wm(m.get("watermark", "DUDS"))
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        self.vig = (1 - 0.35 * np.clip(((xx / self.W - .5) / .6) ** 2 + ((yy / self.H - .47) / .7) ** 2, 0, 1.4))[..., None]
        print(f"{len(self.shots)} shots, drop at {self.drop:.2f}s, avg {np.mean(np.diff(self.F)) / FPS:.2f}s per shot")

    # ---------- text ----------
    def _caption(self, text):
        if not text:
            return None
        lay = Image.new("L", (self.W, self.H), 0)
        d = ImageDraw.Draw(lay)
        f = ImageFont.truetype(FONT, int(0.052 * self.H))
        while d.textlength(text, font=f) > 0.86 * self.W:
            f = ImageFont.truetype(FONT, int(f.size * 0.94))
        d.text((self.W / 2, 0.70 * self.H), text, font=f, fill=255, anchor="mm", stroke_width=max(2, int(0.004 * self.W)),
               stroke_fill=0)
        a = np.asarray(lay).astype(np.float32) / 255
        st = Image.new("L", (self.W, self.H), 0)
        ImageDraw.Draw(st).text((self.W / 2, 0.70 * self.H), text, font=f, fill=255, anchor="mm",
                                stroke_width=max(3, int(0.006 * self.W)), stroke_fill=255)
        return a, np.asarray(st).astype(np.float32) / 255

    def _wm(self, text):
        lay = Image.new("L", (self.W, self.H), 0)
        f = ImageFont.truetype(r"C:\Windows\Fonts\bahnschrift.ttf", int(self.H * 0.018))
        ImageDraw.Draw(lay).text((self.W / 2, self.H * 0.955), " ".join(text), font=f, fill=200, anchor="mm")
        return np.asarray(lay).astype(np.float32)[..., None] / 255

    # ---------- motion ----------
    def hit_env(self, k, tau=5.0):
        e = 0.0
        for f, s in self.hits:
            d = k - f
            if 0 <= d < 5 * tau:
                e += s * math.exp(-d / tau)
        return min(e, 1.5)

    def shot_img(self, sh, k):
        """Velocity edit: source position p(u) = u + 0.6 sin(2pi u)/(2pi)
        -> speed 1.6x at the cut, 0.4x mid-shot, back to 1.6x into the next cut.
        Fast parts are motion-blurred by blending sub-frames."""
        n = sh["b"] - sh["a"] + 1
        span = max(n * 1.15, 12)
        u = (k - sh["a"]) / max(1, n)
        amp = 0.6 if n > 18 else 0.3
        p = lambda uu: sh["off"] + span * (uu + amp * math.sin(2 * math.pi * uu) / (2 * math.pi))
        speed = 1.15 * (1 + amp * math.cos(2 * math.pi * u))
        c = sh["clip"]
        if speed > 1.3:
            subs = [c.frame(p(u + d / n)) for d in (-0.33, 0.0, 0.33)]
            return sum(subs) / 3
        return c.frame(p(u))

    def shot_frame(self, i, k):
        sh = self.shots[i]
        img = self.shot_img(sh, k)
        n = sh["b"] - sh["a"] + 1
        j = k - sh["a"]
        e = self.hit_env(k)
        # punch-in at the cut + slow push + hit punches
        s = 1.0 + 0.10 * (1 - ease_out(j / 8, 3)) + 0.04 * j / n + 0.045 * e
        rng = np.random.default_rng(k * 13 + 1)
        sx, sy = rng.normal(0, 1, 2) * (2 + 9 * e) * self.W / 1080
        rot = rng.normal(0, 1) * 0.5 * e
        M = cv2.getRotationMatrix2D((self.W / 2, self.H * 0.45), rot, s)
        M[0, 2] += sx
        M[1, 2] += sy
        img = cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if j < 6:                                              # radial zoom blur right after the cut
            amt = 0.05 * (1 - j / 6)
            acc = img.copy()
            for q in (1, 2, 3):
                acc += cv2.warpAffine(img, cv2.getRotationMatrix2D((self.W / 2, self.H * 0.45), 0, 1 + amt * q / 3),
                                      (self.W, self.H), borderMode=cv2.BORDER_REFLECT)
            img = acc / 4
        if e > 0.5:                                            # RGB split on strong hits
            d = int(min(10, 6 * e) * self.W / 1080)
            img = np.stack([np.roll(img[..., 0], d, 1), img[..., 1], np.roll(img[..., 2], -d, 1)], -1)
        return img

    def whip(self, img, dx, blur):
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        out = cv2.warpAffine(img, M, (self.W, self.H), borderMode=cv2.BORDER_REFLECT)
        kw = int(min(blur, self.W * 0.5)) | 1
        return cv2.blur(out, (kw, 1)) if kw > 2 else out

    def frame(self, k):
        i = max(j for j in range(len(self.shots)) if self.shots[j]["a"] <= k)
        sh = self.shots[i]
        t = k / FPS
        img = self.shot_frame(i, k)
        in_drop = t >= self.drop - 0.02
        W = self.W
        # transitions: last 3 frames whip out, first 3 frames whip in (drop);
        # intro uses the punch-in only
        if in_drop and i + 1 < len(self.shots):
            to_end = sh["b"] - k
            if to_end < 3:
                v = (3 - to_end) / 3
                img = self.whip(img, sh["dir"] * ease_in(v, 2) * 0.35 * W, ease_in(v, 2) * 0.45 * W)
        if in_drop and i > 0:
            j = k - sh["a"]
            if j < 4:
                v = 1 - ease_out(j / 4, 2.5)
                img = self.whip(img, -self.shots[i - 1]["dir"] * v * 0.35 * W, v * 0.45 * W)
        # grade: keep the clip, lift contrast, tinted bloom of the highlights
        L = img.mean(-1, keepdims=True)
        hi = np.clip((L - 150) / 105, 0, 1)
        bloom = cv2.GaussianBlur(hi[..., 0], (0, 0), 18 * W / 1080)[..., None]
        img = np.clip((img - 128) * 1.12 + 128, 0, 255) + bloom * self.tint * 0.45
        img *= self.vig
        # flashes: white on the drop and on downbeats in the drop, light pops on hits
        for f in self.downs:
            d = k - f
            if 0 <= d < 8 and f / FPS >= self.drop - 0.02:
                a = (0.95 if abs(f / FPS - self.drop) < 0.05 else 0.55) * math.exp(-d / 2.2)
                img = img * (1 - a) + 255 * a
        e = self.hit_env(k, 3.0)
        img = img * (1 + 0.18 * e)
        # fades
        if k < 18:
            img *= k / 18
        if k > self.n - 24:
            img *= max(0, (self.n - k) / 24)
        if self.cap is not None and t < self.drop - 0.05:
            a, st = self.cap
            fade = min(1, k / 20) * min(1, (self.drop - t) / 0.15)
            img = img * (1 - st[..., None] * 0.6 * fade)
            img = img * (1 - a[..., None] * fade) + 250 * a[..., None] * fade
        img = img * (1 - self.wm * 0.8) + 230 * self.wm * 0.8
        return np.clip(img, 0, 255).astype(np.uint8)

    def render(self, out):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".video.mp4")
        wr = imageio_ffmpeg.write_frames(str(tmp), (self.W, self.H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "16", "-preset", "medium"], macro_block_size=2)
        wr.send(None)
        for k in range(self.n):
            wr.send(np.ascontiguousarray(self.frame(k)))
        wr.close()
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp), "-i",
                        str(ROOT / self.m["music"]), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "256k", "-t", f"{self.n / FPS:.3f}", str(out)], check=True)
        tmp.unlink()
        return out


if __name__ == "__main__":
    p = Path(sys.argv[1])
    m = json.loads((p if p.is_absolute() else ROOT / p).read_text(encoding="utf-8-sig"))
    out = ROOT / "output" / f"{m['id']}.mp4"
    print(Freestyle(m, preview="--preview" in sys.argv).render(out))
