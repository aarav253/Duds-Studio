"""'Rise' edit engine (template 8, measured from the Jinwoo reference).

Structure (frame-exact from the template timeline):
  intro   long hold on a silhouetted character lit blue/purple, slow ease-out
          zoom-out, black caption box (2nd line's first word red) baked into
          the picture so it zooms with it
  rise    the character as a CUTOUT with a blue rim glow rises slowly from the
          bottom (path measured: small peek, then a large ease-out rise) while
          the background flashes through panels every ~5 frames - grayscale,
          radial zoom-blurred, a white burst behind the head, brightness
          following the reference frame by frame. Repeats are fine (the
          viewer can't see them)
  drop    ~30-frame shots, vivid blue grade, fast ease-out zoom-in (1.3-1.7x)
  glitch  2-3 frame halftone-dot transitions between everything
  logo    white flash, then the watermark text glitching on black

Output is 1080x1920 with the 1080x1350 picture letterboxed like the reference.

    python src/rise_edit.py edits/<id>.json [--preview]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wekisava_edit import ROOT, apply_lut, cutout_mask, load_rgb, luma, warp  # noqa: E402
from xavier_edit import XavierEdit, zoom_blur  # noqa: E402

CAPTION_FONT = r"C:\Windows\Fonts\impact.ttf"


class RiseEdit(XavierEdit):
    def __init__(self, manifest, preview=False):
        super().__init__(manifest, preview)
        fw, fh = self.t.get("frame_size", self.t["size"])
        sc = self.W / self.t["size"][0]
        self.FW, self.FH = int(fw * sc) // 2 * 2, int(fh * sc) // 2 * 2
        self.y_off = int(round(self.t.get("content_y", 0) * sc))
        self.pool = manifest.get("bg_pool", [])
        self._auto_center = set(self.slots)

    # ---------------- helpers ----------------
    def _cut(self, slot):
        """(rgb float, mask) of a slot's image at source resolution, cached."""
        key = ("cut", slot)
        if key not in self._cache:
            path = self.slots[slot]
            src = load_rgb(path)
            m = cutout_mask(path)
            m = cv2.resize(m, (src.shape[1], src.shape[0]))
            self._cache[key] = (src, m)
        return self._cache[key]

    def _grade(self, g255, name, strength=0.9):
        rgb = np.repeat(np.clip(g255, 0, 255)[..., None], 3, -1)
        return self.match_section(rgb, name, name, strength=strength)

    def _grade_color(self, rgb, name, amt):
        """Light-touch grade for colour stills: the measured LUT's colour cast
        blended into the ORIGINAL colours (amt 0 = untouched)."""
        rgb = rgb.astype(np.float32)
        L = np.clip(rgb @ np.array([0.299, 0.587, 0.114], np.float32), 0, 255)
        lut = apply_lut(L.astype(np.uint8), self.luts[name]).astype(np.float32)
        return rgb * (1 - amt) + lut * amt

    def _crunch_img(self, img):
        g = luma(img).astype(np.float32)
        lo, hi = np.percentile(g, [3, 97])
        x = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
        x = np.clip((x - 0.5) * self.st.get("contrast", 1.25) + 0.5, 0, 1)
        return (x * x * (3 - 2 * x) * 0.5 + x * 0.5) * 255

    def _rim(self, m, color, width=6, glow=18, inner=0.8):
        mm = (m > 0.5).astype(np.uint8)
        r = max(3, int(width * self.W / 1080)) | 1
        edge = (cv2.dilate(mm, np.ones((r, r), np.uint8)) - mm).astype(np.float32)
        e = cv2.GaussianBlur(edge, (0, 0), 1.0)
        g = cv2.GaussianBlur(e, (0, 0), glow * self.W / 1080)
        ie = (mm - cv2.erode(mm, np.ones((3, 3), np.uint8))).astype(np.float32)
        return (e * 0.9 + g * 2.2)[..., None] * np.array(color, np.float32), cv2.GaussianBlur(ie, (0, 0), 0.8) * inner

    # ---------------- intro ----------------
    def _caption_on(self, img):
        cap = self.m.get("intro_caption")
        if not cap:
            return img
        W, H = self.W, self.H
        c = self.st.get("caption_box", {"y": 0.665, "h": 0.135, "x0": 0.0, "x1": 1.0})
        y0, y1 = int(c["y"] * H), int((c["y"] + c["h"]) * H)
        out = img.copy()
        out[y0:y1, int(c["x0"] * W):int(c["x1"] * W)] *= 0.08
        lay = Image.new("RGB", (W, H), (0, 0, 0))
        msk = Image.new("L", (W, H), 0)
        d, dm = ImageDraw.Draw(lay), ImageDraw.Draw(msk)
        f = ImageFont.truetype(CAPTION_FONT, int(0.047 * H))
        lines = cap if isinstance(cap, list) else [cap]
        for i, line in enumerate(lines):
            yc = y0 + (y1 - y0) * (0.30 + 0.42 * i)
            words = line.split(" ")
            red_first = i == 1
            total = d.textlength(line, font=f)
            x = W / 2 - total / 2
            for wi, wd in enumerate(words):
                col = (232, 30, 40) if (red_first and wi == 0) else (245, 245, 245)
                txt = wd + (" " if wi < len(words) - 1 else "")
                d.text((x, yc), txt, font=f, fill=col, anchor="lm")
                dm.text((x, yc), txt, font=f, fill=255, anchor="lm")
                x += d.textlength(txt, font=f)
        a = np.asarray(msk).astype(np.float32)[..., None] / 255
        return out * (1 - a) + np.asarray(lay).astype(np.float32) * a

    def intro_look(self, seg, k):
        key = ("introimg", seg["slot"])
        if key not in self._cache:
            pl = self.placed(seg["slot"], seg.get("overscan", 1.0))
            img = self._grade_color(pl.img, "intro", self.st.get("intro_grade", 0.35))
            self._cache[key] = self._caption_on(img)
        img = self._cache[key]
        sc, an, rot, dx, dy = self._mo(seg, k)
        return warp(img, sc, an, rot, dx, dy)

    # ---------------- drop ----------------
    def drop_look(self, seg, k):
        key = ("dropimg", seg["slot"])
        if key not in self._cache:
            ov = seg.get("overscan", 1.0)
            pl = self.placed(seg["slot"], ov)
            img = self._grade_color(pl.img, "drop", self.st.get("drop_grade", 0.4))
            self._cache[key] = img
        img = self._cache[key]
        sc, an, rot, dx, dy = self._mo(seg, k)
        img = warp(img, sc, an, rot, dx, dy)
        c = seg["motion"]["curve"]
        j = min(k - seg["start"], len(c) - 1)
        if j > 0:
            img = zoom_blur(img, (c[j] / c[j - 1]) ** self.st.get("zoom_blur", 1.6), an, n=9)
        return img

    # ---------------- rise ----------------
    def _bg_panel(self, seg, k):
        changes = seg.get("bg_changes", [])
        epoch = sum(1 for f in changes if f <= k)
        key = ("risebg", seg["start"], epoch)
        if key not in self._cache:
            pool = self.pool or list(self.slots.values())
            path = pool[(seg.get("pool_offset", 0) + epoch) % len(pool)]
            src = load_rgb(path)
            h, w = src.shape[:2]
            s = max(self.W / w, self.H / h) * 1.15
            im = cv2.resize(src, (int(w * s) + 1, int(h * s) + 1), interpolation=cv2.INTER_AREA)
            y0, x0 = (im.shape[0] - self.H) // 2, (im.shape[1] - self.W) // 2
            g = self._crunch_img(im[y0:y0 + self.H, x0:x0 + self.W])
            rgb = np.repeat(g[..., None], 3, -1)
            rgb = zoom_blur(rgb, self.st.get("bg_zoom_blur", 1.25), (0.5, 0.33), n=11)
            self._cache[key] = rgb
            for old in [c for c in self._cache if c[0] == "risebg" and c[1] == seg["start"] and c[2] < epoch - 1]:
                del self._cache[old]
        return self._cache[key]

    def _burst(self):
        key = ("burst",)
        if key not in self._cache:
            yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
            d = np.hypot((xx / self.W - 0.5) * 1.0, (yy / self.H - 0.30) * 1.25)
            ang = np.arctan2(yy / self.H - 0.30, xx / self.W - 0.5)
            rays = 0.5 + 0.5 * np.cos(ang * 22) * np.cos(ang * 9 + 1.3)
            self._cache[key] = (np.exp(-(d / 0.30) ** 2) * (0.6 + 0.4 * rays))[..., None]
        return self._cache[key]

    def _rise_pos(self, seg, k):
        r = seg.get("rise")
        if r:
            # one continuous exponential ease-out (measured on the reference:
            # top edge 0.33 at +67f, ~0.14 at +151f, 0.08 at +231f -> tau ~55f)
            t = max(0.0, k - seg["start"])
            return r["to"] + (r["from"] - r["to"]) * math.exp(-t / r["tau"]), r["width"]
        path = seg["path"]                         # [[frame, top_frac, width_frac], ...]
        f = np.array([p[0] for p in path], float)
        return float(np.interp(k, f, [p[1] for p in path])), float(np.interp(k, f, [p[2] for p in path]))

    def rise_frame(self, seg, k):
        W, H = self.W, self.H
        bg = self._bg_panel(seg, k)
        ref = seg.get("bg_luma")
        gain = 1.0
        if ref:
            j = min(k - seg["start"], len(ref) - 1)
            gain = ref[j] / max(np.median(ref), 1)
        bg = bg * gain
        bg = bg + self._burst() * 255 * self.st.get("burst", 0.75) * min(1.4, gain)
        bg = self._grade(np.clip(bg.mean(-1), 0, 255), "rise", 0.5)
        # cutout, placed by the measured path
        src, m = self._cut(seg["slot"])
        ys, xs = np.nonzero(m > 0.5)
        if not len(ys):
            return bg
        bx0, bx1 = np.percentile(xs, 1), np.percentile(xs, 99)
        by0 = np.percentile(ys, 1)
        top, wf = self._rise_pos(seg, k)
        s = wf * W / max(bx1 - bx0, 1)
        by1 = np.percentile(ys, 99.5)
        if by1 > 0.97 * m.shape[0]:
            # the body runs off the bottom of the source (a cropped clip
            # frame): ONE constant scale so it still reaches the bottom of the
            # frame at its final height - the cutout never resizes while rising
            fin = seg["rise"]["to"] if seg.get("rise") else top
            s = max(s, (1.04 - fin) * H / max(by1 - by0, 1))
        cx = (bx0 + bx1) / 2
        M = np.float32([[s, 0, W / 2 - cx * s], [0, s, top * H - by0 * s]])
        cs = cv2.warpAffine(src.astype(np.float32), M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
                            borderValue=0)
        cm = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        char = self._grade_color(cs, "drop", self.st.get("rise_grade", 0.3))
        rim, inner = self._rim(cm, [70, 110, 255], width=7, glow=22)
        a = cm[..., None]
        out = bg * (1 - a) + char * a + rim + inner[..., None] * np.array([150, 180, 255], np.float32)
        return out

    # ---------------- glitch / logo ----------------
    def _ink_dots(self, img, dark_gain=1.0, pitch=None):
        """Dark ink dots on white paper (the reference's transition frames),
        with a slight RGB split."""
        pitch = pitch or 14 * self.W / 1080
        H, W = img.shape[:2]
        g = np.clip(img.mean(-1) / 255, 0, 1)
        gh, gw = int(H / pitch), int(W / pitch)
        cells = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
        dark = np.clip((1 - cells) * dark_gain, 0, 1)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        iy = np.minimum((yy / (H / gh)).astype(int), gh - 1)
        ix = np.minimum((xx / (W / gw)).astype(int), gw - 1)
        d = dark[iy, ix]
        dy = yy - (iy + 0.5) * H / gh
        dx = xx - (ix + 0.5) * W / gw
        R = 0.5 * pitch * 1.2 * np.sqrt(d)
        a = np.clip(R - np.hypot(dx, dy) + 0.7, 0, 1)
        v = 255 * (1 - a * 0.95)
        out = np.stack([np.roll(v, 2, 1), v, np.roll(v, -2, 1)], -1)
        return out

    def glitch_frame(self, seg, k):
        prev_seg, nxt = self.neighbours_any(seg)
        j = k - seg["start"]
        n = seg["end"] - seg["start"] + 1
        if j < n - 1 or nxt is None:
            src = self.render_seg(prev_seg, prev_seg["end"]) if prev_seg else self.black()
        else:
            src = self.render_seg(nxt, nxt["start"])
        out = self._ink_dots(src, dark_gain=1.3)
        tgt = seg.get("lum_profile", [])
        if j < len(tgt):
            cur = float(out.mean())
            out = out * (tgt[j] / max(cur, 1))
        return out

    def neighbours_any(self, seg):
        i = self.timeline.index(seg)
        prev_seg = next((s for s in reversed(self.timeline[:i]) if s["type"] in ("shot", "rise")), None)
        nxt = next((s for s in self.timeline[i + 1:] if s["type"] in ("shot", "rise")), None)
        return prev_seg, nxt

    # ---------------- dispatch ----------------
    def render_seg(self, seg, k, apply_motion=True):
        key = ("frame", id(seg), k, apply_motion)
        if key in self._cache:
            return self._cache[key]
        t = seg["type"]
        if t == "shot":
            img = self.intro_look(seg, k) if seg["look"] == "intro" else self.drop_look(seg, k)
            img = self.light(img, seg["look"])
        elif t == "rise":
            img = self.light(self.rise_frame(seg, k), "rise")
        elif t == "glitch":
            img = self.glitch_frame(seg, k)
        elif t == "logo":
            img = self.logo_frame(seg, k)
        else:
            img = self.black()
        img = np.clip(img, 0, 255)
        frames = [c for c in self._cache if c[0] == "frame"]
        if len(frames) > 12:
            for c in frames[:6]:
                del self._cache[c]
        self._cache[key] = img
        return img

    def frame(self, k):
        seg = self.seg_at(k)
        img = self.render_seg(seg, k) if seg else self.black()
        if seg is not None and (seg["type"] == "rise" or (seg["type"] == "shot" and seg.get("look") == "drop")):
            fx = self.flash(seg, k, self.st.get("flash_tint", [120, 150, 255]))
            if fx is not None:
                img = np.clip(img + fx, 0, 255)
        if self.wm is not None and seg is not None and seg["type"] not in ("black", "logo") and img.mean() > 12:
            img = img * (1 - self.wm) + 235 * self.wm
        rng = np.random.default_rng(k)
        img = img + rng.normal(0, self.st["grain"], (self.H // 2, self.W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)
        return np.clip(img, 0, 255).astype(np.uint8)

    def render(self, out_path, log=print):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".video.mp4")
        wr = imageio_ffmpeg.write_frames(str(tmp), (self.FW, self.FH), fps=self.fps, codec="libx264",
                                         pix_fmt_out="yuv420p", output_params=["-crf", "16", "-preset", "medium"],
                                         macro_block_size=2)
        wr.send(None)
        canvas = np.zeros((self.FH, self.FW, 3), np.uint8)
        for k in range(self.n):
            canvas[:] = 0
            canvas[self.y_off:self.y_off + self.H, :self.W] = self.frame(k)
            wr.send(np.ascontiguousarray(canvas))
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
    return RiseEdit(m, preview).render(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(render_manifest(sys.argv[1], preview="--preview" in sys.argv))
