"""Xavier-style manga edit engine (templates 5 and 6).

Built on the wekisava engine (same timeline format, measured camera curves,
AR(1) handheld shake, beat-synced zoom punches, horizontal smear transitions,
white flashes) with the Xavier looks, measured frame by frame on the
references:

  intro   navy duotone panels (blue shadows, white paper), crushed contrast,
          every shot snaps in at ~2.1x and eases out to 1.0 over ~1 s with
          zoom blur; a bold caption with a soft drop shadow sits on top
  duo     the same duotone in the drop section
  red     duotone character; the background's ink turns crimson/pink and
          glowing pink light bars streak sideways behind him
  yellow  the same in acid yellow, plus a white rim line around the character
  lines   dark duotone panel with white horizontal light bars (build-up)
  bright  high-key duotone (the white outro section)
  card    the panel as an inset card over a dark, blurred copy of itself

    python src/xavier_edit.py edits/<id>.json [--preview]
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wekisava_edit import ROOT, WekisavaEdit, apply_lut, lightning_layer, luma, warp  # noqa: E402

FONT = r"C:\Windows\Fonts\seguibl.ttf"          # closest installed match to the reference's heavy geometric sans


def zoom_blur(img, ratio, anchor, n=7):
    """Average of the frame scaled between 1 and `ratio` about `anchor`
    (the motion blur of a fast zoom)."""
    if abs(math.log(ratio)) < 0.012:
        return img
    acc = np.zeros_like(img, np.float32)
    for i in range(n):
        acc += warp(img, ratio ** (i / (n - 1)), anchor)
    return acc / n


class XavierEdit(WekisavaEdit):
    def __init__(self, manifest, preview=False):
        super().__init__(manifest, preview)
        self.acc = {k: {q: np.array(v, np.float32) for q, v in d.items()} for k, d in self.st["accent"].items()}
        self.cap_text = manifest.get("captions", {})
        # every panel is framed on its character unless the manifest pins a focus
        self._auto_center = set(self.slots)

    # ---- panels ----
    def _crunch(self, slot, ov):
        """Panel luma with the reference's crushed contrast (true black ink,
        white paper), cached per slot."""
        key = ("crunch", slot, ov)
        if key not in self._cache:
            g = luma(self.placed(slot, ov).img).astype(np.float32)
            # dense screentone/hatching aliases into shimmer under the handheld
            # shake: soften it just below the pixel grid
            g = cv2.GaussianBlur(g, (0, 0), self.st.get("detail_soften", 0.8) * self.W / 1080)
            lo, hi = np.percentile(g, [3, 97])
            x = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
            c = self.st.get("contrast", 1.25)
            x = np.clip((x - 0.5) * c + 0.5, 0, 1)
            x = x * x * (3 - 2 * x) * 0.5 + x * 0.5
            x = self._shade(x, slot, ov)
            self._cache[key] = x * 255
        return self._cache[key]

    def _shade(self, x, slot, ov):
        """Tone shading for line-art panels. The reference panels are
        cel-shaded (skin/hair/shirt are a mid periwinkle tone, lighter on top);
        clean line art is white paper with black lines. Fill the large white
        areas INSIDE the character with a mid tone that is lighter toward the
        top (light from above) and darkens next to the ink lines."""
        amt = self.st.get("tone_amt", 0.0)
        if amt <= 0:
            return x
        m = self.mask(slot, ov)
        if m.mean() < 0.02:
            return x
        H, W = x.shape
        paper = (x > 0.8).astype(np.uint8)
        paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        d = cv2.distanceTransform(paper, cv2.DIST_L2, 5) / (W / 1080)
        ys, _ = np.nonzero(m > 0.5)
        top, bot = (np.percentile(ys, 2), np.percentile(ys, 98)) if len(ys) else (0, H)
        yy = np.arange(H, dtype=np.float32)[:, None]
        up = np.clip(1 - (yy - top) / max(bot - top, 1), 0, 1)              # 1 at the top of the character
        tone = self.st.get("char_tone", 0.6) + 0.22 * up ** 1.5 - 0.14 * np.exp(-d / 7.0)
        a = cv2.GaussianBlur(paper.astype(np.float32), (0, 0), 1.5) * np.clip(m * 1.2, 0, 1) * amt
        return np.clip(x * (1 - a) + tone * a, 0, 1)

    def _duo(self, g, look, strength=None):
        rgb = np.repeat(g[..., None], 3, -1)
        return self.match_section(rgb, look, look, strength=self.st.get("match", 0.85) if strength is None else strength)

    def _bars(self, seg, k, n=None, thick=(3, 16)):
        """Horizontal glowing light bars scrolling sideways (0..~1.4)."""
        epoch = sum(1 for f in seg.get("bar_reseed", []) if f <= k)    # new bar layout on each reference flicker
        key = ("bars", seg["start"], epoch)
        W, H = self.W, self.H
        if key not in self._cache:
            rng = np.random.default_rng(seg["start"] * 3 + 1 + 7919 * epoch)
            lay = np.zeros((H, W * 2), np.float32)
            for _ in range(n or seg.get("bars", 26)):
                y = int(rng.uniform(0.02, 0.98) * H)
                t = max(1, int(rng.uniform(*thick) * H / 1350))
                x0 = int(rng.uniform(0, 2) * W)
                ln = int(rng.uniform(0.25, 0.9) * W)
                v = float(rng.uniform(0.5, 1.0))
                for x in (x0, x0 - 2 * W):                     # wraps seamlessly
                    cv2.rectangle(lay, (x, y), (x + ln, y + t), v, -1)
            core = cv2.GaussianBlur(lay, (0, 0), 1.2)
            glow = cv2.GaussianBlur(lay, (0, 0), 9 * H / 1350)
            self._cache[key] = (core, glow, float(rng.choice([-1, 1])))
        core, glow, d = self._cache[key]
        off = int((k - seg["start"]) * seg.get("bar_speed", 26) * W / 1080 * d) % (2 * W)
        c = np.roll(core, off, axis=1)[:, :W]
        g = np.roll(glow, off, axis=1)[:, :W]
        return c, g

    def _ink_noise(self, seg):
        key = ("inknoise", seg["start"])
        if key not in self._cache:
            # chunky paint splotches: two octaves of noise, hard-edged threshold
            rng = np.random.default_rng(seg["start"] + 11)
            big = cv2.resize(rng.random((14, 12)).astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_CUBIC)
            small = cv2.resize(rng.random((60, 48)).astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_CUBIC)
            n = 0.65 * big + 0.35 * small
            x = np.clip((n - self.st.get("splotch_thresh", 0.47)) / 0.10, 0, 1)
            self._cache[key] = x * x * (3 - 2 * x)
        return self._cache[key]

    def look_accent(self, seg, k, g, m, look):
        """Reference red/yellow look: the character stays NAVY duotone; the
        background is near-black (dark navy panel texture) with SPARSE, bright,
        glowing accent paint concentrated behind the silhouette, light bars
        with white-hot cores, and a crisp white rim + accent glow on the
        character's outline."""
        A = self.acc[look]
        W = self.W
        char = self._duo(g, self.st.get("char_grade", look)) * self.st.get("char_gain", 1.0)
        bgm = 1 - m
        if m.mean() > 0.45:
            # close-up: the character fills the frame, so (like the reference
            # close-ups) the accent takes over the dark ink toward the edges
            key = ("ring",)
            if key not in self._cache:
                yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
                r = np.hypot((xx / self.W - 0.5) / 0.5, (yy / self.H - 0.42) / 0.55)
                self._cache[key] = np.clip((r - 0.55) / 0.35, 0, 1)
            ink = np.clip((120 - g) / 80, 0, 1)
            bgm = np.maximum(bgm, self._cache[key] * ink * 0.95)
            m = 1 - bgm
        base = self._duo(g * self.st.get("bg_dim", 0.3), "duo", strength=0.0)
        mm = (m > 0.5).astype(np.uint8)
        key = ("prox", seg["slot"])
        if key not in self._cache:
            d = cv2.distanceTransform(1 - mm, cv2.DIST_L2, 5) / W
            self._cache[key] = np.exp(-d / self.st.get("accent_reach", 0.22))
        prox = self._cache[key]
        paint = self._ink_noise(seg) * prox * bgm * self.st.get("accent_amt", 0.75)
        bloom = cv2.GaussianBlur(paint, (0, 0), 16 * W / 1080)
        bgc = base * (1 - paint[..., None]) + A["p50"] * self.st.get("paint_gain", 1.25) * paint[..., None]             + A["p25"] * (bloom * 0.8)[..., None]
        out = char * m[..., None] + bgc * bgm[..., None]
        core, glow = self._bars(seg, k, thick=(4, 22))
        lvl = self.st.get("bar_level", 0.85)
        out = out + (glow * bgm * lvl)[..., None] * A["p50"] * self.st.get("bar_glow", 1.1) +\
            (np.clip(core * bgm, 0, 1) ** 1.5 * lvl)[..., None] * (0.35 * A["p90"] + 0.65 * 255) * self.st.get("bar_core", 1.0)
        r = max(3, int(4 * W / 1080)) | 1
        edge = (cv2.dilate(mm, np.ones((r, r), np.uint8)) - mm).astype(np.float32)
        e = cv2.GaussianBlur(edge, (0, 0), 0.8)
        out = out + cv2.GaussianBlur(e, (0, 0), 8 * W / 1080)[..., None] * A["p90"] * self.st.get("rim_glow", 1.6) \
            + e[..., None] * self.st.get("rim_white", 210)
        return out

    def look_lines(self, seg, k, g, m):
        # reference build-up: dark navy panel (median luma ~55), character a
        # little brighter than the background, white light bars across it
        base = self._duo(g * (self.st.get("lines_dim", 0.3) + 0.35 * m), "duo", strength=0.0)
        core, glow = self._bars(seg, k, thick=(4, 22))
        b = np.clip(glow * 1.2 + core, 0, 1.3) * (1 - m * 0.85)
        out = base + b[..., None] * np.array([205, 215, 255], np.float32) * 0.9
        if "lightning" in seg.get("fx", []) and (k // 3) % 4 != 3:
            ys, xs = np.nonzero(cv2.Canny((m > 0.5).astype(np.uint8) * 255, 50, 150)[::8, ::8])
            gl, co = lightning_layer(self.W, self.H, np.random.default_rng((seg["start"], k // 3)), list(zip(xs * 8, ys * 8)), n=2)
            out = out + gl[..., None] * np.array([150, 170, 255], np.float32) * 0.6 + co[..., None] * 240
        return out

    def look_card(self, seg, k, slot, ov):
        g = self._crunch(slot, ov)
        bg = self._duo(cv2.GaussianBlur(g, (0, 0), 18) * 0.45, "duo", strength=0.3)
        s = seg.get("card_scale", 0.62)
        cw, ch = int(self.W * s), int(self.H * s)
        card = self._duo(cv2.resize(g, (cw, ch), interpolation=cv2.INTER_AREA), "bright" if "bright" in self.luts else "duo")
        x0, y0 = (self.W - cw) // 2, (self.H - ch) // 2
        out = bg.copy()
        out[y0 - 4:y0 + ch + 4, x0 - 4:x0 + cw + 4] = 245
        out[y0:y0 + ch, x0:x0 + cw] = card
        return out

    # ---- frame builder ----
    def shot_frame(self, seg, k, apply_motion=True):
        look, slot = seg["look"], seg["slot"]
        ov = seg.get("overscan", 1.0)
        if look not in ("intro", "lines"):
            # framing room depends on the image: face crops are already tight,
            # full-body panels need room to frame the face
            p = self.slots[slot]
            ov = 1.0 if "_face" in p else 1.15
        if look == "card":
            img = self.look_card(seg, k, slot, ov)
        else:
            if look == "intro" and self.st.get("intro_raw", False):
                # intro panels: shown exactly as supplied (unedited)
                img = self.placed(slot, ov).img.astype(np.float32)
            else:
                g = self._crunch(slot, ov)
            if look == "intro" and self.st.get("intro_raw", False):
                pass
            elif look in ("intro", "duo", "bright"):
                img = self._duo(g, look)
            else:
                m = self.mask(slot, ov)
                img = self.look_lines(seg, k, g, m) if look == "lines" else self.look_accent(seg, k, g, m, look)
        if apply_motion:
            sc, an, rot, dx, dy = self._mo(seg, k)
            img = warp(img, sc, an, rot, dx, dy)
            # zoom blur from the measured scale velocity (the intro snap-in)
            if k > seg["start"] and seg.get("motion", {}).get("type") == "curve":
                c = seg["motion"]["curve"]
                j = min(k - seg["start"], len(c) - 1)
                ratio = c[j - 1] / c[j]
                img = zoom_blur(img, ratio ** self.st.get("zoom_blur", 1.0), an, n=9)
        if look == "intro" and self.st.get("intro_raw", False):
            return np.clip(img, 0, 255)
        return np.clip(self.light(img, look), 0, 255)

    def _light_field(self, look):
        key = ("lightfield", look)
        if key not in self._cache:
            p = self.st.get("light", {}).get(look, {})
            yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
            u, v = yy / self.H, xx / self.W
            top = 1 + p.get("top", 0.0) * (1 - 2 * u)                   # light from above
            r2 = ((v - 0.5) / 0.62) ** 2 + ((u - 0.47) / 0.68) ** 2
            vig = 1 - p.get("vig", 0.0) * np.clip(r2, 0, 1.6)            # darker edges
            f = top * vig
            f = f / f.mean()                                             # redistribute light, keep the exposure
            self._cache[key] = f[..., None].astype(np.float32)
        return self._cache[key]

    def light(self, img, look):
        """Camera-space lighting shared by every look: top-down light,
        vignette, navy split-tone in the shadows and a true black point."""
        img = img * self._light_field(look)
        L = np.clip(img @ np.array([0.299, 0.587, 0.114], np.float32), 0, 255) / 255
        mx, mn = img.max(-1), img.min(-1)
        sat = np.clip(((mx - mn) / (mx + 1) - 0.25) * 2, 0, 1)            # leave the accent colours alone
        sh = ((1 - L) * (1 - sat))[..., None]
        img = img + sh * np.array(self.st.get("shadow_tint", [-8, -2, 22]), np.float32)
        bp = self.st.get("black_point", 0.0)
        return (img - bp) * (255 / (255 - bp))

    def smear_frame(self, seg, k):
        """Reference smears are crisp black/white streaks with a navy cast and
        surviving accent colour - not a grey wash: stretch to true black and
        white, re-tint with the duotone, then match the measured brightness."""
        out = super().smear_frame(seg, k)
        L = np.clip(out @ np.array([0.299, 0.587, 0.114], np.float32), 0, 255)
        lo, hi = np.percentile(L, [4, 97])
        Ls = np.clip((L - lo) / max(hi - lo, 1), 0, 1) * 255
        mx, mn = out.max(-1), out.min(-1)
        sat = np.clip(((mx - mn) / (mx + 1) - 0.3) * 3, 0, 1)[..., None]
        sat = np.maximum(sat, self.st.get("smear_color", 0.0))
        duo = apply_lut(Ls.astype(np.uint8), self.luts["duo"]).astype(np.float32)
        stretched = out * (Ls / np.maximum(L, 1))[..., None]
        out = duo * (1 - sat) + stretched * sat
        prof = seg.get("lum_profile")
        j = k - seg["start"]
        if prof and j < len(prof):
            tgt = prof[j]
            x = np.clip(out / 255, 0, 1)
            lo, hi = 0.15, 4.0
            for _ in range(14):                                         # bisect the gamma that hits the target mean
                gm = (lo * hi) ** 0.5
                cur = float(((x ** gm) @ np.array([0.299, 0.587, 0.114], np.float32)).mean()) * 255
                lo, hi = (gm, hi) if cur > tgt else (lo, gm)
            out = (x ** ((lo * hi) ** 0.5)) * 255
        return out

    def white_frame(self, seg, k):
        """Reference flash frames (T5 598, 632, 693, 892): the incoming shot
        smeared sideways and over-exposed almost to white - streaks of its
        colour (yellow / blue / red) survive. Not a flat white sheet."""
        prev_seg, nxt = self.neighbours(seg)
        if nxt is None:                                  # flash between two smears: use the nearest shot ahead
            i = self.timeline.index(seg)
            nxt = next((x for x in self.timeline[i + 1:] if x["type"] == "shot"), None)
        src_seg = nxt or prev_seg
        if src_seg is None:
            return super().white_frame(seg, k)
        j = k - seg["start"]
        L = seg["end"] - seg["start"] + 1
        f = src_seg["start"] if src_seg is nxt else src_seg["end"]
        base = self.render_seg(src_seg, f, apply_motion=True).astype(np.float32)
        W, H = self.W, self.H
        # row bands shifted sideways (same idea as the smear), then a long blur
        rng = np.random.default_rng(seg["start"] * 5 + j)
        out = base.copy()
        y = 0
        while y < H:
            bh = int(rng.uniform(6, 40) * H / 1350)
            out[y:y + bh] = np.roll(out[y:y + bh], int(rng.normal(0, 0.12) * W), axis=1)
            y += bh
        kw = int(self.st.get("flash_streak", 0.22) * W) | 1
        out = cv2.blur(out, (kw, 3))
        # over-expose: brights blow to white, colour survives in the streaks
        u = (j + 0.5) / L
        expo = self.st.get("flash_expo", 2.6) * (1.0 - 0.35 * u)
        out = 255 * (1 - np.exp(-out / 255 * expo))
        mx, mn = out.max(-1, keepdims=True), out.min(-1, keepdims=True)
        out = mn + (out - mn) * self.st.get("flash_sat", 1.5)         # keep the accent streaks vivid
        tgt = self.st.get("flash_level_luma", 228)                      # reference flash frames: luma ~230
        cur = float((out @ np.array([0.299, 0.587, 0.114], np.float32)).mean())
        if cur < tgt:
            w = (tgt - cur) / max(1.0, 255 - cur)
            out = out * (1 - w) + 255 * w
        fade = seg.get("fade_into_next", 0)
        if fade and j >= L - fade and nxt is not None:
            a = (j - (L - fade) + 1) / (fade + 1)
            out = out * (1 - a) + base * a
        return np.clip(out, 0, 255)

    def motion(self, seg, k):
        """Intro / build-up: the reference camera is a pure measured zoom -
        no handheld shake, no beat punches (those read as jitter there)."""
        if seg.get("look") in self.st.get("calm_looks", ("intro", "lines")):
            mo = seg.get("motion", {})
            c = mo.get("curve") or [1.0]
            j = min(max(k - seg["start"], 0), len(c) - 1)
            dy = mo["rise"][min(j, len(mo["rise"]) - 1)] * self.H if "rise" in mo else 0.0
            sc = c[j] / min(c)
            if mo.get("beat_bounce"):
                _, punch = self._hit_env(seg["start"], k)
                sc *= math.exp(punch * mo["beat_bounce"])
            return sc, mo.get("anchor", (0.5, 0.45)), 0.0, 0.0, dy
        return super().motion(seg, k)

    # ---- light flashes ----
    def flash(self, seg, k, tint):
        """Reference 'lights': on each drum hit a white-hot light flare hits
        for 1-3 frames and fades - a diagonal beam with a wide glow, a bloom
        burst and an exposure pop (T5 611, 742, 750). Returns an additive
        RGB layer (0 when nothing is flashing)."""
        out = None
        env = self.st.get("flash_env", [1.0, 0.7, 0.42, 0.22, 0.1])
        src = seg["flashes"] if "flashes" in seg else self.t.get("hits", [])
        for f, s in src:
            if s < self.st.get("flash_min", 0.45):
                continue
            j = int(round(k - f))
            if j < 0 or j >= len(env) or f < seg["start"] - 1 or f > seg["end"]:
                continue
            a = env[j] * min(1.2, s) * self.st.get("flash_level", 1.0)
            lay = self._flare(int(round(f)))
            add = lay[0][..., None] * 255 + lay[1][..., None] * np.array(tint, np.float32) * 1.4 + 22
            out = add * a if out is None else out + add * a
        return out

    def _flare(self, f):
        key = ("flare", f)
        if key not in self._cache:
            W, H = self.W, self.H
            rng = np.random.default_rng(f * 7 + 1)
            core = np.zeros((H, W), np.float32)
            ang = rng.uniform(-0.9, 0.9) + (0 if rng.random() < 0.5 else np.pi)
            cx, cy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.1, 0.55) * H
            L = 1.6 * max(W, H)
            p0 = (int(cx - L * np.cos(ang)), int(cy - L * np.sin(ang)))
            p1 = (int(cx + L * np.cos(ang)), int(cy + L * np.sin(ang)))
            cv2.line(core, p0, p1, 1.0, max(2, int(rng.uniform(0.012, 0.03) * W)))
            for _ in range(int(rng.integers(0, 3))):                       # thin secondary streaks
                o = rng.uniform(-0.12, 0.12) * W
                q0 = (int(p0[0] + o * np.sin(ang)), int(p0[1] - o * np.cos(ang)))
                q1 = (int(p1[0] + o * np.sin(ang)), int(p1[1] - o * np.cos(ang)))
                cv2.line(core, q0, q1, 0.6, max(1, int(0.004 * W)))
            core = cv2.GaussianBlur(core, (0, 0), 2.0 * W / 1080)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            bloom = np.exp(-(((xx - cx) / (0.28 * W)) ** 2 + ((yy - cy) / (0.28 * W)) ** 2))
            glow = cv2.GaussianBlur(core, (0, 0), 30 * W / 1080) * 3 + bloom * 0.55
            self._cache[key] = (np.clip(core, 0, 1), np.clip(glow, 0, 1.5))
        return self._cache[key]

    # ---- captions / watermark ----
    def _caption_layer(self, cap):
        key = ("caption", id(cap))
        if key not in self._cache:
            W, H = self.W, self.H
            lay = Image.new("L", (W, H), 0)
            d = ImageDraw.Draw(lay)
            for ln in cap["lines"]:
                text = self.cap_text.get(ln["key"], "")
                if not text:
                    continue
                f = ImageFont.truetype(FONT, int(ln["size"] * H))
                # shrink long captions to fit 88% of the width
                while d.textlength(text, font=f) > 0.88 * W and f.size > 10:
                    f = ImageFont.truetype(FONT, int(f.size * 0.94))
                d.text((W / 2, ln["y"] * H), text, font=f, fill=255, anchor="mm")
            a = np.asarray(lay).astype(np.float32) / 255
            sh = np.asarray(lay.filter(ImageFilter.GaussianBlur(7 * H / 1350))).astype(np.float32) / 255
            sh = np.roll(np.roll(sh, int(4 * H / 1350), 0), int(3 * W / 1080), 1)
            self._cache[key] = (a, np.clip(sh * 1.3, 0, 1))
        return self._cache[key]

    def _watermark(self, text):
        if not text:
            return None
        W, H = self.W, self.H
        f = ImageFont.truetype(FONT, int(H * 0.028))
        lay = Image.new("L", (W, H), 0)
        ImageDraw.Draw(lay).text((W / 2, H * self.st.get("wm_y", 0.822)), text.upper(), font=f, fill=140, anchor="mm")
        return np.asarray(lay).astype(np.float32)[..., None] / 255

    def frame(self, k):
        seg = self.seg_at(k)
        img = self.render_seg(seg, k) if seg else self.black()
        if seg is not None and seg.get("lum_follow"):
            # the reference's own per-frame brightness pattern (build-up strobe)
            ref = seg["lum_follow"]
            j = min(k - seg["start"], len(ref) - 1)
            r = ref[j] / max(seg["lum_base"], 1)
            if r >= 1:
                a = min(0.9, (r - 1) * seg["lum_base"] / max(1.0, 250 - seg["lum_base"]))
                img = img * (1 - a) + 250 * a
            else:
                img = img * max(r, 0.15)
        if seg is not None and seg["type"] == "shot" and seg.get("look") not in self.st.get("calm_looks", ()):
            tint = self.acc[seg["look"]]["p90"] if seg.get("look") in self.acc else self.st.get("flash_tint", [190, 205, 255])
            fx = self.flash(seg, k, tint)
            if fx is not None:
                img = img + fx
        if seg is not None and seg.get("orb") and seg["orb"][0] <= k:
            a0, a1 = seg["orb"]
            u = min(1.0, (k - a0) / max(1, a1 - a0))
            key = ("orbgrid",)
            if key not in self._cache:
                yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
                self._cache[key] = np.hypot(xx / self.W - seg.get("orb_at", [0.5, 0.62])[0],
                                            (yy / self.H - seg.get("orb_at", [0.5, 0.62])[1]) * self.H / self.W)
            r = 0.05 + 1.15 * u ** 3.0                      # eases in: grows slowly, then floods the frame
            d = self._cache[key]
            core = np.clip((r - d) / (0.02 + 0.08 * u), 0, 1)
            glow = np.exp(-np.maximum(d - r, 0) / (0.06 + 0.2 * u)) * 0.6
            a = np.clip(core + glow * (1 - core), 0, 1)[..., None]
            img = img * (1 - a) + 252 * a
        if seg is not None and seg["type"] == "shot":
            fl = self._shake_seq(seg, rho=0.3)[min(k - seg["start"], seg["end"] - seg["start"])][3]
            amt = self.st.get("flicker_by_look", {}).get(seg.get("look"), self.st.get("exposure_flicker", 0.0))
            img = img * math.exp(fl * amt)
        for cap in self.t.get("captions", []):
            if cap["start"] <= k <= cap["end"]:
                a, sh = self._caption_layer(cap)
                img = img * (1 - sh[..., None] * 0.55) + np.array([10, 12, 40], np.float32) * sh[..., None] * 0.55
                img = img * (1 - a[..., None]) + 252 * a[..., None]
        if self.wm is not None and seg is not None and seg["type"] not in ("black",) and img.mean() > 12:
            img = img * (1 - self.wm) + 235 * self.wm
        rng = np.random.default_rng(k)
        img = img + rng.normal(0, self.st["grain"], (self.H // 2, self.W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)
        return np.clip(img, 0, 255).astype(np.uint8)


def render_manifest(manifest_path, preview=False, out=None):
    p = Path(manifest_path)
    m = json.loads((p if p.is_absolute() else ROOT / p).read_text(encoding="utf-8-sig"))
    out = out or ROOT / "output" / f"{m['id']}.mp4"
    t = json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig"))
    if t.get("style", {}).get("layered"):
        from layered_edit import LayeredEdit           # drop shots rebuilt as layers
        return LayeredEdit(m, preview).render(out)
    return XavierEdit(m, preview).render(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(render_manifest(sys.argv[1], preview="--preview" in sys.argv))
