"""Layered reconstruction of the Xavier drop shots (templates 5 and 6).

Every drop shot is rebuilt the way it was made in the editing app, as a
stack of layers, each analysed separately on the reference:

  1. BACKGROUND  another panel, recoloured with the section's accent (black ->
                 crimson / acid yellow -> pink-white), its brightness matched to
                 the reference's background layer (bgL / bgP95 per shot), with a
                 wide glow of its bright parts. It can scroll on its own
                 (parallax) when the reference background does.
  2. CHARACTER   the cutout from the shot's panel - NOT re-drawn: only the
                 navy duotone grade, with its brightness matched to the
                 reference cutout of that shot (mean / p10 / p90), plus a thin
                 rim light.
     (1 + 2 move together with the measured camera curve of the shot.)
  3. POPUPS      screen-space overlays with their own entry/exit motion:
                   strip  - a horizontal band (eyes) that whips in from a side
                   card   - an inset card that enters big + blurred and
                            settles (or grows), optional shard reveal / shard exit
  4. FLASHES     the light flares / exposure pops on top (xavier_edit.flash).

Template shots opt in with  "layers": {...}  (see templates 5/6).
"""
import math

import cv2
import numpy as np

from wekisava_edit import cutout_mask, load_rgb, luma, warp
from xavier_edit import XavierEdit


def ease_out(u, p=3.0):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** p


def cover_fit(src, W, H, cx=0.5, cy=0.42):
    h, w = src.shape[:2]
    s = max(W / w, H / h)
    nw, nh = int(math.ceil(w * s)), int(math.ceil(h * s))
    im = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x0 = int(min(max(cx * nw - W / 2, 0), nw - W))
    y0 = int(min(max(cy * nh - H / 2, 0), nh - H))
    return im[y0:y0 + H, x0:x0 + W]


class LayeredEdit(XavierEdit):
    # ------------------------------------------------------------ layers
    def _field_bg(self, seg, look, j):
        """Reference background (T5/T6 red & yellow shots): NOT a detailed
        panel - a soft, greyed-out soccer field, dimmed to the measured
        background brightness, with the section colour carried by blurry
        horizontal light lines that drift sideways, plus a glow."""
        L = seg["layers"]
        W, H = self.W, self.H
        key = ("field", seg["start"])
        if key not in self._cache:
            pool = self.m.get("field_pool") or self.m["bg_pool"]
            src = load_rgb(pool[(seg["start"] // 7) % len(pool)])
            g = luma(cover_fit(src, W, H)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), self.st.get("field_blur", 7) * W / 1080)       # not defined
            g = g / max(float(g.mean()), 1) * L["bgL"] * self.st.get("field_level", 0.75)
            grey = self.match_section(np.repeat(np.clip(g, 0, 255)[..., None], 3, -1), "duo", "duo", strength=0.0)
            sat = grey.mean(-1, keepdims=True)
            grey = sat + (grey - sat) * 0.35                                              # greyed out
            self._cache[key] = grey.astype(np.float32)
        out = self._cache[key].copy()
        acc = self.acc.get(look)
        # deep, saturated accent for the lines (the reference's crimson / acid yellow),
        # not the washed-out highlight colour
        tint = np.clip(acc["p50"] * self.st.get("line_gain", 1.45), 0, 255) if acc is not None else np.array([200, 210, 255], np.float32)
        core, glow = self._bars(seg, seg["start"] + j, thick=(6, 34))
        core = cv2.GaussianBlur(core, (0, 0), 2.5 * W / 1080)                             # soft lines
        lvl = self.st.get("line_level", 0.9)
        out = out + (glow * 1.2 * lvl)[..., None] * tint * 0.85 + (core * lvl)[..., None] * (0.88 * tint + 0.12 * 255)
        # overall glow (wide bloom of the light lines)
        out = out + cv2.GaussianBlur(core, (0, 0), 30 * W / 1080)[..., None] * tint * 0.5 * L.get("glow", 1.0)
        return out

    def _bg_layer(self, seg, look):
        L = seg["layers"]
        key = ("lbg", seg["start"])
        if key not in self._cache:
            src = load_rgb(self.m["bg"].get(seg["slot"]) or self.m["bg_pool"][seg["start"] % len(self.m["bg_pool"])])
            ov = 1.0 + abs(L.get("bg_scroll", [0, 0])[1]) * (seg["end"] - seg["start"]) / self.H + 0.1
            g = luma(cover_fit(src, int(self.W * ov), int(self.H * ov))).astype(np.float32)
            # target brightness distribution of the reference background layer
            lo, hi = np.percentile(g, [5, 99])
            x = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
            gam = math.log(max(L["bgL"], 4) / 255) / math.log(max(float(x.mean()), 1e-3))
            x = np.clip(x ** max(0.4, min(gam, 4.0)), 0, 1) * min(1.0, L["bgP95"] / 235)
            acc = self.acc.get(look)
            if acc is not None and L.get("bgSat", 0) > 60:
                # black -> accent -> pinkish white (the reference's recoloured panel)
                c1, c2 = acc["p25"] / 255, acc["p90"] / 255
                t = x[..., None]
                rgb = np.where(t < 0.55, c1 * (t / 0.55) * 1.6, c1 * 1.6 + (c2 - c1 * 1.6) * ((t - 0.55) / 0.45))
                rgb = np.clip(rgb, 0, 1) * 255
                Lc = float((rgb @ np.array([0.299, 0.587, 0.114], np.float32)).mean())
                rgb = rgb * (L["bgL"] / max(Lc, 1))                    # measured background brightness
            else:
                rgb = self.match_section(np.repeat((x * 255)[..., None], 3, -1), "duo", "duo", strength=0.6)
            # the background GLOWS: wide bloom of its bright parts, accent tinted
            br = np.clip((x - 0.55) / 0.45, 0, 1)
            glow = cv2.GaussianBlur(br, (0, 0), 24 * self.W / 1080) * L.get("glow", 1.0) * min(1.0, L["bgL"] / 90)
            tint = acc["p90"] if acc is not None else np.array([200, 210, 255], np.float32)
            rgb = rgb + glow[..., None] * tint * 0.9
            self._cache[key] = (rgb.astype(np.float32), ov)
        return self._cache[key]

    def _char_layer(self, seg, look):
        L = seg["layers"]
        key = ("lchar", seg["start"])
        if key not in self._cache:
            pl = self.placed(seg["slot"], 1.0)
            m = self.mask(seg["slot"], 1.0)
            g = luma(pl.img).astype(np.float32)
            gm = g[m > 0.5] if (m > 0.5).any() else g.ravel()
            # brightness of the reference cutout (mean, p10, p90) - a tone map
            # only, the drawing itself is untouched
            s_lo, s_hi = np.percentile(gm, [10, 90])
            t_lo, t_hi = L["charP10"], L["charP90"]
            x = (g - s_lo) / max(s_hi - s_lo, 1)
            g2 = np.clip(t_lo + x * (t_hi - t_lo), 0, 255)
            cur = float(g2[m > 0.5].mean()) if (m > 0.5).any() else float(g2.mean())
            gam = math.log(max(L["charL"], 4) / 255) / math.log(max(cur / 255, 1e-3))
            g2 = 255 * (g2 / 255) ** max(0.6, min(gam, 1.6))
            rgb = self.match_section(np.repeat(g2[..., None], 3, -1), "duo", "duo", strength=0.0)
            mm = (m > 0.5).astype(np.uint8)
            r = max(3, int(4 * self.W / 1080)) | 1
            edge = cv2.GaussianBlur((cv2.dilate(mm, np.ones((r, r), np.uint8)) - mm).astype(np.float32), (0, 0), 0.8)
            acc = self.acc.get(look)
            rim = edge[..., None] * 170 + cv2.GaussianBlur(edge, (0, 0), 7 * self.W / 1080)[..., None] * (
                acc["p90"] if acc is not None else np.array([190, 200, 255], np.float32)) * 0.9
            self._cache[key] = (rgb.astype(np.float32), m.astype(np.float32), rim.astype(np.float32))
        return self._cache[key]

    def _popup_src(self, slot, w, h, cy=0.42):
        key = ("lpop", slot, w, h, cy)
        if key not in self._cache:
            src = load_rgb(self.m["popups"][slot])
            g = luma(cover_fit(src, w, h, cy=cy)).astype(np.float32)
            self._cache[key] = self.match_section(np.repeat(g[..., None], 3, -1), "duo", "duo", strength=0.5)
        return self._cache[key]

    def _strip(self, img, p, k):
        """Eye slot (T5 733-760, measured): a band centred at y=0.50 that OPENS
        vertically like an eye - 5% of the frame tall on its first frame,
        widening smoothly to ~13% - and (as asked) closes again at the end.
        The picture inside stays fixed; only the opening grows/shrinks."""
        W, H = self.W, self.H
        cy = p.get("cy", 0.502)
        h0, h1 = p.get("open", [0.054, 0.135])
        n = max(1, p["end"] - p["start"])
        close = p.get("close", 6)
        u = (k - p["start"]) / max(1, n - close)
        h = h0 + (h1 - h0) * ease_out(min(u, 1.0), 2.0)
        if k > p["end"] - close:                                   # close: ease-in to a line
            v = (k - (p["end"] - close)) / close
            h *= (1 - v) ** 2
        if k == p["start"]:
            h *= 0.55                                              # first frame: just cracking open
        full = int(p["max_h"] * H) if p.get("max_h") else int(h1 * H)
        band = self._popup_src(p["slot"], W, full, cy=p.get("src_y", 0.42))
        hh = max(1, int(h * H))
        y0 = int(cy * H - hh / 2)
        cut = band[(full - hh) // 2:(full - hh) // 2 + hh]
        a = np.ones((hh, W), np.float32)
        a[:2] = a[-2:] = 0.6                                       # soft edges
        out = img.copy()
        ys0, ys1 = max(0, y0), min(H, y0 + hh)
        out[ys0:ys1] = out[ys0:ys1] * (1 - a[ys0 - y0:ys1 - y0, :, None]) + cut[ys0 - y0:ys1 - y0] * a[ys0 - y0:ys1 - y0, :, None]
        for yy in (ys0, ys1 - 1):                                  # thin white rims
            out[max(0, yy - 1):yy + 2] = np.maximum(out[max(0, yy - 1):yy + 2], 230)
        return out

    def _card(self, img, p, k):
        W, H = self.W, self.H
        u = (k - p["start"]) / max(1, p.get("settle", 12))
        s0, s1 = p.get("scale", [0.95, 0.58])
        s = s0 + (s1 - s0) * ease_out(u, 2.6)
        cw, ch = int(W * s), int(H * s * p.get("aspect", 1.0))
        card = self._popup_src(p["slot"], int(W * max(s0, s1)), int(H * max(s0, s1) * p.get("aspect", 1.0)))
        card = cv2.resize(card, (cw, ch), interpolation=cv2.INTER_AREA)
        bw = max(2, int(0.008 * W))
        card = cv2.copyMakeBorder(card, bw, bw, bw, bw, cv2.BORDER_CONSTANT, value=(12, 12, 20))
        vel = abs(s - (s0 + (s1 - s0) * ease_out((k - 1 - p["start"]) / max(1, p.get("settle", 12)), 2.6))) * W
        if vel > 2:
            card = cv2.GaussianBlur(card, (0, 0), min(8.0, vel * 0.25))
        cx, cy = p.get("at", [0.5, 0.47])
        x0, y0 = int(cx * W - card.shape[1] / 2), int(cy * H - card.shape[0] / 2)
        out = img.copy()
        if p.get("dim_bg"):
            out *= p["dim_bg"]
        # drop shadow
        sh = np.zeros((H, W), np.float32)
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(W, x0 + card.shape[1]), min(H, y0 + card.shape[0])
        if xs1 <= xs0 or ys1 <= ys0:
            return out
        sh[ys0:ys1, xs0:xs1] = 1
        sh = cv2.GaussianBlur(np.roll(np.roll(sh, int(0.012 * H), 0), int(0.008 * W), 1), (0, 0), 14 * W / 1080)
        out = out * (1 - 0.6 * sh[..., None])
        crop = card[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        a = np.ones(crop.shape[:2], np.float32)
        if p.get("reveal") == "shatter":
            # card appears through growing shattered shards with white edges
            r = ease_out((k - p["start"]) / max(1, p.get("reveal_frames", 6)), 1.6)
            key = ("shard", p["start"], crop.shape)
            if key not in self._cache:
                rng = np.random.default_rng(p["start"])
                n = cv2.resize(rng.random((9, 7)).astype(np.float32), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]].astype(np.float32)
                d = np.hypot((xx / crop.shape[1] - 0.5), (yy / crop.shape[0] - 0.4))
                self._cache[key] = d * 1.6 + n * 0.5
            field = self._cache[key]
            a = (field < r * 1.6).astype(np.float32)
            edge = cv2.dilate(a, np.ones((5, 5), np.uint8)) - a
            crop = crop * (1 - edge[..., None]) + 255 * edge[..., None]
            a = np.maximum(a, edge)
            if r < 0.6:
                crop = crop * 0.4 + 255 * 0.6
        out[ys0:ys1, xs0:xs1] = out[ys0:ys1, xs0:xs1] * (1 - a[..., None]) + crop * a[..., None]
        if p.get("exit") == "shards" and k > p["end"] - 5:
            v = (k - (p["end"] - 5)) / 5
            lay = np.zeros((H, W), np.float32)
            for (ax, ay), (bx, by), (qx, qy) in (((0, 0), (0.45, 0), (0, 0.55)), ((1, 1), (0.55, 1), (1, 0.45)),
                                                 ((1, 0), (1, 0.35), (0.65, 0)), ((0, 1), (0.35, 1), (0, 0.65))):
                pts = np.array([[ax * W, ay * H], [ax * W + (bx - ax) * W * v, ay * H + (by - ay) * H * v],
                                [ax * W + (qx - ax) * W * v, ay * H + (qy - ay) * H * v]], np.int32)
                cv2.fillConvexPoly(lay, pts, 1.0)
            out = out * (1 - lay[..., None]) + 245 * lay[..., None]
        return out

    def _layer_xf(self, img, pth, j, extra=1.0, anchor=(0.5, 0.45)):
        """Apply a measured layer path (cumulative tx/ty in px@1080, scale)."""
        if not pth:
            return img
        jj = min(j, len(pth["s"]) - 1)
        s = pth["s"][jj] * extra
        W, H = self.W, self.H
        dx, dy = pth["tx"][jj] - pth["tx"][-1] / 2, pth["ty"][jj] - pth["ty"][-1] / 2   # centred travel
        M = np.float32([[s, 0, (1 - s) * anchor[0] * W + dx * W / 1080], [0, s, (1 - s) * anchor[1] * H + dy * W / 1080]])
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _fit_scale(self, seg, m):
        """Scale the cutout so it fills the frame like the reference's
        character layer (SAM 2 coverage)."""
        key = ("fitscale", seg["start"])
        if key not in self._cache:
            want = seg["layers"].get("char_cov")
            have = float((m > 0.5).mean())
            k = 1.0 if not want or have < 0.01 else float(np.clip(math.sqrt(want / have), 0.85, 2.3))
            self._cache[key] = k
        return self._cache[key]

    # ------------------------------------------------------------ frame
    def shot_frame(self, seg, k, apply_motion=True):
        if not seg.get("layers"):
            return super().shot_frame(seg, k, apply_motion)
        look = seg["look"]
        L = seg["layers"]
        char, m, rim = self._char_layer(seg, look)
        W, H = self.W, self.H
        j = k - seg["start"]
        if self.st.get("layer_paths") and L.get("char_path"):
            # every layer moves on its own, exactly as measured on the reference
            base = self._field_bg(seg, look, j) if (self.st.get("field_bg") and look in ("red", "yellow")) \
                else self._bg_layer(seg, look)[0][:H, :W]
            fs = self._fit_scale(seg, m)
            base = self._layer_xf(base, L["bg_path"], j, 1.04)
            ch = self._layer_xf(char + rim, L["char_path"], j, fs)
            cm = self._layer_xf(m, L["char_path"], j, fs)
            comp = base * (1 - cm[..., None]) + ch * cm[..., None]
            for p in L.get("popups", []):
                if p["start"] <= k <= p["end"]:
                    comp = self._strip(comp, p, k) if p["type"] == "strip" else self._card(comp, p, k)
            return np.clip(self.light(comp, look), 0, 255)
        if self.st.get("field_bg") and look in ("red", "yellow"):
            base = self._field_bg(seg, look, j)
            comp = base * (1 - m[..., None]) + char * m[..., None] + rim
            return self._finish_layers(seg, look, k, j, comp, apply_motion)
        bg, ov = self._bg_layer(seg, look)
        # background: its own scroll (parallax), cropped from the oversized layer
        sx, sy = L.get("bg_scroll", [0, 0])
        bh, bw = bg.shape[:2]
        x0 = int(np.clip((bw - W) / 2 - sx * j * W / 1080, 0, bw - W))
        y0 = int(np.clip((bh - H) / 2 - sy * j * H / 1350, 0, bh - H))
        base = bg[y0:y0 + H, x0:x0 + W]
        comp = base * (1 - m[..., None]) + char * m[..., None] + rim
        return self._finish_layers(seg, look, k, j, comp, apply_motion)

    def _finish_layers(self, seg, look, k, j, comp, apply_motion):
        L = seg["layers"]
        if apply_motion:
            sc, an, rot, dx, dy = self._mo(seg, k)
            comp = warp(comp, sc, an, rot, dx, dy)
            c = seg["motion"].get("curve")
            if j > 0 and c:
                jj = min(j, len(c) - 1)
                from xavier_edit import zoom_blur
                comp = zoom_blur(comp, (c[jj - 1] / c[jj]) ** self.st.get("zoom_blur", 1.3), an, n=7)
        for p in L.get("popups", []):
            if p["start"] <= k <= p["end"]:
                comp = self._strip(comp, p, k) if p["type"] == "strip" else self._card(comp, p, k)
        return np.clip(self.light(comp, look), 0, 255)
