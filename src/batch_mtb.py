"""'Meant To Be' (template 24) as a Blue Lock edit - Isagi, Rin, Sae, Nagi.

Shot-for-shot from a frame-by-frame read of the reference (10 fps sheets +
per-frame difference / saturation). Medium follows the reference: colour still
panels where it uses colour art, the user's B&W manga where it uses manga,
colour anime clips where it uses footage, cutouts where it uses cutouts.

  0    - 12.6   per character: small whole-figure cutout centred on a starfield, four colour panels
                slide in behind it (reference order + timing), lyric words, grid dims out
  12.6 - 18.83  fast B&W manga per character
  18.83- 31.37  per character: four colour eye strips drop in over B&W manga, then a colour image
                enters in halves (left half from the left, right half from the right), then it
                darkens / washes out into the next character's manga
  31.37- 36.8   animated manga clips of each character, one per beat, then a redrawn Japan flag (no reference footage)
  36.8 - 43.83  washed manga, mixed-cast eye strips, BLUE LOCK title, washed manga, eye strips
                again, white / colour flicker
  43.83- 56.3   animated manga panels (twixtor clips) per character (blue / teal / red / purple grade), white flashes between
  56.3 -        the standard DUDS outro card + fade (same as every other edit)

    python src/batch_mtb.py        -> edits/nx_meant_to_be.json
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw  # noqa: E402

BL = bw.BL
CAST = ["Isagi", "Rin", "Sae", "Nagi"]
NAME = {"Isagi": ("ISAGI YOICHI", [70, 110, 255]), "Rin": ("ITOSHI RIN", [40, 215, 120]),
        "Sae": ("ITOSHI SAE", [215, 25, 40]), "Nagi": ("NAGI SEISHIRO", [245, 245, 245])}


def name_box(who):
    """the character's name in their colour (shown over their fast manga), no box."""
    txt, col = NAME[who]
    return {"text": txt, "y": 0.82, "font": FONT, "size": 0.07, "color": col, "stroke": 0.003, "stroke_color": [0, 0, 0],
            "glow": 0.5, "glow_color": col}
TINT = {"Isagi": [60, 110, 255], "Rin": [40, 210, 170], "Sae": [225, 35, 60], "Nagi": [150, 90, 230]}
FONT = "C:/Windows/Fonts/impact.ttf"
BW = {"bw": True, "contrast": 1.4}
DIM = {"bw": True, "contrast": 0.8, "bright": 0.42}
PALE = {"bw": True, "contrast": 0.45, "bright": 1.7}
_next = {}
_cache = {}


def _rel(p):
    return str(Path(p).relative_to(ROOT)).replace("\\", "/")


def images(who, kind):
    """the user's art: 'color' / 'bw' (split by saturation)."""
    key = ("img", who, kind)
    if key not in _cache:
        ps = []
        for q in sorted((ROOT / BL / bw.FOLDER[who] / "images").glob("*.jpg")):
            sat = cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2HSV)[..., 1].mean()
            if (sat >= 25) == (kind == "color"):
                ps.append(_rel(q))
        _cache[key] = ps
    return _cache[key]


def eye_strips(who):
    """colour eye strips, best first (src/eye_strips.py); falls back to strips cut from the images."""
    key = ("eye", who)
    if key not in _cache:
        idx = ROOT / BL / bw.FOLDER[who] / "eyes/index.json"
        rows = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else []
        ps = [r["file"] for r in rows if r["colour"] >= 30]
        if len(ps) < 4:
            ps += [_rel(q) for q in sorted((ROOT / BL / bw.FOLDER[who] / "images/eyes").glob("*.jpg"))]
        _cache[key] = ps
    return _cache[key]


def colour_clips(who):
    """the animated manga panels (Musstq twixtor clips) used in the earlier edits: the verified pool,
    original clip folders only, normal framing, bright enough."""
    key = ("clip", who)
    if key not in _cache:
        d = json.loads((ROOT / BL / bw.FOLDER[who] / "clips_pool.json").read_text(encoding="utf-8-sig"))["clips"]
        _cache[key] = [(c["path"], c["frames"]) for c in d
                       if "/clips_x" not in c["path"] and not c.get("framing") and c["brightness"] > 45]
    return _cache[key]


def nxt(who, kind):
    """the next unused panel of a kind ('color', 'bw', 'eye', 'clip') for a character."""
    if kind == "clip":
        ps = colour_clips(who)
        if not ps:
            return nxt(who, "color")
        i = _next.get((who, kind), 0); _next[(who, kind)] = i + 1
        p, n = ps[i % len(ps)]
        return {"clip": p, "start": min(6, n // 4)}
    ps = eye_strips(who) if kind == "eye" else images(who, kind)
    i = _next.get((who, kind), 0); _next[(who, kind)] = i + 1
    return {"clip": ps[i % len(ps)], "start": 0}


def cutout_png(who):
    return f"{BL}{bw.FOLDER[who]}/cutouts/main.png"


def word(text, a, b, col, y=0.56, size=0.075):
    return {"text": text, "a": round(a, 3), "b": round(b, 3), "y": y, "font": FONT, "size": size, "color": col,
            "stroke": 0.004, "stroke_color": [0, 0, 0], "glow": 0.6, "glow_color": col}


def single(a, b, spec, grade, fx=(), **k):
    z = k.pop("zoom", [1.08, 1.0])
    return dict(spec, a=round(a, 3), b=round(b, 3), layout="single", zoom=z, grade=dict(grade), fx=list(fx), **k)


def run(times, end, make):
    """consecutive shots at `times`, the last running to `end`."""
    return [make(i, a, b) for i, (a, b) in enumerate(zip(times, times[1:] + [end]))]


# ---------------------------------------------------------------- 0 - 12.6
def grids():
    quads = {"TL": ((0, 0, .5, .5), "top"), "TR": ((.5, 0, 1, .5), "right"), "BL": ((0, .5, .5, 1), "left"),
             "BR": ((.5, .5, 1, 1), "bottom")}
    spans = [(0.00, 3.20, [("TL", .17), ("BL", .47), ("BR", .83), ("TR", 1.20)]),
             (3.20, 6.30, [("TL", .10), ("TR", .33), ("BL", .77), ("BR", 1.17)]),
             (6.30, 9.45, [("BL", .13), ("BR", .37), ("TR", .77), ("TL", 1.17)]),
             (9.45, 12.50, [("BL", .08), ("TL", .35), ("TR", .75), ("BR", 1.12)])]
    blue, red, white, green = [70, 110, 255], [205, 18, 30], [245, 245, 245], [40, 215, 120]
    purple = white                                                # "WITH" is white too (user)
    words = [[("1", 1.25, blue), ("2", 1.80, blue), ("3", 2.50, blue)],
             [("MEANT", 4.83, green), ("TO", 5.20, green), ("BE", 5.70, green)],
             [("THIS", 7.93, red), ("A", 8.33, red), ("TRICK", 8.73, red)],
             [("I", 11.00, white), ("WILL", 11.45, white), ("STICK", 11.85, white), ("WITH", 12.30, purple)]]
    segs = []
    for (a, b, pops), who, ws in zip(spans, CAST, words):
        tiles = []
        for q, at in pops:
            rect, frm = quads[q]
            tiles.append(dict(nxt(who, "color"), rect=list(rect), at=round(max(0.0, at - 0.07), 3), dur=0.13, still=True, **{"from": frm}))
        ends = [w[1] for w in ws[1:]] + [b - 0.1]
        caps = [word(t, wa - a, we - a, col) for (t, wa, col), we in zip(ws, ends)]
        segs.append({"a": a, "b": b, "layout": "tiles", "bg": [0, 0, 0], "stars": True, "tiles": tiles, "tile_grade": {"contrast": 1.1},
                     "captions": caps, "dark_last": 0.22,
                     "cutout": {"png": cutout_png(who), "x": 0.5, "y": 0.52, "h": 0.34, "max_w": 0.5, "zoom": [1.0, 1.0],
                                "drift": [0, 0], "glow": 0.25, "pop": 0.0001}})
    segs.append({"a": 12.50, "b": 12.60, "layout": "card", "bg": [0, 0, 0], "stars": True, "fx": ["glitch_in", "rgb_in"],
                 "captions": [word("WITH", 0, 0.1, purple)]})
    return segs


# ---------------------------------------------------------------- 12.6 - 18.83
FAST = {"Isagi": [12.60, 12.73, 12.87, 13.03, 13.17, 13.30, 13.40, 13.53, 13.67, 13.77, 13.87, 14.00],
        "Rin": [14.07, 14.23, 14.37, 14.57, 14.73, 14.93, 15.00, 15.13, 15.33, 15.47, 15.60, 15.73],
        "Sae": [15.90, 16.07, 16.20, 16.37, 16.53, 16.67, 16.83, 17.00, 17.17],
        "Nagi": [17.40, 17.53, 17.67, 17.80, 17.93, 18.07, 18.20, 18.33, 18.47, 18.60, 18.73]}


def fast_manga():
    """super-fast B&W manga of each character (the earlier, approved cut list), with the measured reference cuts
    merged in; no panel is ever held longer than 0.17 s."""
    R = json.loads((ROOT / "references/template_24/cuts.json").read_text())
    ends = {"Isagi": 14.07, "Rin": 15.90, "Sae": 17.40, "Nagi": 18.83}
    segs, a0 = [], 12.60
    for who in CAST:
        cuts = sorted(set(FAST[who]) | {r for r in R if a0 + 0.03 < r < ends[who] - 0.03})
        merged = [cuts[0]]
        for c in cuts[1:]:
            if c - merged[-1] >= 0.05:
                merged.append(c)
        full = [merged[0]]
        for y in merged[1:] + [ends[who]]:
            while y - full[-1] > 0.17:
                full.append(round(full[-1] + 0.13, 3))
            if y < ends[who]:
                full.append(y)
        panels = [nxt(who, "bw") for _ in range(5)]
        segs += run(full, ends[who],
                    lambda i, a, b, p=panels, w=who: single(a, b, p[i % 5], BW, ["mblur", "shake"] if i % 2 else ["radial_in", "zoom_punch"],
                                                            mblur=0.05, mblur_angle=-15 if i % 2 else 20, zoom=[1.12, 1.0],
                                                            captions=[name_box(w)]))
        a0 = ends[who]
    return segs


# ---------------------------------------------------------------- 18.83 - 31.37
# (who, window start, strip landings, halves enter, hold until, after-shot, after end, next-manga cuts, manga end)
EYES = [("Isagi", 18.83, [18.83, 19.20, 19.57, 19.93], 20.30, 21.23, "dark", 21.73, [21.73, 21.87, 22.00], 22.17),
        ("Rin", 22.17, [22.17, 22.37, 22.73, 23.10], 23.47, 24.37, "pale", 24.97, [25.00, 25.13, 25.23], 25.40),
        ("Sae", 25.40, [25.40, 25.57, 25.83, 26.23], 26.57, 27.43, "pale", 28.17, [], 28.17),
        ("Nagi", 28.17, [28.20, 28.53, 28.93, 29.33], 29.67, 30.60, "pale", 31.07, [31.07, 31.27], 31.37)]


def strips_seg(a, b, lands, strip_specs, under_who, under_end, halves=None, bright_first=0.4):
    """four eye strips drop in from the top with impact (arrive smeared + zoomed, settle sharp ~0.25 s, like the
    reference); optional colour image halves: left half slides in from the left, the right half from the right
    ~0.27 s later, both landing with the same impact."""
    tiles = [dict(sp, rect=[0, i / 4, 1, (i + 1) / 4], at=round(max(0.0, t - a - 0.08), 3), dur=0.08, smear_amt=0.12,
                  impact=0.25, impact_zoom=0.16, impact_blur=0.06, grade={"contrast": 1.15, "sat": 1.15}, **{"from": "top"})
             for i, (t, sp) in enumerate(zip(lands, strip_specs))]
    if halves:
        img, at = halves
        for side, frm, rect, lag in (("L", "left", [0, 0, .5, 1], 0.0), ("R", "right", [.5, 0, 1, 1], 0.27)):
            tiles.append(dict(img, rect=rect, half=side, at=round(at - a + lag, 3), dur=0.17, smear_amt=0.2, impact=0.2,
                              impact_zoom=0.1, impact_blur=0.08, grade={"contrast": 1.15, "sat": 1.15, "glow": 0.25}, **{"from": frm}))
    under, t, k = [], a, 0
    while t < under_end - 0.02:                                      # manga keeps cutting under the strips
        step = 0.13 if k % 2 else 0.17
        under.append(dict(nxt(under_who, "bw"), a=round(t - a, 3), b=round(min(t + step, under_end) - a, 3),
                          grade=dict(BW if t - a < bright_first else DIM), mblur=0.04, angle=20 if k % 2 else -15))
        t += step; k += 1
    return {"a": round(a, 3), "b": round(b, 3), "layout": "tiles", "bg": [0, 0, 0], "tiles": tiles, "under": under, "fx": ["glitch_in"]}


HALVES = {"Isagi": BL + "Isagi Yoichi/images/isagi_10.jpg"}       # user-picked replacements for the split image


def eyes():
    """per character: eyes drop in, the colour halves join, then super-fast manga of that character runs
    until the next character's eyes."""
    R = json.loads((ROOT / "references/template_24/cuts.json").read_text())
    segs = []
    for n, (who, a0, lands, halves_at, *_rest) in enumerate(EYES):
        nxt_start = EYES[n + 1][1] if n + 1 < len(EYES) else 31.37
        joined = round(halves_at + 0.27 + 0.17 + 0.2, 3)            # right half landed + impact settled
        img = {"clip": HALVES[who], "start": 0} if who in HALVES else nxt(who, "color")
        segs.append(strips_seg(a0, joined, lands, [nxt(who, "eye") for _ in range(4)], who, halves_at, (img, halves_at),
                               bright_first=0.45 if who == "Isagi" else 0.0))
        cuts = [joined] + [r for r in R if joined + 0.06 < r < nxt_start - 0.05]
        full = [cuts[0]]                                             # fill to the reference's fast cadence (~0.1 s)
        for x, y in zip(cuts, cuts[1:] + [nxt_start]):
            while y - full[-1] > 0.16:
                full.append(round(full[-1] + 0.1, 3))
            if y < nxt_start:
                full.append(y)
        segs += run(full, nxt_start, lambda i, a, b, w=who: single(a, b, nxt(w, "bw"), BW,
                                                                  ["white_in", "mblur"] if i == 0 else (["mblur", "shake"] if i % 2 else ["zoom_punch", "mblur"]),
                                                                  mblur=0.05, mblur_angle=-15 if i % 2 else 20, zoom=[1.12, 1.0]))
    return segs


# ---------------------------------------------------------------- 31.37 - 36.8 the reference's own shots
def reference_part():
    """31.37-36.8 (was the reference's own human eyes + Japan flag): animated manga clips of each character,
    one per beat of the song, a harder hit on each downbeat."""
    beats = [31.37, 31.72, 32.12, 32.50, 32.90, 33.28, 33.68, 34.06, 34.46, 34.84, 35.22, 35.62]
    downs = {31.37, 32.90, 34.46}
    g = {"contrast": 1.2, "sat": 1.1, "glow": 0.3}
    segs = run(beats, 35.90, lambda i, a, b: single(a, b, nxt(CAST[i % 4], "clip"), g,
                                                    ["white_in", "zoom_punch", "shake"] if a in downs else ["zoom_punch", "rgb_in"],
                                                    mblur=0, zoom=[1.1, 1.0], shake_amt=0.015))
    # Japan flag, redrawn (no footage from the reference), with the reference's transition: the right half of the
    # flag slides in from the right with a smear and lands with impact (35.90), the current clip keeps playing on the
    # left, then the left half slides in from the left (36.27) and the two join; hold to 36.80
    flag = {"clip": "assets/japan_flag.png", "start": 0}
    under = [dict(nxt("Nagi", "clip"), a=0.0, b=0.90, grade=dict(g))]
    tiles = [dict(flag, rect=[.5, 0, 1, 1], half="R", at=0.0, dur=0.13, smear_amt=0.2, impact=0.18, impact_zoom=0.08,
                  impact_blur=0.06, **{"from": "right"}),
             dict(flag, rect=[0, 0, .5, 1], half="L", at=0.37, dur=0.13, smear_amt=0.2, impact=0.18, impact_zoom=0.08,
                  impact_blur=0.06, **{"from": "left"})]
    segs[-1]["b"] = 35.90
    segs.append({"a": 35.90, "b": 36.80, "layout": "tiles", "bg": [0, 0, 0], "tiles": tiles, "under": under,
                 "fx": ["glitch_in"], "hits": [36.40], "hit_zoom": 0.04})
    return segs


# ---------------------------------------------------------------- 36.8 - 43.83
def middle():
    segs, mix = [], lambda i: CAST[i % 4]
    pink = {"bw": True, "contrast": 0.45, "bright": 1.65, "tint": [235, 170, 190], "tint_amt": 0.35}
    segs += run([36.80, 36.87, 36.97, 37.07], 37.17, lambda i, a, b: single(a, b, nxt(mix(i), "bw"), pink, ["mblur"], mblur=0.07))
    segs += run([37.17, 37.27, 37.37, 37.47], 37.57, lambda i, a, b: single(a, b, nxt(mix(i + 1), "bw"), PALE, ["mblur"], mblur=0.06))
    segs.append({"a": 37.57, "b": 37.62, "layout": "card", "bg": [40, 40, 42]})
    segs.append(strips_seg(37.62, 39.03, [37.62, 38.22, 38.33, 38.47], [nxt(w, "eye") for w in CAST], "Rin", 38.30, bright_first=0.6))
    segs.append({"a": 39.03, "b": 40.17, "layout": "card", "bg": [250, 250, 250],
                 "captions": [{"text": "BLUE LOCK", "y": 0.5, "font": FONT, "size": 0.17, "color": [25, 60, 200], "stroke": 0.007,
                               "stroke_color": [10, 10, 25], "glow": 0.5, "glow_color": [60, 110, 255], "slide": ["left", 0.18],
                               "blur_out": 0.22}]})
    quick = [round(40.17 + 0.08 * i, 3) for i in range(10)]          # super-quick manga of all four, no white flashes
    segs += run(quick, 40.97, lambda i, a, b: dict(single(a, b, nxt(mix(i), "bw"), BW, ["mblur", "shake"] if i % 2 else ["zoom_punch"],
                                                           mblur=0.05, mblur_angle=-15 if i % 2 else 20, zoom=[1.12, 1.0]), no_flash=True))
    st = strips_seg(40.97, 43.37, [40.97, 41.00, 41.40, 41.73], [nxt(w, "eye") for w in CAST], "Sae", 41.80, bright_first=0.0)
    st.update(steps=[[42.23, 1.03, 0.14], [42.64, 1.18, 0.09]],   # one set of eyes, two zoom jumps on the beats (measured)
              hits=[42.64], hit_zoom=0.0)                          # + a hard shake on the big jump
    segs.append(st)
    segs += run([43.37, 43.50, 43.53, 43.63, 43.67, 43.70, 43.73], 43.83,
                lambda i, a, b: single(a, b, nxt(mix(i), "bw"), PALE, ["mblur"], mblur=0.05) if i % 2 == 0 else
                single(a, b, nxt(mix(i), "color"), {"tint": [220, 30, 50], "tint_amt": 0.6, "contrast": 1.25}, ["rgb_in"]))
    return segs


# ---------------------------------------------------------------- 43.83 - 56.3 colour clips
CLIPS = {"Isagi": ([43.83, 44.60, 45.03, 45.13, 45.23, 45.37, 45.73], 46.17),
         "Rin": ([46.17, 46.27, 46.37, 46.50, 46.57, 46.67, 46.93, 47.33, 47.40, 47.70, 48.17, 48.27], 48.47),
         "Sae": ([48.57, 48.67, 48.90, 49.23, 49.33, 49.43, 49.67, 49.73, 49.90, 50.07, 50.20, 50.43, 50.83, 51.27, 51.37, 51.47,
                  51.57, 51.87, 51.97, 52.43, 52.50, 52.57, 52.63, 52.83, 52.93], 53.17),
         "Nagi": ([53.33, 53.57, 53.93, 54.33, 54.43, 54.57, 55.13, 55.47, 55.57, 55.67, 55.77, 55.87, 55.97, 56.07, 56.13, 56.23], 56.30)}
MANGA_IN_CLIPS = {"Sae": (48.90, 50.07), "Nagi": (53.33, 54.33)}   # the reference drops tinted manga in here


def end_cutouts(who):
    key = ("endcut", who)
    if key not in _cache:
        _cache[key] = json.loads((ROOT / "edits/nx_meant_to_be_cutouts.json").read_text())[who]
    return _cache[key]


def cutout_over_manga(a, b, who, k, enter):
    """the reference's main device in the ending: a character cutout slides into frame (smeared, then a slow push)
    while B&W manga of the same character cuts every ~0.08 s behind it, washed with the section colour."""
    col = TINT[who]
    under, t, n = [], a, 0
    while t < b - 0.02:
        under.append(dict(nxt(who, "bw"), a=round(t - a, 3), b=round(min(t + 0.08, b) - a, 3),
                          grade={"bw": True, "contrast": 1.3, "bright": 1.05, "tint": col, "tint_amt": 0.28},
                          mblur=0, angle=20 if n % 2 else -15))
        t += 0.08; n += 1
    cuts = end_cutouts(who)
    return {"a": round(a, 3), "b": round(b, 3), "layout": "tiles", "bg": [0, 0, 0], "tiles": [], "under": under,
            "fx": ["shake"], "shake_amt": 0.01,
            "_cutout_removed": {"png": cuts[k % len(cuts)], "x": 0.5, "y": 0.56, "h": 0.64, "max_w": 0.82, "zoom": [1.0, 1.1], "drift": [0, 0],
                       "glow": 0.35, "pop": 0.0001, "enter": enter, "enter_dur": 0.13,
                       "grade": {"tint": col, "tint_amt": 0.4, "contrast": 1.2, "sat": 1.1, "glow": 0.3}}}


# per character: the reference ending as phases - ("flash", a) colour flash, ("clips", a, b, cut times) tinted clips,
# ("cut", a, b) cutouts sliding in over fast B&W manga - all at the reference's measured times
# ending, driven by the song (measured on music/template_24_song.wav): the bass "doo" on every beat gets a longer
# animated-manga shot cut exactly on it; the hi-hat rolls (where the music speeds up) get the super-fast B&W manga.
SECTIONS = [("Isagi", 43.83, 46.17), ("Rin", 46.17, 49.57), ("Sae", 49.67, 53.17), ("Nagi", 53.33, 56.30)]
DOOS = [43.91, 44.28, 44.69, 45.07, 45.47, 45.85, 46.25, 46.63, 47.03, 47.41, 47.81, 48.19, 48.62, 48.98, 49.37, 49.76,
        50.15, 50.53, 50.92, 51.31, 51.73, 52.09, 52.52, 52.88, 53.27, 53.66, 54.06, 54.44, 54.84, 55.22, 55.62]
ROLLS = [(45.04, 45.27), (48.18, 48.42), (49.33, 49.57), (52.07, 52.31), (55.07, 56.30)]
SAE_FAST = [(50.53, 50.92), (51.73, 52.52), (52.88, 53.17)]          # extra speed manga in Sae's part (user, on version A)


VARIANT = "A"
# A  one clip per doo, super-fast manga on the rolls
# B  longer shots: one clip every two doos, super-fast manga on the rolls
# C  energetic: clips on every hi-hat (half-beat), white flash on each doo, super-fast manga on the rolls
# D  one clip per doo; on the rolls the current clip stutters (repeat + zoom punches) instead of manga
# E  one shot per doo alternating colour clip / held B&W manga panel, super-fast manga on the rolls, harder shakes


def clips_part():
    v = VARIANT
    segs = []
    fxs = [["zoom_punch", "rgb_in"], ["shake", "glitch_in"], ["zoom_punch", "shake"], ["shake"]]
    for who, a0, b0 in SECTIONS:
        col = TINT[who]
        g = {"tint": col, "tint_amt": 0.5, "contrast": 1.25, "sat": 1.15, "glow": 0.45}
        segs.append({"a": a0, "b": round(a0 + 0.067, 3), "layout": "card", "bg": [int(c * 0.9) for c in col], "fx": ["glitch_in"]})
        rolls = ROLLS + (SAE_FAST if who == "Sae" and v == "A" else [])
        doos = [d for d in DOOS if a0 + 0.1 < d < b0 - 0.1 and not any(r0 - 0.05 <= d < r1 + 0.05 for r0, r1 in rolls)]
        if v == "B":
            doos = doos[::2]
        if v == "C":
            doos = sorted(set(doos) | {round((x + y) / 2, 3) for x, y in zip(doos, doos[1:]) if y - x < 0.5})
        marks = sorted({round(a0 + 0.067, 3)} | set(doos) | {r[0] for r in rolls if a0 < r[0] < b0}
                       | {r[1] for r in rolls if a0 < r[1] < b0 - 0.1})
        i, last = 0, None
        for x, y in zip(marks, marks[1:] + [b0]):
            roll = any(r0 - 0.01 <= x < r1 for r0, r1 in rolls)
            if roll and v == "D" and last:                            # stutter the last clip on the roll
                t, k = x, 0
                while t < y - 0.02:
                    segs.append(single(t, min(t + 0.07, y), last, g, ["zoom_punch", "shake"], mblur=0, zoom=[1.12 + 0.03 * (k % 3), 1.0]))
                    t += 0.07; k += 1
            elif roll:
                segs.append(cutout_over_manga(round(x, 3), round(y, 3), who, 0, "right"))
            elif v == "E" and i % 2 == 1:
                segs.append(single(x, y, nxt(who, "bw"), BW, ["zoom_punch", "shake"], mblur=0, zoom=[1.1, 1.0], shake_amt=0.018)); i += 1
            else:
                last = nxt(who, "clip")
                sh = single(x, y, last, g, fxs[i % 4], mblur=0, zoom=[1.1, 1.0])
                if v == "C" and x in DOOS:
                    sh["fx"] = sh["fx"] + ["white_in"]
                if v == "E":
                    sh["fx"] = list(dict.fromkeys(sh["fx"] + ["shake"])); sh["shake_amt"] = 0.018
                segs.append(sh); i += 1
    segs.append({"a": 49.57, "b": 49.67, "layout": "card", "bg": [250, 250, 250]})
    segs.append({"a": 53.17, "b": 53.33, "layout": "card", "bg": [250, 245, 255]})
    return segs


# ---------------------------------------------------------------- 56.3 - 58.07 sign-off
def sign_off():
    big = {"text": "DUDS", "y": 0.5, "font": FONT, "size": 0.2, "color": [255, 255, 255], "stroke": 0.0, "glow": 1.4, "glow_color": [255, 255, 255]}
    small = dict(big, size=0.06, glow=0.6)
    return [{"a": 56.30, "b": 56.40, "layout": "card", "bg": [0, 0, 0], "captions": [big]},
            {"a": 56.40, "b": 56.47, "layout": "card", "bg": [0, 0, 0], "captions": [dict(big, blur_out=0.07)]},
            {"a": 56.47, "b": 56.57, "layout": "card", "bg": [0, 0, 0], "captions": [big]},
            {"a": 56.57, "b": 56.80, "layout": "card", "bg": [0, 0, 0], "captions": [small], "fx": ["glitch_in"]},
            {"a": 56.80, "b": 57.05, "layout": "card", "bg": [0, 0, 0]},
            {"a": 57.05, "b": 57.95, "layout": "card", "bg": [0, 0, 0], "captions": [dict(small, size=0.035, glow=0.4, fade_in=0.2)],
             "dark_last": 0.4},
            {"a": 57.95, "b": 58.07, "layout": "card", "bg": [0, 0, 0]}]


def snap(segs, tol=0.05, end=56.30):
    """move each segment start onto the nearest measured reference cut (references/template_24/cuts.json),
    never collapsing a one-frame flash; the previous segment then ends there."""
    R = json.loads((ROOT / "references/template_24/cuts.json").read_text())
    for i, s in enumerate(segs):
        if i == 0 or s.get("layout") == "video" or segs[i - 1].get("layout") == "video":
            continue
        c = min(R, key=lambda r: abs(r - s["a"]))
        if 0 < abs(c - s["a"]) <= tol and segs[i - 1]["a"] + 0.025 < c < s["b"] - 0.025:
            s["a"] = c
    for x, y in zip(segs, segs[1:]):
        x["b"] = y["a"]
    segs[-1]["b"] = end
    return segs


def flashes_shakes(segs):
    """the reference's white flashes and strong shakes (references/template_24/flash_shake.json, measured from
    frame luminance spikes / large frame-to-frame displacement), applied to whatever shot is on screen then."""
    fs = json.loads((ROOT / "references/template_24/flash_shake.json").read_text())
    for a, b in fs["ref_flash"]:
        a, b = a - 0.02, max(b, a + 0.034) + 0.015
        long_ = b - a > 0.25
        for s in segs:
            if s.get("layout") == "video" or s.get("no_flash") or s["b"] <= a or s["a"] >= b:
                continue
            if long_:                                              # washed-out white stretch: pulse white on each cut
                s.setdefault("flashes", []).append([round(max(a, s["a"]), 3), round(min(b, s["a"] + 0.05), 3), 0.7])
                s.setdefault("flashes", []).append([round(max(a, s["a"] + 0.05), 3), round(min(b, s["b"]), 3), 0.22])
            else:
                s.setdefault("flashes", []).append([round(a, 3), round(b, 3), 0.85])
    for a, b in fs["ref_shake"]:
        for s in segs:
            if s.get("layout") != "video" and s["a"] < b + 0.1 and s["b"] > a - 0.1:
                s["fx"] = list(dict.fromkeys(s.get("fx", []) + ["shake"])); s["shake_amt"] = 0.022
    return segs


def build(variant=None):
    global VARIANT
    if variant:
        VARIANT = variant
    segs = grids() + fast_manga() + eyes() + reference_part() + middle() + clips_part()
    segs.sort(key=lambda s: s["a"])
    segs = flashes_shakes(snap(segs))
    for x in segs:                                                # no blur at all in the ending (user)
        if x["a"] >= 43.83:
            x["no_blur"] = True
            x["fx"] = [f for f in x.get("fx", []) if f not in ("mblur", "radial_in", "end_blur")]
    m = {"id": "nx_meant_to_be" + (f"_end{variant}" if variant else ""), "title": "Meant To Be | Blue Lock", "caption": "it was meant to be...", "watermark": "DUDS",
         "end_text": "DUDS", "size": [1080, 1350], "fps": 60,
         "description": "it was meant to be...\n\nMusic: Meant To Be (sped up)\n\n#bluelock #isagi #rin #sae #nagi #mangaedit #anime #shorts",
         "tags": ["blue lock", "manga edit", "anime edit", "shorts", "bluelock", "isagi yoichi", "itoshi rin", "itoshi sae", "nagi seishiro"],
         "music": {"file": "music/template_24_song.wav"}, "body_end": 56.30, "segments": segs, "voicelines": [],
         "look": {"bloom": 0.35, "bloom_thr": 150, "diffuse": 0.25, "sat": 1.08, "contrast": 1.04, "shaky_glitch": 0.10}}
    (ROOT / "edits" / f"{m['id']}.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m["id"]


if __name__ == "__main__":
    import os
    print(build(os.environ.get("MTB_VARIANT")))
