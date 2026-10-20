"""Templates 28, 29, 32 as Blue Lock edits (Oct 16-17 request).

  t28_top4       "My Top 4 Favourite Blue Lock Characters": B&W manga intro with the title, manga-page pans,
                 then #4 Yukimiya (purple), #3 Rin (ice blue), #2 Teddy (deep blue), #1 Sae (B&W + gold),
                 each with name + ratings appearing.  Song: Sem Tempo (Super Slowed) - INSVNIA, sample-exact.
  t29_most_ego   "Most Ego": Isagi B&W manga intro (4 held panels), then a red-tinted Barou edit.
                 Song: the reference's own audio (not identified on YouTube).
  t32_antagonist "The Real Antagonist": high-contrast B&W, words SHOW / ME / WHO / YOU / ARE on the beat,
                 then the title card held over fast Shidou cuts.  Song: the reference's own audio.

    python src/batch_next3.py [t28|t29|t32]
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw  # noqa: E402

BL = bw.BL
_next = {}
SERIF = "C:/Windows/Fonts/georgiai.ttf"
IMPACT = "C:/Windows/Fonts/impact.ttf"
BOLD = "C:/Windows/Fonts/bahnschrift.ttf"


PICKS = json.loads((ROOT / "edits/clip_picks.json").read_text(encoding="utf-8"))
_pick_set = None


def use_picks(name):
    """route clip(who) through the hand-picked list for one edit (edits/clip_picks.json)."""
    global _pick_set
    _pick_set = PICKS.get(name)
    for k in [k for k in _next if k[0] == "p"]:
        _next.pop(k)


def clip(who):
    if _pick_set and _pick_set.get(who):
        L = _pick_set[who]; i = _next.get(("p", who), 0); _next[("p", who)] = i + 1
        return {"clip": L[i % len(L)], "start": 6}
    d = json.loads((ROOT / BL / bw.FOLDER[who] / "clips_pool.json").read_text(encoding="utf-8-sig"))["clips"]
    ps = [c for c in d if not c.get("framing") and c["brightness"] > 45 and "/clips_x" not in c["path"]] or \
         [c for c in d if not c.get("framing") and c["brightness"] > 45]
    i = _next.get(("c", who), 0); _next[("c", who)] = i + 1
    c = ps[i % len(ps)]
    return {"clip": c["path"], "start": min(6, c["frames"] // 4)}


def image(who, bw_=True):
    d = ROOT / BL / bw.FOLDER[who] / "images"
    ps = [q for q in sorted(d.glob("*.jpg"))
          if (cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2HSV)[..., 1].mean() < 25) == bw_] if d.exists() else []
    if not ps:
        return clip(who)
    i = _next.get(("i", who, bw_), 0); _next[("i", who, bw_)] = i + 1
    return {"clip": str(ps[i % len(ps)].relative_to(ROOT)).replace("\\", "/"), "start": 0}


def single(a, b, spec, grade, fx=(), **k):
    z = k.pop("zoom", [1.06, 1.0])
    return dict(spec, a=round(a, 3), b=round(b, 3), layout="single", zoom=z, grade=dict(grade), fx=list(fx), **k)


def card(a, b, bg):
    return {"a": round(a, 3), "b": round(b, 3), "layout": "card", "bg": bg}


def cap(text, y, size, color, font=BOLD, **k):
    return dict({"text": text, "y": y, "size": size, "color": color, "font": font, "stroke": 0.003, "stroke_color": [0, 0, 0]}, **k)


def seal(segs, end):
    segs.sort(key=lambda s: s["a"])
    for x, y in zip(segs, segs[1:]):
        x["b"] = y["a"]
    segs[-1]["b"] = end
    return [s for s in segs if s["b"] - s["a"] > 0.005]


def meta(id_, title, caption, tags, song, music, body_end, segs, look=None, voices=()):
    m = {"id": id_, "title": title, "caption": caption, "watermark": "DUDS", "end_text": "DUDS", "size": [1080, 1350], "fps": 60,
         "description": f"{caption}\n\nMusic: {song}\n\n#bluelock #mangaedit #anime #shorts",
         "tags": ["blue lock", "manga edit", "anime edit", "shorts", "bluelock"] + tags,
         "music": {"file": music}, "body_end": body_end, "segments": segs, "voicelines": list(voices),
         "duck_db": 7.0, "voice_over_music": 2.8}
    if look:
        m["look"] = look
    (ROOT / "edits" / f"{id_}.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return id_


def cuts_of(n):
    return json.loads((ROOT / f"references/template_{n}/cuts.json").read_text())


# ---------------------------------------------------------------- 28
def t28():
    R = cuts_of(28)
    BWG = {"bw": True, "contrast": 1.35, "bright": 1.05}
    mix = ["Rin", "Sae", "Isagi", "Nagi"]
    title = [cap("MY TOP 4 FAVOURITE", 0.455, 0.045, [235, 235, 245], a=0.2, b=9.73),
             cap("BLUE LOCK", 0.505, 0.1, [70, 110, 235], IMPACT, a=0.2, b=9.73, stroke=0.004, glow=0.4, glow_color=[70, 110, 235]),
             cap("CHARACTERS", 0.555, 0.042, [235, 235, 245], a=0.2, b=9.73)]
    segs = [card(0, 0.2, [250, 250, 250])]
    intro = [0.2, 0.683, 1.35, 1.967, 2.6, 3.2, 3.8, 4.33, 4.6]
    for i, (a, b) in enumerate(zip(intro, intro[1:] + [4.9])):
        segs.append(single(a, b, image(mix[i % 4]), BWG, ["fade_in"] if i == 0 else ["radial_in"], fade_time=0.3, zoom=[1.08, 1.0]))
    segs.append(card(4.9, 5.0, [250, 250, 250]))
    pages = [5.0, 7.07, 8.07, 9.0]                                  # slow pans across manga pages
    for i, (a, b) in enumerate(zip(pages, pages[1:] + [9.73])):
        segs.append(single(a, b, image(mix[(i + 1) % 4]), BWG, ["radial_in"], zoom=[1.25, 1.15], pan=[0.06 if i % 2 else -0.06, 0.08]))
    segs.append(dict(single(9.73, 9.9, image("Rin"), BWG, ["glitch_in", "rgb_in"]), glitch_red=False))
    for s in segs:
        s["captions"] = [dict(c, a=max(0, c["a"] - s["a"]), b=c["b"] - s["a"]) for c in title if s["a"] < c["b"]]
    use_picks("t28")
    ranks = [("Yukimiya", "YUKIMIYA KENYU", 4, 9.9, 12.3, [170, 60, 230]), ("Rin", "ITOSHI RIN", 3, 12.3, 14.7, [150, 215, 245]),
             ("Teddy", "TEDDY KNIGHT", 2, 14.7, 16.93, [40, 90, 255]), ("Sae", "ITOSHI SAE", 1, 16.93, 19.68, [230, 190, 60])]
    beats = [10.04, 10.64, 11.28, 11.88, 12.5, 13.12, 13.74, 14.34, 14.96, 15.58, 16.2, 16.8, 17.42, 18.04, 18.66, 19.28]
    g = {"contrast": 1.08, "sat": 1.05, "glow": 0.2}               # the clips' own colours
    dirs = ["left", "up", "right", "down"]
    k = 0
    for who, name, n, a0, b0, col in ranks:
        caps = [cap(str(n), 0.665, 0.045, [245, 245, 245], SERIF),
                cap(name, 0.71, 0.06, col if n != 4 else [235, 200, 255], SERIF, glow=0.5, glow_color=col)]
        bt = [x for x in beats if a0 + 0.15 < x < b0 - 0.15]
        # first shot: the manga panel holds for at least ~0.7 s, the colour wipes in ON a beat, the coloured
        # version runs one more beat; then a new panel every half beat
        reveal_at = next((x for x in bt if x - a0 >= 0.7), bt[0] if bt else a0)
        rest = [x for x in bt if x > reveal_at + 0.1]
        first_end = rest[0] if rest else b0
        marks = [a0, first_end]
        for x, y in zip(rest, rest[1:] + [b0]):
            marks.append(x)
            if y - x > 0.5:
                marks.append(round((x + y) / 2, 3))
        marks = sorted(set(m for m in marks if m < b0 - 0.1))
        for i, (a, b) in enumerate(zip(marks, marks[1:] + [b0])):
            on_beat = any(abs(a - x) < 0.02 for x in beats)
            s_ = single(a, b, clip(who), g, [], zoom=[1.0, 1.08] if i % 2 else [1.08, 1.0], captions=caps)
            s_["zoom"] = [1.0, 1.06]; s_["zoom_bounce"] = True     # smooth: gentle drift, every cut slides in
            if i == 0:
                s_["manga_reveal"] = {"at": round((reveal_at - a) / (b - a), 3), "wipe": 0.18}
                s_["push"] = {"dir": "left", "dur": 0.32}                 # slide in from the previous character
                s_["flashes"] = [[round(reveal_at - 0.02, 3), round(reveal_at + 0.07, 3), 0.85],      # flash as the colour hits
                                 [round(reveal_at + 0.07, 3), round(reveal_at + 0.16, 3), 0.35]]
            elif on_beat:
                s_["fx"] = ["white_in"]                            # soft flash on the beat
                s_["push"] = {"dir": dirs[k % 4], "dur": 0.24}; k += 1
            else:
                s_["push"] = {"dir": dirs[k % 4], "dur": 0.2}; k += 1
            segs.append(s_)
    segs.append(card(19.68, 19.7, [250, 250, 250]))
    return meta("t28_top4", "My Top 4 Favourite Blue Lock Characters", "my top 4 favourite blue lock characters",
                ["itoshi sae", "teddy knight", "itoshi rin", "yukimiya kenyu"], "Sem Tempo (Super Slowed) - INSVNIA",
                "music/template_28_song.wav", 19.7, seal(segs, 19.7), {"bloom": 0.3, "bloom_thr": 160, "shaky_glitch": 0.06})


# ---------------------------------------------------------------- 29
def t29(red_scheme=True, scheme=None, drop="split"):
    """intro: four different characters (B&W manga) labelled EGO; the edit: Teddy, labelled NO EGO, one shot per beat of
    the song (130 bpm, beats measured on the audio), extra cuts on the hard hits, smooth pushes on the downbeats, and the
    last shot held to the downbeat at 14.94 before the outro. red_scheme=False -> plain high-contrast B&W."""
    white = cap('" EGO "', 0.7, 0.05, [250, 250, 250], BOLD, variation="Bold", stroke=0.004)
    red = dict(white, text='" NO EGO "', color=[230, 25, 35] if red_scheme else [250, 250, 250], glow=0.5,
               glow_color=[230, 25, 35] if red_scheme else [0, 0, 0])
    BWG = {"bw": True, "contrast": 1.35, "bright": 1.05}
    intro = [0.0, 1.717, 3.567, 5.4]
    I = lambda f: {"clip": BL + f, "start": 0}
    picks = [I("Isagi Yoichi/images/isagi_18.jpg"), I("Rin Itoshi/images/rin_29.jpg"),     # hand-picked: clear, ego faces
             I("Sae Itoshi/images/sae_15.jpg"), I("Nagi Seishiro/images/nagi_26.jpg")]
    segs = [single(a, b, picks[i], BWG, ["radial_in", "fade_in"] if i == 0 else ["radial_in"], fade_time=0.15,
                   zoom=[1.0, 1.06], captions=[white]) for i, (a, b) in enumerate(zip(intro, intro[1:] + [7.25]))]
    segs.append(card(7.25, 7.283, [250, 250, 250]))
    g = {"bw": True, "contrast": 1.3, "tint": [200, 25, 35], "tint_amt": 0.65, "glow": 0.25} if red_scheme else         {"bw": True, "contrast": 1.4, "bright": 1.05, "glow": 0.3}
    if scheme == "orig":                                          # the clips' own colours
        g = {"contrast": 1.08, "glow": 0.25}
        red = dict(red, color=[250, 250, 250], glow=0.5, glow_color=[0, 0, 0])
    if scheme == "blue":
        g = {"bw": True, "contrast": 1.25, "tint": [120, 200, 255], "tint_amt": 0.6, "glow": 0.35}
        red = dict(red, color=[150, 215, 255], glow=0.6, glow_color=[90, 180, 255])
    beats = [7.283, 7.54, 8.02, 8.48, 8.94, 9.4, 9.86, 10.34, 10.8, 11.26, 11.72, 12.18, 12.64, 13.1, 13.56, 14.02, 14.48]
    hits = [8.66, 12.34, 14.64]                                     # strongest drum hits between beats
    downs = {7.54, 9.4, 11.26, 13.1}
    cs = sorted(set(beats) | set(hits))
    end = 14.94
    T = BL + "Teddy Knight/clips/scene_%02d.mp4"                    # hand-picked: clear Teddy faces / actions
    teddy = [T % n for n in (30, 10, 13, 27, 12, 17, 14, 19, 25, 11, 37, 16, 20, 44)]
    dirs = ["left", "up", "right", "down"]
    for i, (a, b) in enumerate(zip(cs, cs[1:] + [end])):
        s_ = single(a, b, {"clip": teddy[i % len(teddy)], "start": 6}, g, ["zoom_punch"] if a in hits else [],
                    zoom=[1.0, 1.1] if i % 2 else [1.1, 1.0], captions=[red])
        if a in downs:
            s_["push"] = {"dir": dirs[(i // 2) % 4], "dur": 0.16}
        elif a not in hits and i:
            s_["fx"] = ["white_in"] if i % 3 == 0 else ["shake"]
        segs.append(s_)
    if scheme == "blue":                                          # the beat-drop transition (7.25 - 8.02)
        segs = [x for x in segs if not (7.25 <= x["a"] < 8.02)]
        segs += drop_t29(drop, teddy, g, red)
    return meta("t29_most_ego" + (f"_{scheme}" if scheme else "" if red_scheme else "_bw") + ("" if drop == "split" else f"_{drop}"), "Ego vs No Ego | Teddy Knight", "ego... no ego",
                ["teddy knight", "teddy", "isagi yoichi", "itoshi rin"], "original edit audio", "music/template_29_ref.wav",
                end, seal(segs, end))


def drop_t29(kind, teddy, g, cap_):
    """five beat-drop transitions borrowed from the other edits; all fill 7.25 - 8.02 (drop on 7.283, beat 7.54)."""
    T = lambda i, **k: dict({"clip": teddy[i % len(teddy)], "start": 6}, **k)
    manga = [dict(image(w), a=round(k * 0.09, 3), b=round(k * 0.09 + 0.09, 3), grade={"bw": True, "contrast": 1.3}, mblur=0.04)
             for k, w in enumerate(["Isagi", "Rin", "Sae", "Nagi", "Isagi", "Rin", "Sae", "Nagi", "Isagi"])]
    if kind == "split":                                          # Meant To Be: face halves slam in from both sides
        img = T(0, still=True)
        tiles = [dict(img, rect=[0, 0, .5, 1], half="L", at=0.0, dur=0.17, smear_amt=0.2, impact=0.2, impact_zoom=0.1,
                      impact_blur=0.08, **{"from": "left"}),
                 dict(img, rect=[.5, 0, 1, 1], half="R", at=0.257, dur=0.17, smear_amt=0.2, impact=0.2, impact_zoom=0.1,
                      impact_blur=0.08, **{"from": "right"})]
        return [{"a": 7.25, "b": 8.02, "layout": "tiles", "bg": [0, 0, 0], "tiles": tiles, "under": manga, "tile_grade": g,
                 "fx": ["white_in", "glitch_in"], "hits": [7.537], "captions": [cap_]}]
    if kind == "eyes":                                           # Meant To Be: four strips drop in from the top with impact
        d = json.loads((ROOT / BL / "Teddy Knight/clips_pool.json").read_text(encoding="utf-8"))["clips"]
        cl = [c["path"] for c in d if c.get("framing") == "closeup"][:4] or teddy[:4]
        tiles = [dict({"clip": c, "start": 8}, rect=[0, i / 4, 1, (i + 1) / 4], fy=0.42, at=round(i * 0.12, 3), dur=0.08,
                      smear_amt=0.12, impact=0.2, impact_zoom=0.14, impact_blur=0.06, **{"from": "top"}) for i, c in enumerate(cl)]
        return [{"a": 7.25, "b": 8.02, "layout": "tiles", "bg": [0, 0, 0], "tiles": tiles, "under": manga, "tile_grade": g,
                 "fx": ["white_in"], "hits": [7.537], "captions": [cap_]}]
    if kind == "grid":                                           # Meant To Be intro / Best Striker: 2x2 grid slides in on the beats
        quads = [((0, 0, .5, .5), "left"), ((.5, 0, 1, .5), "top"), ((0, .5, .5, 1), "bottom"), ((.5, .5, 1, 1), "right")]
        tiles = [T(i + 1, rect=list(r), at=round(0.03 + i * 0.12, 3), dur=0.12, still=True, smear_amt=0.12, impact=0.15,
                   impact_zoom=0.08, **{"from": f}) for i, (r, f) in enumerate(quads)]
        return [{"a": 7.25, "b": 7.79, "layout": "tiles", "bg": [8, 12, 20], "tiles": tiles, "tile_grade": g, "fx": ["white_in"],
                 "captions": [cap_]},
                single(7.79, 8.02, T(0), g, ["white_in", "zoom_punch", "shake"], zoom=[1.15, 1.0], captions=[cap_])]
    if kind == "static":                                         # Weak/Strong: TV static, then an inverted flash into Teddy
        return [{"a": 7.25, "b": 7.42, "layout": "card", "bg": [150, 170, 190], "fx": ["static"], "static_tint": [140, 210, 255]},
                single(7.42, 7.54, T(0), g, ["invert_in", "rgb_in", "shake"], zoom=[1.2, 1.1], shake_amt=0.03),
                single(7.54, 8.02, T(0), g, ["white_in", "zoom_punch", "shake"], zoom=[1.12, 1.0], shake_amt=0.015, captions=[cap_])]
    if kind == "punch":                                          # Kaiser Glow / Best Striker: radial blast + stutter zooms on the beat
        segs = [single(7.25, 7.33, T(0), g, ["white_in"], zoom=[1.4, 1.3])]
        t, k = 7.33, 0
        for z in (1.3, 1.22, 1.15):                              # three stutter punches leading into the beat
            segs.append(single(t, t + 0.07, T(0), g, ["radial_in", "rgb_in"], zoom=[z, z - 0.04], shake_amt=0.02)); t += 0.07; k += 1
        segs.append(single(t, 7.54, T(0), g, ["radial_in"], zoom=[1.1, 1.05]))
        segs.append(single(7.54, 8.02, T(0), g, ["white_in", "zoom_punch", "shake", "radial_in"], zoom=[1.12, 1.0], captions=[cap_]))
        for x in segs:
            x.setdefault("captions", [cap_])
        return segs
    raise ValueError(kind)


# ---------------------------------------------------------------- 32
def t32():
    R = cuts_of(32)
    HC = {"bw": True, "contrast": 2.3, "bright": 1.12}
    words = [(0.5, "SHOW"), (1.183, "ME"), (1.783, "WHO"), (2.483, "YOU"), (3.083, "ARE"), (3.7, "ARE."), (4.4, "ARE..")]
    R2 = BL + "Rin Itoshi/"
    kid = [{"clip": R2 + "clips_2/scene_41.mp4", "start": 10}, {"clip": R2 + "images/rin_11.jpg", "start": 0},
           {"clip": R2 + "clips_2/scene_42.mp4", "start": 10}, {"clip": R2 + "images/rin_55.jpg", "start": 0},
           {"clip": R2 + "clips_2/scene_42.mp4", "start": 35}, {"clip": R2 + "clips_2/scene_41.mp4", "start": 60},
           {"clip": R2 + "images/rin_38.jpg", "start": 0}, {"clip": R2 + "clips_2/scene_42.mp4", "start": 60},
           {"clip": R2 + "clips_2/scene_41.mp4", "start": 35}]
    segs = [single(0.0, 0.5, kid[0], HC, ["fade_in"], fade_time=0.1, zoom=[1.04, 1.0])]
    for i, (a, w) in enumerate(words):
        b = words[i + 1][0] if i + 1 < len(words) else 5.117
        col = [220, 25, 35] if w == "WHO" else [250, 250, 250]
        segs.append(single(a, b, kid[i + 1], HC, ["radial_in"], zoom=[1.0, 1.04],
                           captions=[cap(w, 0.7, 0.05, col, BOLD, variation="Bold", glow=0.6, glow_color=[0, 0, 0])]))
    segs.append(single(5.117, 5.767, kid[8], HC, ["zoom_punch"], zoom=[1.0, 1.06]))
    segs.append(card(5.767, 5.9, [0, 0, 0]))
    title = [cap("THE REAL ANTAGONIST", 0.675, 0.04, [235, 235, 235], BOLD),
             cap('"THE DESTROYER"', 0.71, 0.04, [220, 25, 35], BOLD, glow=0.4, glow_color=[220, 25, 35]),
             cap("ITOSHI RIN", 0.745, 0.036, [220, 25, 35], BOLD)]
    cs = [5.9] + [r for r in R if 5.95 < r < 16.5]
    for i, (a, b) in enumerate(zip(cs, cs[1:] + [16.533])):
        fx = ["radial_in", "zoom_punch"] if b - a > 0.2 else ["glitch_in", "shake"]
        s = single(a, b, clip("Rin"), HC, fx, zoom=[1.1, 1.0], captions=title)
        if b - a < 0.06:
            s = dict(card(a, b, [0, 0, 0] if i % 2 else [250, 250, 250]), captions=title)
        segs.append(s)
    return meta("t32_antagonist", "The Real Antagonist | Itoshi Rin", "the real antagonist...", ["itoshi rin", "rin"],
                "original edit audio", "music/template_32_ref.wav", 16.533, seal(segs, 16.533))


if __name__ == "__main__":
    want = sys.argv[1:] or ["t28", "t29", "t32"]
    print("\n".join(globals()[w]() for w in want))


# ---------------------------------------------------------------- 31
def t31():
    """'The strongest characters in Blue Lock' (template 31, NIK0LH's Haikyuu edit): a fast parade of the cast under
    the title, then nine characters, two shots each, name glowing in their colour, epithet in brackets above it."""
    import json as _j
    import glob as _g
    cards = sorted(str(Path(p).relative_to(ROOT)).replace("\\", "/") for p in _g.glob(str(ROOT / BL / "*/body/official_white.jpg")))
    cards += sorted(str(Path(p).relative_to(ROOT)).replace("\\", "/") for p in _g.glob(str(ROOT / BL / "_cards/*_white.jpg")))
    cards = [c for c in cards if "/Sae Itoshi/" not in c and "/Isagi Yoichi/" not in c]   # they get their own section
    import random as _r
    _r.Random(7).shuffle(cards)                                    # 31 different players, no repeats in the parade
    title = [cap("THE STRONGEST CHARACTERS IN", 0.43, 0.046, [20, 24, 40], BOLD, variation="Bold", a=0.9, b=14.0, glow=0.0, stroke=0.0,
                 glow_color=[255, 255, 255]),
             cap("BLUE LOCK", 0.49, 0.1, [60, 110, 240], IMPACT, a=0.9, b=14.0, stroke=0.004, glow=0.6, glow_color=[60, 110, 240])]
    segs = []
    t, k = 0.0, 0
    while t < 14.0 - 0.05:                                         # a new character every ~0.23 s
        b = min(14.0, t + (0.9 if k == 0 else 0.233))
        w = cards[k % len(cards)]                                   # official full-body art on white, like the template
        s = single(t, b, {"clip": w, "start": 0}, {"contrast": 1.03, "glow": 0.15},
                   ["fade_in"] if k == 0 else [], fade_time=0.9, zoom=[1.0, 1.0])
        s["captions"] = [dict(c, a=max(0, c["a"] - t), b=c["b"] - t) for c in title if t < c["b"] and b > c["a"]]
        segs.append(s); t = b; k += 1
    cast = [("Isagi", "ISAGI YOICHI", "(THE GENIUS OF ADAPTABILITY)", [70, 130, 255], (14.0, 16.533, 16.7, 18.867)),
            ("Sae", "ITOSHI SAE", "(THE GENIUS MIDFIELDER)", [240, 70, 90], (19.3, 21.867, 21.867, 24.533)),
            ("Teddy", "TEDDY KNIGHT", "(THE STRONGEST DEFENDER)", [150, 215, 255], (24.767, 27.2, 27.2, 29.533)),
            ("Bunny", "BUNNY IGLESIAS", "(THE TRICKSTER)", [245, 150, 200], (29.867, 32.5, 32.5, 35.167)),
            ("Loki", "JULIAN LOKI", "(THE FASTEST MAN)", [250, 190, 60], (35.267, 37.833, 37.9, 40.167)),
            ("Hugo", "HUGO", "(THE STRATEGIST)", [230, 60, 70], (40.533, 43.167, 43.167, 45.5)),
            ("Kaiser", "MICHAEL KAISER", "(THE EMPEROR)", [80, 120, 255], (45.867, 48.5, 48.5, 50.833)),
            ("Lorenzo", "DON LORENZO", "(THE ZOMBIE)", [240, 190, 50], (51.267, 53.833, 54.0, 56.533))]
    rank = ["Isagi", "Lorenzo", "Kaiser", "Teddy", "Sae", "Hugo", "Bunny", "Loki"]   # the user's ranking, #8 first -> #1 last
    slots = [c[4] for c in cast]
    by = {c[0]: c for c in cast}
    cast = [by[w][:4] + (slots[i],) for i, w in enumerate(rank)]
    use_picks("t31")
    prev_end = 14.0
    for n, (who, name, epi, col, (a1, b1, a2, b2)) in enumerate(cast):
        if a1 > prev_end + 0.01:                                    # the reference's black / blur gaps between characters
            segs.append(card(prev_end, a1, [0, 0, 0]))
        g = {"contrast": 1.15, "sat": 1.1, "glow": 0.35}
        nm = cap(name, 0.565, 0.058, col, BOLD, variation="Bold", glow=1.0, glow_color=col,
                 band={"h": 0.05, "blur": 0.02, "alpha": 0.55, "color": [10, 10, 14]})
        ep = cap(epi, 0.525, 0.036, [245, 245, 250], BOLD, variation="Bold", glow=0.6, glow_color=col)
        sec_a = a1
        for (a, b), z in (((a1, b1), [1.05, 1.0]), ((a2, b2), [1.0, 1.04])):
            if b - a < 0.05:
                continue
            s = single(a, b, clip(who), g, ["radial_in"], zoom=z)
            s["captions"] = [dict(nm, a=max(0, sec_a + 0.25 - a)),   # name only (no bracket line), rank above it
                             cap(f"#{8 - n}", 0.49, 0.075, [245, 245, 250], IMPACT, stroke=0.004, glow=0.9, glow_color=col,
                                 a=max(0, sec_a + 0.1 - a))]
            segs.append(s)
        prev_end = max(b1, b2)
    return meta("t31_strongest", "The Strongest Characters In Blue Lock", "the strongest characters in blue lock",
                ["isagi yoichi", "itoshi sae", "teddy knight", "bunny iglesias", "julian loki", "michael kaiser", "don lorenzo"],
                "BRODYAGA FUNK (Super Slowed + Reverb)", "music/template_31_song.wav", 56.6, seal(segs, 56.6),
                {"bloom": 0.35, "bloom_thr": 150, "sat": 1.05})
