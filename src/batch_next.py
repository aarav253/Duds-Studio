"""Next batch (Oct 15): replicas of references 24-26 with Blue Lock characters.

  nx_meant_to_be   (template 24, "Meant To Be" trend)  mixed cast: 2x2 pop-in grids with lyric words,
                   then fast manga cuts, motion-blur hits and stacked eye strips
  nx_weak_strong   (template 25, "Players")            "WEAK PLAYERS" in blue, then "STRONG PLAYERS" in pink
  nx_kaiser_glow   (template 26, "Glow")               Kaiser: B&W manga intro, "You make me", fast cuts

Each edit plays its reference's own audio, so the reference's cut times are kept
as they are (picture and sound come from the same file).

    python src/batch_next.py      -> edits/nx_*.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw  # noqa: E402

BL = bw.BL
OUT = ROOT / "edits"
FULL = dict(bw.FULL, Teddy="TEDDY KNIGHT", Zantetsu="ZANTETSU TSURUGI")
BW = {"bw": True, "contrast": 1.35}
_next = {}


def take(who, k=1):
    """the next k unused clips of a character (cycling the pool)."""
    p = bw.pool(who); out = []
    for _ in range(k):
        i = _next.get(who, 0); _next[who] = i + 1
        c = p[i % len(p)]
        start = 4 + (i // len(p)) * 40
        out.append({"clip": c["path"], "start": min(start, max(2, c["frames"] - 34))})
    return out


def closeups(who):
    d = json.loads((ROOT / BL / bw.FOLDER[who] / "clips_pool.json").read_text(encoding="utf-8-sig"))
    return [c["path"] for c in d["clips"] if c.get("framing") == "closeup"]


def seg(a, b, spec, **k):
    return dict({"a": round(a, 3), "b": round(b, 3), "layout": "single", "zoom": [1.0, 1.05]}, **spec, **k)


def save(m):
    (OUT / f"{m['id']}.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m["id"]


def meta(id_, title, caption, tags, song):
    return {"id": id_, "title": title, "caption": caption, "watermark": "DUDS", "end_text": "DUDS", "size": [1080, 1350], "fps": 60,
            "description": f"{caption}\n\nMusic: {song}\n\n#bluelock #mangaedit #anime #shorts",
            "tags": ["blue lock", "manga edit", "anime edit", "shorts", "bluelock"] + tags}


# ---------------------------------------------------------------- 24: Meant To Be (mixed cast)
def meant_to_be():
    quads = {"TL": (0, 0, .5, .5), "TR": (.5, 0, 1, .5), "BL": (0, .5, .5, 1), "BR": (.5, .5, 1, 1)}
    # measured in the reference: when each quadrant pops in, and when each grid clears
    grids = [(0.00, 3.03, [("TL", .17), ("BL", .47), ("BR", .83), ("TR", 1.20)], ["Isagi", "Rin", "Nagi", "Kaiser"]),
             (3.03, 6.03, [("TL", .27), ("TR", .50), ("BL", .94), ("BR", 1.34)], ["Sae", "Bachira", "Barou", "Shidou"]),
             (6.03, 9.20, [("BL", .40), ("BR", .64), ("TR", 1.04), ("TL", 1.44)], ["Yukimiya", "Lorenzo", "Rin", "Isagi"]),
             (9.20, 12.40, [("BL", .33), ("TL", .60), ("TR", 1.00), ("BR", 1.37)], ["Kaiser", "Nagi", "Sae", "Isagi"])]
    segs = []
    for a, b, pops, who in grids:
        tiles = [dict(take(w)[0], rect=list(quads[q]), at=at) for (q, at), w in zip(pops, who)]
        segs.append({"a": a, "b": b, "layout": "tiles", "bg": [0, 0, 0], "stars": True, "tiles": tiles, "fx": ["glitch_in"]})
    segs.append({"a": 12.40, "b": 12.60, "layout": "card", "bg": [0, 0, 0], "stars": True, "fx": ["glitch_in", "rgb_in"]})
    red = dict(font="C:/Windows/Fonts/impact.ttf", size=0.085, color=[225, 25, 35], stroke=0.004, stroke_color=[0, 0, 0], glow=0.6,
               glow_color=[225, 25, 35])
    lyr = [("MEANT", 4.98, 5.48), ("TO", 5.48, 6.03), ("THIS", 7.98, 8.48), ("A", 8.48, 8.98), ("TRICK", 8.98, 9.20),
           ("WILL", 11.48, 11.98), ("STICK", 11.98, 12.40), ("BE", 12.40, 12.60)]
    caps = [dict(red, text=w, a=a, b=b, y=0.5) for w, a, b in lyr]

    cast = ["Isagi", "Rin", "Nagi", "Kaiser", "Sae", "Shidou", "Barou", "Bachira", "Yukimiya"]
    eyes = [c for w in ["Isagi", "Rin", "Nagi", "Kaiser", "Sae", "Shidou", "Yukimiya", "Teddy", "Barou"] for c in closeups(w)]
    ci, ei = [0], [0]

    def nxt():
        w = cast[ci[0] % len(cast)]; ci[0] += 1
        return take(w)[0]

    def strips(n):
        out = []
        for i in range(n):
            c = eyes[ei[0] % len(eyes)]; ei[0] += 1
            out.append({"clip": c, "start": 8, "rect": [0, i / n, 1, (i + 1) / n], "at": 0.0, "fy": 0.42, "pop": False})
        return out

    # shot list measured from the reference (start, kind); each runs to the next start
    shots = [(12.60, "bw"), (14.30, "bw"), (14.57, "bw"), (15.00, "bwb"), (15.13, "bw"), (15.40, "bw"), (15.53, "bwb"), (16.13, "bw"),
             (16.30, "bw"), (16.67, "bwb"), (16.83, "bw"), (17.23, "bw"), (18.07, "colb"), (21.27, "dim"), (21.40, "dim"), (21.60, "dim"),
             (21.73, "bw"), (21.87, "col"), (22.20, "s4"), (24.47, "white"), (24.67, "bwb"), (24.80, "bw"), (25.23, "col"), (25.57, "s3"),
             (27.47, "white"), (27.67, "dim"), (27.80, "dim"), (28.00, "dim"), (28.20, "bw"), (28.40, "col"), (28.53, "s4"), (30.60, "white"),
             (31.07, "dim"), (31.27, "bw"), (31.40, "bw"), (31.53, "s4"), (31.90, "s4"), (33.70, "col"), (33.83, "dim"), (33.97, "col"),
             (34.10, "white"), (34.30, "s3bw"), (37.00, "bw"), (37.37, "bw"), (37.57, "col"), (37.70, "s3"), (40.17, "bw"), (40.40, "white"),
             (40.60, "col"), (40.87, "s3"), (41.10, "s4"), (43.37, "pale"), (43.50, "pale"), (43.70, "pale"), (43.83, "col"), (45.37, "colb"),
             (45.73, "col"), (46.17, "col"), (46.57, "colb"), (48.47, "col"), (49.43, "colb"), (51.87, "colb"), (53.17, "col"), (54.43, "colb"),
             (55.13, "colb"), (55.47, "colb"), (55.97, "col")]
    end = 56.50
    for (a, kind), (b, _) in zip(shots, shots[1:] + [(end, None)]):
        if kind == "white":
            segs.append({"a": a, "b": b, "layout": "card", "bg": [245, 245, 245], "fx": ["glitch_in"]})
        elif kind.startswith("s"):
            n = int(kind[1]); g = BW if kind.endswith("bw") else {"contrast": 1.2, "sat": 1.1}
            segs.append({"a": a, "b": b, "layout": "tiles", "bg": [0, 0, 0], "tiles": strips(n), "tile_grade": g, "fx": ["glitch_in", "rgb_in"]})
        else:
            s = seg(a, b, nxt(), zoom=[1.08, 1.0])
            s["grade"] = {"bw": True, "contrast": 1.35} if kind.startswith("bw") else {"contrast": 1.2, "sat": 1.15, "glow": 0.25}
            s["fx"] = ["radial_in"]
            if kind.endswith("b"):
                s["fx"] = ["mblur", "radial_in"]; s["mblur"] = 0.06; s["mblur_angle"] = 25 if a % 2 > 1 else -20
            if kind == "dim":
                s["grade"] = {"bw": True, "contrast": 0.7, "bright": 0.45}; s["fx"] = ["mblur"]; s["mblur"] = 0.04
            if kind == "pale":
                s["grade"] = {"sat": 0.5, "contrast": 0.6, "bright": 1.45}; s["fx"] = ["mblur"]; s["mblur"] = 0.05
            segs.append(s)
    segs[-1]["fx"] = segs[-1].get("fx", []) + ["end_blur"]
    m = meta("nx_meant_to_be", "Meant To Be | Blue Lock", "it was meant to be...", ["isagi yoichi", "itoshi rin", "nagi seishiro", "michael kaiser"],
             "Meant To Be (sped up)")
    m.update(music={"file": "music/template_24_ref.wav"}, body_end=end, segments=segs, captions=caps, voicelines=[])
    return save(m)


# ---------------------------------------------------------------- 25: Players -> weak / strong
_PK = json.loads((ROOT / "edits/clip_picks.json").read_text(encoding="utf-8"))["weak"]


def picked(who):
    """hand-picked, well-framed clip for this edit (edits/clip_picks.json), cycling."""
    if who in ("Chigiri", "Reo", "Karasu"):                        # no clips: their official art, framed on the face
        return {"clip": BL + f"_cards/{who.lower()}_face.jpg", "start": 0, "zoom": [1.0, 1.05]}
    L = _PK.get(who)
    if not L:
        return take(who)[0]
    i = _next.get(("pk", who), 0); _next[("pk", who)] = i + 1
    return {"clip": L[i % len(L)], "start": 6}


def weak_strong():
    blue = {"tint": [120, 130, 225], "tint_amt": 0.8, "contrast": 1.2, "bright": 1.05}
    pink = {"tint": [235, 105, 150], "tint_amt": 0.8, "contrast": 1.2, "bright": 1.0}
    capw = dict(text='" BLUE LOCK "', y=0.665, size=0.045, color=[255, 255, 255], stroke=0.003, stroke_color=[0, 0, 0], glow=0.4)
    caps_ = dict(text='" NEW GENERATION 11 "', y=0.665, size=0.045, color=[230, 25, 40], stroke=0.003, stroke_color=[20, 0, 0],
                 glow=0.5, glow_color=[230, 25, 40])
    # (start, who, fx) - every shot runs to the next start; micro-cuts inside a shot become shake/glitch hits
    weak = [(0.00, "Isagi", []), (0.87, "Rin", ["rgb_in", "shake"]), (1.88, "Bachira", ["glitch_in"]),     # 11 different players
            (2.87, "Nagi", ["radial_in"]), (3.88, "Barou", ["glitch_in", "shake"]), (4.87, "Shidou", ["radial_in"]),
            (5.87, "Yukimiya", ["glitch_in"]), (6.87, "Zantetsu", ["rgb_in", "shake"]), (7.87, "Isagi", ["radial_in"]),
            (8.87, "Rin", ["glitch_in", "shake"]), (9.87, "Nagi", ["radial_in"])]
    strong = [(11.80, "Sae", ["glitch_in"]), (12.87, "Sae", ["rgb_in", "shake"]), (13.87, "Kaiser", ["radial_in"]),
              (14.55, None, "dark"), (14.68, "Kaiser", ["glitch_in", "shake"]), (15.83, None, "static"),
              (15.98, "Lorenzo", ["glitch_in"]), (16.87, "Lorenzo", ["rgb_in", "shake"]), (17.87, "Loki", ["radial_in"]),
              (18.87, "Loki", ["glitch_in"]), (19.83, None, "static"), (20.08, "Bunny", ["flicker"]), (20.68, None, "dark"),
              (20.87, "Hugo", ["flicker", "shake"]), (21.75, None, "dark"), (21.87, "Hugo", ["glitch_in", "shake"]),
              (22.87, "Teddy", ["radial_in"])]
    end = 23.87
    segs = []

    def build(lst, stop, g, cap):
        for (a, who, fx), (b, *_) in zip(lst, lst[1:] + [(stop,)]):
            if fx == "dark":
                segs.append({"a": a, "b": b, "layout": "card", "bg": [8, 4, 8], "captions": [dict(cap)]})
            elif fx == "static":
                segs.append({"a": a, "b": b, "layout": "card", "bg": [150, 150, 160], "fx": ["static"], "static_tint": [140, 230, 240]})
            else:
                pk_ = picked(who)
                s = seg(a, b, pk_, zoom=pk_.pop("zoom", [1.0, 1.04]), grade=dict(g), fx=list(fx), captions=[dict(cap)])
                if "flicker" in fx:
                    s["flicker_frames"] = 4
                segs.append(s)

    build(weak, 10.87, blue, capw)
    segs += [{"a": 10.87, "b": 10.98, "layout": "card", "bg": [235, 235, 245]},
             {"a": 10.98, "b": 11.17, "layout": "card", "bg": [70, 70, 80], "stars": True},
             {"a": 11.17, "b": 11.30, "layout": "card", "bg": [235, 235, 245]},
             {"a": 11.30, "b": 11.80, "layout": "card", "bg": [5, 5, 8], "stars": True}]
    build(strong, end, pink, caps_)
    m = meta("nx_weak_strong", "Blue Lock vs New Generation 11", "blue lock... new generation 11",
             ["isagi yoichi", "itoshi rin", "michael kaiser", "nagi seishiro"], "Players (Wekisava edit audio)")
    m.update(music={"file": "music/template_25_ref.wav"}, body_end=end, segments=segs, voicelines=[])
    return save(m)


# ---------------------------------------------------------------- 26: Kaiser - Glow
def kaiser_glow():
    pur = {"tint": [190, 140, 235], "tint_amt": 0.35, "contrast": 1.25, "glow": 0.3}
    bwg = {"bw": True, "contrast": 1.4, "bright": 1.1}
    K = lambda: take("Kaiser")[0]
    segs = [{"a": 0.0, "b": 0.25, "layout": "card", "bg": [0, 0, 0]},
            seg(0.25, 1.25, K(), grade=bwg, zoom=[1.0, 1.12], pan=[0.04, 0], fx=["fade_in"], fade_time=0.15),
            seg(1.25, 2.00, K(), grade=bwg, zoom=[1.05, 1.15], fx=["radial_in"]),
            seg(2.00, 3.25, K(), grade=bwg, zoom=[1.0, 1.1], pan=[-0.04, 0], fx=["mblur"], mblur=0.03),
            seg(3.25, 3.50, K(), grade=bwg, zoom=[1.1, 1.0]),
            {"a": 3.50, "b": 3.77, "layout": "card", "bg": [210, 210, 215], "fx": ["static"], "static_tint": [235, 235, 235]}]
    ym = dict(font="C:/Windows/Fonts/georgiai.ttf", size=0.06, y=0.5, color=[245, 245, 245], stroke=0.002, stroke_color=[0, 0, 0], glow=0.5)
    flick = [(3.77, None, "You"), (3.85, "p", "You make"), (3.97, "p", "You make"), (4.09, "p", "You make"), (4.15, None, "You make me"),
             (4.22, "p", "You make me"), (4.40, None, "You make me"), (4.47, "p", "You make me")]
    panel = K()
    for (a, kind, txt), (b, *_) in zip(flick, flick[1:] + [(4.59,)]):
        cap = [dict(ym, text=txt)]
        if kind:
            segs.append(seg(a, b, panel, grade=bwg, zoom=[1.04, 1.04], captions=cap))
        else:
            segs.append({"a": a, "b": b, "layout": "card", "bg": [4, 4, 6], "stars": True, "captions": cap})
    segs.append({"a": 4.59, "b": 5.00, "layout": "card", "bg": [40, 60, 160], "fx": ["static"], "static_tint": [90, 120, 255]})
    segs.append(seg(5.00, 5.39, K(), grade=dict(pur), zoom=[1.0, 1.04], fx=["invert_in", "zoom_punch"]))
    cuts = [5.39, 6.94, 7.77, 8.34, 8.46, 8.66, 9.56, 9.99, 10.06, 10.48, 10.54, 10.96, 11.03, 11.43, 11.49, 12.23, 13.03, 13.86, 14.68,
            15.52, 15.60, 16.42, 16.87]
    end = 18.40
    splits = {9.99, 10.48, 10.96}
    for i, (a, b) in enumerate(zip(cuts, cuts[1:] + [end])):
        if a == 15.52:
            segs.append({"a": a, "b": b, "layout": "card", "bg": [245, 240, 250], "fx": ["glitch_in"]})
        elif a in splits:
            segs.append({"a": a, "b": b, "layout": "versus", "clips": [K()["clip"], K()["clip"]], "start": 6, "grade": bwg,
                         "divider_color": [250, 250, 250]})
        else:
            fx = [["mblur", "radial_in"], ["glitch_in", "shake"], ["rgb_in", "zoom_punch"], ["radial_in"]][i % 4]
            s = seg(a, b, K(), grade=dict(pur) if i % 3 else dict(bwg), zoom=[1.08, 1.0], fx=fx, mblur=0.05)
            if a == 5.39:
                s["fx"] = ["white_in", "mblur"]; s["mblur"] = 0.08
            segs.append(s)
    segs[-1]["fx"] = segs[-1]["fx"] + ["end_blur"]
    m = meta("nx_kaiser_glow", "Michael Kaiser Edit | Blue Lock", "you make me glow", ["michael kaiser", "kaiser"], "Mr.Kitty - Glow")
    m.update(music={"file": "music/template_26_ref.wav"}, body_end=end, segments=segs, voicelines=[])
    return save(m)


if __name__ == "__main__":
    print("\n".join([meant_to_be(), weak_strong(), kaiser_glow()]))
