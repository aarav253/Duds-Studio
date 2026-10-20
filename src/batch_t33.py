"""Templates 33-35 as Blue Lock edits (Oct 17 request: "use whatever characters you want").

  t33_seven_sins     "7 Deadly Sins": title + the seven sins one by one (numbered, in their colour, the character
                     faint behind), then one ~2.45 s section per sin on the reference's cuts:
                     Pride Kaiser, Greed Lorenzo, Lust Shidou, Envy Rin, Gluttony Barou, Wrath Isagi, Sloth Nagi.
                     Song: MONTAGEM GLORIA (Slowed) - SUKA.
  t34_goated_bros    "POV: The Goated Brothers" (template 34, Naruto/Minato family edit) with Rin & Sae: quick intro,
                     stacked strips under the POV bar, the two long hero shots, fast alternating cuts, glowing ITOSHI.
                     Song: LOS VOLTAJE FUNK (Slowed) - Fyex.
  t35_nagi_teal      Template 35 (Kakashi teal edit) with Nagi: B&W manga flicker, then a teal halftone edit with
                     white flash out (no on-screen name).  Audio: the reference's own, declipped (song not
                     found; its vocals are sung chops, no dialogue).

    python src/batch_t33.py [t33|t34|t35]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARGS, sys.argv = sys.argv[1:], sys.argv[:1]                       # batch_week reads argv on import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw  # noqa: E402
from wide_clip import wide  # noqa: E402
from batch_next3 import single, card, cap, seal, meta, cuts_of, image, IMPACT, BOLD  # noqa: E402

BL = bw.BL
PICKS = json.loads((ROOT / "edits/clip_picks.json").read_text(encoding="utf-8"))
_i = {}


def pool(who):
    """hand-picked clips first (every pick list), then the rest of the well-framed pool."""
    L = []
    for name in ("t28", "t27", "t31", "weak"):
        L += [p for p in PICKS.get(name, {}).get(who, []) if p not in L]
    d = json.loads((ROOT / BL / bw.FOLDER[who] / "clips_pool.json").read_text(encoding="utf-8-sig"))["clips"]
    L += [c["path"] for c in d if not c.get("framing") and c["brightness"] > 45 and c["path"] not in L]
    return L


def clip(who):
    L = pool(who); i = _i.get(who, 0); _i[who] = i + 1
    return {"clip": L[i % len(L)], "start": 6}


def merge(cuts, a, b, gap):
    out = [a]
    for r in cuts:
        if a + 0.01 < r < b - gap and r - out[-1] >= gap:
            out.append(r)
    return out


# ---------------------------------------------------------------- 33
SINS = [("PRIDE", "Kaiser", "MICHAEL KAISER", [170, 90, 255]), ("GREED", "Lorenzo", "DON LORENZO", [245, 195, 60]),
        ("LUST", "Shidou", "SHIDOU RYUSEI", [255, 90, 160]), ("ENVY", "Rin", "ITOSHI RIN", [60, 230, 170]),
        ("GLUTTONY", "Barou", "BAROU SHOEI", [255, 140, 40]), ("WRATH", "Isagi", "ISAGI YOICHI", [240, 40, 50]),
        ("SLOTH", "Nagi", "NAGI SEISHIRO", [90, 160, 255])]


def t33():
    R = cuts_of(33)
    segs = [card(0.0, 0.05, [0, 0, 0])]
    title = [cap("7", 0.40, 0.14, [245, 245, 250], IMPACT, stroke=0.0, glow=0.5, glow_color=[255, 255, 255]),
             cap("DEADLY SINS", 0.50, 0.085, [245, 245, 250], IMPACT, stroke=0.0, glow=0.5, glow_color=[255, 255, 255])]
    s0 = card(0.05, 1.0, [0, 0, 0]); s0["captions"] = title; s0["fx"] = ["white_in", "zoom_punch"]
    segs.append(s0)
    step = 4.0 / 7                                                # a sin every ~0.57 s, like the reference's wheel
    for i, (sin, who, _, col) in enumerate(SINS):
        a = 1.0 + i * step
        s = dict(card(a, a + step, [0, 0, 0]), **clip(who), clip_alpha=0.32, zoom=[1.08, 1.0], grade={"bw": True, "contrast": 1.2})
        s["captions"] = [cap(str(i + 1), 0.40, 0.05, col, BOLD, variation="Bold", stroke=0.0, glow=0.8, glow_color=col),
                         cap(sin.lower(), 0.47, 0.08, [245, 245, 250], BOLD, variation="Bold", glow=0.9, glow_color=col),
                         cap("7 DEADLY SINS", 0.10, 0.04, [200, 200, 210], IMPACT, stroke=0.0)]
        s["fx"] = ["rgb_in", "zoom_punch"] if i % 2 else ["glitch_in", "zoom_punch"]
        s["glitch_red"] = False
        segs.append(s)
    _i.clear()                                                    # sections start from each character's best pick
    bounds = [5.02, 7.48, 9.92, 12.37, 14.8, 17.22, 19.73, 22.2]
    fxs = [["white_in", "zoom_punch"], ["rgb_in", "mblur"], ["glitch_in", "shake"], ["radial_in", "zoom_punch"]]
    for n, ((sin, who, name, col), a0, b0) in enumerate(zip(SINS, bounds, bounds[1:])):
        caps = [cap(sin, 0.83, 0.075, col, IMPACT, stroke=0.004, glow=0.9, glow_color=col),
                cap(name, 0.885, 0.034, [245, 245, 250], BOLD, variation="Bold", glow=0.4, glow_color=col)]
        marks = merge(R, a0, b0, 0.45)
        g = {"contrast": 1.12, "sat": 1.08, "glow": 0.3, "tint": col, "tint_amt": 0.18}
        for i, (a, b) in enumerate(zip(marks, marks[1:] + [b0])):
            s = single(a, b, clip(who), g, fxs[(i + n) % 4] if i else ["white_in", "zoom_punch", "rgb_in"],
                       zoom=[1.0, 1.06] if i % 2 else [1.06, 1.0], mblur=0.03, shake_amt=0.01, glitch_red=False)
            s["captions"] = [dict(c, a=max(0.0, a0 + 0.1 - a)) for c in caps]
            if i == 0:
                s["flashes"] = [[a, a + 0.06, 0.9]]
            segs.append(s)
    return meta("t33_seven_sins", "The 7 Deadly Sins of Blue Lock", "the 7 deadly sins of blue lock",
                ["michael kaiser", "don lorenzo", "shidou ryusei", "itoshi rin", "barou shoei", "isagi yoichi", "nagi seishiro",
                 "7 deadly sins"], "MONTAGEM GLORIA (Slowed) - SUKA.", "music/template_33_song.wav", 22.2, seal(segs, 22.2),
                {"bloom": 0.3, "bloom_thr": 160, "shaky_glitch": 0.06},
                [{"file": f, "at": round(a0 + 0.06 - bw.onset(f), 3)} for f, a0 in zip(SIN_VOICE, bounds)])


# user, Oct 19 "add voicelines for 33": one short line per sin, first word just after the section's cut; every line is
# shorter than its ~2.45 s section so none overlap.  Rin = 02c "ore wa egoist" (rin-voiceline rules).
SIN_VOICE = [BL + "Micheal Kaiser/voicelines/lines/10_my_name_is_kaiser.wav",
             BL + "Don Lorenzo/voicelines/lines/01_you_re_my_killer.wav",          # lorenzo_voice.wav trimmed to the line
             BL + "Shidou Ryusei/voicelines/lines/02_are_you_an_idiot.wav",
             BL + "Rin Itoshi/voicelines/lines/02c_ore_wa_egoist.wav",
             BL + "Barou Shoei/voicelines/lines/06_remember_this_you_scrubs.wav",
             BL + "Isagi Yoichi/voicelines/lines/02_i_m_here_to_change_my_life.wav",
             BL + "Nagi Seishiro/voicelines/lines/10_that_s_more_like_it.wav"]


# ---------------------------------------------------------------- 34
def t34():
    """user notes (Oct 19): the half/strip panels had bad eye close-ups -> only hand-picked medium shots (clip_picks 't34',
    separate *_strip lists with the face in the upper-middle); 'more smooth' -> push transitions + gentle zoom, no
    glitch/rgb/shake hits; the white POV bar must touch the top edge (no gap)."""
    R = cuts_of(34)
    RIN, SAE = [60, 230, 200], [245, 120, 120]
    T = PICKS["t34"]; k_ = {}

    def pk(key):
        i = k_.get(key, 0); k_[key] = i + 1
        return {"clip": T[key][i % len(T[key])], "start": 6}
    pov = cap("POV : THE GOATED BROTHERS", 0.026, 0.026, [20, 20, 24], BOLD, variation="Bold", stroke=0.0,   # band spans y -0.004..0.056
              band={"h": 0.06, "blur": 0.0005, "alpha": 1.0, "color": [250, 250, 250]})
    warm = {"contrast": 1.1, "sat": 1.1, "glow": 0.3}
    dirs = ["left", "up", "right", "down"]

    def sm(a, b, who, i, **k):                                     # smooth shot: slides in, gentle drift
        s = single(a, b, pk(who), warm, k.pop("fx", []), zoom=[1.0, 1.05] if i % 2 else [1.05, 1.0], **k)
        s["push"] = {"dir": dirs[i % 4], "dur": min(0.22, 0.45 * (b - a))}
        return s
    segs = [single(0.0, 0.5, pk("Sae"), warm, ["fade_in"], fade_time=0.15),
            sm(0.5, 0.98, "Rin", 0), sm(0.98, 1.43, "Sae", 1)]
    for s in segs[1:]:
        s["captions"] = [pov]

    def strips(a, b, rows, caps):
        n = len(rows); g = 0.004
        top = 0.056 if caps and caps[0] is pov else 0.0             # strips start under the POV bar
        h = (1 - top) / n
        tiles = []
        for i, who in enumerate(rows):                              # strips use the uncropped 16:9 shot (whole head,
            w_, fy = wide(pk(who + "_strip")["clip"])              # not an eye-line band of the portrait crop)
            tiles.append({"clip": w_, "start": 6, "wide": True, "fy": min(0.62, fy + 0.1), "at": 0.1 * i,
                          "from": ["left", "right"][i % 2],
                          "dur": 0.22, "smear_amt": 0.05, "play": True,
                          "rect": [0, top + i * h + (g if i else 0), 1, top + (i + 1) * h - (g if i < n - 1 else 0)]})
        return {"a": a, "b": b, "layout": "tiles", "tiles": tiles, "bg": [245, 245, 245], "grade": warm, "captions": caps}
    segs.append(strips(1.43, 3.5, ["Sae", "Rin", "Sae"], [pov]))
    segs.append(strips(3.5, 5.52, ["Rin", "Sae"], [cap("ITOSHI SAE", 0.75, 0.03, [245, 245, 250], BOLD, variation="Bold",
                                                       band={"h": 0.04, "blur": 0.004, "alpha": 0.7, "color": [10, 10, 12]})]))
    rin = sm(5.52, 7.35, "Rin", 2, captions=[cap("ITOSHI RIN", 0.84, 0.045, RIN, BOLD, variation="Bold", glow=0.8, glow_color=RIN, a=0.3)])
    rin["zoom"] = [1.0, 1.1]
    segs.append(rin)
    sae = single(7.35, 9.23, pk("Sae"), warm, [], zoom=[1.1, 1.0],
                 captions=[cap("ITOSHI SAE", 0.84, 0.045, SAE, BOLD, variation="Bold", glow=0.8, glow_color=SAE, a=0.3)])
    sae["flashes"] = [[7.35, 7.45, 0.6], [7.45, 7.55, 0.25]]        # the reference's flash, softer
    segs.append(sae)
    marks = merge(R, 9.23, 16.6, 0.3)
    for i, (a, b) in enumerate(zip(marks, marks[1:] + [16.6])):
        segs.append(sm(a, b, "Rin" if i % 2 == 0 else "Sae", i, fx=["mblur"] if i % 3 == 2 else [], mblur=0.025))
    end = 17.8
    fin = dict(card(16.6, end, [0, 0, 0]), **pk("Rin"), clip_alpha=0.3, zoom=[1.1, 1.0], grade={"bw": True})
    fin["captions"] = [cap("ITOSHI", 0.5, 0.11, [120, 240, 230], IMPACT, stroke=0.0, glow=1.2, glow_color=[60, 220, 210], a=0.15)]
    fin["fx"] = ["white_in"]
    segs.append(fin)
    return meta("t34_goated_bros", "POV: The Goated Brothers | Itoshi Rin & Sae", "pov: the goated brothers",
                ["itoshi rin", "itoshi sae", "itoshi brothers"], "LOS VOLTAJE FUNK (Slowed) - Fyex",
                "music/template_34_song.wav", end, seal(segs, end), {"bloom": 0.35, "bloom_thr": 150, "shaky_glitch": 0.06})


# ---------------------------------------------------------------- 35
def t35():
    BWG = {"bw": True, "contrast": 1.4, "bright": 1.05}
    TEAL = {"tint": [40, 175, 195], "tint_amt": 0.8, "contrast": 1.2, "dots": 0.4, "glow": 0.35}
    intro = [0.0, 0.24, 0.44, 0.6, 0.8, 1.0]
    segs = [single(a, b, image("Nagi"), BWG, ["white_in"] if i == 0 else [], zoom=[1.12, 1.0])
            for i, (a, b) in enumerate(zip(intro, intro[1:] + [1.16]))]
    marks = [1.16, 1.76, 2.36, 3.0, 3.52, 5.96, 6.56, 7.2, 7.76, 8.32, 9.6, 10.2, 10.76]
    fxs = [["glitch_in"], ["radial_in"], ["rgb_in", "zoom_punch"], ["mblur"]]
    for i, (a, b) in enumerate(zip(marks, marks[1:] + [10.84])):
        s = single(a, b, clip("Nagi"), TEAL, fxs[i % 4], zoom=[1.0, 1.08] if i % 2 else [1.08, 1.0], glitch_red=False,
                   static_tint=[120, 230, 235])
        if a == 3.52:                                              # the long shot (user, Oct 19: NO letter-by-letter name)
            s["zoom"] = [1.0, 1.15]; s["fx"] = ["white_in"]
        segs.append(s)
    segs.append(card(10.84, 10.95, [250, 250, 250]))
    # user, Oct 19: "clean the audio" -> the reference rip is clipped/crushed; use the declipped copy (src/declip.py)
    return meta("t35_nagi_teal", "Nagi Seishiro | Blue Lock Manga Edit", "nagi...", ["nagi seishiro"],
                "edit audio", "music/template_35_clean.wav", 10.95, seal(segs, 10.95), {"bloom": 0.3, "bloom_thr": 150})


if __name__ == "__main__":
    for name in ARGS or ["t33", "t34", "t35"]:
        _i.clear()
        print(globals()[name]())
