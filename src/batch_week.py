"""Builds the manifests for the week's batch (Oct 12 request).

Every focus section gets a voiceline whose first word lands on the moment the
focus character takes over. Cut times come from each template's measured cuts
(snapped to the song's beats/onsets); clips come from the verified CCIP pools.

    python src/batch_week.py            -> edits/wk_*.json (+ list printed)
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_mix import onset  # noqa: E402

BL = "panels/Blue Lock/"
SP = Path(sys.argv[1]) if len(sys.argv) > 1 else None
AN = json.loads((SP / "analysis.json").read_text()) if SP else {}
BT = json.loads((SP / "beats.json").read_text()) if SP else {}
OUT = ROOT / "edits"
FOLDER = {"Rin": "Rin Itoshi", "Nagi": "Nagi Seishiro", "Isagi": "Isagi Yoichi", "Kaiser": "Micheal Kaiser", "Barou": "Barou Shoei",
          "Shidou": "Shidou Ryusei", "Sae": "Sae Itoshi", "Bachira": "Bachira Meguru", "Yukimiya": "Yukimiya Kenyu",
          "Lorenzo": "Don Lorenzo", "Teddy": "Teddy Knight", "Zantetsu": "Zantetsu Tsurugi",
          "Loki": "Julian Loki", "Bunny": "Bunny Iglesias", "Hugo": "Hugo"}
FULL = {"Rin": "ITOSHI RIN", "Nagi": "NAGI SEISHIRO", "Isagi": "ISAGI YOICHI", "Kaiser": "MICHAEL KAISER", "Barou": "BAROU SHOEI",
        "Shidou": "SHIDOU RYUSEI", "Sae": "ITOSHI SAE", "Bachira": "BACHIRA MEGURU", "Yukimiya": "YUKIMIYA KENYU", "Lorenzo": "DON LORENZO"}
TINT = {"Rin": [60, 215, 170], "Nagi": [235, 235, 245], "Isagi": [70, 120, 245], "Kaiser": [80, 120, 255], "Barou": [225, 40, 60],
        "Shidou": [235, 70, 190], "Sae": [255, 120, 150], "Bachira": [245, 200, 60], "Yukimiya": [120, 200, 255], "Lorenzo": [240, 190, 50]}
VOICE = {"Rin": ["02c_ore_wa_egoist.wav"],
         "Nagi": ["08_i_ll_crush_you.wav", "10_that_s_more_like_it.wav", "13_i_m_going_to_destroy_you.wav"],
         "Isagi": ["04_i_m_a_striker.wav", "03_i_m_here_to_become_the_best_in_the_world.wav"],
         "Kaiser": ["10_my_name_is_kaiser.wav", "06_my_body_is_worth_300_million_yen.wav"],
         "Barou": ["09_on_the_pitch_im_the_king.wav", "06_remember_this_you_scrubs.wav", "01_ill_devour_you_and_the_whole_match.wav"],
         "Shidou": ["08_i_m_the_strongest_beast_on_the_field_especially.wav", "16_my_goal_my_will.wav", "11_dragon_dragon.wav"],
         "Sae": ["13_i_don_t_have_any_value_for_you.wav", "03_this_is_the_end_of_our_dream.wav"],
         "Bachira": ["01_i_like_it_i_like_it.wav", "02_i_m_having_so_much_fun.wav"],
         "Yukimiya": ["02_number_5_yukimiya_kane.wav", "01_the_basics_of_soccer_are_one_on_one_if_you_.wav"]}

_pool_cache, _used = {}, {}


def pool(who):
    if who not in _pool_cache:
        d = json.loads((ROOT / "panels/Blue Lock" / FOLDER[who] / "clips_pool.json").read_text())
        _pool_cache[who] = [c for c in d["clips"] if c["brightness"] > 35 and not c.get("framing")]
        _used[who] = 0
    return _pool_cache[who]


def clips(who, n, offset=0):
    """n clip specs cycling the verified pool; repeats use a later part of the clip."""
    p = pool(who); out = []
    for i in range(n):
        c = p[(offset + i) % len(p)]
        rep = (offset + i) // len(p)
        start = 4 + rep * 40
        if start + 30 > c["frames"]:
            start = max(2, c["frames"] - 34)
        out.append({"clip": c["path"], "start": start})
    return out


def vpath(who, k=0):
    if who == "Lorenzo":
        return BL + "Don Lorenzo/lorenzo_voice.wav"
    return BL + FOLDER[who] + "/voicelines/lines/" + VOICE[who][k]


def voice(who, t, k=0):
    """voiceline whose first word lands at t."""
    p = vpath(who, k)
    return {"file": p, "at": round(t - onset(p), 3)}


def snap(t, cuts):
    return min(cuts, key=lambda c: abs(c - t))


def beats(n):
    return BT[str(n)]["beats"]


def cap(text, **k):
    return dict({"text": text}, **k)


def meta(id_, who, caption, title=None, song=None):
    tag = who.lower() if who else "bluelock"
    title = title or f"{FULL.get(who, who).title()} Edit | #bluelock #{tag} #mangaedit #anime #shorts"
    desc = caption + ("\n\nMusic: " + song if song else "") + f"\n\n#bluelock #{tag} #mangaedit #anime #shorts"
    return {"id": id_, "title": title, "caption": caption, "description": desc, "watermark": "DUDS", "end_text": "DUDS",
            "tags": ["blue lock", "manga edit", "anime edit", "shorts", "bluelock"] + ([FULL[who].lower(), who.lower()] if who in FULL else [])}


def save(m):
    (OUT / f"{m['id']}.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m["id"]


def post_segments(times, end, who, offset=0, fx=("radial_in",), tint_amt=0.55, extra=None, alt_fx=None):
    cs = clips(who, len(times), offset)
    segs = []
    for i, (a, b) in enumerate(zip(times, list(times[1:]) + [end])):
        s = {"a": round(a, 3), "b": round(b, 3), "layout": "single", **cs[i], "zoom": [1.07, 1.0] if i % 2 else [1.0, 1.06],
             "fx": list(alt_fx if (alt_fx and i % 3 == 2) else fx),
             "grade": {"tint": TINT[who], "tint_amt": tint_amt, "contrast": 1.12, "glow": 0.25}}
        if extra:
            s.update(extra)
        segs.append(s)
    return segs


# ------------------------------------------------------------------ edits
def t16_nagi():
    import music_analysis as ma
    song = "music/library/slava_funk.wav"
    a = ma.analyze(ROOT / song)
    drop = a["drop"]
    intro_end = 4.70
    off = round(drop - intro_end, 3)
    bt = [b - off for b in a["beats"] if b - off > intro_end - 0.01]
    period = (bt[-1] - bt[0]) / (len(bt) - 1)
    cut = [intro_end] + [b for i, b in enumerate(bt[1:], 1) if (i < 8 and i % 2 == 0) or i >= 8]
    cut = [c for c in cut if c < intro_end + 15.0]
    end = cut[-1] + period
    segs = [{"a": 0.0, "b": intro_end, "layout": "video", "src": "references/template_16/reference.mp4", "t0": 0.0, "zoom": [1.0, 1.0],
             "fx": ["fade_in"], "fade_time": 0.25}]
    segs += post_segments(cut, end, "Nagi", 0, fx=("white_in", "zoom_punch"), alt_fx=("radial_in",), tint_amt=0.35)
    segs[1]["fx"] = ["white_in", "zoom_punch", "shake"]
    m = meta("wk_nagi_mentality", "Nagi", "POV: football mentality", song="MVSTERIOUS, hxmr, yngastrobeatz - SLAVA FUNK! (Slowed)")
    m.update(size=[1080, 1350], fps=60, body_end=round(end, 3), music={"file": song, "offset": off}, segments=segs,
             voicelines=[voice("Nagi", intro_end + 0.05, 1)])
    return save(m)


def t12(who, caption):
    t = json.loads((ROOT / "templates/video_edit_template_12.json").read_text(encoding="utf-8"))
    others = [("Nagi", 0), ("Rin", 3), ("Barou", 0), ("Isagi", 2), ("Kaiser", 1), ("Rin", 6), ("Shidou", 0)]
    others = [o for o in others if o[0] != who][:7]
    while len(others) < 7:
        others.append(("Sae", len(others)))
    intro = [clips(w, 1, k)[0] for w, k in others]
    m = meta(f"wk_{who.lower()}_t12", who, caption)
    m.update(template="templates/video_edit_template_12.json", captions={"intro": caption.upper().replace("... ", "...\n", 1)},
             tint=TINT[who], popup={"clip": pool(who)[0]["path"]}, clips=intro + clips(who, 16, 1),
             voicelines=[voice(who, t["segments"][7]["a"] + 0.02, 0)])
    return save(m)


def t13(who, intro_who="Isagi"):
    t = json.loads((ROOT / "templates/video_edit_template_13.json").read_text(encoding="utf-8"))
    m = meta(f"wk_{who.lower()}_t13", who, "The real show stealer of Blue Lock")
    split = [clips(who, 1, 2)[0], clips("Nagi" if who != "Nagi" else "Rin", 1, 1)[0], clips("Barou" if who != "Barou" else "Kaiser", 1, 0)[0]]
    m.update(template="templates/video_edit_template_13.json",
             captions={"main": f"I'M THE MAIN CHARACTER OF BLUE LOCK", "split": "REAL SHOW STEALER OF\nBLUE LOCK"},
             clips=[dict(clips(intro_who, 1, 4)[0], start=20), split] + clips(who, 21, 3),
             voicelines=[voice(who, t["segments"][2]["a"] + 0.02, 0), voice(who, 13.0, 1)])
    return save(m)


def t11(who, intro):
    base = json.loads((ROOT / "edits/barou_shoei_edit_1.json").read_text(encoding="utf-8"))
    t = json.loads((ROOT / "templates/video_edit_template_11_ext.json").read_text(encoding="utf-8"))
    n_post = sum(1 for s in t["shots"] if s["role"] == "post")
    m = dict(base)
    m.update(meta(f"wk_{who.lower()}_t11", who, "there's no perfect striker", song="LXSTFFACE - MONTAGEM ROOMLOST (Slowed)"))
    m["template"] = "templates/video_edit_template_11_ext.json"
    m["shots"] = [dict(clips(w, 1, k)[0], who=w) for w, k in intro] + [dict(c, who=who) for c in clips(who, n_post, 0)]
    gr = lambda k: round(0.06 + 0.515 * k, 3)
    m["voicelines"] = [voice(who, 8.263, 0), voice(who, gr(29), 1 if len(VOICE.get(who, [])) > 1 else 0)]
    return save(m)


def tpl_edit(n, who, caption, still_from=0):
    t = json.loads((ROOT / f"templates/video_edit_template_{n}.json").read_text(encoding="utf-8-sig"))
    still = ROOT / "panels/Blue Lock" / FOLDER[who] / "clips" / "intro_still.png"
    if not still.exists():
        c = pool(who)[still_from]; cap_ = cv2.VideoCapture(str(ROOT / c["path"])); cap_.set(1, c["frames"] // 2)
        ok, f = cap_.read(); cv2.imwrite(str(still), f)
    shots = []
    for i, c in enumerate(clips(who, len(t["shots"]), 1)):
        shots.append({"clip": c["clip"], "clip_start": 3})
    if n == 9:                                                       # popup pairs continue the same clip
        for a, b in ((3, 4), (9, 10), (15, 16)):
            shots[b] = {"clip": shots[a]["clip"], "clip_start": 20}
        for a, b in ((4, 5), (10, 11), (16, 17)):
            shots[b] = {"clip": shots[a]["clip"], "clip_start": 38}
    m = meta(f"wk_{who.lower()}_t{n}", who, caption)
    m.update(type="template_edit", template=f"templates/video_edit_template_{n}.json", intro_image=str(still.relative_to(ROOT)).replace("\\", "/"),
             style_overrides={"zoom_max": 1.3, "card_from_next": True, "pop_from": "first_shot", "color": 1.0, "saturation": 1.2},
             shots=shots, voicelines=[voice(who, t["cuts"][0] + 0.02, 0)])
    return save(m)


def t14_isagi():
    c = AN["14"]["cuts"]; who, rival = "Isagi", "Rin"
    segs, intro = [], [0.0] + c[1:6]
    rc = clips(rival, len(intro), 2)
    for i, (a, b) in enumerate(zip(intro, intro[1:] + [c[6]])):
        segs.append({"a": a, "b": b, "layout": "single", **rc[i], "zoom": [1.0, 1.06], "fx": ["white_in"] if i == 0 else ["radial_in"],
                     "grade": {"tint": [140, 200, 255], "tint_amt": 0.6, "bright": 1.12, "glow": 0.6}})
    vs = [clips(rival, 1, 9)[0]["clip"], clips(who, 1, 9)[0]["clip"]]
    segs.append({"a": c[6], "b": c[8], "layout": "versus", "clips": vs, "fx": ["glitch_in", "rgb_in"], "divider_color": [255, 60, 60],
                 "grade": {"contrast": 1.2, "sat": 1.2}})
    segs += post_segments(c[8:], AN["14"]["dur"] - 0.2, who, 0, fx=("glitch_in", "rgb_in"), alt_fx=("radial_in", "shake"), tint_amt=0.0,
                          extra={"grade": {"tint": [230, 40, 50], "tint_amt": 0.55, "contrast": 1.25, "bright": 0.8, "glow": 0.3}})
    m = meta("wk_isagi_t14", who, "The demon king")
    m.update(size=[1080, 1350], fps=60, body_end=round(AN["14"]["dur"] - 0.2, 3), music={"file": "music/template_14_ref.wav"},
             segments=segs, voicelines=[voice(who, c[8] + 0.02, 0)])
    return save(m)


def t15_sae():
    who = "Sae"; b = beats(15)
    skills = [("METAVISION", "Isagi"), ("DRIBBLING", "Bachira"), ("POWER", "Barou"), ("FLAIR", "Shidou"), ("GENIUS", "Nagi"), ("PRECISION", "Rin")]
    edges = [0.0] + [round(x, 3) for x in b[1:7]]               # one beat each
    segs = []
    for i, (sk, w) in enumerate(skills):
        segs.append({"a": edges[i], "b": edges[i + 1] if i + 1 < len(edges) else 3.82, "layout": "single", **clips(w, 1, i + 2)[0],
                     "zoom": [1.0, 1.05], "fx": ["radial_in"] if i else ["fade_in"], "grade": {"bw": True, "contrast": 1.3, "bright": 1.1},
                     "captions": [cap(sk, y=0.5, size=0.045, variation="Bold")]})
    segs[-1]["b"] = 3.82
    segs.append({"a": 3.82, "b": 5.04, "layout": "single", **clips(who, 1, 0)[0], "zoom": [1.0, 1.08], "fx": ["glitch_in"],
                 "grade": {"tint": [150, 60, 245], "tint_amt": 0.7, "contrast": 1.2},
                 "captions": [cap("All in one.", y=0.5, size=0.05, variation="Bold")]})
    cuts = [x for x in AN["15"]["cuts"] if x >= 5.04]
    cuts = [5.04] + [x for x in cuts if x > 5.1 and x < 15.95]
    dedup = []
    for x in cuts:
        if not dedup or x - dedup[-1] > 0.12: dedup.append(x)
    segs += post_segments(dedup, 16.2, who, 1, fx=("radial_in", "rgb_in"), alt_fx=("glitch_in",),
                          extra={"grade": {"tint": [150, 60, 245], "tint_amt": 0.7, "contrast": 1.2, "glow": 0.35}})
    m = meta("wk_sae_t15", who, "All in one.")
    m.update(size=[1080, 1350], fps=60, body_end=16.2, music={"file": "music/template_15_ref.wav"}, segments=segs,
             voicelines=[voice(who, 5.06, 0)])
    return save(m)


def t17_lorenzo():
    who = "Lorenzo"; c = AN["17"]["cuts"]
    band = {"h": 0.075, "blur": 0.002, "alpha": 0.82, "color": [18, 18, 20]}
    segs = [{"a": 0.0, "b": 18.5, "layout": "single", **clips("Nagi", 1, 5)[0], "speed": 0.25, "zoom": [1.0, 1.1], "zoom_ease": "lin",
             "fx": ["fade_in"], "fade_time": 0.5, "grade": {"bw": True, "contrast": 1.1, "bright": 1.15},
             "captions": [cap("“ME SEEING SHE IS HAPPY”\nWITHOUT ME", y=0.5, size=0.032, band=band)]},
            {"a": 18.5, "b": c[0], "layout": "single", **clips(who, 1, 0)[0], "speed": 0.4, "zoom": [1.0, 1.05], "fx": ["fade_in"],
             "fade_time": 0.4, "grade": {"bw": True, "contrast": 1.15, "bright": 1.15},
             "captions": [cap("WHO CARES", y=0.5, size=0.036, band=band)]}]
    segs += post_segments(c, 33.2, who, 1, fx=("radial_in",), alt_fx=("glitch_in",),
                          extra={"grade": {"bw": True, "contrast": 1.3, "bright": 1.05}})
    m = meta("wk_lorenzo_t17", who, "who cares")
    m.update(size=[1080, 1350], fps=60, body_end=33.2, music={"file": "music/template_17_ref.wav"}, segments=segs,
             voicelines=[voice(who, c[0] + 0.02, 0)])
    return save(m)


def t18_barou_shidou():
    return t18_versus("Barou", "Shidou", "wk_barou_vs_shidou", "BAROU SHOEI\n(THE KING)", "SHIDOU RYUSEI\n(THE DEVIL)",
                      [255, 90, 100], [255, 120, 220], (1, 0), [255, 40, 60], [235, 70, 190])


def t18_versus(A, B, id_, label_a, label_b, col_a, col_b, vk, glow_a=None, glow_b=None, picked=False):
    """template 18 stat battle: A (top) vs B (bottom), a stat on both, then the winner's close-up.
    picked=True: hand-picked clips first (edits/clip_picks.json), each versus half framed around the upper face area."""
    c = AN["18"]["cuts"]; end = 32.2
    glow_a, glow_b = glow_a or TINT[A], glow_b or TINT[B]
    if picked:
        P = json.loads((ROOT / "edits/clip_picks.json").read_text(encoding="utf-8"))
        order = {}
        for w in (A, B):
            L = []
            for lst in P.values():
                L += [x for x in lst.get(w, []) if x not in L]
            order[w] = L + [x["path"] for x in pool(w) if x["path"] not in L]
        clips = lambda w, n, k: [{"clip": order[w][(k + j) % len(order[w])], "start": 6} for j in range(n)]  # noqa: E731
    else:
        clips = globals()["clips"]
    stats = [("OVR. STRENGTH", A), ("OVR. SPEED", B), ("INTELLIGENCE", A), ("FLAIR", B), ("EXPERIENCE", A), ("SHOOTING", B),
             ("POWER", A), ("WEAPON", B), ("PASSING", A), ("CREATIVITY", B), ("ACCURACY", A), ("EGO", B), ("KILLER INSTINCT", A),
             ("VISION", B), ("CLUTCH", A), ("CONFIDENCE", B), ("OVERALL", None)]
    name = lambda w: cap(FULL[w], y=0.5, size=0.04, variation="Bold", color=[255, 255, 255], glow=0.9, glow_color=TINT[w])
    segs = [{"a": 0.0, "b": c[0], "layout": "versus", "clips": [clips(A, 1, 3)[0]["clip"], clips(B, 1, 3)[0]["clip"]], "fx": ["fade_in", "glitch_in"],
             "fade_time": 0.4, "divider_color": [255, 255, 255],
             "captions": [cap("VERSUS", y=0.5, size=0.05, variation="Bold", color=[255, 50, 70], glow=1.0, glow_color=[255, 30, 60], a=0.3),
                          cap(label_a, y=0.18, size=0.036, color=col_a, glow=0.8, glow_color=glow_a, a=0.6),
                          cap(label_b, y=0.82, size=0.036, color=col_b, glow=0.8, glow_color=glow_b, a=0.6)]}]
    ia = ib = 0; si = 0
    pairs = list(zip(c, c[1:] + [end]))
    for i, (a, b) in enumerate(pairs):
        st, win = stats[min(si, len(stats) - 1)]
        if i % 2 == 0:                                   # stat: both on screen
            ca, cb = clips(A, 1, 4 + ia)[0], clips(B, 1, 4 + ib)[0]; ia += 1; ib += 1
            segs.append({"a": a, "b": b, "layout": "versus", "clips": [ca["clip"], cb["clip"]], "fx": ["rgb_in"],
                         "grade": {"contrast": 1.15}, "captions": [cap(st, y=0.5, size=0.038, variation="Bold", color=[245, 245, 255], glow=0.6, glow_color=[120, 160, 255])]})
        else:                                            # winner close-up
            w = win or (A if si % 2 else B)
            k = (ia if w == A else ib) + 10
            segs.append({"a": a, "b": b, "layout": "single", **clips(w, 1, k)[0], "zoom": [1.06, 1.0], "fx": ["radial_in"],
                         "grade": {"tint": TINT[w], "tint_amt": 0.55, "contrast": 1.2, "glow": 0.3}, "captions": [name(w)]})
            si += 1
    if picked:
        for s_ in segs:
            if s_["layout"] == "versus":
                s_["fy"] = [0.4, 0.4]
    segs[-1]["captions"] = [cap("WHO WINS?", y=0.5, size=0.05, variation="Bold", glow=0.9, glow_color=[255, 60, 80])]
    m = meta(id_, A, f"{A} vs {B}", title=f"{A} vs {B} | #bluelock #{A.lower()} #{B.lower()} #mangaedit #anime #shorts")
    m["tags"] += [FULL[B].lower(), B.lower()]
    m.update(size=[1080, 1350], fps=60, body_end=end, music={"file": "music/template_18_ref.wav"}, segments=segs,
             voicelines=[voice(A, c[1] + 0.02, vk[0]), voice(B, c[3] + 0.02, vk[1])])
    return save(m)


def t19_rin():
    who = "Rin"; c = AN["19"]["cuts"]
    R = BL + "Rin Itoshi/"
    def card(a, b, bg, text, clip, col=(255, 255, 255)):
        return {"a": a, "b": b, "layout": "card", "bg": bg, "clip": clip, "start": 6, "clip_alpha": 0.55, "zoom": [1.0, 1.08],
                "fx": ["white_in"], "captions": [cap(text, y=0.5, size=0.07, variation="Bold", color=list(col), glow=0.6, glow_color=list(col))]}
    segs = [card(0.0, 1.92, [40, 170, 70], "CHILDHOOD", R + "clips_2/scene_41.mp4"),
            card(1.92, 3.74, [20, 40, 210], "BLUE LOCK", R + "clips_2/scene_10.mp4"),
            {"a": 3.74, "b": 4.50, "layout": "card", "bg": [210, 25, 35], "fx": ["white_in"]},
            card(4.50, 5.98, [245, 245, 245], "U-20 JAPAN", R + "clips_1/scene_03.mp4", col=(20, 20, 30)),
            card(5.98, 7.46, [15, 15, 18], "P.X.G", R + "clips_2/scene_35.mp4", col=(235, 40, 50))]
    post = [x for x in c if 7.46 <= x < 14.6]
    ded = []
    for x in post:
        if not ded or x - ded[-1] > 0.15: ded.append(x)
    segs += post_segments(ded, 14.9, who, 2, fx=("glitch_in", "rgb_in"), alt_fx=("radial_in",),
                          extra={"grade": {"tint": [235, 35, 45], "tint_amt": 0.6, "contrast": 1.25, "glow": 0.35}})
    segs[-1]["fx"] = ["radial_in", "dark_out"]
    m = meta("wk_rin_t19", who, "The best version of Rin")
    m.update(size=[1080, 1350], fps=60, body_end=14.9, music={"file": "music/template_19_ref.wav"}, segments=segs,
             voicelines=[voice(who, 7.48, 0)])
    return save(m)


def countdown(id_, n, title, ranks, rank_start, rank_end, body_end, title_clips, palette, quotes=None, voice_rank=0):
    cuts = [x for x in AN[str(n)]["cuts"]]
    k = len(ranks); step = (rank_end - rank_start) / k
    bounds = [rank_start] + [snap(rank_start + step * i, cuts) for i in range(1, k)] + [rank_end]
    segs = []
    tc = [x for x in cuts if 0.3 < x < rank_start]
    tb = [0.0] + tc + [rank_start]
    for i, (a, b) in enumerate(zip(tb, tb[1:])):
        w = title_clips[i % len(title_clips)]
        segs.append({"a": a, "b": b, "layout": "single", **clips(w, 1, i + 3)[0], "zoom": [1.0, 1.06], "fx": ["radial_in"] if i else ["fade_in"],
                     "grade": {"bw": True, "contrast": 1.2, "bright": 0.85},
                     "captions": [cap(title, y=0.45, size=0.045, variation="Bold", color=[255, 255, 255], glow=0.7, glow_color=palette[0])]})
    vls = []
    for r, (w, label) in enumerate(ranks):
        a, b = bounds[r], bounds[r + 1]
        inner = [a] + [x for x in cuts if a + 0.3 < x < b - 0.3]
        col = palette[r % len(palette)]
        txt = f"{k - r}. {label}"
        caps_ = [cap(txt, y=0.62, size=0.042, variation="Bold", color=[255, 255, 255], glow=0.9, glow_color=col)]
        if quotes:
            caps_.append(cap(f"“{quotes[r]}”", y=0.67, size=0.026, color=[235, 235, 235]))
        for j, (x, y) in enumerate(zip(inner, inner[1:] + [b])):
            g = {"tint": col, "tint_amt": 0.55, "contrast": 1.2, "glow": 0.3} if j % 2 == 0 else {"bw": True, "contrast": 1.25}
            segs.append({"a": round(x, 3), "b": round(y, 3), "layout": "single", **clips(w, 1, r + j)[0], "zoom": [1.07, 1.0],
                         "fx": ["white_in", "glitch_in"] if j == 0 else ["radial_in"], "grade": g, "captions": caps_})
        if r == voice_rank or r == k - 1:
            if w in VOICE or w == "Lorenzo":
                vls.append(voice(w, a + 0.05, 0))
    segs.append({"a": rank_end, "b": body_end, "layout": "card", "bg": [12, 8, 14], "fx": ["fade_in"], "fade_time": 0.3})
    m = meta(id_, ranks[-1][0], title.title().replace("\n", " "), title=f"{title.title()} | #bluelock #mangaedit #anime #shorts")
    m.update(size=[1080, 1350], fps=60, body_end=body_end, music={"file": f"music/template_{n}_ref.wav"}, segments=segs, voicelines=vls)
    return save(m)


def freestyle(id_, who, song, intro_src, intro_t0, intro_len, cap1, cap2, bridge_clip_k, song_title, tint_amt=0.5, intro_grade=None):
    import music_analysis as ma
    a = ma.analyze(ROOT / song)
    drop = a["drop"]
    t_drop = intro_len + 1.2                                  # intro footage, then a 1.2 s bridge shot of the character
    off = round(drop - t_drop, 3)
    if off < 0:
        off = 0.0; t_drop = drop
    bt = [b - off for b in a["beats"] if t_drop - 0.01 <= b - off]
    period = (bt[-1] - bt[0]) / max(1, len(bt) - 1)
    cut = [t_drop] + [b for i, b in enumerate(bt[1:], 1) if (i % 2 == 0 and i < 8) or i >= 8]
    cut = [x for x in cut if x < t_drop + 14.0]
    end = round(cut[-1] + period, 3)
    segs = [{"a": 0.0, "b": intro_len, "layout": "video", "src": intro_src, "t0": intro_t0, "zoom": [1.0, 1.08], "zoom_ease": "lin",
             "fx": ["fade_in"], "fade_time": 0.4, "grade": intro_grade or {"contrast": 1.1, "sat": 0.85},
             "captions": [cap(cap1, y=0.5, size=0.045, variation="Bold", glow=0.6, glow_color=TINT[who], a=0.4)]},
            {"a": intro_len, "b": t_drop, "layout": "single", **clips(who, 1, bridge_clip_k)[0], "speed": 0.5, "zoom": [1.12, 1.0],
             "fx": ["white_in"], "grade": {"bw": True, "contrast": 1.2},
             "captions": [cap(cap2, y=0.5, size=0.045, variation="Bold", glow=0.8, glow_color=TINT[who])]}]
    segs += post_segments(cut, end, who, bridge_clip_k + 1, fx=("white_in", "zoom_punch"), alt_fx=("radial_in", "rgb_in"), tint_amt=tint_amt)
    segs[2]["fx"] = ["white_in", "zoom_punch", "shake"]
    m = meta(id_, who, cap2.replace("\n", " ").strip(".").capitalize(), song=song_title)
    m.update(size=[1080, 1350], fps=60, body_end=end, music={"file": song, "offset": off}, segments=segs,
             voicelines=[voice(who, t_drop + 0.05, 0)])
    return save(m)


if __name__ == "__main__":
    ids = [
        t16_nagi(),
        t12("Bachira", "Blue Lock's best dribblers... "),
        t17_lorenzo(),
        t18_barou_shidou(),
        t19_rin(),
        t13("Sae"),
        t11("Kaiser", [("Nagi", 0), ("Rin", 2), ("Isagi", 3), ("Barou", 1), ("Rin", 5)]),
        tpl_edit(2, "Yukimiya", "POV: you're the 1v1 king"),
        tpl_edit(9, "Shidou", "POV: the devil was let loose"),
        tpl_edit(1, "Barou", "POV: you're the king"),
        t14_isagi(),
        t15_sae(),
        countdown("wk_top7_strikers", 20, "TOP 7 STRIKERS IN\nTHE NEO EGOIST LEAGUE",
                  [("Yukimiya", "YUKIMIYA KENYU"), ("Bachira", "BACHIRA MEGURU"), ("Shidou", "SHIDOU RYUSEI"), ("Barou", "BAROU SHOEI"),
                   ("Nagi", "NAGI SEISHIRO"), ("Kaiser", "MICHAEL KAISER"), ("Isagi", "ISAGI YOICHI")],
                  5.13, 21.4, 22.4, ["Isagi", "Kaiser", "Rin"],
                  [[120, 200, 255], [245, 200, 60], [235, 70, 190], [225, 40, 60], [235, 235, 245], [80, 120, 255], [70, 120, 245]]),
        countdown("wk_top5_coldest", 21, "TOP 5 COLDEST\nCHARACTERS IN BLUE LOCK",
                  [("Kaiser", "MICHAEL KAISER"), ("Barou", "BAROU SHOEI"), ("Shidou", "SHIDOU RYUSEI"), ("Rin", "ITOSHI RIN"), ("Sae", "ITOSHI SAE")],
                  3.5, 24.42, 24.9, ["Sae", "Rin"], [[90, 130, 255], [225, 40, 60], [235, 70, 190], [60, 215, 170], [255, 120, 150]]),
        countdown("wk_top5_backstories", 23, "TOP 5 SADDEST\nBACKSTORIES",
                  [("Nagi", "NAGI SEISHIRO"), ("Bachira", "BACHIRA MEGURU"), ("Yukimiya", "YUKIMIYA KENYU"), ("Rin", "ITOSHI RIN"), ("Kaiser", "MICHAEL KAISER")],
                  3.92, 22.6, 23.4, ["Isagi", "Rin"], [[220, 40, 50], [235, 235, 240], [220, 40, 50], [235, 235, 240], [220, 40, 50]],
                  quotes=["nothing ever felt worth the effort", "the monster was my only friend", "racing the dark before it wins",
                          "forever chasing his brother's back", "never wanted. never loved."]),
        freestyle("wk_free_nagi", "Nagi", "music/library/close_eyes.wav", "downloads/intros/cat_yawn.mp4", 11.0, 4.2,
                  "THE LAZIEST GUY\nON THE PITCH...", "...IS THE BIGGEST\nTALENT IN BLUE LOCK", 2, "DVRST - Close Eyes", tint_amt=0.3),
        freestyle("wk_free_barou", "Barou", "music/library/murder_in_my_mind.wav", "downloads/intros/chess_king_standing.mp4", 6.0, 5.0,
                  "EVERYONE FALLS...", "...ONLY THE KING\nSTAYS STANDING", 3, "KORDHELL - MURDER IN MY MIND",
                  intro_grade={"contrast": 1.2, "sat": 0.3, "tint": [225, 40, 60], "tint_amt": 0.25}),
        freestyle("wk_free_rin", "Rin", "music/library/aura.wav", "downloads/intros/mirror_shatter.mp4", 1.0, 3.6,
                  "HE DOESN'T PLAY\nFOOTBALL...", "...HE DESTROYS IT", 4, "Ogryzek - AURA (Slowed)",
                  intro_grade={"contrast": 1.2, "tint": [60, 215, 170], "tint_amt": 0.35}),
    ]
    print("\n".join(ids))
