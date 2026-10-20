"""Round-4 fixes (Oct 13 second feedback).   python src/batch_fix3.py <scratchpad>"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw   # noqa: E402
import batch_fix as bf    # noqa: E402
import batch_fix2 as bf2  # noqa: E402
from voice_mix import onset  # noqa: E402

BL = bw.BL
L = lambda who, f: BL + bw.FOLDER[who] + "/voicelines/lines/" + f
dur = lambda p: sf.info(str(ROOT / p)).duration


def greedy_lines(strong, order, end, slack=0.1, maxn=3):
    """first word of each line on a strong transition; never overlapping."""
    vl, last, k = [], -9, 0
    for t in strong:
        if k >= len(order) or len(vl) >= maxn:
            break
        if t + slack < last:
            continue
        p = order[k]
        if t + dur(p) - onset(p) > end + 0.3:
            continue
        vl.append({"file": p, "at": round(t + 0.02 - onset(p), 3)}); last = t + dur(p) - onset(p); k += 1
    return vl


# ---------------------------------------------------------------- 1: Barou vs Shidou, 6 panels
def vs():
    m = bf.load("wk_barou_vs_shidou")
    B, S = BL + "Barou Shoei/clips/scene_%02d.mp4", BL + "Shidou Ryusei/clips/scene_%02d.mp4"
    pair1, pair2, full = (B % 16, S % 20), (B % 14, S % 19), {"Barou": B % 12, "Shidou": S % 1}
    fy = lambda c: round((bf2.face(c) or (0.42,))[0], 3)
    first = True
    for s in m["segments"]:
        if s.get("layout") == "versus":
            pr = pair1 if first else pair2; first = False
            s.update(clips=list(pr), fy=[fy(pr[0]), fy(pr[1])], start=6, half_zoom=1.0)
        elif s.get("captions") and s["captions"][0]["text"] in ("BAROU SHOEI", "SHIDOU RYUSEI"):
            who = "Barou" if "BAROU" in s["captions"][0]["text"] else "Shidou"
            s.update(clip=full[who], start=6, focus=[0.5, fy(full[who])], zoom=[1.05, 1.0])
        elif s.get("layout") == "single":                         # e.g. the WHO WINS? end shot
            s.update(layout="versus", clips=list(pair2), fy=[fy(pair2[0]), fy(pair2[1])], start=6)
    return bf.save(m)


# ---------------------------------------------------------------- 2: Rin cards - 4 sliding B&W panels
def rin_tiles():
    import music_analysis as ma
    m = bf.load("wk_rin_t19")
    R = BL + "Rin Itoshi/"
    a = ma.analyze(ROOT / m["music"]["file"])
    ons = sorted([o["t"] for o in a["drum_onsets"] if o["strength"] >= 0.3] + a["beats"])
    snap = lambda t: min(ons, key=lambda x: abs(x - t)) if min(abs(x - t) for x in ons) < 0.08 else t
    eras = [("CHILDHOOD", [40, 170, 70], (255, 255, 255), [R + "clips_2/scene_41.mp4", R + "clips_2/scene_42.mp4", R + "clips_2/scene_43.mp4", BL + "Sae Itoshi/clips/scene_03.mp4"]),
            ("BLUE LOCK", [20, 40, 210], (255, 255, 255), [R + "clips_2/scene_10.mp4", R + "clips_2/scene_02.mp4", R + "clips_2/scene_22.mp4", R + "clips_2/scene_07.mp4"]),
            ("U-20 JAPAN", [245, 245, 245], (20, 30, 120), [R + "clips_1/scene_03.mp4", R + "clips_1/scene_04.mp4", R + "clips_1/scene_14.mp4", R + "clips_1/scene_09.mp4"]),
            ("P.X.G", [200, 20, 30], (255, 255, 255), [R + "clips_2/scene_35.mp4", R + "clips_2/scene_19.mp4", R + "clips_2/scene_34.mp4", R + "clips_2/scene_27.mp4"])]
    phases = [(0.0, 1.80), (1.95, 3.68), (3.82, 5.56), (5.70, 7.30)]
    quads = [((0, 0, .5, .5), "top"), ((.5, 0, 1, .5), "right"), ((.5, .5, 1, 1), "bottom"), ((0, .5, .5, 1), "left")]
    land = [0.38, 0.62, 0.88, 1.12]                     # measured in the reference, seconds after each card appears
    import cv2
    segs = []
    for (name, bg, col, clips), (a0, b0), nxt in zip(eras, phases, [1.95, 3.82, 5.70, 7.46]):
        tiles = []
        for (rect, frm), c, l in zip(quads, clips, land):
            n = int(cv2.VideoCapture(str(ROOT / c)).get(7))
            tiles.append({"clip": c, "rect": list(rect), "from": frm, "at": round(snap(a0 + l) - a0 - 0.14, 3), "dur": 0.16,
                          "start": n // 2, "still": True})
        segs.append({"a": a0, "b": b0, "layout": "tiles", "bg": bg, "tiles": tiles, "tile_grade": {"bw": True, "contrast": 1.25, "bright": 1.05},
                     "captions": [bw.cap(name, y=0.5, size=0.075, font="assets/fonts/Orbitron.ttf", variation="Black", color=list(col),
                                         stroke=0.004, stroke_color=[0, 0, 0], stroke_alpha=0.9, glow=0.8, glow_color=list(col),
                                         band={"h": 0.09, "blur": 0.004, "alpha": 0.9, "color": bg})]})
        segs.append({"a": b0, "b": nxt, "layout": "card", "bg": [210, 25, 35], "fx": ["white_in"]})
    m["segments"] = segs + [s for s in m["segments"] if s["a"] >= 7.46]
    return bf.save(m)


# ---------------------------------------------------------------- 3: Bachira - front-facing Rin / Shidou
def bachira():
    m = bf.load("wk_bachira_t12")
    m["clips"][5] = BL + "Rin Itoshi/clips_2/scene_11.mp4"
    m["clips"][6] = BL + "Shidou Ryusei/clips/scene_02.mp4"
    return bf.save(m)


# ---------------------------------------------------------------- 4: Kaiser T11 - 3 lines
def kaiser_t11():
    m = bf.load("wk_kaiser_t11")
    gr = lambda k: round(0.06 + 0.515 * k, 3)
    order = [L("Kaiser", "10_my_name_is_kaiser.wav"), L("Kaiser", "08_it_s_a_dream.wav"), L("Kaiser", "15_yoichi.wav")]
    m["voicelines"] = greedy_lines([8.263, gr(25), gr(37)], order, 24.8)
    return bf.save(m)


# ---------------------------------------------------------------- 5: Shidou T9 - lines on main transition + popup landings
def shidou_t9():
    m = bf.load("wk_shidou_t9")
    t = json.loads((ROOT / "templates/video_edit_template_9.json").read_text(encoding="utf-8-sig"))
    strong = [t["cuts"][0]] + [t["cuts"][c["src"]] for c in t["cutouts"]]
    order = [L("Shidou", "02_are_you_an_idiot.wav"), L("Shidou", "16_my_goal_my_will.wav"), L("Shidou", "05_i_will_do_my_best_to_take_the_goal.wav")]
    m["voicelines"] = greedy_lines(strong, order, t["duration"] - 0.6)
    return bf.save(m)


# ---------------------------------------------------------------- 6: Barou T1 - lines on main transition + card drops
def barou_t1():
    m = bf.load("wk_barou_t1")
    t = json.loads((ROOT / "templates/video_edit_template_1.json").read_text(encoding="utf-8-sig"))
    strong = [t["cuts"][0]] + [t["cuts"][i] for i, s in enumerate(t["shots"]) if s["kind"] == "card"]
    order = [L("Barou", f) for f in ("05_listen_up.wav", "06_remember_this_you_scrubs.wav", "03_what_isnt_it_obvious.wav", "09_on_the_pitch_im_the_king.wav")]
    m["voicelines"] = greedy_lines(strong, order, t["duration"] - 0.6)
    return bf.save(m)


# ---------------------------------------------------------------- 7: Sae - post-drop cuts generated from the music
def music_cuts(path, start, end, min_gap=0.2, strong=0.6):
    import music_analysis as ma
    a = ma.analyze(ROOT / path)
    cuts = [b for b in a["beats"] if start - 0.01 <= b < end - 0.2]
    for o in a["drum_onsets"]:
        if o["strength"] >= strong and start < o["t"] < end - 0.2 and min(abs(o["t"] - c) for c in cuts) >= min_gap:
            cuts.append(o["t"])
    return sorted(round(c, 3) for c in cuts), a


def sae():
    m = bf.load("wk_sae_t15")
    pre = [s for s in m["segments"] if s["a"] < 5.0]
    drop = [s for s in m["segments"] if s["a"] >= 5.0][0]["a"]
    cuts, _ = music_cuts(m["music"]["file"], drop, 16.2)
    post = bw.post_segments(cuts, 16.2, "Sae", 1, fx=("radial_in", "rgb_in"), alt_fx=("glitch_in",),
                            extra={"grade": {"tint": [150, 60, 245], "tint_amt": 0.7, "contrast": 1.2, "glow": 0.35}})
    pre[-1]["b"] = drop
    m["segments"] = pre + post
    m["voicelines"] = greedy_lines([drop, cuts[len(cuts) // 3], cuts[2 * len(cuts) // 3]],
                                   [p for p in bf.clean_lines("Sae", 2.8)][:3], 16.2)
    return bf.save(m)


# ---------------------------------------------------------------- 8, 9: countdowns rebuilt on the music
def music_countdown(mid, song, title, ranks, palette, title_clips, rank_end, body_end, voices="all", font_y=0.62):
    import music_analysis as ma
    a = ma.analyze(ROOT / song)
    beats, downs = a["beats"], a["downbeats"]
    d0 = min(downs, key=lambda d: abs(d - a["drop"]))
    # energy per beat -> the biggest jump after the first drop is a second drop: a rank must start there
    y, sr = sf.read(str(ROOT / song), always_2d=True); y = y.mean(1)
    rms = [np.sqrt((y[int(b * sr):int(b2 * sr)] ** 2).mean() + 1e-9) for b, b2 in zip(beats, beats[1:])]
    k = len(ranks)
    cand = [d for d in downs if d0 < d < rank_end - 1.0]
    targets = [d0 + i * (rank_end - d0) / k for i in range(k)]
    bounds = [d0] + [min(cand, key=lambda d: abs(d - t)) for t in targets[1:]]
    jumps = [(rms[i] / rms[i - 1], beats[i]) for i in range(1, len(rms)) if d0 + 1.5 < beats[i] < rank_end - 1.0]
    if jumps:
        r, tj = max(jumps)
        tj = min(downs, key=lambda d: abs(d - tj))
        if r > 1.4:
            i = int(np.argmin([abs(b - tj) for b in bounds[1:]])) + 1
            bounds[i] = tj
    bounds = sorted(set(bounds)) + [rank_end]
    segs = []
    tb = [b for b in beats if b < d0][::2]
    tb = [0.0] + [b for b in tb if 0.3 < b < d0 - 0.2] + [d0]
    for i, (x, y_) in enumerate(zip(tb, tb[1:])):
        w = title_clips[i % len(title_clips)]
        segs.append({"a": round(x, 3), "b": round(y_, 3), "layout": "single", **bw.clips(w, 1, i + 3)[0], "zoom": [1.0, 1.06],
                     "fx": ["radial_in"] if i else ["fade_in"], "grade": {"bw": True, "contrast": 1.2, "bright": 0.85},
                     "captions": [bw.cap(title, y=0.45, size=0.045, variation="Bold", color=[255, 255, 255], glow=0.7, glow_color=palette[0])]})
    starts = []
    for r, ((w, label), x0, x1) in enumerate(zip(ranks, bounds, bounds[1:])):
        col = palette[r % len(palette)]
        cap_ = [bw.cap(f"{k - r}. {label}", y=font_y, size=0.042, variation="Bold", color=[255, 255, 255], glow=0.9, glow_color=col)]
        inner = [b for b in beats if x0 - 0.01 <= b < x1 - 0.15]
        inner = inner[::1] if (x1 - x0) / max(1, len(inner)) > 0.3 else inner[::2]
        for j, (u, v) in enumerate(zip(inner, inner[1:] + [x1])):
            g = {"tint": col, "tint_amt": 0.55, "contrast": 1.2, "glow": 0.3} if j % 2 == 0 else {"bw": True, "contrast": 1.25}
            segs.append({"a": round(u, 3), "b": round(v, 3), "layout": "single", **bw.clips(w, 1, r + j)[0], "zoom": [1.07, 1.0],
                         "fx": ["white_in", "glitch_in"] if j == 0 else ["radial_in"], "grade": g, "captions": cap_})
        starts.append((x0, x1, w))
    segs.append({"a": rank_end, "b": body_end, "layout": "card", "bg": [12, 8, 14], "fx": ["fade_in"], "fade_time": 0.3})
    m = bf.load(mid)
    vl = []
    for x0, x1, w in starts:
        if voices != "all" and w not in voices:
            continue
        ls = [p for p in bf.clean_lines(w, 9) if dur(p) - onset(p) < x1 - x0 - 0.1]
        if ls:
            vl.append({"file": ls[0], "at": round(x0 + 0.03 - onset(ls[0]), 3)})
    m.update(segments=segs, music={"file": song}, body_end=body_end, voicelines=vl)
    return bf.save(m)


def top7():
    return music_countdown("wk_top7_strikers", "music/template_20_song.wav", "TOP 7 STRIKERS IN\nTHE NEO EGOIST LEAGUE",
                           [("Yukimiya", "YUKIMIYA KENYU"), ("Bachira", "BACHIRA MEGURU"), ("Shidou", "SHIDOU RYUSEI"), ("Barou", "BAROU SHOEI"),
                            ("Nagi", "NAGI SEISHIRO"), ("Kaiser", "MICHAEL KAISER"), ("Isagi", "ISAGI YOICHI")],
                           [[120, 200, 255], [245, 200, 60], [235, 70, 190], [225, 40, 60], [235, 235, 245], [80, 120, 255], [70, 120, 245]],
                           ["Isagi", "Kaiser"], 21.4, 22.4, voices=["Isagi"])


def coldest():
    return music_countdown("wk_top5_coldest", "music/template_21_song.wav", "TOP 5 COLDEST\nCHARACTERS IN BLUE LOCK",
                           [("Kaiser", "MICHAEL KAISER"), ("Barou", "BAROU SHOEI"), ("Shidou", "SHIDOU RYUSEI"), ("Rin", "ITOSHI RIN"), ("Sae", "ITOSHI SAE")],
                           [[90, 130, 255], [225, 40, 60], [235, 70, 190], [60, 215, 170], [255, 120, 150]], ["Sae", "Rin"], 24.42, 24.9)


# ---------------------------------------------------------------- 10: Yukimiya - fit better, no repeated intro panel
def yukimiya():
    m = bf.load("wk_yukimiya_t2")
    used = {s["clip"] for s in m["shots"]}
    spare = [c["path"] for c in bw.pool("Yukimiya") if c["path"] not in used and not c["path"].endswith("scene_18.mp4")]
    m["shots"] = [s if not s["clip"].endswith("scene_18.mp4") else {"clip": spare.pop(0), "clip_start": 3} for s in m["shots"]]
    m["style_overrides"]["zoom_max"] = 1.12
    return bf.save(m)


# ---------------------------------------------------------------- 11, 12, 13
def shooters():
    m = bf.load("wk_top5_shooters")
    last = m["voicelines"][-1]; t = last["at"] + onset(last["file"])
    p = L("Rin", "02_i_m_an_egoist_i_don_t_know.wav")
    m["voicelines"][-1] = {"file": p, "at": round(t - onset(p), 3)}
    return bf.save(m)


def rin3():
    m = bf.load("rin_itoshi_edit_3")
    m["shots"][4] = {"clip": BL + "Rin Itoshi/clips_2/scene_24.mp4", "start": 4, "who": "Rin"}
    return bf.save(m)


def kaiser_shooters():
    m = bf.load("kaiser_best_shooters_1")
    m["clips"][5] = BL + "Sae Itoshi/clips/scene_01.mp4"
    return bf.save(m)


if __name__ == "__main__":
    bw._pool_cache.clear()
    print("\n".join([vs(), rin_tiles(), bachira(), kaiser_t11(), shidou_t9(), barou_t1(), sae(), top7(), coldest(),
                     yukimiya(), shooters(), rin3(), kaiser_shooters()]))
