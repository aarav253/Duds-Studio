"""Round-2 fixes for the week batch (user feedback, Oct 12).

    python src/batch_fix.py <scratchpad dir>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw  # noqa: E402
from voice_mix import onset  # noqa: E402
from face_still import face_still  # noqa: E402

BL = bw.BL
E = ROOT / "edits"


def song(n):
    p = ROOT / f"music/template_{n}_song.json"
    if p.exists() and json.loads(p.read_text(encoding="utf-8")).get("ok") and (ROOT / f"music/template_{n}_song.wav").exists():
        return f"music/template_{n}_song.wav"
    return None


def song_title(n):
    p = ROOT / f"music/template_{n}_song.json"
    return json.loads(p.read_text(encoding="utf-8"))["best"]["title"].split("|")[0].strip() if p.exists() else None


def clean_lines(who, maxd=2.8):
    """clean (QA-passed) voicelines, shortest first."""
    if who == "Lorenzo":
        return [BL + "Don Lorenzo/lorenzo_voice.wav"]
    idx = json.loads((ROOT / "panels/Blue Lock" / bw.FOLDER[who] / "voicelines/lines/index.json").read_text(encoding="utf-8"))
    ok = [r for r in idx if r.get("qa", {}).get("ok", True)]
    short = sorted([r for r in ok if r["duration"] <= maxd], key=lambda r: r["duration"]) or sorted(ok, key=lambda r: r["duration"])
    return [BL + bw.FOLDER[who] + "/voicelines/lines/" + r["file"] for r in short]


def vline(who, t, k=0, maxd=2.8):
    ls = clean_lines(who, maxd)
    p = ls[k % len(ls)]
    return {"file": p, "at": round(t - onset(p), 3)}


def load(mid):
    return json.loads((E / f"{mid}.json").read_text(encoding="utf-8"))


def save(m):
    (E / f"{m['id']}.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m["id"]


def snap_beats(cuts, beats, tol=0.15):
    return [min(beats, key=lambda b: abs(b - c)) if min(abs(b - c) for b in beats) < tol else c for c in cuts]


# ------------------------------------------------------------------ 4: Barou vs Shidou
def vs():
    import music_analysis as ma
    mid = "wk_barou_vs_shidou"; m = load(mid)
    sf_ = song(18)
    beats = ma.analyze(ROOT / sf_)["beats"]
    segs = m["segments"]
    starts = snap_beats([s["a"] for s in segs], beats)
    for i, s in enumerate(segs):
        s["a"] = round(starts[i], 3)
        if i:
            segs[i - 1]["b"] = s["a"]
    ORB = "assets/fonts/Orbitron.ttf"
    for s in segs:
        for c in s.get("captions", []):
            t = c["text"]
            c.update(font=ORB, variation="Black", stroke=0.0025, stroke_color=[0, 0, 0], stroke_alpha=0.8)
            if t == "VERSUS":
                c.update(size=0.06, color=[255, 60, 80], glow=1.2, glow_color=[255, 30, 60])
            elif "\n(" in t:
                c.update(size=0.032)
            elif t in ("BAROU SHOEI", "SHIDOU RYUSEI"):
                col = bw.TINT["Barou" if "BAROU" in t else "Shidou"]
                c.update(size=0.042, color=[255, 255, 255], glow=1.1, glow_color=col,
                         band={"h": 0.07, "blur": 0.01, "alpha": 0.55, "color": [10, 10, 14]})
            elif t == "WHO WINS?":
                c.update(size=0.06, glow=1.2, glow_color=[255, 60, 80])
            else:                                   # stat names
                c.update(size=0.036, color=[235, 245, 255], glow=1.0, glow_color=[90, 200, 255],
                         band={"h": 0.06, "blur": 0.008, "alpha": 0.6, "color": [8, 10, 18]})
    wins = [(s["a"], s["captions"][0]["text"]) for s in segs if s.get("captions") and s["captions"][0]["text"] in ("BAROU SHOEI", "SHIDOU RYUSEI")]
    vl, kb, ks, last = [], 0, 0, -9
    for t, name in wins:
        if t - last < 3.2:                          # never overlap; one line per ~3 s
            continue
        who = "Barou" if "BAROU" in name else "Shidou"
        vl.append(vline(who, t + 0.02, kb if who == "Barou" else ks, maxd=2.2))
        kb, ks = (kb + 1, ks) if who == "Barou" else (kb, ks + 1); last = t
        if len(vl) >= 4:
            break
    m.update(music={"file": sf_}, segments=segs, voicelines=vl)
    m["description"] = m["caption"] + f"\n\nMusic: {song_title(18)}\n\n#bluelock #barou #shidou #mangaedit #anime #shorts"
    return save(m)


# ------------------------------------------------------------------ 5: Rin best version (tiles)
def rin_tiles():
    mid = "wk_rin_t19"; m = load(mid)
    R = BL + "Rin Itoshi/"
    b = bw.beats(19)
    grid = [(0.02, 0.02, 0.49, 0.49), (0.51, 0.51, 0.98, 0.98), (0.51, 0.02, 0.98, 0.49), (0.02, 0.51, 0.49, 0.98)]
    def card(a, bnd, bg, text, clips_, col):
        beats_in = [x - a for x in b if a + 0.25 < x < bnd - 0.1][:4]
        tiles = [{"clip": c, "rect": list(grid[i]), "at": round(beats_in[i] if i < len(beats_in) else 0.3 + 0.35 * i, 3), "start": 6}
                 for i, c in enumerate(clips_)]
        return {"a": a, "b": bnd, "layout": "tiles", "bg": bg, "tiles": tiles, "fx": ["white_in"],
                "captions": [bw.cap(text, y=0.5, size=0.075, font="assets/fonts/Orbitron.ttf", variation="Black", color=list(col),
                                    stroke=0.004, stroke_color=[0, 0, 0], stroke_alpha=0.9, glow=0.8, glow_color=list(col))]}
    segs = [card(0.0, 1.92, [40, 170, 70], "CHILDHOOD", [R + "clips_2/scene_41.mp4", R + "clips_2/scene_42.mp4"], (255, 255, 255)),
            card(1.92, 3.74, [20, 40, 210], "BLUE LOCK", [R + "clips_2/scene_10.mp4", R + "clips_2/scene_02.mp4", R + "clips_2/scene_07.mp4"], (255, 255, 255)),
            {"a": 3.74, "b": 4.50, "layout": "card", "bg": [210, 25, 35], "fx": ["white_in"]},
            card(4.50, 5.98, [245, 245, 245], "U-20 JAPAN", [R + "clips_1/scene_03.mp4", R + "clips_1/scene_04.mp4", R + "clips_1/scene_14.mp4"], (20, 30, 120)),
            card(5.98, 7.46, [15, 15, 18], "P.X.G", [R + "clips_2/scene_35.mp4", R + "clips_2/scene_19.mp4", R + "clips_2/scene_34.mp4", R + "clips_2/scene_27.mp4"], (235, 40, 50))]
    post = [s for s in m["segments"] if s["a"] >= 7.46]
    m.update(segments=segs + post, music={"file": song(19)}, voicelines=[vline("Rin", 7.48, 0, 3.0)])
    m["description"] = m["caption"] + f"\n\nMusic: {song_title(19)}\n\n#bluelock #rin #mangaedit #anime #shorts"
    return save(m)


# ------------------------------------------------------------------ 10, 19: reveal edits
def kaiser_t11():
    m = load("wk_kaiser_t11")
    m["shots"][4] = {"clip": BL + "Micheal Kaiser/clips/scene_33.mp4", "start": 4, "who": "Kaiser"}   # Kaiser smiling before "brutal."
    return save(m)


def rin_edit3():
    m = load("rin_itoshi_edit_3")
    m["shots"][4] = {"clip": BL + "Rin Itoshi/clips_2/scene_07.mp4", "start": 4, "who": "Rin"}
    return save(m)


# ------------------------------------------------------------------ 11, 12, 13: clip-template edits
def tpl(n, who, intro_clip=None, zoom_face=True):
    mid = bw.tpl_edit(n, who, load(f"wk_{who.lower()}_t{n}")["caption"])
    m = load(mid)
    still = f"panels/Blue Lock/{bw.FOLDER[who]}/clips/intro_still_t{n}.png"
    src = intro_clip or bw.pool(who)[0]["path"]
    face_still(src, still, at=0.5, face_y=0.40 if zoom_face else 0.25, zoom=1.3 if zoom_face else 1.0)
    m["intro_image"] = still
    t = json.loads((ROOT / f"templates/video_edit_template_{n}.json").read_text(encoding="utf-8-sig"))
    m["voicelines"] = [vline(who, t["cuts"][0] + 0.02, 0, 3.4 if who == "Yukimiya" else 2.8)]
    if song(n):
        m["music_file"] = song(n)
    return save(m)


# ------------------------------------------------------------------ 15: Sae all in one
def sae_t15():
    m = load("wk_sae_t15")
    m["music"] = {"file": song(15)}
    post = [s["a"] for s in m["segments"] if s["a"] >= 5.04]
    times = [5.06] + [min(post, key=lambda x: abs(x - t)) for t in (8.7, 12.37)]
    m["voicelines"] = [vline("Sae", t + 0.02, k, 2.8) for k, t in enumerate(times)]
    m["description"] = m["caption"] + f"\n\nMusic: {song_title(15)}\n\n#bluelock #sae #mangaedit #anime #shorts"
    return save(m)


# ------------------------------------------------------------------ 16, 17: countdowns with a short line per rank
def countdown_voices(mid, rank_start=None):
    m = load(mid)
    segs = m["segments"]
    ranks = []
    for s in segs:
        c = s.get("captions", [])
        if c and c[0]["text"][:2].rstrip(".").isdigit():
            if not ranks or ranks[-1][1] != c[0]["text"]:
                ranks.append((s["a"], c[0]["text"]))
    name2who = {v: k for k, v in bw.FULL.items()}
    vl = []
    for a, txt in ranks:
        who = name2who.get(txt.split(". ", 1)[1])
        if who:
            vl.append(vline(who, a + 0.05, 0, 2.6))
    m["voicelines"] = vl
    return m


def top7():
    bw.AN["20"] = bw.AN["20"]                          # title shortened: ranks start on the 5th beat
    rs = round(bw.beats(20)[4], 3)
    mid = bw.countdown("wk_top7_strikers", 20, "TOP 7 STRIKERS IN\nTHE NEO EGOIST LEAGUE",
                       [("Yukimiya", "YUKIMIYA KENYU"), ("Bachira", "BACHIRA MEGURU"), ("Shidou", "SHIDOU RYUSEI"), ("Barou", "BAROU SHOEI"),
                        ("Nagi", "NAGI SEISHIRO"), ("Kaiser", "MICHAEL KAISER"), ("Isagi", "ISAGI YOICHI")],
                       rs, 21.4, 22.4, ["Isagi", "Kaiser"],
                       [[120, 200, 255], [245, 200, 60], [235, 70, 190], [225, 40, 60], [235, 235, 245], [80, 120, 255], [70, 120, 245]])
    m = countdown_voices(mid)
    m["music"] = {"file": song(20)}
    m["description"] = m["caption"] + f"\n\nMusic: {song_title(20)}\n\n#bluelock #mangaedit #anime #shorts"
    return save(m)


def coldest():
    mid = bw.countdown("wk_top5_coldest", 21, "TOP 5 COLDEST\nCHARACTERS IN BLUE LOCK",
                       [("Kaiser", "MICHAEL KAISER"), ("Barou", "BAROU SHOEI"), ("Shidou", "SHIDOU RYUSEI"), ("Rin", "ITOSHI RIN"), ("Sae", "ITOSHI SAE")],
                       3.5, 24.42, 24.9, ["Sae", "Rin"], [[90, 130, 255], [225, 40, 60], [235, 70, 190], [60, 215, 170], [255, 120, 150]])
    m = countdown_voices(mid)
    if song(21):
        m["music"] = {"file": song(21)}
    return save(m)


# ------------------------------------------------------------------ 20: Kaiser best shooters intro
def kaiser_shooters():
    m = load("kaiser_best_shooters_1")
    m["clips"][5] = bw.clips("Sae", 1, 2)[0]["clip"]          # calm close-ups that match the reaction-shot intro
    m["clips"][6] = bw.clips("Nagi", 1, 3)[0]["clip"]
    m["music_file"] = song(12)
    return save(m)


def bachira_audio():
    m = load("wk_bachira_t12"); m["music_file"] = song(12); return save(m)


if __name__ == "__main__":
    bw.VOICE["Shidou"] = ["02_are_you_an_idiot.wav", "13_i_m_going_to_dance.wav", "16_my_goal_my_will.wav"]
    bw._pool_cache.clear()
    done = [vs(), rin_tiles(), kaiser_t11(), rin_edit3(),
            tpl(2, "Yukimiya"), tpl(9, "Shidou"), tpl(1, "Barou", intro_clip=BL + "Barou Shoei/clips/scene_11.mp4", zoom_face=False),
            sae_t15(), top7(), coldest(), kaiser_shooters(), bachira_audio()]
    print("\n".join(done))
