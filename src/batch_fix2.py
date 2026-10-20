"""Round-3 fixes (Oct 13 feedback).   python src/batch_fix2.py <scratchpad>"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw   # noqa: E402
import batch_fix as bf    # noqa: E402
from voice_mix import onset  # noqa: E402

BL = bw.BL
_face = {}


def face(clip):
    """(face centre y 0-1, face height 0-1) on the middle frame, or None."""
    if clip not in _face:
        from imgutils.detect import detect_faces
        c = cv2.VideoCapture(str(ROOT / clip)); n = int(c.get(7)); c.set(1, n // 2); ok, f = c.read()
        r = None
        if ok:
            fs = detect_faces(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            if fs:
                (x0, y0, x1, y1), _, sc = max(fs, key=lambda d: d[2] * (d[0][3] - d[0][1]))
                r = ((y0 + y1) / 2 / f.shape[0], (y1 - y0) / f.shape[0], (x0 + x1) / 2 / f.shape[1])
        _face[clip] = r
    return _face[clip]


def faced(who):
    """pool clips with a detected face that fits a half-screen window, best first."""
    good, rest = [], []
    for c in bw.pool(who):
        r = face(c["path"])
        if r and 0.06 < r[1] < 0.7:
            good.append((c["path"], r))
        else:
            rest.append((c["path"], (0.42, 0.3, 0.5)))       # no clear face: centre-ish window
    return good + rest                                       # every clip, faces first -> a fresh panel per category


def music_beats(path):
    import music_analysis as ma
    a = ma.analyze(ROOT / path)
    return sorted(set([round(b, 3) for b in a["beats"]] + [round(o["t"], 3) for o in a["drum_onsets"] if o["strength"] >= 0.55])), a


def snap_segments(segs, grid, tol=0.12, keep=()):
    for i, s in enumerate(segs):
        if i == 0 or i in keep:
            continue
        near = min(grid, key=lambda g: abs(g - s["a"]))
        if abs(near - s["a"]) < tol and near > segs[i - 1]["a"] + 0.1:
            s["a"] = near
            segs[i - 1]["b"] = near
    return segs


def lines_fitting(who, room):
    return [p for p in bf.clean_lines(who, 9.0) if bw_dur(p) <= room]


def bw_dur(p):
    import soundfile as sf
    i = sf.info(str(ROOT / p)); return i.frames / i.samplerate


def vl_at(p, t):
    return {"file": p, "at": round(t - onset(p), 3)}


# ------------------------------------------------------------------ 1: Barou vs Shidou
def vs():
    m = bf.load("wk_barou_vs_shidou")
    A, B = faced("Barou"), faced("Shidou")
    ia = ib = 0
    segs = m["segments"]
    for i, s in enumerate(segs):
        if s.get("layout") == "versus":
            (ca, fa), (cb, fb_) = A[ia % len(A)], B[ib % len(B)]
            s["clips"] = [ca, cb]; s["fy"] = [round(fa[0], 3), round(fb_[0], 3)]; s["half_zoom"] = 1.0
            s["_pair"] = (ia, ib); ia += 1; ib += 1
        elif s.get("layout") == "single" and s.get("captions") and s["captions"][0]["text"] in ("BAROU SHOEI", "SHIDOU RYUSEI"):
            who = "Barou" if "BAROU" in s["captions"][0]["text"] else "Shidou"
            pool = A if who == "Barou" else B
            k = (ia if who == "Barou" else ib)                    # the winner's own next panel - the set for this category
            c, f_ = pool[k % len(pool)]
            s.update(clip=c, start=6, focus=[0.5, round(f_[0], 3)], zoom=[1.06, 1.0])
            if who == "Barou": ia += 1
            else: ib += 1
    for s in segs:
        s.pop("_pair", None)
    grid, _ = music_beats(m["music"]["file"])
    segs = snap_segments(segs, grid)
    wins = [(s["a"], s["b"], "Barou" if "BAROU" in s["captions"][0]["text"] else "Shidou") for s in segs
            if s.get("captions") and s["captions"][0]["text"] in ("BAROU SHOEI", "SHIDOU RYUSEI")]
    vl, used, last = [], {"Barou": 0, "Shidou": 0}, -9
    for a, b, who in wins:
        if a - last < 3.0:
            continue
        ls = lines_fitting(who, 3.8)
        if used[who] >= len(ls):
            continue
        p = ls[used[who]]; used[who] += 1
        vl.append(vl_at(p, a + 0.02)); last = a + bw_dur(p)
    m.update(segments=segs, voicelines=vl)
    return bf.save(m)


# ------------------------------------------------------------------ 2: Rin cards with 4 pictures
def rin_tiles():
    m = bf.load("wk_rin_t19")
    R = BL + "Rin Itoshi/"
    eras = {"CHILDHOOD": [(R + "clips_2/scene_41.mp4", 20), (R + "clips_2/scene_42.mp4", 30), (R + "clips_2/scene_41.mp4", 80), (R + "clips_2/scene_43.mp4", 40)],
            "BLUE LOCK": [(R + "clips_2/scene_10.mp4", 40), (R + "clips_2/scene_02.mp4", 40), (R + "clips_2/scene_22.mp4", 40), (R + "clips_2/scene_07.mp4", 40)],
            "U-20 JAPAN": [(R + "clips_1/scene_03.mp4", 40), (R + "clips_1/scene_04.mp4", 40), (R + "clips_1/scene_14.mp4", 40), (R + "clips_1/scene_07.mp4", 40)],
            "P.X.G": [(R + "clips_2/scene_35.mp4", 40), (R + "clips_2/scene_19.mp4", 40), (R + "clips_2/scene_34.mp4", 40), (R + "clips_2/scene_27.mp4", 40)]}
    b = bw.beats(19)
    for s in m["segments"]:
        if s.get("layout") != "tiles":
            continue
        name = s["captions"][0]["text"]
        inside = [x - s["a"] for x in b if s["a"] + 0.15 < x < s["b"] - 0.05]
        step = (s["b"] - s["a"] - 0.2) / 4
        ats = inside[:4] if len(inside) >= 4 else [0.1 + step * i for i in range(4)]
        for tl, (c, fr), at in zip(s["tiles"] + [dict(s["tiles"][0]) for _ in range(4 - len(s["tiles"]))], eras[name], ats):
            tl.update(clip=c, start=fr, still=True, at=round(at, 3))
        s["tiles"] = (s["tiles"] + [dict(s["tiles"][0]) for _ in range(4)])[:4]
        grid = [(0.02, 0.02, 0.49, 0.49), (0.51, 0.51, 0.98, 0.98), (0.51, 0.02, 0.98, 0.49), (0.02, 0.51, 0.49, 0.98)]
        for tl, (c, fr), at, g in zip(s["tiles"], eras[name], ats, grid):
            tl.update(clip=c, start=fr, still=True, at=round(at, 3), rect=list(g))
    return bf.save(m)


# ------------------------------------------------------------------ 3: Bachira - new Rin / Shidou intro panels
def bachira():
    m = bf.load("wk_bachira_t12")
    m["clips"][1] = BL + "Rin Itoshi/clips_2/scene_06.mp4"
    shid = [c for c, r in faced("Shidou")]
    m["clips"][6] = shid[1] if len(shid) > 1 else shid[0]
    return bf.save(m)


# ------------------------------------------------------------------ 4: Kaiser T11 - more voicelines on bars
def kaiser_t11():
    m = bf.load("wk_kaiser_t11")
    gr = lambda k: round(0.06 + 0.515 * k, 3)
    ls = lines_fitting("Kaiser", 2.6)
    times = [8.263, gr(21), gr(25), gr(29), gr(33), gr(41)]
    m["voicelines"] = [vl_at(ls[i % len(ls)], t) for i, t in enumerate(times)]
    return bf.save(m)


# ------------------------------------------------------------------ 5/6: template_edit voicelines
def tpl_voices(mid, who, n, count, room=3.6, gap=3.6):
    m = bf.load(mid)
    t = json.loads((ROOT / f"templates/video_edit_template_{n}.json").read_text(encoding="utf-8-sig"))
    cuts = t["cuts"]
    ls = lines_fitting(who, room)
    vl, last, k = [], -9, 0
    for c in cuts:
        if c - last < gap:
            continue
        p = ls[k % len(ls)]; k += 1
        vl.append(vl_at(p, c + 0.02)); last = c
        if len(vl) >= count:
            break
    m["voicelines"] = vl
    return m


def shidou_t9():
    return bf.save(tpl_voices("wk_shidou_t9", "Shidou", 9, 3, 5.4))


def barou_t1():
    from face_still import face_still
    m = tpl_voices("wk_barou_t1", "Barou", 1, 3, 3.0, 2.6)
    face_still(BL + "Barou Shoei/clips/scene_11.mp4", "panels/Blue Lock/Barou Shoei/clips/intro_still_t1.png", at=0.5, face_y=0.30, zoom=1.35)
    return bf.save(m)


# ------------------------------------------------------------------ 7: Sae - tighter beat sync
def sae():
    m = bf.load("wk_sae_t15")
    grid, a = music_beats(m["music"]["file"])
    drop = min(a["downbeats"], key=lambda d: abs(d - 5.04))
    segs = m["segments"]
    i_drop = next(i for i, s in enumerate(segs) if s["a"] >= 5.0)
    segs[i_drop]["a"] = drop; segs[i_drop - 1]["b"] = drop
    segs = snap_segments(segs, grid, keep={i_drop})
    m["segments"] = segs
    m["voicelines"][0]["at"] = round(drop + 0.02 - onset(m["voicelines"][0]["file"]), 3)
    return bf.save(m)


# ------------------------------------------------------------------ 8: Top 7 - ranks start on the drop, lines that fit
def top7():
    path = "music/template_20_song.wav"
    grid, a = music_beats(path)
    drop = round(min(a["downbeats"], key=lambda d: abs(d - a["drop"])), 3)
    bw.AN["20"]["cuts"] = sorted(set(bw.AN["20"]["cuts"]))
    mid = bw.countdown("wk_top7_strikers", 20, "TOP 7 STRIKERS IN\nTHE NEO EGOIST LEAGUE",
                       [("Yukimiya", "YUKIMIYA KENYU"), ("Bachira", "BACHIRA MEGURU"), ("Shidou", "SHIDOU RYUSEI"), ("Barou", "BAROU SHOEI"),
                        ("Nagi", "NAGI SEISHIRO"), ("Kaiser", "MICHAEL KAISER"), ("Isagi", "ISAGI YOICHI")],
                       drop, 21.4, 22.4, ["Isagi", "Kaiser"],
                       [[120, 200, 255], [245, 200, 60], [235, 70, 190], [225, 40, 60], [235, 235, 245], [80, 120, 255], [70, 120, 245]])
    m = bf.load(mid)
    m["segments"] = snap_segments(m["segments"], grid)
    ranks = []
    for s in m["segments"]:
        c = s.get("captions", [])
        if c and c[0]["text"][:1].isdigit() and (not ranks or ranks[-1][1] != c[0]["text"]):
            ranks.append([s["a"], c[0]["text"]])
    name2who = {v: k for k, v in bw.FULL.items()}
    vl = []
    for i, (a0, txt) in enumerate(ranks):
        nxt = ranks[i + 1][0] if i + 1 < len(ranks) else 22.4
        who = name2who.get(txt.split(". ", 1)[1])
        ls = lines_fitting(who, nxt - a0 - 0.15) if who else []
        if ls:
            vl.append(vl_at(ls[0], a0 + 0.03))
    m.update(music={"file": path}, voicelines=vl, duck_db=7.0, voice_over_music=2.8)
    m["description"] = m["caption"] + f"\n\nMusic: {bf.song_title(20)}\n\n#bluelock #mangaedit #anime #shorts"
    return bf.save(m)


if __name__ == "__main__":
    bw._pool_cache.clear()
    print("\n".join([vs(), rin_tiles(), bachira(), kaiser_t11(), shidou_t9(), barou_t1(), sae(), top7()]))
