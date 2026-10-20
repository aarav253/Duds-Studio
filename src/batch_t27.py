"""'Who is the strongest' (template 27, Kito Kawi's Madara edit) as "who is the best Blue Lock striker":
intro Bachira, Barou, Shidou, Isagi; focus Rin.

Measured from the reference (60 fps, cut list references/template_27/cuts.json):
  0     - 0.82   hook: the focus character on a pale manga panel, POV question on a white band at the top
  0.82  - 4.17   one held colour shot per contender (0.83 s each), blur-in on the cut, slow push
  4.17  - 6.18   2x2 grid of the four with a white cross and a centre emblem, panels arrive one by one,
                 "MADE BY DUDS" in yellow
  6.18  - 6.67   flicker between the grid and the focus character
  6.67  - 14.53  the focus edit on the song's cuts: colour clips, white flashes, zoom punches, "MADE BY DUDS"
  14.53 -        standard DUDS outro
Song: MONTAGEM UNNAMED (Slowed) - Avenxir, aligned sample-exact (music/template_27_song.wav).

    python src/batch_t27.py   -> edits/t27_best_striker.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw  # noqa: E402

BL = bw.BL
FOCUS = "Rin"
INTRO = ["Bachira", "Barou", "Shidou", "Isagi"]
FONT = str(ROOT / "assets/fonts/Bangers.ttf")
_next = {}


PICKS = json.loads((ROOT / "edits/clip_picks.json").read_text(encoding="utf-8"))["t27"]


def clip(who, kind="pool"):
    """hand-picked clips (edits/clip_picks.json) first, then the verified pool (original folders first)."""
    if who in PICKS:
        i = _next.get((who, "pick"), 0); _next[(who, "pick")] = i + 1
        return {"clip": PICKS[who][i % len(PICKS[who])], "start": 6}
    d = json.loads((ROOT / BL / bw.FOLDER[who] / "clips_pool.json").read_text(encoding="utf-8-sig"))["clips"]
    ps = [c for c in d if not c.get("framing") and c["brightness"] > 45 and "/clips_x" not in c["path"]] or \
         [c for c in d if not c.get("framing") and c["brightness"] > 45]
    i = _next.get((who, kind), 0); _next[(who, kind)] = i + 1
    c = ps[i % len(ps)]
    return {"clip": c["path"], "start": min(6, c["frames"] // 4)}


def image(who, bw_=True):
    import cv2
    d = ROOT / BL / bw.FOLDER[who] / "images"
    ps = [q for q in sorted(d.glob("*.jpg")) if (cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2HSV)[..., 1].mean() < 25) == bw_] if d.exists() else []
    return {"clip": str(ps[0].relative_to(ROOT)).replace("\\", "/"), "start": 0} if ps else clip(who)


def single(a, b, spec, grade, fx=(), **k):
    z = k.pop("zoom", [1.06, 1.0])
    return dict(spec, a=round(a, 3), b=round(b, 3), layout="single", zoom=z, grade=dict(grade), fx=list(fx), **k)


def pov():
    return {"text": "POV : WHO IS THE BEST BLUE LOCK STRIKER?", "y": 0.04, "font": FONT, "size": 0.046, "color": [15, 15, 15],
            "stroke": 0.0, "band": {"h": 0.11, "blur": 0.0005, "alpha": 1.0, "color": [252, 252, 252]}}  # flush with the top edge


def made_by(y=0.72):
    return {"text": "MADE BY DUDS", "y": y, "font": FONT, "size": 0.05, "color": [255, 225, 40], "stroke": 0.003,
            "stroke_color": [0, 0, 0], "glow": 0.8, "glow_color": [255, 210, 0]}


def rin_lines():
    """Rin speaks as the edit lands on him (6.67) and again on a later bass hit (11.15)."""
    from voice_mix import onset
    L = BL + "Rin Itoshi/voicelines/lines/"
    out = []
    for f, t in (("02c_ore_wa_egoist.wav", 6.70), ("07_kill_kill_kill.wav", 11.17)):   # two different Rin lines
        out.append({"file": L + f, "at": round(t - onset(L + f), 3)})
    return out


def build():
    R = json.loads((ROOT / "references/template_27/cuts.json").read_text())
    col = {"contrast": 1.15, "sat": 1.1, "glow": 0.3}
    segs = [single(0.0, 0.817, clip("Nagi"), col, ["fade_in"], fade_time=0.12, zoom=[1.0, 1.05], captions=[pov()])]
    for (a, b), who in zip([(0.817, 1.667), (1.667, 2.483), (2.483, 3.317), (3.317, 4.167)], INTRO):
        segs.append(single(a, b, clip(who), col, ["radial_in", "white_in"], zoom=[1.0, 1.06], captions=[pov()] if b <= 3.4 else []))
    quads = [((0, 0, .5, .5), "left", 0.20), ((.5, .5, 1, 1), "right", 0.78), ((.5, 0, 1, .5), "top", 1.30), ((0, .5, .5, 1), "bottom", 1.70)]
    grid_clip = {"Bachira": "panels/Blue Lock/Bachira Meguru/clips/scene_22.mp4", "Barou": "panels/Blue Lock/Barou Shoei/clips/scene_01.mp4", "Shidou": "panels/Blue Lock/Shidou Ryusei/clips/scene_02.mp4", "Isagi": "panels/Blue Lock/Isagi Yoichi/clips/scene_04.mp4"}                         # coloured manga-edit panels
    tiles = [dict({"clip": grid_clip[w], "start": 6}, rect=list(r), at=at, dur=0.14, still=True, smear_amt=0.12, impact=0.18, impact_zoom=0.08, **{"from": frm})
             for w, (r, frm, at) in zip(INTRO, quads)]
    grid = {"a": 4.167, "b": 6.183, "layout": "tiles", "bg": [245, 245, 248], "tiles": tiles, "tile_grade": {"contrast": 1.1, "glow": 0.25},
            "fx": ["white_in"], "captions": [],
            "cutout": {"png": "assets/grid_cross_ball.png", "x": 0.5, "y": 0.5, "h": 1.0, "max_w": 1.0, "zoom": [1.0, 1.0],
                       "drift": [0, 0], "pop": 0.0001}}
    segs.append(grid)
    f1 = clip(FOCUS)                                               # flicker grid <-> focus on the reference's cuts
    for i, (a, b) in enumerate(zip([6.183, 6.267, 6.4, 6.517], [6.267, 6.4, 6.517, 6.667])):
        if i % 2 == 0:
            segs.append(single(a, b, f1, dict(col, bright=1.3), ["white_in", "glitch_in"], zoom=[1.1, 1.05]))
        else:
            segs.append(dict(grid, a=a, b=b, fx=["glitch_in"], captions=[], tiles=[dict(t, at=0.0, dur=0.001) for t in tiles]))
    # the edit: one shot per bass hit of the song (measured on music/template_27_song.wav, ~0.63 s apart)
    booms = [6.667, 7.40, 8.01, 8.66, 9.30, 9.89, 10.52, 11.15, 11.81, 12.40, 13.04, 13.70, 14.30]
    for k, (a, b) in enumerate(zip(booms, booms[1:] + [14.53])):
        fx = [["radial_in", "zoom_punch"], ["white_in", "shake"], ["rgb_in", "zoom_punch"], ["radial_in", "shake"]][k % 4]
        segs.append(single(a, b, clip(FOCUS), col, fx, zoom=[1.1, 1.0] if k % 2 else [1.0, 1.08]))
    segs.sort(key=lambda s: s["a"])
    for x, y in zip(segs, segs[1:]):
        x["b"] = y["a"]
    m = {"id": "t27_best_striker", "title": "Who Is The Best Blue Lock Striker? | Itoshi Rin", "caption": "who is the best striker?",
         "watermark": "DUDS", "end_text": "DUDS", "size": [1080, 1350], "fps": 60,
         "description": "who is the best striker?\n\nMusic: MONTAGEM UNNAMED (Slowed) - Avenxir\n\n#bluelock #rin #itoshirin #mangaedit #anime #shorts",
         "tags": ["blue lock", "manga edit", "anime edit", "shorts", "bluelock", "itoshi rin", "rin", "bachira", "barou", "shidou", "isagi"],
         "music": {"file": "music/template_27_song.wav"}, "body_end": 14.53, "segments": segs, "voicelines": rin_lines(),
         "duck_db": 7.0, "voice_over_music": 2.8,
         "look": {"bloom": 0.3, "bloom_thr": 160, "sat": 1.05, "contrast": 1.03}}
    (ROOT / "edits" / f"{m['id']}.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m["id"]


if __name__ == "__main__":
    print(build())
