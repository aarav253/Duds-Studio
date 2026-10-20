"""Kaiser - Glow (own edit) built around the manga -> colour reveal.

Song: the Glow edit audio (template 26), 70 bpm, a beat every ~0.86 s.
  0     - 3.77   slow pans over Kaiser shown as pure manga (ink + screentone), never revealed
  3.77  - 4.59   "You / You make / You make me" flickering over the manga
  4.59  - 5.39   blue static into the drop
  5.39  - 17.17  one shot per beat: each starts as the manga panel and wipes to colour on the half-beat;
                 downbeats get a white flash + zoom punch, MICHAEL KAISER on the first coloured shot
  17.17 - 18.40  last shot holds, blurs out, standard DUDS outro

    python src/batch_glow.py   -> edits/kaiser_glow_v2.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw  # noqa: E402

BL = bw.BL
SERIF = "C:/Windows/Fonts/georgiai.ttf"
BOLD = "C:/Windows/Fonts/bahnschrift.ttf"


def kaiser():
    d = json.loads((ROOT / BL / "Micheal Kaiser/clips_pool.json").read_text(encoding="utf-8-sig"))["clips"]
    return [c["path"] for c in d if not c.get("framing") and c["brightness"] > 45]


def build():
    K = kaiser()
    k = iter(K * 3)
    shot = lambda: {"clip": next(k), "start": 8}
    col = {"contrast": 1.08, "glow": 0.3}
    ink_only = {"at": 1.01, "wipe": 0.01}
    segs = [{"a": 0.0, "b": 0.25, "layout": "card", "bg": [0, 0, 0]}]
    intro = [0.25, 1.20, 2.06, 2.92]
    for i, (a, b) in enumerate(zip(intro, intro[1:] + [3.77])):
        segs.append(dict(shot(), a=a, b=b, layout="single", grade={"contrast": 1.05}, zoom=[1.0, 1.1],
                         pan=[0.04 if i % 2 else -0.04, 0], manga_reveal=ink_only, fx=["fade_in"] if i == 0 else ["radial_in"],
                         fade_time=0.25))
    you = {"font": SERIF, "size": 0.065, "y": 0.5, "color": [245, 245, 245], "stroke": 0.002, "stroke_color": [0, 0, 0], "glow": 0.5}
    p = shot()
    for (a, txt, panel), b in zip([(3.77, "You", 0), (3.85, "You make", 1), (4.09, "You make", 0), (4.15, "You make me", 1),
                                   (4.40, "You make me", 0)], [3.85, 4.09, 4.15, 4.40, 4.59]):
        s = {"a": a, "b": b, "captions": [dict(you, text=txt)]}
        s.update(dict(p, layout="single", grade={"contrast": 1.05}, zoom=[1.05, 1.05], manga_reveal=ink_only) if panel else
                 {"layout": "card", "bg": [4, 4, 6], "stars": True})
        segs.append(s)
    segs.append({"a": 4.59, "b": 5.39, "layout": "card", "bg": [40, 60, 160], "fx": ["static"], "static_tint": [90, 120, 255]})
    cuts = [5.39, 6.25, 7.11, 7.97, 8.83, 9.69, 10.55, 11.41, 12.27, 13.13, 13.99, 14.85, 15.71, 16.57]
    downs = {5.39, 8.06, 11.41, 14.85}
    name = {"text": "MICHAEL KAISER", "y": 0.78, "font": BOLD, "variation": "Bold", "size": 0.05, "color": [120, 160, 255],
            "stroke": 0.003, "stroke_color": [0, 0, 0], "glow": 0.8, "glow_color": [80, 120, 255], "a": 0.45}
    for i, (a, b) in enumerate(zip(cuts, cuts[1:] + [17.17])):
        s = dict(shot(), a=a, b=b, layout="single", grade=dict(col), zoom=[1.1, 1.0] if i % 2 else [1.0, 1.08],
                 manga_reveal={"at": 0.5, "wipe": 0.12}, fx=["shake"] if i % 3 == 1 else [], shake_amt=0.008)
        if any(abs(a - d) < 0.12 for d in downs):
            s["fx"] = ["white_in", "zoom_punch"]
        if i == 0:
            s["captions"] = [name]
        segs.append(s)
    segs.append(dict(shot(), a=17.17, b=18.40, layout="single", grade=dict(col), zoom=[1.0, 1.1], fx=["end_blur"],
                     manga_reveal={"at": 0.3, "wipe": 0.12}))
    m = {"id": "kaiser_glow_v2", "title": "Michael Kaiser Edit | Blue Lock", "caption": "you make me glow", "watermark": "DUDS",
         "end_text": "DUDS", "size": [1080, 1350], "fps": 60,
         "description": "you make me glow\n\nMusic: Mr.Kitty - Glow\n\n#bluelock #kaiser #michaelkaiser #mangaedit #anime #shorts",
         "tags": ["blue lock", "manga edit", "anime edit", "shorts", "bluelock", "michael kaiser", "kaiser"],
         "music": {"file": "music/template_26_ref.wav"}, "body_end": 18.40, "segments": segs, "voicelines": [],
         "look": {"bloom": 0.3, "bloom_thr": 160, "shaky_glitch": 0.06}}
    (ROOT / "edits" / f"{m['id']}.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m["id"]


if __name__ == "__main__":
    print(build())
