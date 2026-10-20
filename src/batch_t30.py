"""'Blue Lock Special Grade' (template 30): Blue Lock players walk in by grade (3, 2, 1) as full-body official art,
then the New Generation 11 get the SPECIAL GRADE edit.

  0     - 13.9   grade intro on a training-room backdrop; each player slides in from the side and holds,
                 GRADE label at the top in the grade colour; reference audio kept as-is (its voices)
  13.9  - 35.6   SPECIAL GRADE: one New Gen 11 player per section, hand-picked clips, cut on the reference's
                 cuts (merged so no shot is under 0.15 s), flashes on the hard hits; vocals removed from the audio
  35.6 -         standard DUDS outro

    python src/batch_t30.py   -> edits/t30_special_grade.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw  # noqa: E402

BL = bw.BL
BOLD = "C:/Windows/Fonts/bahnschrift.ttf"
IMPACT = "C:/Windows/Fonts/impact.ttf"


def card_png(name):
    p = ROOT / BL / "_cards" / f"{name}.png"
    if p.exists():
        return str(p.relative_to(ROOT)).replace("\\", "/")
    return f"{BL}{name}/body/official.png"


GRADES = [("GRADE 3", [80, 220, 120], 0.30, 3.00, ["igarashi", "gagamaru", "raichi", "aryu", "tokimitsu"]),
          ("GRADE 2", [80, 220, 230], 3.00, 6.80, ["niko", "otoya", "hiori", "nanase", "karasu", "Zantetsu Tsurugi"]),
          ("GRADE 1", [235, 50, 60], 6.80, 13.90, ["chigiri", "reo", "Yukimiya Kenyu", "Barou Shoei", "Shidou Ryusei",
                                                  "Bachira Meguru", "Nagi Seishiro", "Rin Itoshi", "Isagi Yoichi"])]
NG = [("Loki", "Julian Loki"), ("Bunny", "Bunny Iglesias"), ("Hugo", "Hugo"), ("Lorenzo", "Don Lorenzo"),
      ("Teddy", "Teddy Knight"), ("Kaiser", "Micheal Kaiser"), ("Sae", "Sae Itoshi")]


def picks():
    p = json.loads((ROOT / "edits/clip_picks.json").read_text(encoding="utf-8"))
    out = {}
    for who, folder in NG:
        L = (p["weak"].get(who) or []) + (p["t31"].get(who) or [])
        d = json.loads((ROOT / BL / folder / "clips_pool.json").read_text(encoding="utf-8"))["clips"]
        L += [c["path"] for c in d if not c.get("framing") and c["brightness"] > 45 and c["path"] not in L]
        out[who] = L
    return out


def build():
    segs = [{"a": 0.0, "b": 0.30, "layout": "card", "bg": [0, 0, 0], "fx": ["fade_in"]}]
    for label, col, a0, b0, cast in GRADES:
        step = (b0 - a0) / len(cast)
        cap = {"text": label, "y": 0.06, "font": BOLD, "variation": "Bold", "size": 0.06, "color": col, "stroke": 0.004,
               "stroke_color": [0, 0, 0], "glow": 0.6, "glow_color": col}
        for i, name in enumerate(cast):
            a, b = round(a0 + i * step, 3), round(a0 + (i + 1) * step, 3)
            segs.append({"a": a, "b": b, "layout": "single", "clip": "assets/grade_room.jpg", "start": 0, "zoom": [1.0, 1.02],
                         "fx": ["glitch_in"] if i == 0 else [], "captions": [cap],
                         "cutout": {"png": card_png(name), "x": 0.5, "y": 0.56, "h": 0.78, "max_w": 0.9, "zoom": [1.0, 1.0],
                                    "drift": [0, 0], "pop": 0.0001, "enter": "right" if i % 2 else "left", "enter_dur": 0.16}})
    R = json.loads((ROOT / "references/template_30/cuts.json").read_text())
    cuts = [13.9]
    for r in R:
        if r > 13.95 and r - cuts[-1] >= 0.15:
            cuts.append(r)
    end = 35.6
    P = picks()
    span = (end - 13.9) / len(NG)
    sg = {"text": "SPECIAL GRADE", "y": 0.86, "font": IMPACT, "size": 0.05, "color": [235, 40, 50], "stroke": 0.004,
          "stroke_color": [0, 0, 0], "glow": 0.7, "glow_color": [235, 40, 50]}
    used = {}
    for i, (a, b) in enumerate(zip(cuts, cuts[1:] + [end])):
        who = NG[min(len(NG) - 1, int((a - 13.9) / span))][0]
        k = used.get(who, 0); used[who] = k + 1
        fx = [["white_in", "zoom_punch"], ["mblur", "shake"], ["rgb_in", "zoom_punch"], ["glitch_in", "shake"]][i % 4]
        segs.append({"a": round(a, 3), "b": round(b, 3), "layout": "single", "clip": P[who][k % len(P[who])], "start": 6,
                     "zoom": [1.06, 1.0] if i % 2 else [1.0, 1.05], "grade": {"contrast": 1.12, "sat": 1.08, "glow": 0.3},
                     "fx": fx, "mblur": 0.03, "shake_amt": 0.012, "captions": [sg]})
    segs.sort(key=lambda s: s["a"])
    for x, y in zip(segs, segs[1:]):
        x["b"] = y["a"]
    m = {"id": "t30_special_grade", "title": "Blue Lock Special Grade | New Generation 11", "caption": "special grade...",
         "watermark": "DUDS", "end_text": "DUDS", "size": [1080, 1350], "fps": 60,
         "description": "special grade...\n\n#bluelock #newgeneration11 #mangaedit #anime #shorts",
         "tags": ["blue lock", "manga edit", "anime edit", "shorts", "bluelock", "new generation 11", "itoshi sae", "michael kaiser"],
         "music": {"file": "music/template_30_clean.wav"}, "body_end": end, "segments": segs, "voicelines": [],
         "look": {"bloom": 0.3, "bloom_thr": 160, "shaky_glitch": 0.06}}
    (ROOT / "edits" / f"{m['id']}.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m["id"]


if __name__ == "__main__":
    print(build())
