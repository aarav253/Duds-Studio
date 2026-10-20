"""Round-5 fixes (Oct 14 feedback).   python src/batch_fix4.py <scratchpad> [clips with bars...]"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw    # noqa: E402
import batch_fix as bf     # noqa: E402
import batch_fix3 as bf3   # noqa: E402
from voice_mix import onset  # noqa: E402
from clip_bars import _worst_bars, debar  # noqa: E402,F401

BL = bw.BL
FF = imageio_ffmpeg.get_ffmpeg_exe()
L = bf3.L


def wide(folder, i, size=(1080, 672)):
    """the scene at full 16:9 width from the source video, bars removed, cover-fitted to a versus half."""
    d = ROOT / BL / folder / "clips"
    out = d / "wide" / f"scene_{i:02d}.mp4"
    if out.exists():
        return str(out.relative_to(ROOT)).replace("\\", "/")
    S = json.loads((d / "scenes.json").read_text(encoding="utf-8"))
    sc = next(s for s in S["scenes"] if s["i"] == i)
    c = cv2.VideoCapture(str(ROOT / S["source"])); fs = []
    for k in range(sc["start"], sc["end"] + 1, 3):
        c.set(1, k); ok, f = c.read()
        if ok:
            fs.append(f)
    t, b, l, r = _worst_bars(fs)
    H, W = fs[0].shape[:2]
    out.parent.mkdir(exist_ok=True)
    vf = (f"crop={W - l - r}:{H - t - b}:{l}:{t},scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase,"
          f"crop={size[0]}:{size[1]}")
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{sc['start'] / S['fps']:.4f}", "-i", str(ROOT / S["source"]),
                    "-frames:v", str(sc["end"] - sc["start"] + 1), "-vf", vf, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-an", str(out)], check=True)
    return str(out.relative_to(ROOT)).replace("\\", "/")


# ---------------------------------------------------------------- 1: versus halves = whole wide scenes, not face crops
def vs():
    m = bf.load("wk_barou_vs_shidou")
    pair1 = (wide("Barou Shoei", 2), wide("Shidou Ryusei", 7))
    pair2 = (wide("Barou Shoei", 1), wide("Shidou Ryusei", 5))
    first = True
    for s in m["segments"]:
        if s.get("layout") == "versus":
            pr = pair1 if first else pair2; first = False
            s["clips"] = list(pr); s["start"] = 6
            s.pop("fy", None); s.pop("half_zoom", None)
    return bf.save(m)


# ---------------------------------------------------------------- 6: Barou T1 - third card was an eye close-up
def barou_t1():
    m = bf.load("wk_barou_t1")
    m["shots"][11] = {"clip": BL + "Barou Shoei/clips/scene_02.mp4", "clip_start": 3}
    return bf.save(m)


# ---------------------------------------------------------------- 8: Top 7 in the right order (1st Kaiser ... 7th Nagi)
def top7():
    return bf3.music_countdown("wk_top7_strikers", "music/template_20_song.wav", "TOP 7 STRIKERS IN\nTHE NEO EGOIST LEAGUE",
                               [("Nagi", "NAGI SEISHIRO"), ("Bachira", "BACHIRA MEGURU"), ("Shidou", "SHIDOU RYUSEI"), ("Barou", "BAROU SHOEI"),
                                ("Isagi", "ISAGI YOICHI"), ("Rin", "ITOSHI RIN"), ("Kaiser", "MICHAEL KAISER")],
                               [[235, 235, 245], [245, 200, 60], [235, 70, 190], [225, 40, 60], [70, 120, 245], [60, 215, 170], [150, 90, 255]],
                               ["Kaiser", "Rin"], 21.4, 22.4, voices=["Isagi"])


# ---------------------------------------------------------------- 9: Coldest - Rin gets a line, Kaiser a different one
def coldest():
    m = bf.load("wk_top5_coldest")
    ranks = []
    for s in m["segments"]:
        c = s.get("captions", [])
        if c and c[0]["text"][:1].isdigit() and (not ranks or ranks[-1][1] != c[0]["text"]):
            ranks.append((s["a"], c[0]["text"]))
    at = {txt.split(". ", 1)[1]: a for a, txt in ranks}
    swap = {"MICHAEL KAISER": L("Kaiser", "01_damn_you_hizamazuke.wav"), "ITOSHI RIN": L("Rin", "01_this_is_me.wav")}
    vl = [v for v in m["voicelines"] if "Micheal Kaiser" not in v["file"] and "Rin Itoshi" not in v["file"]]
    for name, p in swap.items():
        vl.append({"file": p, "at": round(at[name] + 0.03 - onset(p), 3)})
    m["voicelines"] = sorted(vl, key=lambda v: v["at"])
    return bf.save(m)


# ---------------------------------------------------------------- 13: Kaiser shooters - Sae shooting instead of a portrait
def kaiser_shooters():
    m = bf.load("kaiser_best_shooters_1")
    m["clips"][5] = BL + "Sae Itoshi/clips/scene_14.mp4"
    return bf.save(m)


if __name__ == "__main__":
    bw._pool_cache.clear()
    for c in sys.argv[2:]:                                   # clips found with bars
        print(*debar(c))
    print("\n".join([vs(), barou_t1(), top7(), coldest(), kaiser_shooters()]))
