"""Round-8 fixes (Oct 14 fourth feedback).   python src/batch_fix7.py <scratchpad>"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_fix as bf     # noqa: E402
import batch_fix3 as bf3   # noqa: E402
from voice_mix import load, onset  # noqa: E402

L = bf3.L
on = lambda p, t: {"file": p, "at": round(t + 0.02 - onset(p), 3)}
WINNERS = ("BAROU SHOEI", "SHIDOU RYUSEI")


# ---------------------------------------------------------------- 1: Barou vs Shidou - every category exactly 2 beats
def vs():
    import music_analysis as ma
    m = bf.load("wk_barou_vs_shidou")
    segs = m["segments"]
    beats = [b for b in ma.analyze(ROOT / m["music"]["file"])["beats"] if b >= segs[1]["a"] - 0.05]
    intro, body, end = segs[0], segs[1:-1], segs[-1]                 # VERSUS intro / categories / WHO WINS?
    cats, k = [], 0
    while k < len(body):                                              # (category panel, winner panel); a category with no winner is dropped
        if k + 1 < len(body) and body[k + 1].get("captions", [{}])[0].get("text") in WINNERS:
            cats.append((body[k], body[k + 1])); k += 2
        else:
            k += 1
    out = [intro]
    for i, (c, w) in enumerate(cats):
        c["a"], c["b"] = round(beats[2 * i], 3), round(beats[2 * i + 1], 3)
        w["a"], w["b"] = round(beats[2 * i + 1], 3), round(beats[2 * i + 2], 3)
        out += [c, w]
    intro["b"] = out[1]["a"]
    end["a"] = out[-1]["b"]
    m["segments"] = out + [end]
    final = out[-1]
    who = "Shidou" if "SHIDOU" in final["captions"][0]["text"] else "Barou"
    line = {"Shidou": "16_my_goal_my_will.wav", "Barou": "09_on_the_pitch_im_the_king.wav"}[who]
    m["voicelines"] = [on(L(who, line), final["a"])]
    return bf.save(m)


# ---------------------------------------------------------------- 2: Barou T1 - the template's own audio
def barou_t1():
    m = bf.load("wk_barou_t1")
    m["music_file"] = "music/template_1_ref.wav"
    return bf.save(m)


# ---------------------------------------------------------------- 3: Top 7 - Rin's line cut to "ore wa egoist"
def top7():
    m = bf.load("wk_top7_strikers")
    src = L("Rin", "02_i_m_an_egoist_i_don_t_know.wav"); dst = L("Rin", "02c_ore_wa_egoist.wav")
    x = load(src); sr = 44100
    a = x[:int(1.85 * sr)]
    a[-int(0.08 * sr):] *= np.linspace(1, 0, int(0.08 * sr))[:, None]
    sf.write(str(ROOT / dst), a, sr)
    vl = [v for v in m["voicelines"] if "Rin Itoshi" not in v["file"] and "Micheal Kaiser" not in v["file"]]
    vl += [on(dst, 15.82), on(L("Kaiser", "10_my_name_is_kaiser.wav"), 18.08)]
    m["voicelines"] = sorted(vl, key=lambda v: v["at"])
    return bf.save(m)


if __name__ == "__main__":
    print("\n".join([vs(), barou_t1(), top7()]))
