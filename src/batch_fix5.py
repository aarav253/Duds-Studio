"""Round-6 fixes (Oct 14 second feedback).   python src/batch_fix5.py <scratchpad>"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_week as bw    # noqa: E402
import batch_fix as bf     # noqa: E402
import batch_fix3 as bf3   # noqa: E402
from voice_mix import onset  # noqa: E402

BL = bw.BL
L = bf3.L
on = lambda p, t: {"file": p, "at": round(t + 0.02 - onset(p), 3)}


def win_starts(m, name):
    return [s["a"] for s in m["segments"] if s.get("layout") == "single" and s.get("captions") and s["captions"][0]["text"] == name]


# ---------------------------------------------------------------- 1: Barou vs Shidou - a Barou line on most Barou wins
def vs():
    m = bf.load("wk_barou_vs_shidou")
    w = win_starts(m, "BAROU SHOEI")                       # 5.8, 8.12, 11.22, 14.3, 16.405, 19.72, 22.581, 25.12
    plan = [(0, "05_listen_up.wav"), (1, "06_remember_this_you_scrubs.wav"), (2, "09_on_the_pitch_im_the_king.wav"),
            (4, "03_what_isnt_it_obvious.wav"), (5, "07_to_me_the_ball_isnt_a_friend.wav"), (6, "02_ore_yell.wav")]
    files = {p.split("/")[-1][:20]: p for p in bf.clean_lines("Barou", 9)}
    m["voicelines"] = [on(files[f[:20]], w[i]) for i, f in plan]
    return bf.save(m)


# ---------------------------------------------------------------- 2: Barou T1 - shot at 14 s was a dark half-face
def barou_t1():
    m = bf.load("wk_barou_t1")
    m["shots"][8] = {"clip": BL + "Barou Shoei/clips/scene_10.mp4", "clip_start": 45}
    return bf.save(m)


# ---------------------------------------------------------------- 3: Top 7 - one copy with every character's line, one with none
def top7_variants():
    base = bf.load("wk_top7_strikers")
    ranks = []
    for s in base["segments"]:
        c = s.get("captions", [])
        if c and c[0]["text"][:1].isdigit() and (not ranks or ranks[-1][1] != c[0]["text"]):
            ranks.append((s["a"], c[0]["text"]))
    pick = {"NAGI SEISHIRO": L("Nagi", "08_i_ll_crush_you.wav"), "BACHIRA MEGURU": L("Bachira", "07_this_is_the_beginning_of_english.wav"),
            "SHIDOU RYUSEI": L("Shidou", "02_are_you_an_idiot.wav"), "BAROU SHOEI": L("Barou", "06_remember_this_you_scrubs.wav"),
            "ISAGI YOICHI": L("Isagi", "02_i_m_here_to_change_my_life.wav"), "ITOSHI RIN": L("Rin", "01_this_is_me.wav"),
            "MICHAEL KAISER": L("Kaiser", "10_my_name_is_kaiser.wav")}
    out = []
    for suffix, voiced in (("all_voices", True), ("no_voices", False)):
        m = copy.deepcopy(base)
        m["id"] = f"wk_top7_strikers_{suffix}"
        m["voicelines"] = [on(pick[txt.split(". ", 1)[1]], a) for a, txt in ranks] if voiced else []
        out.append(bf.save(m))
    return "\n".join(out)


# ---------------------------------------------------------------- 5: Kaiser shooters - Yukimiya standing in place of Sae
def kaiser_shooters():
    m = bf.load("kaiser_best_shooters_1")
    m["clips"][5] = sys.argv[2] if len(sys.argv) > 2 else BL + "Yukimiya Kenyu/clips/scene_02.mp4"
    return bf.save(m)


if __name__ == "__main__":
    print("\n".join([vs(), barou_t1(), top7_variants(), kaiser_shooters()]))
