"""Round-7 fixes (Oct 14 third feedback).   python src/batch_fix6.py <scratchpad>"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d as uf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_fix as bf     # noqa: E402
import batch_fix3 as bf3   # noqa: E402
from voice_mix import load, onset  # noqa: E402

L = bf3.L
on = lambda p, t: {"file": p, "at": round(t + 0.02 - onset(p), 3)}


# ---------------------------------------------------------------- 1: Barou vs Shidou - silent until the overall winner speaks
def vs():
    m = bf.load("wk_barou_vs_shidou")
    segs = m["segments"]
    i = next(k for k, s in enumerate(segs) if s.get("captions") and s["captions"][0]["text"] == "OVERALL")
    win = next(s for s in segs[i + 1:] if s.get("layout") == "single")      # the panel that answers the first OVERALL
    who = "Shidou" if "SHIDOU" in win["captions"][0]["text"] else "Barou"
    line = {"Shidou": "16_my_goal_my_will.wav", "Barou": "09_on_the_pitch_im_the_king.wav"}[who]
    m["voicelines"] = [on(L(who, line), win["a"])]
    return bf.save(m)


# ---------------------------------------------------------------- 2: Barou T1 - the song's own intro fade, levelled
def flat_intro(src, dst, until=2.8, max_db=9.0):
    x = load(src); sr = 44100
    e = np.sqrt(uf(x.mean(1) ** 2, int(0.6 * sr)) + 1e-12)
    ref = np.median(e[int(0.1 * sr):int(0.6 * sr)])
    g = np.clip(ref / e, 1.0, 10 ** (max_db / 20))
    n = int(until * sr); ramp = int(0.08 * sr)
    w = np.ones(len(x)); w[n:] = 0; w[n - ramp:n] = np.linspace(1, 0, ramp)
    g = 1 + (g - 1) * w
    sf.write(str(ROOT / dst), (x * g[:, None]).astype(np.float32), sr)
    return dst


def barou_t1():
    m = bf.load("wk_barou_t1")
    m["music_file"] = flat_intro("music/template_1_song.wav", "music/template_1_song_flat.wav")
    return bf.save(m)


# ---------------------------------------------------------------- 3: Top 7 - Rin says "ore wa egoist, Itoshi Rin"
def rin_short():
    """the line with its 1.7 s pause shortened to 0.2 s."""
    src = L("Rin", "02_i_m_an_egoist_i_don_t_know.wav"); dst = L("Rin", "02b_ore_wa_egoist_itoshi_rin.wav")
    x = load(src); sr = 44100
    a, b = x[:int(1.80 * sr)], x[int(3.45 * sr):]
    sf.write(str(ROOT / dst), np.concatenate([a, np.zeros((int(0.2 * sr), 2), np.float32), b]), sr)
    return dst


def top7():
    m = bf.load("wk_top7_strikers_all_voices")
    m["id"] = "wk_top7_strikers"
    p = rin_short()
    vl = [v for v in m["voicelines"] if "Rin Itoshi" not in v["file"] and "Micheal Kaiser" not in v["file"]]
    vl.append(on(p, 15.82))
    rin_end = 15.82 + sf.info(str(ROOT / p)).duration - onset(p)
    k = L("Kaiser", "10_my_name_is_kaiser.wav")
    vl.append(on(k, max(18.08, rin_end + 0.25)))
    m["voicelines"] = sorted(vl, key=lambda v: v["at"])
    return bf.save(m)


# ---------------------------------------------------------------- 5: Kaiser shooters - intro text stays until Kaiser pops in
def kaiser_shooters():
    m = bf.load("kaiser_best_shooters_1")
    m["caption_times"] = {"intro": [0.0, 5.12]}
    return bf.save(m)


if __name__ == "__main__":
    print("\n".join([vs(), barou_t1(), top7(), kaiser_shooters()]))
