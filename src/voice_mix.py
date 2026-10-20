"""Voicelines over music, shared by every engine.

add_voices(music, voicelines, sr) -> mix  : pedalboard cleanup + compression on
each line, music ducked under the voice, voice levelled ~3 dB over the ducked
music. finish(mix, sr, body_s) -> mix   : -14 LUFS + peak limiting + fade.

Also a CLI that re-mixes voicelines into an already rendered video (used for the
template_edit clip edits whose renderer only muxes the song):

    python src/voice_mix.py <video.mp4> <manifest.json>     (manifest needs "voicelines")
"""
import json
import subprocess
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d as mf, uniform_filter1d as uf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent


def _p(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load(path, sr=44100):
    x, s = sf.read(str(_p(path)), always_2d=True)
    if s != sr:
        g = gcd(sr, s); x = resample_poly(x, sr // g, s // g, axis=0)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    return x.astype(np.float32)


def onset(path):
    """seconds of silence before the first word of a voiceline file."""
    x = load(path); e = uf(np.abs(x).max(1), 441)
    return float(np.argmax(e > 0.08 * e.max()) / 44100)


def add_voices(music, voicelines, sr=44100, duck_db=7.0, voice_over_music=2.8):
    from pedalboard import Pedalboard, HighpassFilter, Compressor, PeakFilter
    board = Pedalboard([HighpassFilter(90), PeakFilter(3200, 2.5, 1.0), Compressor(-18, 3.0, 5, 120)])
    n = len(music)
    voice = np.zeros_like(music)
    for v in voicelines:
        x = board(load(v["file"], sr).T, sr).T * 10 ** (v.get("gain_db", 0) / 20)
        i = int(v["at"] * sr); j = min(n, i + len(x))
        if j > i:
            voice[i:j] += x[:j - i]
    if np.abs(voice).max() == 0:
        return music
    act = (uf(np.abs(voice).max(1), int(0.03 * sr)) > 0.01).astype(np.float32)
    act = uf(mf(act, int(0.25 * sr)), int(0.12 * sr))
    music = music * (1 - act * (1 - 10 ** (-duck_db / 20)))[:, None]
    vr = np.sqrt(np.mean(voice[act > 0.5] ** 2) + 1e-9); mr = np.sqrt(np.mean(music[act > 0.5] ** 2) + 1e-9)
    return music + voice * min(10.0, mr * voice_over_music / vr)


def finish(mix, sr, body_s, lufs=-14.0, fade=None):
    import pyloudnorm as pyln
    meter = pyln.Meter(sr); body = int(body_s * sr)
    for _ in range(3):
        mix = pyln.normalize.loudness(mix, meter.integrated_loudness(mix[:body]), lufs)
        env = mf(np.abs(mix).max(1), int(0.01 * sr))
        g = np.minimum(1.0, 0.94 / np.maximum(env, 1e-6))
        mix = mix * np.minimum(g, uf(g, int(0.02 * sr)))[:, None]
    if fade:
        a = int(fade[0] * sr); mix[a:] *= (np.cos(np.linspace(0, np.pi / 2, len(mix) - a)) ** 1.5)[:, None]
    return mix.astype(np.float32)


if __name__ == "__main__":
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vid = _p(sys.argv[1]); m = json.loads(_p(sys.argv[2]).read_text(encoding="utf-8-sig"))
    tmp = vid.with_suffix(".orig.wav")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(vid), "-vn", "-ac", "2", "-ar", "44100", str(tmp)], check=True)
    music = load(tmp)
    mix = add_voices(music, m["voicelines"], duck_db=m.get("duck_db", 4.0), voice_over_music=m.get("voice_over_music", 1.4))
    mix = finish(mix, 44100, len(mix) / 44100 - 0.8)
    sf.write(str(tmp), mix, 44100)
    out = vid.with_name(vid.stem + ".voiced.mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(vid), "-i", str(tmp), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "320k", "-shortest", str(out)], check=True)
    tmp.unlink(); out.replace(vid)
    print("voiced", vid.name)
