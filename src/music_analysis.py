"""Music timing analysis for cut placement.

- beats / downbeats: beat_this (neural beat + downbeat tracker, CPJKU)
- drum hits: onsets detected on the Demucs-separated DRUM stem, so cuts land
  on real percussive hits instead of vocal/synth transients in the full mix
- drop: biggest sustained energy jump in drums+bass, snapped to the nearest
  downbeat (drops land on the "1" of a bar)

Results are cached in .cache/music/<hash>.json (Demucs takes ~1s per second of
audio on CPU, so each track is only separated once). If torch / the models
fail for any reason, falls back to plain librosa analysis of the full mix.

    python src/music_analysis.py music/track.mp3
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "music"
VERSION = 3  # bump to invalidate cached analyses


def _file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _as_wav(path):
    """librosa can't decode .m4a/.aac on Windows without an audio backend:
    convert with the bundled ffmpeg (cached next to the analysis cache)."""
    p = Path(path)
    if p.suffix.lower() in (".wav", ".flac", ".ogg", ".mp3"):
        return p
    import subprocess
    import imageio_ffmpeg
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{_file_hash(p)}.wav"
    if not out.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(p),
                        "-ac", "2", "-ar", "44100", str(out)], check=True)
    return out


def _load(path, sr, mono=True):
    import librosa
    y, _ = librosa.load(str(_as_wav(path)), sr=sr, mono=mono)
    return y


# --------------------------------------------------------------------------
# Neural pieces
# --------------------------------------------------------------------------

_BEAT_MODEL = None
_DEMUCS = None


def _beats_neural(path):
    global _BEAT_MODEL
    from beat_this.inference import File2Beats
    if _BEAT_MODEL is None:
        _BEAT_MODEL = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = _BEAT_MODEL(str(_as_wav(path)))
    return [float(b) for b in beats], [float(d) for d in downbeats]


def _stems(path):
    """Returns (sr, {"drums": mono, "bass": mono, "other": mono, "vocals": mono})."""
    global _DEMUCS
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    if _DEMUCS is None:
        _DEMUCS = get_model("htdemucs")
        _DEMUCS.eval()
    sr = _DEMUCS.samplerate
    y = _load(path, sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    x = torch.tensor(y, dtype=torch.float32)[None]
    ref = x.mean(1)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(_DEMUCS, (x - mean) / std, device="cpu", progress=False)[0]
    out = out * std + mean
    return sr, {name: out[i].mean(0).numpy() for i, name in enumerate(_DEMUCS.sources)}


# --------------------------------------------------------------------------
# Signal helpers
# --------------------------------------------------------------------------

def _onsets(y, sr):
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, backtrack=True, units="frames")
    times = librosa.frames_to_time(frames, sr=sr)
    peak = env.max() or 1.0
    # backtracked onsets sit at the energy minimum BEFORE the hit, so take the
    # envelope peak over the next ~0.1s as the hit's strength
    win = max(2, int(0.1 * sr / 512))
    strength = [float(env[f:f + win].max() / peak) if f < len(env) else 0.0 for f in frames]
    return [float(t) for t in times], strength


def _energy_curve(y, sr, hop=512, smooth_s=0.25):
    import librosa
    from scipy.ndimage import uniform_filter1d
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms = uniform_filter1d(rms, size=max(1, int(smooth_s * sr / hop)))
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    return times, rms


def _find_drop(y, sr, downbeats, min_start=1.5, window=0.75):
    """Biggest sustained energy jump (mean over `window` after vs before),
    snapped to the nearest downbeat within 0.25s."""
    times, rms = _energy_curve(y, sr)
    hop_t = times[1] - times[0] if len(times) > 1 else 0.023
    w = max(1, int(window / hop_t))
    if len(rms) < 2 * w + 2:
        return None, 0.0
    c = np.concatenate([[0], np.cumsum(rms)])
    idx = np.arange(w, len(rms) - w)
    before = (c[idx] - c[idx - w]) / w
    after = (c[idx + w] - c[idx]) / w
    score = (after - before) / (rms.max() + 1e-9)
    score[times[idx] < min_start] = -np.inf
    best = int(np.argmax(score))
    if not np.isfinite(score[best]) or score[best] < 0.2:
        return None, float(max(score[best], 0)) if np.isfinite(score[best]) else 0.0
    t = float(times[idx[best]])
    if downbeats:
        near = min(downbeats, key=lambda d: abs(d - t))
        if abs(near - t) <= 0.25:
            t = near
    return t, float(score[best])


def _sections(y, sr, beats, n_bars=4):
    """Coarse loudness per 4-bar block (helps spot build-ups vs drops)."""
    times, rms = _energy_curve(y, sr)
    out = []
    if len(beats) < 2:
        return out
    step = 4 * n_bars
    for i in range(0, len(beats), step):
        a = beats[i]
        b = beats[i + step] if i + step < len(beats) else times[-1]
        if b - a < 0.5:
            continue
        m = rms[(times >= a) & (times < b)]
        out.append({"start": round(a, 3), "end": round(float(b), 3),
                    "energy": round(float(m.mean() / (rms.max() + 1e-9)), 3) if len(m) else 0.0})
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def analyze(path, use_neural=True, verbose=False):
    """Returns dict: duration, tempo, beats, downbeats, drum_onsets
    [{t, strength}], drop, drop_score, sections, method."""
    path = Path(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{_file_hash(path)}_v{VERSION}_{'nn' if use_neural else 'lr'}"
    cache_file = CACHE / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8-sig"))

    import librosa
    result = None
    if use_neural:
        try:
            if verbose:
                print(f"  [music] beat_this + demucs on {path.name} ...")
            beats, downbeats = _beats_neural(path)
            sr, stems = _stems(path)
            drums = stems["drums"]
            on_t, on_s = _onsets(drums, sr)
            drop, drop_score = _find_drop(drums + stems["bass"], sr, downbeats)
            mix = sum(stems.values())
            result = {"method": "beat_this+demucs", "beats": beats, "downbeats": downbeats,
                      "drum_onsets": [{"t": round(t, 4), "strength": round(s, 3)} for t, s in zip(on_t, on_s)],
                      "drop": drop, "drop_score": round(drop_score, 3),
                      "duration": round(len(mix) / sr, 3), "sections": _sections(mix, sr, beats)}
        except Exception as e:  # noqa: BLE001 - any failure -> librosa fallback
            print(f"  [music] neural analysis failed ({type(e).__name__}: {e}); using librosa")
    if result is None:
        sr = 22050
        y = _load(path, sr)
        _, bframes = librosa.beat.beat_track(y=y, sr=sr)
        beats = [float(t) for t in librosa.frames_to_time(bframes, sr=sr)]
        downbeats = beats[::4]
        on_t, on_s = _onsets(y, sr)
        drop, drop_score = _find_drop(y, sr, downbeats)
        result = {"method": "librosa", "beats": beats, "downbeats": downbeats,
                  "drum_onsets": [{"t": round(t, 4), "strength": round(s, 3)} for t, s in zip(on_t, on_s)],
                  "drop": drop, "drop_score": round(drop_score, 3),
                  "duration": round(len(y) / sr, 3), "sections": _sections(y, sr, beats)}
    b = result["beats"]
    # slope of a line through all beats: far more precise than one median gap
    # (beat_this reports on a 20ms grid, so single gaps are quantized)
    result["tempo"] = round(60.0 / float(np.polyfit(np.arange(len(b)), b, 1)[0]), 1) if len(b) > 2 else None
    cache_file.write_text(json.dumps(result), encoding="utf-8")
    return result


def strong_onsets(analysis, min_strength=0.25):
    return [o["t"] for o in analysis["drum_onsets"] if o["strength"] >= min_strength]


if __name__ == "__main__":
    a = analyze(sys.argv[1], verbose=True)
    print(f"method={a['method']} tempo={a['tempo']} duration={a['duration']}s")
    print(f"beats={len(a['beats'])} downbeats={len(a['downbeats'])} drum_onsets={len(a['drum_onsets'])}")
    print(f"drop={a['drop']} (score {a['drop_score']})")
    for s in a["sections"]:
        print(f"  {s['start']:7.2f}-{s['end']:7.2f}  energy {'#' * int(s['energy'] * 30)}")

