"""Clean voicelines out of a reference edit when the clean song is known.

voice = separated voice stem (demucs htdemucs_ft), then every time-frequency
bin where the (band-gain-matched) clean song is much louder than the voice
estimate is attenuated - that removes the music bleed the separator leaves
under each line. Outside the given line ranges the output is silent.

    python src/voice_extract.py <ref_mix.wav> <song.wav> <vocals_stem.wav> <out.wav> a-b,a-b,...
"""
import sys

import librosa
import numpy as np
import soundfile as sf


def extract(mix, song, voc, sr, ranges, n_fft=2048, hop=512, ramp=0.04):
    out = np.zeros_like(voc)
    for ch in range(voc.shape[1]):
        R = librosa.stft(mix[:, ch], n_fft=n_fft, hop_length=hop)
        S = librosa.stft(song[:, ch], n_fft=n_fft, hop_length=hop)
        V = librosa.stft(voc[:, ch], n_fft=n_fft, hop_length=hop)
        # per-band gain song -> reference (editor EQ / level), from voice-free frames
        vq = np.abs(V).sum(0)
        quiet = vq < np.percentile(vq, 40)
        g = np.median(np.abs(R[:, quiet]), 1) / (np.median(np.abs(S[:, quiet]), 1) + 1e-6)
        Sg = np.abs(S) * np.clip(g, 0.1, 4)[:, None]
        # soft mask: keep bins where the voice clearly stands above the song
        ratio = np.abs(V) / (Sg + 1e-6)
        M = np.clip((ratio - 0.15) / 0.6, 0.0, 1.0) ** 1.5
        out[:, ch] = librosa.istft(V * M, hop_length=hop, length=len(voc))
    gate = np.zeros(len(voc), np.float32)
    k = int(ramp * sr)
    for a, b in ranges:
        ia, ib = int(a * sr), int(b * sr)
        gate[ia:ib] = 1
        gate[max(0, ia - k):ia] = np.maximum(gate[max(0, ia - k):ia], np.linspace(0, 1, ia - max(0, ia - k)))
        gate[ib:ib + k] = np.maximum(gate[ib:ib + k], np.linspace(1, 0, len(gate[ib:ib + k])))
    return out * gate[:, None]


if __name__ == "__main__":
    mix, sr = sf.read(sys.argv[1], always_2d=True)
    song, _ = sf.read(sys.argv[2], always_2d=True)
    voc, _ = sf.read(sys.argv[3], always_2d=True)
    n = min(len(mix), len(song), len(voc))
    ranges = [tuple(map(float, r.split("-"))) for r in sys.argv[5].split(",")]
    out = extract(mix[:n], song[:n], voc[:n], sr, ranges)
    sf.write(sys.argv[4], out, sr, subtype="PCM_24")
    print("voices ->", sys.argv[4], "peak", float(np.abs(out).max()))
