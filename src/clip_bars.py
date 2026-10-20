"""Letterbox-bar detection and removal for 1080x1350 character clips.

    debar(clip) crops bars seen on ANY frame (bars that slide in during a
    Twixtor zoom included) and cover-fits back; the original goes to
    <clips>/_with_bars/.
"""
import shutil
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FF = imageio_ffmpeg.get_ffmpeg_exe()


def _bars(f, thr=22):
    g = f.max(2)
    r = np.where((g > thr).mean(1) > 0.02)[0]; c = np.where((g > thr).mean(0) > 0.02)[0]
    if not len(r) or not len(c):
        return 0, 0, 0, 0
    return int(r[0]), int(f.shape[0] - 1 - r[-1]), int(c[0]), int(f.shape[1] - 1 - c[-1])


def _worst_bars(frames):
    """largest bar on each side over every frame (bars that animate in a Twixtor zoom included)."""
    b = np.array([_bars(f) for f in frames])
    return [int(v) + 2 if v > 4 else 0 for v in b.max(0)]


def debar(clip):
    """crop bars that appear on any frame of a 4:5 clip and cover-fit back; original kept in _with_bars/."""
    p = ROOT / clip
    c = cv2.VideoCapture(str(p)); fps = c.get(5); frames = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        frames.append(f)
    c.release()
    t, b, l, r = _worst_bars(frames)
    if max(t, b, l, r) == 0:
        return clip, "clean"
    H, W = frames[0].shape[:2]
    if H - t - b < 0.5 * H or W - l - r < 0.5 * W:              # mostly-black clip (fade/disclaimer), not bars
        return clip, "skipped (too dark to measure)"
    keep = p.parent / "_with_bars" / p.name
    keep.parent.mkdir(exist_ok=True)
    if not keep.exists():
        shutil.copy2(p, keep)
    H, W = frames[0].shape[:2]
    tmp = p.with_suffix(".tmp.mp4")
    vf = f"crop={W - l - r}:{H - t - b}:{l}:{t},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(keep), "-vf", vf, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-r", str(fps), "-an", str(tmp)], check=True)
    tmp.replace(p)
    return clip, f"debarred t{t} b{b} l{l} r{r}"
