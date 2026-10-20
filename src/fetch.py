"""Download a YouTube video's picture, sound, or both (yt-dlp), at the best
quality YouTube serves, straight into the project.

    python src/fetch.py <url>                         # video + audio -> downloads/
    python src/fetch.py <url> --audio                 # audio only -> music/  (WAV, plus the original stream)
    python src/fetch.py <url> --video                 # picture only (no sound)
    python src/fetch.py <url> --to "panels/Shidou Ryusei"
    python src/fetch.py <url> --start 0:26.7 --end 1:13   # just that part
    python src/fetch.py <url> --name teni_teni --max-height 2160

Video is H.264 MP4 when YouTube has it at the best resolution (it plays and
edits everywhere); otherwise VP9/AV1 is converted to H.264 at high quality.
Audio is YouTube's best stream (Opus ~160 kbps or AAC ~128 kbps), kept as-is
and also decoded to a WAV for the edit engines.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import yt_dlp
from yt_dlp.utils import download_range_func

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _secs(t):
    if t is None:
        return None
    parts = [float(p) for p in str(t).split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def _safe(name):
    name = re.sub(r"[^\w\-. ()]+", "", name).strip()
    return re.sub(r"\s+", " ", name)[:90] or "download"


def _codec(path):
    """(video codec, audio codec) via ffmpeg's stream listing."""
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stderr
    v = re.search(r"Video: (\w+)", err)
    a = re.search(r"Audio: (\w+)", err)
    return (v.group(1) if v else None), (a.group(1) if a else None)


def fetch(url, mode="both", to=None, start=None, end=None, name=None, max_height=1080):
    out_dir = ROOT / (to or ("music" if mode == "audio" else "downloads"))
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "ffmpeg_location": FFMPEG,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "restrictfilenames": False,
        "windowsfilenames": True,
        "outtmpl": str(out_dir / ((_safe(name) if name else "%(title).80s") + ".%(ext)s")),
    }
    if mode == "audio":
        opts["format"] = "bestaudio/best"
    else:
        # best resolution up to max_height; among equals prefer H.264 + AAC (edit-friendly)
        opts["format_sort"] = [f"res:{max_height}", "fps", "vcodec:h264", "acodec:aac"]
        opts["format"] = "bv*" if mode == "video" else "bv*+ba/b"
        opts["merge_output_format"] = "mp4"
    s, e = _secs(start), _secs(end)
    if s is not None or e is not None:
        opts["download_ranges"] = download_range_func(None, [(s or 0, e if e is not None else float("inf"))])
        opts["force_keyframes_at_cuts"] = True
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():                                   # merger may have changed the extension
        cands = sorted(path.parent.glob(path.stem + ".*"), key=lambda p: p.stat().st_mtime)
        path = cands[-1] if cands else path
    res = {"title": info.get("title"), "channel": info.get("channel"), "url": info.get("webpage_url", url),
           "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}
    if mode == "audio":
        wav = path.with_suffix(".wav")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(path), "-ac", "2", "-ar", "44100", str(wav)], check=True)
        res["wav"] = str(wav.relative_to(ROOT))
        res["audio_codec"] = _codec(path)[1]
        res["abr_kbps"] = info.get("abr")
    else:
        vc, ac = _codec(path)
        if vc and vc != "h264":                             # VP9/AV1: convert so every tool can read it
            h264 = path.with_name(path.stem + ".h264.mp4")
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-crf", "14",
                   "-preset", "slow", "-pix_fmt", "yuv420p"]
            cmd += ["-c:a", "aac", "-b:a", "256k"] if ac else ["-an"]
            subprocess.run(cmd + [str(h264)], check=True)
            path.unlink()
            h264.rename(path.with_suffix(".mp4"))
            path = path.with_suffix(".mp4")
            res["file"] = str(path.relative_to(ROOT))
            res["converted_from"] = vc
        res.update(width=info.get("width"), height=info.get("height"), fps=info.get("fps"),
                   video_codec=vc, audio_codec=ac)
    (path.parent / (path.stem + ".source.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download from YouTube into the project.")
    ap.add_argument("url")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--audio", action="store_true", help="audio only (WAV + original stream)")
    g.add_argument("--video", action="store_true", help="picture only, no sound")
    ap.add_argument("--to", help="folder inside the project (default downloads/ or music/)")
    ap.add_argument("--start", help="start time, e.g. 0:26.7")
    ap.add_argument("--end", help="end time, e.g. 1:13")
    ap.add_argument("--name", help="file name without extension")
    ap.add_argument("--max-height", type=int, default=1080, help="max video height (default 1080; 2160 for 4K)")
    a = ap.parse_args()
    mode = "audio" if a.audio else "video" if a.video else "both"
    r = fetch(a.url, mode, a.to, a.start, a.end, a.name, a.max_height)
    print(json.dumps(r, indent=1, ensure_ascii=False))
