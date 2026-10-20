"""Auto-generate manifests from the episode files YOU placed in clips/<Show Name>/.

    python src/generate_manifests.py "Show Name" --types edit,meme,narration
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_selector  # noqa: E402
import content_gen  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".m4v"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def rel(p):
    return str(Path(p).relative_to(ROOT)).replace("\\", "/")


def pick_music(mood):
    """Prefer a track named '<mood>_...', else any track, else None (silent)."""
    tracks = [p for p in (ROOT / "music").glob("*") if p.suffix.lower() in AUDIO_EXT]
    moody = [p for p in tracks if p.name.lower().startswith(mood.lower() + "_")]
    pool = moody or tracks
    return rel(random.choice(pool)) if pool else None


def generate_for_show(show, types=("edit", "meme", "narration")):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8-sig"))
    cs = cfg["clip_selection"]
    folder = ROOT / "clips" / show
    if not folder.is_dir():
        raise FileNotFoundError(f"put episode files in {folder}")
    episodes = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not episodes:
        raise FileNotFoundError(f"no video files in {folder}")
    written = []
    for ep in episodes:
        windows = clip_selector.find_candidate_clips(ep, cs["min_clip_seconds"], cs["max_clip_seconds"],
                                                     cs["clips_per_episode"])
        for wi, (start, end) in enumerate(windows):
            mood = "hype" if wi == 0 else "emotional"
            copy = content_gen.generate(show, mood)
            base_id = f"{_slug(show)}_{_slug(ep.stem)}_{wi + 1:02d}"
            common = {"source_clip": rel(ep), "start": round(start, 2), "end": round(end, 2),
                      "title": copy["title"], "tags": copy["tags"],
                      "description": f"{copy['title']}\n\n#shorts #{_slug(show).replace('_', '')}"}
            for t in types:
                m = {"id": f"{base_id}_{t}", "type": t, **common}
                if t in ("edit", "meme"):
                    m["music"] = pick_music(mood)
                if t == "meme":
                    m["caption"] = copy["meme_caption"]
                if t == "narration":
                    import tts
                    audio = ROOT / "narration_audio" / f"{m['id']}.wav"
                    tts.synthesize(copy["narration_script"], audio,
                                   rate_wpm=cfg["content_types"]["narration"]["tts_rate"])
                    m["script"] = copy["narration_script"]
                    m["narration_audio"] = rel(audio)
                    if cfg["music_selection"]["include_in_narration"]:
                        m["music"] = pick_music(mood)
                out = ROOT / "edits" / f"{m['id']}.json"
                out.write_text(json.dumps(m, indent=2), encoding="utf-8")
                written.append(rel(out))
                print("wrote", rel(out))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("show")
    ap.add_argument("--types", default="edit,meme,narration")
    a = ap.parse_args()
    generate_for_show(a.show, [t.strip() for t in a.types.split(",") if t.strip()])
