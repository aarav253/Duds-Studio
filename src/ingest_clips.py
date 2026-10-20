"""Add clip videos to a character: download -> scene split -> face crops ->
bar trim -> rebuild the verified pool.

    python src/ingest_clips.py "<Character folder>" <tag> <url> [<url> ...]
Video n goes to panels/Blue Lock/<Character>/<tag><n>.mp4 and its scenes to
clips_<tag><n>/. Steps already done are skipped, so a stopped run can resume.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import fetch  # noqa: E402

PY = sys.executable


def run(*a):
    subprocess.run([PY, *a], check=True, cwd=ROOT)


def ingest(char, tag, urls):
    d = ROOT / "panels/Blue Lock" / char
    for n, url in enumerate(urls, 1):
        name = f"{tag}{n}"
        vid = d / f"{name}.mp4"
        if not vid.exists():
            fetch(url, to=str(d.relative_to(ROOT)), name=name)
        cd = d / f"clips_{name}"
        if not (cd / "scenes.json").exists():
            run("src/scene_split.py", str(vid.relative_to(ROOT)), f"clips_{name}")
        S = json.loads((cd / "scenes.json").read_text(encoding="utf-8"))
        todo = [s["i"] for s in S["scenes"] if not (cd / f"scene_{s['i']:02d}.mp4").exists()]
        if todo:                                               # resumes: only scenes without a clip yet
            run("src/clip_crops.py", str((cd / "scenes.json").relative_to(ROOT)), ",".join(map(str, todo)))
        from clip_bars import debar
        for c in sorted(cd.glob("scene_*.mp4")):
            debar(str(c.relative_to(ROOT)).replace("\\", "/"))
        print(char, name, len(list(cd.glob("scene_*.mp4"))), "clips", flush=True)
    run("src/char_pool.py", char)
    pool = json.loads((d / "clips_pool.json").read_text(encoding="utf-8"))["clips"]
    print(char, "pool", len(pool), flush=True)


if __name__ == "__main__":
    ingest(sys.argv[1], sys.argv[2], sys.argv[3:])
