"""Upload built videos to YouTube as PRIVATE. Never public - see publish.py.

    python src/upload.py            # every entry with status "built"
    python src/upload.py <key>      # one manifest id
"""
import json
import sys
from pathlib import Path

import yaml
from googleapiclient.http import MediaFileUpload

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_queue as rq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Private-first is a hard safety gate, not a preference: refuse to run otherwise.
    assert cfg["youtube"]["privacy_on_upload"] == "private", \
        "config.yaml youtube.privacy_on_upload must be 'private' - publishing happens only via publish.py"
    return cfg


def _description(m):
    desc = m.get("description", "")
    try:
        from media_search import credits_for_manifest
        credits = credits_for_manifest(m)
    except Exception:  # noqa: BLE001 - credits are best-effort
        credits = []
    if credits:
        desc = (desc + "\n\nCredits:\n" + "\n".join(credits)).strip()
    return desc[:5000]


def upload_key(key, youtube=None):
    cfg = load_config()
    entry = rq.get(key)
    if not entry:
        raise KeyError(f"{key} not in review_queue.json")
    if entry.get("video_id"):
        print(f"{key}: already uploaded ({entry['video_id']}), skipping")
        return entry["video_id"]
    out = ROOT / entry["output_path"]
    if not out.exists():
        raise FileNotFoundError(out)
    m = json.loads((ROOT / entry["manifest"]).read_text(encoding="utf-8-sig"))
    yt_cfg = cfg["youtube"]
    tags = m.get("tags") or yt_cfg["default_tags"]
    body = {
        "snippet": {
            "title": m.get("title", key)[:100],
            "description": _description(m),
            "tags": tags,
            "categoryId": str(m.get("category_id", yt_cfg["category_id"])),
        },
        "status": {
            "privacyStatus": "private",     # hardcoded on purpose, matches the asserted config
            "selfDeclaredMadeForKids": bool(yt_cfg["made_for_kids"]),
            "embeddable": True,
            "publicStatsViewable": True,
        },
    }
    if youtube is None:
        from youtube_auth import get_youtube
        youtube = get_youtube()
    print(f"Uploading {key} as PRIVATE ...")
    req = youtube.videos().insert(part="snippet,status", body=body,
                                  media_body=MediaFileUpload(str(out), chunksize=-1, resumable=True))
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            print(f"  {int(status.progress() * 100)}%")
    video_id = resp["id"]
    thumb = m.get("thumbnail_path")
    if thumb and (ROOT / thumb).exists():
        try:
            youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(ROOT / thumb))).execute()
        except Exception as e:  # noqa: BLE001 - custom thumbnails need a verified channel
            print(f"  [warn] thumbnail not set: {e}")
    rq.update(key, status="uploaded_private", video_id=video_id)
    print(f"  done: https://youtu.be/{video_id} (private)")
    return video_id


def upload_all():
    ids = []
    youtube = None
    for key, e in rq.load().items():
        if e.get("status") == "built" and not e.get("video_id"):
            if youtube is None:
                from youtube_auth import get_youtube
                youtube = get_youtube()
            try:
                ids.append(upload_key(key, youtube))
            except Exception as ex:  # noqa: BLE001 - keep going with the rest
                print(f"  [error] {key}: {ex}")
    return ids


if __name__ == "__main__":
    if len(sys.argv) > 1:
        upload_key(sys.argv[1])
    else:
        print(f"Uploaded {len(upload_all())} video(s).")
