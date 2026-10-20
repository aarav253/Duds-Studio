"""Upload edits to a given channel as PRIVATE, each scheduled to go public one
day apart at a fixed local time (YouTube's own publishAt does the publishing).

    python src/upload_channel.py <channel> <first YYYY-MM-DD> <HH:MM> <id> [<id> ...]

<channel> is a token name from youtube_auth (e.g. duds_edits). The n-th id is
scheduled for first date + n days. Already uploaded ids are skipped, so a run
stopped by the daily API quota (an upload costs 1600 of 10,000 units) can be
re-run the next day with the same arguments.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_queue as rq  # noqa: E402
from upload import _description, load_config  # noqa: E402
from youtube_auth import get_youtube  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def upload_scheduled(youtube, channel, key, when):
    """when=None: upload PRIVATE with no publish time (it stays private until published from the dashboard)."""
    entry = rq.get(key) or {}
    if entry.get("video_id"):
        print(f"{key}: already uploaded ({entry['video_id']}), skipping")
        return entry["video_id"]
    cfg = load_config()["youtube"]
    m = json.loads((ROOT / "edits" / f"{key}.json").read_text(encoding="utf-8-sig"))
    out = ROOT / "output" / f"{key}.mp4"
    if when is not None and when <= datetime.now().astimezone():
        raise ValueError(f"{key}: {when} is in the past")
    body = {
        "snippet": {"title": " ".join(m.get("title", key).split())[:100], "description": _description(m),
                    "tags": m.get("tags") or cfg["default_tags"], "categoryId": str(m.get("category_id", cfg["category_id"]))},
        "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": bool(cfg["made_for_kids"]),
                   "embeddable": True, "publicStatsViewable": True},
    }
    if when is not None:
        body["status"]["publishAt"] = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    print(f"Uploading {key} -> {channel}, " + (f"public at {when:%a %m/%d %I:%M %p}" if when else "PRIVATE (no publish time)"))
    req = youtube.videos().insert(part="snippet,status", body=body,
                                  media_body=MediaFileUpload(str(out), chunksize=-1, resumable=True))
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    vid = resp["id"]
    rq.update(key, status="scheduled" if when else "uploaded_private", manifest=f"edits/{key}.json", output_path=f"output/{key}.mp4",
              title=body["snippet"]["title"], error=None, video_id=vid, channel=channel)
    if when:
        rq.update(key, scheduled_for=when.isoformat(timespec="minutes"))
    print(f"  done: https://youtu.be/{vid}")
    return vid


if __name__ == "__main__":
    channel, first, hhmm, keys = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    yt = get_youtube(channel=channel)
    me = yt.channels().list(part="snippet", mine=True).execute()["items"][0]["snippet"]["title"]
    print("Channel:", me)
    day0 = datetime.strptime(f"{first}T{hhmm}", "%Y-%m-%dT%H:%M").astimezone()
    for n, key in enumerate(keys):
        try:
            upload_scheduled(yt, channel, key, day0 + timedelta(days=n))
        except HttpError as e:
            if "quota" in str(e).lower() or "uploadlimitexceeded" in str(e).lower():
                print(f"Daily upload limit reached at {key} - re-run tomorrow with the same arguments.")
                break
            raise
