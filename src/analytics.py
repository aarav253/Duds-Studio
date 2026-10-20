"""Basic views / likes / comments via the YouTube Data API -> review_queue.json."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_queue as rq  # noqa: E402


def refresh_all(youtube=None, channel=None):
    """channel: token name (None = the original Duds channel); only that channel's videos are refreshed."""
    entries = {k: e for k, e in rq.load().items() if e.get("video_id") and e.get("channel") == channel}
    if not entries:
        return {}
    if youtube is None:
        from youtube_auth import get_youtube
        youtube = get_youtube(interactive=False, channel=channel)
    by_id = {e["video_id"]: k for k, e in entries.items()}
    ids = list(by_id)
    results = {}
    for i in range(0, len(ids), 50):
        resp = youtube.videos().list(part="statistics,status", id=",".join(ids[i:i + 50])).execute()
        for it in resp.get("items", []):
            s = it.get("statistics", {})
            stats = {"views": int(s.get("viewCount", 0)), "likes": int(s.get("likeCount", 0)),
                     "comments": int(s.get("commentCount", 0)),
                     "privacy": it.get("status", {}).get("privacyStatus"),
                     "fetched_at": datetime.now().isoformat(timespec="minutes")}
            key = by_id[it["id"]]
            rq.update(key, stats=stats)
            results[key] = stats
    return results


if __name__ == "__main__":
    for key, s in refresh_all().items():
        print(f"{key:40s} views={s['views']:>7} likes={s['likes']:>5} comments={s['comments']:>4} [{s['privacy']}]")
