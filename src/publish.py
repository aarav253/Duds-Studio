"""The ONLY code path that can make a video public.

    python src/publish.py <video_id>      # asks y/N first

The web dashboard's Publish button click is treated as the equivalent explicit
confirmation. schedule_video() defers going public to YouTube's own publishAt
field - still an explicit human choice of when. Nothing else in the pipeline
(batch build, upload) may flip privacy.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_queue as rq  # noqa: E402
from youtube_auth import get_youtube  # noqa: E402


def _status(youtube, video_id):
    items = youtube.videos().list(part="status", id=video_id).execute().get("items", [])
    if not items:
        raise LookupError(f"video {video_id} not found on the channel")
    return items[0]["status"]


def _keep(st):
    """videos.update(part=status) resets every status field it isn't sent
    (embedding and public view counts silently turned OFF) - resend them."""
    return {"embeddable": True, "publicStatsViewable": True,
            "license": st.get("license", "youtube")}


def publish_video(video_id, channel=None):
    youtube = get_youtube(channel=channel)
    st = _status(youtube, video_id)
    body = {"id": video_id, "status": {"privacyStatus": "public",
                                       "selfDeclaredMadeForKids": st.get("selfDeclaredMadeForKids", False),
                                       **_keep(st)}}
    youtube.videos().update(part="status", body=body).execute()
    key, _ = rq.find_by_video_id(video_id)
    if key:
        rq.update(key, status="published")
        rq.remove_fields(key, "scheduled_for")
    return True


def schedule_video(video_id, publish_at_local, channel=None):
    """publish_at_local: naive 'YYYY-MM-DDTHH:MM' in this machine's timezone.
    YouTube requires privacyStatus=private together with publishAt."""
    local = datetime.strptime(publish_at_local, "%Y-%m-%dT%H:%M").astimezone()
    if local <= datetime.now().astimezone():
        raise ValueError("scheduled time must be in the future")
    utc = local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    youtube = get_youtube(channel=channel)
    st = _status(youtube, video_id)
    body = {"id": video_id, "status": {"privacyStatus": "private", "publishAt": utc,
                                       "selfDeclaredMadeForKids": st.get("selfDeclaredMadeForKids", False),
                                       **_keep(st)}}
    youtube.videos().update(part="status", body=body).execute()
    key, _ = rq.find_by_video_id(video_id)
    if key:
        rq.update(key, status="scheduled", scheduled_for=local.isoformat(timespec="minutes"))
    return utc


def unschedule_video(video_id, channel=None):
    youtube = get_youtube(channel=channel)
    st = _status(youtube, video_id)
    # sending status without publishAt clears the schedule
    body = {"id": video_id, "status": {"privacyStatus": "private",
                                       "selfDeclaredMadeForKids": st.get("selfDeclaredMadeForKids", False),
                                       **_keep(st)}}
    youtube.videos().update(part="status", body=body).execute()
    key, _ = rq.find_by_video_id(video_id)
    if key:
        rq.update(key, status="uploaded_private")
        rq.remove_fields(key, "scheduled_for")


def delete_video(video_id, channel=None):
    """Permanently delete from YouTube. Only for videos that were never public
    (e.g. replacing a private upload after re-rendering a fix)."""
    youtube = get_youtube(channel=channel)
    st = _status(youtube, video_id)
    if st.get("privacyStatus") == "public":
        raise PermissionError("refusing to delete a PUBLIC video from the pipeline - do that in YouTube Studio")
    youtube.videos().delete(id=video_id).execute()
    key, _ = rq.find_by_video_id(video_id)
    if key:
        rq.update(key, status="built")
        rq.remove_fields(key, "video_id", "scheduled_for", "stats", "detailed_stats")


def sync_scheduled_status(channel=None):
    """YouTube flips scheduled videos public on its own (no webhook), so poll
    each locally-'scheduled' video's real privacy and update the ledger."""
    scheduled = {k: e for k, e in rq.load().items()
                 if e.get("status") == "scheduled" and e.get("video_id") and e.get("channel") == channel}
    if not scheduled:
        return 0
    youtube = get_youtube(interactive=False, channel=channel)
    changed = 0
    ids = [e["video_id"] for e in scheduled.values()]
    for i in range(0, len(ids), 50):
        resp = youtube.videos().list(part="status", id=",".join(ids[i:i + 50])).execute()
        live = {it["id"]: it["status"] for it in resp.get("items", [])}
        for key, e in scheduled.items():
            st = live.get(e["video_id"])
            if st and st.get("privacyStatus") == "public":
                rq.update(key, status="published")
                rq.remove_fields(key, "scheduled_for")
                changed += 1
    return changed


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python src/publish.py <video_id>")
        sys.exit(1)
    vid = sys.argv[1]
    key, entry = rq.find_by_video_id(vid)
    label = f"{entry.get('title')} ({key})" if entry else vid
    ans = input(f"Make '{label}' PUBLIC on YouTube now? [y/N] ").strip().lower()
    if ans == "y":
        publish_video(vid)
        print(f"Published: https://youtu.be/{vid}")
    else:
        print("Cancelled - video stays private.")
