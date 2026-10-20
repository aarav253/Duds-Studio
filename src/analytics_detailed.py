"""Retention / watch time / average view % via the YouTube Analytics API.

Needs the yt-analytics.readonly scope. Note: this API lags YouTube Studio by
2-3+ days; a video live only a few hours usually returns no rows yet - that's
YouTube's processing delay, not a bug.
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_queue as rq  # noqa: E402

METRICS = "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained"


def refresh_all(analytics=None, start="2020-01-01", channel=None):
    entries = {k: e for k, e in rq.load().items()
               if e.get("video_id") and e.get("status") == "published" and e.get("channel") == channel}
    if not entries:
        return {}
    if analytics is None:
        from youtube_auth import get_analytics
        analytics = get_analytics(interactive=False, channel=channel)
    end = (date.today() + timedelta(days=1)).isoformat()
    results = {}
    for key, e in entries.items():
        resp = analytics.reports().query(ids="channel==MINE", startDate=start, endDate=end,
                                         metrics=METRICS, filters=f"video=={e['video_id']}").execute()
        rows = resp.get("rows") or []
        cols = [h["name"] for h in resp.get("columnHeaders", [])]
        detail = dict(zip(cols, rows[0])) if rows else {"note": "no data yet (Analytics API lags 2-3 days)"}
        detail["fetched_at"] = datetime.now().isoformat(timespec="minutes")
        rq.update(key, detailed_stats=detail)
        results[key] = detail
    return results


if __name__ == "__main__":
    for key, d in refresh_all().items():
        print(key, d)
