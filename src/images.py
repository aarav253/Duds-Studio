"""Real topic photos from the Pexels API for facts/story narration videos.

Pexels photos are licensed for free reuse (no attribution required). This was
chosen deliberately instead of a Google Images scraper, which would be the same
class of problem as scraping anime footage.

Needs PEXELS_API_KEY. Without it, functions return None / [] and callers fall
back to visuals.generate_theme_bg.
"""
import os
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "clips" / "TopicImages"
API = "https://api.pexels.com/v1/search"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _slug(query):
    return re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_") or "topic"


def _cached(folder):
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def fetch_topic_images(query, n=3, orientation="landscape"):
    """Return up to n local image paths for `query`, downloading only what
    the cache doesn't already have."""
    folder = CACHE_DIR / _slug(query)
    have = _cached(folder)
    if len(have) >= n:
        return have[:n]
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return have
    try:
        r = requests.get(API, headers={"Authorization": key},
                         params={"query": query, "per_page": max(n, 5), "orientation": orientation},
                         timeout=15)
        r.raise_for_status()
        photos = r.json().get("photos", [])
    except (requests.RequestException, ValueError) as e:
        print(f"  [images] Pexels search failed for {query!r}: {e}")
        return have
    folder.mkdir(parents=True, exist_ok=True)
    for ph in photos:
        if len(have) >= n:
            break
        dest = folder / f"pexels_{ph['id']}.jpg"
        if dest.exists():
            continue
        try:
            img = requests.get(ph["src"]["large2x"], timeout=30)
            img.raise_for_status()
            dest.write_bytes(img.content)
            have.append(str(dest))
        except (requests.RequestException, KeyError) as e:
            print(f"  [images] download failed: {e}")
    return have[:n]


def fetch_topic_image(query):
    paths = fetch_topic_images(query, 1)
    return paths[0] if paths else None


if __name__ == "__main__":
    import sys
    print(fetch_topic_images(sys.argv[1] if len(sys.argv) > 1 else "black hole", 3))
