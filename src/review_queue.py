"""Tiny load/save/update helper for review_queue.json.

Every other script reads and writes build/upload/publish status through here.
Entry schema (keyed by manifest id):
    status: built | error | uploaded_private | scheduled | published
    manifest, output_path, title, video_id, error, scheduled_for, stats, detailed_stats
"""
import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "review_queue.json"

_lock = threading.Lock()  # the Flask dashboard can call into this from several threads


def load():
    if not QUEUE_PATH.exists():
        return {}
    try:
        with open(QUEUE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save(queue):
    import time
    tmp = QUEUE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)
    # atomic replace, so a crash mid-write can't corrupt the ledger. The
    # project lives in OneDrive, whose sync briefly locks files - retry
    # instead of losing the update (a lost write once left a deleted
    # video's id in the ledger).
    for attempt in range(40):
        try:
            os.replace(tmp, QUEUE_PATH)
            return
        except PermissionError:
            time.sleep(0.25)
    os.replace(tmp, QUEUE_PATH)


def update(key, **fields):
    """Merge fields into queue[key] and persist. Returns the updated entry."""
    with _lock:
        queue = load()
        entry = queue.get(key, {})
        entry.update(fields)
        queue[key] = entry
        save(queue)
        return entry


def remove_fields(key, *names):
    with _lock:
        queue = load()
        entry = queue.get(key)
        if entry is None:
            return None
        for n in names:
            entry.pop(n, None)
        save(queue)
        return entry


def get(key):
    return load().get(key)


def find_by_video_id(video_id):
    for key, entry in load().items():
        if entry.get("video_id") == video_id:
            return key, entry
    return None, None
