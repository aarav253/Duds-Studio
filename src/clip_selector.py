"""Scene-cut detection to auto-pick candidate clip windows from an episode file."""
from pathlib import Path


def _duration(path):
    from moviepy.editor import VideoFileClip
    with VideoFileClip(str(path)) as c:
        return float(c.duration)


def find_candidate_clips(video_path, min_len=8, max_len=45, count=3, threshold=27.0):
    """Returns up to `count` (start, end) windows in seconds: the longest
    detected scenes within [min_len, max_len]. Falls back to fixed-length
    windows spread across the file if no scene lands in range."""
    from scenedetect import ContentDetector, detect
    scenes = []
    try:
        for s, e in detect(str(video_path), ContentDetector(threshold=threshold)):
            a, b = s.get_seconds(), e.get_seconds()
            if b - a > max_len:
                b = a + max_len  # a long unbroken scene still makes a usable window
            if b - a >= min_len:
                scenes.append((a, b))
    except Exception as e:  # noqa: BLE001 - detection failure -> fallback windows
        print(f"  [clip_selector] scene detection failed: {e}")
    if scenes:
        scenes.sort(key=lambda w: w[1] - w[0], reverse=True)
        return sorted(scenes[:count])
    total = _duration(video_path)
    length = min(max_len, max(min_len, total / max(count, 1)))
    if total < min_len:
        return [(0.0, total)]
    windows = []
    step = (total - length) / max(count - 1, 1) if count > 1 else 0
    for i in range(count):
        a = round(i * step, 2)
        if a + length <= total + 1e-6:
            windows.append((a, round(a + length, 2)))
    return windows


if __name__ == "__main__":
    import sys
    print(find_candidate_clips(Path(sys.argv[1])))
