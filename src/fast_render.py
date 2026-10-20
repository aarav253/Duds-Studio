"""Parallel render of a seg_edit manifest: the frame range is split into N chunks
rendered at the same time (every frame is computed from its index alone, so the
result is identical to a single-process render), then joined losslessly and the
mixed audio muxed on.

    python src/fast_render.py <id> [workers=8]
    python src/fast_render.py <id> --section 18.5 23 [--full]   # quick check of one stretch (half-res by default)
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
PY = str(ROOT / "venv/Scripts/python.exe")
FF = imageio_ffmpeg.get_ffmpeg_exe()


def _chunk(manifest, k0, k1, out, preview):
    cmd = [PY, "src/seg_edit.py", str(manifest), "--frames", str(k0), str(k1), str(out)] + (["--preview"] if preview else [])
    for attempt in range(3):                                   # Windows can briefly run out of resources with many workers
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            return out
        if "WinError 1450" not in r.stderr:
            break
        time.sleep(5 + 10 * attempt)
    raise RuntimeError(r.stderr[-1500:])


def render(mid, workers=8, section=None, preview=False):
    from seg_edit import SegEdit
    t0 = time.time()
    manifest = ROOT / "edits" / f"{mid}.json"
    m = json.loads(manifest.read_text(encoding="utf-8-sig"))
    e = SegEdit(m, preview)
    k_start, k_end = 0, e.n
    if section:
        k_start, k_end = int(section[0] * e.fps), min(e.n, int(section[1] * e.fps))
    tag = f"_sec{section[0]:g}-{section[1]:g}" if section else ""
    out = ROOT / "output" / f"{mid}{tag}{'_preview' if preview else ''}.mp4"
    parts_dir = ROOT / "output" / f".parts_{mid}"
    parts_dir.mkdir(exist_ok=True)
    total = k_end - k_start
    step = -(-total // workers)
    jobs = [(k_start + i * step, min(k_end, k_start + (i + 1) * step), parts_dir / f"p{i:02d}.mp4") for i in range(workers)
            if k_start + i * step < k_end]
    wav = parts_dir / "mix.wav"
    with ThreadPoolExecutor(len(jobs) + 1) as ex:
        fa = ex.submit(e.mix, wav)
        futs = [ex.submit(_chunk, manifest, a, b, o, preview) for a, b, o in jobs]
        for i, f in enumerate(as_completed(futs), 1):          # progress line per finished chunk (the dashboard reads it)
            f.result()
            print(f"[progress] {i}/{len(futs)} chunks", flush=True)
        parts = [f.result() for f in futs]
        lufs = fa.result()
    lst = parts_dir / "list.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    joined = parts_dir / "joined.mp4"
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)], check=True)
    a_off = ["-ss", f"{k_start / e.fps:.4f}"] if k_start else []
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(joined), *a_off, "-i", str(wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", str(out)], check=True)
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    print(f"DONE {out.name}: {total / e.fps:.1f}s of video in {time.time() - t0:.0f}s ({len(jobs)} workers), {lufs:.1f} LUFS", flush=True)
    return out


if __name__ == "__main__":
    mid = sys.argv[1]
    if "--section" in sys.argv:
        i = sys.argv.index("--section")
        render(mid, 4, (float(sys.argv[i + 1]), float(sys.argv[i + 2])), preview="--full" not in sys.argv)
    else:
        render(mid, int(sys.argv[2]) if len(sys.argv) > 2 else max(2, min(12, (os.cpu_count() or 8) // 2)))
