"""Search YouTube for a song, download the top audio candidates and pick the
one that matches a reference edit exactly (src/song_match.py).

    python src/find_song.py <template_n> "<query>" [max_candidates]
Writes music/candidates/t<n>/*.wav and music/template_<n>_song.json with the
winning file, rate and offset when the match is confirmed.
"""
import json
import sys
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import fetch  # noqa: E402
from song_match import match  # noqa: E402


def candidates(query, k):
    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True}) as y:
        es = [e for e in y.extract_info(f"ytsearch{k * 3}:{query}", download=False)["entries"] if e]
    out = []
    for e in es:
        t = (e.get("title") or "").lower()
        if (e.get("duration") or 999) > 420 or any(w in t for w in ("hour", "1h", "edit", "amv", "mix", "lyric video")):
            continue
        out.append(e)
        if len(out) >= k:
            break
    return out


def run(n, query, k=4, extra=()):
    ref = ROOT / f"music/template_{n}_ref.wav"
    d = ROOT / f"music/candidates/t{n}"
    d.mkdir(parents=True, exist_ok=True)
    results = []
    files = [(ROOT / x).resolve() for x in extra]
    for i, e in enumerate(candidates(query, k)):
        name = f"c{i}_{e['id']}"
        if not (d / f"{name}.wav").exists():
            for _ in range(3):
                try:
                    fetch(f"https://www.youtube.com/watch?v={e['id']}", mode="audio", to=str(d.relative_to(ROOT)), name=name)
                    break
                except Exception:  # noqa: BLE001
                    pass
        if (d / f"{name}.wav").exists():
            files.append(d / f"{name}.wav")
            (d / f"{name}.title.txt").write_text(f"{e.get('title')} | {e.get('channel')} | https://youtu.be/{e['id']}", encoding="utf-8")
    for f in files:
        r = match(ref, f)
        title = (f.with_name(f.stem + ".title.txt").read_text(encoding="utf-8") if f.with_name(f.stem + ".title.txt").exists() else f.name)
        results.append({"file": str(f.relative_to(ROOT)).replace("\\", "/"), "title": title, **r})
        print(f"  T{n} {r['corr']:.3f} rate {r['rate']} off {r['offset']:.2f} confirm {r['confirm']} | {title[:70]}", flush=True)
    results.sort(key=lambda r: -(min(r["confirm"]) if r["confirm"] else 0))
    best = results[0] if results else None
    ok = bool(best and best["confirm"] and min(best["confirm"]) > 0.6)
    (ROOT / f"music/template_{n}_song.json").write_text(json.dumps({"query": query, "ok": ok, "best": best, "all": results}, indent=1,
                                                               ensure_ascii=False), encoding="utf-8")
    print(f"T{n}: {'MATCH' if ok else 'no confirmed match'} -> {best['title'][:60] if best else None}", flush=True)
    return ok, best


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4, sys.argv[4:])
