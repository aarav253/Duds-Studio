"""Build music/library/ - clean copies of songs used in anime/Blue Lock edits.

For each (title, artist) it searches YouTube and takes the best upload from the
artist's own channel or their auto-generated "<artist> - Topic" channel
(cleanest audio, no edit sound effects); skips anything else so we never pull
a re-upload with effects baked in. Audio only, WAV + original stream.

    python src/song_library.py
"""
import json
import re
import sys
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import fetch  # noqa: E402

SONGS = [  # (search, artist keywords that the channel must contain, file name)
    ("TENI TENI slowed HEAZY", ["heazy"], "teni_teni_slowed"),
    ("MONTAGEM ROOMLOST slowed LXSTFFACE", ["lxstfface"], "montagem_roomlost_slowed"),
    ("Dead Inside sped up", [], "dead_inside_sped_up"),
    ("DEADLOCK slowed", [], "deadlock_slowed"),
    ("MONTAGEM CORAL", [], "montagem_coral"),
    ("SLAVA FUNK!", [], "slava_funk"),
    ("FUNK INFERNAL", [], "funk_infernal"),
    ("Murder In My Mind Kordhell", ["kordhell"], "murder_in_my_mind"),
    ("Close Eyes DVRST", ["dvrst"], "close_eyes"),
    ("METAMORPHOSIS INTERWORLD", ["interworld"], "metamorphosis"),
    ("RAPTURE INTERWORLD", ["interworld"], "rapture"),
    ("Override KSLV Noh", ["kslv"], "override"),
    ("Aura Ogryzek", ["ogryzek"], "aura"),
    ("Sleepwalker akiaura", ["akiaura"], "sleepwalker"),
    ("Memory Reboot VOJ Narvent", ["narvent", "vøj", "voj"], "memory_reboot"),
    ("PASSO BEM SOLTO", [], "passo_bem_solto"),
    ("ELOY NVXUS", ["nvxus"], "eloy"),
    ("VINGANCA DEMONIACA phonk", [], "vinganca_demoniaca"),
    ("MONTAGEM XONADA", [], "montagem_xonada"),
    ("Brazilian Phonk Mano", [], "brazilian_phonk_mano"),
]


def pick(query, artists):
    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True}) as y:
        es = [e for e in y.extract_info(f"ytsearch15:{query}", download=False)["entries"] if e]
    def ok(e):
        ch = (e.get("channel") or "").lower()
        t = (e.get("title") or "").lower()
        if (e.get("duration") or 0) > 420 or any(w in t for w in ("1 hour", "1h", "10 hours", "hour", "mix", "edit", "amv", "lyrics")):
            return False
        return (not artists) or any(a in ch for a in artists) or ch.endswith("- topic")
    good = [e for e in es if ok(e)]
    # prefer Topic / artist channels, then views
    good.sort(key=lambda e: (not (e.get("channel") or "").lower().endswith("- topic"), -(e.get("view_count") or 0)))
    return good[0] if good else None


if __name__ == "__main__":
    out = ROOT / "music" / "library"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for q, artists, name in SONGS:
        if (out / f"{name}.wav").exists():
            rows.append({"name": name, "status": "already there"}); continue
        e = pick(q, artists)
        if not e:
            print(f"  ! no clean upload found for {q}"); rows.append({"name": name, "status": "not found"}); continue
        r = None
        for attempt in range(3):                               # YouTube sometimes answers 403 on the first try
            try:
                r = fetch(f"https://www.youtube.com/watch?v={e['id']}", mode="audio", to="music/library", name=name)
                break
            except Exception as ex:  # noqa: BLE001
                print(f"  retry {attempt + 1} for {name}: {str(ex)[:80]}")
        if r is None:
            rows.append({"name": name, "status": "download failed"}); continue
        rows.append({"name": name, "title": r["title"], "channel": r["channel"], "url": r["url"], "wav": r["wav"]})
        print(f"  {name:28s} <- {r['title'][:55]} | {r['channel']}", flush=True)
    (out / "library.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
