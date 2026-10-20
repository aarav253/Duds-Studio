"""Keep references/ numbered: one folder per reference edit, numbered like
its template (references/template_N/ <-> templates/video_edit_template_N.json).

    references/template_N/reference.mp4   the reference edit
    references/template_N/analysis/       frame-by-frame analysis
    references/template_N/compare.mp4     side-by-side checks (optional)
    references/template_N/source.txt      the file's original name

    python src/references.py           # number any new video dropped into references/
    python src/references.py --list

New videos get the next free number. Building the template afterwards:
    python src/template_builder.py references/template_N/reference.mp4 video_edit_template_N
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "references"
VIDEO = (".mp4", ".mov", ".webm", ".mkv")


def numbers():
    return sorted(int(m.group(1)) for p in REF.iterdir() if p.is_dir() and (m := re.fullmatch(r"template_(\d+)", p.name)))


def folder(n):
    return REF / f"template_{n}"


def place(src, n, original=None):
    d = folder(n)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / ("reference" + src.suffix.lower())
    shutil.move(str(src), str(dst))
    (d / "source.txt").write_text(original or src.name, encoding="utf-8")
    tp = ROOT / "templates" / f"video_edit_template_{n}.json"
    if tp.exists():                                   # keep the template pointing at its reference
        t = json.loads(tp.read_text(encoding="utf-8-sig"))
        t["reference"] = f"references/template_{n}/{dst.name}"
        tp.write_text(json.dumps(t, indent=1), encoding="utf-8")
    return dst


def number_new():
    out = []
    for p in sorted(REF.iterdir(), key=lambda q: q.stat().st_mtime):
        if p.is_file() and p.suffix.lower() in VIDEO:
            n = (numbers() or [0])[-1] + 1
            out.append((n, place(p, n)))
    return out


if __name__ == "__main__":
    if "--list" in sys.argv:
        for n in numbers():
            src = folder(n) / "source.txt"
            print(n, (src.read_text(encoding="utf-8") if src.exists() else "").encode("ascii", "replace").decode())
    else:
        for n, p in number_new():
            print(f"template {n}: {p.relative_to(ROOT)}")
