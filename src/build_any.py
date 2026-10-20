"""Build any manifest in edits/ with the engine it was made for, and record the result in review_queue.json.

The project grew several renderers; the manifest's shape says which one it belongs to:
  "segments"            -> seg_edit, through fast_render.py (parallel chunks)    batch_t27/t30/t33, batch_week ...
  "caption_pre"         -> reveal_edit.py   (template 11 "brutal." edits)
  "popup"               -> focus_edit.py    (template 12 pop-in edits)
  type "sins_edit"      -> sins_edit.py     (template 10)
  type "freestyle"      -> freestyle_edit.py
  anything else         -> assemble.py      (the original engines: template_edit, wekisava, xavier, rise, ...)

    python src/build_any.py <id> [--preview]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_queue as rq  # noqa: E402

PY = str(ROOT / "venv/Scripts/python.exe")


def engine(m):
    if "segments" in m:
        return "seg_edit"
    if "caption_pre" in m:
        return "reveal_edit"
    if "popup" in m:
        return "focus_edit"
    if m.get("type") in ("sins_edit", "freestyle"):
        return m["type"]
    return "assemble"


def command(key, preview=False):
    """(engine, argv, output path) for one manifest."""
    mf = ROOT / "edits" / f"{key}.json"
    m = json.loads(mf.read_text(encoding="utf-8-sig"))
    eng = engine(m)
    rel = f"edits/{key}.json"
    if eng == "seg_edit":
        if preview:                                       # half-res preview of the whole edit
            b = float(m["body_end"])
            return eng, [PY, "src/fast_render.py", key, "--section", "0", f"{b:g}"], ROOT / "output" / f"{key}_sec0-{b:g}_preview.mp4"
        return eng, [PY, "src/fast_render.py", key], ROOT / "output" / f"{key}.mp4"
    if eng == "assemble":                                 # assemble has no preview mode; it always renders full
        return eng, [PY, "src/assemble.py", rel], ROOT / "output" / f"{key}.mp4"
    script = {"reveal_edit": "reveal_edit.py", "focus_edit": "focus_edit.py", "sins_edit": "sins_edit.py",
              "freestyle": "freestyle_edit.py"}[eng]
    argv = [PY, f"src/{script}", rel] + (["--preview"] if preview else [])
    return eng, argv, ROOT / "output" / f"{key}{'_preview' if preview else ''}.mp4"


def build(key, preview=False):
    """Run the engine, streaming its output to stdout (the dashboard captures it into the job log)."""
    eng, argv, out = command(key, preview)
    print(f"[build] {key} with {eng}{' (preview)' if preview else ''}\n$ {' '.join(argv[1:])}", flush=True)
    p = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    tail = []
    for line in p.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line.strip()])[-5:]
    if p.wait() != 0:
        if not preview:
            rq.update(key, error=f"{eng} failed: {' | '.join(tail)[-300:]}")
        raise RuntimeError(f"{eng} exited with code {p.returncode}")
    if preview:
        return str(out.relative_to(ROOT)).replace("\\", "/")
    title = json.loads((ROOT / "edits" / f"{key}.json").read_text(encoding="utf-8-sig")).get("title", key)
    e = rq.get(key) or {}
    # a re-render after upload must not lose the YouTube status, so only fresh entries become "built"
    rq.update(key, status=e.get("status") if e.get("video_id") else "built", manifest=f"edits/{key}.json",
              output_path=str(out.relative_to(ROOT)).replace("\\", "/"), title=e.get("title") or title, error=None)
    run_qc(key)
    return str(out.relative_to(ROOT))


def run_qc(key):
    """frame-by-frame QC against the reference: seg_qc for seg_edit renders, quality_check / wekisava_qc for the
    template_edit engines (the same calls batch_build makes).  Other engines have no reference-based QC."""
    path = ROOT / "edits" / f"{key}.json"
    m = json.loads(path.read_text(encoding="utf-8-sig"))
    eng = engine(m)
    if eng == "seg_edit":
        import seg_qc
        return seg_qc.run(key)
    if m.get("type") == "template_edit":
        tpl = json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig")).get("engine")
        import quality_check
        import wekisava_qc
        s = (wekisava_qc if tpl in ("wekisava", "xavier", "rise") else quality_check).run(path, render_path=ROOT / "output" / f"{key}.mp4")
        rq.update(key, qc={"overall": s["overall"], "counts": s["counts"], "report": s["report"],
                           "issues": [f"[{i['section']}] {i['check']}: {i['detail']}" for i in s["items"] if i["status"] != "PASS"]})
        return s
    print(f"[qc] no reference-based QC for the {eng} engine")
    return None


if __name__ == "__main__":
    print(build(sys.argv[1], "--preview" in sys.argv))
