"""Render a list of edit manifests with N parallel workers, picking the engine
from the manifest, and adding voicelines afterwards for template_edit renders.

    python src/render_queue.py 3 wk_a wk_b ...
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "venv" / "Scripts" / "python.exe")


def engine(m):
    if m.get("type") == "template_edit":
        return "src/template_edit.py"
    if "segments" in m:
        return "src/seg_edit.py"
    t = json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig"))
    return {"focus": "src/focus_edit.py", "reveal": "src/reveal_edit.py"}.get(t.get("engine"), "src/template_edit.py")


def run(mid):
    p = ROOT / "edits" / f"{mid}.json"
    m = json.loads(p.read_text(encoding="utf-8-sig"))
    t0 = time.time()
    r = subprocess.run([PY, engine(m), str(p)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0
    if ok and m.get("type") == "template_edit" and m.get("voicelines"):
        r2 = subprocess.run([PY, "src/voice_mix.py", f"output/{mid}.mp4", str(p)], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        ok = r2.returncode == 0
        if not ok:
            r = r2
    msg = f"{'DONE' if ok else 'FAIL'} {mid} ({time.time() - t0:.0f}s)"
    if not ok:
        msg += "\n" + (r.stderr or r.stdout)[-1500:]
    print(msg, flush=True)
    return ok


if __name__ == "__main__":
    n = int(sys.argv[1])
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(run, sys.argv[2:]))
    print(f"{sum(res)}/{len(res)} rendered", flush=True)
