"""Re-frame character clips that came out too zoomed in, and mark extreme
close-ups so the edit pools skip them.

    python src/panel_refine.py <plan.json>
plan.json: {"reframe": [{"clip": "...scene_07.mp4", "cx": 0.46}, ...],     # full-height 4:5 window at cx
            "closeup": ["...scene_04.mp4", ...],                           # pool entry gets "framing": "closeup"
            "poor": ["...scene_24.mp4", ...]}                              # no character / too dark: "framing": "poor"
Each re-framed scene is written into crops_override.json (so clip_crops.py
reruns keep it), exported again from the source and bar-checked; the old
clip goes to <clips>/_zoomed/.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_crops import export  # noqa: E402


def reframe(clip, cx, h=1.0):
    p = ROOT / clip
    cd = p.parent
    i = int(p.stem.split("_")[1])
    S = json.loads((cd / "scenes.json").read_text(encoding="utf-8-sig"))
    sc = next(s for s in S["scenes"] if s["i"] == i)
    ovp = cd / "crops_override.json"
    ov = json.loads(ovp.read_text(encoding="utf-8-sig")) if ovp.exists() else {}
    ov[str(i)] = {"cx": round(cx, 3), "cy": 0.4, "h": h}
    ovp.write_text(json.dumps(ov, indent=1), encoding="utf-8")
    (cd / "_zoomed").mkdir(exist_ok=True)
    if not (cd / "_zoomed" / p.name).exists():
        shutil.copy2(p, cd / "_zoomed" / p.name)
    n = sc["end"] - sc["start"] + 1
    cap = cv2.VideoCapture(str(ROOT / S["source"]))
    export(cap, sc["start"], sc["end"], np.full(n, cx), np.full(n, 0.4), np.full(n, h), p)
    stale = cd / "_with_bars" / p.name                       # a bar backup of the OLD crop must not be reused
    if stale.exists():
        stale.unlink()
    from clip_bars import debar
    return debar(clip)[1]


def mark(paths, label):
    """pool entries with a "framing" label are skipped by batch_week.pool()."""
    by_pool = {}
    for c in paths:
        char = Path(c).parts[2]
        by_pool.setdefault(char, []).append(c)
    for char, cs in by_pool.items():
        pp = ROOT / "panels/Blue Lock" / char / "clips_pool.json"
        d = json.loads(pp.read_text(encoding="utf-8-sig"))
        for e in d["clips"]:
            if e["path"] in cs:
                e["framing"] = label
        pp.write_text(json.dumps(d, indent=1), encoding="utf-8")


if __name__ == "__main__":
    plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for r in plan.get("reframe", []):
        print("reframed", r["clip"], reframe(r["clip"], r["cx"], r.get("h", 1.0)), flush=True)
    for label in ("closeup", "poor"):
        mark(plan.get(label, []), label)
        print(f"{label} marked:", len(plan.get(label, [])))
