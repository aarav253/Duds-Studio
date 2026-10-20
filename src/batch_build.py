"""Render every pending manifest in edits/.

    python src/batch_build.py            # only manifests not already built
    python src/batch_build.py --force    # rebuild ALL (resets local status - use carefully)
    python src/batch_build.py <id> ...   # specific manifests
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_queue as rq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EDITS = ROOT / "edits"
DONE_STATUSES = {"built", "uploaded_private", "scheduled", "published"}


def manifests():
    return sorted(p for p in EDITS.glob("*.json") if not p.name.startswith("_"))


def pending(force=False, only=None):
    queue = rq.load()
    out = []
    for p in manifests():
        try:
            m = json.loads(p.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            print(f"  [skip] {p.name}: invalid JSON ({e})")
            continue
        key = m.get("id") or p.stem
        if only and key not in only:
            continue
        if not force and queue.get(key, {}).get("status") in DONE_STATUSES:
            continue
        out.append((key, p, m))
    return out


def build_one(key, path, m, logger="bar"):
    import assemble
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    try:
        out = assemble.assemble(path, logger=logger)
        rq.update(key, status="built", manifest=rel, output_path=str(out.relative_to(ROOT)).replace("\\", "/"),
                  title=m.get("title", key), error=None)
        engine = json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig")).get("engine") \
            if m.get("type") == "template_edit" else None
        if engine in ("wekisava", "xavier", "rise"):
            import wekisava_qc
            s = wekisava_qc.run(path, render_path=out)
            rq.update(key, qc={"overall": s["overall"], "counts": s["counts"], "report": s["report"],
                               "issues": [f"[{i['section']}] {i['check']}: {i['detail']}"
                                          for i in s["items"] if i["status"] != "PASS"]})
        elif m.get("type") == "template_edit":
            # frame-by-frame quality check against the reference, after every build
            try:
                import quality_check
                s = quality_check.run(path, render_path=out)
                rq.update(key, qc={"overall": s["overall"], "counts": s["counts"], "report": s["report"],
                                   "issues": [f"[{i['section']}] {i['check']}: {i['detail']}"
                                              for i in s["items"] if i["status"] != "PASS"]})
            except Exception as e:  # noqa: BLE001 - QC must never lose a finished render
                traceback.print_exc()
                rq.update(key, qc={"overall": "ERROR", "issues": [f"QC crashed: {e}"]})
        return True
    except Exception as e:  # noqa: BLE001 - record and continue the batch
        traceback.print_exc()
        rq.update(key, status="error", manifest=rel, title=m.get("title", key), error=f"{type(e).__name__}: {e}")
        return False


def run(force=False, only=None, logger="bar"):
    todo = pending(force, only)
    if force:
        live = [k for k, _, _ in todo if rq.get(k) and rq.get(k).get("status") in ("published", "scheduled")]
        if live:
            print(f"  [note] --force rebuilds already-live videos locally: {live}. "
                  "Their YouTube uploads are NOT changed.")
    ok = 0
    for key, p, m in todo:
        print(f"== {key}")
        ok += build_one(key, p, m, logger)
    print(f"Built {ok}/{len(todo)}")
    return ok, len(todo)


if __name__ == "__main__":
    args = sys.argv[1:]
    run(force="--force" in args, only=[a for a in args if not a.startswith("--")] or None)
