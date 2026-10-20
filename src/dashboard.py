"""Duds Studio dashboard: a local Flask app over the real pipeline.  http://127.0.0.1:5000

    start_dashboard.bat        (or: venv\\Scripts\\python.exe src\\dashboard.py)

Pages: Videos (render, preview, QC, upload, publish), Manifests (edit / create from a template), Media (browse and add
clips, panels, music), Stats (YouTube views / likes).

Every action calls the same code the command line uses: build_any.py picks the render engine for a manifest
(fast_render/seg_edit, reveal_edit, focus_edit, sins_edit, freestyle_edit, assemble), seg_qc / quality_check
compare the render to its reference, upload_channel / publish / analytics talk to YouTube with each channel's
OAuth token.  Nothing here re-implements pipeline logic.

Safety:
  - binds 127.0.0.1 only, and rejects any Host header that isn't localhost (DNS rebinding)
  - every POST needs a random per-launch token header, so another website can't trigger an upload or a publish
  - /media serves only the media folders, so token*.json / client_secret.json (project root) are never reachable
  - uploads are always PRIVATE with no publish time; "Publish" (typed confirmation) or "Schedule" is the only release
"""
import contextlib
import io
import json
import mimetypes
import os
import queue
import re
import secrets
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_any  # noqa: E402
import review_queue as rq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EDITS = ROOT / "edits"
PORT = int(os.environ.get("DUDS_PORT", "5000"))
TOKEN = secrets.token_urlsafe(24)
MEDIA_ROOTS = ("panels", "clips", "music", "references", "output")     # browsable / servable
WRITABLE = ("panels", "clips", "music")                                 # where new media may be added
EXT = {"video": {".mp4", ".mov", ".mkv", ".webm", ".m4v"}, "image": {".jpg", ".jpeg", ".png", ".webp", ".gif"},
       "audio": {".wav", ".mp3", ".m4a", ".ogg", ".flac"}}
CHANNELS = {"duds_edits": "Duds_Edits", "": "Duds (original)"}          # token name -> label ("" = token.json)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3


@app.before_request
def _guard():
    if (request.host or "").split(":")[0] not in ("127.0.0.1", "localhost"):
        abort(403)
    if request.method == "POST" and request.headers.get("X-Duds-Token") != TOKEN:
        abort(403)


def _safe(rel, roots=MEDIA_ROOTS):
    """project-relative path inside one of the allowed folders, or 403."""
    p = (ROOT / (rel or "")).resolve()
    try:
        parts = p.relative_to(ROOT.resolve()).parts
    except ValueError:
        abort(403)
    if not parts or parts[0] not in roots:
        abort(403)
    return p


def _rel(p):
    return str(Path(p).resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _json():
    return request.get_json(force=True, silent=True) or {}


# ------------------------------------------------------------------ background jobs (one at a time)
JOBS, _Q, _n = {}, queue.Queue(), [0]


class _Log(io.TextIOBase):
    def __init__(self, job):
        self.job = job

    def write(self, s):
        if s:
            self.job["log"] = (self.job["log"] + s)[-60000:]
            for m in re.finditer(r"\[progress\] (\d+)/(\d+)", s):        # fast_render's per-chunk line
                self.job["progress"] = int(m.group(1)) / int(m.group(2))
        return len(s)


def submit(name, fn, *a, **k):
    _n[0] += 1
    jid = str(_n[0])
    JOBS[jid] = {"id": jid, "name": name, "status": "queued", "log": "", "result": None, "progress": None,
                 "created": time.time(), "started": None, "ended": None}
    _Q.put((jid, fn, a, k))
    return jid


def _worker():
    while True:
        jid, fn, a, k = _Q.get()
        job = JOBS[jid]; job["status"] = "running"; job["started"] = time.time()
        try:
            with contextlib.redirect_stdout(_Log(job)), contextlib.redirect_stderr(_Log(job)):
                job["result"] = fn(*a, **k)
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001 - the error goes to the job panel
            job["log"] += "\n" + traceback.format_exc()
            job["status"], job["result"] = "error", f"{type(e).__name__}: {e}"
        job["ended"] = time.time()


threading.Thread(target=_worker, daemon=True).start()


# ------------------------------------------------------------------ state
def _manifest_files():
    """edits/*.json that are real video manifests (have an id and something to render)."""
    out = {}
    for p in sorted(EDITS.glob("*.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            continue
        if isinstance(m, dict) and m.get("id") and any(k in m for k in ("segments", "shots", "sections", "clips", "type")):
            out[p.stem] = m
    return out


def _video(key, m, e):
    out = ROOT / (e.get("output_path") or f"output/{key}.mp4")
    try:
        _, _, pv = build_any.command(key, preview=True)
    except Exception:  # noqa: BLE001
        pv = None
    preview = pv if pv and pv.exists() and pv != out else None
    return {"key": key, "title": e.get("title") or m.get("title") or key, "engine": build_any.engine(m),
            "status": e.get("status") or ("built" if out.exists() else "not built"),
            "channel": e.get("channel") or "", "video_id": e.get("video_id"), "scheduled_for": e.get("scheduled_for"),
            "error": e.get("error"), "qc": e.get("qc"), "stats": e.get("stats"), "detailed": e.get("detailed_stats"),
            "output": _rel(out) if out.exists() else None, "mtime": out.stat().st_mtime if out.exists() else 0,
            "preview": _rel(preview) if preview else None, "preview_mtime": preview.stat().st_mtime if preview else 0}


@app.get("/api/state")
def api_state():
    q = rq.load()
    vids = [_video(k, m, q.get(k, {})) for k, m in _manifest_files().items()]
    if not AUTH_OK:
        _check_auth()
    chans = [{"id": c, "label": lbl, "authorized": AUTH_OK.get(c, False)} for c, lbl in CHANNELS.items()]
    jobs = sorted(JOBS.values(), key=lambda j: -j["created"])[:12]
    return jsonify(videos=vids, channels=chans,
                   jobs=[{k: j[k] for k in ("id", "name", "status", "result", "progress", "started", "ended")} for j in jobs])


AUTH_OK = {}                                                               # channel -> last verified login state


def _check_auth():
    import youtube_auth
    for ch in CHANNELS:
        try:
            youtube_auth._get_credentials(interactive=False, channel=ch or None)   # refreshes the token if it can
            AUTH_OK[ch] = True
        except Exception:  # noqa: BLE001 - missing, expired or revoked
            AUTH_OK[ch] = False


@app.post("/api/auth")
def api_auth():
    """Google's own login page opens in the browser on this PC; the new token is saved next to the old one."""
    ch = _json().get("channel") or ""

    def go():
        import youtube_auth
        yt = youtube_auth.get_youtube(interactive=True, channel=ch or None)
        AUTH_OK[ch] = True
        return yt.channels().list(part="snippet", mine=True).execute()["items"][0]["snippet"]["title"]
    return jsonify(job=submit(f"log in to {CHANNELS.get(ch, ch)}", go))


@app.get("/api/job/<jid>")
def api_job(jid):
    return jsonify(JOBS.get(jid) or abort(404))


# ------------------------------------------------------------------ render + QC
@app.post("/api/build")
def api_build():
    d = _json(); key = d.get("key")
    if key not in _manifest_files():
        abort(404)
    pv = bool(d.get("preview"))
    return jsonify(job=submit(f"{'preview' if pv else 'render'} {key}", build_any.build, key, pv))


@app.post("/api/qc")
def api_qc():
    key = _json().get("key")
    if not (ROOT / "output" / f"{key}.mp4").exists():
        return jsonify(error="render it first"), 400
    return jsonify(job=submit(f"QC {key}", build_any.run_qc, key))


# ------------------------------------------------------------------ YouTube
def _entry(key):
    e = rq.get(key) or abort(404)
    return e, (e.get("channel") or None)


@app.post("/api/upload")
def api_upload():
    """PRIVATE upload with no publish time. It stays private until Publish or Schedule."""
    d = _json(); key = d.get("key"); ch = d.get("channel") or None
    if not (ROOT / "output" / f"{key}.mp4").exists():
        return jsonify(error="render it first"), 400
    if (rq.get(key) or {}).get("video_id"):
        return jsonify(error="already uploaded"), 400

    def go():
        import upload_channel
        from youtube_auth import get_youtube
        return upload_channel.upload_scheduled(get_youtube(channel=ch), ch, key, None)
    return jsonify(job=submit(f"upload {key} (private) -> {CHANNELS.get(ch or '', ch)}", go))


@app.post("/api/publish")
def api_publish():
    d = _json()
    if d.get("confirm") != "PUBLISH":                                       # typed confirmation in the UI
        abort(400)
    e, ch = _entry(d.get("key"))
    import publish
    return jsonify(job=submit(f"PUBLISH {d['key']}", publish.publish_video, e["video_id"], channel=ch))


@app.post("/api/schedule")
def api_schedule():
    d = _json(); e, ch = _entry(d.get("key"))
    import publish
    return jsonify(job=submit(f"schedule {d['key']} @ {d['when']}", publish.schedule_video, e["video_id"], d["when"], channel=ch))


@app.post("/api/unschedule")
def api_unschedule():
    d = _json(); e, ch = _entry(d.get("key"))
    import publish
    return jsonify(job=submit(f"unschedule {d['key']}", publish.unschedule_video, e["video_id"], channel=ch))


@app.post("/api/stats")
def api_stats():
    detailed = bool(_json().get("detailed"))

    def go():
        import analytics
        import analytics_detailed
        import publish
        import youtube_auth
        failed = []
        for ch in CHANNELS:
            c = ch or None
            if not youtube_auth.token_path(c).exists():
                continue
            try:                                                       # one channel's expired login mustn't stop the other
                print(f"[{CHANNELS[ch]}] scheduled -> published: {publish.sync_scheduled_status(channel=c)}")
                print(f"[{CHANNELS[ch]}] stats for {len(analytics.refresh_all(channel=c))} videos")
                if detailed:
                    print(f"[{CHANNELS[ch]}] retention for {len(analytics_detailed.refresh_all(channel=c))} videos")
            except Exception as e:  # noqa: BLE001
                print(f"[{CHANNELS[ch]}] FAILED: {e}"); failed.append(CHANNELS[ch]); AUTH_OK[ch] = False
        if failed:
            raise RuntimeError(f"stats refreshed, except {', '.join(failed)} (log in again from the header)")
        return "stats refreshed"
    return jsonify(job=submit("refresh YouTube stats", go))


# ------------------------------------------------------------------ manifests
@app.get("/api/manifest/<key>")
def api_manifest(key):
    p = EDITS / f"{key}.json"
    if key not in _manifest_files():
        abort(404)
    return p.read_text(encoding="utf-8-sig"), 200, {"Content-Type": "text/plain; charset=utf-8"}


def _missing_files(m):
    """referenced media paths (clip/png/file/src values) that don't exist on disk."""
    found = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str) and k in ("clip", "png", "file", "src", "intro_image", "music_file") and "/" in v:
                    found.add(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                if isinstance(v, str) and "/" in v and Path(v).suffix.lower() in EXT["video"] | EXT["image"] | EXT["audio"]:
                    found.add(v)
                walk(v)
    walk(m)
    return sorted(f for f in found if not (ROOT / f).exists())


@app.post("/api/manifest")
def api_manifest_save():
    d = _json()
    try:
        m = json.loads(d.get("content", ""))
    except json.JSONDecodeError as e:
        return jsonify(error=f"invalid JSON: {e}"), 400
    mid = re.sub(r"[^A-Za-z0-9_-]+", "_", str(m.get("id", ""))).strip("_")
    if not mid:
        return jsonify(error='the manifest needs an "id"'), 400
    target = EDITS / f"{mid}.json"
    if d.get("new") and target.exists():
        return jsonify(error=f"{mid} already exists; pick a new id"), 400
    if not d.get("new") and d.get("original") and d["original"] != mid:
        return jsonify(error="changing the id of an existing manifest isn't allowed; use 'New from template'"), 400
    m["id"] = mid
    target.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return jsonify(ok=True, id=mid, engine=build_any.engine(m), missing=_missing_files(m))


# ------------------------------------------------------------------ media
@app.get("/api/browse")
def api_browse():
    rel = request.args.get("path", "")
    if not rel:
        return jsonify(path="", dirs=[r for r in MEDIA_ROOTS if (ROOT / r).exists()], files=[], writable=False)
    p = _safe(rel)
    if not p.is_dir():
        abort(404)
    dirs = sorted(x.name for x in p.iterdir() if x.is_dir() and not x.name.startswith((".", "_")))
    files = []
    for x in sorted(p.iterdir()):
        kind = next((k for k, s in EXT.items() if x.suffix.lower() in s), None)
        if x.is_file() and kind:
            files.append({"name": x.name, "path": _rel(x), "kind": kind, "size": x.stat().st_size})
    return jsonify(path=_rel(p), dirs=dirs, files=files[:400], total=len(files),
                   writable=_rel(p).split("/")[0] in WRITABLE)


@app.post("/api/media/upload")
def api_media_upload():
    d = _safe(request.form.get("path"), WRITABLE)
    if not d.is_dir():
        abort(400)
    saved, skipped = [], []
    for f in request.files.getlist("files"):
        name = re.sub(r"[^A-Za-z0-9 _.()-]+", "", os.path.basename(f.filename or "")).strip(" .")
        if not name or not any(Path(name).suffix.lower() in s for s in EXT.values()):
            skipped.append(f.filename); continue
        p = d / name; i = 2
        while p.exists():
            p = d / f"{Path(name).stem}_{i}{Path(name).suffix}"; i += 1
        f.save(p); saved.append(_rel(p))
    return jsonify(saved=saved, skipped=skipped)


@app.post("/api/media/mkdir")
def api_media_mkdir():
    d = _json()
    parent = _safe(d.get("path"), WRITABLE)
    name = re.sub(r"[^A-Za-z0-9 _-]+", "", d.get("name", "")).strip()
    if not name:
        abort(400)
    (parent / name).mkdir(exist_ok=True)
    return jsonify(ok=True)


@app.get("/media/<path:rel>")
def media(rel):
    p = _safe(rel)
    if not p.is_file():
        abort(404)
    return send_file(p, mimetype=mimetypes.guess_type(p.name)[0], conditional=True)


@app.get("/")
def index():
    return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)


def main():
    url = f"http://127.0.0.1:{PORT}"
    print(f"Duds Studio dashboard: {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
