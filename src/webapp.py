"""Duds Studio dashboard - http://127.0.0.1:5000 (local only, never exposed).

Wraps generate / build / upload / publish / schedule / stats, plus a media
library (drag-and-drop uploads, licensed image & music search) and a
manifest editor. Every action calls the same importable functions the CLI
scripts use - no shelling out.

Safety:
  - binds 127.0.0.1 only
  - every POST needs a per-launch token header (a random website can't forge
    a request that publishes/deletes a video - custom headers force a CORS
    preflight this app never approves)
  - Host header must be localhost (blocks DNS-rebinding)
  - publishing still requires an explicit confirm in the UI
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
from pathlib import Path

import requests
from flask import Flask, abort, jsonify, request, send_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import media_search  # noqa: E402
import review_queue as rq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EDITS = ROOT / "edits"
PORT = int(os.environ.get("DUDS_PORT", "5000"))
TOKEN = secrets.token_urlsafe(24)
MEDIA_ROOTS = {"output", "clips", "panels", "music", "narration_audio"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".oga", ".ogx"}
MAX_UPLOAD = 2 * 1024 ** 3

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD


# ==========================================================================
# Request guards
# ==========================================================================

@app.before_request
def _guard():
    host = (request.host or "").split(":")[0]
    if host not in ("127.0.0.1", "localhost"):
        abort(403)
    if request.method == "POST" and request.headers.get("X-Duds-Token") != TOKEN:
        abort(403)


def _safe_path(rel, roots=MEDIA_ROOTS):
    """Resolve a project-relative path, refusing anything outside allowed folders."""
    if not rel:
        abort(400)
    p = (ROOT / rel).resolve()
    try:
        parts = p.relative_to(ROOT.resolve()).parts
    except ValueError:
        abort(403)
    if not parts or parts[0] not in roots:
        abort(403)
    return p


def _rel(p):
    return str(Path(p).resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _clean_name(name, default="file"):
    stem, ext = os.path.splitext(os.path.basename(name or ""))
    stem = re.sub(r"[^A-Za-z0-9 _.-]+", "", stem).strip(" .") or default
    return stem[:80] + ext.lower()


def _clean_folder(name):
    return re.sub(r"[^A-Za-z0-9 _-]+", "", name or "").strip()[:60]


# ==========================================================================
# Background job runner (one job at a time; stdout captured into the job log)
# ==========================================================================

JOBS = {}
_JOB_Q = queue.Queue()
_job_counter = [0]


class _JobLog(io.TextIOBase):
    def __init__(self, job):
        self.job = job

    def write(self, s):
        if s:
            self.job["log"] += s
            if len(self.job["log"]) > 60000:
                self.job["log"] = self.job["log"][-50000:]
        return len(s)


def submit(name, fn, *args, **kwargs):
    _job_counter[0] += 1
    jid = str(_job_counter[0])
    JOBS[jid] = {"id": jid, "name": name, "status": "queued", "log": "", "result": None,
                 "created": time.time()}
    _JOB_Q.put((jid, fn, args, kwargs))
    return jid


def _worker():
    while True:
        jid, fn, args, kwargs = _JOB_Q.get()
        job = JOBS[jid]
        job["status"] = "running"
        log = _JobLog(job)
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                job["result"] = fn(*args, **kwargs)
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            job["log"] += "\n" + traceback.format_exc()
            job["status"] = "error"
            job["result"] = f"{type(e).__name__}: {e}"


threading.Thread(target=_worker, daemon=True).start()


# ==========================================================================
# State
# ==========================================================================

def _manifests():
    out = {}
    for p in sorted(EDITS.glob("*.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            m = {"id": p.stem, "_invalid": True}
        out[m.get("id") or p.stem] = {"file": p.name, "type": m.get("type", "edit"),
                                      "title": m.get("title", ""), "invalid": m.get("_invalid", False),
                                      "example": p.name.startswith("_")}
    return out


def _auth_state():
    import youtube_auth
    return {"client_secret": youtube_auth.CLIENT_SECRET.exists(), "authorized": youtube_auth.is_authorized()}


@app.get("/api/state")
def api_state():
    queue_ = rq.load()
    mans = _manifests()
    videos = []
    for key in sorted(set(queue_) | {k for k, v in mans.items() if not v["example"]}):
        e = dict(queue_.get(key, {}))
        e["key"] = key
        e.setdefault("status", "pending")
        e["manifest_file"] = mans.get(key, {}).get("file")
        e["type"] = mans.get(key, {}).get("type")
        e.setdefault("title", mans.get(key, {}).get("title", ""))
        out = ROOT / e.get("output_path", f"output/{key}.mp4")
        e["has_output"] = out.exists()
        e["output_rel"] = _rel(out) if out.exists() else None
        e["mtime"] = out.stat().st_mtime if out.exists() else 0
        videos.append(e)
    jobs = sorted(JOBS.values(), key=lambda j: -j["created"])[:15]
    return jsonify(videos=videos, manifests=mans, auth=_auth_state(),
                   providers=media_search.available_providers(),
                   jobs=[{k: j[k] for k in ("id", "name", "status", "result")} for j in jobs])


@app.get("/api/job/<jid>")
def api_job(jid):
    j = JOBS.get(jid) or abort(404)
    return jsonify(j)


# ==========================================================================
# Pipeline actions
# ==========================================================================

def _json():
    return request.get_json(force=True, silent=True) or {}


@app.post("/api/build")
def api_build():
    import batch_build
    d = _json()
    only = [d["key"]] if d.get("key") else None
    force = bool(d.get("force")) or bool(d.get("key"))
    return jsonify(job=submit(f"build {d.get('key') or 'all pending'}",
                              batch_build.run, force=force, only=only, logger=None))


@app.post("/api/upload")
def api_upload():
    import upload
    d = _json()
    if d.get("key"):
        return jsonify(job=submit(f"upload {d['key']} (private)", upload.upload_key, d["key"]))
    return jsonify(job=submit("upload all built (private)", upload.upload_all))


@app.post("/api/publish")
def api_publish():
    # the UI's explicit confirm() click is the human confirmation for going public
    import publish
    d = _json()
    if d.get("confirm") != "PUBLISH":
        abort(400)
    return jsonify(job=submit(f"PUBLISH {d['video_id']}", publish.publish_video, d["video_id"]))


@app.post("/api/schedule")
def api_schedule():
    import publish
    d = _json()
    return jsonify(job=submit(f"schedule {d['video_id']} @ {d['when']}", publish.schedule_video,
                              d["video_id"], d["when"]))


@app.post("/api/unschedule")
def api_unschedule():
    import publish
    d = _json()
    return jsonify(job=submit(f"unschedule {d['video_id']}", publish.unschedule_video, d["video_id"]))


@app.post("/api/delete_youtube")
def api_delete_youtube():
    import publish
    d = _json()
    if d.get("confirm") != "DELETE":
        abort(400)
    return jsonify(job=submit(f"delete private upload {d['video_id']}", publish.delete_video, d["video_id"]))


@app.post("/api/refresh_stats")
def api_refresh_stats():
    import analytics
    import analytics_detailed
    import publish

    def go():
        n = publish.sync_scheduled_status()
        print(f"scheduled -> published: {n}")
        print(analytics.refresh_all())
        if _json_detailed:
            print(analytics_detailed.refresh_all())
    _json_detailed = bool(_json().get("detailed"))
    return jsonify(job=submit("refresh stats", go))


@app.post("/api/auth")
def api_auth():
    import youtube_auth

    def go():
        yt = youtube_auth.get_youtube(interactive=True)
        me = yt.channels().list(part="snippet", mine=True).execute().get("items", [])
        return me[0]["snippet"]["title"] if me else "(no channel)"
    return jsonify(job=submit("connect YouTube account", go))


@app.post("/api/generate")
def api_generate():
    import generate_manifests
    d = _json()
    types = [t for t in d.get("types", []) if t in ("edit", "meme", "narration")] or ["edit"]
    return jsonify(job=submit(f"generate for {d['show']}", generate_manifests.generate_for_show,
                              d["show"], types))


@app.post("/api/thumbnail")
def api_thumbnail():
    import thumbnail
    d = _json()
    mf = EDITS / _manifest_file_for(d["key"])

    def go():
        out = thumbnail.make_thumbnail(str(mf), d.get("text") or None)
        m = json.loads(mf.read_text(encoding="utf-8-sig"))
        m["thumbnail_path"] = _rel(out)
        mf.write_text(json.dumps(m, indent=2), encoding="utf-8")
        return _rel(out)
    return jsonify(job=submit(f"thumbnail {d['key']}", go))


def _manifest_file_for(key):
    info = _manifests().get(key) or abort(404)
    return info["file"]


# ==========================================================================
# Manifest editor
# ==========================================================================

TEMPLATES = {
    "edit": {"id": "new_edit", "type": "edit", "source_clip": "clips/Show/ep1.mp4", "start": 0, "end": 30,
             "title": "", "description": "", "tags": ["shorts"], "music": None},
    "meme": {"id": "new_meme", "type": "meme", "source_clip": "clips/Show/ep1.mp4", "start": 0, "end": 12,
             "caption": "pov: ...", "title": "", "description": "", "tags": ["shorts"], "music": None},
    "narration": {"id": "new_narration", "type": "narration", "source_clip": "clips/Show/ep1.mp4",
                  "start": 0, "end": 40, "script": "Write 3-4 sentences here.",
                  "narration_audio": "narration_audio/new_narration.wav", "caption_style": "phrase",
                  "pop_keywords": [], "title": "", "description": "", "tags": ["shorts"]},
    "facts_split": {"id": "new_facts", "type": "narration", "layout": "split",
                    "script": "Write 3-4 sentences here.", "narration_audio": "narration_audio/new_facts.wav",
                    "gameplay_source": "clips/satisfying_bg.mp4", "gameplay_start": 0, "top_frac": 0.33,
                    "theme": "space", "segments": [{"query": "space", "start_frac": 0, "end_frac": 1}],
                    "title": "", "description": "", "tags": ["shorts", "facts"]},
    "manga_edit": {"id": "new_manga_edit", "type": "manga_edit", "panels": [], "music": "music/track.mp3",
                   "panel_fit": "crop", "auto_drop": True, "cut_on_onsets": True, "intro_caption": "pov: ...",
                   "title_overlays": [], "flash_beats": [], "title": "", "description": "", "tags": ["shorts"]},
    "manga_hype_edit": {"id": "new_hype", "type": "manga_hype_edit", "music": "music/track.mp3",
                        "width": 1080, "height": 1400, "intro_caption": "pov: ...",
                        "intro_duration_seconds": 6.0, "duration_seconds": 20.0,
                        "panels": ["(exactly 13 image paths)"], "title": "", "description": "",
                        "tags": ["shorts"]},
}


@app.get("/api/templates")
def api_templates():
    out = dict(TEMPLATES)
    try:
        import template_edit
        for name in template_edit.list_templates():
            out[f"template: {name}"] = template_edit.manifest_skeleton(name, f"new_{name}")
    except Exception as e:  # noqa: BLE001 - templates are optional
        print(f"[webapp] template list failed: {e}")
    return jsonify(out)


@app.get("/api/manifest/<key>")
def api_manifest_get(key):
    return (EDITS / _manifest_file_for(key)).read_text(encoding="utf-8-sig"), 200, {"Content-Type": "text/plain"}


@app.post("/api/manifest")
def api_manifest_save():
    d = _json()
    try:
        m = json.loads(d["content"])
    except (json.JSONDecodeError, KeyError) as e:
        return jsonify(error=f"invalid JSON: {e}"), 400
    mid = re.sub(r"[^A-Za-z0-9_-]+", "_", str(m.get("id", ""))).strip("_")
    if not mid:
        return jsonify(error="manifest needs an \"id\""), 400
    m["id"] = mid
    target = EDITS / f"{mid}.json"
    old = d.get("original_file")
    if old and old != target.name and (EDITS / old).exists():
        (EDITS / old).unlink()   # renamed id -> move file
    target.write_text(json.dumps(m, indent=2), encoding="utf-8")
    msg = ""
    if m.get("type") == "narration" and m.get("script") and m.get("narration_audio") and d.get("tts"):
        import tts
        submit(f"TTS for {mid}", tts.synthesize, m["script"], str(ROOT / m["narration_audio"]))
        msg = "TTS job queued"
    return jsonify(ok=True, file=target.name, msg=msg)


@app.post("/api/manifest/delete")
def api_manifest_delete():
    f = EDITS / _manifest_file_for(_json()["key"])
    f.unlink()
    return jsonify(ok=True)


# ==========================================================================
# Library: list / drag-and-drop upload / URL import / delete / serve
# ==========================================================================

def _lib_entry(p):
    info = media_search.read_license(p)
    return {"path": _rel(p), "name": p.name, "size": p.stat().st_size,
            "license": (info or {}).get("license"), "attribution": (info or {}).get("attribution")}


@app.get("/api/library")
def api_library():
    lib = {"clips": {}, "panels": {}, "music": []}
    for base, exts in (("clips", VIDEO_EXT), ("panels", IMAGE_EXT)):
        d = ROOT / base
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix.lower() in exts and "TopicImages" not in f.parts:
                folder = _rel(f.parent)[len(base) + 1:] or "(root)"
                lib[base].setdefault(folder, []).append(_lib_entry(f))
    for f in sorted((ROOT / "music").glob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXT:
            lib["music"].append(_lib_entry(f))
    return jsonify(lib)


def _target_dir(kind, folder):
    base = {"clips": "clips", "panels": "panels", "music": "music"}.get(kind) or abort(400)
    d = ROOT / base
    if kind != "music" and _clean_folder(folder):
        d = d / _clean_folder(folder)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _allowed_ext(kind):
    return {"clips": VIDEO_EXT, "panels": IMAGE_EXT, "music": AUDIO_EXT}[kind]


def _unique(d, name):
    p = d / name
    stem, ext = os.path.splitext(name)
    i = 2
    while p.exists():
        p = d / f"{stem}_{i}{ext}"
        i += 1
    return p


@app.post("/api/library/upload")
def api_library_upload():
    kind = request.form.get("kind", "")
    d = _target_dir(kind, request.form.get("folder"))
    mood = re.sub(r"[^a-z0-9]+", "", (request.form.get("mood") or "").lower())
    saved, skipped = [], []
    for f in request.files.getlist("files"):
        name = _clean_name(f.filename)
        if Path(name).suffix.lower() not in _allowed_ext(kind):
            skipped.append(f.filename)
            continue
        if kind == "music" and mood and not name.lower().startswith(mood + "_"):
            name = f"{mood}_{name}"
        p = _unique(d, name)
        f.save(p)
        saved.append(_rel(p))
    return jsonify(saved=saved, skipped=skipped)


@app.post("/api/library/import_url")
def api_library_import_url():
    """For a link dragged in from another browser tab. Only direct media files;
    you are responsible for having the rights to what you import."""
    d = _json()
    kind = d.get("kind", "")
    url = (d.get("url") or "").strip()
    if not re.match(r"^https?://", url):
        return jsonify(error="not an http(s) URL"), 400
    target = _target_dir(kind, d.get("folder"))
    try:
        r = requests.get(url, timeout=60, stream=True, headers=media_search.UA)
        r.raise_for_status()
        ext = media_search._ext_for(r, url, "")
        if ext not in _allowed_ext(kind):
            return jsonify(error=f"that link isn't a direct {kind} file ({ext or 'unknown type'})"), 400
        data = b""
        for chunk in r.iter_content(256 * 1024):
            data += chunk
            if len(data) > 500 * 1024 * 1024:
                return jsonify(error="file too large"), 400
    except requests.RequestException as e:
        return jsonify(error=str(e)), 400
    name = _clean_name(Path(url.split("?")[0]).stem + ext, "imported" + ext)
    mood = re.sub(r"[^a-z0-9]+", "", (d.get("mood") or "").lower())
    if kind == "music" and mood:
        name = f"{mood}_{name}"
    p = _unique(target, name)
    p.write_bytes(data)
    media_search.write_license(p, {"provider": "manual-url", "url": url, "license": "unknown - user supplied",
                                   "attribution": None})
    return jsonify(saved=_rel(p))


@app.post("/api/library/delete")
def api_library_delete():
    p = _safe_path(_json().get("path"), {"clips", "panels", "music"})
    if p.is_file():
        p.unlink()
        lp = media_search.license_path(p)
        if lp.exists():
            lp.unlink()
    return jsonify(ok=True)


@app.get("/api/search")
def api_search():
    try:
        res = media_search.search(request.args["kind"], request.args["q"],
                                  request.args.get("provider", "openverse"), 24)
        return jsonify(results=res)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=f"{type(e).__name__}: {e}"), 400


@app.post("/api/search/download")
def api_search_download():
    d = _json()
    try:
        return jsonify(saved=media_search.download(d["item"], d["kind"], d.get("folder"), d.get("mood")))
    except Exception as e:  # noqa: BLE001
        return jsonify(error=f"{type(e).__name__}: {e}"), 400


@app.get("/media/<path:rel>")
def media(rel):
    p = _safe_path(rel)
    if not p.is_file():
        abort(404)
    return send_file(p, mimetype=mimetypes.guess_type(p.name)[0], conditional=True)


# ==========================================================================
# Page
# ==========================================================================

@app.get("/")
def index():
    return PAGE.replace("__TOKEN__", TOKEN)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Duds Studio</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1e222b;--line:#2a2f3a;--text:#e7e9ee;--dim:#8b93a3;
--accent:#ffe066;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--blue:#60a5fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:12px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
header h1{font-size:18px;margin:0;letter-spacing:.5px}header h1 span{color:var(--accent)}
nav button{background:none;border:0;color:var(--dim);font:inherit;padding:6px 10px;border-radius:6px;cursor:pointer}
nav button.on{color:var(--text);background:var(--panel2)}
.auth{margin-left:auto;font-size:12px;color:var(--dim)}
main{padding:20px;max-width:1300px;margin:0 auto}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button.b{background:var(--panel2);color:var(--text);border:1px solid var(--line);padding:6px 11px;border-radius:7px;cursor:pointer;font:inherit}
button.b:hover{border-color:#3d4452}button.p{background:var(--accent);color:#111;border-color:var(--accent);font-weight:600}
button.d{color:var(--bad)}button:disabled{opacity:.45;cursor:default}
input,select,textarea{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:6px 8px;font:inherit}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin-top:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.card video{width:100%;aspect-ratio:9/14;background:#000;object-fit:contain}
.card .ph{aspect-ratio:9/14;display:flex;align-items:center;justify-content:center;color:var(--dim);background:#0b0c0f}
.card .body{padding:10px;display:flex;flex-direction:column;gap:6px}
.title{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill{display:inline-block;font-size:11px;padding:1px 7px;border-radius:99px;border:1px solid currentColor}
.s-pending{color:var(--dim)}.s-built{color:var(--blue)}.s-error{color:var(--bad)}.s-uploaded_private{color:var(--warn)}
.s-scheduled{color:#c084fc}.s-published{color:var(--ok)}
.small{font-size:12px;color:var(--dim)}.err{color:var(--bad);font-size:12px;white-space:pre-wrap}
.drop{border:2px dashed var(--line);border-radius:10px;padding:18px;text-align:center;color:var(--dim)}
.drop.over{border-color:var(--accent);color:var(--text)}
section.box{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px}
section.box h3{margin:0 0 10px;font-size:14px}
.files{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-top:8px}
.file{background:var(--panel2);border-radius:8px;padding:6px;font-size:11px;position:relative;word-break:break-all}
.file img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:5px}
.file .x{position:absolute;top:4px;right:4px;background:#000a;border:0;color:#fff;border-radius:5px;cursor:pointer}
.res{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin-top:10px}
.res .file img{aspect-ratio:4/3}
#jobs{position:fixed;right:16px;bottom:16px;width:380px;max-height:45vh;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px;font-size:12px;box-shadow:0 8px 30px #0008}
#jobs .j{padding:6px 10px;border-bottom:1px solid var(--line);cursor:pointer}
#joblog{white-space:pre-wrap;font:11px/1.4 Consolas,monospace;max-height:220px;overflow:auto;padding:8px 10px;background:#0b0c0f;display:none}
textarea.code{width:100%;min-height:420px;font:12px/1.5 Consolas,monospace}
.split{display:grid;grid-template-columns:260px 1fr;gap:16px}
.list button{display:block;width:100%;text-align:left;margin-bottom:4px}
@media (max-width:760px){.split{grid-template-columns:1fr}#jobs{width:auto;left:16px}}
</style></head><body>
<header><h1><span>Duds</span> Studio</h1>
<nav><button data-tab="videos" class="on">Videos</button><button data-tab="library">Library</button>
<button data-tab="find">Find media</button><button data-tab="manifests">Manifests</button><button data-tab="generate">Generate</button></nav>
<div class="auth" id="auth"></div></header>
<main>
<div id="tab-videos">
  <div class="row"><button class="b p" onclick="post('/api/build',{})">Build all pending</button>
  <button class="b" onclick="post('/api/upload',{})">Upload all built (private)</button>
  <button class="b" onclick="post('/api/refresh_stats',{detailed:true})">Refresh stats</button>
  <select id="vfilter" onchange="render()"><option value="">All statuses</option><option>pending</option><option>built</option><option>error</option><option>uploaded_private</option><option>scheduled</option><option>published</option></select>
  <span class="small">Uploads are always private. Only "Publish" (with confirmation) makes a video public.</span></div>
  <div class="grid" id="videos"></div>
</div>
<div id="tab-library" hidden>
  <section class="box"><h3>Clips (source footage) <span class="small">- clips/&lt;Show&gt;/</span></h3>
    <div class="row">Show folder <input id="clipfolder" placeholder="e.g. Blue Lock"></div>
    <div class="drop" data-kind="clips">Drop video files here (or click)</div><div id="lib-clips"></div></section>
  <section class="box"><h3>Panels / images <span class="small">- panels/&lt;folder&gt;/</span></h3>
    <div class="row">Folder <input id="panelfolder" placeholder="e.g. Sae Itoshi"></div>
    <div class="drop" data-kind="panels">Drop images here, or drag an image straight from another browser tab</div><div id="lib-panels"></div></section>
  <section class="box"><h3>Music <span class="small">- music/ (name prefix = mood, e.g. hype_track.mp3)</span></h3>
    <div class="row">Mood prefix <input id="musicmood" placeholder="hype / emotional / chill"></div>
    <div class="drop" data-kind="music">Drop audio files here (or click)</div><div id="lib-music"></div></section>
  <p class="small">Only add media you have the rights to use. Anything added from a pasted/dragged link is marked "unknown license".</p>
</div>
<div id="tab-find" hidden>
  <section class="box"><h3>Licensed media search</h3>
  <div class="row"><select id="fkind" onchange="fillProviders()"><option value="images">Images</option><option value="music">Music</option></select>
  <select id="fprov"></select><input id="fq" placeholder="search..." style="flex:1;min-width:200px" onkeydown="if(event.key==='Enter')doSearch()">
  <button class="b p" onclick="doSearch()">Search</button></div>
  <div class="row" style="margin-top:8px"><span class="small">Save images to panels/</span><input id="ffolder" placeholder="folder" style="width:160px">
  <span class="small">music mood prefix</span><input id="fmood" placeholder="hype" style="width:100px"></div>
  <p class="small">Results are Creative Commons / public domain / stock-licensed and filtered to licenses that allow commercial use (no NC/ND).
  CC BY / BY-SA credits are added to the YouTube description automatically on upload. Google results use Google's usage-rights filter, which isn't always accurate, so check the source page.</p>
  <div id="fres" class="res"></div></section>
</div>
<div id="tab-manifests" hidden>
  <div class="split"><div><div class="row" style="margin-bottom:8px"><select id="tpl"></select><button class="b" onclick="newFromTpl()">New</button></div>
  <div class="list" id="mlist"></div></div>
  <div><div class="row" style="margin-bottom:8px"><b id="mname">-</b><button class="b p" onclick="saveManifest(false)">Save</button>
  <button class="b" onclick="saveManifest(true)">Save + generate TTS</button><button class="b d" onclick="delManifest()">Delete manifest</button><span id="mmsg" class="small"></span></div>
  <textarea id="med" class="code" spellcheck="false"></textarea>
  <p class="small">Paths are relative to the project folder, e.g. panels/Sae Itoshi/01.png. Copy exact paths from the Library tab (click a file).</p></div></div>
</div>
<div id="tab-generate" hidden>
  <section class="box"><h3>Auto-generate manifests from episodes in clips/&lt;Show&gt;/</h3>
  <div class="row"><select id="gshow"></select>
  <label><input type="checkbox" class="gt" value="edit" checked> edit</label>
  <label><input type="checkbox" class="gt" value="meme" checked> meme</label>
  <label><input type="checkbox" class="gt" value="narration"> narration</label>
  <button class="b p" onclick="gen()">Generate</button></div>
  <p class="small">Scene detection picks clip windows, then Claude writes titles/captions/scripts (template fallback without ANTHROPIC_API_KEY).</p></section>
</div>
</main>
<div id="jobs"><div class="row" style="padding:6px 10px;border-bottom:1px solid var(--line);cursor:pointer" onclick="toggleJobs()"><b>Jobs</b><span class="small" id="jobsum"></span><span style="margin-left:auto" id="jobtog">▾</span></div>
<div id="jobbody"><div id="joblist"></div><div id="joblog"></div></div></div>
<script>
const TOKEN="__TOKEN__";let S={},LIB={},openJob=null,curManifest=null;
const $=s=>document.querySelector(s),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function post(url,body,isForm){const o={method:'POST',headers:{'X-Duds-Token':TOKEN}};
 if(isForm)o.body=body;else{o.headers['Content-Type']='application/json';o.body=JSON.stringify(body)}
 const r=await fetch(url,o);let j={};try{j=await r.json()}catch(e){}
 if(!r.ok){alert(j.error||('Request failed: '+r.status));return null}
 if(j.job){openJob=j.job;refresh()}return j}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('main>div').forEach(d=>d.hidden=d.id!=='tab-'+b.dataset.tab);
 if(b.dataset.tab==='library')loadLib();if(b.dataset.tab==='manifests')loadManifests();if(b.dataset.tab==='generate')loadShows();if(b.dataset.tab==='find')fillProviders()});
async function refresh(){S=await (await fetch('/api/state')).json();render();renderJobs()}
function render(){const a=S.auth||{};$('#auth').innerHTML=a.authorized?'YouTube: <span style="color:var(--ok)">connected</span>':
 (a.client_secret?'YouTube: not connected <button class="b" onclick="post(\'/api/auth\',{})">Connect</button>':'YouTube: add client_secret.json to connect');
 const f=$('#vfilter').value;const vs=(S.videos||[]).filter(v=>!f||v.status===f).sort((x,y)=>(y.mtime||0)-(x.mtime||0));
 $('#videos').innerHTML=vs.map(card).join('')||'<p class="small">No manifests yet. Create one in Manifests or Generate.</p>'}
function card(v){const id=v.video_id,st=v.status,k=esc(v.key);let btns=[];
 btns.push(`<button class="b" onclick="post('/api/build',{key:'${k}'})">${v.has_output?'Rebuild':'Build'}</button>`);
 if(v.has_output&&!id&&st==='built')btns.push(`<button class="b" onclick="post('/api/upload',{key:'${k}'})">Upload (private)</button>`);
 if(v.has_output)btns.push(`<button class="b" onclick="thumb('${k}')">Thumbnail</button>`);
 if(id&&st!=='published'){btns.push(`<button class="b p" onclick="pub('${id}','${k}')">Publish</button>`);
  btns.push(st==='scheduled'?`<button class="b" onclick="post('/api/unschedule',{video_id:'${id}'})">Unschedule</button>`:
   `<button class="b" onclick="sched('${id}')">Schedule</button>`);
  btns.push(`<button class="b d" onclick="delYT('${id}','${k}')">Delete upload</button>`)}
 if(id)btns.push(`<a class="b" style="text-decoration:none;padding:6px 11px;border:1px solid var(--line);border-radius:7px;color:var(--text)" href="https://youtu.be/${id}" target="_blank" rel="noopener">Open</a>`);
 const stats=v.stats?`<div class="small">views ${v.stats.views} · likes ${v.stats.likes} · comments ${v.stats.comments}</div>`:'';
 const qc=v.qc?`<div class="small">QC: <span class="pill ${v.qc.overall==='PASS'?'s-published':v.qc.overall==='WARN'?'s-uploaded_private':'s-error'}">${esc(v.qc.overall)}</span>
   ${v.qc.counts?` ${v.qc.counts.PASS}/${v.qc.counts.WARN}/${v.qc.counts.FAIL}`:''}
   ${v.qc.report?` <a href="/media/${encodeURI(v.qc.report)}/qc_report.md" target="_blank">report</a> · <a href="/media/${encodeURI(v.qc.report)}/cuts_01-04.png" target="_blank">frames</a>`:''}
   ${(v.qc.issues||[]).length?`<details><summary>${v.qc.issues.length} issue(s)</summary>${v.qc.issues.map(x=>`<div class="err">${esc(x)}</div>`).join('')}</details>`:''}</div>`:'';
 const det=v.detailed_stats&&v.detailed_stats.averageViewPercentage!=null?`<div class="small">avg viewed ${(+v.detailed_stats.averageViewPercentage).toFixed(1)}% · ${(+v.detailed_stats.averageViewDuration).toFixed(1)}s</div>`:'';
 return `<div class="card">${v.has_output?`<video src="/media/${encodeURI(v.output_rel)}?t=${v.mtime}" controls preload="metadata"></video>`:'<div class="ph">not built</div>'}
 <div class="body"><div class="title" title="${esc(v.title)}">${esc(v.title||v.key)}</div>
 <div class="row"><span class="pill s-${esc(st)}">${esc(st)}</span><span class="small">${esc(v.type||'')} · ${k}</span></div>
 ${v.scheduled_for?`<div class="small">goes public ${esc(v.scheduled_for.replace('T',' '))}</div>`:''}
 ${v.error?`<div class="err">${esc(v.error)}</div>`:''}${qc}${stats}${det}<div class="row">${btns.join('')}</div></div></div>`}
function pub(id,k){if(confirm(`Make "${k}" PUBLIC on YouTube now?\n\nThis is the only step that publishes. Anyone can see it afterwards.`))post('/api/publish',{video_id:id,confirm:'PUBLISH'})}
function delYT(id,k){if(confirm(`Permanently delete the PRIVATE upload of "${k}" from YouTube?\nThe local video file is kept.`))post('/api/delete_youtube',{video_id:id,confirm:'DELETE'})}
function sched(id){const d=new Date(Date.now()+3600e3);const def=new Date(d-d.getTimezoneOffset()*6e4).toISOString().slice(0,16);
 const w=prompt('Go public at (your local time, YYYY-MM-DDTHH:MM):',def);if(w)post('/api/schedule',{video_id:id,when:w})}
function thumb(k){const t=prompt('Thumbnail text (blank = title):','');if(t!==null)post('/api/thumbnail',{key:k,text:t})}
function renderJobs(){$('#joblist').innerHTML=(S.jobs||[]).map(j=>`<div class="j" onclick="openJob='${j.id}';showLog()"><span class="pill ${j.status==='error'?'s-error':j.status==='done'?'s-published':'s-built'}">${j.status}</span> ${esc(j.name)}${j.status==='error'?`<div class="err">${esc(j.result)}</div>`:''}</div>`).join('');
 const js=S.jobs||[],run=js.filter(j=>j.status==='running'||j.status==='queued').length;
 $('#jobsum').textContent=run?`${run} running`:`${js.length} recent`;
 $('#jobs').style.display=js.length?'block':'none';$('#jobbody').hidden=jobsCollapsed;$('#jobtog').textContent=jobsCollapsed?'▸':'▾';showLog()}
let jobsCollapsed=false;try{jobsCollapsed=localStorage.getItem('jobsCollapsed')==='1'}catch(e){}
function toggleJobs(){jobsCollapsed=!jobsCollapsed;try{localStorage.setItem('jobsCollapsed',jobsCollapsed?'1':'0')}catch(e){}renderJobs()}
async function showLog(){if(!openJob)return;const j=await (await fetch('/api/job/'+openJob)).json();const el=$('#joblog');el.style.display='block';
 el.textContent=`[${j.name}] ${j.status}\n`+j.log.slice(-6000);el.scrollTop=el.scrollHeight}
setInterval(()=>{if((S.jobs||[]).some(j=>j.status==='running'||j.status==='queued'))refresh()},2000);
// ---------- library ----------
async function loadLib(){LIB=await (await fetch('/api/library')).json();
 const grp=(o,kind)=>Object.entries(o).map(([f,items])=>`<div class="small" style="margin-top:8px">${esc(f)} (${items.length})</div><div class="files">${items.map(i=>fileTile(i,kind)).join('')}</div>`).join('')||'<p class="small">empty</p>';
 $('#lib-clips').innerHTML=grp(LIB.clips,'clips');$('#lib-panels').innerHTML=grp(LIB.panels,'panels');
 $('#lib-music').innerHTML=`<div class="files" style="grid-template-columns:1fr">${LIB.music.map(i=>fileTile(i,'music')).join('')||'<p class="small">empty</p>'}</div>`}
function fileTile(i,kind){const src='/media/'+encodeURI(i.path);const lic=i.license?`<div style="color:var(--dim)">${esc(i.license)}</div>`:'';
 const media=kind==='panels'?`<img src="${src}" loading="lazy">`:kind==='music'?`<audio src="${src}" controls preload="none" style="width:100%"></audio>`:'';
 return `<div class="file" title="click to copy path" onclick="copyPath(event,'${esc(i.path)}')">${media}<div>${esc(i.name)}</div>${lic}<button class="x" onclick="event.stopPropagation();delFile('${esc(i.path)}')">✕</button></div>`}
function copyPath(e,p){if(e.target.tagName==='AUDIO')return;navigator.clipboard&&navigator.clipboard.writeText(p);e.currentTarget.style.outline='1px solid var(--accent)';setTimeout(()=>loadLib(),600)}
async function delFile(p){if(confirm('Delete '+p+' from disk?')){await post('/api/library/delete',{path:p});loadLib()}}
function folderFor(kind){return kind==='clips'?$('#clipfolder').value:kind==='panels'?$('#panelfolder').value:''}
async function sendFiles(kind,files){if(kind==='clips'&&!$('#clipfolder').value.trim())return alert('Enter a show folder name first');
 const fd=new FormData();fd.append('kind',kind);fd.append('folder',folderFor(kind));fd.append('mood',$('#musicmood').value);
 [...files].forEach(f=>fd.append('files',f));const r=await post('/api/library/upload',fd,true);
 if(r&&r.skipped.length)alert('Skipped (wrong type): '+r.skipped.join(', '));loadLib()}
document.querySelectorAll('.drop').forEach(z=>{const kind=z.dataset.kind;
 z.ondragover=e=>{e.preventDefault();z.classList.add('over')};z.ondragleave=()=>z.classList.remove('over');
 z.ondrop=async e=>{e.preventDefault();z.classList.remove('over');
  if(e.dataTransfer.files.length)return sendFiles(kind,e.dataTransfer.files);
  const url=(e.dataTransfer.getData('text/uri-list')||e.dataTransfer.getData('text/plain')||'').split('\n')[0].trim();
  if(url){if(!confirm('Import this link?\n'+url+'\n\nOnly import media you have the rights to use.'))return;
   const r=await post('/api/library/import_url',{kind,url,folder:folderFor(kind),mood:$('#musicmood').value});if(r)loadLib()}};
 z.onclick=()=>{const i=document.createElement('input');i.type='file';i.multiple=true;
  i.accept=kind==='clips'?'video/*':kind==='panels'?'image/*':'audio/*';i.onchange=()=>sendFiles(kind,i.files);i.click()}});
// ---------- find ----------
function fillProviders(){const k=$('#fkind').value,p=(S.providers||{})[k]||{};
 $('#fprov').innerHTML=Object.entries(p).map(([n,ok])=>`<option value="${n}" ${ok?'':'disabled'}>${n}${ok?'':' (needs API key)'}</option>`).join('')}
let FRES=[];
async function doSearch(){const k=$('#fkind').value,q=$('#fq').value.trim();if(!q)return;$('#fres').innerHTML='<p class="small">searching…</p>';
 const r=await fetch(`/api/search?kind=${k}&provider=${$('#fprov').value}&q=${encodeURIComponent(q)}`);const j=await r.json();
 if(!r.ok){$('#fres').innerHTML=`<p class="err">${esc(j.error)}</p>`;return}FRES=j.results;
 $('#fres').innerHTML=FRES.map((it,i)=>`<div class="file">${k==='images'?`<img src="${esc(it.thumb)}" loading="lazy" referrerpolicy="no-referrer">`:`<audio src="${esc(it.url)}" controls preload="none" style="width:100%"></audio>`}
 <div>${esc(it.title).slice(0,70)}</div><div style="color:var(--dim)">${esc(it.creator)} · ${esc(it.license)}${it.duration?` · ${Math.round(it.duration)}s`:''}</div>
 <div class="row" style="margin-top:4px"><button class="b" onclick="grab(${i},this)">Add</button>${it.source_page?`<a class="small" href="${esc(it.source_page)}" target="_blank" rel="noopener">source</a>`:''}</div></div>`).join('')||'<p class="small">no results</p>'}
async function grab(i,btn){btn.disabled=true;btn.textContent='…';const k=$('#fkind').value;
 const r=await post('/api/search/download',{kind:k,item:FRES[i],folder:$('#ffolder').value||'downloads',mood:$('#fmood').value});
 btn.textContent=r?'Added ✓':'Add';if(!r)btn.disabled=false}
// ---------- manifests ----------
let TPL={};
async function loadManifests(){await refresh();TPL=await (await fetch('/api/templates')).json();$('#tpl').innerHTML=Object.keys(TPL).map(t=>`<option>${t}</option>`).join('');
 $('#mlist').innerHTML=Object.entries(S.manifests).map(([k,m])=>`<button class="b" onclick="openManifest('${esc(k)}')">${esc(k)} <span class="small">${esc(m.type)}${m.invalid?' · INVALID':''}</span></button>`).join('')}
async function openManifest(k){curManifest=k;$('#mname').textContent=S.manifests[k].file;$('#med').value=await (await fetch('/api/manifest/'+encodeURIComponent(k))).text();$('#mmsg').textContent=''}
function newFromTpl(){curManifest=null;$('#mname').textContent='(new)';$('#med').value=JSON.stringify(TPL[$('#tpl').value],null,2)}
async function saveManifest(tts){const r=await post('/api/manifest',{content:$('#med').value,original_file:curManifest?S.manifests[curManifest].file:null,tts});
 if(r){$('#mmsg').textContent='saved '+r.file+(r.msg?' · '+r.msg:'');await loadManifests();const id=r.file.replace(/\.json$/,'');if(S.manifests[id])curManifest=id}}
async function delManifest(){if(curManifest&&confirm('Delete manifest '+S.manifests[curManifest].file+'? (rendered video is kept)')){await post('/api/manifest/delete',{key:curManifest});curManifest=null;$('#med').value='';loadManifests()}}
// ---------- generate ----------
async function loadShows(){const l=await (await fetch('/api/library')).json();$('#gshow').innerHTML=Object.keys(l.clips).filter(s=>s!=='(root)').map(s=>`<option>${esc(s)}</option>`).join('')||'<option value="">(add clips first)</option>'}
function gen(){const show=$('#gshow').value;if(!show)return;post('/api/generate',{show,types:[...document.querySelectorAll('.gt:checked')].map(c=>c.value)})}
refresh();
</script></body></html>"""


def main():
    url = f"http://127.0.0.1:{PORT}"
    print(f"Duds Studio dashboard: {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
