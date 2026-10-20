"""Search + download LICENSED images and music into the library folders.

Every source here returns content that is explicitly licensed for reuse
(Creative Commons / public domain / stock licenses). This deliberately does
NOT scrape Google Images or rip music: results there are mostly copyrighted,
and a video built on them risks Content ID claims or strikes on the channel.

Providers
  images: openverse (no key), pexels (PEXELS_API_KEY),
          google (GOOGLE_API_KEY + GOOGLE_CSE_ID, Programmable Search with the
                  Creative-Commons usage-rights filter)
  music:  openverse (no key; Freesound / Wikimedia / Jamendo CC audio),
          jamendo (JAMENDO_CLIENT_ID)

Each download gets a sidecar "<file>.license.json" with title, creator,
license and attribution text. upload.py appends required credits (CC BY /
BY-SA) to the YouTube description automatically.

    python src/media_search.py images "black hole" --provider openverse
    python src/media_search.py music "epic orchestral" --provider openverse
"""
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "DudsStudio/1.0 (local video tool)"}
# licenses allowed on a monetized channel (no NC / ND)
COMMERCIAL_OK = {"cc0", "pdm", "by", "by-sa", "pexels", "jamendo-cc-by", "jamendo-cc-by-sa"}
NEEDS_CREDIT = {"by", "by-sa"}


def _slug(s, n=40):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n] or "file"


# --------------------------------------------------------------------------
# Search providers -> list of normalized dicts:
# {provider, id, title, url, thumb, creator, license, license_url, source_page,
#  attribution, duration (music, seconds)}
# --------------------------------------------------------------------------

def _openverse(kind, query, n):
    endpoint = "images" if kind == "images" else "audio"
    r = requests.get(f"https://api.openverse.org/v1/{endpoint}/", headers=UA, timeout=20,
                     # anonymous access is capped at 20 per page (more returns 401)
                     params={"q": query, "license_type": "commercial", "page_size": min(n, 20)})
    r.raise_for_status()
    out = []
    for x in r.json().get("results", []):
        lic = (x.get("license") or "").lower()
        out.append({
            "provider": "openverse", "id": x.get("id"), "title": x.get("title") or "untitled",
            "url": x.get("url"), "thumb": x.get("thumbnail") or x.get("url"),
            "creator": x.get("creator") or "unknown",
            "license": lic, "license_version": x.get("license_version"),
            "license_url": x.get("license_url"), "source_page": x.get("foreign_landing_url"),
            "attribution": x.get("attribution"),
            "duration": (x.get("duration") or 0) / 1000.0 if kind == "music" else None,
        })
    return out


def _pexels(query, n):
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        raise RuntimeError("PEXELS_API_KEY is not set")
    r = requests.get("https://api.pexels.com/v1/search", headers={"Authorization": key},
                     params={"query": query, "per_page": min(n, 80)}, timeout=20)
    r.raise_for_status()
    return [{
        "provider": "pexels", "id": p["id"], "title": p.get("alt") or query,
        "url": p["src"]["large2x"], "thumb": p["src"]["medium"],
        "creator": p.get("photographer"), "license": "pexels",
        "license_url": "https://www.pexels.com/license/", "source_page": p.get("url"),
        "attribution": f'Photo by {p.get("photographer")} on Pexels',
    } for p in r.json().get("photos", [])]


def _google(query, n):
    key, cx = os.environ.get("GOOGLE_API_KEY"), os.environ.get("GOOGLE_CSE_ID")
    if not (key and cx):
        raise RuntimeError("GOOGLE_API_KEY and GOOGLE_CSE_ID must both be set")
    out = []
    for start in range(1, min(n, 30) + 1, 10):
        r = requests.get("https://www.googleapis.com/customsearch/v1", timeout=20, params={
            "key": key, "cx": cx, "q": query, "searchType": "image", "num": 10, "start": start,
            # Creative-Commons commercial-use filter (no NC / ND)
            "rights": "cc_publicdomain|cc_attribute|cc_sharealike",
            "safe": "active"})
        if r.status_code != 200:
            raise RuntimeError(f"Google API error {r.status_code}: {r.text[:200]}")
        for it in r.json().get("items", []):
            out.append({
                "provider": "google", "id": it.get("link"), "title": it.get("title"),
                "url": it.get("link"), "thumb": (it.get("image") or {}).get("thumbnailLink"),
                "creator": it.get("displayLink"), "license": "cc (google filter - verify)",
                "license_url": None, "source_page": (it.get("image") or {}).get("contextLink"),
                "attribution": f'{it.get("title")} via {it.get("displayLink")}',
            })
    return out[:n]


def _jamendo(query, n):
    cid = os.environ.get("JAMENDO_CLIENT_ID")
    if not cid:
        raise RuntimeError("JAMENDO_CLIENT_ID is not set")
    r = requests.get("https://api.jamendo.com/v3.0/tracks/", timeout=20, params={
        "client_id": cid, "format": "json", "limit": min(n, 50), "search": query,
        "audioformat": "mp32", "include": "licenses"})
    r.raise_for_status()
    out = []
    for t in r.json().get("results", []):
        lic_url = t.get("license_ccurl") or ""
        lic = "jamendo-" + ("cc-by-sa" if "by-sa" in lic_url else "cc-by" if "/by/" in lic_url else "other")
        if "-nc" in lic_url or "-nd" in lic_url:
            continue  # not usable on a monetized channel
        out.append({
            "provider": "jamendo", "id": t["id"], "title": t.get("name"), "url": t.get("audio"),
            "thumb": t.get("image"), "creator": t.get("artist_name"), "license": lic,
            "license_url": lic_url, "source_page": t.get("shareurl"),
            "attribution": f'"{t.get("name")}" by {t.get("artist_name")} ({lic_url})',
            "duration": t.get("duration"),
        })
    return out


PROVIDERS = {
    "images": {"openverse": lambda q, n: _openverse("images", q, n), "pexels": _pexels, "google": _google},
    "music": {"openverse": lambda q, n: _openverse("music", q, n), "jamendo": _jamendo},
}


def search(kind, query, provider="openverse", n=20):
    if kind not in PROVIDERS or provider not in PROVIDERS[kind]:
        raise ValueError(f"unknown {kind} provider: {provider}")
    results = PROVIDERS[kind][provider](query, n)
    return [r for r in results if r.get("url")]


def available_providers():
    return {
        "images": {"openverse": True, "pexels": bool(os.environ.get("PEXELS_API_KEY")),
                   "google": bool(os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_CSE_ID"))},
        "music": {"openverse": True, "jamendo": bool(os.environ.get("JAMENDO_CLIENT_ID"))},
    }


# --------------------------------------------------------------------------
# Download into the library
# --------------------------------------------------------------------------

def _ext_for(resp, url, default):
    ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
    ext = mimetypes.guess_extension(ct) if ct else None
    if ext in (".jpe", None):
        ext = Path(urlparse(url).path).suffix.lower() or default
    return {".jpeg": ".jpg", ".mpga": ".mp3"}.get(ext, ext)


def dest_dir(kind, folder):
    base = ROOT / ("music" if kind == "music" else "panels")
    d = base / _slug(folder) if folder else base
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(item, kind, folder=None, mood=None):
    """Download one search result into panels/<folder>/ (images) or music/
    (named "<mood>_title.ext" so generate_manifests' mood picker finds it).
    Writes a .license.json sidecar. Returns the saved path (project-relative)."""
    r = requests.get(item["url"], headers=UA, timeout=60, stream=True)
    r.raise_for_status()
    size_cap = 60 * 1024 * 1024
    data = b""
    for chunk in r.iter_content(64 * 1024):
        data += chunk
        if len(data) > size_cap:
            raise RuntimeError("file too large (>60MB)")
    ext = _ext_for(r, item["url"], ".mp3" if kind == "music" else ".jpg")
    allowed = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".oga", ".ogx"} if kind == "music" else \
              {".jpg", ".png", ".webp", ".gif"}
    if ext not in allowed:
        raise RuntimeError(f"unexpected file type {ext}")
    stem = _slug(item.get("title") or "file")
    if kind == "music" and mood:
        stem = f"{_slug(mood, 15)}_{stem}"
    d = dest_dir(kind, folder if kind != "music" else None)
    path = d / f"{stem}{ext}"
    i = 2
    while path.exists():
        path = d / f"{stem}_{i}{ext}"
        i += 1
    path.write_bytes(data)
    write_license(path, {k: item.get(k) for k in (
        "provider", "title", "creator", "license", "license_version", "license_url",
        "source_page", "attribution", "url")})
    return str(path.relative_to(ROOT)).replace("\\", "/")


def license_path(p):
    p = Path(p)
    return p.with_name(p.name + ".license.json")


def write_license(p, info):
    license_path(p).write_text(json.dumps(info, indent=2), encoding="utf-8")


def read_license(p):
    lp = license_path(ROOT / p if not Path(p).is_absolute() else p)
    if lp.exists():
        try:
            return json.loads(lp.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            return None
    return None


def credits_for_manifest(m):
    """Attribution lines required for the assets a manifest uses."""
    paths = list(m.get("panels", [])) + [m.get("music"), m.get("source_clip"), m.get("gameplay_source")]
    lines, seen = [], set()
    for p in paths:
        if not p:
            continue
        info = read_license(p)
        if not info:
            continue
        lic = (info.get("license") or "").lower()
        needs = lic in NEEDS_CREDIT or "cc-by" in lic or "verify" in lic
        line = info.get("attribution") or f'{info.get("title")} by {info.get("creator")}'
        if needs and line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["images", "music"])
    ap.add_argument("query")
    ap.add_argument("--provider", default="openverse")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--download", type=int, default=0, help="download the first N results")
    ap.add_argument("--folder", help="panels/<folder> for images")
    ap.add_argument("--mood", help="filename prefix for music (e.g. hype)")
    a = ap.parse_args()
    res = search(a.kind, a.query, a.provider, a.n)
    for i, it in enumerate(res):
        print(f"[{i}] {it['title'][:60]!r} by {it['creator']} - {it['license']}  {it['source_page']}")
    for it in res[:a.download]:
        print("saved", download(it, a.kind, a.folder, a.mood))
