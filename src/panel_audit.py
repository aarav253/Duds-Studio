"""Audit every character clip for framing: how much the 4:5 crop zoomed into the
source, how big the face is in the crop, whether the crop cuts the face off.

    python src/panel_audit.py <scratchpad>
Writes <scratchpad>/audit.json and one contact sheet per character
(<scratchpad>/audit_<char>.jpg: current crop | source frame with the crop box).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BL = ROOT / "panels/Blue Lock"


def _faces(bgr):
    from imgutils.detect import detect_faces, detect_heads
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    f = detect_faces(im) or []
    h = detect_heads(im) or []
    return f, h


def _biggest(dets, H):
    if not dets:
        return None
    (x0, y0, x1, y1), _, sc = max(dets, key=lambda d: (d[0][3] - d[0][1]) * d[2])
    return [x0, y0, x1, y1, sc]


def _frame(path, at):
    c = cv2.VideoCapture(str(path)); n = int(c.get(7)); c.set(1, int(n * at)); ok, f = c.read(); c.release()
    return f if ok else None


def audit_clip(clip, scene, src_cap, crops):
    W, H = 1080, 1350
    r = {"clip": str(clip.relative_to(ROOT)).replace("\\", "/")}
    hs = crops.get(str(scene), {}).get("h")
    r["crop_h"] = round(float(np.mean(hs)), 3) if hs else None
    face_h, clipped, head_h = [], 0, []
    for at in (0.25, 0.5, 0.75):
        f = _frame(clip, at)
        if f is None:
            continue
        fs, hd = _faces(f)
        b = _biggest(fs, H)
        if b:
            face_h.append((b[3] - b[1]) / H)
            if b[0] < 0.01 * W or b[2] > 0.99 * W or b[1] < 0.01 * H or b[3] > 0.99 * H:
                clipped += 1
        hb = _biggest(hd, H)
        if hb:
            head_h.append((hb[3] - hb[1]) / H)
    r["face_h"] = round(float(np.median(face_h)), 3) if face_h else None
    r["head_h"] = round(float(np.median(head_h)), 3) if head_h else None
    r["clipped"] = clipped
    return r


def main(sp):
    sp = Path(sp)
    out = []
    only = set(sys.argv[2:])
    for char in sorted(p for p in BL.iterdir() if p.is_dir() and p.name != "backgrounds" and (not only or p.name in only)):
        rows = []
        for cd in sorted(char.glob("clips*")):
            sj = cd / "scenes.json"
            if not sj.exists():
                continue
            S = json.loads(sj.read_text(encoding="utf-8-sig"))
            crops = json.loads((cd / "crops.json").read_text(encoding="utf-8-sig")) if (cd / "crops.json").exists() else {}
            cap = cv2.VideoCapture(str(ROOT / S["source"]))
            for s in S["scenes"]:
                clip = cd / f"scene_{s['i']:02d}.mp4"
                if not clip.exists() or s["i"] == 0:
                    continue
                r = audit_clip(clip, s["i"], cap, crops)
                r.update(char=char.name, folder=cd.name, scene=s["i"])
                cap.set(1, (s["start"] + s["end"]) // 2); ok, src = cap.read()
                fs, hd = _faces(src) if ok else ([], [])
                b = _biggest(fs, src.shape[0]) if ok else None
                r["src_face_h"] = round((b[3] - b[1]) / src.shape[0], 3) if b else None
                r["src_face_cx"] = round((b[0] + b[2]) / 2 / src.shape[1], 3) if b else None
                hb = _biggest(hd, src.shape[0]) if ok else None
                r["src_head_cx"] = round((hb[0] + hb[2]) / 2 / src.shape[1], 3) if hb else None
                r["src_head_h"] = round((hb[3] - hb[1]) / src.shape[0], 3) if hb else None
                rows.append((r, clip, src))
                out.append(r)
                print(f"{char.name[:10]:10} {cd.name}/{s['i']:02d} crop_h {r['crop_h']} face {r['face_h']} head {r['head_h']} "
                      f"clip {r['clipped']} | src face {r['src_face_h']} head {r['src_head_h']}", flush=True)
        tiles = []
        for r, clip, src in rows:
            a = cv2.resize(_frame(clip, 0.5), (216, 270))
            b = cv2.resize(src, (480, 270))
            cv2.putText(a, f"{r['folder'][-1] if r['folder'] != 'clips' else ''}{r['scene']:02d}", (4, 24), 0, 0.8, (0, 255, 255), 2)
            tiles.append(np.hstack([a, b, np.zeros((270, 8, 3), np.uint8)]))
        if tiles:
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
            cv2.imwrite(str(sp / f"audit_{char.name.split()[0]}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    (sp / ("audit_new.json" if only else "audit.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
