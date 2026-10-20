"""Make a 1080x1350 intro still with the character's face centred (dghs-imgutils
anime face detection), from a clip frame.   face_still(clip, out, at=0.5, face_y=0.42)"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

def face_still(clip, out, at=0.5, face_y=0.42, W=1080, H=1350, body=False, zoom=1.0):
    from imgutils.detect import detect_faces
    c = cv2.VideoCapture(str(ROOT / clip)); n = int(c.get(7))
    best = None
    for a in (at, at - 0.15, at + 0.15, 0.3, 0.7):
        c.set(1, int(n * min(max(a, 0.05), 0.95))); ok, f = c.read()
        if not ok: continue
        faces = detect_faces(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        if faces:
            (x0, y0, x1, y1), _, score = max(faces, key=lambda d: d[2] * (d[0][2] - d[0][0]))
            best = (f, (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0); break
    if best is None:
        c.set(1, int(n * at)); ok, f = c.read(); best = (f, f.shape[1] / 2, f.shape[0] * 0.4, f.shape[0] * 0.3)
    f, cx, cy, fh = best
    h, w = f.shape[:2]
    s = max(W / w, H / h) * zoom
    img = cv2.resize(f, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_CUBIC)
    cx, cy = cx * s, cy * s
    x = int(np.clip(cx - W / 2, 0, img.shape[1] - W)); y = int(np.clip(cy - face_y * H, 0, img.shape[0] - H))
    cv2.imwrite(str(ROOT / out), img[y:y + H, x:x + W])
    return (cx - x) / W, (cy - y) / H

if __name__ == "__main__":
    print(face_still(sys.argv[1], sys.argv[2]))


def best_face_still(clips, out, W=1080, H=1350, face_h=0.30, face_y=0.40):
    """Search several clips/frames for a clear face that can sit centred, crop
    1080x1350 around it (zooming so the face is ~face_h of the height)."""
    from imgutils.detect import detect_faces
    best = None
    for clip in clips:
        c = cv2.VideoCapture(str(ROOT / clip)); n = int(c.get(7))
        for a in (0.25, 0.5, 0.75):
            c.set(1, int(n * a)); ok, f = c.read()
            if not ok:
                continue
            for (x0, y0, x1, y1), _, sc in detect_faces(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))):
                h, w = f.shape[:2]; s0 = max(W / w, H / h)
                fh = (y1 - y0) * s0
                z = float(np.clip(face_h * H / max(fh, 1), 1.0, 1.7))
                s = s0 * z; cx, cy = (x0 + x1) / 2 * s, (y0 + y1) / 2 * s
                X = np.clip(cx - W / 2, 0, w * s - W); Y = np.clip(cy - face_y * H, 0, h * s - H)
                dev = abs((cx - X) / W - 0.5) + abs((cy - Y) / H - face_y)
                score = sc - 2.0 * dev - 0.3 * (z - 1)
                if best is None or score > best[0]:
                    best = (score, f, s, int(X), int(Y), clip)
    if best is None:
        return None
    _, f, s, X, Y, clip = best
    img = cv2.resize(f, (int(round(f.shape[1] * s)), int(round(f.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(ROOT / out), img[Y:Y + H, X:X + W])
    return clip
