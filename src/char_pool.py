"""Verified clip pools per character (dghs-imgutils CCIP).

Each character folder holds clips cut from a twixtor compilation of that
character, but some scenes show other players, the ball, crowds or the
disclaimer card. For every clip we take 3 frames, embed them with CCIP and keep
the clips whose best frame sits inside the folder's dominant identity cluster
(close to the medoid of all frames). Writes <char>/clips_pool.json:
  {"clips": [{path, frames, score, brightness}...]}  best match first

    python src/char_pool.py                 # every character
    python src/char_pool.py "Rin Itoshi"
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BL = ROOT / "panels" / "Blue Lock"


def frames(p, n=3):
    c = cv2.VideoCapture(str(p)); k = int(c.get(7)); out = []
    for a in np.linspace(0.2, 0.8, n):
        c.set(1, int(k * a)); ok, f = c.read()
        if ok:
            out.append(f)
    return out, k


def _ccip():
    """imgutils' CCIP imports scikit-learn's clustering at module level, and
    Windows Application Control blocks one of sklearn's compiled files on this
    machine. We only need feature extraction + difference (onnxruntime), so
    give it placeholder clustering classes instead of loading the blocked DLL."""
    import types
    if "sklearn.cluster" not in sys.modules:
        try:
            import sklearn.cluster  # noqa: F401
        except ImportError:
            fake = types.ModuleType("sklearn.cluster")
            fake.DBSCAN = fake.OPTICS = object
            sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
            sys.modules["sklearn.cluster"] = fake
    from imgutils.metrics import ccip_extract_feature, ccip_difference, ccip_default_threshold
    return ccip_extract_feature, ccip_difference, ccip_default_threshold


def pool(char):
    ccip_extract_feature, ccip_difference, ccip_default_threshold = _ccip()
    d = BL / char
    clips = sorted(p for p in d.glob("clips*/scene_*.mp4") if p.stem != "scene_00")
    feats, meta = [], []
    for p in clips:
        fs, k = frames(p)
        for f in fs:
            feats.append(ccip_extract_feature(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
            meta.append((p, k, float(f.mean())))
    n = len(feats)
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = ccip_difference(feats[i], feats[j])
    thr = ccip_default_threshold()
    # medoid of the biggest "same character" neighbourhood
    support = (D < thr).sum(1)
    core = np.argsort(-support)[: max(5, n // 5)]
    score = np.median(D[:, core], 1)                       # distance of every frame to the core
    best = {}
    for (p, k, br), s in zip(meta, score):
        if p not in best or s < best[p][0]:
            best[p] = (float(s), k, br)
    keep = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "frames": k, "score": round(s, 3), "brightness": round(br)}
            for p, (s, k, br) in sorted(best.items(), key=lambda kv: kv[1][0]) if s < thr * 1.1]
    old = d / "clips_pool.json"                            # keep framing labels from panel_refine.py
    if old.exists():
        lab = {c["path"]: c["framing"] for c in json.loads(old.read_text(encoding="utf-8-sig"))["clips"] if c.get("framing")}
        for c in keep:
            if c["path"] in lab:
                c["framing"] = lab[c["path"]]
    (d / "clips_pool.json").write_text(json.dumps({"threshold": thr, "clips": keep}, indent=1), encoding="utf-8")
    return keep, len(clips)


if __name__ == "__main__":
    chars = sys.argv[1:] or [p.name for p in BL.iterdir() if p.is_dir() and p.name != "backgrounds"]
    for c in chars:
        keep, total = pool(c)
        print(f"{c:18s} {len(keep):3d}/{total} clips verified", flush=True)
