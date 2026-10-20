"""Deep comparison of a wekisava render vs its reference: fluidity, zoom,
lighting and transitions, section by section.

    python src/fluidity_metrics.py <render.mp4> [template.json]

Metrics (each printed REF vs OURS):
  zoom       per-shot log-scale curve error vs the reference (tracked ORB)
  jerk       frame-to-frame motion smoothness inside held shots: std of the
             change in global motion between consecutive frames (lower = smoother)
  flicker    frame-to-frame change of mean brightness inside shots
  light_lf   share of brightness variance that is low-frequency (big smooth
             pools of light vs small busy detail)
  dark       share of near-black pixels;  hi = share of bright pixels
  smear_coh  inside transitions: correlation of consecutive frames (a fluid
             slide keeps high coherence; per-frame random tearing drops it)
  smear_lum  per-frame brightness profile of transitions (shape error)
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_orb = cv2.ORB_create(1500)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def load(path, w=270):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, b = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(b, (w, int(w * b.shape[0] / b.shape[1])), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY).astype(np.float32))
    return out


def sim(a, b):
    k1, d1 = _orb.detectAndCompute(a.astype(np.uint8), None)
    k2, d2 = _orb.detectAndCompute(b.astype(np.uint8), None)
    if d1 is None or d2 is None:
        return None
    m = _bf.match(d1, d2)
    if len(m) < 15:
        return None
    p1 = np.float32([k1[x.queryIdx].pt for x in m])
    p2 = np.float32([k2[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl.sum() < 12:
        return None
    return math.hypot(M[0, 0], M[1, 0]), M[0, 2], M[1, 2]


def shot_stats(F, s, e):
    logs, mot = [0.0], []
    for k in range(s + 1, e + 1):
        r = sim(F[k - 1], F[k])
        if r and 0.85 < r[0] < 1.2:
            logs.append(logs[-1] + math.log(r[0]))
            mot.append((math.log(r[0]) * 100, r[1], r[2]))
        else:
            logs.append(logs[-1])
    mot = np.array(mot) if mot else np.zeros((1, 3))
    jerk = float(np.mean(np.abs(np.diff(mot, axis=0)))) if len(mot) > 2 else 0.0
    lum = np.array([F[k].mean() for k in range(s, e + 1)])
    flick = float(np.mean(np.abs(np.diff(lum)))) if len(lum) > 2 else 0.0
    return np.array(logs), jerk, flick


def light_stats(frames):
    lf, dark, hi = [], [], []
    for g in frames:
        low = cv2.GaussianBlur(g, (0, 0), 12)
        tot = g.var() + 1e-6
        lf.append(low.var() / tot)
        dark.append((g < 25).mean())
        hi.append((g > 170).mean())
    return float(np.mean(lf)), float(np.mean(dark)), float(np.mean(hi))


def smear_stats(F, s, e):
    coh = [float(np.corrcoef(F[k - 1].ravel(), F[k].ravel())[0, 1]) for k in range(s + 1, e + 1)]
    lum = np.array([F[k].mean() for k in range(s, e + 1)])
    return float(np.mean(coh)), lum


def compare(render, template=ROOT / "templates" / "video_edit_template_4.json", verbose=True):
    t = json.loads(Path(template).read_text(encoding="utf-8-sig"))
    R, Q = load(ROOT / t["reference"]), load(render)
    n = min(len(R), len(Q))
    rows = {"zoom_err": [], "jerk_r": [], "jerk_q": [], "flick_r": [], "flick_q": [],
            "coh_r": [], "coh_q": [], "smear_lum_err": []}
    sections = {"intro": [], "cyan": [], "eyes": []}
    for seg in t["timeline"]:
        s, e = seg["start"], min(seg["end"], n - 1)
        if seg["type"] == "shot" and e - s > 5 and not seg.get("black_frames"):
            lr, jr, fr_ = shot_stats(R, s, e)
            lq, jq, fq = shot_stats(Q, s, e)
            rows["zoom_err"].append(float(np.sqrt(np.mean((lr - lq) ** 2))) * 100)
            rows["jerk_r"].append(jr); rows["jerk_q"].append(jq)
            rows["flick_r"].append(fr_); rows["flick_q"].append(fq)
            sec = {"bw": "intro", "cyan": "cyan", "cyan_strip": "cyan", "eyes": "eyes"}[seg["look"]]
            sections[sec] += list(range(s + 2, e - 1, 3))
        elif seg["type"] == "smear" and e - s >= 3:
            cr, lr = smear_stats(R, s, e)
            cq, lq = smear_stats(Q, s, e)
            rows["coh_r"].append(cr); rows["coh_q"].append(cq)
            rows["smear_lum_err"].append(float(np.mean(np.abs(lr - lq))))
    res = {"zoom_err_pct": np.mean(rows["zoom_err"]),
           "jerk": (np.mean(rows["jerk_r"]), np.mean(rows["jerk_q"])),
           "flicker": (np.mean(rows["flick_r"]), np.mean(rows["flick_q"])),
           "smear_coherence": (np.mean(rows["coh_r"]), np.mean(rows["coh_q"])),
           "smear_lum_err": np.mean(rows["smear_lum_err"])}
    for name, ks in sections.items():
        if ks:
            res[f"light_{name}"] = (light_stats([R[k] for k in ks]), light_stats([Q[k] for k in ks]))
    if verbose:
        print(f"zoom curve error (avg % scale, lower better): {res['zoom_err_pct']:.2f}")
        print(f"motion jerk      REF {res['jerk'][0]:.3f}  OURS {res['jerk'][1]:.3f}")
        print(f"flicker          REF {res['flicker'][0]:.2f}  OURS {res['flicker'][1]:.2f}")
        print(f"smear coherence  REF {res['smear_coherence'][0]:.3f}  OURS {res['smear_coherence'][1]:.3f}")
        print(f"smear brightness profile error: {res['smear_lum_err']:.1f}")
        for name in sections:
            k = f"light_{name}"
            if k in res:
                (a, b, c), (d, e_, f) = res[k]
                print(f"light {name:5s}  low-freq REF {a:.2f} OURS {d:.2f} | dark REF {b:.2f} OURS {e_:.2f} | "
                      f"bright REF {c:.2f} OURS {f:.2f}")
    return res


if __name__ == "__main__":
    compare(sys.argv[1], *(sys.argv[2:3]))
