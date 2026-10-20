"""Wide (16:9, uncropped) version of a portrait scene clip, cut straight from its source video, plus the face's
vertical position in it -- for wide strip panels, where the portrait crop only leaves an eye-line band.

    from wide_clip import wide   ->  wide("panels/Blue Lock/Sae Itoshi/clips_x3/scene_02.mp4") = (path, face_y)
"""
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def wide(clip):
    p = ROOT / clip
    i = int(p.stem.split("_")[1])
    sc = json.loads((p.parent / "scenes.json").read_text(encoding="utf-8-sig"))
    s = next(x for x in sc["scenes"] if x["i"] == i)
    cr = json.loads((p.parent / "crops.json").read_text(encoding="utf-8-sig")).get(str(i), {})
    fy = float(np.mean(cr["cy"])) if cr.get("cy") else 0.4
    out = p.parent.parent / "clips_wide" / f"{p.parent.name}_{p.stem}.mp4"
    if not out.exists():
        out.parent.mkdir(exist_ok=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-ss", str(s["t0"]), "-to", str(s["t1"]),
                        "-i", str(ROOT / sc["source"]), "-vf", "scale=1080:-2", "-an", "-c:v", "libx264", "-crf", "16", str(out)],
                       check=True)
    return str(out.relative_to(ROOT)).replace("\\", "/"), round(fy, 3)
