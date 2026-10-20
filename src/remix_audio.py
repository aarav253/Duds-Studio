"""Re-mix an edit's audio (music + voicelines) and swap it into the rendered
video without re-rendering the picture.   python src/remix_audio.py <id> [...]"""
import json, os, subprocess, sys
from pathlib import Path
import imageio_ffmpeg
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
FF = imageio_ffmpeg.get_ffmpeg_exe()
LOUD = {"duck_db": 7.0, "voice_over_music": 2.8}

def mux(video, wav):
    tmp = video.with_suffix(".remux.mp4")
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "320k", "-shortest", str(tmp)], check=True)
    os.replace(tmp, video)

def remix(mid):
    p = ROOT / "edits" / f"{mid}.json"; m = json.loads(p.read_text(encoding="utf-8-sig")); m.update(LOUD)
    p.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    video = ROOT / "output" / f"{mid}.mp4"; wav = video.with_suffix(".remix.wav")
    if m.get("type") == "template_edit":
        t = json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig"))
        mux(video, ROOT / m.get("music_file", t["music"]))
        subprocess.run([sys.executable, str(ROOT / "src/voice_mix.py"), str(video), str(p)], check=True, capture_output=True)
        return
    if "segments" in m:
        import seg_edit; seg_edit.SegEdit(m).mix(wav)
    else:
        t = json.loads((ROOT / m["template"]).read_text(encoding="utf-8-sig"))
        if t.get("engine") == "focus":
            import focus_edit; focus_edit.FocusEdit(m).mix(wav)
        else:
            import reveal_edit; reveal_edit.Reveal(m).mix(wav)
    mux(video, wav); wav.unlink()

if __name__ == "__main__":
    for mid in sys.argv[1:]:
        remix(mid); print("louder voices:", mid, flush=True)
