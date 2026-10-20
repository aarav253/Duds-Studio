"""Test a list of candidate songs (YouTube search, top result per query) against
several reference edits at once; writes music/unknown_matches.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_song import candidates
from fetch import fetch
from song_match import match
Q = ["MONTAGEM GLORIA ultra slowed", "MONTAGEM GLORIA slowed", "MONTAGEM DOMINANDO super slowed", "FUNK MANDALA", "INVOCADO funk",
     "FUNK POWER ultra slowed", "FUNK ECLIPSE slowed", "MONTAGEM FLUXO", "MONTAGEM AMBRA", "Montagem Zoro", "LOUCURA LETAL FUNK slowed",
     "MTG DIABOLICA", "MTG DIABOLICA super slowed", "NERVY FUNK ultra slowed", "FUNK MONOCLE ultra slowed", "Raya slowed funk", "SEM DEMORA slowed",
     "SEM DEMORA ultra slowed", "NO RADAR slowed funk", "MONTAGEM OSCURIDA ultra slowed", "MONTAGEM URANIUM", "BAILA LENTO funk", "SPACE CHESS slowed"]
refs = [1, 2, 9, 21]
d = ROOT / "music/candidates/unknown"; d.mkdir(parents=True, exist_ok=True)
res = {str(n): [] for n in refs}
rates = [round(0.80 + 0.02 * i, 2) for i in range(23)]
for q in Q:
    es = candidates(q, 1)
    if not es: continue
    e = es[0]; name = "u_" + e["id"]
    if not (d / f"{name}.wav").exists():
        for _ in range(3):
            try:
                fetch(f"https://www.youtube.com/watch?v={e['id']}", mode="audio", to=str(d.relative_to(ROOT)), name=name); break
            except Exception: pass
    f = d / f"{name}.wav"
    if not f.exists(): continue
    for n in refs:
        r = match(ROOT / f"music/template_{n}_ref.wav", f, rates=rates)
        score = min(r["confirm"]) if r["confirm"] else 0
        res[str(n)].append({"q": q, "title": e.get("title"), "url": f"https://youtu.be/{e['id']}", "file": str(f.relative_to(ROOT)).replace("\\", "/"), "score": score, **r})
        if score > 0.4: print(f"  HIT T{n}: {q} -> {score:.2f} rate {r['rate']}", flush=True)
    print("tested", q, flush=True)
    (ROOT / "music/unknown_matches.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
