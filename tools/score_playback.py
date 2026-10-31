"""Score a live --speakers session against the known play plan from play_speakers.py.

  python tools/score_playback.py results/play_<ts>.json logs/<session>

Each Remote phrase is matched to the played piece it overlaps most (wall clock:
summary.json session_t0_unix + phrase times). Reports labels in order, agreement
(each Speaker N mapped to the reader it mostly contains), speakers created, the
first A -> B -> A, unmatched phrases, and latency including the speaker step.
Writes results/score_playback_<session>.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    plan_path, log = Path(sys.argv[1]), Path(sys.argv[2])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary = json.loads((log / "summary.json").read_text(encoding="utf-8"))
    t0 = summary.get("session_t0_unix")
    if t0 is None:
        raise SystemExit("summary.json has no session_t0_unix (log from an older version)")
    rows = [json.loads(l) for l in (log / "segments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    remote = [r for r in rows if r["source"] == "loopback"]
    matched, unmatched = [], []
    for r in remote:
        a, b = t0 + r["start_s"], t0 + r["end_s"]
        ov = [(max(0.0, min(b, p["end_unix"]) - max(a, p["start_unix"])), k) for k, p in enumerate(plan["plan"])]
        best, k = max(ov)
        if best <= 0:
            unmatched.append(r)
            continue
        p = plan["plan"][k]
        matched.append({"piece": f"{p['label']}{p['piece']}", "true": p["label"], "label": r["label"],
                        "speaker": r.get("speaker"), "reason": r.get("spk_reason"),
                        "similarity": r.get("spk_similarity"), "audio_s": r["audio_s"], "hy": r["hy"],
                        "latency_s": r["latency_s"], "spk_s": r.get("spk_s", 0.0)})
    for m in matched:
        print(f"{m['piece']:4} {m['true']} -> {m['label']:10} {m['audio_s']:4.1f}s  "
              f"{m['reason'] or '':14} sim {m['similarity']}  lat {m['latency_s']:.2f}s (spk {m['spk_s']:.3f})")
    pieces_heard = {m["piece"] for m in matched}
    missed = [f"{p['label']}{p['piece']}" for p in plan["plan"] if f"{p['label']}{p['piece']}" not in pieces_heard]
    mapping = {}
    for spk in {m["speaker"] for m in matched}:
        mapping[spk] = Counter(m["true"] for m in matched if m["speaker"] == spk).most_common(1)[0][0]
    correct = sum(mapping[m["speaker"]] == m["true"] for m in matched)
    # first A -> B -> A in the played order, using the first phrase heard of each piece
    first = {}
    for m in matched:
        first.setdefault(m["piece"], m["label"])
    order = [f"{p['label']}{p['piece']}" for p in plan["plan"]]
    aba = [first.get(order[i]) for i in range(3)]
    aba_ok = aba == ["Speaker 1", "Speaker 2", "Speaker 1"]
    lat = np.array([m["latency_s"] for m in matched]) if matched else np.array([0.0])
    spk = np.array([m["spk_s"] for m in matched]) if matched else np.array([0.0])
    out = {"plan": str(plan_path), "log": str(log), "remote_phrases": len(remote), "matched": len(matched),
           "unmatched": len(unmatched), "pieces_without_any_phrase": missed,
           "speakers_created": len(mapping), "mapping": {f"Speaker {k}": v for k, v in sorted(mapping.items())},
           "agreement": round(correct / max(len(matched), 1), 3),
           "aba_first_three_pieces": {"labels": aba, "status": "PASS" if aba_ok else "FAIL"},
           "latency_s": {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2),
                         "max": round(float(lat.max()), 2), "over_5s": int((lat > 5).sum())},
           "spk_s": {"median": round(float(np.median(spk)), 3), "max": round(float(spk.max()), 3)},
           "phrases": matched}
    print(json.dumps({k: v for k, v in out.items() if k != "phrases"}, indent=1, ensure_ascii=False))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"score_playback_{log.name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
