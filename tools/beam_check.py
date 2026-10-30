"""Compare NLLB beam 1 and beam 4 on the phrases flagged in eval/t-live-4_refs.jsonl.

  python tools/beam_check.py

For each flagged phrase, translates (a) the user-corrected Armenian and (b) the
live ASR output, with beam 1 and beam 4. Time per translation = median of 5
repeats after a warm-up. English quality is judged by a person (no automatic
score on 5 phrases). Writes results/beam_check.json.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from livetrans import models
    from livetrans.translate import Translator

    rows = [json.loads(l) for l in (ROOT / "eval" / "t-live-4_refs.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows if r.get("beam_check")]
    unconfirmed = [r["wav"] for r in rows if r.get("ref_status") != "confirmed"]
    if unconfirmed:
        print("WARNING: references not confirmed yet for:", *unconfirmed, sep="\n  ")
    tr = Translator(models.MT_DIR)
    tr.translate("Բարև ձեզ, ո՞նց եք։")                       # warm-up
    results = []
    for r in rows:
        for kind in ("ref_hy", "asr_hy"):
            entry = {"wav": r["wav"], "input": kind, "hy": r[kind], "ref_status": r.get("ref_status")}
            for beam in (1, 4):
                tr.beam_size = beam
                times = []
                for _ in range(5):
                    t = time.perf_counter()
                    en = tr.translate(r[kind])
                    times.append(time.perf_counter() - t)
                entry[f"en_b{beam}"] = en
                entry[f"ms_b{beam}"] = round(statistics.median(times) * 1000)
            results.append(entry)
            print(f"\n{kind:6} HY: {entry['hy']}\n  beam1 ({entry['ms_b1']:4d} ms): {entry['en_b1']}"
                  f"\n  beam4 ({entry['ms_b4']:4d} ms): {entry['en_b4']}", flush=True)
    b1 = [e["ms_b1"] for e in results]
    b4 = [e["ms_b4"] for e in results]
    summary = {"phrases": len(rows), "translations_per_beam": len(results),
               "ms_median_b1": statistics.median(b1), "ms_median_b4": statistics.median(b4),
               "same_output": sum(e["en_b1"] == e["en_b4"] for e in results),
               "ctranslate2": version("ctranslate2"), "python": sys.version.split()[0], "platform": sys.platform}
    print("\n" + json.dumps(summary, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "beam_check.json").write_text(json.dumps({"summary": summary, "results": results},
                                                                  indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
