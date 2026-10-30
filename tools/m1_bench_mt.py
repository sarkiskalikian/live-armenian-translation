"""M1 step 3: translation quality, speed and memory on this machine.

  python tools/m1_bench_mt.py             # needs data/fleurs_hy.jsonl (+ results/m1_asr_fleurs.jsonl)

Translates the FLEURS Armenian sentences one at a time (as in live use), from
  - the clean reference text, and
  - the ASR output from m1_bench_asr.py (the realistic, chained case),
with beam sizes 1 and 4, against the FLEURS/FLORES English references.
Writes results/m1_mt.jsonl and results/m1_mt_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m1_common import DATA, RESULTS, ROOT, machine_info, peak_ram_mb, read_jsonl, write_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--beams", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import sacrebleu
    from livetrans.translate import Translator

    rows = read_jsonl(DATA / "fleurs_hy.jsonl")[: args.limit]
    asr_path = RESULTS / "m1_asr_fleurs.jsonl"
    asr_hyp = {r["id"]: r["hyp_hy"] for r in read_jsonl(asr_path)} if asr_path.exists() else {}
    inputs = {"clean": [r["ref_hy"] for r in rows]}
    if asr_hyp:
        rows = [r for r in rows if r["id"] in asr_hyp]
        inputs = {"clean": [r["ref_hy"] for r in rows], "asr": [asr_hyp[r["id"]] for r in rows]}
    else:
        print("note: results/m1_asr_fleurs.jsonl not found; only clean-input MT is measured")
    refs = [r["ref_en"] for r in rows]

    t = time.perf_counter()
    tr = Translator(ROOT / "models/mt-nllb-600m-int8", threads=args.threads)
    load_s = time.perf_counter() - t
    tr.translate("Բարև։")  # warm-up

    summary = {"sentences": len(rows), "model_load_s": round(load_s, 2), "threads": args.threads,
               "runs": {}, "machine": machine_info()}
    per_sentence = [{"id": r["id"], "ref_hy": r["ref_hy"], "ref_en": r["ref_en"],
                     "asr_hy": asr_hyp.get(r["id"])} for r in rows]
    for beam in args.beams:
        tr.beam_size = beam
        for cond, texts in inputs.items():
            hyps, lat = [], []
            for i, text in enumerate(texts):
                t = time.perf_counter()
                hyps.append(tr.translate(text))
                lat.append(time.perf_counter() - t)
                per_sentence[i][f"en_{cond}_b{beam}"] = hyps[-1]
            lat = np.array(lat)
            key = f"{cond}_beam{beam}"
            summary["runs"][key] = {
                "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
                "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
                "ms_p50": round(float(np.percentile(lat, 50)) * 1000),
                "ms_p95": round(float(np.percentile(lat, 95)) * 1000),
                "ms_max": round(float(lat.max()) * 1000),
            }
            print(key, summary["runs"][key])
    summary["peak_ram_mb"] = peak_ram_mb()

    RESULTS.mkdir(exist_ok=True)
    write_jsonl(RESULTS / "m1_mt.jsonl", per_sentence)
    (RESULTS / "m1_mt_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print(f"peak RAM {summary['peak_ram_mb']} MB, load {summary['model_load_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
