"""M1 step 2: ASR accuracy, speed and memory on this machine.

  python tools/m1_bench_asr.py --set fleurs          # FLEURS Eastern Armenian read speech
  python tools/m1_bench_asr.py --set own             # your own read sentences (m1_record_reading.py)
  python tools/m1_bench_asr.py --set fleurs --threads 2

Writes results/m1_asr_<set>.jsonl (per clip) and results/m1_asr_<set>_summary.json.
WER/CER use normalised text (no case/punctuation); see m1_common.normalize_hy.
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

from m1_common import DATA, RESULTS, ROOT, machine_info, normalize_hy, peak_ram_mb, read_jsonl, \
    read_wav_16k_mono, write_jsonl  # noqa: E402

MANIFESTS = {"fleurs": DATA / "fleurs_hy.jsonl", "own": DATA / "own_reading.jsonl"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(MANIFESTS), default="fleurs")
    ap.add_argument("--threads", type=int, default=None, help="ONNX Runtime threads (default: runtime decides)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import jiwer
    from livetrans.asr import Transcriber

    rows = read_jsonl(MANIFESTS[args.set])[: args.limit]
    ram_before = peak_ram_mb()
    t = time.perf_counter()
    asr = Transcriber(ROOT / "models/asr-hy-fastconformer", threads=args.threads)
    load_s = time.perf_counter() - t
    asr.transcribe(np.zeros(16_000, dtype=np.float32))  # warm-up, not timed

    out = []
    for i, r in enumerate(rows, 1):
        audio = read_wav_16k_mono(DATA / r["wav"])
        t = time.perf_counter()
        hyp = asr.transcribe(audio)
        dt = time.perf_counter() - t
        dur = len(audio) / 16_000
        ref_n, hyp_n = normalize_hy(r["ref_hy"]), normalize_hy(hyp)
        out.append({**r, "hyp_hy": hyp, "audio_s": round(dur, 2), "asr_s": round(dt, 3),
                    "rtf": round(dt / dur, 3),
                    "wer": round(jiwer.wer(ref_n, hyp_n), 3) if ref_n else None,
                    "cer": round(jiwer.cer(ref_n, hyp_n), 3) if ref_n else None})
        print(f"[{i:3}/{len(rows)}] {dur:5.1f}s audio  {dt:5.2f}s asr  WER {out[-1]['wer']}")

    refs = [normalize_hy(r["ref_hy"]) for r in out]
    hyps = [normalize_hy(r["hyp_hy"]) for r in out]
    lat = np.array([r["asr_s"] for r in out])
    durs = np.array([r["audio_s"] for r in out])
    summary = {
        "set": args.set, "clips": len(out), "audio_minutes": round(durs.sum() / 60, 2),
        "corpus_wer": round(jiwer.wer(refs, hyps), 4), "corpus_cer": round(jiwer.cer(refs, hyps), 4),
        "rtf_mean": round(lat.sum() / durs.sum(), 3),
        "asr_s_p50": round(float(np.percentile(lat, 50)), 3),
        "asr_s_p95": round(float(np.percentile(lat, 95)), 3),
        "asr_s_max": round(float(lat.max()), 3),
        "longest_clip_s": round(float(durs.max()), 2),
        "model_load_s": round(load_s, 2),
        "peak_ram_mb": peak_ram_mb(), "ram_before_load_mb": ram_before,
        "threads": args.threads or "default", "machine": machine_info(),
    }
    RESULTS.mkdir(exist_ok=True)
    write_jsonl(RESULTS / f"m1_asr_{args.set}.jsonl", out)
    (RESULTS / f"m1_asr_{args.set}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "machine"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
