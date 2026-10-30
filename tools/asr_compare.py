"""Score one ASR export on intact clips: WER/CER (normalised) and time per clip.

  python tools/asr_compare.py --model ctc  --set fleurs
  python tools/asr_compare.py --model rnnt --set fleurs
  python tools/asr_compare.py --model rnnt --quant int8 --set fleurs     # only after fp32 is judged
  python tools/asr_compare.py --model ctc  --set calls                   # eval/t-live-4_refs.jsonl

Sets: "fleurs" = data/fleurs_hy.jsonl (100 FLEURS hy_am test clips, read Eastern Armenian,
float32 audio read without rounding); "calls" = eval/t-live-4_refs.jsonl (user-corrected
references for intact Remote phrases of T-LIVE-4). Normalisation for WER/CER: lower case,
punctuation removed, եւ -> և (tools/m1_common.normalize_hy). Results go to
results/asr_<model>[_int8]_<set>.json with model revision, file hash and runtime versions.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from m1_common import normalize_hy  # noqa: E402


def utf8_console() -> None:
    """Windows consoles/pipes default to cp1252, which cannot encode Armenian."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

SETS = {"fleurs": (ROOT / "data" / "fleurs_hy.jsonl", ROOT / "data"),
        "calls": (ROOT / "eval" / "t-live-4_refs.jsonl", ROOT)}


def edits(ref: list, hyp: list) -> int:
    """Levenshtein distance between two token lists."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("ctc", "rnnt"), required=True)
    ap.add_argument("--quant", choices=("none", "int8"), default="none")
    ap.add_argument("--set", choices=sorted(SETS), default="fleurs")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    utf8_console()
    if args.model == "ctc" and args.quant != "none":
        ap.error("the CTC export has no int8 variant")

    from livetrans import models
    from livetrans.asr import Transcriber
    from livetrans.audio.wavio import read_wav_float, to_16k_mono_float

    manifest, base = SETS[args.set]
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.set == "calls":   # only intact (loopback) audio with a reference the user confirmed
        rows = [r for r in rows if r.get("intact") and r.get("ref_status") == "confirmed"]
        if not rows:
            raise SystemExit("no confirmed intact rows in eval/t-live-4_refs.jsonl yet")
    rows = rows[: args.limit]
    key = "asr-ctc" if args.model == "ctc" else "asr-rnnt"
    reg = models.MODELS[key]
    quant = None if args.quant == "none" else "int8"
    t = time.perf_counter()
    if args.model == "ctc":
        asr = Transcriber(reg["dir"])
        main_file = "model.onnx"
    else:
        asr = Transcriber(reg["dir"], model_type="nemo-conformer-rnnt", quantization=quant)
        main_file = "encoder-model.int8.onnx" if quant else "encoder-model.onnx"
    load_s = time.perf_counter() - t

    audio = [to_16k_mono_float(*read_wav_float(base / r["wav"])) for r in rows]
    asr.transcribe(audio[0])                                   # warm-up, not timed
    out, w_err, w_tot, c_err, c_tot, times, durs = [], 0, 0, 0, 0, [], []
    for r, a in zip(rows, audio):
        t = time.perf_counter()
        hyp = asr.transcribe(a)
        dt = time.perf_counter() - t
        ref_n, hyp_n = normalize_hy(r["ref_hy"]), normalize_hy(hyp)
        we, ce = edits(ref_n.split(), hyp_n.split()), edits(list(ref_n), list(hyp_n))
        w_err, w_tot, c_err, c_tot = w_err + we, w_tot + len(ref_n.split()), c_err + ce, c_tot + len(ref_n)
        times.append(dt)
        durs.append(len(a) / 16000)
        out.append({"wav": r["wav"], "ref_hy": r["ref_hy"], "hyp_hy": hyp, "word_errors": we,
                    "ref_words": len(ref_n.split()), "asr_s": round(dt, 3), "audio_s": round(len(a) / 16000, 2)})
        print(f"[{len(out):3}/{len(rows)}] {dt:5.2f}s  word errors {we}/{len(ref_n.split())}  {hyp[:60]}", flush=True)

    name = f"asr_{args.model}{'_int8' if quant else ''}_{args.set}"
    summary = {
        "model": args.model, "quantization": quant or "fp32", "set": args.set, "clips": len(out),
        "wer": round(w_err / max(w_tot, 1), 4), "cer": round(c_err / max(c_tot, 1), 4),
        "word_errors": w_err, "ref_words": w_tot,
        "asr_s_median": round(statistics.median(times), 3), "asr_s_max": round(max(times), 3),
        "rtf": round(sum(times) / sum(durs), 3), "load_s": round(load_s, 2),
        "provenance": {"repo": reg["repo"], "revision_pinned": reg["revision"],
                       "revision_downloaded": models.downloaded_revision(reg["dir"], main_file),
                       "main_file": main_file, "sha256": models._sha256(reg["dir"] / main_file),
                       "onnxruntime": version("onnxruntime"), "onnx-asr": version("onnx-asr"),
                       "python": sys.version.split()[0], "platform": sys.platform},
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps({"summary": summary, "clips": out}, indent=1,
                                                              ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
