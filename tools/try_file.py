"""Run the full chain on a recorded WAV: format conversion -> VAD -> ASR -> MT.

  python tools/try_file.py recordings/20261030-201834/mic.wav --source mic
  python tools/try_file.py recordings/20261030-201834/loopback.wav --source loopback --start 20 --end 35

Uses exactly the same conversion, segmentation and model code as the live
program, so a problem seen here is a problem in the pipeline, not in capture.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livetrans import models  # noqa: E402
from livetrans.audio.convert import To16kMono  # noqa: E402
from livetrans.audio.wavio import read_wav  # noqa: E402
from livetrans.pipeline import LABELS, hms  # noqa: E402
from livetrans.segmenter import Segmenter, SileroVad  # noqa: E402


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--source", choices=["mic", "loopback"], default="mic")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--vad-only", action="store_true", help="only show the segments, skip the models")
    ap.add_argument("--beam", type=int, default=4)
    args = ap.parse_args()

    pcm, rate = read_wav(args.wav)
    a = int(args.start * rate)
    b = int(args.end * rate) if args.end else len(pcm)
    pcm = pcm[a:b]
    print(f"{args.wav}: {pcm.shape[1]} ch, {rate} Hz, using {args.start:.1f}-{args.start + len(pcm) / rate:.1f} s")

    segments = []
    seg = Segmenter(args.source, SileroVad(models.vad_model_path()), segments.append)
    conv = To16kMono(rate)
    step = int(rate * 0.02)                       # feed 20 ms chunks, like live capture
    for i in range(0, len(pcm), step):
        seg.push(conv(pcm[i:i + step], last=i + step >= len(pcm)))
    seg.flush()
    print(f"VAD: {len(segments)} phrases ({seg.dropped_short} too-short blips dropped)\n")

    asr = mt = None
    if not args.vad_only:
        from livetrans.asr import Transcriber
        from livetrans.translate import Translator
        t = time.perf_counter()
        asr = Transcriber(models.ASR_DIR)
        mt = Translator(models.MT_DIR, beam_size=args.beam)
        print(f"models loaded in {time.perf_counter() - t:.1f} s\n")

    for s in segments:
        start = args.start + s.start_s
        head = (f"[{hms(start)}] {LABELS[s.source]}  {start:.2f}-{args.start + s.end_s:.2f} s "
                f"({len(s.audio) / 16000:.1f} s audio{', cut at max length' if s.forced else ''})")
        if asr is None:
            print(head)
            continue
        t = time.perf_counter()
        hy = asr.transcribe(s.audio)
        t_asr = time.perf_counter() - t
        t = time.perf_counter()
        en = mt.translate(hy) if hy else ""
        t_mt = time.perf_counter() - t
        print(f"{head}\n  HY: {hy or '(nothing recognised)'}\n  EN: {en}\n  .. asr {t_asr:.2f}s, mt {t_mt:.2f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
