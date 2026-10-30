"""Segment queue -> ASR -> translation -> terminal, with per-segment latency.

Latency of a line = time it is printed - moment the speaker stopped (segment.end_s
on the shared sample clock). It is split into:
  vad   - waiting for the end-of-phrase silence (min_silence) + capture delivery
  queue - waiting behind other segments (the single worker is busy)
  asr   - speech recognition
  mt    - translation
"""
from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from livetrans.segmenter import Segment

LABELS = {"mic": "Me", "loopback": "Remote"}
ARMENIAN_LETTER = re.compile("[\u0531-\u0556\u0561-\u0587]")
BUDGET_S = 5.0


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


@dataclass
class Stats:
    rows: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    empty_asr: int = 0
    max_queue: int = 0


class Worker(threading.Thread):
    """Single consumer: models run one segment at a time so they never fight for CPU."""

    def __init__(self, asr, mt, t0: float, log_dir: Path | None = None, save_audio: bool = False):
        super().__init__(daemon=True, name="worker")
        self.asr, self.mt, self.t0 = asr, mt, t0
        self.q: queue.Queue[Segment | None] = queue.Queue()
        self.stats = Stats()
        self.log_dir, self.save_audio = log_dir, save_audio
        self._print_lock = threading.Lock()
        self._log = open(log_dir / "segments.jsonl", "a", encoding="utf-8") if log_dir else None

    # called from capture threads
    def submit(self, seg: Segment) -> None:
        seg_t_queued = time.perf_counter()
        self.q.put((seg, seg_t_queued))
        depth = self.q.qsize()
        self.stats.max_queue = max(self.stats.max_queue, depth)
        if depth >= 3:
            self.say(f"!! backlog: {depth} segments waiting - latency will exceed the budget")

    def stop(self) -> None:
        self.q.put(None)

    def say(self, text: str) -> None:
        with self._print_lock:
            print(text, flush=True)

    def run(self) -> None:
        while (item := self.q.get()) is not None:
            seg, t_queued = item
            try:
                self._process(seg, t_queued)
            except Exception as e:  # report and keep going
                msg = f"{LABELS[seg.source]} @ {hms(seg.start_s)}: {type(e).__name__}: {e}"
                self.stats.failures.append(msg)
                self.say(f"!! processing failed - {msg}")
                traceback.print_exc()

    def _process(self, seg: Segment, t_queued: float) -> None:
        t_start = time.perf_counter()
        if self.asr is None:                     # --no-models diagnostic mode
            hy = "(models off)"
        else:
            hy = self.asr.transcribe(seg.audio)
        t_asr = time.perf_counter()
        # NLLB invents a sentence for input with no words (e.g. a lone "֊"), so skip it
        has_words = bool(ARMENIAN_LETTER.search(hy))
        en = self.mt.translate(hy) if (has_words and self.mt is not None) else ""
        t_mt = time.perf_counter()

        speech_end_wall = self.t0 + seg.end_s
        total = t_mt - speech_end_wall
        parts = {"vad": t_queued - speech_end_wall, "queue": t_start - t_queued,
                 "asr": t_asr - t_start, "mt": t_mt - t_asr}
        label = LABELS[seg.source]
        dur = len(seg.audio) / 16_000
        flag = "  !! OVER 5 s" if total > BUDGET_S else ""
        timing = (f"latency {total:.1f}s = vad {parts['vad']:.1f} + queue {parts['queue']:.1f} "
                  f"+ asr {parts['asr']:.1f} + mt {parts['mt']:.1f}  ({dur:.1f}s audio"
                  f"{', cut at max length' if seg.forced else ''}){flag}")
        if not has_words and self.asr is not None:
            self.stats.empty_asr += 1
            self.say(f"[{hms(seg.start_s)}] {label}   (no words recognised{': ' + hy if hy else ''}; {timing})")
        else:
            self.say(f"[{hms(seg.start_s)}] {label}\n  HY: {hy}\n  EN: {en}\n  .. {timing}")

        row = {"source": seg.source, "label": label, "start_s": round(seg.start_s, 2),
               "end_s": round(seg.end_s, 2), "audio_s": round(dur, 2), "forced": seg.forced,
               "hy": hy, "en": en, "latency_s": round(total, 3),
               **{f"{k}_s": round(v, 3) for k, v in parts.items()}}
        self.stats.rows.append(row)
        if self._log:
            self._log.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._log.flush()
        if self.save_audio and self.log_dir:
            from livetrans.audio.wavio import write_wav16k
            write_wav16k(self.log_dir / f"{seg.start_s:08.2f}_{label}.wav", seg.audio)


def summarize(stats: Stats) -> dict:
    out: dict = {"segments": len(stats.rows), "empty_asr": stats.empty_asr,
                 "failures": stats.failures, "max_queue_depth": stats.max_queue, "by_source": {}}
    for src, label in LABELS.items():
        lat = np.array([r["latency_s"] for r in stats.rows if r["source"] == src])
        if lat.size == 0:
            out["by_source"][label] = {"segments": 0}
            continue
        rows = [r for r in stats.rows if r["source"] == src]
        out["by_source"][label] = {
            "segments": int(lat.size),
            "latency_p50_s": round(float(np.percentile(lat, 50)), 2),
            "latency_p95_s": round(float(np.percentile(lat, 95)), 2),
            "latency_max_s": round(float(lat.max()), 2),
            "over_budget": int((lat > BUDGET_S).sum()),
            "mean_parts_s": {k: round(float(np.mean([r[f"{k}_s"] for r in rows])), 2)
                             for k in ("vad", "queue", "asr", "mt")},
        }
    return out


def print_summary(summary: dict, capture_problems: list[str]) -> None:
    print("\n=========== session summary ===========")
    print(f"segments: {summary['segments']}  (no words recognised: {summary['empty_asr']}), "
          f"max queue depth: {summary['max_queue_depth']}")
    for label, s in summary["by_source"].items():
        if not s["segments"]:
            print(f"{label:6}: no segments")
            continue
        p = s["mean_parts_s"]
        print(f"{label:6}: {s['segments']} segments, latency p50 {s['latency_p50_s']}s, "
              f"p95 {s['latency_p95_s']}s, max {s['latency_max_s']}s, over 5 s: {s['over_budget']}  "
              f"[mean vad {p['vad']} / queue {p['queue']} / asr {p['asr']} / mt {p['mt']}]")
    problems = capture_problems + [f"processing: {f}" for f in summary["failures"]]
    print("failures: " + ("none" if not problems else ""))
    for p in problems:
        print(f"  - {p}")
    sys.stdout.flush()
