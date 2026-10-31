"""Develop and check remote-speaker labels on saved phrases with known speakers.

  python tools/speaker_check.py eval/t-live-6_speakers.jsonl
  python tools/speaker_check.py eval/t-live-6_speakers.jsonl --threshold 0.45

Input: a JSONL file with one row per saved Remote phrase, in time order, with
"wav", "speaker" (A, B, ...; "?" = unknown) and "speaker_status" ("confirmed").
Steps (all on this machine, with the live program's own embedder and tracker):
 1. embed every clip >= min_embed_s; time each embedding;
 2. cosine similarity between clips of the SAME vs DIFFERENT speakers;
 3. replay all clips in time order through a fresh tracker for several thresholds;
    score agreement with the confirmed labels (each tracker speaker mapped to the
    true speaker it mostly contains) and count speakers created;
 4. explicit check A -> B -> A: first long A clip, then the next long B clip, then
    the next long A clip, through a fresh tracker; PASS if it prints
    Speaker 1 -> Speaker 2 -> Speaker 1.
Writes results/speaker_check_<input name>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from livetrans import models  # noqa: E402
from livetrans.audio.wavio import read_wav_float, to_16k_mono_float  # noqa: E402
from livetrans.speakers import SpeakerTracker, WeSpeakerEmbedder  # noqa: E402


# labels not scored: unknown, both voices, and the user's own voice echoed back through the remote phone
EXCLUDED = ("?", "X", "ME")


def utf8_console() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def replay(rows, audio, cache, threshold, tracker_kw):
    tr = SpeakerTracker(lambda a: cache[id(a)], threshold=threshold, **tracker_kw)
    return [tr.assign(audio[i]) for i in range(len(rows))]


def score(rows, decisions):
    known = [(r["speaker"], d.speaker) for r, d in zip(rows, decisions)
             if r.get("speaker_status") == "confirmed" and r["speaker"] not in EXCLUDED]
    mapping = {}
    for spk in {p for _, p in known}:
        mapping[spk] = Counter(t for t, p in known if p == spk).most_common(1)[0][0]
    correct = sum(mapping[p] == t for t, p in known)
    true_spk = {t for t, _ in known}
    return {"scored_clips": len(known), "agreement": round(correct / max(len(known), 1), 3),
            "tracker_speakers": len({d.speaker for d in decisions}), "true_speakers": len(true_spk),
            "mapping": {f"Speaker {k}": v for k, v in sorted(mapping.items())}}


def main() -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--threshold", type=float, default=0.30, help="threshold used for the A->B->A check")
    ap.add_argument("--sweep", type=float, nargs="+", default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.labels.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r.get("start_s", 0))
    audio = [to_16k_mono_float(*read_wav_float(ROOT / r["wav"])) for r in rows]
    durs = [len(a) / 16000 for a in audio]
    kw = {}                                     # tracker defaults (min_embed_s etc.) from livetrans.speakers
    defaults = SpeakerTracker(lambda a: a)
    min_embed, min_new = defaults.min_embed_s, defaults.min_new_s

    t = time.perf_counter()
    emb = WeSpeakerEmbedder(models.SPK_DIR)
    load_s = time.perf_counter() - t
    emb((0.003 * np.random.default_rng(0).standard_normal(24_000)).astype(np.float32))   # warm-up
    cache, times = {}, []
    for a, d in zip(audio, durs):
        if d >= min_embed:
            t = time.perf_counter()
            e = emb(a)
            times.append(time.perf_counter() - t)
            cache[id(a)] = e / np.linalg.norm(e)
    print(f"embedded {len(cache)} of {len(rows)} clips (>= {min_embed} s); load {load_s:.1f} s; "
          f"embedding time median {statistics.median(times) * 1000:.0f} ms, max {max(times) * 1000:.0f} ms")

    # 2. same vs different speaker similarity (confirmed, embedded clips only)
    lab = [(i, r["speaker"]) for i, r in enumerate(rows)
           if r.get("speaker_status") == "confirmed" and r["speaker"] not in EXCLUDED and id(audio[i]) in cache]
    same, diff = [], []
    for x in range(len(lab)):
        for y in range(x + 1, len(lab)):
            (i, si), (j, sj) = lab[x], lab[y]
            s = float(cache[id(audio[i])] @ cache[id(audio[j])])
            (same if si == sj else diff).append(s)
    def desc(v):
        return "n=0" if not v else (f"n={len(v)} mean {np.mean(v):.2f} min {min(v):.2f} "
                                    f"p10 {np.percentile(v, 10):.2f} p90 {np.percentile(v, 90):.2f} max {max(v):.2f}")
    print(f"cosine same speaker: {desc(same)}")
    print(f"cosine diff speaker: {desc(diff)}")

    # 3. threshold sweep, full replay in time order
    sweep = {}
    print("\nthreshold  agreement  tracker-speakers  true-speakers  mapping")
    for th in args.sweep:
        dec = replay(rows, audio, cache, th, kw)
        s = score(rows, dec)
        sweep[str(th)] = {**s, "labels": [d.speaker for d in dec], "reasons": [d.reason for d in dec]}
        print(f"{th:9.2f}  {s['agreement']:9.1%}  {s['tracker_speakers']:16d}  {s['true_speakers']:13d}  {s['mapping']}")

    # 4. A -> B -> A
    def next_long(spk, after):
        for i, r in enumerate(rows):
            if i > after and r.get("speaker_status") == "confirmed" and r["speaker"] == spk and durs[i] >= min_new:
                return i
        return None
    a1 = next_long("A", -1)
    b1 = next_long("B", a1) if a1 is not None else None
    a2 = next_long("A", b1) if b1 is not None else None
    aba = {"status": "NOT POSSIBLE", "reason": "no long A, B, A clips in order"}
    if None not in (a1, b1, a2):
        tr = SpeakerTracker(lambda a: cache[id(a)], threshold=args.threshold)
        dec = [tr.assign(audio[i]) for i in (a1, b1, a2)]
        got = [d.label for d in dec]
        ok = got == ["Speaker 1", "Speaker 2", "Speaker 1"]
        aba = {"status": "PASS" if ok else "FAIL", "clips": [rows[i]["wav"] for i in (a1, b1, a2)],
               "labels": got, "similarities": [d.similarity for d in dec], "threshold": args.threshold}
    print(f"\nA -> B -> A at threshold {args.threshold}: {aba['status']}"
          + (f"  {aba.get('labels')}  clips {[Path(c).name for c in aba.get('clips', [])]}" if "labels" in aba else
             f"  ({aba['reason']})"))

    out = {"input": str(args.labels), "clips": len(rows), "embedded": len(cache),
           "embed_ms_median": round(statistics.median(times) * 1000), "embed_ms_max": round(max(times) * 1000),
           "load_s": round(load_s, 2), "cos_same": same, "cos_diff": diff, "sweep": sweep, "aba": aba,
           "tracker": {k: getattr(defaults, k) for k in ("min_embed_s", "min_new_s", "min_update_s", "max_speakers")},
           "provenance": {"model_revision": models.downloaded_revision(models.SPK_DIR, "voxceleb_resnet34_LM.onnx"),
                          "onnxruntime": version("onnxruntime"), "onnx-asr": version("onnx-asr"),
                          "python": sys.version.split()[0], "platform": sys.platform}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"speaker_check_{args.labels.stem}.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
