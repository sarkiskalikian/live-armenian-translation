"""Inspect a live session's log and saved phrase audio (deterministic file analysis).

  python tools/inspect_session.py logs/20261031-173439

Separates what the files CAN show from what they cannot:
- capture: delivered ratio, unrecovered-lag events (summary.json), and runs of exact
  digital zeros inside saved phrases (mic: inserted by our aligner; loopback: usually
  the call app's own digital silence);
- phrase detection: phrases that start with speech already loud in the first 50 ms
  (the 0.2 s pre-roll may have clipped the onset), and phrases cut at the 10 s maximum;
- recognition: needs user references (eval/*.jsonl); not judged here.
Only detected phrases are saved (--save-audio). Speech that the VAD never detected,
and audio between phrases, is not on disk, so this tool cannot say whether any was missed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from livetrans.audio.wavio import read_wav_float  # noqa: E402

FRAME = 800                  # 50 ms at 16 kHz
ZERO_MIN = 160               # exact-zero runs >= 10 ms are reported
ONSET_HOT_DB = -6.0          # first 50 ms within 6 dB of the phrase's loud level = speech already under way


def zero_runs(x: np.ndarray, rate: int) -> list[tuple[float, float]]:
    z = np.concatenate([[0], (x == 0).astype(np.int8), [0]])
    d = np.diff(z)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(s / rate, (e - s) / rate) for s, e in zip(starts, ends) if e - s >= ZERO_MIN]


def db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    d = Path(sys.argv[1])
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (d / "segments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    wavs = {w.name: w for w in d.glob("*.wav")}
    print(f"session {d.name}: {len(rows)} phrases, {len(wavs)} saved WAVs")
    for label, c in summary.get("capture", {}).items():
        print(f"capture {label:6}: delivered {c['delivered_ratio']}, unrecovered-lag events "
              f"{c.get('unrecovered_events', c.get('loss_events'))} ({c.get('unrecovered_lag_s', c.get('lost_s'))} s), "
              f"overflows {c['overflows']}")
    out, counts = [], {"zero_mic": 0, "zero_remote": 0, "hot_onset": 0, "forced": 0, "missing_wav": 0}
    for r in rows:
        cand = [w for n, w in wavs.items() if n.endswith(f"_{r['label']}.wav")
                and abs(float(n.split("_")[0]) - r["start_s"]) < 0.02]
        if not cand:
            counts["missing_wav"] += 1
            continue
        x, rate = read_wav_float(cand[0])
        x = x.mean(axis=1)
        n = len(x) // FRAME
        frames = x[: n * FRAME].reshape(n, FRAME) if n else x[None, :]
        lvl = np.array([db(f) for f in frames])
        loud = float(np.percentile(lvl, 90))
        onset = float(lvl[0])
        zr = zero_runs(x, rate)
        hot = onset > loud + ONSET_HOT_DB
        counts["hot_onset"] += hot
        counts["forced"] += bool(r.get("forced"))
        if zr:
            counts["zero_mic" if r["source"] == "mic" else "zero_remote"] += 1
        out.append({"wav": cand[0].name, "label": r["label"], "audio_s": round(len(x) / rate, 2),
                    "onset_db": round(onset, 1), "loud_db": round(loud, 1), "hot_onset": hot,
                    "forced": r.get("forced", False), "zero_runs": [(round(a, 2), round(b * 1000)) for a, b in zr],
                    "hy": r["hy"]})
    for o in out:
        flags = []
        if o["hot_onset"]:
            flags.append(f"onset {o['onset_db']} dB vs loud {o['loud_db']} dB -> start may be clipped")
        if o["zero_runs"]:
            flags.append("exact zeros " + ", ".join(f"at {a}s {ms} ms" for a, ms in o["zero_runs"]))
        if o["forced"]:
            flags.append("cut at max length")
        if flags:
            print(f"{o['wav']:24} {o['hy'][:40]:42} " + "; ".join(flags))
    print(f"\nphrases with exact-zero runs: mic {counts['zero_mic']}, remote {counts['zero_remote']}; "
          f"possibly clipped onsets: {counts['hot_onset']}; cut at max length: {counts['forced']}; "
          f"phrases without a saved WAV: {counts['missing_wav']}")
    print("Not visible in these files: speech the VAD did not detect and any audio between phrases.")
    (d / "inspection.json").write_text(json.dumps({"counts": counts, "phrases": out}, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
