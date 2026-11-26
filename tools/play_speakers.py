"""Play a known two-speaker sequence into the system audio, for a live speaker-label test.

  terminal 1:  python -m livetrans --mic 14 --loopback 16 --speakers --save-audio
  terminal 2:  python tools/play_speakers.py --list-outputs
               python tools/play_speakers.py --output <index of the device that loopback 16 records>

Takes the longest FEMALE and the longest MALE clip of data/fleurs_hy.jsonl (one reader
each, so each "speaker" is guaranteed to be one person), cuts each into pieces at quiet
points, and plays A1 B1 A2 B2 B3 A3 ... with gaps, through the chosen output device.
The live program hears it on the loopback like call audio. The exact wall-clock time of
every piece is saved to results/play_<timestamp>.json for tools/score_playback.py.
Limits: not a real call (no Meet processing); woman vs man is easier than similar voices.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from livetrans.audio.wavio import read_wav_float, to_16k_mono_float  # noqa: E402

ORDER = ["A", "B", "A", "B", "B", "A", "A", "B", "A", "B"]   # starts with A -> B -> A


def utf8_console() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def pieces(x: np.ndarray, n: int) -> list[np.ndarray]:
    """Cut x (16 kHz mono) into n pieces at the quietest 50 ms frame near each cut point."""
    frame = 800
    m = len(x) // frame
    lvl = (x[: m * frame].reshape(m, frame) ** 2).mean(axis=1)
    cuts = [0]
    for k in range(1, n):
        target = int(m * k / n)
        lo, hi = max(target - 20, 1), min(target + 20, m - 1)      # search +-1 s
        cuts.append((lo + int(np.argmin(lvl[lo:hi]))) * frame)
    cuts.append(len(x))
    return [x[a:b] for a, b in zip(cuts, cuts[1:])]


def main() -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-outputs", action="store_true")
    ap.add_argument("--output", type=int, help="WASAPI output device index (see --list-outputs)")
    ap.add_argument("--gap", type=float, default=1.5, help="silence between pieces, seconds")
    ap.add_argument("--peak", type=float, default=0.3, help="normalise each reader to this peak")
    args = ap.parse_args()

    import pyaudiowpatch as pyaudio
    import soxr

    with pyaudio.PyAudio() as pa:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        outs = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
        outs = [d for d in outs if d["hostApi"] == wasapi["index"] and d["maxOutputChannels"] > 0
                and not d.get("isLoopbackDevice")]
        if args.list_outputs or args.output is None:
            for d in outs:
                star = "*" if d["index"] == wasapi["defaultOutputDevice"] else " "
                print(f" {star}[{d['index']:>3}] {d['name']}  ({d['maxOutputChannels']} ch, {int(d['defaultSampleRate'])} Hz)")
            print("\nPick the output whose loopback you pass to livetrans (e.g. the Buds 'Headphones').")
            return 0 if args.list_outputs else 2
        dev = next((d for d in outs if d["index"] == args.output), None)
        if dev is None:
            print(f"ERROR: no WASAPI output device with index {args.output}")
            return 2

        rows = [json.loads(l) for l in (ROOT / "data" / "fleurs_hy.jsonl").read_text(encoding="utf-8").splitlines()]
        readers = {}
        for label, gender in (("A", "FEMALE"), ("B", "MALE")):
            best = max((r for r in rows if r["gender"] == gender),
                       key=lambda r: len(read_wav_float(ROOT / "data" / r["wav"])[0]))
            x = to_16k_mono_float(*read_wav_float(ROOT / "data" / best["wav"]))
            x = x * (args.peak / (np.abs(x).max() + 1e-9))
            n = ORDER.count(label)
            readers[label] = {"wav": best["wav"], "gender": gender, "pieces": pieces(x, n)}
        rate, ch = int(dev["defaultSampleRate"]), min(2, dev["maxOutputChannels"])
        stream = pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, output=True,
                         output_device_index=dev["index"])
        print(f"Playing on [{dev['index']}] {dev['name']}: {' '.join(ORDER)}  "
              f"(A = {readers['A']['wav'].split('/')[-1]}, B = {readers['B']['wav'].split('/')[-1]})")
        print("Starting in 3 s - keep the live program running in the other terminal.")
        time.sleep(3)
        used = {"A": 0, "B": 0}
        plan = []
        silence = np.zeros((int(rate * args.gap), ch), dtype=np.float32)
        for label in ORDER:
            piece = readers[label]["pieces"][used[label]]
            used[label] += 1
            y = soxr.resample(piece, 16_000, rate).astype(np.float32)
            y = np.repeat(y[:, None], ch, axis=1)
            t_start = time.time()
            stream.write(y.tobytes())
            t_end = time.time()
            plan.append({"label": label, "piece": used[label], "start_unix": round(t_start, 3),
                         "end_unix": round(t_end, 3), "audio_s": round(len(piece) / 16_000, 2)})
            print(f"  {label}{used[label]}  {len(piece) / 16_000:4.1f} s")
            stream.write(silence.tobytes())
        stream.stop_stream()
        stream.close()

    out = {"device": dev["name"], "order": ORDER, "gap_s": args.gap, "peak": args.peak,
           "readers": {k: {"wav": v["wav"], "gender": v["gender"]} for k, v in readers.items()},
           "plan": plan}
    (ROOT / "results").mkdir(exist_ok=True)
    path = ROOT / "results" / f"play_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved {path.relative_to(ROOT)} - now stop the live program (Ctrl+C), then run "
          f"tools/score_playback.py {path.relative_to(ROOT)} logs/<its folder>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
