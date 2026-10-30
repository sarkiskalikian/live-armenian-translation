"""M1 (optional but important): record yourself reading FLEURS sentences.

  python tools/m1_record_reading.py --mic 14 --n 15

Shows Armenian sentences from the same FLEURS subset used in m1_bench_asr.py.
Read each one naturally, the way you normally speak (don't imitate a standard
accent): press Enter to start, Enter to stop; 's' + Enter skips, 'r' + Enter redoes.
Because the sentences are the same, your WER can be compared directly with the
FLEURS speakers on identical text. Audio stays in data/own/ (git-ignored).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m1_common import DATA, read_jsonl, write_jsonl, write_wav_16k_mono  # noqa: E402


def record_until_enter(pa, pyaudio, dev) -> np.ndarray:
    chunks: list[bytes] = []

    def cb(in_data, frame_count, time_info, status):
        chunks.append(in_data)
        return (None, pyaudio.paContinue)

    stream = pa.open(format=pyaudio.paInt16, channels=dev.channels, rate=dev.rate, input=True,
                     input_device_index=dev.index, frames_per_buffer=int(dev.rate * 0.02), stream_callback=cb)
    input("   recording... press Enter when done ")
    time.sleep(0.3)                      # keep the tail of the last word
    stream.stop_stream()
    stream.close()
    pcm = np.frombuffer(b"".join(chunks), dtype=np.int16).reshape(-1, dev.channels)
    audio = pcm.astype(np.float32).mean(axis=1) / 32768.0
    if dev.rate != 16_000:
        import soxr
        audio = soxr.resample(audio, dev.rate, 16_000).astype(np.float32)
    return audio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", type=int, required=True, help="mic index from m0_record.py --list-devices")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()

    import pyaudiowpatch as pyaudio
    from livetrans.audio.devices import list_devices, resolve

    sentences = read_jsonl(DATA / "fleurs_hy.jsonl")[: args.n]
    out_dir = DATA / "own"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with pyaudio.PyAudio() as pa:
        dev = resolve(list_devices(pa), args.mic, "mic")
        print(f"Mic: [{dev.index}] {dev.name}\n")
        for i, s in enumerate(sentences, 1):
            while True:
                print(f"\n({i}/{len(sentences)})  {s['ref_hy']}")
                cmd = input("   Enter = start, s = skip > ").strip().lower()
                if cmd == "s":
                    break
                audio = record_until_enter(pa, pyaudio, dev)
                peak = float(np.abs(audio).max()) if audio.size else 0.0
                print(f"   {len(audio) / 16000:.1f} s, peak {20 * np.log10(peak + 1e-9):.0f} dBFS")
                if input("   keep? Enter = yes, r = redo > ").strip().lower() == "r":
                    continue
                name = f"own_{i:02d}.wav"
                write_wav_16k_mono(out_dir / name, audio)
                rows.append({"id": s["id"], "wav": f"own/{name}", "ref_hy": s["ref_hy"], "ref_en": s["ref_en"],
                             "gender": "own"})
                break
    write_jsonl(DATA / "own_reading.jsonl", rows)
    print(f"\nSaved {len(rows)} recordings. Next: python tools/m1_bench_asr.py --set own")
    return 0


if __name__ == "__main__":
    sys.exit(main())
