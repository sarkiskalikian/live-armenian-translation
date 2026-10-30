"""M0: record the microphone and the system audio (WASAPI loopback) at the same time.

  python tools/m0_record.py --list-devices
  python tools/m0_record.py --mic <N> --loopback <M> --guided      # 60 s scripted test
  python tools/m0_record.py --mic <N> --loopback <M> --seconds 30  # free recording

Both devices must be chosen explicitly; nothing is picked silently.
Writes recordings/<timestamp>/{mic.wav, loopback.wav, session.json}.
Audio recordings stay local: recordings/ is git-ignored.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyaudiowpatch as pyaudio  # noqa: E402

from livetrans.audio.capture import CaptureStream  # noqa: E402
from livetrans.audio.devices import list_devices, print_devices, resolve  # noqa: E402

# Guided test script: (start_s, end_s, phase_id, instruction)
PHASES = [
    (0, 5, "ready", "SILENCE - nobody speaks (measuring background noise)"),
    (5, 20, "me", "ME only - you talk, remote person stays silent"),
    (20, 35, "remote", "REMOTE only - signal them to start; you stay silent"),
    (35, 50, "both", "BOTH at once - you start talking too, they keep talking"),
    (50, 60, "silence", "SILENCE - signal them to stop; nobody speaks"),
]
MIC_STALL_S = 2.0   # the mic always delivers buffers; silence from it means trouble


def meter(db: float, width: int = 18) -> str:
    filled = int(max(0.0, min(1.0, (db + 70) / 70)) * width)
    return "#" * filled + "." * (width - filled)


def open_wav(path: Path, dev) -> wave.Wave_write:
    w = wave.open(str(path), "wb")
    w.setnchannels(dev.channels)
    w.setsampwidth(2)
    w.setframerate(dev.rate)
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--mic", type=int, help="index of the microphone device")
    ap.add_argument("--loopback", type=int, help="index of the system-audio loopback device")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--guided", action="store_true", help="run the 60 s scripted test")
    ap.add_argument("--out", type=Path, default=Path("recordings"))
    args = ap.parse_args()

    with pyaudio.PyAudio() as pa:
        devices = list_devices(pa)
        if args.list_devices:
            print_devices(devices)
            return 0
        if args.mic is None or args.loopback is None:
            ap.error("choose both devices: --mic N --loopback M (see --list-devices)")
        try:
            mic = resolve(devices, args.mic, "mic")
            lb = resolve(devices, args.loopback, "loopback")
        except ValueError as e:
            print(f"ERROR: {e}")
            return 2

        seconds = PHASES[-1][1] if args.guided else args.seconds
        out = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        print(f"MIC      [{mic.index}] {mic.name}  ({mic.channels} ch, {mic.rate} Hz)")
        print(f"LOOPBACK [{lb.index}] {lb.name}  ({lb.channels} ch, {lb.rate} Hz)")
        print(f"Recording {seconds:.0f} s -> {out}   (Ctrl+C stops early and still saves)\n")

        wav_mic, wav_lb = open_wav(out / "mic.wav", mic), open_wav(out / "loopback.wav", lb)
        t0 = time.perf_counter()
        streams = {
            "mic": CaptureStream(pa, mic, t0, lambda a: wav_mic.writeframes(a.tobytes())),
            "loopback": CaptureStream(pa, lb, t0, lambda a: wav_lb.writeframes(a.tobytes())),
        }
        errors: list[str] = []
        started = []
        try:
            for name, s in streams.items():
                try:
                    s.start()
                    started.append(name)
                except OSError as e:
                    errors.append(f"{name}: could not open device: {e}")
            if len(started) < 2:
                raise RuntimeError("; ".join(errors))

            phase_idx = -1
            stall_reported = False
            while (now := time.perf_counter() - t0) < seconds:
                if args.guided:
                    for i, (a, b, _, text) in enumerate(PHASES):
                        if a <= now < b and i != phase_idx:
                            phase_idx = i
                            print(f"\n\a>>> [{a:02d}-{b:02d} s] {text}")
                for name, s in streams.items():
                    if not s.is_active() and f"{name} inactive" not in errors:
                        errors.append(f"{name} inactive")
                        print(f"\n!!! {name} stream stopped (device unplugged or switched?)")
                    if s.error and s.error not in errors:
                        errors.append(s.error)
                last = streams["mic"].last_callback
                if not stall_reported and last and time.perf_counter() - last > MIC_STALL_S:
                    stall_reported = True
                    errors.append("mic stalled: no buffers for >2 s")
                    print("\n!!! microphone stopped delivering audio")
                m = streams["mic"].aligner.level_db()
                s_db = streams["loopback"].aligner.level_db()
                print(f"\r  {now:5.1f}s  MIC {meter(m)} {m:6.1f} dB | SYS {meter(s_db)} {s_db:6.1f} dB ",
                      end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        except RuntimeError as e:
            print(f"\nERROR: {e}")
        finally:
            t_end = time.perf_counter()
            for s in streams.values():
                s.stop()
            for s in streams.values():
                s.aligner.pad_to(t_end)
            wav_mic.close()
            wav_lb.close()

        session = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": round(t_end - t0, 3),
            "guided": args.guided,
            "phases": [dict(start=a, end=b, id=p, text=t) for a, b, p, t in PHASES] if args.guided else [],
            "errors": errors,
            "streams": {
                name: {"device": asdict(s.device), **asdict(s.aligner.stats),
                       "written_seconds": round(s.aligner.written / s.device.rate, 3)}
                for name, s in streams.items()
            },
        }
        (out / "session.json").write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved to {out}\nNext: python tools/m0_check.py {out}")
        return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
