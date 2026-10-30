"""Live Armenian -> English transcription of a call (walking skeleton).

  python -m livetrans --list-devices
  python -m livetrans --mic 14 --loopback 16
  python -m livetrans --mic 14 --loopback 16 --save-audio     # also keep each segment as WAV

Sources are labelled "Me" (microphone) and "Remote" (system audio). Each line
shows the Armenian recognition AND the English translation, plus where the
latency went. Ctrl+C prints a summary. A log of the session (text + timings)
goes to logs/<timestamp>/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def main() -> int:
    # CTranslate2 uses OpenMP; stop its threads busy-waiting between calls so they
    # do not compete with the audio threads (must be set before ctranslate2 loads).
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    for s in (sys.stdout, sys.stderr):        # Armenian text on any Windows console
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--mic", type=int)
    ap.add_argument("--loopback", type=int)
    ap.add_argument("--asr-threads", type=int, default=4)
    ap.add_argument("--mt-threads", type=int, default=4)
    ap.add_argument("--beam", type=int, default=4, help="translation beam size (1 = fastest)")
    ap.add_argument("--min-silence", type=float, default=0.5, help="seconds of silence that end a phrase")
    ap.add_argument("--max-speech", type=float, default=10.0, help="force a cut after this many seconds")
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--no-models", action="store_true",
                    help="diagnostic: capture + phrase cutting only (no ASR/MT), to isolate capture problems")
    args = ap.parse_args()

    import pyaudiowpatch as pyaudio

    from livetrans import models
    from livetrans.audio.capture import CaptureStream
    from livetrans.audio.convert import To16kMono
    from livetrans.audio.devices import list_devices, print_devices, resolve
    from livetrans.pipeline import Worker, print_summary, summarize
    from livetrans.segmenter import Segmenter, SileroVad

    with pyaudio.PyAudio() as pa:
        devices = list_devices(pa)
        if args.list_devices:
            print_devices(devices)
            return 0
        if args.mic is None or args.loopback is None:
            ap.error("choose both devices: --mic N --loopback M (see --list-devices)")
        try:
            mic, lb = resolve(devices, args.mic, "mic"), resolve(devices, args.loopback, "loopback")
        except ValueError as e:
            print(f"ERROR: {e}")
            return 2

        vad = SileroVad(models.vad_model_path())
        asr = mt = None
        if args.no_models:
            print("DIAGNOSTIC MODE: --no-models (no recognition or translation)")
        else:
            print("Loading models...", flush=True)
            t = time.perf_counter()
            from livetrans.asr import Transcriber
            from livetrans.translate import Translator
            asr = Transcriber(models.ASR_DIR, threads=args.asr_threads)
            mt = Translator(models.MT_DIR, threads=args.mt_threads, beam_size=args.beam)
            # warm-up so the first real phrase is not slowed by lazy initialisation
            import numpy as np
            mt.translate(asr.transcribe(np.zeros(16_000, dtype=np.float32)) or "Բարև։")
            print(f"Models ready in {time.perf_counter() - t:.1f} s")

        log_dir = Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Me     = [{mic.index}] {mic.name}\nRemote = [{lb.index}] {lb.name}")
        print(f"Log: {log_dir}   Listening... (Ctrl+C to stop)\n", flush=True)

        t0 = time.perf_counter()
        worker = Worker(asr, mt, t0, log_dir, save_audio=args.save_audio)
        worker.start()
        seg_kw = dict(min_silence_s=args.min_silence, max_speech_s=args.max_speech)
        pipes = {}
        def loss_warning(label):
            return lambda gap: worker.say(f"!! {label}: {gap.duration_s:.2f} s of audio LOST around "
                                          f"{gap.start_s:.0f} s (device did not deliver it)")

        for name, dev in (("mic", mic), ("loopback", lb)):
            label = "Me" if name == "mic" else "Remote"
            seg = Segmenter(name, vad, worker.submit, **seg_kw)
            conv = To16kMono(dev.rate)
            pipes[name] = (seg, conv, CaptureStream(pa, dev, t0, lambda pcm, s=seg, c=conv: s.push(c(pcm)),
                                                    on_loss=loss_warning(label)))

        problems: list[str] = []
        try:
            for name, (_, _, cap) in pipes.items():
                cap.start()
            stall_reported = False
            while True:
                time.sleep(0.25)
                for name, (_, _, cap) in pipes.items():
                    label = "Me" if name == "mic" else "Remote"
                    if not cap.is_active() and f"{label}: capture stopped" not in problems:
                        problems.append(f"{label}: capture stopped (device unplugged or switched?)")
                        worker.say(f"!! {label}: capture stopped (device unplugged or switched?)")
                    if cap.error and f"{label}: {cap.error}" not in problems:
                        problems.append(f"{label}: {cap.error}")
                        worker.say(f"!! {label}: {cap.error}")
                last = pipes["mic"][2].last_callback
                if not stall_reported and last and time.perf_counter() - last > 2.0:
                    stall_reported = True
                    problems.append("Me: microphone stopped delivering audio for >2 s")
                    worker.say("!! Me: microphone stopped delivering audio")
        except KeyboardInterrupt:
            print("\nStopping...", flush=True)
        except OSError as e:
            problems.append(f"could not open a device: {e}")
            print(f"!! could not open a device: {e}", flush=True)
        finally:
            for name, (seg, conv, cap) in pipes.items():
                cap.stop()
                cap.aligner.flush()
                seg.flush()
            worker.stop()
            worker.join(timeout=30)

        t_stop = time.perf_counter()
        capture = {}
        for name, (seg, _, cap) in pipes.items():
            label = "Me" if name == "mic" else "Remote"
            st = cap.aligner.stats
            ratio = cap.aligner.delivered_ratio(t_stop)
            capture[label] = {"session_s": round(t_stop - t0, 1), "delivered_ratio": ratio,
                              "lost_s": round(st.lost_s, 2), "loss_events": len(st.lost),
                              "late_recovered": len(st.late), "idle_filled_s": round(st.filled_s, 2),
                              "overflows": st.overflows, "max_lateness_s": st.max_lateness_s,
                              "start_offset_s": st.start_offset_s}
            print(f"capture {label:6}: delivered {ratio if ratio is None else f'{ratio:.1%}'} of expected audio, "
                  f"lost {st.lost_s:.2f} s in {len(st.lost)} events, overflows {st.overflows}")
            if st.lost:
                problems.append(f"{label}: {st.lost_s:.2f} s of audio LOST in {len(st.lost)} events "
                                f"(delivered {ratio:.1%} of expected samples)")
            if st.overflows:
                problems.append(f"{label}: {st.overflows} input buffer overflows (audio lost)")
        summary = summarize(worker.stats)
        summary["capture"] = capture
        summary["capture_problems"] = problems
        summary["settings"] = vars(args)
        (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print_summary(summary, problems)
        print(f"Log saved to {log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
