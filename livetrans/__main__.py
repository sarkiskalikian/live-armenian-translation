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

import numpy as np
from pathlib import Path


def main() -> int:
    for s in (sys.stdout, sys.stderr):        # Armenian text on any Windows console
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--mic", type=int)
    ap.add_argument("--loopback", type=int)
    # Candidate mitigations for the mic frame shortfall (DECISIONS.md D10). Defaults = T-LIVE-1 config;
    # change ONE per test run.
    ap.add_argument("--asr-threads", type=int, default=None, help="ONNX Runtime threads (default: all cores)")
    ap.add_argument("--asr-no-spin", action="store_true", help="stop ONNX Runtime busy-waiting between runs")
    ap.add_argument("--omp-passive", action="store_true", help="set OMP_WAIT_POLICY=PASSIVE for CTranslate2")
    ap.add_argument("--capture-buffer-ms", type=int, default=20, help="PortAudio buffer per callback")
    ap.add_argument("--mt-threads", type=int, default=4)
    ap.add_argument("--beam", type=int, default=1,
                    help="translation beam size (default 1, chosen in E2; 4 = previous default)")
    ap.add_argument("--min-silence", type=float, default=0.5, help="seconds of silence that end a phrase")
    ap.add_argument("--max-speech", type=float, default=10.0, help="force a cut after this many seconds")
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--warmup", choices=("minimal", "representative"), default="minimal",
                    help="model warm-up before listening (start-up cost, not live latency)")
    ap.add_argument("--speakers", action="store_true",
                    help="label remote phrases Speaker 1/2/... by voice (default off until verified)")
    ap.add_argument("--spk-threshold", type=float, default=0.30,
                    help="cosine similarity to join a speaker (0.30 chosen in T-SPK-1)")
    ap.add_argument("--no-models", action="store_true",
                    help="diagnostic: capture + phrase cutting only (no ASR/MT), to isolate capture problems")
    args = ap.parse_args()
    if args.omp_passive:                          # must be set before ctranslate2 is imported
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    print(f"config: asr_threads={args.asr_threads or 'default'} asr_spin={'off' if args.asr_no_spin else 'on'} "
          f"omp_passive={'on' if args.omp_passive else 'off'} capture_buffer={args.capture_buffer_ms} ms "
          f"models={'off' if args.no_models else 'on'} beam={args.beam} "
          f"speakers={'on (threshold ' + str(args.spk_threshold) + ')' if args.speakers else 'off'}")

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
        speakers = None
        startup = {}
        if args.no_models:
            print("DIAGNOSTIC MODE: --no-models (no recognition or translation)")
        else:
            print("Loading models...", flush=True)
            t = time.perf_counter()
            from livetrans.asr import Transcriber
            from livetrans.translate import Translator
            asr = Transcriber(models.ASR_DIR, threads=args.asr_threads, spin=not args.asr_no_spin)
            mt = Translator(models.MT_DIR, threads=args.mt_threads, beam_size=args.beam)
            startup["load_s"] = round(time.perf_counter() - t, 2)
            from livetrans.warmup import warm_up
            startup["warmup_s"] = round(warm_up(asr, mt, args.warmup), 2)
            startup["warmup_mode"] = args.warmup
            if args.speakers:
                from livetrans.speakers import SpeakerTracker, WeSpeakerEmbedder
                t = time.perf_counter()
                embedder = WeSpeakerEmbedder(models.SPK_DIR)
                embedder((0.003 * np.random.default_rng(0).standard_normal(24_000)).astype(np.float32))  # warm-up
                speakers = SpeakerTracker(embedder, threshold=args.spk_threshold)
                startup["speaker_model_s"] = round(time.perf_counter() - t, 2)
            print(f"Start-up (not part of live latency): models loaded in {startup['load_s']:.1f} s, "
                  f"warm-up '{args.warmup}' {startup['warmup_s']:.1f} s"
                  + (f", speaker model {startup['speaker_model_s']:.1f} s" if args.speakers else ""))

        log_dir = Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Me     = [{mic.index}] {mic.name}\nRemote = [{lb.index}] {lb.name}")
        print(f"Log: {log_dir}   Listening... (Ctrl+C to stop)\n", flush=True)

        t0 = time.perf_counter()
        t0_unix = time.time()          # wall clock of t0, to line up logs with other tools
        worker = Worker(asr, mt, t0, log_dir, save_audio=args.save_audio, speakers=speakers)
        worker.start()
        seg_kw = dict(min_silence_s=args.min_silence, max_speech_s=args.max_speech)
        pipes = {}
        def lag_warning(label):
            return lambda gap: worker.say(f"!! {label}: stream {gap.duration_s:.2f} s behind real time around "
                                          f"{gap.start_s:.0f} s and not catching up (fewer frames than expected; "
                                          f"cause unknown)")

        for name, dev in (("mic", mic), ("loopback", lb)):
            label = "Me" if name == "mic" else "Remote"
            seg = Segmenter(name, vad, worker.submit, **seg_kw)
            conv = To16kMono(dev.rate)
            pipes[name] = (seg, conv, CaptureStream(pa, dev, t0, lambda pcm, s=seg, c=conv: s.push(c(pcm)),
                                                    buffer_s=args.capture_buffer_ms / 1000,
                                                    on_unrecovered=lag_warning(label)))

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
                              "unrecovered_lag_s": round(st.unrecovered_s, 2), "unrecovered_events": len(st.unrecovered),
                              "late_recovered": len(st.late), "idle_filled_s": round(st.filled_s, 2),
                              "overflows": st.overflows, "max_lateness_s": st.max_lateness_s,
                              "start_offset_s": st.start_offset_s}
            print(f"capture {label:6}: delivered {ratio if ratio is None else f'{ratio:.1%}'} of expected audio, "
                  f"unrecovered lag {st.unrecovered_s:.2f} s in {len(st.unrecovered)} events, overflows {st.overflows}")
            if st.unrecovered:
                problems.append(f"{label}: frame shortfall - {st.unrecovered_s:.2f} s of unrecovered lag in "
                                f"{len(st.unrecovered)} events; cause unknown "
                                f"(delivered {ratio:.1%} of expected samples)")
            if st.overflows:
                problems.append(f"{label}: {st.overflows} input overflows reported by PortAudio")
        summary = summarize(worker.stats)
        summary["capture"] = capture
        summary["startup"] = startup
        summary["session_t0_unix"] = round(t0_unix, 3)
        summary["capture_problems"] = problems
        summary["settings"] = vars(args)
        (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print_summary(summary, problems)
        print(f"Log saved to {log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
