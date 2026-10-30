"""M0: check a recording made by m0_record.py.

  python tools/m0_check.py recordings/<timestamp>

Reports, per guided phase, whether speech landed in the right stream, plus any
dropouts, buffer overflows, clipping and length mismatches. Prints the report
and saves it as m0_report.txt next to the recording (text only, no audio).
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

FRAME_S = 0.05          # level is measured in 50 ms frames
EDGE_S = 1.5            # ignore the first/last 1.5 s of each phase (people react late)
ACTIVE_ABOVE_FLOOR = 15  # dB above the noise floor counts as "active"
ACTIVE_MIN_DB = -60.0   # ...and never below this absolute level
SPEECH_MIN = 0.30       # a talking phase needs >= 30 % active frames (speech has pauses)
QUIET_MAX = 0.10        # a quiet phase allows <= 10 % active frames
IDLE_EDGE_DB = -50.0    # loopback gap with quiet edges = nothing was playing


def load(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)
    return pcm, rate


def frame_db(pcm: np.ndarray, rate: int) -> np.ndarray:
    mono = pcm.astype(np.float32).mean(axis=1) / 32768.0
    n = int(rate * FRAME_S)
    usable = len(mono) // n * n
    ms = (mono[:usable].reshape(-1, n) ** 2).mean(axis=1)
    return 10 * np.log10(ms + 1e-12)


def phase_slice(db: np.ndarray, start: float, end: float) -> np.ndarray:
    a, b = int((start + EDGE_S) / FRAME_S), int((end - EDGE_S) / FRAME_S)
    return db[a:max(a, b)]


def check(session_dir: Path) -> tuple[str, bool]:
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    lines: list[str] = []
    ok = True

    def verdict(passed: bool, warn_only: bool = False) -> str:
        nonlocal ok
        if passed:
            return "PASS"
        if not warn_only:
            ok = False
        return "WARN" if warn_only else "FAIL"

    lines.append(f"M0 capture check - {session_dir}")
    lines.append(f"Recorded {meta['created']}, wall time {meta['wall_seconds']:.1f} s")
    if meta["errors"]:
        ok = False
        lines.append("Runtime errors: " + "; ".join(meta["errors"]))

    db, raw = {}, {}
    lines.append("\n== Streams ==")
    for name in ("mic", "loopback"):
        st = meta["streams"][name]
        pcm, rate = load(session_dir / f"{name}.wav")
        raw[name] = pcm
        db[name] = frame_db(pcm, rate)
        dur = len(pcm) / rate
        clip = float(np.mean(np.abs(pcm.astype(np.int32)) >= 32767)) * 100
        dev = st["device"]
        lines.append(f"{name.upper():9} [{dev['index']}] {dev['name']} ({dev['channels']} ch, {dev['rate']} Hz)")
        lines.append(f"  length {dur:.2f} s vs wall {meta['wall_seconds']:.2f} s  "
                     f"[{verdict(abs(dur - meta['wall_seconds']) < 0.1)}]")
        lines.append(f"  first buffer after {st['first_buffer_s']:.3f} s; "
                     f"{st['callbacks']} buffers; max queue {st['max_queue']}")
        lines.append(f"  input overflows {st['overflows']}  [{verdict(st['overflows'] == 0)}]")
        lines.append(f"  clipped samples {clip:.3f} %  [{verdict(clip < 0.1, warn_only=True)}]")
        gaps = st["gaps"]
        if name == "mic":
            lines.append(f"  dropouts {len(gaps)} totalling {st['filled_s']:.2f} s  [{verdict(not gaps)}]")
            for g in gaps:
                lines.append(f"    gap at {g['start_s']:.2f} s for {g['duration_s']:.2f} s")
        else:
            idle = [g for g in gaps if g["db_before"] < IDLE_EDGE_DB and g["db_after"] < IDLE_EDGE_DB]
            odd = [g for g in gaps if g not in idle]
            lines.append(f"  gaps {len(gaps)}: {len(idle)} idle (nothing playing - expected for WASAPI loopback), "
                         f"{len(odd)} during sound  [{verdict(not odd)}]")
            for g in odd:
                lines.append(f"    gap at {g['start_s']:.2f} s for {g['duration_s']:.2f} s "
                             f"(level before {g['db_before']} dB, after {g['db_after']} dB)")

    if not meta["phases"]:
        lines.append("\n(free recording: no phase checks)")
        return "\n".join(lines), ok

    quiet_ids = {"ready", "silence"}
    floor = {}
    for name in db:
        quiet = np.concatenate([phase_slice(db[name], p["start"], p["end"])
                                for p in meta["phases"] if p["id"] in quiet_ids])
        floor[name] = float(np.percentile(quiet, 50)) if quiet.size else -90.0
    thr = {n: max(floor[n] + ACTIVE_ABOVE_FLOOR, ACTIVE_MIN_DB) for n in db}

    expect = {  # phase -> (mic should talk?, loopback should talk?)
        "ready": (False, False), "me": (True, False), "remote": (False, True),
        "both": (True, True), "silence": (False, False),
    }
    lines.append("\n== Guided phases ==")
    lines.append(f"noise floor: mic {floor['mic']:.1f} dB, loopback {floor['loopback']:.1f} dB "
                 f"(active = above {thr['mic']:.1f} / {thr['loopback']:.1f} dB)")
    lines.append(f"{'phase':8} {'mic active':>10} {'mic med':>8} {'sys active':>11} {'sys med':>8}  result")
    notes = []
    for p in meta["phases"]:
        want_mic, want_sys = expect[p["id"]]
        cells, results = [], []
        for name, want in (("mic", want_mic), ("loopback", want_sys)):
            seg = phase_slice(db[name], p["start"], p["end"])
            frac = float(np.mean(seg > thr[name])) if seg.size else 0.0
            med = float(np.median(seg)) if seg.size else -120.0
            cells += [f"{frac * 100:9.0f}%", f"{med:8.1f}"]
            results.append(frac >= SPEECH_MIN if want else frac <= QUIET_MAX)
            if not (frac >= SPEECH_MIN if want else frac <= QUIET_MAX):
                label = "mic" if name == "mic" else "system audio"
                if want:
                    notes.append(f"'{p['id']}': expected speech on {label}, found {frac:.0%} active frames.")
                else:
                    hint = (" Remote voice may be leaking into the mic (speakers or loud headphones)."
                            if name == "mic" and p["id"] == "remote" else "")
                    notes.append(f"'{p['id']}': expected quiet {label}, found {frac:.0%} active frames.{hint}")
        lines.append(f"{p['id']:8} {cells[0]:>10} {cells[1]} {cells[2]:>11} {cells[3]}  "
                     f"[{verdict(all(results))}]")
    lines += [f"  - {n}" for n in notes]

    lines.append(f"\nOVERALL: {'PASS' if ok else 'NEEDS ATTENTION'}")
    return "\n".join(lines), ok


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = Path(sys.argv[1])
    report, ok = check(d)
    print(report)
    (d / "m0_report.txt").write_text(report + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
