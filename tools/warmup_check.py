"""Measure the first-phrase cold-start cost with each warm-up mode, in fresh processes.

  python tools/warmup_check.py                       # 3 runs per mode, default clips
  python tools/warmup_check.py --runs 5 --beam 1

Each run is a new Python process (a real cold start): load models -> warm-up -> run
ASR+MT on clip1, clip2, clip3, then clip1 again. "clip1 first" vs "clip1 again" is the
cold-start cost on identical input. Start-up (load + warm-up) is reported separately.
Writes results/warmup_check.json. Uses the live program's own model and warm-up code.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CLIPS = ["logs/20261030-221541/00019.81_Remote.wav",
                 "logs/20261030-221541/00037.95_Remote.wav",
                 "logs/20261030-221541/00070.08_Remote.wav"]


def utf8_console() -> None:
    """Windows consoles/pipes default to cp1252, which cannot encode Armenian."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def child(mode: str, clips: list[str], beam: int) -> dict:
    from livetrans import models
    from livetrans.audio.convert import To16kMono
    from livetrans.audio.wavio import read_wav
    from livetrans.asr import Transcriber
    from livetrans.translate import Translator
    from livetrans.warmup import warm_up

    audio = []
    for c in clips:
        pcm, rate = read_wav(ROOT / c)
        audio.append(To16kMono(rate)(pcm, last=True))
    t = time.perf_counter()
    asr = Transcriber(models.ASR_DIR)
    mt = Translator(models.MT_DIR, beam_size=beam)
    load_s = time.perf_counter() - t
    warm_s = warm_up(asr, mt, mode)
    steps = []
    for name, a in [("clip1 first", audio[0]), ("clip2", audio[1]), ("clip3", audio[2]), ("clip1 again", audio[0])]:
        t = time.perf_counter()
        hy = asr.transcribe(a)
        t_asr = time.perf_counter() - t
        t = time.perf_counter()
        en = mt.translate(hy) if hy else ""
        t_mt = time.perf_counter() - t
        steps.append({"step": name, "asr_s": round(t_asr, 3), "mt_s": round(t_mt, 3), "hy": hy, "en": en})
    return {"mode": mode, "load_s": round(load_s, 2), "warmup_s": round(warm_s, 2), "steps": steps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--clips", nargs=3, default=DEFAULT_CLIPS)
    ap.add_argument("--child", choices=("minimal", "representative"))
    args = ap.parse_args()
    utf8_console()
    if args.child:
        print(json.dumps(child(args.child, args.clips, args.beam), ensure_ascii=False))
        return 0

    runs = []
    for i in range(args.runs):
        for mode in ("minimal", "representative"):          # alternate so drift affects both
            out = subprocess.run([sys.executable, __file__, "--child", mode, "--beam", str(args.beam),
                                  "--clips", *args.clips], capture_output=True, text=True, encoding="utf-8",
                                 env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            if out.returncode != 0:
                print(out.stderr[-2000:])
                return 1
            r = json.loads(out.stdout.strip().splitlines()[-1])
            runs.append(r)
            s = {x["step"]: x for x in r["steps"]}
            print(f"run {i + 1} {mode:15} load {r['load_s']:.1f}s warm-up {r['warmup_s']:.1f}s | "
                  + " | ".join(f"{k}: asr {v['asr_s']:.2f} mt {v['mt_s']:.2f}" for k, v in s.items()), flush=True)

    print("\nmedian over runs (seconds)")
    print(f"{'mode':15} {'load':>5} {'warm':>5} | {'clip1 first':>17} | {'clip1 again':>17} | {'clip2':>17}")
    summary = {}
    for mode in ("minimal", "representative"):
        rs = [r for r in runs if r["mode"] == mode]
        med = lambda f: round(statistics.median(f(r) for r in rs), 3)
        step = lambda name, k: med(lambda r: next(x for x in r["steps"] if x["step"] == name)[k])
        summary[mode] = {"load_s": med(lambda r: r["load_s"]), "warmup_s": med(lambda r: r["warmup_s"]),
                         **{f"{n}_{k}": step(n, k) for n in ("clip1 first", "clip1 again", "clip2", "clip3")
                            for k in ("asr_s", "mt_s")}}
        m = summary[mode]
        print(f"{mode:15} {m['load_s']:5.1f} {m['warmup_s']:5.1f} | "
              f"asr {m['clip1 first_asr_s']:.2f} mt {m['clip1 first_mt_s']:.2f} | "
              f"asr {m['clip1 again_asr_s']:.2f} mt {m['clip1 again_mt_s']:.2f} | "
              f"asr {m['clip2_asr_s']:.2f} mt {m['clip2_mt_s']:.2f}")
    out = {"beam": args.beam, "clips": args.clips, "runs": runs, "median": summary}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "warmup_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved results/warmup_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
