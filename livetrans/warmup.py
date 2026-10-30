"""Model warm-up before listening starts (start-up cost, never counted as live latency).

minimal         - today's behaviour: ASR on 1 s of silence, MT on its output or "Բարև։".
representative  - ASR on low-level noise at three typical phrase lengths, and MT on two
                  real Armenian sentences (short + long) at the live beam size, so the
                  first live phrase does not pay one-time initialisation costs.
Which one is the default is decided by measurement (tools/warmup_check.py, TESTS.md).
"""
from __future__ import annotations

import time

import numpy as np

MODES = ("minimal", "representative")
NOISE_LENGTHS_S = (1.5, 4.0, 8.0)          # VAD phrases in T-LIVE-4 were 0.5-5.1 s; max is 10 s
MT_SENTENCES = (
    "Բարև ձեզ, ո՞նց եք։",
    "Ես այս պահին Երևանում եմ ապրում և աշխատում եմ կլինիկական հետազոտությունների ոլորտում։",
)


def warm_up(asr, mt, mode: str = "minimal") -> float:
    """Run the warm-up; returns its duration in seconds."""
    t = time.perf_counter()
    if mode == "minimal":
        mt.translate(asr.transcribe(np.zeros(16_000, dtype=np.float32)) or "Բարև։")
    elif mode == "representative":
        rng = np.random.default_rng(0)
        for seconds in NOISE_LENGTHS_S:
            asr.transcribe((0.003 * rng.standard_normal(int(16_000 * seconds))).astype(np.float32))
        for sentence in MT_SENTENCES:
            mt.translate(sentence)
    else:
        raise ValueError(f"unknown warm-up mode {mode!r}; choose from {MODES}")
    return time.perf_counter() - t
