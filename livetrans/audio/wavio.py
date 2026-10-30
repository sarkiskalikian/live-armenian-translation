"""Minimal WAV helpers (16-bit PCM) using only the standard library + numpy."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Returns (int16 array shaped (frames, channels), sample_rate)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        rate, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)
    return pcm, rate


def write_wav16k(path: str | Path, audio: np.ndarray) -> None:
    """Write float32 mono 16 kHz audio."""
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(pcm.tobytes())
