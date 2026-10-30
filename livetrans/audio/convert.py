"""Audio format bridge: device format (int16, N channels, 44.1/48 kHz) -> model format
(float32 mono 16 kHz). Streaming, so one converter instance per source keeps
resampler state across chunks and the total sample count stays exact."""
from __future__ import annotations

import numpy as np

TARGET_RATE = 16_000


class To16kMono:
    def __init__(self, in_rate: int):
        self.in_rate = in_rate
        self._rs = None
        if in_rate != TARGET_RATE:
            import soxr
            self._rs = soxr.ResampleStream(in_rate, TARGET_RATE, 1, dtype="float32", quality="HQ")

    def __call__(self, pcm: np.ndarray, last: bool = False) -> np.ndarray:
        """pcm: int16 array shaped (frames, channels). Returns float32 1-D at 16 kHz."""
        mono = pcm.astype(np.float32).mean(axis=1) / 32768.0
        if self._rs is None:
            return mono
        return self._rs.resample_chunk(mono, last=last)
