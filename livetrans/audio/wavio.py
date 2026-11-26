"""Minimal WAV helpers (16-bit PCM) using only the standard library + numpy."""
from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np


def read_wav_float(path: str | Path) -> tuple[np.ndarray, int]:
    """Returns (float32 array in [-1, 1] shaped (frames, channels), sample_rate).
    Accepts 16-bit PCM and 32-bit float WAV. Float input is passed through
    unchanged (FLEURS is float32 and often very quiet; rounding it to 16 bit
    would cost resolution before recognition)."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    fmt = data = None
    i = 12
    while i + 8 <= len(raw):
        cid, size = raw[i:i + 4], struct.unpack("<I", raw[i + 4:i + 8])[0]
        body = raw[i + 8:i + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
        i += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == 0xFFFE and len(fmt) >= 26:              # WAVE_FORMAT_EXTENSIBLE: real tag in SubFormat
        tag = struct.unpack("<H", fmt[24:26])[0]
    if tag == 1 and bits == 16:
        x = np.frombuffer(data, dtype="<i2").reshape(-1, ch).astype(np.float32) / 32768.0
    elif tag == 3 and bits == 32:
        x = np.frombuffer(data, dtype="<f4").reshape(-1, ch).astype(np.float32)
    else:
        raise ValueError(f"{path}: unsupported WAV format (tag {tag}, {bits} bit)")
    return x, rate


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Returns (int16 array shaped (frames, channels), sample_rate), as the live code expects.
    16-bit input is returned bit-exact; float input is scaled and clipped to int16."""
    x, rate = read_wav_float(path)
    pcm = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)
    return pcm, rate


def to_16k_mono_float(x: np.ndarray, rate: int) -> np.ndarray:
    """Float (frames, channels) -> float32 mono 16 kHz, without any int16 rounding."""
    mono = x.mean(axis=1).astype(np.float32)
    if rate == 16_000:
        return mono
    import soxr
    return soxr.resample(mono, rate, 16_000).astype(np.float32)


def write_wav16k(path: str | Path, audio: np.ndarray) -> None:
    """Write float32 mono 16 kHz audio."""
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(pcm.tobytes())
