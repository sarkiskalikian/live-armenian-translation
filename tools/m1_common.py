"""Shared helpers for the M1 benchmarks (evaluation only, not used at runtime)."""
from __future__ import annotations

import json
import re
import unicodedata
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def normalize_hy(text: str) -> str:
    """Normalisation for WER/CER: case, punctuation and the եւ/և spelling variant
    are not counted as errors (the model emits punctuation; FLEURS raw text has it)."""
    t = unicodedata.normalize("NFC", text).lower()
    t = t.replace("եւ", "և")
    # Armenian emphasis/exclamation/question marks (՛ ՜ ՞) and abbreviation mark (՟)
    # sit inside a word, e.g. «ինչպե՞ս»: delete them so the word is not split.
    t = re.sub("[՛-՟’']", "", t)
    t = "".join(" " if unicodedata.category(c).startswith("P") else c for c in t)
    return re.sub(r"\s+", " ", t).strip()


def read_wav_16k_mono(path: Path) -> np.ndarray:
    """Read a 16-bit PCM WAV, downmix to mono, resample to 16 kHz if needed."""
    with wave.open(str(path), "rb") as w:
        rate, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)
    audio = pcm.astype(np.float32).mean(axis=1) / 32768.0
    if rate != 16_000:
        import soxr
        audio = soxr.resample(audio, rate, 16_000).astype(np.float32)
    return audio


def write_wav_16k_mono(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(pcm.tobytes())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def peak_ram_mb() -> float:
    """Peak working set on Windows; current RSS elsewhere."""
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1)


def machine_info() -> dict:
    import os
    import platform
    import psutil
    return {"cpu": platform.processor(), "logical_cores": os.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "os": platform.platform()}
