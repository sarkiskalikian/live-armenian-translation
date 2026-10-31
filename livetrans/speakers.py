"""Remote-speaker labels: "Speaker 1", "Speaker 2", ... for loopback phrases.

Each remote phrase gets a voice embedding (WeSpeaker ResNet34 via onnx-asr); the
tracker compares it with the speakers seen so far (cosine similarity to each
speaker's centroid). Online, incremental - the rules borrowed from Coria et al.
(diart), nothing else:
  - similarity >= threshold to the best speaker  -> that speaker;
  - otherwise a NEW speaker, but only from a phrase long enough to trust
    (min_new_s); a shorter phrase is given the best existing speaker instead;
  - a speaker's centroid is updated only from phrases >= min_update_s, so short,
    noisy phrases cannot drag a voice profile toward someone else;
  - phrases shorter than min_embed_s are not embedded at all: they inherit the
    previous remote label (backchannels like «հա» carry too little voice).
Labels never change after they are printed. Microphone phrases are always "Me"
and never reach this module. Thresholds are tuned in tools/speaker_check.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RATE = 16_000


class WeSpeakerEmbedder:
    """WeSpeaker ONNX model through onnx-asr (no extra package)."""

    def __init__(self, model_dir: str | Path, threads: int | None = 2):
        import onnxruntime as rt
        from onnx_asr.loader import Manager

        opts = rt.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.se = Manager(opts, ["CPUExecutionProvider"]).create_se(None, Path(model_dir))

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return np.asarray(self.se.embedding(audio.astype(np.float32), sample_rate=RATE), dtype=np.float32)


@dataclass
class Decision:
    label: str
    speaker: int                  # 1-based
    reason: str                   # "match" | "new" | "short-inherit" | "short-first" | "no-new-best" | "cap-best"
    similarity: float | None      # cosine to the chosen speaker (None if not embedded)
    embed_s: float                # time spent computing the embedding
    updated: bool                 # centroid updated with this phrase


@dataclass
class SpeakerTracker:
    embed: callable
    threshold: float = 0.45
    min_embed_s: float = 1.0
    min_new_s: float = 1.5
    min_update_s: float = 1.5
    max_speakers: int = 8
    sums: list[np.ndarray] = field(default_factory=list)       # unnormalised sums of embeddings
    counts: list[int] = field(default_factory=list)
    last: int | None = None

    def _centroids(self) -> np.ndarray:
        c = np.stack(self.sums)
        return c / np.linalg.norm(c, axis=1, keepdims=True)

    def _new(self, e: np.ndarray) -> int:
        self.sums.append(e.copy())
        self.counts.append(1)
        return len(self.sums) - 1

    def assign(self, audio: np.ndarray) -> Decision:
        dur = len(audio) / RATE
        if dur < self.min_embed_s:
            if self.last is None:                 # nothing known yet: provisional Speaker 1, no centroid
                self.last = 0
                return Decision("Speaker 1", 1, "short-first", None, 0.0, False)
            return Decision(f"Speaker {self.last + 1}", self.last + 1, "short-inherit", None, 0.0, False)

        t = time.perf_counter()
        e = self.embed(audio)
        embed_s = time.perf_counter() - t
        e = e / (np.linalg.norm(e) + 1e-12)

        if not self.sums:
            k = self._new(e)
            self.last = k
            return Decision(f"Speaker {k + 1}", k + 1, "new", None, embed_s, True)

        sims = self._centroids() @ e
        best = int(np.argmax(sims))
        sim = float(sims[best])
        if sim >= self.threshold:
            k, reason = best, "match"
        elif dur < self.min_new_s:
            k, reason = best, "no-new-best"       # too short to found a new speaker
        elif len(self.sums) >= self.max_speakers:
            k, reason = best, "cap-best"
        else:
            k = self._new(e)
            self.last = k
            return Decision(f"Speaker {k + 1}", k + 1, "new", sim, embed_s, True)

        updated = dur >= self.min_update_s
        if updated:
            self.sums[k] += e
            self.counts[k] += 1
        self.last = k
        return Decision(f"Speaker {k + 1}", k + 1, reason, sim, embed_s, updated)
