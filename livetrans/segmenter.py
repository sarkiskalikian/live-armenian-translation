"""Cut a continuous 16 kHz stream into spoken phrases with Silero VAD (ONNX).

Silero is run frame by frame (512 samples = 32 ms) with its recurrent state
carried between frames, so it works on a live stream. Times are measured on
the sample clock: sample n of a source is n / 16000 s after the session t0,
because capture.ChunkAligner keeps every source aligned to t0.

A phrase ends after `min_silence_s` of non-speech; `end_s` is the end of the
last speech frame (the moment the speaker stopped), which is what the 5 s
latency budget is measured from. Phrases longer than `max_speech_s` are cut
so one long monologue cannot block the pipeline.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RATE = 16_000
FRAME = 512          # Silero frame at 16 kHz
CONTEXT = 64         # Silero expects the last 64 samples of the previous frame prepended


@dataclass
class Segment:
    source: str          # "mic" | "loopback"
    start_s: float       # since session t0 (sample clock)
    end_s: float         # end of the last speech frame
    emitted_s: float     # sample-clock time when the segment was closed (end_s + silence wait)
    audio: np.ndarray    # float32 16 kHz mono, with a little padding
    forced: bool = False  # True if cut because it hit max_speech_s


class SileroVad:
    """Stateless wrapper around the ONNX session; each stream keeps its own state."""

    def __init__(self, model_path: str | Path):
        import onnxruntime as rt
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = rt.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(RATE, dtype=np.int64)

    def prob(self, frame_with_context: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        out, new_state = self.session.run(["output", "stateN"],
                                          {"input": frame_with_context[None, :], "state": state, "sr": self._sr})
        return float(out[0, 0]), new_state


class Segmenter:
    def __init__(self, source: str, vad: SileroVad, on_segment, *, start_th: float = 0.5,
                 end_th: float = 0.35, min_silence_s: float = 0.5, min_speech_s: float = 0.25,
                 max_speech_s: float = 10.0, pad_s: float = 0.2):
        self.source, self.vad, self.on_segment = source, vad, on_segment
        self.start_th, self.end_th = start_th, end_th
        self.min_silence = int(min_silence_s * RATE)
        self.min_speech = int(min_speech_s * RATE)
        self.max_speech = int(max_speech_s * RATE)
        self.pad = int(pad_s * RATE)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._n = 0                                       # samples consumed by the VAD so far
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, self.pad // FRAME + 1))
        self._active = False
        self._frames: list[np.ndarray] = []
        self._seg_start = 0
        self._last_speech_end = 0
        self._silence = 0
        self.frames_seen = 0
        self.dropped_short = 0

    def push(self, audio: np.ndarray) -> None:
        self._pending = np.concatenate([self._pending, audio.astype(np.float32, copy=False)])
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            p, self._state = self.vad.prob(np.concatenate([self._context, frame]), self._state)
            self._context = frame[-CONTEXT:]
            self._step(frame, p)

    def _step(self, frame: np.ndarray, p: float) -> None:
        start, end = self._n, self._n + FRAME
        self._n = end
        self.frames_seen += 1
        if not self._active:
            if p >= self.start_th:
                self._active = True
                self._frames = list(self._preroll) + [frame]
                self._seg_start = start - FRAME * len(self._preroll)
                self._last_speech_end = end
                self._silence = 0
                self._preroll.clear()
            else:
                self._preroll.append(frame)
            return

        self._frames.append(frame)
        if p >= self.end_th:
            self._last_speech_end = end
            self._silence = 0
        else:
            self._silence += FRAME
        if self._silence >= self.min_silence:
            self._close(forced=False)
        elif end - self._seg_start >= self.max_speech:
            self._close(forced=True)

    def _close(self, forced: bool) -> None:
        audio = np.concatenate(self._frames)
        if forced:
            end = self._n
        else:
            end = self._last_speech_end
            audio = audio[: min(len(audio), end - self._seg_start + self.pad)]
        speech_len = end - self._seg_start
        seg = Segment(self.source, max(self._seg_start, 0) / RATE, end / RATE, self._n / RATE, audio, forced)
        if forced:
            # keep going: the speaker has not stopped, the next segment starts here
            self._frames, self._seg_start, self._silence = [], self._n, 0
        else:
            self._active, self._frames = False, []
        if speech_len >= self.min_speech:
            self.on_segment(seg)
        else:
            self.dropped_short += 1

    def flush(self) -> None:
        """Close an open phrase (end of session or end of file)."""
        if self._active and self._frames:
            self._close(forced=False)
