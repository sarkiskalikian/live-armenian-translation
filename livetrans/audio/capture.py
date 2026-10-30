"""Capture one audio device and keep its samples aligned to a shared wall clock.

Two pieces:
- CaptureStream: opens a PyAudio callback stream. The callback only time-stamps
  the buffer and puts it on a queue (never blocks, never does work).
- ChunkAligner: consumes those buffers in another thread and turns them into a
  continuous signal that starts at the session's t0, so sample n of the mic and
  sample n of the loopback refer to the same moment.

Timing rules (learned from M0 and from load tests):
- Buffers can only arrive *late*, never early. A stream's start time is therefore
  estimated from the earliest-arriving buffer of a short warm-up window, not
  from the first buffer alone (a late first buffer followed by a catch-up burst
  would otherwise shift the whole stream late for the rest of the session).
- Late delivery is not lost audio. Under CPU load (ASR running) callbacks can be
  late by hundreds of ms and then arrive in a burst; inserting zeros there would
  corrupt speech. So zeros are only inserted when `fill_gaps` is on (loopback)
  AND the stream was quiet before the hole: WASAPI loopback simply stops
  delivering while nothing is playing. Other late deliveries are only logged.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

GAP_TOLERANCE_S = 0.3     # lateness below this is normal jitter
LOSS_CONFIRM_S = 1.0      # lateness that does not recover within this long = audio was lost
WARMUP_S = 0.5            # audio collected before the start time is fixed
IDLE_DB = -50.0           # "quiet" level for deciding that a loopback hole is idle time


@dataclass
class Gap:
    start_s: float        # seconds since session t0
    duration_s: float
    db_before: float      # level just before / after the gap; lets us tell an
    db_after: float       # idle loopback (silence both sides) from a glitch


@dataclass
class StreamStats:
    callbacks: int = 0
    frames: int = 0
    overflows: int = 0
    underflows: int = 0
    first_buffer_s: float | None = None   # when the first buffer arrived, since t0
    start_offset_s: float | None = None   # estimated time of the stream's first sample, since t0
    gaps: list[Gap] = field(default_factory=list)       # holes filled with zeros (idle loopback)
    filled_s: float = 0.0                  # total zeros inserted inside the recording
    late: list[Gap] = field(default_factory=list)       # late deliveries that recovered (audio intact)
    lost: list[Gap] = field(default_factory=list)       # samples the device never delivered
    lost_s: float = 0.0
    max_lateness_s: float = 0.0
    max_queue: int = 0


def to_db(mean_square: float) -> float:
    return float(10.0 * np.log10(mean_square + 1e-12))


class ChunkAligner:
    """Turns time-stamped int16 buffers into an aligned stream. Pure Python + numpy,
    so it can be tested without audio hardware."""

    def __init__(self, rate: int, channels: int, t0: float, sink, fill_gaps: bool = False, on_loss=None):
        self.rate = rate
        self.channels = channels
        self.t0 = t0
        self.sink = sink              # callable(np.ndarray int16, shape (n, channels))
        self.fill_gaps = fill_gaps
        self.on_loss = on_loss        # callable(Gap) when lost audio is confirmed (live warning)
        self._late_since: float | None = None
        self.written = 0              # frames handed to sink, including filled zeros
        self.stats = StreamStats()
        self._warm: list[tuple[float, np.ndarray, float]] | None = []
        self._recent: deque[tuple[float, float, int]] = deque()   # (t, mean_sq, n)
        self._last_db = -120.0
        self._lock = threading.Lock()

    # -- level meter (read from the main thread) --------------------------------
    def level_db(self, window_s: float = 0.3) -> float:
        now = time.perf_counter()
        with self._lock:
            while self._recent and now - self._recent[0][0] > window_s:
                self._recent.popleft()
            if not self._recent:
                return -120.0
            num = sum(ms * n for _, ms, n in self._recent)
            den = sum(n for _, _, n in self._recent)
        return to_db(num / den)

    def _fill(self, seconds: float) -> None:
        n = int(round(seconds * self.rate))
        if n > 0:
            self.sink(np.zeros((n, self.channels), dtype=np.int16))
            self.written += n

    def _emit(self, pcm: np.ndarray, ms: float) -> None:
        self.sink(pcm)
        self.written += pcm.shape[0]
        self._last_db = to_db(ms)

    def push(self, t_arrival: float, data: bytes, status: int = 0,
             overflow_flag: int = 2, underflow_flag: int = 1) -> None:
        pcm = np.frombuffer(data, dtype=np.int16).reshape(-1, self.channels)
        n = pcm.shape[0]
        st = self.stats
        st.callbacks += 1
        st.frames += n
        if status & overflow_flag:
            st.overflows += 1
        if status & underflow_flag:
            st.underflows += 1
        ms = float(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)) if n else 0.0
        with self._lock:
            self._recent.append((t_arrival, ms, n))
        if st.first_buffer_s is None:
            st.first_buffer_s = t_arrival - self.t0

        if self._warm is not None:                         # still collecting the warm-up window
            self._warm.append((t_arrival, pcm, ms))
            if sum(p.shape[0] for _, p, _ in self._warm) >= WARMUP_S * self.rate:
                self._finish_warmup()
            return
        self._place(t_arrival, pcm, ms)

    def _finish_warmup(self) -> None:
        cum, starts = 0, []
        for t_arr, p, _ in self._warm:
            cum += p.shape[0]
            starts.append(t_arr - cum / self.rate)          # implied time of the stream's first sample
        start = min(starts)
        self.stats.start_offset_s = round(start - self.t0, 4)
        self._fill(max(start - self.t0, 0.0))              # lead-in: device started after t0
        for _, p, ms in self._warm:
            self._emit(p, ms)
        self._warm = None

    def _place(self, t_arrival: float, pcm: np.ndarray, ms: float) -> None:
        """Lateness = how far behind the wall clock this stream's samples are.
        Three cases when it exceeds the jitter tolerance:
          - loopback after quiet audio: WASAPI idle time -> fill with zeros now;
          - it recovers within LOSS_CONFIRM_S: buffers were only late (CPU load) -> keep;
          - it does not recover: the device dropped samples -> report as LOST and
            re-anchor the timeline with zeros so timestamps and latency stay true."""
        st = self.stats
        n = pcm.shape[0]
        lateness = t_arrival - (self.t0 + (self.written + n) / self.rate)
        st.max_lateness_s = max(st.max_lateness_s, round(lateness, 3))
        if lateness <= GAP_TOLERANCE_S:
            self._late_since = None
        else:
            gap = Gap(round(self.written / self.rate, 3), round(lateness, 3),
                      round(self._last_db, 1), round(to_db(ms), 1))
            if self.fill_gaps and self._last_db < IDLE_DB:
                st.gaps.append(gap)
                st.filled_s += lateness
                self._fill(lateness)
                self._late_since = None
            elif self._late_since is None:
                self._late_since = t_arrival
                st.late.append(gap)
            elif t_arrival - self._late_since >= LOSS_CONFIRM_S:
                st.lost.append(gap)
                st.lost_s += lateness
                self._fill(lateness)
                self._late_since = None
                if self.on_loss:
                    self.on_loss(gap)
        self._emit(pcm, ms)

    def delivered_ratio(self, t_now: float) -> float | None:
        """Samples actually delivered / samples expected since the stream started."""
        if self.stats.start_offset_s is None:
            return None
        expected = (t_now - self.t0 - self.stats.start_offset_s) * self.rate
        return round(self.stats.frames / expected, 4) if expected > 0 else None

    def flush(self) -> None:
        """Release warm-up audio if the stream ended before warm-up completed."""
        if self._warm:
            self._finish_warmup()

    def pad_to(self, t_end: float) -> None:
        """Pad with zeros up to t_end so all sources end at the same time."""
        self.flush()
        self._fill(max(t_end - (self.t0 + self.written / self.rate), 0.0))


class CaptureStream:
    """One PyAudio input stream feeding a ChunkAligner via a worker thread."""

    def __init__(self, pa, device, t0: float, sink, buffer_s: float = 0.02, on_loss=None):
        import pyaudiowpatch as pyaudio   # imported here so ChunkAligner works without it
        self._pyaudio = pyaudio
        self.pa = pa
        self.device = device
        self.aligner = ChunkAligner(device.rate, device.channels, t0, sink,
                                    fill_gaps=device.kind == "loopback", on_loss=on_loss)
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._drain, daemon=True,
                                         name=f"drain-{device.kind}")
        self.last_callback = None
        # every callback as (arrival perf_counter, frames, status flags): cheap list append,
        # used to tell dropped packets apart from a slow device clock (TESTS.md T-REC-2)
        self.callback_log: list[tuple[float, int, int]] = []
        self.stream = None
        self.error: str | None = None
        self._fpb = max(int(device.rate * buffer_s), 64)

    def _callback(self, in_data, frame_count, time_info, status):
        t = time.perf_counter()
        self.last_callback = t
        self.callback_log.append((t, frame_count, status))
        self._q.put((t, in_data, status))
        return (None, self._pyaudio.paContinue)

    def _drain(self) -> None:
        pa = self._pyaudio
        while not (self._stop.is_set() and self._q.empty()):
            try:
                t, data, status = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.aligner.stats.max_queue = max(self.aligner.stats.max_queue, self._q.qsize() + 1)
            try:
                self.aligner.push(t, data, status, pa.paInputOverflow, pa.paInputUnderflow)
            except Exception as e:  # never let a bad buffer kill the thread silently
                self.error = f"{type(e).__name__}: {e}"

    def start(self) -> None:
        self.stream = self.pa.open(
            format=self._pyaudio.paInt16,
            channels=self.device.channels,
            rate=self.device.rate,
            input=True,
            input_device_index=self.device.index,
            frames_per_buffer=self._fpb,
            stream_callback=self._callback,
        )
        self._worker.start()

    def save_callback_log(self, path, t0: float) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("t_s,frames,status\n")
            for t, n, st in self.callback_log:
                f.write(f"{t - t0:.6f},{n},{st}\n")

    def is_active(self) -> bool:
        try:
            return bool(self.stream and self.stream.is_active())
        except OSError:
            return False

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except OSError as e:          # e.g. device unplugged mid-session
                self.error = self.error or f"close failed: {e}"
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=5)
