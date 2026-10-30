# Decisions

Each entry: what was decided, why, and its status. "Unverified" means it has
not yet been confirmed on the target laptop.

**D1 - System audio via WASAPI loopback (PyAudioWPatch).** Windows exposes a
loopback twin of every output device, so no virtual cable (BlackHole
equivalent) is needed and the user still hears the call. Verified in T-M0-1.

**D2 - Microphone = laptop array `[14]`, not the Buds' mic.** Opening the
Buds' "Headset" mic switches them to the Bluetooth hands-free profile; call
audio then moves to a different endpoint and loopback `[16]` would go
silent. In-ear Buds also leak little sound into the laptop mic (T-M0-1).

**D3 - ASR: NVIDIA FastConformer-Hybrid Large (hy), ONNX CTC export via
onnx-asr.** Best open model on ArmBench-ASR v1.0 (combined WER 19.9 vs
Whisper-large-v3 65.1), 115M params, runs without NeMo/PyTorch. Known risks:
~61 % WER on conversational movie audio in that benchmark; trained on read
Eastern Armenian; CTC head (published numbers are RNN-T). Not benchmarked
locally (D9).

**D4 - MT: NLLB-200 distilled 600M, int8 CTranslate2 (third-party
conversion), Meta's own SentencePiece model.** Fits in ~1 GB RAM; T-MT-1:
chrF 61-62 on clean FLEURS input. Licence CC-BY-NC (fine for this
challenge). Beam 4 is the current default; beam 1 is nearly as good and
faster (T-MT-1) - not changed yet.

**D5 - Phrase segmentation: Silero VAD run frame by frame (32 ms) with its
state carried**, 0.5 s silence ends a phrase, 10 s maximum. Latency is
measured from the end of the last speech frame. Verified offline (T-VAD-1).

**D6 - One worker thread, FIFO across both sources.** Models never compete
with each other for CPU; overlapping speech shows up as measurable queue
wait instead of hidden slowdowns. Labels are "Me" / "Remote" for now;
per-speaker labels come later.

**D7 - Sample-clock timing with a shared t0.** A stream's start time is the
earliest arrival in a 0.5 s warm-up window (a late first buffer used to shift
the whole stream: T-M0-1's +0.10 s). Late buffers are never padded with
zeros; zeros are inserted only for idle loopback, or when lateness has not
recovered for 1 s (confirmed loss), which is then reported as LOST.
Verified with stub tests only.

**D8 - Errors are reported, not hidden.** Lines over 5 s are flagged; lost
audio, overflows, stalled devices and processing exceptions print `!!` lines
live and are listed in the session summary. Wordless ASR output is not sent
to MT (NLLB invents sentences for it).

**D9 - Skip the 100-clip ASR benchmark and own-voice reading test** (user
decision, 2026-10-30): prioritise an end-to-end walking skeleton. Cost: no
local WER number; accent risk unmeasured.

**D10 - Debug the microphone loss one variable at a time.** The baseline
commit already contains three mitigations applied together (ASR thread cap
+ no spinning, `OMP_WAIT_POLICY=PASSIVE`, capture buffer 20 -> 50 ms). An
improvement with all three on would not show which one helped. Plan: make
them switchable with defaults equal to the T-LIVE-1 configuration, reproduce
the failure (T-LIVE-3), then test each on its own. Status: open. Flags now
exist (`--asr-threads N`, `--asr-no-spin`, `--omp-passive`, `--capture-buffer-ms N`);
with none given the program runs the T-LIVE-1 configuration.
