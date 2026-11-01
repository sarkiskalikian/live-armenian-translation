# Decisions

Each entry: **decision, why, evidence, status, commit**. Status is one of
*active*, *open*, *superseded* (replaced text kept, struck through) or
*proposed, not tested*. Evidence IDs refer to TESTS.md.

## Current pipeline (2026-10-31)

```
mic [14] ─┐                                                       ┌─ "Me"
          ├─ WASAPI capture ─ 16 kHz mono ─ Silero VAD phrases ─ FastConformer-hy CTC ─ Armenian text
loopback ─┘   (PyAudioWPatch)    (soxr)      (0.5 s silence)      (ONNX, fp32)            │
 [16]                                                                                     ▼
                                       terminal: [time] label / HY / EN / latency ◄─ NLLB-200 600M int8, beam 1
```

Audio -> FastConformer (CTC head) -> Armenian text -> NLLB -> English text.
Labels: "Me" (mic); remote phrases "Speaker 1", "Speaker 2", ... by voice (D20).

---

**D1 - System audio via WASAPI loopback (PyAudioWPatch).**
Why: Windows exposes a loopback twin of each output device, so no virtual
cable is needed and the user still hears the call. Evidence: T-M0-1.
Status: active. Commit: 4c6bb5d.

**D2 - Microphone = laptop array `[14]`, not the Buds' mic.**
Why: opening the Buds' "Headset" mic switches them to the hands-free profile,
moving call audio off loopback `[16]`; in-ear Buds leak little into the laptop
mic. Evidence: T-M0-1. Status: active.

**D3 - ASR: NVIDIA FastConformer-Hybrid Large (hy), CTC head, OpenVoiceOS ONNX
export (fp32) via onnx-asr.**
Why: best open model on ArmBench-ASR v1.0 (combined WER 19.9 vs
Whisper-large-v3 65.1), 115M params, no NeMo/PyTorch. Evidence: E3 - FLEURS
WER 11.92 % (95 % CI 9.75-14.22), CER 5.00 %, 0.145 s per 11.7 s clip on
the laptop. ~~"Not benchmarked locally"~~ superseded by E3. Known risks:
conversational audio (ArmBench movies ~61 % WER; our 4 call clips 23.8 %),
read Eastern Armenian training data, own-voice accent untested.
Status: active, confirmed by the user 2026-10-31. Commit: pinned in 692d014.

**D4 - MT: NLLB-200 distilled 600M, int8 CTranslate2 (third-party conversion,
pinned), Meta's SentencePiece model.**
Why: ~1 GB RAM, chrF 61-62 on clean FLEURS (T-MT-1). Licence CC-BY-NC (fine
for this challenge). ~~"Beam 4 is the current default"~~ superseded by D15
(beam 1). Known limitation: place names (E2: «Երեւանում» dropped,
«Վանաձոր» -> "Vancouver"). Status: active. Commit: pinned in 692d014.

**D5 - Phrase segmentation: Silero VAD frame by frame (32 ms), state carried;
0.5 s silence ends a phrase; 10 s maximum.** Latency is measured from the end
of the last speech frame. Evidence: T-VAD-1 (sandbox). Status: active;
not re-checked in Windows with the pinned Silero file.

**D6 - One worker thread, FIFO across both sources.** Why: models never
compete for CPU; overlap shows up as measured queue wait. Status: active.

**D7 - Sample-clock timing with a shared t0.** Stream start = earliest arrival
in a 0.5 s warm-up window; late buffers are never padded; zeros are inserted
for idle loopback, or when lateness has not recovered for 1 s (unrecovered
lag). Evidence: T-STUB. Status: active, **with a known defect**: the
re-anchoring inserts the whole correction as one hole at detection time,
which can fall inside a word (391 ms in `00053.76_Me.wav`). Proposed fix,
not made: offset timestamps instead of inserting audio.

**D8 - Errors are reported, not hidden.** Lines over 5 s, frame shortfalls,
PortAudio overflows, stalled devices and exceptions are printed live and
listed in the summary. Status: active. Commits: 2b1cebf (wording).

**D9 - Skip the 100-clip ASR benchmark and own-voice reading test**
(user, 2026-10-30), to build the walking skeleton first.
Status: ~~ASR benchmark skipped~~ superseded by E3 (FLEURS 100 clips run).
Own-voice test: still not run.

**D10 - Debug the mic frame shortfall one variable at a time.** Three
mitigations (ASR thread cap / no spinning, OMP passive, 50 ms buffer) were
made switchable, default = T-LIVE-1 configuration (03c8fb5). T-LIVE-2 and
T-REC-2 then showed the shortfall without models or pipeline work.
Status: paused, unresolved (user, 2026-10-30 22:14); flags exist, untested.

**D11 - Judge ASR/MT mainly on "Remote" lines** while the mic shortfall is
open. Status: active.

**D12 - Minimum-letters rule is a translation guard, not a quality score.**
Text with < 2 Armenian letters is not sent to NLLB (it invents sentences).
Short replies «այո», «ոչ», «հա», «չէ» stay translatable. Evidence: T-OFF-1.
Status: active; not yet seen live. Commit: cfc291a.

**D13 - Pin everything a result depends on; report only laptop-Windows
numbers.** Model files by Hugging Face commit + sha256, packages by exact
version (`python -m livetrans.models --verify`). ~~"the sandbox is used to
check scripts run"~~ superseded (user, 2026-10-31): all checks run in the
laptop's Windows environment; the sandbox is used only to edit files and run
git. Clarified 2026-10-31: deterministic reading of saved files (no models,
no timing, identical output on any OS - e.g. T-INSP-1) may run there and is
labelled laptop-sandbox; anything with models or timing runs in Windows.
Status: active. Commits: 692d014, 109c5bd.

**D14 - ASR head: keep CTC (E3).** Motivation for testing RNN-T: Karpov et
al., Interspeech 2025, Section 3 (hybrid FastConformer trained with RNN-T and
CTC decoders; RNN-T described as generally better) and Tables 1-2
(RNN-T-only WER). Motivation, not proof. Outcome: RNN-T fp32 -0.52 WER
points on FLEURS, 95 % CI -1.28 .. +0.26, p = 0.15, ~5x slower; not better
on 4 call clips. Decision: keep CTC; RNN-T int8 not tested (precondition not
met); RNN-T export stays pinned (ec92cef). Status: closed. Commit: 109c5bd.

**D15 - Warm-up and beam by measurement.** Warm-up: keep "minimal" (E1: first
phrase within ~0.07 s of warm; "representative" +2 s start-up, no gain).
Start-up is reported separately from live latency. Beam: **beam 1**
(E2: same meaning as beam 4 on 5 corrected call phrases, 203.5 vs 256.0 ms;
T-MT-1: p50 1.31 vs 1.67 s, -0.7 chrF). Live check T-LIVE-5: 34 phrases,
0 over 5 s, max 1.46 s, MT mean 0.37-0.41 s. Status: closed - warm-up
"minimal" and beam 1 are the defaults (040b505).
Commits: d46bbcf, 8b8a230, 040b505.

**D16 - Final model pipeline.** Silero VAD -> [loopback only: WeSpeaker
ResNet34 speaker label] -> FastConformer-hy CTC (fp32) -> NLLB-200 600M int8,
beam 1. Status: active; verified live in T-LIVE-5.

**D17 - NLLB-200 distilled 1.3B: proposed, not tested.** Candidate for the
place-name errors in E2. No measurement exists; no quality or speed claim is
made. Deferred (user, 2026-10-31). Status: proposed, not tested.

**D18 - Deferred for now (user, 2026-10-31):** NLLB-1.3B (D17), call-reference
expansion (10 unconfirmed rows) and further dataset work, own-voice test,
mic shortfall investigation (D10), zero-insertion fix (D7).

**D19 - Next priorities (user, 2026-10-31):** (1) ~~verify beam 1 in a short
live call~~ done (T-LIVE-5); (2) ~~remote-speaker labels~~ done (D20, T-SPK-2); (3)
final acceptance test against the brief. Status: open.

**D20 - Remote-speaker labels: WeSpeaker ResNet34 embeddings + online
clustering (diart rules only).** Each loopback phrase >= 1.0 s is embedded
(WeSpeaker voxceleb ResNet34-LM ONNX, loaded through onnx-asr - no new
package); cosine >= threshold to the best centroid -> that speaker; else a new
speaker, but only from phrases >= 1.5 s; centroids updated only from phrases
>= 1.5 s; phrases < 1.0 s inherit the previous remote label. Labels never
change once printed; mic phrases are always "Me". Why: small CPU model, no
extra dependency, follows the incremental-clustering rules from the plan
(Coria et al.) without their segmentation model. Known limits: one label per
phrase (overlap and two voices in one phrase are not split); short replies
may inherit the wrong label; Meet's noise gating removes parts of quiet
speech; VoxCeleb-trained model, untested on Armenian telephony audio.
Status: implemented behind `--speakers` (default off). Threshold: ~~0.45
placeholder~~ superseded by T-SPK-1 -> **0.30** (0.25-0.35 found the correct
2 speakers with 92.3 % agreement; >= 0.45 split one person into several).
Embedding cost ~35-42 ms per phrase. Model pinned (f0c48c2, d1551c3).
A -> B -> A at 0.30: PASS on real clips (T-SPK-1). A second real remote person
is not available now (user, 18:58), so the separate live validation uses
held-out recordings played into the loopback (T-SPK-2) instead of T-LIVE-7;
a real two-person call remains part of the final acceptance test.
T-SPK-2 passed (11/11 phrases, 2 speakers, A -> B -> A, max latency 1.94 s,
speaker step ~0.11 s): **speaker labels are on by default**; `--no-speakers`
prints "Remote" as before. Status: active.
