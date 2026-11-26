# PLAN — Live Armenian → English transcription with speaker labels

> Status (2026-10-30 evening): original plan, kept as written. Changes since:
> M1 benchmarks were cut back to the translation benchmark only (D9), work
> moved to an end-to-end walking skeleton, and speaker-embedding tests move to
> M5. See DECISIONS.md and TESTS.md for what actually happened.

Status: planning only. Nothing installed or implemented yet.
Target machine: Windows 11, i5-13500H, 8 GB RAM, Intel Iris Xe (no CUDA). Everything runs on CPU.

## 1. Goal

Build a Python terminal program that:
- captures the **microphone** (labelled `Me`) and the **system audio** (the call side) at the same time;
- cuts each stream into spoken phrases and transcribes them as Armenian text;
- translates each phrase into English;
- labels call-side phrases `Speaker 1`, `Speaker 2`, … and keeps each label stable for the whole session;
- prints `[time] speaker / HY / EN`. The target is ≤ 5 s from the end of a phrase to its English line. This is **a target to measure, not a promise.**

Local models only. Keep it small. Spare time goes to reliability, not features.

## 2. What the research changes

| Finding | Source | Consequence |
|---|---|---|
| NVIDIA FastConformer-Hybrid (hy), 115M params, is the **best open model** on the independent ArmBench-ASR v1.0 (combined WER 19.9; Whisper-large-v3 zero-shot 65.1). | ArmBench-ASR leaderboard; Karpov et al., Interspeech 2025 | Primary ASR candidate. It is small enough for CPU. |
| Even that model scores **~61% WER on movie dialogue**, against 7.7% on Common Voice read speech. It was trained on read speech and its card calls it noise-sensitive. | ArmBench-ASR; model card | Conversational call audio is the **main quality risk**. The README must state this honestly. |
| An **ONNX export** exists (CTC head, fp32) that runs with `onnx-asr` and ONNX Runtime. It needs **no NeMo or PyTorch**. There is no int8 version. | `OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx` | It avoids installing NeMo on Windows. The paper's numbers are for the RNN-T head, so CTC quality must be measured. |
| The model card says it is **non-streaming**. | model card | Run it once per VAD segment, not as a stream. |
| Armenian fine-tunes of Whisper exist (turbo ~0.8B, large-v3) but rank **below** FastConformer on ArmBench and are 7–13× larger. | ArmBench-ASR | Fallback only, if FastConformer fails on our audio. |
| Coria et al. (diart): rolling-buffer segmentation plus **incremental clustering**. New speaker if distance > δ_new. Update a centroid **only from long-enough segments** (ρ_update). Speaker confusion rises as latency drops. Labels improve as the conversation goes on. | Paper provided | Borrow the clustering rules and nothing else. Expect labels to be shaky in the first minute. |
| Newer streaming diarizers exist (FASTDIAR 2026, ~10M params, 5× real-time on one CPU thread). | microsoft/fastdiar | Fallback only, if the simple approach measurably fails. |
| sherpa-onnx ships ungated ONNX speaker-embedding models (3D-Speaker CAM++, WeSpeaker) for CPU. | sherpa-onnx docs | Gives speaker embeddings without pyannote or a Hugging Face token. |
| Translation: NLLB-200 distilled 600M is available as CTranslate2 int8 (~0.6 GB). TranslateGemma-4B (Gemma terms) Q4 is ~2.5 GB. Gemma-3/NLLB-3.3B score well on hye→eng, but those sizes are too big here. | HF model cards; OpenWHO eval | NLLB-600M int8 is primary. TranslateGemma-4B is an optional experiment, only if RAM and latency allow. |
| ArmBench and almost all training data are **Eastern Armenian**. | ArmBench; Chakmakjian & Wang 2022 | Your Western-influenced pronunciation on the mic is a specific risk. Test it explicitly. |

**Not taken from the papers:** data collection, pseudo-labelling, fine-tuning, overlap-aware segmentation models, hyper-parameter optimisation pipelines.

## 3. Architecture

```
 mic (WASAPI)       ─► Capture ─► VAD/Segmenter ─┐
 loopback (WASAPI)  ─► Capture ─► VAD/Segmenter ─┤
                                                 ▼
                                        segment queue (FIFO)
                                                 ▼
                       Worker: Speaker → ASR → Translate → Printer
```

- One capture thread per source. One shared inference worker, because parallel models on 4 performance cores would fight each other. If measurement shows ASR and MT pipelining helps, that becomes a two-stage worker.
- The pipeline is the same for live audio and for **WAV replay**, so every test can be repeated.
- Every segment carries timestamps, logged at each stage, so latency is measured rather than guessed.

## 4. Modules and their inputs/outputs

| Module | Input | Output | Initial choice (to confirm in M1) |
|---|---|---|---|
| `audio/devices.py` | — | printed device list; resolved device IDs from `--mic` / `--loopback` | PyAudioWPatch (WASAPI loopback, no virtual cable) |
| `audio/capture.py` | device ID | `(source, float32 mono 16 kHz chunk, t_capture)` | downmix + resample (soxr) |
| `audio/segmenter.py` | chunks of one source | `Segment(source, t_start, t_end, audio)` | Silero VAD (ONNX); silence hang-over ~0.5 s; max segment length ~10 s (both tunable) |
| `speakers.py` | `Segment` | `"Me"` for mic; `"Speaker N"` for loopback | CAM++ or ECAPA embedding + online cosine clustering with δ_new; centroid updated only for segments ≥ ~1.5 s |
| `asr.py` | segment audio | Armenian text (punctuated) | FastConformer-hy ONNX (CTC) via onnx-asr |
| `translate.py` | Armenian text | English text | NLLB-200-distilled-600M, CTranslate2 int8 |
| `output.py` | labelled result + timings | terminal lines; optional `--debug-latency` column | plain print |
| `pipeline.py` / `main.py` | CLI flags | wiring, threads, queue, shutdown | stdlib `threading` + `queue` |
| `tools/replay.py`, `tools/bench_*.py` | WAV files | the same pipeline offline; metric tables | jiwer, sacrebleu |

Each model module exposes one function so it can be swapped (`transcribe(audio) -> str`, `translate(text) -> str`, `assign(segment) -> label`). There are no plugin frameworks.

## 5. Major risks

1. **ASR quality on conversational and call audio.** The domain gap from read speech is large (ArmBench Movies ~61% WER). Codecs, noise, and short replies like «հա», «լավ» make it worse.
2. **Western-accented speech on the mic.** The model is Eastern-trained and the accent effect is unmeasured.
3. **CPU contention and latency.** Both sides plus the call app share one laptop. Backlog in the queue directly breaks the 5 s budget.
4. **8 GB RAM.** Models (~2 GB estimated, *to measure*) plus Windows plus the call app plus a browser.
5. **CTC vs RNN-T.** The ONNX export uses the CTC head. Quality may be below the published numbers.
6. **Diarization on short segments.** Embeddings from segments under ~1–1.5 s are unreliable. Cold start produces label churn. A VAD segment may contain two speakers if there's no pause. Overlap is ignored. Codec audio can make voices sound alike.
7. **Translation of noisy or fragmentary ASR output.** NLLB can hallucinate on very short or broken input. NLLB is CC-BY-NC, which is fine for this challenge but must be noted.
8. **Windows audio quirks.** The loopback device changes when output switches between headphones, Bluetooth and speakers. Sample rates and channel counts vary.
9. **Audio leakage without headphones.** Echo handling is out of scope, but leakage causes duplicate `Me` lines. This will be documented rather than fixed.

## 6. Milestones and how each is tested

| # | Milestone | Done when / test |
|---|---|---|
| M0 | **Audio capture.** List devices, pick them by flag, record mic and loopback together to WAV. | Record 60 s of a real call (Meet, Zoom, Telegram) with headphones and with speakers. Both WAVs are audible, there are no dropouts, and they are time-aligned. Unplugging or switching the device gives a clear error, not a hang. |
| M1 | **Model spike (offline benchmark, decision gate).** No live code yet. | ASR: ~50 FLEURS `hy_am` + ~20 Common Voice clips + your own recordings, split into Eastern speakers and your voice. Report CER/WER, real-time factor and peak RAM for FastConformer-CTC, plus Whisper-turbo-hy as a reference. MT: ~100 FLORES hye→eng sentences, clean and with ASR output as input. Report chrF and ms per sentence for NLLB-600M, with TranslateGemma-4B only if RAM allows. Speakers: cosine-score distributions for same vs. different speaker at 1, 2 and 4 s. **Outputs: a choice of models plus written numbers for the README.** |
| M2 | **Live segmentation.** VAD on both streams prints `[time] source segment-length`. | Replay WAVs at real-time speed and count segments against hand-marked phrases. Live: speaking short and long phrases creates no merged or chopped segments. Tune the hang-over and the maximum length here. |
| M3 | **ASR in the loop.** Live Armenian text with latency logging. | Replay test set: CER matches M1, which shows the integration adds no errors. The latency log shows p50/p95 from end of segment to text. If p95 is already near 5 s, stop and re-plan before adding MT. |
| M4 | **Translation in the loop.** | Same replay with EN output. Spot-check ~30 lines by hand (you're fluent in both languages). The p95 end-to-end latency is reported. |
| M5 | **Speaker labels on loopback.** | Build a replay file of 3 known speakers taking alternating turns (from your recordings or CV clips with different `client_id`s). Report segment-level label accuracy and the number of spurious new speakers. Tune δ_new and the minimum update duration on this file only, then confirm on a second file. |
| M6 | **Full live run and hardening.** | 30-min session with both sides active: RAM is stable, the queue doesn't grow, Ctrl+C exits cleanly. Rehearse the interview setup with ≥ 2 people on the call side, including you speaking with your accent. Record the honest results. |
| M7 | **Deliverables.** | README: setup, devices, models with alternatives and numbers, architecture, limitations. Export every chat session to `chats/` as separate `.md` files. Confirm the repo contains no weights, audio or keys (`.gitignore` plus a final check). |

**Decision gates:**
- After M1: if FastConformer-CTC is clearly worse than expected, try exporting the RNN-T head or switching to the Whisper fallback.
- After M3/M4: if latency is too high, shorten the maximum segment length, cut thread counts or use a smaller MT model.
- After M5: only if label accuracy is clearly poor, consider a streaming diarizer (diart or FASTDIAR).

## 7. Explicitly not doing
GUI, packaging, echo cancellation, named or persistent speakers, saving transcripts, non-Armenian input, fine-tuning, LLM post-editing, or GPU/OpenVINO work. The last one is revisited only if M3/M4 latency forces it.

## 8. References
- Coria, Bredin, Ghannay, Rosset — *Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation* (diart).
- Karpov et al. — *From Scarcity to Sufficiency: Speech Recognition Pipeline for Zero-resource Language*, Interspeech 2025.
- Metric AI Lab — ArmBench-ASR v1.0 leaderboard (metric-ai-armbench-asr.static.hf.space).
- nvidia/stt_hy_fastconformer_hybrid_large_pc; OpenVoiceOS ONNX export.
- Torgashov & Köpüklü — FASTDIAR (arXiv 2610.02941), as a fallback only.
- Chakmakjian & Wang — *Towards a Unified ASR System for the Armenian Standards*, DigitAm 2022.
