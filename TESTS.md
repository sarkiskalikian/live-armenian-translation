# Test log

Each experiment records: **question, environment, configuration, measured
outcome, decision, limitations, commit**. "Measured" = read from logs,
recordings or result files; anything inferred is labelled as such.
Conclusions overturned by later tests are kept and marked **SUPERSEDED**.

**Environments**
- **[laptop-Windows]** - the target laptop: Windows 11, Intel i5-13500H
  (12 physical / 16 logical cores), 7.6 GB usable RAM, no GPU; Python 3.13.14
  `.venv` with the pinned versions in `requirements.txt` (onnxruntime 1.31.1,
  onnx-asr 0.12.0, ctranslate2 4.8.4, sentencepiece 0.2.2, numpy 2.5.4, soxr
  1.1.0, PyAudioWPatch 0.2.12.9). Devices: mic `[14]` Intel Smart Sound array
  (2 ch, 48 kHz); loopback `[16]` Galaxy Buds3 Pro "Headphones" (2 ch, 48 kHz).
  Call app: Google Meet in a browser. **All reported numbers come from here.**
- [laptop-sandbox] - a Linux VM on the same laptop (Python 3.10, onnxruntime
  1.23.2, 2 vCPUs). Used for T-VAD-1 and T-OFF-1 only; from 2026-10-31 used
  only for editing files and git, never for results (user decision).
- [cloud] - assistant's workspace, no audio hardware: logic tests with stub
  models only.

**Terminology.** *Frame shortfall*: a stream delivered fewer frames than wall
time x nominal rate. *Unrecovered-lag event*: lateness > 0.3 s that did not
recover within 1 s. Neither proves the device dropped audio (a slow device
clock would look the same).

## Index

| ID | Question | Env | Result | Status |
|---|---|---|---|---|
| T-M0-1 | Do mic and loopback capture both voices? | laptop-Windows | yes, no overflows | done |
| T-VAD-1 | Does Silero cut phrases correctly? | laptop-sandbox | 9 + 6 phrases in the right places | done |
| T-STUB | Aligner/segmenter/worker logic | cloud | logic correct with stubs | done |
| T-MT-1 | NLLB-600M quality/speed on FLEURS | laptop-Windows | chrF 61.1 (b1) / 61.7 (b4) | done |
| T-LIVE-0 | First live run | laptop-Windows | latency 2.2-3.2 s, mic OK | done |
| T-LIVE-1 | Two-person call | laptop-Windows | mic frame shortfall (92.9 %) | failed |
| T-LIVE-2 | Shortfall without models? | laptop-Windows | yes (84.0 %, 15.6 s run) | done |
| T-REC-2 | Shortfall in plain recorder? | laptop-Windows | yes (87.3 %) | done |
| T-LIVE-4 | Armenian conversation, HY+EN | laptop-Windows | latency mostly < 5 s; ASR main error source | done |
| T-OFF-1 | Min-letters rule on T-LIVE-4 text | laptop-sandbox | changes 3 of 36 lines | done |
| E0 | Pinned models/runtime verify | laptop-Windows | not reported | open |
| E1 | Warm-up minimal vs representative | laptop-Windows | no gain; keep minimal | done |
| E2 | Beam 1 vs 4 | laptop-Windows | same meaning, beam 1 faster | done -> beam 1 |
| E3 | ASR CTC vs RNN-T fp32 | laptop-Windows | -0.52 pt WER, not significant | done -> keep CTC |
| T-LIVE-5 | Beam 1 in a live call | laptop-Windows | 0/34 over 5 s, max 1.46 s | done -> beam 1 kept |
| T-LIVE-6 | Call with two remote people (dev recording) | laptop-Windows | 0/44 over 5 s, max 1.75 s | done |
| T-INSP-1 | Where did missed speech go? (T-LIVE-6 files) | laptop-sandbox | Meet gates speech to digital silence | done |
| T-SPK-0 | Speaker tracker logic | cloud | A->B->A = 1->2->1 with stub embeddings | done |
| T-SPK-1 | Speaker labels on real T-LIVE-6 clips | laptop-Windows | 2 speakers found at 0.25-0.35, 92.3 % agreement; A->B->A PASS at 0.45 | done -> threshold 0.30 |
| T-LIVE-7 | Separate live call with --speakers | laptop-Windows | - | pending |

---

## T-M0-1 - Capture both sources (2026-10-30 20:18)
- **Question:** do the mic and the system audio both get captured, separated and complete?
- **Environment:** laptop-Windows; Meet call, Buds as output, laptop mic.
- **Configuration:** `tools/m0_record.py --guided` (60 s scripted phases), 20 ms buffer, `tools/m0_check.py`. Recording `recordings/20261030-201834`.
- **Measured:** mic 2,854,080 frames, length 60.01 s = wall 60.01 s; loopback 2,857,920 frames, 60.11 s (+0.10 s); 0 overflows; max queue 1; no clipping. Remote-only speech (21-30 s): mic at its floor (~-75 dB). Overlap (33-43 s): both streams -16 to -22 dB.
- **Decision:** capture design accepted (D1, D2).
- **Limitations:** speakers-instead-of-headphones and device-unplug tests not run. The +0.10 s was traced to anchoring a stream on its first buffer; fixed in code (D7), not re-measured.
- **Commit:** run before the repo existed (code later committed in e7817a9 with changes); recorded in 1f710b5.

## T-VAD-1 - Phrase segmentation on the M0 recording
- **Question:** does frame-by-frame Silero VAD cut phrases at the right places?
- **Environment:** laptop-sandbox (Silero 6.2.3 from PyPI, not the pinned file).
- **Configuration:** `tools/try_file.py --vad-only`, 0.5 s silence, 10 s max.
- **Measured:** mic 9 phrases inside 5-17 s and 33-45 s; loopback 6 phrases inside 21-30 s and 33-43 s; none in silence; no remote voice in the mic stream.
- **Decision:** keep D5 settings.
- **Limitations:** not re-run in Windows with the pinned Silero file; no hand-marked phrase boundaries.
- **Commit:** pre-repo code; recorded in 1f710b5.

## T-STUB - Logic tests without hardware
- **Question:** does the timing/queue logic behave as designed?
- **Environment:** cloud, synthetic buffers, stub models.
- **Measured:** phrase ends within 20 ms; queue wait accounted when sources overlap; late first buffer + burst no longer shifts a stream; late buffers not padded; idle loopback gap filled; a simulated 7 % drop reported as 6.4 of 6.7 s unrecovered lag, ratio 0.929.
- **Decision:** D7 aligner rules.
- **Limitations:** logic only; says nothing about the real device. Did not anticipate the zero-insertion defect found later (see "Open defects").
- **Commit:** code in e7817a9 / later aligner commits; recorded in 1f710b5.

## T-MT-1 - Translation benchmark on FLEURS (2026-10-30 ~20:44)
- **Question:** NLLB-600M int8 quality and speed, beam 1 vs 4.
- **Environment:** laptop-Windows.
- **Configuration:** `tools/m1_bench_mt.py`, 100 FLEURS hy_am test sentences, reference (clean) Armenian input, 4 threads. `results/m1_mt_summary.json`.
- **Measured:** beam 1: chrF 61.08, BLEU 35.38, p50 1307 ms, p95 2486, max 3637. Beam 4: chrF 61.74, BLEU 36.29, p50 1674, p95 3543, max 5937. Peak RAM 996 MB.
- **Decision:** none at the time (beam 4 kept); used later as evidence for beam 1 (E2).
- **Limitations:** clean input only (ASR-output condition not run); long read sentences, not call phrases. ~~"The ASR benchmark was not run"~~ **SUPERSEDED by E3** (FLEURS ASR run).
- **Commit:** tool in e7817a9; recorded in 1f710b5.

## T-LIVE-0 - First live run (2026-10-30 21:04)
- **Question:** does the end-to-end pipeline run live?
- **Environment:** laptop-Windows, Meet.
- **Configuration:** beam 4, ASR threads = runtime default, 20 ms buffer, min-silence 0.5 s. `logs/20261030-210400`. Only the mic produced speech (4 phrases, 4.7 s).
- **Measured:** latency 2.2-3.2 s (all < 5 s); ASR mean 1.18 s, MT mean 1.10 s; mic VAD-wait 0.56-0.60 s with no growth (derived sample-clock ratio 0.999).
- **Decision:** proceed to two-person test.
- **Limitations:** 4 phrases; no remote speech.
- **Commit:** code state before e7817a9 (not in git; its configuration = defaults of b72c767); recorded in 1f710b5.

## T-LIVE-1 - Two-person Meet call (2026-10-30 21:13) - FAILED (mic timing)
- **Question:** does the pipeline keep up in a real two-person call?
- **Environment / configuration:** as T-LIVE-0. `logs/20261030-211345`. User: "when we talked a bit faster it did not catch all the words".
- **Measured:** 14 Me + 11 Remote phrases. Remote VAD-wait 0.50-0.55 s throughout. Me VAD-wait grew from 0.55 s (t = 4.0 s) to 7.07 s (t = 91.5 s), fit `-0.07 + 0.0769 t`. 4315 buffers > 0.3 s late, max 6.93 s; 0 overflow flags. Latency Me p50 4.62 s, p95 8.18 s, 6/14 over 5 s; Remote p50 1.85 s, max 2.08 s, 0 over. ASR mean 0.44 s, MT 0.77 s, queue 0.
- **Derived:** mic sample clock at 92.9 % of wall time (raw frame counts not logged in this version).
- **Decision:** add frame-count logging and a no-models mode; investigate (D10).
- **Limitations:** ~~H1 "model CPU starvation" as the explanation~~ **SUPERSEDED by T-LIVE-2 and T-REC-2** (shortfall occurs without models). Remaining hypotheses unconfirmed.
- **Commit:** code as T-LIVE-0; recorded in 1f710b5.

## T-LIVE-2 - Models off (2026-10-30 21:54) - mic frame shortfall reproduced
- **Question:** does the mic shortfall occur without ASR/MT load?
- **Environment:** laptop-Windows, same Meet setup.
- **Configuration:** commit b72c767, `--no-models` (capture + resampling + VAD only), T-LIVE-1 settings, 20 ms buffer. `logs/20261030-215404`. Ran 15.6 s (planned 2 min).
- **Measured:** Me delivered ratio 0.840, 3 unrecovered-lag events (1.57 s), max lateness 0.556 s, 0 overflows. Remote 0.993, no events.
- **Decision:** model CPU load is not the sole cause; the ASR/OMP mitigation flags are not the first lever.
- **Limitations:** short run; ~~"shortfall comes in ~0.5 s blocks"~~ **SUPERSEDED by T-REC-2**: the regular ~0.45 s events are an artefact of the lag detector.
- **Commit:** code b72c767; recorded in 7938c53.

## T-REC-2 - Plain recorder in the same call (2026-10-30 22:01) - shortfall reproduced
- **Question:** does the shortfall occur without the live pipeline's per-buffer work?
- **Environment:** laptop-Windows, Meet, both people talking.
- **Configuration:** commit 7938c53, `tools/m0_record.py --seconds 120`, 20 ms buffer. `recordings/20261030-220126`.
- **Measured:** mic 4,985,280 frames = 103.86 s over 118.91 s -> 0.873; 32 unrecovered-lag events (14.70 s) every ~3 s; 0 overflows. Loopback 1.002.
- **Decision:** neither models nor pipeline processing is necessary for the shortfall; environment (H3: driver / Meet sharing the mic / power state) leading but unconfirmed. Investigation paused 22:14 by user decision.
- **Limitations:** dropped packets vs slow device clock not distinguished (callback log added afterwards, eafc972, not yet used). User's listening impression ("no problem") disagreed with the frame counts.
- **Commit:** code 7938c53; recorded in 15671ca.

Mic delivery over time (same device, 20 ms buffer): T-M0-1 20:18 **1.000**; T-LIVE-0 21:04 0.999 (derived); T-LIVE-1 21:13 0.929 (derived); T-LIVE-2 21:54 0.840; T-REC-2 22:01 0.873; T-LIVE-4 22:15 0.909.

## T-LIVE-4 - Armenian conversation, HY and EN (2026-10-30 22:15)
- **Question:** how good are recognition, translation and latency in a real call?
- **Environment:** laptop-Windows, Meet, two people, 124 s.
- **Configuration:** commit eafc972, beam 4, T-LIVE-1 settings, `--save-audio`. `logs/20261030-221541`.
- **Measured:** 36 phrases (22 Me, 14 Remote). Latency Me p50 1.97 s, p95 5.14 s, max 5.32 s, 2 over 5 s (the first two phrases: ASR 1.8 s + MT 2.9 s, then 3.4 s queue); Remote p50 1.70 s, max 2.53 s, 0 over. Mean ASR 0.34-0.49 s, MT 0.70-0.80 s. Me delivered 0.909; Remote 0.997.
- **Quality (assistant's reading at the time):** ~11 lines correct or close, ~13 recognition errors carried into English, ~5 correct Armenian/wrong English, ~7 one-letter hallucinations. ~~"~7 hallucinations fixed by a letter rule"~~ **SUPERSEDED by T-OFF-1** (3). ~~"Երեւանում -> Yes, you are is an MT error"~~ **SUPERSEDED by the user's references** (it was an ASR error: «Այո» for «Ես»). ~~"slow first phrase = cold start"~~ **SUPERSEDED by E1** (not model cold start; cause unconfirmed).
- **Decision:** recognition is the main error source; fix one-letter hallucinations (T-OFF-1); test warm-up, beam, ASR head (E1-E3).
- **Limitations:** quality judged informally except the 5 phrases later referenced by the user; Me audio affected by the shortfall.
- **Commit:** code eafc972; recorded in 5d86997.

## T-OFF-1 - Minimum-letters rule on T-LIVE-4 output
- **Question:** which lines does "translate only if >= 2 Armenian letters" change?
- **Environment:** laptop-sandbox (deterministic text replay, no timing).
- **Configuration:** commit 86a44f3; the 36 recognised strings of `logs/20261030-221541`.
- **Measured:** 3 lines change («Հ։» x2, «Ս։»); 33 unchanged; «Հովհ.» and «Հահ։» still translated (3-4 letters).
- **Decision:** keep the rule as a translation guard only (D12); short replies («այո», «ոչ», «հա», «չէ») stay translatable.
- **Limitations:** not yet observed in a live run (will be in T-LIVE-5); does not catch longer hallucination triggers.
- **Commit:** 86a44f3.

## E0 - Pinned models and runtime (`python -m livetrans.models --verify`)
- **Question:** do the installed files and packages match the pins?
- **Environment:** laptop-Windows (to be run); model-file hashes were checked against the laptop's files on 2026-10-30 (all four OK; RNN-T files OK after download).
- **Status:** the Windows `--verify` output has not been reported. Package versions were read from the `.venv` dist-info folders.
- **Commit:** pins in 270fea2; RNN-T pin in cdb7cca.

## E1 - Warm-up: minimal vs representative (2026-10-31 ~00:37)
- **Question:** does a more representative warm-up remove first-phrase slowness?
- **Environment:** laptop-Windows.
- **Configuration:** commit b834391, `tools/warmup_check.py`, 3 fresh processes per mode, alternating; beam 4; clips T-LIVE-4 Remote 00019.81 / 00037.95 / 00070.08. `results/warmup_check.json`.
- **Measured (medians, ASR / MT s):** minimal: load 6.03, warm-up 1.11, clip1 first 0.131 / 0.629, clip1 again 0.065 / 0.571. Representative: load 8.88, warm-up 3.16, clip1 first 0.183 / 0.689, clip1 again 0.058 / 0.506. One run per mode was slow on every step (machine-wide slowdown).
- **Decision:** keep "minimal" (default unchanged).
- **Limitations:** 3 runs per mode; measured with beam 4 (live default now beam 1); quiet laptop, not during a call.
- **Commit:** code b834391; recorded in a2a792a.

## Call references (2026-10-31 00:59-01:13)
`eval/t-live-4_refs.jsonl`, user (native speaker) listened to each clip; unclear words were asked about before confirming.

| clip | reference | live CTC output | word errors |
|---|---|---|---|
| 00019.81_Remote | Հաճելի է, իմ անունը գարուն է։ | Պատելի է, ... | 1/6 |
| 00049.31_Remote | Ես Երեւանում, դու՞ւք։ | Այո Երեւանում, ... | 1/3 |
| 00053.76_Me | Ես Վանաձորից եմ։ | (same) | 0/3 (MT test only) |
| 00070.08_Remote | Ես զբաղվում եմ կլինիկական հետազոտություններ հետ։ | ... զբաղում ... քինիկական ... | 2/6 |
| 00095.46_Remote | Մեր մոտ ժամը տաս տասնհինգ անց։ | ... տ տասնհինգ անց | 1/6 |

00049.31 was first confirmed as recognised (01:02); **SUPERSEDED** at 01:13 (first word «Ես»). Open: whether «գարուն» is a name («Գարուն»); the reference keeps the recogniser's spelling «դու՞ւք» (affects E3 calls scoring). 10 other Remote rows unconfirmed.
- **Commits:** 7144ea5, 83173cf, 6b929c6, 8dbee71, 75c930d, 7641f54.

## E2 - Beam 1 vs beam 4 on 5 call phrases (2026-10-31 01:14)
- **Question:** does beam 1 keep translation quality while lowering latency?
- **Environment:** laptop-Windows.
- **Configuration:** commit b834391 tool, refs 7641f54, `tools/beam_check.py`; NLLB-600M int8, 4 threads; each translation timed as median of 5 after warm-up; inputs = corrected Armenian and live ASR output. `results/beam_check.json`.
- **Measured:** median 203.5 ms (b1) vs 256.0 ms (b4); identical English in 6/10, in 4/5 on corrected Armenian. Corrected-Armenian outputs (b1 = b4 unless shown): "It's nice. My name is Spring." / "I'm coming, are you?" (Yerevan lost) / "I'm from Vancouver." / "I'm in clinical trials." / b1 "We're about 10:15 in the morning." vs b4 "It's about 10:15 in the morning."
- **Decision:** switch the live default to beam 1 (d194c54), pending T-LIVE-5.
- **Limitations:** 5 phrases cannot detect small quality differences; English judged by the assistant, user may correct; place-name errors (Երեւանում, Վանաձոր) occur with both beams.
- **Commit:** recorded in 9eb54c5; default changed in d194c54.

## E3 - ASR head: CTC fp32 vs RNN-T fp32, same NVIDIA weights (2026-10-31 01:17)
- **Question:** does the RNN-T head of the same model recognise better than our CTC export?
- **Environment:** laptop-Windows.
- **Configuration:** `tools/asr_compare.py` (commit b834391), refs 7641f54; CTC = OpenVoiceOS export @ f7db1fa (model.onnx sha256 09ed4278...); RNN-T = Vana-Labs export @ ec92cef (encoder 541c3462..., decoder_joint 5bb91df1...); same onnx-asr 80-band preprocessing; default threads; one untimed warm-up clip; WER/CER after normalisation (case, punctuation, եւ/և); bootstrap 10,000 clip resamples. `results/asr_{ctc,rnnt}_{fleurs,calls}.json`.
- **Measured, FLEURS (100 clips, 1930 words, 11.7 s average):**

| | CTC fp32 | RNN-T fp32 |
|---|---|---|
| WER (95 % CI) | 11.92 % (9.75-14.22) | 11.40 % (9.33-13.58) |
| word errors | 230 | 220 |
| CER | 5.00 % | 4.81 % |
| clips perfect | 28 % | 30 % |
| clips WER <= 10 % / 10-30 % / > 30 % | 58 / 32 / 10 % | 55 / 37 / 8 % |
| worst clip | 50 % | 42 % |
| time per clip median / p95 / max | 0.145 / 0.245 / 0.445 s | 0.772 / 1.162 / 1.663 s |

  Paired: RNN-T better on 25 clips, worse on 15, equal on 60; difference -0.52 points (95 % CI -1.28 .. +0.26), sign test p = 0.15.
- **Measured, calls (4 confirmed intact Remote clips, 21 words):** CTC 5 errors (23.8 %), RNN-T 7 (33.3 %).
- **Decision:** keep CTC as the live model (confirmed by the user 2026-10-31); do not run RNN-T int8 (E4), since its precondition (RNN-T fp32 better) was not met. RNN-T stays pinned for a possible later comparison.
- **Limitations:** FLEURS is clean read Eastern Armenian; the call set is too small and one RNN-T "error" is a reference spelling artefact («դու՞ւք»). Karpov et al. (motivation, not proof) report RNN-T-only numbers.
- **Commit:** pins 270fea2; recorded and RNN-T pinned in cdb7cca.

## T-LIVE-5 - Beam 1 in a short live call (2026-10-31 17:23)
- **Question:** with beam 1 as default, are HY/EN lines and latency acceptable live?
- **Environment:** laptop-Windows, Meet, two people (same remote person as T-LIVE-4), 140.9 s.
- **Configuration:** commit bdc9c97 (code = d194c54 defaults): beam 1, CTC, warm-up minimal, ASR threads default, 20 ms buffer, min-silence 0.5 s, `--save-audio`. `logs/20261031-172334`.
- **Measured:**
  - 34 phrases (18 Me, 16 Remote); 0 processing failures; max queue depth 1.
  - Latency Me p50 0.98 s, p95 1.30 s, max 1.46 s; Remote p50 0.99 s, p95 1.29 s, max 1.31 s; **0 of 34 over 5 s**, first phrase 1.46 s.
  - Mean parts: VAD wait 0.52-0.61 s, queue 0.00-0.02, ASR 0.06 s, MT 0.37-0.41 s.
  - Start-up (separate from live latency): load 8.08 s, warm-up 0.89 s.
  - Capture: Me delivered 0.992 (2 unrecovered-lag events, 0.71 s); Remote 0.997.
  - Min-letters rule seen live: «Հ։» not translated. «Հ հ։» (2 letters) -> "What is it?", «Սովհ։» -> "I'm not sure.", «Հովհ։» -> "John, you know?" still translated (known limitation, D12).
  - Short reply «Այո։» was translated as "I did." (should be "Yes.").
- **Comparison with T-LIVE-4 (beam 4, 2026-10-30):** MT mean 0.37-0.41 vs 0.70-0.80 s; latency p50 ~1.0 vs 1.70-1.97 s; max 1.46 vs 5.32 s. **Not a one-variable comparison:** different conversation, different day, and ASR time also fell (0.06 vs 0.34-0.49 s) although ASR settings were unchanged, so part of the gain is machine/session state. E2 (same inputs) attributes ~50 ms per short phrase to beam 1.
- **Quality (assistant's reading, not user-referenced):** most wrong lines start with wrong Armenian (e.g. «Առեւ ձեզ» for «Բարեւ ձեզ» -> "Without you"); translation errors on apparently correct Armenian include «Այո» -> "I did.", «Պար չեմ գնում» -> "...not going for money", "...you're a program". No beam 4 output exists for these lines, so beam cannot be blamed or cleared for them.
- **Decision:** keep beam 1 (D15 verified live: no line over budget, no new failure mode).
- **Limitations:** quality not scored against references; single remote speaker; mic shortfall was small this session (0.992 vs 0.87-0.91 on 2026-10-30) - recorded under Open defects, not explained.
- **Commit:** code d194c54; recorded in this commit.

## T-LIVE-6 - Call with two remote people on one phone (2026-10-31 17:34)
- **Question:** development recording for remote-speaker labels; also HY/EN and latency with beam 1.
- **Environment:** laptop-Windows, Meet; remote side = Գարուն and her brother sharing one phone; brother far from the phone.
- **Configuration:** commit 1b5ed69 defaults (beam 1, CTC), `--save-audio`. `logs/20261031-173439`.
- **Measured:** 44 phrases (16 Me, 28 Remote); latency Me p50 1.22 s, max 1.75 s; Remote p50 1.06 s, max 1.52 s; 0 over 5 s; MT mean 0.43-0.48 s, ASR 0.06-0.07 s. Capture: Me 0.987 (6 unrecovered-lag events, 2.09 s), Remote 0.999.
- **User's account:** felt fast; some speech missed - partly accent, partly microphones, partly other reasons; model working well overall.
- **Speaker labels (user, partial):** clips 1-7 and 9-13 = A (Գարուն); clip 8 and 14-28 not yet labelled. User transcript corrections stored as references in `eval/t-live-6_speakers.jsonl` (10 confirmed, 2 unclear).
- **Commit:** code 1b5ed69; labels 906f022.

## T-INSP-1 - Capture gaps vs phrase detection vs recognition, T-LIVE-6 files
- **Question:** for the missed speech in T-LIVE-6, what can the saved files attribute to capture, phrase detection or recognition?
- **Environment:** laptop-sandbox - deterministic reading of the saved WAV/JSON files (no models, no timing; the same numbers on any OS).
- **Configuration:** `tools/inspect_session.py logs/20261031-173439` (this commit). Writes `inspection.json` in the log folder.
- **Measured:**
  - Remote stream: delivered 0.9993, no unrecovered-lag events, no idle filling - our code inserted nothing into remote audio. Yet **all 28 Remote clips contain runs of exact digital zeros**: ~130-210 ms at the start and ~200 ms at the end of almost every clip, and mid-phrase runs in several (e.g. `00054.40_Remote` 468 ms at 0.72 s; `00109.15_Remote` 192 ms at 1.41 s). This digital silence comes from the call app (Meet's noise suppression / gating) before the audio reaches the laptop.
  - Mic stream: 6 unrecovered-lag events (2.09 s) but **0 saved Me phrases contain exact-zero runs** - this time the aligner's inserted silence fell outside speech.
  - Phrase onsets: 1 of 44 clips starts already at full level (possible clipped onset); 0 phrases cut at the 10 s maximum.
- **Inferred:**
  - Capture: on the remote side, parts of quiet/distant speech (the brother) were replaced by silence **by Meet**, so they never existed in our input. Our capture lost nothing measurable on that side.
  - Phrase detection: no evidence of clipped onsets or forced cuts inside saved phrases. **Speech that the VAD never detected is not on disk** (only detected phrases are saved), so the files cannot show whether any speech was missed between phrases.
  - Recognition: fragments such as «Հ մազերը» for «Ձեր մազերը» occur in clips whose start is Meet's digital silence followed by speech; whether the first consonant was gated by Meet or misrecognised cannot be separated from these files. Accent effects need the user's references.
- **Limitations:** no continuous recording of the session exists; to measure detection misses, a full-session recording (m0_record.py alongside, or a save-everything option) would be needed.
- **Commit:** this commit.

## T-SPK-0 - Speaker tracker logic with stub embeddings
- **Question:** does the online tracker give stable labels and follow its rules?
- **Environment:** cloud; synthetic 256-d voice vectors (same voice cosine ~0.75, different ~0.0); stub ASR/MT in the worker.
- **Configuration:** `livetrans/speakers.py` and `livetrans/pipeline.py` (this commit), threshold 0.45, min_embed 1.0 s, min_new 1.5 s, min_update 1.5 s.
- **Measured:** A->B->A -> Speaker 1, 2, 1; three voices A,B,C,A,C,B -> 1,2,3,1,3,2; a 1.2 s clip of a new voice is not allowed to create a speaker; a short clip (< 1.0 s) inherits the previous remote label - including when another person said it (A 0.7 s after B -> "Speaker 2"); short matches do not move a centroid. In the worker: loopback lines labelled Speaker N, mic lines stay "Me", `spk` time reported; without `--speakers` output is unchanged ("Remote").
- **Decision:** wire into the live program behind `--speakers` (default off until T-SPK-1 and T-LIVE-7 pass).
- **Limitations:** logic only; real embeddings and thresholds untested.
- **Commit:** this commit.

## T-SPK-1 - Speaker labels on the real T-LIVE-6 Remote clips (2026-10-31 18:28)
- **Question:** do WeSpeaker embeddings + the online tracker separate the two remote people, at which threshold, and does A -> B -> A give Speaker 1 -> 2 -> 1?
- **Environment:** laptop-Windows (Python 3.13.14, onnxruntime 1.31.1, onnx-asr 0.12.0).
- **Configuration:** commit 5454b11, `tools/speaker_check.py eval/t-live-6_speakers.jsonl`; WeSpeaker ResNet34-LM @ f0c48c2 (sha256 7bb2f06e...), 2 threads; tracker min_embed 1.0 s, min_new 1.5 s, min_update 1.5 s. Labels by the user (all 28 clips): A 22, B 4 (8, 14, 15, 26), excluded: 19 (user's own voice echoed via the remote phone) and 24 (B, "not clear"). `results/speaker_check_t-live-6_speakers.json`.
- **Measured:**
  - Embedding time per clip: median 35 ms, max 104 ms (25 of 28 clips >= 1.0 s embedded); model load 0.63 s.
  - Cosine similarity: same speaker n=196, mean 0.34 (p10 0.15, p90 0.53); different speakers n=80, mean 0.11 (p10 -0.01, p90 0.23, max 0.64). Same-speaker pairs below 0.30: 40 %; different-speaker pairs at or above 0.30: 7 %.
  - Full replay in time order (26 scored clips):

| threshold | agreement | speakers created |
|---|---|---|
| 0.25 / 0.30 / 0.35 | 92.3 % | 2 (correct) |
| 0.40 | 88.5 % | 3 |
| 0.45 / 0.50 / 0.55 | 88.5 % | 5 / 7 / 8 |
| 0.60 / 0.65 | 92.3 % | 8 |

  - At 0.30 the two errors are clip 8 (B, 1.35 s, too short to found a speaker -> given Speaker 1) and clip 10 (A, 1.6 s, audio "not clear" per user) - **clip 10 founded Speaker 2**, which B's clips 14, 15, 26 then joined. Clip 19 (echoed own voice) -> Speaker 1.
  - A -> B -> A (clips 00008.26 A, 00083.07 B, 00104.67 A) at threshold 0.45: **PASS** (Speaker 1, 2, 1).
- **Decision:** default threshold 0.30 (middle of the range that found the right number of speakers; 0.45 and above split one person into several). Speaker labels stay behind `--speakers` until the separate live call (T-LIVE-7).
- **Limitations:** threshold tuned on the same clips it is scored on (no held-out data); B has only 4 clips, all far from the phone; "agreement" maps each tracker speaker to its majority label, so it hides fragmentation (use "speakers created" too); same- and different-speaker similarities overlap (Meet-processed audio lowers same-speaker similarity), so two similar voices may merge at 0.30; the A -> B -> A check was run at 0.45, not yet at 0.30.
- **Commit:** labels 5454b11; threshold change in this commit.

---

## Open defects and paused work
- **Mic frame shortfall - PAUSED, UNRESOLVED.** Mic delivered ~7-16 % fewer frames than expected in every test from 2026-10-30 21:13 to 22:15; on 2026-10-31 17:23 (T-LIVE-5) it delivered 0.992 with the same settings. Time-varying, consistent with H3 (environment) but cause unknown. Not the sole cause: model load (T-LIVE-2), pipeline processing (T-REC-2). Ready but not run: callback-log analysis (eafc972), T-REC-3 (recorder with Meet closed). Mitigation flags (D10) untested.
- **Zero insertion mid-speech (found 2026-10-31).** `00053.76_Me.wav` contains 391 ms of exact digital zeros at 0.69 s: the aligner's re-anchoring after an unrecovered-lag event (D7) inserts the whole correction as one hole at detection time, here inside a word. Proposed fix (not made): correct timestamps by an offset instead of inserting audio. Remote clips' exact-zero runs (0.1-0.17 s) are Meet's own digital silence.

## Proposed, not tested
- **NLLB-200 distilled 1.3B (int8)** for the place-name errors seen in E2 («Երեւանում» dropped, «Վանաձոր» -> "Vancouver"). Deferred by user decision (2026-10-31). No measurement exists; nothing about its quality or speed on this laptop is claimed.
- **More call references** (10 unconfirmed Remote rows) and a CTC vs RNN-T comparison on them. Deferred.
- **Own-voice (Western-accented) ASR test.** Not run (D9).
