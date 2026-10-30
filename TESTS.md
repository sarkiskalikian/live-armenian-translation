# Test log

**Terminology (revised 2026-10-30 23:50).** *Frame shortfall*: a stream delivered
fewer frames than wall time x nominal rate (measured from frame counts or
derived from timing). *Unrecovered-lag event*: lateness > 0.3 s that did not
recover within 1 s (measured by the aligner). Neither proves that the device
dropped audio - a slow device clock or frames dropped elsewhere would look the
same. Earlier wording ("lost", "loss events") has been changed accordingly.

Every entry separates **measured** facts (read from logs/recordings) from
**inferred** conclusions and **hypotheses**. Audio, logs and models are not in
the repo; the log folder names are given so results can be re-checked locally.

Machine: Windows 11, Intel i5-13500H (12 physical / 16 logical cores), 7.6 GB
usable RAM, no GPU. Python 3.13 in `.venv`. Devices: mic `[14]` Intel Smart
Sound microphone array (2 ch, 48 kHz); loopback `[16]` Galaxy Buds3 Pro
"Headphones" endpoint (2 ch, 48 kHz). Call app: Google Meet in a browser.

---

## T-M0-1 - Capture both sources (2026-10-30 20:18)
`recordings/20261030-201834`, `tools/m0_record.py --guided`, `tools/m0_check.py`

Measured:
- No mic dropouts, 0 input overflows, max queue 1, no clipping (both streams).
- Mic: 2,854,080 frames, file length 60.01 s = wall time 60.01 s.
- Loopback: 2,857,920 frames, file 60.11 s vs wall 60.01 s (+0.10 s).
- Separation: during remote-only speech (21-30 s) the mic stayed at its noise
  floor (about -75 dB); during overlap (33-43 s) both streams carried speech at
  -16 to -22 dB; the loopback was never idle (Meet keeps playing ~-99 dB).

Inferred: capture of both sources works; no leakage with the Buds.
The +0.10 s was later traced to the aligner anchoring a stream to its first
buffer (see D7); fixed in code, not yet re-measured on the device.

## T-VAD-1 - Phrase segmentation on the M0 recording (offline)
`tools/try_file.py --vad-only` with Silero 6.2.3 (PyPI copy), run in a Linux
sandbox on the same laptop.

Measured: mic -> 9 phrases, all inside the stretches where the user spoke
(5-17 s, 33-45 s); loopback -> 6 phrases inside 21-30 s and 33-43 s; no
phrases in silence; none of the remote voice in the mic stream.

## T-STUB - Logic tests without hardware (cloud workspace)
Synthetic buffers through aligner -> converter -> segmenter -> worker with
stub models. Verified: phrase end times within 20 ms; queue wait accounted
when sources overlap; late first buffer + burst no longer shifts a stream;
no zeros inserted into speech for late buffers; idle loopback gap filled;
a simulated 7 % buffer drop is reported as unrecovered lag (6.4 of 6.7 s) with delivered ratio
0.929. These are logic tests only, not evidence about the real device.

## T-MT-1 - Translation benchmark (2026-10-30 ~20:44, partial M1)
`results/m1_mt_summary.json` - 100 FLEURS hy_am test sentences, **clean**
(reference) Armenian input, NLLB-200-distilled-600M int8, 4 threads.

| beam | chrF | BLEU | ms p50 | ms p95 | ms max |
|---|---|---|---|---|---|
| 1 | 61.08 | 35.38 | 1307 | 2486 | 3637 |
| 4 | 61.74 | 36.29 | 1674 | 3543 | 5937 |

Peak RAM 996 MB. Inferred: beam 1 costs ~0.7 chrF and is ~25-40 % faster;
long FLEURS sentences make MT the slowest stage. The ASR benchmark and the
own-voice reading test were **not run** (skipped, see D9).

## T-LIVE-0 - First live run (2026-10-30 21:04)
`logs/20261030-210400`, default settings (beam 4, ASR threads = runtime default,
20 ms capture buffer). Only the mic produced speech (4 phrases, 4.7 s audio).

Measured: mic VAD-wait component 0.56-0.60 s across 27-85 s (no growth;
linear fit slope 0.0007 s/s, i.e. sample clock at 99.9 % of wall time).
Latency 2.2-3.2 s, all under 5 s. ASR mean 1.18 s, MT mean 1.10 s.

## T-LIVE-1 - Two-person Meet call (2026-10-30 21:13) - FAILED (mic timing)
`logs/20261030-211345`, same code and settings as T-LIVE-0. User report:
"when we talked a bit faster it did not catch all the words".

Measured (per-phrase log, 14 mic + 11 remote phrases, 48.5 s of speech):
- For each phrase the log has `vad_s` = (wall time the phrase was queued) -
  (t0 + phrase end computed from the **sample count**). For a healthy stream
  this is the 0.5 s end-of-phrase silence wait.
- Remote: `vad_s` 0.50-0.55 s throughout (fit slope -0.0002 s/s).
- Me: `vad_s` grew from 0.55 s (phrase ending at 4.0 s) to 7.07 s (phrase
  ending at 91.5 s); linear fit `vad_s = -0.07 + 0.0769 t`.
- The capture code logged 4315 buffers arriving > 0.3 s later than their
  sample position predicted; max lateness 6.93 s. 0 input-overflow flags.
- Latency: Me p50 4.62 s, p95 8.18 s, 6 of 14 over 5 s; Remote p50 1.85 s,
  max 2.08 s, 0 over 5 s. ASR mean 0.44 s, MT mean 0.77 s, queue wait 0.
- Raw sample counts per stream were **not** logged in this version, so the
  ratio below is derived from timing, not counted.

Derived: the mic's sample clock advanced at 1/(1+0.0769) = 92.9 % of wall
time, i.e. a shortfall of ~7 s of samples after ~90 s. Phrase timestamps for "Me"
were early by that amount (e.g. a reply stamped 74.0 s, before the remote
sentence it answers, was really spoken around 79.8 s).

Hypothesis (not proven): the missing samples were never delivered,
which matches the user's missed words. Alternatives not yet excluded: the
device running at a lower real sample rate under some condition, or samples
lost inside PortAudio without an overflow flag. T-LIVE-0 and T-M0-1 show the
same mic delivering ~100 % of samples, so a fixed rate mismatch is unlikely.

Hypotheses (all **unconfirmed**):
- H1 CPU starvation: ONNX Runtime (all cores, busy-waiting) and CTranslate2
  (OpenMP) delay the mic's audio thread until WASAPI data is lost. Weak
  support: T-LIVE-0 (4.7 s of speech processed) did not drift, T-LIVE-1
  (48.5 s processed) did. Against: the loopback, under the same load, did
  not drift.
- H2 Python GIL contention delaying the PyAudio callback.
- H3 Intel Smart Sound mic driver behaviour while Meet uses the same mic.

Accuracy notes (same log): when the Armenian was right the English was right
("Լսո՞ւմ ես։" -> "Do you hear me?"; "Որտե՞ղ եք ապրում։" -> "Where do you
live?"). Bad lines started with bad recognition ("Հեռեւ ձեզ, ո՞նցեքք" for
"Բարև ձեզ, ո՞նց եք" -> "Away from you"; a lone "֊։" -> invented "What is it?").

## T-LIVE-2 - Models off, T-LIVE-1 settings (2026-10-30 21:54) - mic frame shortfall REPRODUCED
`logs/20261030-215404`, commit b72c767, `--no-models` (capture + resampling +
Silero VAD only; no ASR/MT loaded), capture buffer 20 ms, same Meet call setup.
Planned 2 min; the run lasted **15.6 s** (user stopped it), so ratios are coarse.

Measured (`summary.json` -> `capture`; sample counts are now logged):
- Me: delivered_ratio 0.840 (frames received / frames expected since the
  stream's first sample); 3 unrecovered-lag events totalling 1.57 s (live
  warnings at ~10 s and ~12 s, about 0.5 s each); 6 late-but-recovered events;
  max lateness 0.556 s; 0 overflow flags.
- Remote: delivered_ratio 0.993; 0 unrecovered-lag events; max lateness 0.003 s.
- 4 phrases (3 Me, 1 Remote), latency 0.54-0.96 s (VAD wait only).

Inferred:
- The mic shows the frame shortfall **without** any ASR/MT CPU load, so H1 (model CPU
  starvation) cannot be the sole cause. Whether model load makes it worse is
  unknown. The ASR/OMP mitigations from D10 are therefore not the first lever.
- ~~The shortfall comes in ~0.5 s blocks~~ **Corrected after T-REC-2:** the ~0.5 s
  events are an artefact of the lag detector (it confirms a deficit once
  lateness has exceeded 0.3 s for 1 s), so a steady shortfall also shows up
  as regular ~0.45 s events. The data do not show whether the shortfall is bursty.
- 1.57 s of unrecovered lag in ~14.7 s explains ~11 %; the rest of the 16 % gap is
  lateness still unconfirmed when the run stopped (lag is only logged as unrecovered
  after 1 s without recovery).

Contrast with T-M0-1: same mic, same Meet setup with remote speech, 20 ms
buffer, same CaptureStream class -> 100 % delivered over 60 s. Difference: the
recorder only writes WAV per buffer; the live program also resamples (soxr)
and runs Silero VAD per buffer in the drain thread. T-M0-1 ran with the older
aligner, but frame counts are independent of the aligner.

Hypotheses now (unconfirmed):
- H2 Python-side: work in the drain threads (GIL contention) delays the
  PyAudio callback of the mic until WASAPI drops data.
- H3 environment: Intel Smart Sound driver / Meet sharing the mic; a
  condition that differed from the T-M0-1 session (time-varying).
- H1 model CPU load: may still contribute; not the sole cause.

## T-REC-2 - Plain recorder in the same Meet call (2026-10-30 22:01) - mic frame shortfall REPRODUCED
`recordings/20261030-220126`, commit 7938c53, `tools/m0_record.py --seconds 120`,
20 ms buffer; no VAD, no resampling, no models (per buffer: write WAV only).
Both people talked (mic active 36 %, loopback 31 % of 50 ms frames).
User impression after the run: "no problem in capturing the sounds".

Measured (raw frame counts from PortAudio, independent of the aligner):
- Mic: 4,985,280 frames = 103.86 s over 118.91 s since its first sample ->
  **0.873** delivered; 32 unrecovered-lag events (14.70 s) at a regular ~3 s spacing,
  each 0.35-0.50 s, in speech and in silence alike; 41 late-but-recovered;
  0 overflow flags.
- Loopback: 5,718,720 frames = 119.14 s over 118.87 s -> 1.002; no shortfall.

Inferred:
- The plain recorder shows a mic frame shortfall of the same order as the
  live program, so neither the models (H1) nor the live pipeline's per-buffer
  work (H2) is necessary for the shortfall.
- The regular spacing matches a steady deficit of ~13 % seen through the
  detector (0.127 s/s reaches the 0.3 s threshold, is confirmed 1 s later at
  ~0.43 s, then resets), see the correction under T-LIVE-2.
- The user's listening impression and the frame counts disagree; the frame
  counts are taken as the evidence. Not yet known: whether the shortfall is
  dropped packets (missing audio) or the device delivering at a lower actual
  rate (time-stretched audio) - per-buffer timing is needed to tell.

Mic delivery over time (same device, same 20 ms buffer):

| Test | Time | Mic delivered | Notes |
|---|---|---|---|
| T-M0-1 | 20:18 | 1.000 | recorder, Meet call |
| T-LIVE-0 | 21:04 | 0.999 (derived) | live, models on, mic only |
| T-LIVE-1 | 21:13 | 0.929 (derived) | live, models on, both talking |
| T-LIVE-2 | 21:54 | 0.840 (15 s run) | live, models off |
| T-REC-2 | 22:01 | 0.873 | recorder, Meet call |

Leading hypothesis now: **H3 environment** - a condition that started between
21:04 and 21:13 and persisted (candidates: Meet's use of the same mic, e.g. its
audio processing; Bluetooth activity; power/thermal state). Unconfirmed.

## T-LIVE-4 - Armenian conversation, models on, T-LIVE-1 settings (2026-10-30 22:15)
`logs/20261030-221541` (segment WAVs saved), commit eafc972 code, beam 4, 20 ms
buffer. Two-person Meet call, 124 s, 36 phrases (22 Me, 14 Remote).
User's judgement: "the English translation was correct but understanding the
Armenian was not that much correct".

Measured:
- Latency: Me p50 1.97 s, p95 5.14 s, max 5.32 s, 2 over 5 s; Remote p50
  1.70 s, p95 2.50 s, max 2.53 s, 0 over 5 s. Means: VAD wait 0.6-0.7, queue
  0.14-0.21, ASR 0.34-0.49, MT 0.70-0.80 s.
- Both >5 s lines are the first two phrases: phrase 1 ASR 1.8 s + MT 2.9 s
  (cold start despite the warm-up call), phrase 2 then queued 3.4 s.
- Capture: Me delivered 0.909, 10.36 s of unrecovered lag in 25 events (known issue);
  Remote 0.997, no shortfall.

Quality, assistant's reading of the 36 lines (approximate, to be confirmed
by the user, who is a native speaker):
- ~11 both correct or close ("Ես Հայաստանում եմ։" -> "I am in Armenia.").
- ~13 recognition errors that carry into English ("Պատելի" for "Հաճելի";
  "քինիկական" for "կլինիկական" -> "kinetic research"; "Կեանի՞ ես ... կանիսն ա"
  for "ժամը քանիսն ա").
- ~5 correct Armenian, wrong English ("Ես Վանաձորից եմ" -> "I'm from
  Vancouver"; "Հաջողություն" -> "How can we succeed?"; "Ո՞նց եք" -> "Who are
  you?"; "Երեւանո՞ւմ" -> "Where are they going?"); also the name "Գարուն"
  translated as "Spring".
- ~7 one-letter or near-empty recognitions (backchannels/laughter) that MT
  turns into invented sentences ("Հ։" -> "A: What are you talking about?"
  twice; "Ս։" -> "Q. What is this?"; "Հովհ." -> "John: How did you know?").
  The existing filter only skips output with no Armenian letter at all.

Inferred: recognition is the largest error source; MT adds errors on short
colloquial phrases and names; one-letter outputs are a separate, cheap-to-fix
defect. "Me" errors may be partly caused by the mic frame shortfall.

## T-OFF-1 - Minimum-letters rule replayed on T-LIVE-4 output (offline)
Change: translate only if the recognised text has >= 2 Armenian alphabet
letters (Ա-Ֆ, ա-և; punctuation/spaces not counted); otherwise print the
Armenian with "(not translated ...)". Previously only text with 0 letters
was skipped.

Measured on the 36 recognised strings of `logs/20261030-221541`:
- 3 lines change: "Հ։" (Remote, twice) and "Ս։" (Me), previously translated
  as "A: What are you talking about?" / "Q. What is this?".
- 33 lines unchanged. Still translated and still wrong: "Հովհ." (4 letters)
  -> "John: How did you know?", "Հահ։" (3) -> "Oh, my God." A letter count
  cannot catch these. "Հ է։" (2) -> "That's right." is kept.
- Earlier estimate of "~7" hallucination lines fixed by this rule was an
  overstatement; the rule addresses 3.

Not yet observed in a live run.

## E1 - Warm-up: minimal vs representative (2026-10-31 ~00:37, laptop)
`results/warmup_check.json`, commit b834391, beam 4, default ASR threads.
3 fresh processes per mode, alternating; clips = T-LIVE-4 Remote phrases
00019.81, 00037.95, 00070.08 (3.1-4.4 s). Times are ASR / MT seconds, medians.

| warm-up | load | warm-up | clip1 first | clip1 again | clip2 | clip3 |
|---|---|---|---|---|---|---|
| minimal | 6.03 | 1.11 | 0.131 / 0.629 | 0.065 / 0.571 | 0.060 / 0.553 | 0.078 / 0.644 |
| representative | 8.88 | 3.16 | 0.183 / 0.689 | 0.058 / 0.506 | 0.067 / 0.624 | 0.107 / 0.547 |

One run of each mode (representative run 2, minimal run 3) was slow on
every step (ASR 0.45-2.06 s, MT 1.14-3.17 s), i.e. a machine-wide
slowdown during that process, not a first-phrase effect; medians absorb it.

Inferred:
- With the existing minimal warm-up, the first phrase in a fresh process
  costs ~0.07 s ASR and ~0.06 s MT more than the same phrase warm. The
  representative warm-up gives no measurable gain and adds ~2 s start-up.
  Decision: keep "minimal" (D15).
- The T-LIVE-4 first phrase (ASR 1.8 s + MT 2.9 s) is not explained by
  model cold start; it coincided with the start of capture during a Meet
  call. Cause unconfirmed.
- On this laptop MT (~0.55 s at beam 4) dominates per-phrase compute;
  ASR is ~0.06-0.13 s for 3-4 s clips.

## Call references for the 5 beam-check phrases (2026-10-31 00:59-01:11)
`eval/t-live-4_refs.jsonl`, corrected by the user (native speaker) by listening
to the saved T-LIVE-4 clips, one at a time; ambiguous words were asked about
before confirming. Word errors of the live CTC recognition (normalised):

| clip | reference (user) | live ASR | word errors |
|---|---|---|---|
| 00019.81_Remote | Հաճելի է, իմ անունը գարուն է։ | Պատելի է, ... | 1/6 |
| 00049.31_Remote | Ես Երեւանում, դու՞ւք։ | Այո Երեւանում, ... | 1/3 |
| 00053.76_Me | Ես Վանաձորից եմ։ | (same) | 0/3 (MT test only) |
| 00070.08_Remote | Ես զբաղվում եմ կլինիկական հետազոտություններ հետ։ | ... զբաղում ... քինիկական ... | 2/6 |
| 00095.46_Remote | Մեր մոտ ժամը տաս տասնհինգ անց։ | ... տ տասնհինգ անց | 1/6 |

00049.31 was first confirmed as recognised (01:02), then corrected by the
user at 01:13: the first word is «Ես», not «Այո». The other 10 Remote rows
are still unconfirmed. Whether «գարուն» is a name
(would be «Գարուն») was not settled; it affects MT input, not WER.

**Defect found while listening (00053.76_Me):** the user heard a cut. The clip
contains 391 ms of exact digital zeros at 0.69 s - silence inserted by our
aligner when it re-anchors the timeline after an unrecovered-lag event (D7),
placed mid-speech at detection time. Real mic audio never has exact-zero runs
(floor about -75 dB). The frames were missing anyway (frame shortfall), but
concentrating the correction into one 0.4 s hole inside a word is our doing
and can hurt recognition. Possible fix (not made): correct timestamps by an
offset instead of inserting audio. Remote clips also contain exact-zero runs
(0.1-0.17 s) in pauses: Meet's own digital silence, expected.

## Planned (2026-10-30 23:59) - run on the laptop, Windows .venv, pinned versions
- E0 `python -m livetrans.models --verify` - pinned model hashes and runtime versions.
- E1 done (see above).
- E2 `tools/beam_check.py` - beam 1 vs 4 on 5 phrases, corrected Armenian and ASR output.
- E3 `tools/asr_compare.py` - CTC fp32 vs RNN-T fp32 on FLEURS (100) and confirmed
  intact call phrases; E4 RNN-T int8 only if E3 favours RNN-T.
Sandbox runs of these scripts were functional checks only; their numbers are not results.

## Mic capture investigation - PAUSED, UNRESOLVED (2026-10-30 22:14)
Paused by user decision to resume work on recognition/translation output.
Status: the mic delivers ~7-16 % fewer frames than expected in every test since 21:13; cause
unknown. Rejected as *sole* causes: model CPU load (T-LIVE-2), live-pipeline
processing (T-REC-2). Leading but unconfirmed: environment (H3). Until fixed,
"Me" lines can miss words and their timestamps/latencies are less reliable;
"Remote" lines are unaffected (loopback ~100 % in all tests).

Ready but not run:
- Per-callback logging is in place (commit eafc972, `callbacks_<source>.csv`).
- T-REC-3: plain recorder with **Meet closed** (one variable). Shortfall gone ->
  Meet's use of the mic is the trigger; shortfall remains -> system state since 21:13.
- T-LIVE-3: models on, test-1 settings (deferred).
