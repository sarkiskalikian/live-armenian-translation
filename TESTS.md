# Test log

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
a 7 % buffer loss is reported as lost (6.4 of 6.7 s) with delivered ratio
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
time, i.e. ~7 s of samples missing after ~90 s. Phrase timestamps for "Me"
were early by that amount (e.g. a reply stamped 74.0 s, before the remote
sentence it answers, was really spoken around 79.8 s).

Inferred (not proven): the missing samples were never delivered (audio loss),
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

## T-LIVE-2 - Models off, T-LIVE-1 settings (2026-10-30 21:54) - mic loss REPRODUCED
`logs/20261030-215404`, commit b72c767, `--no-models` (capture + resampling +
Silero VAD only; no ASR/MT loaded), capture buffer 20 ms, same Meet call setup.
Planned 2 min; the run lasted **15.6 s** (user stopped it), so ratios are coarse.

Measured (`summary.json` -> `capture`; sample counts are now logged):
- Me: delivered_ratio 0.840 (frames received / frames expected since the
  stream's first sample); 3 confirmed loss events totalling 1.57 s (live
  warnings at ~10 s and ~12 s, about 0.5 s each); 6 late-but-recovered events;
  max lateness 0.556 s; 0 overflow flags.
- Remote: delivered_ratio 0.993; 0 loss events; max lateness 0.003 s.
- 4 phrases (3 Me, 1 Remote), latency 0.54-0.96 s (VAD wait only).

Inferred:
- The mic loses audio **without** any ASR/MT CPU load, so H1 (model CPU
  starvation) cannot be the sole cause. Whether model load makes it worse is
  unknown. The ASR/OMP mitigations from D10 are therefore not the first lever.
- The loss comes in ~0.5 s blocks rather than a steady trickle.
- 1.57 s confirmed lost of ~14.7 s explains ~11 %; the rest of the 16 % gap is
  lateness still unconfirmed when the run stopped (loss is only confirmed
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

## Pending
- T-REC-2: plain recorder (`tools/m0_record.py`) for 60-120 s under the same
  call conditions. Loss there -> H3 (environment/driver); no loss -> the live
  pipeline's per-buffer processing (H2).
- T-LIVE-3: models on, test-1 settings (originally planned; lower priority
  since T-LIVE-2).
