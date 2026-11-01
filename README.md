# Live Armenian transcription and translation

Listens to a call, transcribes spoken Armenian and prints an English
translation in the terminal, labelled by speaker:

```
[00:01:12] Speaker 1
  HY: Բարև, ինչպե՞ս ես։
  EN: Hi, how are you?
  .. latency 1.5s = vad 0.6 + queue 0.0 + spk 0.1 + asr 0.3 + mt 0.5  (2.1s audio)
[00:01:15] Me
  HY: Լավ եմ, շնորհակալություն։
  EN: I'm fine, thank you.
```

- The microphone is labelled **Me**.
- Voices on the system-audio side (the other end of the call) are labelled
  **Speaker 1, Speaker 2, ...** and keep their label for the whole session.
- Every model runs locally on the CPU. The network is only used once, during
  setup, to download the models.

Built and tested on **Windows 11**, Intel i5-13500H, 8 GB RAM, no GPU,
Python 3.13. System-audio capture uses WASAPI loopback, so the program runs on
Windows only (see Limitations).

## 1. Setup

Requirements: Windows 10/11, Python 3.13, about 1.1 GB of disk for the models.

```powershell
git clone https://github.com/sarkiskalikian/live-armenian-translation.git
cd live-armenian-translation
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# download the five models (pinned revisions) into models/ - once, needs internet
.\.venv\Scripts\python.exe -m livetrans.models
# check file hashes, revisions and package versions
.\.venv\Scripts\python.exe -m livetrans.models --verify     # prints ALL OK
```

`models/`, `logs/` and `recordings/` are git-ignored. No model weights or
audio are in the repository.

## 2. Audio devices

Windows offers a **loopback** twin of every output device: it records what
that output plays. That covers the brief's "equivalent" of BlackHole plus a
Multi-Output Device on macOS. You keep hearing the call normally and no
virtual cable is needed (PyAudioWPatch provides WASAPI loopback).

1. Choose the headphones or speakers that the call app plays to, and make that
   device the output in the call app (or the Windows default output).
2. List the devices:
   ```powershell
   .\.venv\Scripts\python.exe -m livetrans --list-devices
   ```
   Note the index of your **microphone** and the index of the **[Loopback]**
   entry with the same name as your output device.
3. Use the laptop's built-in microphone, not a Bluetooth headset microphone.
   Opening a Bluetooth headset's mic switches it to the hands-free profile,
   which moves the call audio to a different output and the loopback goes
   silent (found in test T-M0-1).

The program refuses a loopback index given as `--mic` and the other way
round. It never picks a device silently.

## 3. Run

```powershell
.\.venv\Scripts\python.exe -m livetrans --mic 14 --loopback 16
```

(14 and 16 are the indices on the development laptop; use yours.) Start-up
takes about 10 s while the models load and warm up. Then talk. **Ctrl+C**
stops the program and prints a summary: latency per source, capture problems
and the number of remote speakers. A text log of each session goes to
`logs/<timestamp>/` (git-ignored).

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--no-speakers` | speakers on | label every remote phrase "Remote" instead of Speaker N |
| `--spk-threshold` | 0.30 | cosine similarity needed to match an existing speaker |
| `--min-silence` | 0.5 | seconds of silence that end a phrase |
| `--max-speech` | 10 | force a cut after this many seconds of speech |
| `--beam` | 1 | translation beam size |
| `--save-audio` | off | also keep each phrase as a WAV in the log folder |

## 4. How it works

```
mic ─────┐                                                          ┌─ "Me"
         ├─ WASAPI capture ─ 16 kHz mono ─ Silero VAD ─ phrase ─┬───┤
loopback ┘  (PyAudioWPatch)    (soxr)      (per source)          │   └─ WeSpeaker embedding ─ online clustering ─ "Speaker N"
                                                                  ▼
                                   FastConformer-hy CTC (ONNX) ─ Armenian text ─ NLLB-200 600M int8 ─ English
                                                                  ▼
                                         terminal: [time] label / HY / EN / latency
```

| Module | Does |
|---|---|
| `livetrans/audio/` | `devices.py` lists and checks devices. `capture.py` runs one callback stream per device; the callback only time-stamps the buffer. `ChunkAligner` puts both sources on one shared clock. `convert.py` resamples to 16 kHz mono. |
| `segmenter.py` | Silero VAD (ONNX), frame by frame. 0.5 s of silence ends a phrase; 10 s forces a cut. |
| `speakers.py` | WeSpeaker ResNet34 embedding per remote phrase, then online clustering (below). |
| `asr.py` | FastConformer-Hybrid Large (Armenian), CTC head, through `onnx-asr`. |
| `translate.py` | NLLB-200 distilled 600M, int8, CTranslate2, `hye_Armn -> eng_Latn`. |
| `pipeline.py` | One worker thread takes phrases from a FIFO queue: speaker, ASR, MT, print. Measures latency from the end of speech and splits it into vad / queue / spk / asr / mt. |
| `models.py` | Pinned model registry (repo, revision, sha256); download and `--verify`. |
| `__main__.py` | Command line, start-up, live warnings, session summary. |

Design choices:

- **One phrase at a time.** Each model runs once per phrase, on whole
  phrases. ASR on a 2-5 s phrase takes about 0.1-0.3 s on this CPU, so
  streaming recognition isn't needed to stay under 5 s. One worker keeps the
  models from competing for CPU.
- **Speaker rules**, borrowed from diart (Coria et al.) without its
  segmentation model. Mic phrases are always "Me". For each remote phrase:
  - 1.0 s or longer: it is embedded and compared with each speaker's centroid.
    At similarity 0.30 or above it gets the best-matching label.
  - Below 0.30: a new speaker is created only if the phrase is 1.5 s or longer.
    Otherwise it gets the closest label.
  - Centroids are only updated from phrases of 1.5 s or longer.
  - Phrases shorter than 1.0 s keep the previous remote label.
- **Too little text to translate.** If recognition returns fewer than two
  Armenian letters, the line is shown without a translation. NLLB invents
  whole sentences for one-letter input such as «Հ։» (T-OFF-1).

The development record is in `PLAN.md`, `DECISIONS.md` (decisions D1-D20)
and `TESTS.md` (every experiment: question, environment, configuration,
result, decision, limitations, commit). `ACCEPTANCE.md` checks the brief item
by item. `tools/` holds the measurement scripts and `results/` their outputs.

## 5. Models: what I chose and why

Armenian is low-resource, so each choice was checked against public
benchmarks and then measured on this laptop. All numbers below are from the
laptop (Windows) unless marked otherwise.

| Task | Chosen | Size | Licence |
|---|---|---|---|
| Speech recognition | NVIDIA `stt_hy_fastconformer_hybrid_large_pc`, CTC head, ONNX export by OpenVoiceOS | 115M params, 437 MB | CC-BY-4.0 |
| Translation | Meta `nllb-200-distilled-600M`, int8 CTranslate2 conversion | 618 MB | CC-BY-NC-4.0 (non-commercial) |
| Speakers | WeSpeaker VoxCeleb ResNet34-LM, ONNX | 26 MB | CC-BY-4.0 |
| Phrase detection | Silero VAD, ONNX | 2 MB | MIT |

All revisions and file hashes are pinned in `livetrans/models.py`.

### Speech recognition: FastConformer-Hybrid (hy), CTC

- **Why:** it is the best open model on the independent ArmBench-ASR v1.0
  leaderboard (combined WER 19.9; Whisper-large-v3 zero-shot 65.1). It is
  small enough for a CPU, and the ONNX export runs without NeMo or PyTorch.
- **Measured:**
  - 11.9 % WER (CER 5.0 %) on 100 FLEURS read-speech clips, 0.15 s per
    11.7 s clip (E3).
  - 15.0 % WER on 14 user-checked phrases from real Google Meet calls
    (9 errors in 60 words).
- **Alternatives considered:**
  - **The same weights with the RNN-T head** (Vana-Labs ONNX export), which
    Karpov et al. (Interspeech 2025) motivated testing. 11.4 % vs 11.9 % WER
    on the same 100 clips: the difference is not significant (95 % CI
    -1.3 to +0.3 points), and RNN-T is about 5x slower. **CTC kept** (E3).
  - **Whisper and its Armenian fine-tunes:** ranked below FastConformer on
    ArmBench and 7-13x larger. Not run.
  - **Meta MMS / omnilingual ASR:** lower on ArmBench. Not run.
- **Risk:** ArmBench reports about 61 % WER on movie dialogue for this model.
  Conversation is far harder than read speech; see Limitations.

### Translation: NLLB-200 distilled 600M, int8, beam 1

- **Why:** the strongest open translation model that covers Armenian and is
  small enough to run in well under a second on a CPU with CTranslate2 int8.
- **Measured:**
  - FLEURS chrF 61.1 with beam 1 and 61.7 with beam 4 (T-MT-1).
  - On 5 call phrases, beams 1 and 4 gave the same meaning; median 204 ms vs
    256 ms. **Beam 1 chosen** (E2).
- **Alternatives:**
  - **NLLB-200 1.3B:** might fix the place-name errors, but is about twice
    the size. Proposed, not tested.
  - **Large LLM translators:** too slow on this CPU for a 5 s budget. Not
    tested.

### Speakers: WeSpeaker ResNet34 + online clustering

- **Why:** a 26 MB speaker-embedding model takes about 0.1 s per phrase on
  the CPU, and it loads through `onnx-asr`, which is already installed.
  Clustering with fixed rules gives stable labels without a second model.
- **Measured:**
  - Recorded voices played into the system audio (never used for tuning):
    11/11 phrases labelled correctly, A -> B -> A gave Speaker 1 -> 2 -> 1,
    speaker step about 0.11 s (T-SPK-2).
  - Real Meet clips: 92.3 % agreement with my labels. The threshold was tuned
    on these same clips (T-SPK-1).
- **Alternatives:**
  - **pyannote 3.x pipeline, diart:** need PyTorch and more CPU. They also
    re-segment audio that VAD has already cut.
  - **CAM++ and other embedding models:** not tested.

### Phrase detection: Silero VAD

Small and fast, runs on 16 kHz frames, and is widely used. Phrase cutting was
checked on the first two-voice recording (T-VAD-1).

## 6. Results on the target laptop

| Run | What | Outcome |
|---|---|---|
| T-LIVE-8, 2026-11-01 | Real Google Meet call: me on the laptop mic, two people on the remote side, all defaults | 45 phrases (27 Me, 18 remote), 2 remote speakers created; **0 over 5 s**, p50 1.67 s (Me) and 1.51 s (remote), max 2.37 s; speaker step mean 0.07 s; no processing failures. I judged the Me / Speaker 1 / Speaker 2 labels correct; per-phrase labels were not scored |
| T-LIVE-5 / T-LIVE-6 | Real calls, before speaker labels | 0/34 and 0/44 phrases over 5 s |
| T-SPK-2 | Recorded voices played into the loopback | 11/11 labels, max latency 1.94 s |
| E3 | ASR on 100 FLEURS clips | 11.9 % WER |
| Call clips | 14 user-checked call phrases | 15.0 % WER |

Latency is measured from the end of the spoken phrase (on the audio clock) to
the moment the English line is printed. About 0.5-0.7 s of it is the silence
the VAD waits for before it ends a phrase.

## 7. Known limitations

- **Recognition on real calls is the weak part.** Read speech is about 12 %
  WER. On calls, accents, distance from the mic, Bluetooth mics and Meet's
  noise suppression make it noticeably worse. Wrong Armenian gives wrong English.
- **My accent.** My native Armenian is Western Armenian, while the recognition
  model and the test sets I measured on (FLEURS, ArmBench) are Eastern Armenian.
  During my tests my Western-influenced pronunciation caused extra recognition
  errors. I did not measure how much.
- **Hardware.** Everything was built and measured on one mid-range laptop:
  i5-13500H, 8 GB RAM, no usable GPU. To fit the 5 s budget on this CPU, with
  the call app running at the same time, I chose small CPU models (FastConformer
  115M, NLLB 600M int8) over larger ones that might be more accurate (NLLB 1.3B,
  Whisper large). The latency numbers apply to this laptop only.
- **Time.** The brief expects about 20 hours of work, so testing is limited: a few real
  calls, two remote voices and 14 checked call phrases for WER. Several
  alternatives were chosen from public benchmarks and not run myself.
- **Meet removes quiet speech.** Meet replaces quiet or distant speech with
  digital silence before it reaches the laptop, so that speech never arrives
  (T-INSP-1).
- **Translation errors:** place names («Վանաձոր» became "Vancouver"). Very
  short replies are sometimes over-translated («Այո» became "I did.").
  Phrases under two Armenian letters are shown untranslated.
- **Speaker labels.** In the real call test (T-LIVE-8) the two remote people
  were told apart correctly as Speaker 1 and Speaker 2. The limits are in cases
  that test did not cover:
  - One label per phrase: if two people talk inside one phrase, both get the
    same label. The two speakers were still told apart over the call, but in
    my tests the Armenian recognition made more mistakes on such mixed phrases.
  - Phrases under 1 s keep the previous remote label.
  - The threshold was tuned on little real data (two remote voices).
  - Two similar voices are harder to separate than the voices tested.
  - Early in a session each speaker has few phrases behind its label, so a new voice is more likely to be mislabelled then (expected from the clustering design, not measured).
- **Echo** (my voice coming back through a remote device) shows up as a remote
  phrase. Echo handling is out of scope in the brief.
- **Microphone frame shortfall (open, cause unknown).** On this laptop's mic,
  the stream sometimes delivers fewer samples than expected: 95.8-99.2 % in
  the latest runs, as low as 84 % on the first day. It happens with the
  models off too. The aligner re-anchors the timeline with zeros, which can
  land inside a word.
- **Windows only.** Capture uses WASAPI loopback. macOS would need BlackHole
  plus a different capture module.
- **Mic and system audio share one processing queue.** If many phrases end
  at the same time they wait in line. The queue never grew beyond one phrase
  in the tests.

## 8. What I would improve with more time

1. Test recognition on more real call audio with a wider range of accents,
   and compare the RNN-T head and fine-tuned Whisper models there.
2. Try NLLB-200 1.3B for the place-name and short-reply errors, if it stays
   within the 5 s budget.
3. Improve speaker labels: overlap-aware segmentation for phrases with two
   voices, and more labelled calls to set the threshold.
