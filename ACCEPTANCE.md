# Final acceptance checklist

Derived item by item from the brief (`Mathrix_AI_Coding_Challenge.docx`).
Status: **PASS** = met, with evidence on the target laptop (Windows); **FAIL** =
not met yet; **UNTESTED** = not yet checked the way the interview will check it.
Evidence IDs refer to TESTS.md. State as of 2026-10-31 19:40, commit 3c91ec8.

Two kinds of evidence are kept apart:
- **Real call** - Google Meet, real people speaking Armenian on the remote side.
- **Playback** - recordings played into the system audio (T-SPK-2). Useful, not a real call.

## A. Required features

| # | Requirement (brief) | Real call | Playback | Evidence |
|---|---|---|---|---|
| A1 | Capture microphone and system audio at the same time | **PASS** | PASS | T-M0-1 (both voices, overlap captured, 0 overflows); every T-LIVE run |
| A2 | User still hears the call (macOS: Multi-Output Device; "on other systems, the equivalent") | **PASS** | - | WASAPI loopback records the Buds output while the user hears it (T-M0-1, D1) |
| A3 | User picks input devices (printed list or flags) | **PASS** | PASS | `--list-devices`; `--mic N --loopback M` required, wrong type rejected (devices.py) |
| A4 | Spoken Armenian -> Armenian text | **PASS** (works); quality below | PASS | Every T-LIVE run; quality in section D |
| A5 | Each transcribed segment translated into English | **PASS**, with a deliberate exception | PASS | Every line; text with < 2 Armenian letters is shown untranslated (D12, T-OFF-1) |
| A6 | Microphone labelled "Me" | **PASS** | - | All mic phrases "Me" in T-LIVE-4/5/6 (no mic speech in T-SPK-2) |
| A7 | System-audio speakers told apart as Speaker 1, 2, ... | **UNTESTED live** | **PASS** | Playback T-SPK-2: 11/11, 2 speakers. Real-call audio offline T-SPK-1: 92.3 % (tuned on the same clips). Never run live in a real call |
| A8 | Same person keeps the same label for the whole session | **UNTESTED live** | **PASS** | T-SPK-2 A->B->A = 1,2,1 and no extra speakers; T-SPK-1 offline 2 speakers at 0.30 |
| A9 | Live output: time, speaker, Armenian, English per segment | **PASS** with "Me/Remote"; **UNTESTED** with Speaker N | PASS | T-LIVE-5/6 output; T-SPK-2 output |

## B. Constraints

| # | Requirement | Real call | Playback | Evidence |
|---|---|---|---|---|
| B1 | <= 5 s from end of phrase to English line | **PASS** without speaker step (T-LIVE-5 0/34, T-LIVE-6 0/44 over 5 s, max 1.75 s); **UNTESTED** with speaker step | **PASS** (0/11, max 1.94 s, speaker step ~0.11 s) | Latency measured per line from the end of speech. Caveat: "Me" timings depend on the mic stream, which has an open frame-shortfall issue |
| B2 | Local models only (ASR, MT, speakers) | **PASS** by code and configuration; **UNTESTED** with the network off | - | All models are local ONNX/CTranslate2 files (models.py, pinned); no cloud API in the code. Downloads only during setup |
| B3 | Python | **PASS** | - | Python 3.13, `requirements.txt` pinned |
| B4 | Runs on the laptop used in the interview | **PASS** so far | - | All tests on the target Windows laptop; install from scratch not yet re-tested (E1 in section E) |
| B5 | Models chosen deliberately, explained in the README | **FAIL** (README missing) | - | Reasons exist in DECISIONS.md D3, D4, D14, D20 and TESTS E1-E3, T-SPK-1 |

## C. Deliverables

| # | Requirement | Status | Evidence |
|---|---|---|---|
| C1 | Git repository link | **FAIL** | Local git only, no remote yet |
| C2 | Source code | **PASS** | `livetrans/` (audio, segmenter, asr, translate, speakers, pipeline), `tools/` |
| C3 | README: setup and run, incl. audio device setup | **FAIL** | Missing |
| C4 | README: models chosen, alternatives, why | **FAIL** | Missing (content exists in DECISIONS/TESTS) |
| C5 | README: how the pieces fit together | **FAIL** | Missing (diagram in DECISIONS.md) |
| C6 | README: known limitations, what to improve | **FAIL** | Missing (open defects in TESTS.md) |
| C7 | `chats/`: every AI session, full and unedited, one `.md` per session | **FAIL** | Not exported yet - only the user can export from the app |
| C8 | No model weights, audio recordings or API keys committed | **PASS** | 48 tracked files, 0.46 MB total, none > 200 KB; no audio/model extensions in the current tree or in history; no key-like strings (checked 2026-10-31 19:35). `results/*.jsonl` and `eval/*.jsonl` contain transcribed text only |

## D. What the interviewers will look at live ("How we evaluate" - Live result)

| # | Question | Status | Evidence / numbers |
|---|---|---|---|
| D1 | Does it run? | **PASS** | All live runs since T-LIVE-4 without processing failures |
| D2 | Is the Armenian recognised? | **PASS, with errors** | FLEURS read speech WER 11.9 % (E3, 100 clips); user-confirmed call clips 15.0 % (9 errors / 60 words, 14 clips: T-LIVE-4 refs 23.8 %, T-LIVE-6 refs 10.3 %). Quiet/distant speakers and Meet's noise suppression make it worse (T-INSP-1) |
| D3 | Is the English understandable? | **PASS, with errors** | Correct Armenian -> mostly correct English (E2); place names fail («Վանաձոր» -> "Vancouver"); wrong Armenian -> wrong English |
| D4 | Under 5 seconds? | see B1 | |
| D5 | Speakers kept apart (>= 2 on the call side)? | **UNTESTED in a real call** | see A7, A8 |

## E. Pre-interview checks

| # | Check | Status |
|---|---|---|
| E1 | Fresh setup from the README in a new folder (clone, venv, install, download models, `--verify`) | UNTESTED |
| E2 | `python -m livetrans.models --verify` prints ALL OK in Windows | UNTESTED (never reported) |
| E3 | Real two-person call acceptance run (section F) | UNTESTED |
| E4 | Chat history still available in the app (interviewers open the originals) | user to keep |

## F. Real-call acceptance run - pass criteria (fixed before running)

Setup: Google Meet, **two** remote people speaking Armenian, the user on the
laptop mic, Buds as output, ~3-5 minutes; default program settings
(`--mic 14 --loopback 16 --save-audio`; speaker labels on, threshold 0.30,
beam 1, CTC).

| # | Criterion | Pass if |
|---|---|---|
| F1 | Every line shows time, label, HY, EN | all lines (EN may be absent only for < 2-letter text, D12) |
| F2 | Mic phrases labelled "Me" | 100 % |
| F3 | Remote speakers created | equal to the number of people who spoke on the remote side |
| F4 | Remote label agreement (user labels, clips >= 1.0 s, each Speaker N mapped to its majority person) | >= 85 % |
| F5 | A -> B -> A in the conversation | labels follow 1 -> 2 -> 1 |
| F6 | Latency | 0 lines over 5 s; report p50 / p95 / max and the speaker step |
| F7 | Failures | no processing failures; capture problems reported, not hidden |

Known limitations that may show up and are not acceptance failures by
themselves (to be documented in the README): a phrase containing two voices
gets one label; very short replies (< 1 s) inherit the previous remote label;
the user's own voice echoed back through a remote phone is labelled as a
remote speaker (echo handling is out of scope); quiet or distant remote speech
can be silenced by Meet before it reaches the laptop; the mic frame shortfall
(open defect) can affect "Me" lines.
