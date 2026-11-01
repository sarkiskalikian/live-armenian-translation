# Final acceptance checklist

Derived item by item from the brief (`Mathrix_AI_Coding_Challenge.docx`).
Status: **PASS** = met, with evidence on the target laptop (Windows); **FAIL** =
not met yet; **UNTESTED** = not yet checked the way the interview will check it.
Evidence IDs refer to TESTS.md. State as of 2026-11-01 (README commit). Updated after the real-call run T-LIVE-8.

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
| A6 | Microphone labelled "Me" | **PASS** | - | All mic phrases "Me" in T-LIVE-4/5/6/8 (27/27 in T-LIVE-8) |
| A7 | System-audio speakers told apart as Speaker 1, 2, ... | **PASS** (user judgement, not scored per phrase) | **PASS** | T-LIVE-8: 2 remote people -> Speaker 1 and Speaker 2, judged correct by the user. Playback T-SPK-2: 11/11. Offline T-SPK-1: 92.3 % (tuned on the same clips) |
| A8 | Same person keeps the same label for the whole session | **PASS** (user judgement) | **PASS** | T-LIVE-8: labels 1, 2, then 1 again, no extra speakers; T-SPK-2 A->B->A = 1,2,1 |
| A9 | Live output: time, speaker, Armenian, English per segment | **PASS** | PASS | T-LIVE-8 (Speaker N), T-LIVE-5/6, T-SPK-2 |

## B. Constraints

| # | Requirement | Real call | Playback | Evidence |
|---|---|---|---|---|
| B1 | <= 5 s from end of phrase to English line | **PASS** (T-LIVE-8 with speaker step: 0/45 over 5 s, max 2.37 s; T-LIVE-5 0/34, T-LIVE-6 0/44) | **PASS** (0/11, max 1.94 s, speaker step ~0.11 s) | Latency measured per line from the end of speech. Caveat: "Me" timings depend on the mic stream, which has an open frame-shortfall issue |
| B2 | Local models only (ASR, MT, speakers) | **PASS** by code and configuration; **UNTESTED** with the network off | - | All models are local ONNX/CTranslate2 files (models.py, pinned); no cloud API in the code. Downloads only during setup |
| B3 | Python | **PASS** | - | Python 3.13, `requirements.txt` pinned |
| B4 | Runs on the laptop used in the interview | **PASS** so far | - | All tests on the target Windows laptop; install from scratch not yet re-tested (E1 in section E) |
| B5 | Models chosen deliberately, explained in the README | **PASS** | - | Reasons exist in DECISIONS.md D3, D4, D14, D20 and TESTS E1-E3, T-SPK-1 |

## C. Deliverables

| # | Requirement | Status | Evidence |
|---|---|---|---|
| C1 | Git repository link | **PASS** | https://github.com/sarkiskalikian/live-armenian-translation (private; reviewers need access) |
| C2 | Source code | **PASS** | `livetrans/` (audio, segmenter, asr, translate, speakers, pipeline), `tools/` |
| C3 | README: setup and run, incl. audio device setup | **PASS** | README sections 1-3 |
| C4 | README: models chosen, alternatives, why | **PASS** | README section 5 |
| C5 | README: how the pieces fit together | **PASS** | README section 4 |
| C6 | README: known limitations, what to improve | **PASS** | README sections 7-8 |
| C7 | `chats/`: every AI session, full and unedited, one `.md` per session | **One session submitted** | `chats/challenge-requirements-and-plan.md`, exported in full and unedited |
| C8 | No model weights, audio recordings or API keys committed | **PASS** | 49 tracked files, largest 76 KB; no audio/model files in the tree or in history; no key-like strings (re-checked 2026-11-01). `results/*.jsonl` and `eval/*.jsonl` contain transcribed text only |

## D. What the interviewers will look at live ("How we evaluate" - Live result)

| # | Question | Status | Evidence / numbers |
|---|---|---|---|
| D1 | Does it run? | **PASS** | All live runs since T-LIVE-4 without processing failures |
| D2 | Is the Armenian recognised? | **PASS, with errors** | FLEURS read speech WER 11.9 % (E3, 100 clips); user-confirmed call clips 15.0 % (9 errors / 60 words, 14 clips: T-LIVE-4 refs 23.8 %, T-LIVE-6 refs 10.3 %). Quiet/distant speakers and Meet's noise suppression make it worse (T-INSP-1) |
| D3 | Is the English understandable? | **PASS, with errors** | Correct Armenian -> mostly correct English (E2); place names fail («Վանաձոր» -> "Vancouver"); wrong Armenian -> wrong English |
| D4 | Under 5 seconds? | see B1 | |
| D5 | Speakers kept apart (>= 2 on the call side)? | **PASS** (user judgement) | see A7, A8 |

## E. Pre-interview checks

| # | Check | Status |
|---|---|---|
| E1 | Fresh setup from the README in a new folder (clone, venv, install, download models, `--verify`) | UNTESTED |
| E2 | `python -m livetrans.models --verify` prints ALL OK in Windows | UNTESTED (never reported) |
| E3 | Real two-person call acceptance run (section F) | done (T-LIVE-8) |
| E4 | Chat history still available in the app (interviewers open the originals) | user to keep |

## F. Real-call acceptance run - pass criteria (fixed before running)

**Note:** these criteria were proposed during development; they are **not
requirements of the brief**. The brief's requirements are sections A-C.
Outcome of T-LIVE-8 is given per row.

Setup: Google Meet, **two** remote people speaking Armenian, the user on the
laptop mic, Buds as output, ~3-5 minutes; default program settings
(`--mic 14 --loopback 16 --save-audio`; speaker labels on, threshold 0.30,
beam 1, CTC).

| # | Criterion | Pass if | T-LIVE-8 |
|---|---|---|---|
| F1 | Every line shows time, label, HY, EN | all lines (EN may be absent only for < 2-letter text, D12) | met |
| F2 | Mic phrases labelled "Me" | 100 % | met (27/27) |
| F3 | Remote speakers created | equal to the number of people who spoke on the remote side | met (2 created, 2 people) |
| F4 | Remote label agreement (user labels, clips >= 1.0 s, each Speaker N mapped to its majority person) | >= 85 % | not scored (no per-phrase labels) |
| F5 | A -> B -> A in the conversation | labels follow 1 -> 2 -> 1 | labels 1 -> 2 -> 1 in order; match to people by user judgement only |
| F6 | Latency | 0 lines over 5 s; report p50 / p95 / max and the speaker step | met: 0/45, p50 1.67/1.51 s, p95 2.13/2.35 s, max 2.37 s, speaker step 0.07 s |
| F7 | Failures | no processing failures; capture problems reported, not hidden | met: no failures; mic frame shortfall reported |

Known limitations that may show up and are not acceptance failures by
themselves (to be documented in the README): a phrase containing two voices
gets one label; very short replies (< 1 s) inherit the previous remote label;
the user's own voice echoed back through a remote phone is labelled as a
remote speaker (echo handling is out of scope); quiet or distant remote speech
can be silenced by Meet before it reaches the laptop; the mic frame shortfall
(open defect) can affect "Me" lines.
