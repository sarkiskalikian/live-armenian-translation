# eval/

The tools `asr_compare.py --set calls`, `beam_check.py` and `speaker_check.py`
read reference files from this folder (`t-live-4_refs.jsonl`,
`t-live-6_speakers.jsonl`, `t-live-8_speakers.jsonl`).

Those files hold transcripts of private test calls, so they are not published.
To run the tools, record your own call with `--save-audio`, write references in
the same JSONL format (one object per clip: `wav`, `ref_hy`, `ref_status`,
and `speaker` for the speaker files) and place them here.
