"""M1 step 1: download the models and a FLEURS test subset (one-time, ~1.3 GB).

  python tools/m1_fetch.py            # models + 100 FLEURS hy_am test clips
  python tools/m1_fetch.py --n 50

Everything lands in models/ and data/, both git-ignored.
FLEURS (CC-BY-4.0) sentences are FLORES sentences, so the same IDs in the
en_us split give English reference translations for the MT benchmark.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m1_common import DATA, ROOT, write_jsonl  # noqa: E402

FLEURS = "google/fleurs"


def find(files: list[str], lang: str, suffix: str) -> str:
    hits = [f for f in files if f"/{lang}/" in f and f.endswith(suffix)]
    if not hits:
        raise SystemExit(f"Could not find '{suffix}' for {lang} in {FLEURS}; repo layout changed?")
    return hits[0]


def parse_tsv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for parts in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            wav = next((p for p in parts if p.endswith(".wav")), None)
            if not wav:
                continue
            i = parts.index(wav)
            rows.append({"id": parts[0], "file": wav, "raw": parts[i + 1], "gender": parts[-1]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of FLEURS clips (unique sentences)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    if not args.skip_models:
        print("1/3 ASR model (FastConformer hy, ONNX, ~460 MB)...")
        from livetrans.asr import Transcriber
        Transcriber(ROOT / "models/asr-hy-fastconformer")       # downloads on first load
        print("2/3 MT model (NLLB-200 600M int8, ~620 MB)...")
        from livetrans.translate import download
        download(ROOT / "models/mt-nllb-600m-int8")

    print("3/3 FLEURS test split (hy_am audio + text, en_us text)...")
    raw_dir = DATA / "fleurs_raw"
    files = HfApi().list_repo_files(FLEURS, repo_type="dataset")
    hy_tsv = Path(hf_hub_download(FLEURS, find(files, "hy_am", "test.tsv"), repo_type="dataset", local_dir=raw_dir))
    en_tsv = Path(hf_hub_download(FLEURS, find(files, "en_us", "test.tsv"), repo_type="dataset", local_dir=raw_dir))
    hy, en = parse_tsv(hy_tsv), parse_tsv(en_tsv)
    en_by_id = {r["id"]: r["raw"] for r in en}

    # one clip per sentence id, only ids that also have an English reference
    seen, candidates = set(), []
    for r in hy:
        if r["id"] in en_by_id and r["id"] not in seen:
            seen.add(r["id"])
            candidates.append(r)
    random.Random(args.seed).shuffle(candidates)
    chosen = {r["file"]: r for r in candidates[: args.n]}
    print(f"   {len(hy)} hy_am test clips, {len(candidates)} usable sentences, taking {len(chosen)}")

    tar_path = Path(hf_hub_download(FLEURS, find(files, "hy_am", "test.tar.gz"), repo_type="dataset",
                                    local_dir=raw_dir))
    out_dir = DATA / "fleurs_hy"
    out_dir.mkdir(parents=True, exist_ok=True)
    found = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in chosen:
                (out_dir / name).write_bytes(tar.extractfile(member).read())
                found += 1
    rows = [{"id": r["id"], "wav": f"fleurs_hy/{r['file']}", "ref_hy": r["raw"],
             "ref_en": en_by_id[r["id"]], "gender": r["gender"]}
            for r in chosen.values() if (out_dir / r["file"]).exists()]
    write_jsonl(DATA / "fleurs_hy.jsonl", rows)
    print(f"   extracted {found} clips -> {out_dir}; manifest data/fleurs_hy.jsonl ({len(rows)} rows)")
    print("Done. Next: python tools/m1_bench_asr.py --set fleurs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
