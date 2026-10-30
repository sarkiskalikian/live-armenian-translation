"""Model locations and one-time download (all CPU, all local after download)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASR_DIR = ROOT / "models" / "asr-hy-fastconformer"
MT_DIR = ROOT / "models" / "mt-nllb-600m-int8"
VAD_DIR = ROOT / "models" / "vad-silero"


def vad_model_path() -> Path:
    override = os.environ.get("LIVETRANS_VAD_MODEL")     # for tests with a different copy of Silero
    if override:
        return Path(override)
    hits = sorted(VAD_DIR.glob("*.onnx"))
    if not hits:
        raise FileNotFoundError(f"No Silero VAD model in {VAD_DIR}. Run: python -m livetrans.models")
    return hits[0]


def fetch_all() -> None:
    import onnx_asr

    from livetrans.asr import Transcriber
    from livetrans.translate import download

    print("1/3 VAD: Silero (ONNX, ~2 MB)")
    onnx_asr.load_vad("silero", path=VAD_DIR)
    print("    ->", vad_model_path())
    print("2/3 ASR: FastConformer-Hybrid Large hy (ONNX CTC, ~460 MB)")
    Transcriber(ASR_DIR)
    print("3/3 MT: NLLB-200 distilled 600M int8 (CTranslate2, ~620 MB)")
    download(MT_DIR)
    print("Done. If a download was interrupted, delete that folder under models/ and rerun.")


if __name__ == "__main__":
    fetch_all()
