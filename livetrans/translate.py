"""Armenian text -> English text.

Model: NLLB-200 distilled 600M (Meta), int8 CTranslate2 conversion, on CPU.
Tokenisation uses NLLB's own SentencePiece model directly (no `transformers`).
Licence note: NLLB-200 weights are CC-BY-NC-4.0 (non-commercial).
"""
from __future__ import annotations

from pathlib import Path

NLLB_CT2_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"   # third-party int8 conversion
NLLB_SPM_REPO = "facebook/nllb-200-distilled-600M"                # official tokenizer file
SPM_FILE = "sentencepiece.bpe.model"
SRC, TGT = "hye_Armn", "eng_Latn"


class Translator:
    def __init__(self, model_dir: str | Path = "models/mt-nllb-600m-int8", threads: int = 4,
                 beam_size: int = 4):
        import ctranslate2
        import sentencepiece as spm

        model_dir = Path(model_dir)
        self.translator = ctranslate2.Translator(str(model_dir), device="cpu", compute_type="int8",
                                                 intra_threads=threads)
        self.sp = spm.SentencePieceProcessor(model_file=str(model_dir / SPM_FILE))
        self.beam_size = beam_size

    def translate_batch(self, texts: list[str]) -> list[str]:
        sources = [[SRC] + self.sp.encode(t, out_type=str) + ["</s>"] for t in texts]
        results = self.translator.translate_batch(
            sources, target_prefix=[[TGT]] * len(texts), beam_size=self.beam_size,
            max_decoding_length=256)
        out = []
        for r in results:
            tokens = r.hypotheses[0]
            if tokens and tokens[0] == TGT:
                tokens = tokens[1:]
            out.append(self.sp.decode(tokens).strip())
        return out

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        return self.translate_batch([text])[0]


def download(model_dir: str | Path = "models/mt-nllb-600m-int8") -> Path:
    """Fetch the CT2 model and the official SentencePiece model into model_dir."""
    from huggingface_hub import hf_hub_download, snapshot_download

    model_dir = Path(model_dir)
    snapshot_download(NLLB_CT2_REPO, local_dir=model_dir)
    hf_hub_download(NLLB_SPM_REPO, SPM_FILE, local_dir=model_dir)
    return model_dir
