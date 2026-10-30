"""Model registry: pinned Hugging Face revisions, expected file hashes, download and verify.

  python -m livetrans.models                 # download the core models at their pinned revisions
  python -m livetrans.models --only asr-rnnt # download one optional model
  python -m livetrans.models --verify        # check file hashes and runtime versions

Revisions are the commits that were actually tested (read from the
huggingface_hub download metadata of the files in use, 2026-10-30).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASR_DIR = ROOT / "models" / "asr-hy-fastconformer"
ASR_RNNT_DIR = ROOT / "models" / "asr-hy-fastconformer-rnnt"
MT_DIR = ROOT / "models" / "mt-nllb-600m-int8"
VAD_DIR = ROOT / "models" / "vad-silero"

MODELS = {
    "vad": dict(
        repo="istupakov/silero-vad-onnx", revision="b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
        dir=VAD_DIR, patterns=["silero_vad.onnx", "config.json"],
        sha256={"silero_vad.onnx": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"}),
    "asr-ctc": dict(  # NVIDIA FastConformer-Hybrid Large hy, CTC head, fp32 ONNX export
        repo="OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx",
        revision="f7db1fad9af6bd8a7a9e08fca02068c403b35468",
        dir=ASR_DIR, patterns=["model.onnx", "vocab.txt", "config.json"],
        sha256={"model.onnx": "09ed4278c8536193f5324ae5b1e21be3ef1abaa92779e8c7b4577644071b131a"}),
    "mt": dict(  # NLLB-200 distilled 600M, third-party CTranslate2 int8 conversion
        repo="JustFrederik/nllb-200-distilled-600M-ct2-int8",
        revision="302d78f00e6fdb50a1064059df7c392b735e9d05",
        dir=MT_DIR, patterns=None,
        sha256={"model.bin": "ed1beaf75134de7505315a5223162f56acff397eff6b50638a500d3936fe707b"}),
    "mt-spm": dict(  # Meta's own SentencePiece model for NLLB
        repo="facebook/nllb-200-distilled-600M", revision="f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        dir=MT_DIR, patterns=["sentencepiece.bpe.model"],
        sha256={"sentencepiece.bpe.model": "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a"}),
    # --- optional / under evaluation (not used by the live program) ---
    "asr-rnnt": dict(  # same NVIDIA weights, RNN-T (transducer) head, fp32 files only
        repo="Vana-Labs/stt-fastconformer-armenian",
        revision="ec92cef52199c3874681321b13853e8fe602d23f",   # the revision evaluated in E3
        dir=ASR_RNNT_DIR,
        patterns=["encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt", "config.json"],
        sha256={"encoder-model.onnx": "541c34628b1e67d659f3c3f0a7c204b399c071a2988de8df6194daac54446ec9",
                "decoder_joint-model.onnx": "5bb91df17e3660e8989279fa93152fdae8bae2093a9917bca119438a55edd623"},
        optional=True),
}
CORE = [k for k, m in MODELS.items() if not m.get("optional")]


def vad_model_path() -> Path:
    override = os.environ.get("LIVETRANS_VAD_MODEL")     # for tests with a different copy of Silero
    if override:
        return Path(override)
    hits = sorted(VAD_DIR.glob("*.onnx"))
    if not hits:
        raise FileNotFoundError(f"No Silero VAD model in {VAD_DIR}. Run: python -m livetrans.models")
    return hits[0]


def fetch(names: list[str]) -> None:
    from huggingface_hub import snapshot_download
    for name in names:
        m = MODELS[name]
        rev = m["revision"] or "main"
        print(f"{name}: {m['repo']} @ {rev[:12]} -> {m['dir'].relative_to(ROOT)}")
        snapshot_download(m["repo"], revision=rev, local_dir=m["dir"], allow_patterns=m["patterns"])
    print("Done. If a download was interrupted, delete that folder under models/ and rerun.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def downloaded_revision(model_dir: Path, filename: str) -> str | None:
    """Commit recorded by huggingface_hub when the file was downloaded into model_dir."""
    meta = model_dir / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    return meta.read_text(encoding="utf-8").splitlines()[0].strip() if meta.exists() else None


def verify(names: list[str]) -> bool:
    ok = True
    for name in names:
        m = MODELS[name]
        for fname, want in m["sha256"].items():
            p = m["dir"] / fname
            if not p.exists():
                print(f"MISSING  {name}: {p.relative_to(ROOT)}")
                ok = False
                continue
            got = _sha256(p)
            rev = downloaded_revision(m["dir"], fname)
            rev_ok = rev is None or m["revision"] is None or rev == m["revision"]
            status = "OK      " if got == want and rev_ok else "MISMATCH"
            ok &= status.startswith("OK")
            print(f"{status} {name}: {fname} sha256 {got[:12]} (pinned {want[:12]}), "
                  f"revision {rev[:12] if rev else 'unknown'} (pinned {(m['revision'] or 'none')[:12]})")
    return ok


def verify_runtime() -> bool:
    """Compare installed package versions with the exact pins in requirements.txt."""
    from importlib.metadata import PackageNotFoundError, version
    ok = True
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        pkg, want = line.split("==")
        pkg = pkg.split("[")[0].strip()
        try:
            got = version(pkg)
        except PackageNotFoundError:
            got = "not installed"
        status = "OK      " if got == want.strip() else "MISMATCH"
        ok &= status.startswith("OK")
        print(f"{status} {pkg} {got} (pinned {want.strip()})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(MODELS))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    names = args.only or CORE
    if args.verify:
        ok = verify(names)
        ok &= verify_runtime()
        print("ALL OK" if ok else "SOMETHING DIFFERS")
        return 0 if ok else 1
    fetch(names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
