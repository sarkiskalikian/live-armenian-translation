"""Armenian speech -> Armenian text.

Model: NVIDIA FastConformer-Hybrid Large (hy), 115M params, exported to ONNX
(CTC head) by OpenVoiceOS and run with onnx-asr + ONNX Runtime on CPU.
No NeMo / PyTorch needed. The model is non-streaming: we call it once per
speech segment (one VAD segment = one call).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ASR_REPO = "OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx"
SAMPLE_RATE = 16_000


class Transcriber:
    def __init__(self, model_dir: str | Path = "models/asr-hy-fastconformer", threads: int | None = None,
                 spin: bool = True, model_type: str | None = None, quantization: str | None = None):
        """threads=None and spin=True reproduce ONNX Runtime's defaults (all cores,
        busy-waiting between runs), i.e. the T-LIVE-1 configuration. Capping threads
        and disabling spinning are candidate mitigations for the mic frame shortfall (D10)."""
        import onnx_asr
        import onnxruntime as rt

        opts = None
        if threads is not None or not spin:
            opts = rt.SessionOptions()
            if threads is not None:
                opts.intra_op_num_threads = threads
            if not spin:
                opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
                opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        # Default: the OpenVoiceOS CTC export, identified by its repo id (onnx-asr reads
        # model_dir). model_type="nemo-conformer-rnnt" loads the Vana-Labs RNN-T export
        # from model_dir instead; quantization="int8" picks its *.int8.onnx files.
        self.model = onnx_asr.load_model(model_type or ASR_REPO, path=Path(model_dir),
                                         quantization=quantization, sess_options=opts,
                                         providers=["CPUExecutionProvider"])

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: mono float32 in [-1, 1] at 16 kHz."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return self.model.recognize(audio, sample_rate=SAMPLE_RATE).strip()
