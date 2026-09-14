"""Silero VAD (v5/v6 ONNX graph) without torch — ~30 lines around onnxruntime.

The model consumes 512-sample (32 ms) frames at 16 kHz and keeps a small recurrent state; the
ONNX file is taken from the `silero-vad` pip package (installed with --no-deps).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _find_model() -> Path:
    """Locate silero_vad.onnx inside the installed package without importing it (it imports torch)."""
    import importlib.util
    import site
    import sysconfig

    roots: list[Path] = []
    spec = importlib.util.find_spec("silero_vad")
    if spec and spec.submodule_search_locations:
        roots.extend(Path(p) for p in spec.submodule_search_locations)
    for p in [sysconfig.get_paths().get("purelib"), *site.getsitepackages(), site.getusersitepackages()]:
        if p:
            roots.append(Path(p) / "silero_vad")
    for root in roots:
        for cand in (root / "data" / "silero_vad.onnx", root / "data" / "silero_vad_16k_op15.onnx"):
            if cand.exists():
                return cand
        if root.is_dir():
            found = sorted(root.rglob("silero_vad*.onnx"))
            if found:
                return found[0]
    raise FileNotFoundError("silero_vad.onnx not found — pip install silero-vad --no-deps")


class SileroVAD:
    FRAME = 512
    SR = 16000

    def __init__(self, model_path: Path | None = None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path or _find_model()), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        names = {i.name for i in self.session.get_inputs()}
        self._legacy = "h" in names  # v4 graphs use h/c, v5+ use a single state tensor
        self.reset()

    def reset(self) -> None:
        if self._legacy:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        else:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample float32 frame."""
        if frame.shape[0] != self.FRAME:
            raise ValueError("frame must have 512 samples")
        if self._legacy:
            out, self._h, self._c = self.session.run(None, {
                "input": frame[None, :].astype(np.float32), "sr": np.array(self.SR, dtype=np.int64),
                "h": self._h, "c": self._c})
            return float(out[0][0])
        x = np.concatenate([self._context, frame]).astype(np.float32)[None, :]
        out, self._state = self.session.run(None, {"input": x, "state": self._state, "sr": np.array(self.SR, dtype=np.int64)})
        self._context = frame[-64:].astype(np.float32)
        return float(out[0][0])
