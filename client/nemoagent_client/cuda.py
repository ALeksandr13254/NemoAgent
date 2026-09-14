"""Make pip-installed NVIDIA CUDA libraries visible to onnxruntime-gpu and CTranslate2.

On Windows the `nvidia-*-cu12` wheels drop DLLs into site-packages/nvidia/<lib>/bin; neither
onnxruntime nor ctranslate2 look there by themselves. Call `setup_cuda_paths()` BEFORE importing
onnxruntime / faster_whisper. On Linux the wheels put .so files into .../nvidia/<lib>/lib and the
loaders usually find them through the RPATH baked into the wheels, so this is mostly a no-op.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_done = False


def setup_cuda_paths() -> list[str]:
    global _done
    if _done:
        return []
    _done = True
    added: list[str] = []
    try:
        import nvidia  # type: ignore  # namespace package from the pip wheels
    except Exception:
        return added
    roots = [Path(p) for p in getattr(nvidia, "__path__", [])]
    sub = "bin" if sys.platform == "win32" else "lib"
    for root in roots:
        if not root.is_dir():
            continue
        for lib in sorted(root.iterdir()):
            d = lib / sub
            if not d.is_dir():
                continue
            # cudnn 9 splits into several DLLs in the same bin folder, cublas ships cublasLt too.
            added.append(str(d))
            if sys.platform == "win32":
                try:
                    os.add_dll_directory(str(d))
                except Exception:
                    pass
    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
        if sys.platform != "win32":
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    return added


def cuda_available_for_onnx() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False
