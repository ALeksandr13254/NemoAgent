"""DeepSeek Proof-of-Work solver (DeepSeekHashV1) running the original WASM via wasmtime.

wasm_solve(retptr, challenge_ptr, challenge_len, prefix_ptr, prefix_len, difficulty: f64)
retptr -> 16 bytes: [0..4) i32 success flag, [8..16) f64 answer.
"""
from __future__ import annotations

import base64
import json
import struct
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import wasmtime

WASM_PATH = Path(__file__).parent / "sha3_wasm.wasm"


class WasmSolver:
    def __init__(self, wasm_path: Path = WASM_PATH):
        if not wasm_path.exists():
            raise FileNotFoundError(f"WASM module not found: {wasm_path}")
        self.engine = wasmtime.Engine()
        self.module = wasmtime.Module(self.engine, wasm_path.read_bytes())
        self.linker = wasmtime.Linker(self.engine)
        self._lock = threading.Lock()  # wasmtime Store is not thread-safe
        try:
            self.linker.define_wasi()
        except Exception:
            pass
        self._store = wasmtime.Store(self.engine)
        try:
            self._store.set_wasi(wasmtime.WasiConfig())
        except Exception:
            pass
        self._instance = self.linker.instantiate(self._store, self.module)
        self._exports = self._instance.exports(self._store)
        self._memory = self._exports["memory"]
        self._malloc = self._find_export(["__wbindgen_malloc", "__wbindgen_export_0", "alloc"])
        self._wasm_solve = self._find_export(["wasm_solve", "solve"])
        self._add_to_stack = self._find_export(["__wbindgen_add_to_stack_pointer"], required=False)

    def _find_export(self, names, required=True):
        for n in names:
            try:
                return self._exports[n]
            except KeyError:
                continue
        if required:
            raise RuntimeError(f"none of {names} exported by WASM module")
        return None

    def _write(self, ptr: int, data: bytes) -> None:
        try:
            self._memory.write(self._store, data, ptr)
        except Exception:
            mem = self._memory.data_ptr(self._store)
            for i, b in enumerate(data):
                mem[ptr + i] = b

    def _read(self, ptr: int, length: int) -> bytes:
        try:
            return bytes(self._memory.read(self._store, ptr, ptr + length))
        except Exception:
            mem = self._memory.data_ptr(self._store)
            return bytes(mem[ptr + i] for i in range(length))

    def solve(self, challenge: Dict[str, Any]) -> Dict[str, Any]:
        challenge_bytes = challenge["challenge"].encode("utf-8")
        prefix_bytes = f"{challenge['salt']}_{challenge['expire_at']}_".encode("utf-8")
        difficulty = float(challenge["difficulty"])
        with self._lock:
            if self._add_to_stack is not None:
                retptr = self._add_to_stack(self._store, -16)
            else:
                retptr = self._malloc(self._store, 16, 8)
            c_ptr = self._malloc(self._store, len(challenge_bytes), 1)
            self._write(c_ptr, challenge_bytes)
            p_ptr = self._malloc(self._store, len(prefix_bytes), 1)
            self._write(p_ptr, prefix_bytes)
            self._wasm_solve(self._store, retptr, c_ptr, len(challenge_bytes), p_ptr, len(prefix_bytes), difficulty)
            result = self._read(retptr, 16)
            success = struct.unpack("<i", result[0:4])[0]
            answer = struct.unpack("<d", result[8:16])[0]
            if self._add_to_stack is not None:
                self._add_to_stack(self._store, 16)
        if not success:
            raise RuntimeError("PoW WASM returned success=0 (challenge expired or too hard)")
        return {
            "algorithm": challenge.get("algorithm", "DeepSeekHashV1"),
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": int(answer),
            "signature": challenge["signature"],
            "target_path": challenge["target_path"],
        }


class DeepSeekPOW:
    def __init__(self, wasm_path: Optional[Path] = None) -> None:
        self.solver = WasmSolver(wasm_path or WASM_PATH)

    def solve_and_encode(self, challenge: Dict[str, Any]) -> str:
        return base64.b64encode(json.dumps(self.solver.solve(challenge)).encode()).decode()
