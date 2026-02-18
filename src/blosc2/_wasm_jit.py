#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import os
from pathlib import Path

_HELPERS_REGISTERED = False

_REGISTER_HELPERS_JS = r"""
(() => {
  const g = globalThis;
  if (g.__blosc2_me_jit_helper_ptrs) {
    return g.__blosc2_me_jit_helper_ptrs;
  }

  const moduleObj = g.Module || {};
  const pick = (name) => moduleObj[name] !== undefined ? moduleObj[name] : g[name];
  const runtime = {
    HEAPF32: pick("HEAPF32"),
    HEAPF64: pick("HEAPF64"),
    HEAPU8: pick("HEAPU8"),
    wasmMemory: pick("wasmMemory"),
    wasmTable: pick("wasmTable"),
    addFunction: pick("addFunction"),
    removeFunction: pick("removeFunction"),
    stackSave: pick("stackSave"),
    stackAlloc: pick("stackAlloc"),
    stackRestore: pick("stackRestore"),
    lengthBytesUTF8: pick("lengthBytesUTF8"),
    stringToUTF8: pick("stringToUTF8"),
    err: pick("err"),
  };

  const required = [
    "HEAPF32",
    "HEAPF64",
    "HEAPU8",
    "wasmMemory",
    "wasmTable",
    "addFunction",
    "removeFunction",
    "stackSave",
    "stackAlloc",
    "stackRestore",
    "lengthBytesUTF8",
    "stringToUTF8",
  ];
  const missing = required.filter((name) => !runtime[name]);
  if (missing.length > 0) {
    return { instantiatePtr: 0, freePtr: 0, error: `missing runtime members: ${missing.join(", ")}` };
  }

  if (typeof g._meJitInstantiate !== "function" || typeof g._meJitFreeFn !== "function") {
    return { instantiatePtr: 0, freePtr: 0, error: "me_jit_glue exports unavailable" };
  }

  const instantiateWrapper = (wasmPtr, wasmLen, bridgeLookupFnIdx) => {
    const start = wasmPtr >>> 0;
    const len = wasmLen >>> 0;
    if (start === 0 || len === 0) {
      return 0;
    }
    const wasmBytes = runtime.HEAPU8.slice(start, start + len);
    return g._meJitInstantiate(runtime, wasmBytes, bridgeLookupFnIdx | 0) | 0;
  };
  const freeWrapper = (fnIdx) => {
    g._meJitFreeFn(runtime, fnIdx | 0);
  };

  const instantiatePtr = runtime.addFunction(instantiateWrapper, "iiii");
  const freePtr = runtime.addFunction(freeWrapper, "vi");
  g.__blosc2_me_jit_helper_ptrs = {
    instantiatePtr,
    freePtr,
    instantiateWrapper,
    freeWrapper,
    runtime,
  };
  return g.__blosc2_me_jit_helper_ptrs;
})()
"""


def _trace_enabled() -> bool:
    value = os.environ.get("ME_DSL_TRACE", "")
    return value.lower() in {"1", "true", "on", "yes"}


def _trace(message: str) -> None:
    if _trace_enabled():
        print(f"[blosc2.wasm-jit] {message}")


def _js_eval(js_mod, source: str):
    evaluator = getattr(js_mod, "eval", None)
    if evaluator is not None:
        return evaluator(source)
    return js_mod.globalThis.eval(source)


def _load_glue_once(js_mod) -> bool:
    has_exports = _js_eval(
        js_mod,
        "typeof globalThis._meJitInstantiate === 'function' && "
        "typeof globalThis._meJitFreeFn === 'function'",
    )
    if bool(has_exports):
        return True

    glue_path = Path(__file__).with_name("me_jit_glue.js")
    try:
        glue_source = glue_path.read_text(encoding="utf-8")
    except OSError as exc:
        _trace(f"could not read {glue_path.name}: {exc}")
        return False

    try:
        _js_eval(js_mod, glue_source)
    except Exception as exc:  # pragma: no cover - pyodide-specific error path
        _trace(f"failed to evaluate {glue_path.name}: {exc}")
        return False

    has_exports = _js_eval(
        js_mod,
        "typeof globalThis._meJitInstantiate === 'function' && "
        "typeof globalThis._meJitFreeFn === 'function'",
    )
    return bool(has_exports)


def _create_helper_ptrs(js_mod) -> tuple[int, int] | None:
    try:
        result = _js_eval(js_mod, _REGISTER_HELPERS_JS)
    except Exception as exc:  # pragma: no cover - pyodide-specific error path
        _trace(f"helper setup JS failed: {exc}")
        return None

    try:
        instantiate_ptr = int(result.instantiatePtr)
        free_ptr = int(result.freePtr)
    except Exception as exc:  # pragma: no cover - pyodide-specific error path
        _trace(f"unexpected helper setup result: {exc}")
        return None

    if instantiate_ptr == 0 or free_ptr == 0:
        with_error = getattr(result, "error", None)
        if with_error:
            _trace(str(with_error))
        return None
    return instantiate_ptr, free_ptr


def init_wasm_jit_helpers() -> bool:
    global _HELPERS_REGISTERED
    if _HELPERS_REGISTERED:
        return True

    try:
        import js
    except ImportError:
        return False

    from . import blosc2_ext

    if not hasattr(blosc2_ext, "_register_wasm_jit_helpers"):
        _trace("extension does not expose _register_wasm_jit_helpers")
        return False
    if not _load_glue_once(js):
        _trace("me_jit_glue.js was not loaded")
        return False

    helper_ptrs = _create_helper_ptrs(js)
    if helper_ptrs is None:
        _trace("could not allocate addFunction helper pointers")
        return False

    instantiate_ptr, free_ptr = helper_ptrs
    try:
        blosc2_ext._register_wasm_jit_helpers(instantiate_ptr, free_ptr)
    except Exception as exc:  # pragma: no cover - pyodide-specific error path
        _trace(f"C helper registration failed: {exc}")
        return False
    _HELPERS_REGISTERED = True
    _trace(f"registered wasm JIT helper pointers instantiate={instantiate_ptr} free={free_ptr}")
    return True
