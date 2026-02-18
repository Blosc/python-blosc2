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

  const candidates = [];
  const addCandidate = (name, obj) => {
    if (!obj || (typeof obj !== "object" && typeof obj !== "function")) {
      return;
    }
    candidates.push({ name, obj });
  };

  addCandidate("globalThis", g);
  addCandidate("globalThis.Module", g.Module);
  addCandidate("globalThis.__blosc2_pyodide_module", g.__blosc2_pyodide_module);
  addCandidate("globalThis.__blosc2_pyodide_api", g.__blosc2_pyodide_api);
  addCandidate("globalThis.pyodide", g.pyodide);
  addCandidate("globalThis.pyodide._module", g.pyodide && g.pyodide._module);
  addCandidate("globalThis.pyodide.module", g.pyodide && g.pyodide.module);
  addCandidate("globalThis.pyodide.Module", g.pyodide && g.pyodide.Module);
  addCandidate("globalThis.pyodide._api", g.pyodide && g.pyodide._api);
  addCandidate("globalThis.pyodide._api._module", g.pyodide && g.pyodide._api && g.pyodide._api._module);
  addCandidate("globalThis.pyodide._api.Module", g.pyodide && g.pyodide._api && g.pyodide._api.Module);

  const resolve = (name) => {
    for (const cand of candidates) {
      let value;
      try {
        value = cand.obj[name];
      } catch (_e) {
        value = undefined;
      }
      if (value !== undefined && value !== null) {
        if (typeof value === "function") {
          return value.bind(cand.obj);
        }
        return value;
      }
    }
    if (g[name] !== undefined && g[name] !== null) {
      return g[name];
    }
    return null;
  };

  const wasmExports = resolve("wasmExports") || resolve("exports");
  const asmObj = resolve("asm");

  const isWasmMemory = (value) =>
    typeof WebAssembly !== "undefined" &&
    typeof WebAssembly.Memory !== "undefined" &&
    value instanceof WebAssembly.Memory;
  const isWasmTable = (value) =>
    typeof WebAssembly !== "undefined" &&
    typeof WebAssembly.Table !== "undefined" &&
    value instanceof WebAssembly.Table;

  const findMemoryOrTableByType = (wantMemory) => {
    for (const cand of candidates) {
      const obj = cand.obj;
      if (!obj) {
        continue;
      }
      let keys = [];
      try {
        keys = Object.getOwnPropertyNames(obj);
      } catch (_e) {
        keys = [];
      }
      for (const key of keys) {
        let value;
        try {
          value = obj[key];
        } catch (_e) {
          continue;
        }
        if (wantMemory && isWasmMemory(value)) {
          return value;
        }
        if (!wantMemory && isWasmTable(value)) {
          return value;
        }
        if (value && (typeof value === "object" || typeof value === "function")) {
          if (wantMemory && isWasmMemory(value.memory)) {
            return value.memory;
          }
          if (!wantMemory && isWasmTable(value.__indirect_function_table)) {
            return value.__indirect_function_table;
          }
        }
      }
    }
    return null;
  };

  const wasmMemory =
    resolve("wasmMemory") ||
    resolve("memory") ||
    resolve("wasmMemoryObject") ||
    resolve("__wasmMemory") ||
    (asmObj && asmObj.memory) ||
    (asmObj && asmObj.wasmMemory) ||
    (wasmExports && wasmExports.memory) ||
    findMemoryOrTableByType(true) ||
    null;
  const wasmTable =
    resolve("wasmTable") ||
    resolve("__indirect_function_table") ||
    (asmObj && asmObj.__indirect_function_table) ||
    (asmObj && asmObj.wasmTable) ||
    (wasmExports && wasmExports.__indirect_function_table) ||
    findMemoryOrTableByType(false) ||
    null;
  const runtime = {
    HEAPF32: resolve("HEAPF32"),
    HEAPF64: resolve("HEAPF64"),
    HEAPU8: resolve("HEAPU8"),
    wasmMemory,
    wasmTable,
    addFunction: resolve("addFunction"),
    removeFunction: resolve("removeFunction"),
    stackSave: resolve("stackSave"),
    stackAlloc: resolve("stackAlloc"),
    stackRestore: resolve("stackRestore"),
    lengthBytesUTF8: resolve("lengthBytesUTF8"),
    stringToUTF8: resolve("stringToUTF8"),
    err: resolve("err"),
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
    const aliasKeys = ["wasmMemory", "memory", "wasmExports", "asm", "__indirect_function_table", "wasmTable"];
    const diag = candidates.map((cand) => {
      const have = required.filter((name) => {
        try {
          return !!cand.obj[name];
        } catch (_e) {
          return false;
        }
      });
      const aliases = aliasKeys.filter((name) => {
        try {
          return cand.obj[name] !== undefined && cand.obj[name] !== null;
        } catch (_e) {
          return false;
        }
      });
      return `${cand.name}=[${have.join(",")}],aliases=[${aliases.join(",")}]`;
    }).join(" | ");
    return {
      instantiatePtr: 0,
      freePtr: 0,
      error: `missing runtime members: ${missing.join(", ")}; candidates: ${diag}`,
    };
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


def _inject_pyodide_runtime_handles(js_mod) -> None:
    try:
        import pyodide_js
    except ImportError:
        return

    module_obj = None
    for name in ("_module", "module", "Module"):
        module_obj = getattr(pyodide_js, name, None)
        if module_obj is not None:
            break
    if module_obj is not None:
        js_mod.globalThis.__blosc2_pyodide_module = module_obj
        _trace("captured pyodide_js module handle")

    api_obj = getattr(pyodide_js, "_api", None)
    if api_obj is not None:
        js_mod.globalThis.__blosc2_pyodide_api = api_obj
        _trace("captured pyodide_js API handle")


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

    _inject_pyodide_runtime_handles(js)
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
