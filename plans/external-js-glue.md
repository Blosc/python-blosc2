# Plan: External JS Glue for WASM32 JIT in Side-Module Builds

## Problem Statement

When python-blosc2 is built for Pyodide via cibuildwheel, the extension
(`blosc2_ext.so`) is compiled as an **Emscripten side module**
(`-s SIDE_MODULE=1`).  Side modules cannot use `EM_JS` macros because the
`__em_js__`-prefixed symbols they generate are only resolvable by the main
module's linker.  This makes the two `EM_JS` functions that power the wasm32
JIT (`me_wasm_jit_instantiate` and `me_wasm_jit_free_fn`) unavailable,
currently forcing JIT to be disabled entirely in Pyodide.

## Goal

Keep the full TCC→WASM JIT pipeline working inside a Pyodide side-module
build by moving the JS glue out of `EM_JS` and into a runtime-loaded
external script that Pyodide's main module can invoke.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Pyodide main module (has wasmMemory, wasmTable,    │
│  addFunction, removeFunction, stack helpers …)       │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  me_jit_glue.js  (loaded once at init time)  │   │
│  │  ─ exposes globalThis._meJitInstantiate()    │   │
│  │  ─ exposes globalThis._meJitFreeFn()         │   │
│  └──────────────────────────────────────────────┘   │
│         ▲              ▲                             │
│         │ call         │ call                        │
└─────────┼──────────────┼────────────────────────────┘
          │              │
┌─────────┼──────────────┼────────────────────────────┐
│  blosc2_ext.so  (side module)                        │
│                                                      │
│  miniexpr.c  ──► me_wasm_jit_instantiate_indirect()  │
│                  (calls JS via emscripten_run_script  │
│                   or registered function pointer)     │
│                                                      │
│  dsl_jit_compile_wasm32()  ──► TCC compile ──►       │
│                    write /tmp/me_jit_kernel.wasm      │
│                    read bytes ──► call instantiate    │
└──────────────────────────────────────────────────────┘
```

---

## Detailed Work Items

### Phase 1 — Extract JS Glue Into a Standalone File

**File: `src/me_jit_glue.js`** (new, lives in miniexpr repo)

- [ ] Extract the ~400-line JS body of `me_wasm_jit_instantiate` into a
      self-contained function:
      ```js
      globalThis._meJitInstantiate = function(wasmBytesPtr, wasmLen, bridgeLookupIdx, runtime) { … };
      ```
- [ ] Extract `me_wasm_jit_free_fn`:
      ```js
      globalThis._meJitFreeFn = function(idx, runtime) { … };
      ```
- [ ] The `runtime` parameter is an object the host passes in, containing
      all the Emscripten globals the JS code currently reads as free
      variables:
      ```js
      {
        HEAPU8, HEAPF32, HEAPF64,
        wasmMemory, wasmTable,
        addFunction, removeFunction,
        stackSave, stackRestore, stackAlloc,
        stringToUTF8, lengthBytesUTF8,
      }
      ```
      This decouples the glue from any assumption about whether it runs
      inside the main module's scope.
- [ ] Add a lightweight self-test that can run under Node.js with a mock
      `runtime` object (just verifies parse/patch logic, not full
      instantiation).

### Phase 2 — Add an Indirect Call Path in miniexpr.c

**File: `src/miniexpr.c`** (modify existing)

- [ ] Define two new **function-pointer slots** (file scope, `static`):
      ```c
      typedef int  (*me_wasm_jit_instantiate_fn)(const unsigned char *, int, int);
      typedef void (*me_wasm_jit_free_fn_t)(int);

      static me_wasm_jit_instantiate_fn  me_wasm_jit_instantiate_ptr  = NULL;
      static me_wasm_jit_free_fn_t       me_wasm_jit_free_fn_ptr     = NULL;
      ```
- [ ] Add a **public registration API**:
      ```c
      void me_register_wasm_jit_helpers(me_wasm_jit_instantiate_fn inst,
                                        me_wasm_jit_free_fn_t      free_fn);
      ```
      This is the entry point that the Python/Pyodide layer calls after
      loading the JS glue, passing trampolines that bridge into JS.
- [ ] Gate the existing `EM_JS`-based code so it is only compiled when
      `ME_USE_WASM32_JIT && !ME_WASM32_SIDE_MODULE` (i.e., standalone
      Emscripten main-module builds keep working unchanged).
- [ ] When `ME_WASM32_SIDE_MODULE` is defined:
    - `dsl_jit_compile_wasm32()` uses `me_wasm_jit_instantiate_ptr`
      instead of the `EM_JS` function.
    - `dsl_compiled_program_free()` uses `me_wasm_jit_free_fn_ptr`.
    - Both check for `NULL` and return gracefully (JIT disabled) if the
      host never registered the helpers.
- [ ] Expose the function-pointer slots via `miniexpr.h` so the Python
      extension can call `me_register_wasm_jit_helpers()`.

### Phase 3 — Load the JS Glue From Python / Pyodide

**File: `src/blosc2/__init__.py`** (modify existing, WASM path only)

- [ ] At import time, when `IS_WASM` is true:
      1. Use `pyodide.code.run_js()` (or `js.eval()`) to load
         `me_jit_glue.js` from the package's data directory.
      2. Build the `runtime` object by pulling the necessary globals from
         Pyodide's `Module` (Pyodide exposes `pyodide._module` or similar).
      3. Create two small JS wrapper functions that close over `runtime`
         and delegate to `_meJitInstantiate` / `_meJitFreeFn`.
      4. Convert these JS functions into C-callable function pointers via
         Pyodide's `create_proxy` + `addFunction` (Pyodide re-exports
         Emscripten's `addFunction`).
      5. Call `blosc2_ext.me_register_wasm_jit_helpers(inst_ptr, free_ptr)`
         (exposed as a thin Cython wrapper).

**File: `src/blosc2/blosc2_ext.pyx`** (modify existing)

- [ ] Add a Cython `cdef extern` declaration for
      `me_register_wasm_jit_helpers` and a thin Python-callable wrapper.

**File: `pyproject.toml` / `CMakeLists.txt`**

- [ ] Include `me_jit_glue.js` in the built wheel's package data so it
      ships alongside the `.so`.

### Phase 4 — Wire Up the Runtime Object in Pyodide

The trickiest part is getting the Emscripten runtime references.  Pyodide
exposes them in slightly different ways across versions; the code should
try several paths:

```python
# Pseudocode — exact API depends on Pyodide version
from pyodide.code import run_js

run_js(
    """
    // 'Module' is Pyodide's Emscripten Module object
    const rt = {
        HEAPU8:           Module.HEAPU8,
        HEAPF32:          Module.HEAPF32,
        HEAPF64:          Module.HEAPF64,
        wasmMemory:       Module.wasmMemory  || wasmMemory,
        wasmTable:        Module.wasmTable   || wasmTable,
        addFunction:      Module.addFunction || addFunction,
        removeFunction:   Module.removeFunction || removeFunction,
        stackSave:        Module.stackSave   || stackSave,
        stackRestore:     Module.stackRestore|| stackRestore,
        stackAlloc:       Module.stackAlloc  || stackAlloc,
        stringToUTF8:     Module.stringToUTF8,
        lengthBytesUTF8:  Module.lengthBytesUTF8,
    };
    globalThis._meJitRuntime = rt;
"""
)
```

- [ ] Confirm which Pyodide version(s) expose these globals and document
      the minimum supported version.
- [ ] Add a fallback: if any required global is missing, skip registration
      (JIT stays disabled, interpreter path is used — same as today).

### Phase 5 — Testing

- [ ] **miniexpr standalone (main-module) tests**: Must keep passing
      unchanged — the `EM_JS` path is untouched.
- [ ] **miniexpr side-module unit test**: New CMake target that builds
      miniexpr as a side module, loads the external JS glue via Node.js,
      registers the helpers, and runs a simple JIT kernel.
- [ ] **python-blosc2 Pyodide CI** (`wasm.yml`): After the fix, the CI
      should show `jit runtime built: … compiler=tcc` in traces instead of
      the current `jit runtime skip`.
- [ ] **Fallback test**: Verify that if `me_jit_glue.js` fails to load
      (or Pyodide lacks the required globals), expressions still evaluate
      correctly via the interpreter path.

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pyodide changes its `Module` API between versions | JS glue can't find runtime globals | Probe multiple paths; fail gracefully to interpreter |
| `addFunction` quota is limited in Pyodide's config | Can't register trampolines | Pyodide default table size is generous; document `ALLOW_TABLE_GROWTH` if needed |
| TCC `/tmp` virtual FS not available in Pyodide | Can't write intermediate `.wasm` | Pyodide provides MEMFS at `/tmp` by default; verify in CI |
| Performance overhead of indirect call via JS proxy | JIT kernel invocation is slower | The indirection is only at instantiation time, not per-element; kernel execution goes through the function table directly, same as today |
| `wasmTable.get(bridgeLookupIdx)` may not work if side-module table is separate | Bridge callback unreachable from JS | Use `addFunction` on the Python side to re-register the bridge callback into the main table |

## Alternatives Considered

1. **Build blosc2 as a main module** — Conflicts with Pyodide's extension
   model; every other Python C extension is a side module.
2. **Pre-compiled WASM kernels** — Loses arbitrary-expression flexibility;
   combinatorial explosion of kernel variants.
3. **Disable JIT on WASM entirely** — This is the current workaround and
   what the `miniexpr-wwasm32.patch` implements.  It is the right
   short-term fix but leaves performance on the table.

## Dependencies

- miniexpr must expose `me_register_wasm_jit_helpers()` in its public API.
- python-blosc2 must update its pinned miniexpr commit after the miniexpr
  changes land.
- Minimum Pyodide version must be documented (likely ≥ 0.25 for stable
  `Module` access).

## Suggested Implementation Order

1. Apply the existing `miniexpr-wwasm32.patch` first so CI is green
   (JIT disabled in side modules — the safe baseline).
2. Implement Phases 1–2 in miniexpr (JS file + indirect call path).
3. Implement Phases 3–4 in python-blosc2 (load glue + register helpers).
4. Implement Phase 5 tests, confirm CI shows `jit runtime built`.
5. Remove the `ME_WASM32_SIDE_MODULE` compile-time disable once the
   runtime path is proven stable.
