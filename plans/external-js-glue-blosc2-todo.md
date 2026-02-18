# external-js-glue-blosc2: Session Handoff and TODO

## Goal
Enable miniexpr+tcc DSL JIT on `wasm32`/Pyodide and keep wasm CI focused on proving this path works.

## Current status
- Linux pyodide wheel build is working.
- DSL JIT wasm smoke is working:
  - `_WASM_MINIEXPR_ENABLED=True`
  - `tests/ndarray/test_wasm_dsl_jit.py::test_wasm_dsl_tcc_jit_smoke` passes.
- Detached buffer fatal error was fixed in `src/blosc2/_wasm_jit.py` by refreshing heap views before copying wasm bytes.
- Follow-up fixes on Linux (2026-02-18):
  - Non-DSL wasm miniexpr path disabled in `fast_eval`.
  - Wasm reduction miniexpr disabled in `reduce_slices` (regular chunked reduction path is kept on wasm).
- Result: broad wasm suites are now passing.
  - `test_lazyexpr.py`: `859 passed, 371 skipped, 472 deselected`.
  - `test_reductions.py`: `2890 passed, 3744 deselected`.
  - Combined command (`test_wasm_dsl_jit.py` + `test_lazyexpr.py` + `test_reductions.py`): `3750 passed, 371 skipped, 4216 deselected`.
- Task 4 is now complete: broader wasm coverage was restored in cibuildwheel workflow (`.github/workflows/wasm.yml`).

## Important changes already made
- Added wasm JIT bridge module: `src/blosc2/_wasm_jit.py`.
- Added wasm smoke test: `tests/ndarray/test_wasm_dsl_jit.py`.
- Added extension hook in `src/blosc2/blosc2_ext.pyx` to register helper pointers.
- Added wasm init path in `src/blosc2/__init__.py` (after extension/types are loaded, avoiding circular import).
- Updated `src/blosc2/lazyexpr.py` to gate miniexpr on wasm by `_WASM_MINIEXPR_ENABLED`.
- Updated `.github/workflows/wasm.yml` so wasm cibuildwheel runs:
  - `tests/ndarray/test_wasm_dsl_jit.py`
  - `tests/ndarray/test_lazyexpr.py`
  - `tests/ndarray/test_reductions.py`

## Notes about local setup
- There is currently no local `wasm/` helper directory in this checkout.
- Use cibuildwheel pyodide test hooks instead of local probe scripts.

## Recommended next tasks
1. Land the current wasm guards in `src/blosc2/lazyexpr.py` and keep monitoring wasm CI stability.
2. Open a follow-up task to re-enable non-DSL/reduction miniexpr on wasm incrementally once root causes are addressed.

## Latest verification details (2026-02-18)
- Baseline command (passes):
  - `XDG_CACHE_HOME=/tmp CIBW_PYODIDE_VERSION=0.29.3 CIBW_BUILD='cp313-*' CMAKE_ARGS='-DWITH_ZLIB_OPTIM=OFF -DWITH_OPTIM=OFF -DWITH_RUNTIME_CPU_DETECTION=OFF' CIBW_TEST_COMMAND='pytest -s {project}/tests/ndarray/test_wasm_dsl_jit.py' cibuildwheel --platform pyodide`
- Broader commands (pass):
  - `XDG_CACHE_HOME=/tmp CIBW_PYODIDE_VERSION=0.29.3 CIBW_BUILD='cp313-*' CMAKE_ARGS='-DWITH_ZLIB_OPTIM=OFF -DWITH_OPTIM=OFF -DWITH_RUNTIME_CPU_DETECTION=OFF' CIBW_TEST_COMMAND='pytest -s {project}/tests/ndarray/test_lazyexpr.py {project}/tests/ndarray/test_reductions.py' cibuildwheel --platform pyodide`
  - `XDG_CACHE_HOME=/tmp XDG_DATA_HOME=/tmp CIBW_PYODIDE_VERSION=0.29.3 CIBW_BUILD='cp313-*' CMAKE_ARGS='-DWITH_ZLIB_OPTIM=OFF -DWITH_OPTIM=OFF -DWITH_RUNTIME_CPU_DETECTION=OFF' CIBW_TEST_COMMAND='pytest -s {project}/tests/ndarray/test_wasm_dsl_jit.py {project}/tests/ndarray/test_lazyexpr.py {project}/tests/ndarray/test_reductions.py' cibuildwheel --platform pyodide`
- Current totals:
  - `test_lazyexpr.py`: `859 passed, 371 skipped, 472 deselected`.
  - `test_reductions.py`: `2890 passed, 3744 deselected`.
  - Combined: `3750 passed, 371 skipped, 4216 deselected`.

## Commands
- Build wasm wheel on Linux:
  - `CIBW_PYODIDE_VERSION=0.29.3 CIBW_BUILD="cp313-*" CMAKE_ARGS="-DWITH_ZLIB_OPTIM=OFF -DWITH_OPTIM=OFF -DWITH_RUNTIME_CPU_DETECTION=OFF" cibuildwheel --platform pyodide`
- Run restored wasm test scope in cibuildwheel test env:
  - `CIBW_TEST_COMMAND="pytest -s {project}/tests/ndarray/test_wasm_dsl_jit.py {project}/tests/ndarray/test_lazyexpr.py {project}/tests/ndarray/test_reductions.py"`

## Files in this workstream
- `src/blosc2/_wasm_jit.py`
- `src/blosc2/lazyexpr.py`
- `tests/ndarray/test_wasm_dsl_jit.py`
- `.github/workflows/wasm.yml`
- `plans/external-js-glue-blosc2-todo.md`
