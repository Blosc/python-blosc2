# external-js-glue-blosc2: Session Handoff and TODO

## Goal
Enable miniexpr+tcc DSL JIT on `wasm32`/Pyodide and keep wasm CI focused on proving this path works.

## Current status
- Linux pyodide wheel build is working.
- DSL JIT wasm smoke is working:
  - `_WASM_MINIEXPR_ENABLED=True`
  - `tests/ndarray/test_wasm_dsl_jit.py::test_wasm_dsl_tcc_jit_smoke` passes.
- Detached buffer fatal error was fixed in `src/blosc2/_wasm_jit.py` by refreshing heap views before copying wasm bytes.
- Broad wasm suites (`test_lazyexpr.py`, `test_reductions.py`) still show many failures and should be tracked as separate wasm baseline work, not as DSL JIT bring-up blockers.
- Re-check on Linux (2026-02-18) confirms broad coverage is still not ready for CI restore:
  - `test_wasm_dsl_jit.py`: pass (1 passed).
  - `test_lazyexpr.py` + `test_reductions.py`: `63 failed, 3686 passed, 371 skipped, 4216 deselected`.
- Follow-up fix on Linux (2026-02-18): non-DSL wasm miniexpr path disabled in `fast_eval`.
  - Result: full wasm `test_lazyexpr.py` now passes (`859 passed, 371 skipped, 472 deselected`).

## Important changes already made
- Added wasm JIT bridge module: `src/blosc2/_wasm_jit.py`.
- Added wasm smoke test: `tests/ndarray/test_wasm_dsl_jit.py`.
- Added extension hook in `src/blosc2/blosc2_ext.pyx` to register helper pointers.
- Added wasm init path in `src/blosc2/__init__.py` (after extension/types are loaded, avoiding circular import).
- Updated `src/blosc2/lazyexpr.py` to gate miniexpr on wasm by `_WASM_MINIEXPR_ENABLED`.
- wasm CI test scope is intentionally focused to:
  - `tests/ndarray/test_wasm_dsl_jit.py`

## Notes about local setup
- There is currently no local `wasm/` helper directory in this checkout.
- Use cibuildwheel pyodide test hooks instead of local probe scripts.

## Recommended next tasks
1. Keep wasm CI target narrow: only run DSL smoke while this stream lands.
2. Record the lazyexpr/reductions wasm failures as a separate issue/workstream.
3. After merge, expand wasm test scope incrementally from DSL smoke to selected non-DSL tests.
4. Only when those pass, and after fixing current reductions/lazyexpr wasm regressions, consider restoring broader `test_lazyexpr.py` / `test_reductions.py` coverage in cibuildwheel.

## Latest verification details (2026-02-18)
- Baseline command (passes):
  - `XDG_CACHE_HOME=/tmp CIBW_PYODIDE_VERSION=0.29.3 CIBW_BUILD='cp313-*' CMAKE_ARGS='-DWITH_ZLIB_OPTIM=OFF -DWITH_OPTIM=OFF -DWITH_RUNTIME_CPU_DETECTION=OFF' CIBW_TEST_COMMAND='pytest -s {project}/tests/ndarray/test_wasm_dsl_jit.py' cibuildwheel --platform pyodide`
- Broader command (fails):
  - `XDG_CACHE_HOME=/tmp CIBW_PYODIDE_VERSION=0.29.3 CIBW_BUILD='cp313-*' CMAKE_ARGS='-DWITH_ZLIB_OPTIM=OFF -DWITH_OPTIM=OFF -DWITH_RUNTIME_CPU_DETECTION=OFF' CIBW_TEST_COMMAND='pytest -s {project}/tests/ndarray/test_lazyexpr.py {project}/tests/ndarray/test_reductions.py' cibuildwheel --platform pyodide`
- Failure shape:
  - `test_lazyexpr.py`: previously 2 failures (`test_save`, `test_dtype_infer_scalars[np.int32]`) now fixed by scoping wasm miniexpr to DSL kernels in `fast_eval`.
  - `test_reductions.py`: 61 failures, concentrated around reductions for `fill_value=1` and `axis=None` (notably `min`/`max`/`all`, and multiple `test_save_version*` variants).

## Commands
- Build wasm wheel on Linux:
  - `CIBW_PYODIDE_VERSION=0.29.3 CIBW_BUILD="cp313-*" CMAKE_ARGS="-DWITH_ZLIB_OPTIM=OFF -DWITH_OPTIM=OFF -DWITH_RUNTIME_CPU_DETECTION=OFF" cibuildwheel --platform pyodide`
- Run focused DSL smoke in cibuildwheel test env:
  - `CIBW_TEST_COMMAND="pytest -s {project}/tests/ndarray/test_wasm_dsl_jit.py"`

## Files in this workstream
- `src/blosc2/_wasm_jit.py`
- `tests/ndarray/test_wasm_dsl_jit.py`
- `plans/external-js-glue-blosc2-todo.md`
