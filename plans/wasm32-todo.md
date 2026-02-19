# wasm32 TODO (priority focus)

Last local validation: 2026-02-19 (Pyodide cp313, Emscripten-4.0.9-wasm32).

Scope for this list:
- DSL control-flow stability on wasm32 (runtime OOB risk).
- miniexpr fast-path enablement/coverage on wasm32.

## P0: unblock core DSL JIT stability (runtime safety)

1. [x] Remove the unconditional Emscripten skip in the wasm smoke test.
   - Target: `tests/ndarray/test_wasm_dsl_jit.py::test_wasm_dsl_tcc_jit_smoke`
   - Result: skip removed, test now runs and passes on local Pyodide.

2. [x] Stabilize full-control-flow DSL kernels on wasm32.
   - Target: `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_full_control_flow_kept_as_dsl_function`
   - Result: skip removed, test now runs and passes on local Pyodide.

3. [x] Stabilize while-loop DSL kernels on wasm32.
   - Target: `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_while_kept_as_dsl_function`
   - Result: skip removed, test now runs and passes on local Pyodide.

4. [x] Stabilize scalar-parameter loop DSL kernels on wasm32.
   - Target: `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_accepts_scalar_param_per_call`
   - Result: skip removed, test now runs and passes on local Pyodide.

## P1: recover miniexpr fast-path coverage on wasm32

5. [x] Re-enable DSL miniexpr fast-path instrumentation tests.
   - Targets:
     - `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_index_symbols_keep_full_kernel`
     - `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_with_no_inputs_handles_windows_dtype_policy`
     - `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_index_symbols_float_cast_uses_miniexpr_fast_path`
     - `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_scalar_param_keeps_miniexpr_fast_path`
     - `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_scalar_float_cast_inlined_without_float_call`
     - `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_miniexpr_failure_raises_even_with_strict_disabled`
     - `tests/ndarray/test_dsl_kernels.py::test_lazyudf_jit_policy_forwarding`
   - Result: wasm skips removed and tests pass on local Pyodide.

6. [x] Re-enable lazyexpr miniexpr fast-path behavior tests on wasm32.
   - Targets:
     - `tests/ndarray/test_lazyexpr.py::test_lazyexpr_string_scalar_keeps_miniexpr_fast_path`
     - `tests/ndarray/test_lazyexpr.py::test_lazyexpr_unary_negative_literal_matches_subtraction`
     - `tests/ndarray/test_lazyexpr.py::test_lazyexpr_miniexpr_failure_falls_back_by_default`
     - `tests/ndarray/test_lazyexpr.py::test_lazyexpr_miniexpr_failure_raises_when_strict`
   - Fix applied: removed wasm-only non-DSL miniexpr gate in `src/blosc2/lazyexpr.py` (`fast_eval`).
   - Result: skips removed and all four tests pass on local Pyodide.

7. [x] Revisit int-cast DSL behavior currently expected to fail on wasm.
   - Target: `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_index_symbols_int_cast_matches_expected_ramp`
   - Current local behavior: succeeds on wasm and matches expected `int64` ramp output.
   - 2026-02-19 retest after updating `CMakeLists.txt` to a newer miniexpr and rebuilding `cp313` wasm wheel (`SHA256=45e128507d91ffd535cc070d6846ee970e0054aeeb04bb1336eb454571aa41f5`): behavior unchanged; direct `expr[:]` evaluation still raises.
   - 2026-02-19 retest after pinning miniexpr to `f5e276a151025f9307819c329a033f3f5293a714` and rebuilding `cp313` wasm wheel (`SHA256=cc2ca236ec419da8eeaea464af1fa39db5208ba09f5759952e3b5c44999f55a5`): behavior still unchanged (`expr[:]` raises), but chained cause now includes compile diagnostics (`details: failed to compile DSL expression`).
   - 2026-02-19 retest after fixing wasm-safe dtype mapping in `src/blosc2/blosc2_ext.pyx` and rebuilding `cp313` wasm wheel (`SHA256=fc96898e332e069bfc6764c4243f75eced41cb3837252927fb4098ad6d3972f8`): direct `expr[:]` now succeeds on wasm (`int64`, ramp 0..159), and the previous wasm-only `RuntimeError` expectation was removed from the test.
