# wasm32 TODO (priority focus)

Scope for this list:
- DSL control-flow stability on wasm32 (runtime OOB risk).
- miniexpr fast-path enablement/coverage on wasm32.

## P0: unblock core DSL JIT stability (runtime safety)

1. Remove the unconditional Emscripten skip in the wasm smoke test.
   - `tests/ndarray/test_wasm_dsl_jit.py:22`
   - Current blocker: `tests/ndarray/test_wasm_dsl_jit.py:24` skips on Emscripten.
   - Exit criteria: smoke runs and passes on Pyodide in CI (no skip).

2. Stabilize full-control-flow DSL kernels on wasm32.
   - `tests/ndarray/test_dsl_kernels.py:350`
   - Current reason: "full DSL control-flow kernel is unstable on wasm32 (can trigger runtime OOB)".
   - Exit criteria: unskip and pass consistently.

3. Stabilize while-loop DSL kernels on wasm32.
   - `tests/ndarray/test_dsl_kernels.py:376`
   - Current reason: "while-loop DSL kernel is unstable on wasm32 (can trigger runtime OOB)".
   - Exit criteria: unskip and pass consistently.

4. Stabilize scalar-parameter loop DSL kernels on wasm32.
   - `tests/ndarray/test_dsl_kernels.py:398`
   - Current reason: "scalar-param DSL loop kernel is unstable on wasm32 (can trigger runtime OOB)".
   - Exit criteria: unskip and pass consistently.

## P1: recover miniexpr fast-path coverage on wasm32

5. Re-enable DSL miniexpr fast-path instrumentation tests.
   - `tests/ndarray/test_dsl_kernels.py:202`
   - `tests/ndarray/test_dsl_kernels.py:259`
   - `tests/ndarray/test_dsl_kernels.py:284`
   - `tests/ndarray/test_dsl_kernels.py:422`
   - `tests/ndarray/test_dsl_kernels.py:467`
   - `tests/ndarray/test_dsl_kernels.py:505`
   - `tests/ndarray/test_dsl_kernels.py:531`
   - Current reason: "miniexpr fast path is not available on WASM".
   - Exit criteria: tests run (no skip) and validate `_set_pref_expr`/policy behavior on wasm.

6. Re-enable lazyexpr miniexpr fast-path behavior tests on wasm32.
   - `tests/ndarray/test_lazyexpr.py:1488`
   - `tests/ndarray/test_lazyexpr.py:1523`
   - `tests/ndarray/test_lazyexpr.py:1559`
   - `tests/ndarray/test_lazyexpr.py:1582`
   - Current reason: "miniexpr fast path is not available on WASM".
   - Exit criteria: tests run (no skip), fallback/strict semantics match non-wasm expectations.

7. Revisit int-cast DSL behavior currently expected to fail on wasm.
   - `tests/ndarray/test_dsl_kernels.py:312`
   - Current behavior on wasm: expects `RuntimeError("DSL kernels require miniexpr")`.
   - Exit criteria: either (a) supported and validated, or (b) explicitly documented as out-of-scope with rationale.

## Suggested validation order

1. `tests/ndarray/test_wasm_dsl_jit.py`
2. `tests/ndarray/test_dsl_kernels.py -k "control_flow or while or scalar_param"`
3. `tests/ndarray/test_dsl_kernels.py -k "miniexpr_fast_path or jit_policy_forwarding"`
4. `tests/ndarray/test_lazyexpr.py -k "miniexpr_fast_path or strict_miniexpr or unary_negative_literal"`
