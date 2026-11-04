# Typed expression evaluator rollout

Ordinary expressions now use the typed evaluator by default. No environment
variable is needed:

```sh
python your_program.py
```

During qualification, the previous evaluator remains available as a temporary
escape hatch. Select it before constructing expressions:

```sh
BLOSC2_EXPRESSION_EVALUATOR=legacy python your_program.py
```

Explicit `BLOSC2_EXPRESSION_EVALUATOR=typed` is also accepted. This selector is
temporary, not permanent API configuration. Explicit compatibility DSL kernels
and Python UDFs keep their existing execution contracts. The development default
has switched ahead of fresh full platform CI; that qualification remains required
before deleting the ordinary-expression legacy path.

The typed route reuses Menudet graph plans and admitted compression prefilters.
Other layouts/signatures use bounded chunk evaluation through typed graph
schedules or NumPy. It never retries an ordinary expression with the legacy
uniform-dtype prefilter, and runtime failures are not replayed through NumPy.
Typed scalar captures are not converted into untyped literals.

Rebuilt bindings additionally admit N-D layouts whose physical chunks and blocks
are consecutive C-order runs, with padding only at the end. They reuse the typed
native prefilter rather than decompressing whole chunks for Python evaluation.
Older installed extensions still work but keep the narrower row-slab admission.

This addresses condition/reduction typing and integer-input floating operations
without requiring a second numerical type system. It does **not** yet qualify
arbitrary noncontiguous N-D native grids, WASM native prefilters, or indexed-query performance.
WASM dispatch has representative host-side tests and actual Pyodide wheel tests.
Qualification also requires the full Pyodide suite with typed globally selected;
passing targeted typed tests alone is not that evidence.

On NumPy older than 2, automatic ordinary numerical-expression execution uses the
installed NumPy with bounded Blosc2 chunk/reduction scheduling. Non-numeric
(string/bytes) expressions keep their existing miniexpr-backed route because NumPy
1 has no string ufuncs. Resident preparation, cached
native preparation, native-first dispatch, and typed prefilters decline native
execution. This is an intentional compatibility policy, not a temporary promise
to implement NumPy 1.x value-dependent scalar promotion in the native compiler.
It can reduce acceleration but does not require whole-array decompression.
Explicit `expr.native_kernel()` and `expr.compute(_require_native=True)` remain
available with Menudet's NumPy-2-style numerical contract, even on NumPy 1.x.
Explicit Menudet functions/kernels retain that same contract and acceleration.
Native-only optimization rewrites do not apply to ordinary NumPy 1.x fallback
evaluation; options are validated, but arithmetic follows the installed NumPy.

Non-flat N-D layouts can still be slower than legacy fused execution. The chunk
fallback amortizes native schedule preparation, avoids duplicate persisted edge
reads and unnecessary assignment setup, and uses NumPy for small floating
arithmetic chunks under the default policy. It does not yet reproduce the fused
block pipeline for arbitrary N-D geometry. Profile evidence and remaining gaps
are recorded in the migration plan; performance parity is not claimed for those
layouts.

Use `bench/ndarray/evaluator_migration.py` for an interleaved timing probe and
`bench/ndarray/evaluator_fallbacks.py` for route-aware fallback measurements.
Cross-platform correctness, output/scratch-memory contracts, WASM, and the
recorded performance tradeoffs are still being qualified under the new default.
Some tests intentionally inspect the legacy prefilter or its allocation format;
they are not interchangeable with typed-route storage-contract tests.

The implementation inventory, retirement stages, and acceptance criteria are in
`plans/migrate-typed-expression-evaluator.md` in the repository. Historical
implementation evidence remains in `plans/retire-uniform-expression-evaluator.md`.
