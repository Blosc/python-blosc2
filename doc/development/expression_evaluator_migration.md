# Typed expression evaluator rollout

Ordinary expressions have an experimental typed-only execution route. Enable it
before constructing expressions:

```sh
BLOSC2_EXPRESSION_EVALUATOR=typed python your_program.py
```

The default is currently `legacy`; this temporary selector is intended for
qualification, not permanent API configuration. Explicit compatibility DSL
kernels and Python UDFs keep their existing execution contracts.

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
WASM dispatch is covered by representative host-side tests; those do not replace
an actual Pyodide wheel/runtime test. NumPy 1.x chunk evaluation uses NumPy until
its scalar-promotion compatibility is qualified.

Use `bench/ndarray/evaluator_migration.py` for an interleaved timing probe. The
legacy default must not switch until correctness, output/scratch-memory contracts,
WASM, and performance gates pass. Some existing tests intentionally inspect the
legacy prefilter or its allocation format; they are not interchangeable with
typed-route storage-contract tests.

The implementation inventory, retirement stages, and acceptance criteria are in
`plans/retire-uniform-expression-evaluator.md` in the repository.
