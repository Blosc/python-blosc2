# Retire the uniform-dtype expression evaluator

## Goal and boundary

Ordinary Python/NumPy-style expressions must have one numerical contract,
independent of output storage, slicing, reduction accumulator dtype, JIT, and
WASM. Reuse the existing typed Menudet graph/artifact engine; do not retrofit a
second promotion system into `me_compile`'s uniform-dtype evaluator.

This is distinct from removing explicit compatibility DSL kernels. Those have a
separate language/contract and remain on their existing route until their public
replacement and migration are qualified. Python UDFs remain Python UDFs.

## Inventory

| Entry/path | Existing behavior | Target |
| --- | --- | --- |
| Resident NumPy maps | `native_graph` typed schedules, then chunk fallback | Keep typed schedules; NumPy for unsupported signatures |
| Compressed `.compute()` | Typed prefilter for an admitted subset; otherwise legacy `fast_eval` prefilter | Typed artifact prefilter, then bounded typed chunks |
| Compressed `[:]` / NumPy `out` | Large admitted typed maps; legacy direct-buffer prefilter otherwise | Typed artifact NumPy sink; bounded chunks for other layouts/strides |
| Partial/basic slices, broadcasting, proxy operands | Chunk scheduling plus `evaluate_chunk` | Retain storage scheduling, use typed graph or NumPy only |
| Scalar reductions | Large admitted typed graph reductions; legacy scalar prefilter otherwise | Typed map blocks plus native per-block scalar reducers (`block_reductions.h`) and NumPy final combine; bounded chunk fallback otherwise |
| Axis/index/cumulative reductions | Existing chunk scheduler | Preserve scheduler and result contracts; replace map evaluator only |
| Two-branch `where` | Uniform output dtype can change condition semantics | Typed `select` with independent condition and branch types |
| Cardinality-changing selections | Existing selection scheduler | Keep scheduler, typed/NumPy predicate evaluation |
| Candidate-block bitmap / pruned chunks | Legacy prefilter skips work | Native per-block bitmap skipping and whole-chunk pruning in the Menudet prefilter; bounded chunk fallback otherwise |
| Strings, complex, datetime, float16 | Mixed legacy/NumPy coverage | Explicit NumPy fallback unless typed native support is admitted |
| WASM | More paths reach uniform evaluator; typed compression prefilter is currently disabled | Typed graph interpreter/qualified JIT on resident chunks; no uniform fallback |

Existing building blocks: `native_graph.py`, `graph_plan.py`,
`storage_readers.py`, the shared artifact prefilter in `blosc2_ext.pyx`, and
miniexpr's `dsl_graph*` / `dsl_menudet*` implementation. At inventory time the
native prefilter admitted only 1-D/full-width row slabs. Rebuilt bindings now
also admit consecutive C-order N-D block/chunk runs with tail padding only;
native byte order, no persisted inputs/postfilters, and non-WASM remain required.
It is not an arbitrary N-D replacement yet. Native graph reductions have checked accumulator semantics, so blanket
replacement of NumPy's wrapping reduction scheduler would be incorrect.

## Rollout

1. **Selectable typed-only numerical route.** Add
   `BLOSC2_EXPRESSION_EVALUATOR=typed` (default remains `legacy` during
   qualification). No ordinary expression selected by this route may call
   `_set_pref_expr`; unsupported requests use bounded NumPy chunk evaluation.
   Preserve strong scalars, exact integer literals, output identities, partial
   slices, masks, and no-replay-on-runtime-error behavior. Explicit compatibility
   DSL kernels are not affected by this setting.
2. **Native scheduling integration.** Reuse typed artifact prefilters for
   eligible stored maps and NumPy sinks. Extend generic N-D block geometry,
   persisted/postfiltered inputs, bounded strided outputs, and candidate-block
   pruning only with native admission and lifetime tests. Reuse the typed graph
   compiler; do not specialize typed captures into untyped text.
3. **Qualification.** Run the ordinary suite in both modes; translate tests of
   legacy implementation details into numerical/storage contract tests for the
   new route, retaining DSL compatibility tests. Cover NumPy 1.x/2.x, wheel
   imports, WASM/Pyodide, interpreter/JIT, and thread/scratch budgets. Numerical
   dtype/operator matrices belong in miniexpr; Python tests cover integration.
4. **Performance gates.** Compare resident/compressed `[:]`, `.compute()`, full
   and axis reductions, default and nontrivial N-D layouts, selections, and
   string operations. Measure both elapsed time and peak/scratch memory. Do not
   make the new route default merely because correctness tests pass.
5. **Default switch and deletion.** Once qualified, default to typed, remove
   plain-expression uniform prefilter/reduction branches and their literal/
   `where` guards, then remove the temporary selector. Keep or separately migrate
   explicit DSL compatibility without changing its semantics accidentally.

## Required correctness cases

- `(float32_array > 0.1).sum()` and strongly typed `np.float64(0.1)` comparisons.
- Int64 comparisons near `2**53`, large signed/unsigned captures, negative literals.
- Integer/bool true division and integer-input `hypot`/`nextafter`; map computation
  dtype must not be inferred from the final accumulator/output dtype.
- `where` conditions with mixed types and overflowing narrow-integer arithmetic.
- NumPy versus Blosc2 outputs, C/F/strided destinations, partial/edge chunks,
  broadcasts, sparse selections, and unchanged aliases on preparation failure.
- Once numerical execution starts, propagate errors; never replay through another
  evaluator after partial writes.

The two WASM `int32_array / 2` dtype-policy cases in PR #754 are separate from
native evaluator correctness. Preserve the existing policy until explicitly
changed, and test it rather than marking it as a native numerical failure.

## Initial limits / follow-up

The selectable route is an integration step, not a claim that every native
scheduler has been replaced. In particular, fallback correctness does not qualify
WASM native prefilters, indexed-query performance, or generic N-D native kernels.
PR #754's general dtype-inference fixes are relevant to the default path; the
typed route must infer operation result dtypes independently of uniform output
typing while that PR is pending.

## Initial implementation and evidence

`src/blosc2/expression_evaluator.py` provides the selector, metadata-only binary
result-dtype probes, the typed artifact prefilter adapter, and the typed native
scalar reduction adapter (`compute_typed_reduction`). `fast_eval` and
`reduce_slices` decline the uniform compiler in typed mode; full-domain scalar
reductions (`sum`, `prod`, `min`, `max`, `any`, `all`, and `mean` via `sum`)
execute the typed map per block and reduce directly into a 1-D per-block
accumulator array in C (`src/blosc2/block_reductions.h`) inside
`menudet_menudet_prefilter`, followed by NumPy's final reduction across block
partials. Candidate bitmaps (`_candidate_blocks`) and whole-chunk pruning
(`_prune_chunks`) are handled natively inside the Menudet prefilter as well,
gated by `menudet_prefilter_reductions_available()` so older compiled extensions
fall back cleanly to the bounded chunk path.

The first native scheduling improvement reuses `ndarray.chunks_tile_flat` as a
shared geometry proof. The artifact binding now admits its C-order N-D layouts
without changing numerical compilation or the callback's flat-span ABI. Internal
padding/noncontiguous block grids still decline. The capability queries
`menudet_prefilter_flat_chunks_available` and
`menudet_prefilter_reductions_available` preserve conservative admission with
older editable extensions. Native changes are built/tested in an isolated test
package; the shared conda extension is not overwritten.

Current evidence on macOS/arm64, conda `blosc2`, NumPy 2.5.4, isolated rebuilt
extension:

- Default-mode (`legacy`) full suite: 12,462+ passed; all modified test files
  and doctests pass in both `legacy` and `typed` modes.
- Full-suite `BLOSC2_EXPRESSION_EVALUATOR=typed` qualification: **12,570 passed,
  202 skipped** after updating legacy-specific implementation assertions in
  `tests/ndarray/test_lazyexpr.py`, `tests/ndarray/test_reductions.py`,
  `tests/test_evaluate_numpy_prefilter.py`, `tests/ndarray/test_jit_options.py`,
  and `tests/ndarray/test_safer_jit.py`, and fixing typed-mode edge cases
  (boolean `and`/`or`/`not` rewrites in the NumPy string-expression fallback,
  thread-local `iter_chunks` state for concurrent persisted-array reductions,
  `optimization` policy propagation through `compute_typed_prefilter` /
  `compute_typed_reduction` / `evaluate_chunk`, and `pow(x, y)` lowering in
  `GraphAdapter`).
- Typed-route integration suite (`tests/test_typed_expression_evaluator.py`):
  49 passed with rebuilt bindings, covering simulated WASM dispatch,
  strong/weak scalars, integer floating operations, output identity/layouts,
  unsupported-native fallback, execution-failure non-replay, admitted/rejected
  flat N-D prefilter geometry, native per-block scalar reductions (`sum`, `prod`,
  `min`, `max`, `any`, `all`, `mean`), and native candidate-block/chunk pruning.
- An external 498-check probe of PR #754's expression families (full map,
  `.compute()`, partial slice, with/without host-simulated WASM restrictions)
  matched NumPy; the deliberate int32/int32 division exceptions were excluded.

A dedicated interleaved 2,097,152-element float32 probe, default chunks
`(32, 256, 256)`, blocks `(1, 128, 256)`, 12 threads, seven rounds after warmup:

| Operation | Legacy median | Typed median (initial) | Typed median (flat N-D) | Typed median (native block reductions) |
| --- | ---: | ---: | ---: | ---: |
| `expr[:]` | 1.41 ms | 4.47 ms | 1.63 ms | 1.40 ms |
| `expr.compute()` | 2.05 ms | 4.85 ms | 2.44 ms | 2.02 ms |
| `expr.sum()` | 1.41 ms | 4.15 ms | 4.59 ms | 1.35 ms |
| `expr.sum(axis=0)` | 4.27 ms | 4.21 ms | 4.64 ms | 4.31 ms |

Both the flat N-D map gap and the full scalar reduction gap are now closed in
the isolated rebuilt package (`expr[:]`, `expr.compute()`, `expr.sum()`, and
`expr.sum(axis=0)` are at parity with or slightly faster than `legacy`).

Actual native WASM validation: rebuilt `graph_host`, `graph_corpus`,
`graph_preparation`, `graph_allocation`, and `graph_compiler_allocation` in
miniexpr's existing `build-graph-wasm`; all **10 graph CTests passed** under Node.
This validates the underlying native graph backend, not the Python/Pyodide wheel
or the WASM compression prefilter adapter.

Real Pyodide integration, NumPy 1.x qualification, peak-memory measurements, and
non-flat N-D/persisted native scheduling remain open gates before switching the
default mode from `legacy` to `typed`. String-width metadata still uses the
existing native width query; remove that separate dependency only when the
replacement string contract is qualified.

## Correctness review follow-up

- Explicit boolean accumulators now select native logical OR for `sum(dtype=bool)`
  and logical AND for `prod(dtype=bool)`, with boolean block partials and matching
  final reductions. Values are converted to truth before accumulation, so integer
  cancellation and overflow cannot change the result. Empty identities remain
  False/True respectively.
- NumPy fallback boolean evaluation no longer rewrites `and`/`or`/`not` into
  unconditional eager calls. Admitted graph traversal preserves Python scalar
  short-circuiting and operand-valued results; actual array operands select NumPy
  logical operations. This also avoids injected function names shadowing operands.
  Trusted full-mode syntax outside the admitted graph retains Python evaluation.
- Persisted chunk readers are now owned by an invocation-scoped context, not a
  shared thread-local cursor. Nested evaluations install their own state and
  restore the caller's state; generators are explicitly closed on success/failure
  to stop and join their prefetch workers.

Regression coverage includes scalar control flow in both evaluation policies and
both evaluator modes, boolean accumulator cancellation/overflow/NaN/empty cases,
and nested persisted-array UDF evaluation with successful and failing inner evaluations.

Follow-up validation (conda `blosc2`, installed reduction-capable extension,
NumPy 2.5.4, pytest `-n 0`): legacy full suite **12,534 passed, 96 skipped**;
typed full suite **12,544 passed, 96 skipped**. Ten additional boolean regression
cases were added after legacy-suite collection and also pass in the focused
**100-test** regression run. Ruff lint/format checks and `git diff --check` pass.
