# M5/M6 — logical arrays and native-required graph integration

Reference: NumPy 2.5.3. Local platform: macOS arm64. Date: 2026-10-09.

## Implemented acceptance subset

M5 supplies a versioned native logical-array descriptor/scheduler with checked
allocation bounds, signed strides, byte order, broadcasting (including 0-D and
empty arrays), caller-owned C-order output, disjoint-output enforcement, bounded
tile gathers and an aligned contiguous zero-copy path. Descriptor shape queries
precede output allocation. Native metadata-only reshape, transpose and basic
slice APIs preserve borrowed ownership and reject invalid geometry.

Six logical reductions support normalized selected axes, keepdims, accumulator
dtype, initial values, participating broadcast masks, empty identities/errors,
modular integer accumulation and tile-independent serial floating grouping.
They remain distinct from persisted explicit block-scalar recipes. Mean/variance,
arg/cumulative operations, general indexing and mutable views are deferred.

M6 supplies an explicit `_require_native=True` LazyExpr path: eligible safe graphs
lower to cached immutable portable 1.1 plans, arithmetic/functions fuse, and the
native scheduler owns logical traversal/reduction. Python performs metadata and
storage reads, not numerical fallback. Plans key source/signatures/captures and
never retain operand results. Mutation/repeated execution, basic partial reads,
standalone artifact deployment, root reductions and safe portable persistence are
tested. Table row/partition selection, nested lazy/proxy/remote operands, aliases,
reduction partial reads and acceleration overrides reject before writes.

The useful subset runs with NumExpr import deliberately unavailable, covering
construction, metadata, native-required evaluation/reduction, artifact export/import
and safe persisted recipe reload. Tests also fail on calls to NumExpr/Python graph
evaluation. NumExpr's packaging requirement and default backend selection remain
unchanged: optional packaging for every route needs a wider clean-install audit.

Detailed native API/ownership/accuracy/divergence contract:
`../miniexpr/doc/native-arrays-1.1.md`. Python reference:
`doc/reference/portable_dsl.rst`. Standalone C deployment/regression host:
`../miniexpr/tests/numpy-compat/arrays.c`.

## Validation

- **408 array tests + 9 native graph tests pass.** The numerical matrix spans
  all eleven numeric dtypes, six reductions, five axis choices, keepdims and three
  tile sizes. Layout tests include C/F, reversed, stepped, endian-swapped and
  unaligned arrays. Tests cover bounds rejection, normalization reporting,
  masked exceptional values, zero-copy, initial/empty behavior and modular sums.
- Full Python integration checkpoint: **12,703 passed, 55 skipped**.
- Complete native checkpoint: **426 passed**.
- Native C host, shared corpora and examples: **10 sanitizer checks** and
  **10 standalone Node/WASM checks passed**. The WASM build's new 32-bit stride
  comparison warning was corrected and final rebuilds emitted no new warnings.
- Native interpreter fallback remains explicit: no eligible portable SIMD/JIT
  implementation is claimed. Cython calls release the GIL around native scheduling;
  this is not free-threaded-build or same-storage mutation qualification.
- Floating serial sums meet a seeded standard forward-error bound against
  `math.fsum` and retain bytes across tile sizes. Universal NumPy pairwise bit
  equality is deliberately not the reduction contract.

## Performance evidence and boundaries

`tools/menudet_graph_bench.py` records isolated subprocess comparisons;
`plans/menudet-m6-benchmarks.json` contains 120 affine baseline rows over tiny
(8-element) and large (262,144-element) float32/float64/int64 arrays, C/strided
layouts, scalar broadcasting and reductions. Routes are direct native-array API,
native-required graph, NumPy, one-thread NumExpr and existing Blosc2. Subprocess
order alternates; compilation, setup, first evaluation, five-run warm median,
thread count, iterator buffer size and process peak RSS are distinct.
Supplemental isolated measurements cover mixed types, partial reads, compressed
operands, persisted portable recipes and four independent concurrent native calls.
Persisted recipes retain their existing block scheduler; they are not relabeled as
native logical-array scheduling.

For large C-order affine arrays, observed direct/integrated warm medians are about
**495–520 ms**, versus **0.039–0.230 ms** for NumPy/NumExpr elementwise controls and
**0.47–0.66 ms** for existing Blosc2. Logical sums are similarly about **503–520 ms**
versus **0.071–0.268 ms** for controls. This confirms that frontend integration
adds little to the existing strict interpreter bottleneck; it does **not** establish
a native speedup or performance-release readiness. No universal arbitrary slowdown
threshold was imposed. Workload-specific performance budgets remain a scope/release
decision after this baseline, not a silently assumed agreement.

Zero-copy/gather/scratch and frontend materialization counts are separate. In this
first graph subset, compressed operands may still materialize in full; end-to-end
bounded-RSS scheduling and comprehensive interpreter allocation bounds remain
unproven. Process RSS includes imports, outputs and reference allocations and must
not be equated with native temporary-buffer size.

## CI qualification

Dependency pins remain unchanged. The ordinary Python matrix intentionally still
tests the pinned runtime; newly required capabilities are qualified separately.
`.github/workflows/menudet-paired.yml` builds a selected native checkout against
this Python revision on Linux/macOS/Windows, sets `MENUDET_REQUIRE_ARRAY_RUNTIME=1`
and explicitly selects the shared corpora. It does not hide missing capabilities
behind skips. It uploads exact paired SHAs and native logs. Native branch CI now
also runs on the experimental `numpy-compat` branch.

Remote qualification results will be recorded after the published branches run.
The previous green Python runs at `cf6f25c` and native main runs are not evidence
for these M5/M6 changes. No release/default acceleration or optional-dependency
packaging claim follows from this implementation checkpoint.

## Reproduction

Use the `blosc2` conda environment and explicitly rebuild against the sibling
native checkout. Then:

```sh
MENUDET_REQUIRE_ARRAY_RUNTIME=1 conda run -n blosc2 pytest \
  tests/test_menudet_arrays.py tests/test_menudet_native_graph.py -q -n 0
conda run -n blosc2 ctest --test-dir "$NATIVE_BUILD" --output-on-failure
conda run -n blosc2 python -m tools.menudet_graph_bench \
  --report plans/menudet-m6-benchmarks.json
MENUDET_WORK_DIR="$APPROVED_WORK_DIR" conda run -n blosc2 python \
  -m tools.menudet_graph_bench --report plans/menudet-m6-benchmarks.json --supplement-existing
```
