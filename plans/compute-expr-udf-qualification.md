# compute_expr_udf Menudet migration

## Scope

The notebook uses three public, explicitly typed float32-to-bool Menudet
predicates from `bench/ndarray/compute_expr_udf.py`. They specialize and bind
directly; ordinary Numba and Cython UDFs still use `lazyudf`. No source-backed
legacy execution arm was introduced. Historical outputs are preserved in
[compute-expr-udf-legacy.md](compute-expr-udf-legacy.md), not relabeled as Menudet.

Current measurements check exact boolean values, dtype and shape before timing.
Routes that fail are explicitly unqualified, retain the NumPy assertion diagnostic,
and receive no warm timing, ratio or plot bar. This matters at the original size:
the first full attempt exposed a predicate-2 Numba discrepancy at `[0, 1]`
(Numba `False`, float32 NumPy `True`). Different intermediate precision can flip
a predicate at the boundary; the benchmark does not hide this with approximate
boolean equality. The historical notebook had disabled these checks.
Compressed NDArray compute and NumPy getitem are separate measurement contracts,
shown in separate regenerated plot panels. Direct NumExpr/Numba timings reuse
an output buffer, unlike NumPy expressions; those labels explicitly say `out`.
No timing thresholds are added to tests.

## Reproduction and limits

Use the `blosc2` conda interpreter/kernel, launch Jupyter from the repository root
with that root on `PYTHONPATH`, and execute `bench/ndarray/compute_expr_udf.ipynb`.
The notebook writes `compute-expr-udf-results.json` into the kernel's working
directory. It supports these environment overrides before launching the kernel:

- `BLOSC2_BENCH_SIDE`: square side length, default 10000 (100M elements).
- `BLOSC2_BENCH_THREADS`: Blosc/Numba/NumExpr thread count, default `blosc2.nthreads`.
- `BLOSC2_BENCH_REPEAT`: warm repetitions, default 3.

Smoke qualification used side 33, one repeat and four threads. Full execution is
budgeted at ten minutes by the user. Runs use a temporary working directory and
an explicitly selected conda Python kernel; raw results are copied into
`plans/compute-expr-udf-results.json` and full executed outputs into the notebook.

All current inputs and compressed outputs use explicit LZ4 level 5, with the
historical full-size chunks `(1250, 10000)` and blocks `(2, 10000)`. Historical
input codec settings are unknown. Thus grid/shape matching does not establish
an identical historical storage/CPU/backend configuration. Cython has no explicit
OpenMP compile/link flags, as in the original notebook; Numba uses parallel row
loops. Native and compiler caches are not cleared. Decoration/import and the
Cython compilation cell are untimed; Menudet specialization and first execution
are recorded separately from best-of-three warm execution. Validation and NumPy
conversion of compressed results are outside timing. This is not a peak-memory
study. The final results select a separately built/unpacked wheel rather than
the shared editable-install extension.

Native execution reports are read from `_source_execution_report` as well as
`_native_execution_report`; JIT availability is asserted for each requested
Menudet backend. Null execution reports mean unavailable diagnostics, not proof
of a specific expression backend.
For tiled `.compute()`, reports may describe only the final internal evaluation,
not aggregate byte counts across the complete output.

## Profiling and native fixes

The unoptimized full run is retained in `compute-expr-udf-before.json`. Its TCC
compute times were 2.651, 5.688 and 9.246 seconds; CC took 2.547, 5.045 and
8.589 seconds. Thus compiler choice alone did not solve the storage bottleneck.

`compute-expr-udf-profile-before.txt` profiles predicate 1 at `(3163, 3163)`:
0.242 of 0.278 seconds were in `StorageReader.read`, including 0.180 seconds in
`NDArray._take_sparse_normalized`. The full-width row slabs were repeatedly
treated as sparse index lists.

The fixes are in both repositories:

- Python-Blosc2 admits matching full-width row slabs to the existing native
  compressed prefilter. Logical flattening is valid only when chunks and blocks
  span every trailing dimension, first-axis chunks contain whole blocks, and
  the kernel is context-free/lane-local. Other grids, selections, broadcasts,
  block dependencies and context kernels retain their conservative paths.
- Native scheduling uses products of chunk/block extents, correctly sized
  per-worker scratch, and the actual output dtype for the bounded-memory gate.
  NDArray output keeps its shape/grid; the NumPy sink writes directly into
  C-order output with dynamic chunk workers. Single-chunk decoding also uses the
  full logical item count, not only `shape[0]`.
- miniexpr extends the existing TCC arithmetic-strip bridge to same-type floating
  comparisons. Generated arithmetic uses bounded native vector primitives and
  batched bool output, rather than per-lane TCC comparison scaffolding. NaN strips
  retain the exact scalar comparison/FP-status bridge. Masked, mixed-dtype,
  checked and otherwise unsupported trees keep their existing fallback.
- The private arithmetic bridge ABI is bumped to version 2 and generated code
  changes its cache identity. No artifact schema, public kernel ABI, authoring
  contract or numerical policy is changed.

Native builds use `build-dynamic-scheduling`, with local miniexpr override
`/Users/faltet/blosc/miniexpr`, base `6413c800984ff1b8c63953e336b3b8b245b1c3a5`
plus these fixes. C-Blosc2 is the existing local dependency at
`e5086bf5b23cbffc409c55ba19df0bf46b978f06`. Python checkout base is
`361f87f492f7d4e843eec2eb94b1897ae05806f6`. The host is Apple M4 Pro; system CC
is Apple clang 21.0.0. The shared editable-install extensions are not overwritten;
native tests select a separate package containing the rebuilt modules with
conda Python `-S`. Both repositories' changes are needed to reproduce the TCC
improvement. Before upstream native revisions are available, use the local
`FETCHCONTENT_SOURCE_DIR_MINIEXPR` override when building.

`bench/ndarray/profile-menudet-predicates.py` retains an explicitly internal
`SourceKernel`/`_legacy_source_recipe` control. It measures identical current
inputs, storage, requested compiler and thread counts, validates exact predicate
values before timing, and records a legacy JIT probe. Importantly, a requested
compiler is not proof of actual legacy JIT execution: these controls currently
report JIT for predicate 1, but native interpretation for predicates 2 and 3.
This does not restore legacy public authoring or source-recipe persistence.

## Qualified performance and remaining differences

Same-host controls, 100M elements, identical LZ4-5 input/output grid, three warm
samples (best shown, **milliseconds**). All rows pass exact boolean checks:

| Threads | Predicate | Legacy requested TCC | Menudet TCC | Legacy requested CC | Menudet CC |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 42.815 | 24.033 | 10.766 | 21.007 |
| 4 | 2 | 86.416 | 32.478 | 84.147 | 32.098 |
| 4 | 3 | 333.362 | 362.990 | 333.837 | 191.548 |
| 12 | 1 | 19.832 | 11.382 | 8.104 | 10.053 |
| 12 | 2 | 36.099 | 14.663 | 38.246 | 14.348 |
| 12 | 3 | 140.873 | 162.084 | 143.504 | 86.191 |

Final raw controls: `compute-expr-udf-comparison-{tcc,cc}.json` (four threads) and
`compute-expr-udf-comparison-{tcc,cc}-12.json` (12 threads). The separately retained
`compute-expr-udf-prefilter-{tcc,cc}.json` files show the intermediate storage-only
fix before the TCC comparison bridge; do not treat them as final results.

The original multi-second regression is resolved, not merely hidden by choosing
CC. At four threads TCC compute improves roughly 110x / 175x / 25x over the first
migrated notebook run. At 12 threads all three kernels beat the **historical**
saved DSL times, although those historical ratios are not controlled regressions.

Against the **current** internal legacy controls, TCC wins on predicates 1/2;
predicate 3 remains about 9% slower at four threads and 15% slower at 12 threads.
CC wins on predicates 2/3, while predicate 1 remains 24% slower at 12 threads
(about 2x at four threads). Getitem controls are recorded separately: at 12
threads TCC is 13.415 / 18.037 / 197.134 ms versus legacy
38.148 / 55.776 / 176.916 ms; CC is 11.880 / 17.830 / 110.973 ms versus
legacy 18.588 / 60.040 / 187.955 ms. This is comparable overall throughput, not
a claim that every backend/workload is faster or that TCC and CC are equivalent.
NumPy-output dynamic scheduling uses eight workers for eight physical chunks
when twelve are requested; reports state the effective worker count.

The final notebook reruns all baselines at the host's 12-thread default and
regenerates both plot panels; its raw data is `compute-expr-udf-results.json`.
The four-thread notebook data is retained in `compute-expr-udf-results-4.json`.
The old predicate-2 Numba/Cython boundary mismatch remains visible as unqualified,
with no timing/ranking, rather than weakening boolean comparisons. The native
Menudet and legacy control arms pass all three predicates.

## Validation and rebuild

- Rebuilt-wheel tests: 1068 passed, 34 skipped, 584 native-semantics cases
  deselected; includes Menudet, native graph, storage readers and both benchmarks.
- Native `menudet_jit_parity_tcc` and `menudet_jit_parity_cc_gate` pass. Added
  coverage includes both float widths, all six comparisons, variable/literal and
  arithmetic operands, signed zero, infinities, quiet/signaling NaNs, masked
  execution and strip tails; values and FP flags match native interpretation.
- Storage tests cover 2-D/3-D slabs, compressed/uncompressed bool output, partial
  chunk/block tails and zero padding, single chunks, full reads, selection
  fallback and rejection of noncontiguous grids.
- Independent wheel: `blosc2-4.15.0.dev0-cp311-abi3-macosx_26_0_arm64.whl`, SHA256
  `a5f7ea636f8baea3cef1523129f7b00aafdb5ce3a9ed63f061a86d7c98cf15d8`.
  Build used local miniexpr and C-Blosc2 source overrides; this does not qualify
  the old pinned native source without the local changes.
- Builds retain pre-existing Cython pointer-declaration, generated-C qualifier/
  integer-abs, Accelerate deprecation, CMake dependency-policy and duplicate-library
  warnings. The changed native miniexpr sources build without new warnings.

To activate these fixes in the working editable installation, rebuild against
the local native sources (the qualification did **not** overwrite it):

```sh
conda run -n blosc2 python -m pip install --no-build-isolation -e . \
  --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=/Users/faltet/blosc/miniexpr \
  --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_BLOSC2=/Users/faltet/blosc/python-blosc2/build-portable-python/_deps/blosc2-src
```

For the standalone same-host controls after rebuilding:

```sh
conda run -n blosc2 python bench/ndarray/profile-menudet-predicates.py --threads 12 --output tcc.json
conda run -n blosc2 python bench/ndarray/profile-menudet-predicates.py --threads 12 --compiler cc --output cc.json
conda run -n blosc2 python bench/ndarray/profile-menudet-predicates.py --side 3163 --case 1 --route menudet --profile after.pstats --output profile.json
```
