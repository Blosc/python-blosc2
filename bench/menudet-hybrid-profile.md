# Live-storage hybrid Menudet profiling

## What was tested

`menudet-hybrid-profile.py` is a profiling prototype, not production dispatch.
It supports the benchmark's lane-local float64 expression `x + (y - 1)`, with
matching 1-D compressed operand layouts and positive-step slices.

Unlike the earlier parallel prototype, it does **not** snapshot compressed chunks.
Dense execution calls the live SChunk `decompress_chunk(..., dst=buffer)` API,
evaluates the selected lanes with the portable JIT, and moves to the next chunk.
It reuses one full-chunk input buffer per operand. Sparse execution calls the
existing bounded storage-reader path. No output aliases, reductions, broadcasting,
context-dependent kernels, or concurrent chunk readers are implemented here.

Apple M4 Pro, ZSTD level 5 / shuffle, float64, 25 warm samples, actual prepared JIT;
warm times exclude compilation and initial compression. Each route is checked
against NumPy before timing. Tests used live in-memory storage, not persisted or
remote files. The final partial chunk was also checked in a 2,000,003-value smoke
run with 1,000,000-value chunks.

## Dense read results

Times are milliseconds. TCC columns are from one run; the Clang column is from a
separate run with `ME_DSL_JIT_COMPILER=cc CC=clang`. Dense hybrid uses 12 decoder
threads and sequential chunk evaluation, not a Python worker pool.

| Values / layout | LazyExpr (TCC run) | Current bounded Menudet | Previous whole-partition approach | Dense hybrid, TCC | Dense hybrid, Clang |
|---|---:|---:|---:|---:|---:|
| 100k / one chunk | 0.406 | 1.302 | 0.410 | 0.369 | 0.312 |
| 1M / one chunk | 2.167 | 12.364 | 4.209 | 3.748 | 2.542 |
| 8M / eight 1M chunks | 15.722 | 82.582 | 34.812 | 30.236 | 20.965 |

At 100k, four decoder threads were sufficient: 0.370 ms (TCC) / 0.309 ms (Clang).
At 1M, full-chunk decompression scales meaningfully with threads, unlike getitem:

| Decoder threads | Dense hybrid TCC (ms) |
|---:|---:|
| 1 | 7.899 |
| 2 | 5.492 |
| 4 | 4.198 |
| 12 | 3.748 |

The input buffers are 1.6 MB at 100k, and **16 MB at both 1M and 8M**. For the
8M case, whole-partition input materialization would use 128 MB. The hybrid still
allocates the complete 64 MB output and one transient chunk result (up to 8 MB);
codec/thread scratch and compressed storage are additional. These figures are
not a total RSS measurement or a qualified global scratch-budget guarantee.

## Where dense time now goes

### Follow-up: existing DSL kernel prefilter path

The benchmark also measures `@blosc2.dsl_kernel` through
`blosc2.lazyudf(..., dtype=np.float64, jit=True)[:]`. A separate route probe
confirmed `_set_pref_expr` was called with JIT requested and native compilation
reported `compiled=True`, `jit=True`, `ME_COMPILE_SUCCESS`. This is the existing
C-Blosc2 native prefilter path, not a Python UDF callback.

Fresh 25-sample runs (milliseconds); normal per-read prefilter setup is included:

| Values | LazyExpr, TCC run | Bounded Menudet, TCC | Hybrid Menudet, TCC | Existing DSL kernel, TCC | Hybrid Menudet, Clang | Existing DSL kernel, Clang |
|---|---:|---:|---:|---:|---:|---:|
| 100k | 0.407 | 1.188 | 0.372 | 1.032 | 0.270 | 1.900 |
| 1M | 2.297 | 10.400 | 3.943 | 3.162 | 2.591 | 3.240 |
| 8M | 16.557 | 74.308 | 30.388 | 18.649 | 21.378 | 15.568 |

The existing prefilter path is especially competitive on multi-chunk inputs.
Its small-input setup costs are visible, particularly with the system compiler.
These refreshed numbers vary slightly from the original hybrid run above; they
are not measurements taken simultaneously with that earlier table.

### Component profile

Separate instrumentation, 12 decoder threads, TCC:

| Component | 1M (ms) | 8M (ms) |
|---|---:|---:|
| Full-chunk decompression, both operands | 0.937 | 4.773 |
| JIT binding + evaluation | 2.542 | 20.197 |
| Chunk-result publication into full output | 0 | 4.425 |

Separate component medians need not sum to a median total. cProfile confirms the
shift: at 8M roughly 0.399 / 0.591 profiled seconds were portable evaluation,
0.095 seconds decompression, and the remainder mainly result publication.
The dense hybrid avoids the per-lane reader index arrays, Python reader callbacks,
and repeated serial span decoding. The remaining bottleneck is mostly JIT and,
for multi-chunk outputs, publication copying.

Clang reduced 8M JIT binding/evaluation to about 11.456 ms, while decode remained
4.748 ms and publication 4.040 ms. It helps substantially once the reader
bottleneck is removed, although this prototype still trails LazyExpr at 1M/8M.

## Partial-read results and routing

TCC, 1M values, one chunk, comparing forced routes:

| Selection | Bounded readers (ms) | Full touched chunks + selected JIT (ms) |
|---|---:|---:|
| 4,096 contiguous values | 0.169 | 1.328 |
| Dense 90% prefix | 19.550 | 3.456 |
| Every 1,000th value (1,000 selected) | 1.618 | 1.274 |

This supports hybrid routing, but **output density alone is insufficient**. The
strided selection is only 0.1% of values yet touches every compression block.
At 100k, its full-chunk route was also faster (0.189 vs 0.334 ms); at 8M readers
were slightly faster (4.766 vs 5.043 ms). Chunk sizes, touched blocks, decoder
parallelism, memory budget, and fixed scheduling overhead all matter.

The benchmark deliberately starts with an unqualified 50% selected/touched-chunk
capacity rule. It selects the right route for full reads, tiny contiguous reads,
and dense prefixes, but misses the faster dense route for the small strided cases.
Its JSON reports both forced-route timings and estimated touched-block counts so
a better cost model can be evaluated without claiming this threshold is optimal.

## Automatic dispatch implementation

Automatic Menudet lazy reads now implement the conservative routing policy:

- Eligible full reads of one resident 1-D floating input chunk per operand, with
  total physical input capacity at most 32 MiB: direct full-chunk decompression
  into NumPy buffers, then one portable evaluation.
- Matching resident multi-chunk layouts, or single chunks above that input
  budget: a C-Blosc2 prefilter calls the **prepared portable artifact** per valid
  block. It does not recompile the kernel as legacy DSL. The artifact is shared
  read-only; decoder contexts and input buffers are private to each callback.
- Partial selections, mismatched layouts, persisted inputs, dictionary inputs,
  postfilters and unsupported dense layouts: bounded readers. Reduction/context
  recipes and potentially checked scalar conversions retain original execution.

The prefilter limits input block staging to 32 MiB through worker-count selection
and requires its scheduling chunk buffer to fit in another 32 MiB. Output storage,
the returned NumPy array, codec scratch and evaluator scratch are additional.
At 8M values, the report records 3 MB of input block staging and an 8 MB scheduling
buffer, rather than 128 MB of whole-input materialization. Source buffers and the
artifact remain alive until synchronous workers finish; callback state is cleaned
up on success and error. Codec floating flags are isolated, padded lanes are not
evaluated, numerical flags are aggregated, and execution failures never replay.

Final 25-sample warm comparison (milliseconds), sequential compiler runs:

| Values | Selected route | LazyExpr, TCC run | Existing kernel, TCC | Menudet, TCC | Existing kernel, Clang | Menudet, Clang |
|---|---|---:|---:|---:|---:|---:|
| 100k | Single-chunk | 0.413 | 0.970 | 0.437 | 1.403 | 0.362 |
| 1M | Single-chunk | 1.973 | 2.807 | 3.694 | 2.907 | 2.589 |
| 8M | Portable prefilter | 15.530 | 17.876 | 18.204 | 14.528 | 16.326 |

Forced bounded-reader timings in those same runs were 1.103 / 12.075 / 73.875 ms
with TCC and 0.941 / 9.852 / 69.501 ms with Clang. The automatic route therefore
removes most of the dense reader regression. Multi-chunk execution is now close
to the existing kernel path; compiler-run and system-load variability remain.
The plain compressed LazyExpr path is interpreted, despite the compiler labels.

`--dispatch-only` runs these production-route comparisons without the historical
prototype sweeps. Qualification uses an isolated rebuilt extension; the shared
editable installation is intentionally unchanged.

Qualification: 13,198 configured-suite tests passed, with 76 skips and 12,605
deselected; 161 focused dispatch/reader/decorator/native-first tests passed.
The instrumented Cython binding ASan gate passed 65 tests (bundled native archives
were reused without ASan instrumentation, and leak detection was disabled).
Existing Cython/compiler warnings remain; the changes introduced no new warnings.

## Historical conclusion and remaining opportunities

The hybrid is preferable to an unconditional rollback to whole-partition
materialization: it regains parallel decoding and bounds live input buffers by
chunk size rather than array size. It is also substantially faster than the
current always-reader path on dense inputs.

The initial profiling recommendations were:

1. Add conservative dense execution with an explicit memory budget (now
   implemented for resident matching 1-D floating inputs); retain readers for
   localized partial selections and unsupported layouts.
2. Base selection cost on compression blocks touched, not merely output density.
3. Evaluate compiler policy for larger/hot kernels; Clang has higher cold compile
   cost, which this warm-only table does not measure.
4. Write JIT chunk results directly into the final output to remove publication
   copies, and investigate bounded overlap of decode and evaluation.

Before broadening automatic dense eligibility, qualify storage/postfilter
synchronization, dictionaries, persisted storage, logical partitions and checked
error order. The current automatic implementation excludes those unsupported
dense cases; tests cover failure/no-replay behavior, floating status, padding,
layout fallback and concurrent artifact reuse. Neither dense route should be
generalized to reduction/context recipes merely because the shapes match.

## Reproduction

With the rebuilt storage-reader extension selected in the `blosc2` environment:

```sh
ME_DSL_JIT_COMPILER=tcc conda run --no-capture-output -n blosc2 python bench/menudet-hybrid-profile.py --samples 25
ME_DSL_JIT_COMPILER=cc CC=clang conda run --no-capture-output -n blosc2 python bench/menudet-hybrid-profile.py --samples 25
```

JSON includes package/extension paths, layouts, route timings, per-component
instrumentation and cProfile output. Verify imports: the shared editable extension
was left unchanged; these measurements used an isolated rebuilt test package.
