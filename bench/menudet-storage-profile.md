# Compressed Menudet profiling

## Scope and method

Apple M4 Pro, float64 `x + (y - 1)`, default ZSTD level 5 / shuffle,
50 warm samples per timing, actual prepared JIT. The rebuilt storage-reader
extension was imported from an isolated package, not the shared editable install.
Timings below are warm medians, excluding compilation. Each experimental route
checks its result against NumPy before timing. Runs are sequential, not concurrent.

`bench/menudet-storage-profile.py` records cProfile, separately instrumented
reader/span timings, native scratch reports, tile-size sweeps, thread-policy
sweeps, and optional macOS stack sampling. Instrumented medians are diagnostic;
their separate component medians need not sum exactly to a median total.

## Measurements

| Route | 100k values (ms) | 1M values (ms) |
|---|---:|---:|
| Existing compressed LazyExpr | 0.419 | 2.146 |
| Current bounded-reader Menudet | 1.178 | 11.667 |
| Experimental decode + JIT pipeline, 1 worker | 0.738 | 7.979 |
| Experimental decode + JIT pipeline, 2 workers | 0.413 | 4.836 |
| Experimental decode + JIT pipeline, 4 workers | 0.400 | 3.320 |
| Experimental decode + JIT pipeline, 8 workers | 0.400 | 3.139 |

The pipeline uses block-aligned tiles (50,000 / 62,500 values) and bounded waves
of at most one tile task per worker. The 100k case has only two tasks; requesting
4 or 8 workers does not add useful concurrency. Each task decodes both inputs,
executes the native portable block kernel, and copies its result into the output.
Per-worker scratch grows with the worker count; the complete output is still
allocated. This is not whole-input materialization.

**Prototype limitation:** compressed chunks are snapshotted before timing and
decoded using a profiling-only C helper with private reusable contexts. This
omits live SChunk access, lazy/persisted chunk I/O, dictionaries and postfilters.
It is evidence for a scheduling design, not a qualified implementation or a
guaranteed production speedup.

The 1M default layout has one chunk and 64 blocks of 15,625 values. A 65,536-value
tile touches 79 blocks per input across the traversal; 62,500-value aligned tiles
touch 64. In the tile sweep, alignment reduced about 12.15 ms to 10.85 ms; larger
tiles alone were not a solution (a whole-domain reader tile measured 14.55 ms).

A five-second native stack sample of the default 1M route attributed roughly:

- 70% of main-thread samples to the storage reader callback and its descendants;
- 50% to span decoding/context disposal, including 35% in ZSTD decompression;
- 19% to native portable evaluation, almost entirely generated JIT code;
- 5% to native array traversal itself.

These are inclusive sampled stacks, not additive wall-clock measurements. The
sample corroborates instrumentation: spans dominate, followed by reader plumbing.
Per-call context disposal is visible (~6% of samples), but context reuse by itself
did not yield a dramatic gain in the snapshot-reader experiment.

The reason increasing span `nthreads` did not help is structural:
`blosc2_getitem_bytes_ctx` reaches `_blosc_getitem`, which uses
`context->serial_context` and loops over blocks serially. Setting 2/4/12 threads
on the context does not parallelize this path. Splitting only each reader across
Python workers was also weak: about 9.8 / 9.3 / 11.6 ms with 1/2/4 workers.
Parallelizing complete decode-and-JIT tasks was substantially more effective.

A compiler experiment using `ME_DSL_JIT_COMPILER=cc CC=clang` reduced resident
full-block JIT evaluation from about 2.47 ms (TCC) to 1.41 ms, but current
compressed evaluation still took 11.09 ms. Its 8-worker pipeline took 3.16 ms.
Compiler choice is secondary to fixing the compressed scheduling path.
`ME_DSL_JIT_COMPILER=clang` is **not** a valid compiler override; use `cc` plus `CC`.

## Recommended implementation order

1. **Bounded fused decode/JIT scheduling for simple elementwise kernels.** Use a
   small worker pool, block-aligned tiles, and private invocation-owned decode
   contexts. Retain an explicit scratch budget. Four workers are a sensible
   initial target here; eight bring only a small further benefit.
2. **Native/Cython contiguous source callback.** Decode into the supplied native
   gather destination, avoiding Python index arrays, signed-index conversion,
   `np.diff`, temporary value arrays and the final callback memcpy. Keep arbitrary
   indices/masks on the general reader route. A resident contiguous-reader probe
   took ~3.6–3.9 ms versus ~2.5 ms for resident buffers, showing remaining adapter
   overhead even without decompression.
3. **Block-aware tiling and a bounded boundary-block cache.** Avoid repeated
   decoding at tile boundaries. Do not use an unbounded common-multiple tile size
   when operands have incompatible block layouts.
4. **Compiler/vectorization tuning after storage changes.** Keep strict numerical
   semantics; do not introduce fast-math as a shortcut.

Do not prioritize AST-gate caching (only ~0.16 ms/read in cProfile), larger tiles
alone, or a plain switch from spans to NDArray slices (about 20–22 ms at 1M).

Before production parallel dispatch, qualify synchronization, postfilters,
dictionary ownership, deterministic failure selection, FP-status aggregation,
mask participation, logical partitions, and no replay/publication on failure.
The current prototype does not establish those guarantees. Block reductions and
context-dependent recipes must not be retiled by this elementwise scheduler.

## Reproduction

With the intended rebuilt extension selected in the `blosc2` conda environment:

```sh
ME_DSL_JIT_COMPILER=tcc conda run --no-capture-output -n blosc2 python bench/menudet-storage-profile.py --samples 50
```

The optional decoder helper can be built against the same bundled C-Blosc2 build
(macOS example; choose an approved output location):

```sh
conda run -n blosc2 clang -shared -I build-portable-python/_deps/blosc2-src/include bench/menudet-profile-reader.c build-portable-python/_deps/blosc2-build/blosc/libblosc2.9.dylib -Wl,-rpath,"$PWD/build-portable-python/_deps/blosc2-build/blosc" -o /path/to/menudet-profile-reader.dylib
ME_DSL_JIT_COMPILER=tcc conda run --no-capture-output -n blosc2 python bench/menudet-storage-profile.py --samples 50 --reader-helper /path/to/menudet-profile-reader.dylib
```

Use `--items 100000` for the smaller case and `--native-sample /path/to/report`
for macOS stack sampling. JSON output records the actual package/extension paths;
verify them before trusting a run.
