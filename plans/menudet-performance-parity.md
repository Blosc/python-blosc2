# Menudet/full-DSL performance qualification

Status: local campaign in progress; optimization changes below are not yet qualified.

## Reproduction and scope

The native harness is `../miniexpr/bench/benchmark_profile_parity.c`, driven by
`../miniexpr/bench/profile_parity.py`. The defined matrix has 1,176 recipes,
signatures, distributions and mask variants across Boolean, all standard integer
widths, float32/64, arithmetic, comparisons, logical/bitwise operators, unary math,
casts, mixed types, captures, control flow, reductions, coordinates and libm.
Backends: TCC, Clang, GCC 16 and native interpretation. Sizes: 1K, 64K, 1M, 10M;
interpreter/exception controls are capped at 64K, and large unavailable-JIT pairs
are explicitly reported without timing. Raw output is JSONL with individual
samples, compile/first/warm/allocation timings, JIT availability, semantic
classification, native bridge counts and exact reproduction commands. Generated
C and manifests are retained per job. Normal timings alternate engine order;
first-call timings are diagnostic, always Menudet then legacy.

Local work directory:
`/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/`.
The baseline binary is `profile-parity-baseline`; the baseline results are under
`parity-full-baseline/`. The binary contains native revision `00eb5fd` and is not
rebuilt when optimization sources change. No shared editable-install extension
has been overwritten. Python integration checks use the existing isolated
`integer-test-package` via `isolated_menudet.py` and conda environment `blosc2-3.15`.

## Measurement boundaries

- The profiles have different numerical contracts; timings are not a universal
  equivalence claim. Differences, masks, captures and backend fallback are separate.
- Each worker first verifies up to 1,024 lanes against the Menudet interpreter,
  including IEEE flags. Existing native conformance suites remain authoritative.
- Masked work cannot be compared directly to the unmasked legacy API.
- Threshold: ratio >1.25 and absolute difference >5 us; independent reruns must
  confirm important gaps before attributing them to code generation.
- One host; no cross-platform performance qualification or hard real-time guarantee.

## Implemented optimizations (full performance rerun pending)

- Skip validation scans entirely when no participating mask is supplied.
- Preload input pointers and split generated masked/unmasked entry paths so byte
  stores cannot force pointer-table reloads or prevent unmasked vectorization.
- Carry same-width modular integer expression trees as unsigned bits, avoiding
  repeated signed-representation materialization.
- Inline strong Boolean bitwise operations and guarded NumPy integer shifts/divmod.
- Avoid math-name registry searches for known interpreter arithmetic nodes.
- Prepared semantic callbacks for min/max, round/sign/square, pi-scaled trig,
  logaddexp and exp10; direct fma/ldexp/predicate callbacks.
- Resident partial/stepped NumPy reads evaluate selected safe lane-local recipes
  rather than complete original groups; integer comparison outputs can use workers.

## Validation and current findings

- Rebuilt native libraries and isolated extension successfully. Build warnings
  are existing Accelerate deprecations, generated-C qualifiers/abs and a duplicate
  static library; no new warnings in the changed native sources.
- Expanded TCC native parity suite passes, including all integer widths,
  boundary/negative/oversized shifts, divmod, bitwise/Boolean and prepared math
  values, masks, errors and IEEE status.
- Expanded Clang compact native suite passes. It first caught folded empty
  scalar-reduction `0/0` losing invalid status; reduction-result volatile loads
  now preserve that status without adding volatile work inside the reduction loop.
- Python resident/decorator integration: 84 passed using conda `blosc2` with
  explicitly selected isolated package and rebuilt extension.
- Targeted optimized smoke: 324 result rows, no validation failures; remaining
  semantic/unsupported legacy controls stay separate from parity rankings.
- Preliminary 64K-lane math improvements are substantial, but **not final**:
  fma about 30–93x, ldexp about 196–199x and sinpi about 35–62x faster than the
  frozen baseline on selected TCC/Clang cases. Small wide-type arithmetic timings
  also show regressions/noise that require the scheduled large-size rerun; no
  blanket performance-parity claim is warranted.

The serial follow-up coordinator is `finish_parity_campaign.py` in the local work
directory. It waits for the frozen baseline, then runs the full optimized matrix,
native controls, broader Python tests, paired baseline/optimized application
matrix and three-process confirmations. It must finish before final sign-off.

Final measurements, validation and remaining limitations will replace this status
when the campaign finishes.

## Arithmetic regression investigation and public evaluate dispatch

Fresh-process confirmation (`arithmetic-regression.json` in the work directory):
three processes per version/backend/signature, alternating baseline/candidate
order and seven warm samples per process. At 10M lanes the int64 chain is
178.041 -> 17.640 ms with TCC (10.09x); int8 is 130.171 -> 17.554 ms (7.42x).
Clang int8 is 5.411 -> 0.668 ms (8.10x). The apparent 4.4x Clang int64 regression
at 64K does not reproduce: the separate-process residual is 24% at 64K and 5%
at 10M. Removing restrict or the masked function does not consistently fix it.

To isolate kernels from process/allocation/cache effects,
`check_arithmetic_same_buffers.py` compiles the saved before/after generated C
into separate libraries and alternates 21 calls in one process over exactly the
same inputs and destination. Both aligned and offset views are checked. At 10M:

| dtype | alignment | before ms | after ms | before/after |
|---|---|---:|---:|---:|
| int64 | base | 2.876 | 2.883 | 0.997x |
| int64 | one-element offset | 2.932 | 2.922 | 1.004x |
| float64 | base | 2.233 | 2.220 | 1.006x |
| float64 | one-element offset | 2.321 | 2.298 | 1.010x |

64K same-buffer results are within 1%; float64 hot vector-loop instructions are
identical. No reproducible kernel-level regression remains in this diagnostic;
separate-process timings should not be presented as an arithmetic-lowering bug.

`blosc2.evaluate()` previously excluded math calls and NumPy-input explicit
destinations from automatic native dispatch. Reviewed positional numeric calls
now enter the same native graph/JIT pipeline. Explicit NumPy outputs retain
identity (with a reported copy on the general resident route). New bounded stored
output maps cover integer and mixed NumPy/Blosc2 inputs, including multidimensional
broadcasting, while preserving the existing fast compressed floating prefilter.
Output admission is metadata-only; execution failures are not caught/replayed.
Unsupported cases and backend/accuracy overrides keep existing fallback policy.

Validation: 291 passed across public evaluate routes, NumPy prefilter/recipe-cache,
native-first maps/sources and resident execution. Numeric division, floor division,
modulo, power and shifts also enter native dispatch with representative API tests.
The new tests explicitly forbid legacy numerical fallback
for the representative 3 input kinds x 3 output kinds x 2 recipe families.

## Compressed int16 output outlier

Reproduced `bench/menudet-engines.py --dtype int16 --codec lz4 --clevel 3
--items 8000000`: Menudet compute took 172.5 ms under cProfile. The general
partition-preserving fallback evaluated the original 8M group once per 1M output
chunk (64M lanes). Eight evaluate_block calls consumed 68 ms, unique/hash 31 ms,
input indexing/decompression 28 ms, plus selection/index construction and copies.
The integer literal excluded the otherwise lane-local recipe from bounded storage
admission, and both Python/Cython prefilter gates only admitted float inputs.

Reused the resident modular-integer safety proof for storage admission and opened
numeric prefilter metadata gates accordingly. Checked weak subtrees, casts,
reductions, context dependence and unsupported grids retain fallback. Existing
admission for literal-free recipes is preserved. Integer prefilters are now enabled
for eligible compressed Menudet and native-graph output; they are no longer
universally float-only.

After rebuilding the isolated extension: 512 blocks evaluated once, 16 input chunk
fetches, 750KB worker input scratch. The full benchmark with five warm samples
reported Menudet 2.761 ms (NumPy 6.791, LazyExpr-NumPy 1.531, compressed LazyExpr
6.616, evaluate 1.795 ms; numexpr correctly marked non-parity). An interleaved
same-process control using identical rebuilt kernels and only toggling the old/new
integer dispatch gate measured medians 173.868 -> 4.513 ms (38.52x), five samples
per route after two warmups. These are distinct experiments, not mixed into one
speedup ratio. Profile and comparison scripts/results are in the local work folder
as `profile_int16_stored.*` and `compare_int16_dispatch.*`.

Tests cover int16 wraparound and uint64 compressed output, requested codec/level,
input chunk fetch counts and rejection of full-group/NumPy staging paths. The
Cython gate change requires rebuilding the extension; shared installed extensions
were not overwritten during this qualification.

## LazyExpr/evaluate parity for int16 arrays

Resident profiling reproduced LazyExpr/evaluate ~6.0 ms versus Menudet ~1.5 ms
at 8M lanes, despite all routes already using the same 12-worker resident
scheduler. The graph transports `1` as an immutable weak capture; its narrowed
conversion prevented fusion of the whole modular tree. Generated TCC C retained
three signed representation helper calls per lane instead of one raw-bit store.
Native lowering now absorbs validated, representable immutable integer capture
leaves into the same-width unsigned tree. Checked/computed weak subtrees and
out-of-range capture participation/error behavior are preserved. Cache generation
is bumped. Native tests cover all integer widths, capture bounds, masks and empty
calls, and assert fused source for eligible cases.

At 100M compressed lanes the graph reader also built/validated 64-bit gather
indices for contiguous input spans. Increasing tile size did not help (37–40 ms
after fusion) and increased scratch. The new automatic large-integer chunk-map
route avoids those indices entirely: isolated single-thread decoders fill reusable
worker scratch; the same artifact writes disjoint NumPy output regions. Scratch
is capped at 32 MiB total, limiting this case to eight workers. Only a metadata-
proven lane-local integer recipe on compatible 1-D layouts is admitted; smaller
arrays, checked expressions and other layouts retain existing routes. All workers
join on failure and no execution is replayed. Decoder floating environments are
isolated and restored even on errors.

Representative after-run: int16, LZ4 level 3, 12 configured threads, five warm
samples, same benchmark as the user (milliseconds):

| items | inputs | LazyExpr-NumPy | evaluate | Menudet |
|---|---|---:|---:|---:|
| 100K | NumPy | 0.175 | 0.179 | 0.121 |
| 1M | NumPy | 0.459 | 0.461 | 0.370 |
| 8M | NumPy | 1.853 | 1.928 | 1.681 |
| 100M | NumPy | 18.504 | 18.136 | 18.078 |
| 8M | Blosc2 | 1.340 | 1.697 | 2.827 |
| 100M | Blosc2 | 21.256 | 20.815 | 30.105 |

For NumPy inputs, all three columns return NumPy. For Blosc2 inputs, Menudet's
benchmark column returns compressed Blosc2 output while the other two return
NumPy, so those columns do not have identical output work. Small resident calls
retain ~50–100 us of graph frontend/admission overhead versus a pre-specialized
Menudet kernel; there is no claim of parity at every size/signature.

Diagnostics are retained locally as `profile_evaluate_gap.py`,
`probe_compressed_graph_tiles.py` and `probe_direct_chunk_graph.py`. Tests cover
mixed resident/compressed inputs, integer wraparound, explicit output identity,
scratch budget and worker joining on decoder failures. Native/Cython changes
require rebuilding against the local miniexpr checkout.

Validation for this iteration: 339 Python tests passed (native maps, native-first
selection, evaluate destinations/cache/prefilter, weak captures, storage readers,
resident execution and benchmark correctness); expanded native TCC full and Clang
compact parity suites passed. Lint/format checks passed. Builds retain existing
generated-C qualifier/abs, Cython shared-pointer declaration and duplicate-library
warnings; no new warnings were introduced in the changed native lowering.

### Wide-integer chunk scratch regression

The original whole-chunk scratch strategy capped a 100M int64 map with two
1M-item compressed inputs at two workers: 16,000,000 scratch bytes per worker
under the 32 MiB total budget. Profiling reproduced LazyExpr-NumPy 75.75 ms and
evaluate 75.96 ms, with the execution report explicitly showing two threads.

Scheduling now budgets one physical block per worker first, then selects a
block-aligned sub-chunk batch that fits the unchanged total scratch budget.
For the standard layout this retains 12 workers with 171,875-item batches and
33,000,000 scratch bytes. The decoder accepts bounded aligned offsets/counts;
non-aligned spans reject so the native multi-block destination shortcut cannot
overwrite leading decoded data. Tail batches and integer wraparound are tested.

After-run (int64, LZ4 level 1, system compiler JIT, five warm samples):

| items | inputs | LazyExpr-NumPy ms | evaluate ms | Menudet ms |
|---|---|---:|---:|---:|
| 8M | Blosc2 | 4.089 | 4.041 | 3.681 |
| 100M | Blosc2 | 36.264 | 35.128 | 31.154 |

Menudet returns compressed output in these rows; the other two return NumPy.
The reproduced wide-integer worker-cap regression is removed without raising
scratch memory limits. A companion int16/cc 100M compressed-input run remained
fast (LazyExpr-NumPy 10.285 ms, evaluate 10.136 ms, compressed Menudet 15.496 ms).

### Compression-driven direct NumPy sink investigation

An opt-in prototype extends `PortableArtifactHandle.evaluate_prefilter` with
`numpy_out=`. The existing compression workers, input decoders, artifact call,
FP isolation, and error merge are reused. The callback evaluates straight into
the final NumPy slice. Disposable-output callbacks return nonzero to skip the
filter pipeline; C-Blosc2's existing clevel>0 disposable shortcut emits tiny
zero-run placeholder chunks instead of compressing or retaining result bytes.
No C-Blosc2 changes are required. Blosc2 output keeps its original fused
evaluate/filter/compress path. At this initial stage production dispatch was
unchanged; the coarse-worker implementation below supersedes this prototype.

This removes the old temporary-uncompressed-Blosc2-result/materialization path,
but still uses a small scheduling container and a chunk-sized dummy input
allocation. It is not literally free of all temporary storage. Report fields
identify the destination and placeholder storage size. Integration tests cover
interpreter/JIT execution, reversed binding order, int64 wraparound, output
identity/reuse, canaries around a partial tail, no Blosc2 materialization, no
output zero-fill, and destination validation.

Correctness-gated `bench/menudet-engines.py` measurements (LZ4 level 1, cc JIT,
12 threads, five warm samples; experimental dispatch installed only by the
local `opencode/bench_numpy_prefilter.py` wrapper):

| dtype/items | current LazyExpr-NumPy/evaluate ms | prototype LazyExpr-NumPy/evaluate ms |
|---|---:|---:|
| int64 / 8M | 5.039 / 4.807 | 4.032 / 4.084 |
| int64 / 100M | 33.942 / 35.384 | 62.906 / 62.498 |
| int16 / 100M | earlier 10.285 / 10.136 | 19.500 / 19.884 |

A separate same-input, shuffled-order diagnostic (100M int64 arange, seven
samples, cc JIT) measured direct prefilter fresh output 66.75 ms, reused output
28.19 ms, and the old prefilter/materialization route 112.74 ms. Its current
LazyExpr path measured 48.57 ms; this diagnostic retains more large buffers and
is not interchangeable with the benchmark above. Fresh allocation/first-touch
is a major sensitivity, not yet a proven sole cause. The compression driver
also joins workers once per input chunk, unlike the current coarse partition
of the whole domain; scheduling/locality needs further investigation before
switching dispatch or removing either existing route.

Validation: 136 tests passed across storage readers, Menudet dispatch, native
graph maps, and LazyExpr output dispatch. Builds/imports used the isolated test
package, not the shared editable extension.

### Shared prefilter with coarse chunk workers

The direct NumPy sink now uses the production prefilter callback for eligible
native-first elementwise maps, including mixed NumPy/Blosc2 inputs. NumPy
operands are borrowed directly; compressed operands use the same reusable
decoders and bounded scratch as compressed-output execution. The standalone
integer chunk-map evaluator and its `_portable_chunk_span` binding are removed.
The old temporary-uncompressed-result/materialization branch is also removed.
Persisted, padded-grid, postfiltered, context-dependent, reduction, and other
unsupported cases retain their general bounded-reader/semantic fallbacks.

For sufficient chunk counts, persistent outer workers own contiguous chunk
ranges and private single-threaded compression contexts. Few large chunks
instead use the same callback with C-Blosc2's internal block workers. A nominal
read-only input buffer is shared across outer workers; the prefilter does not
read it. Each private scheduling container has only one chunk, irrespective of
the source domain. Result chunks are not stored: the driver's tiny disposable
chunks are discarded immediately. Compressed output keeps its original tightly
fused evaluate/filter/compress pipeline and requested output block grid.

The first coarse-worker attempt serialized at the Python GIL because
`SChunk.update_data` does not release it by default. NumPy scheduling now uses
the existing GIL-releasing `_prefilter_data` adapter. No global GIL-release
setting is changed and no C-Blosc2 modification is needed. All submitted workers
are joined before returning or raising; errors are inspected in range order.

Tiny source blocks can be grouped into aligned prefilter batches up to 4096
items, respecting chunk divisibility and the shared application-buffer budget.
The callback validates the original input block headers and decodes each batch
as an aligned span. This avoids a per-physical-block callback/placeholder cost
without duplicating decoding or kernel execution. NumPy output padding is never
written. Compressed-output block sizes and padding behavior remain unchanged.

Final correctness-gated benchmark (LZ4 level 1, cc JIT, 12 threads, seven warm
samples, fresh output allocation included):

| items / compressed int64 inputs | LazyExpr-NumPy ms | evaluate ms | compressed Menudet ms |
|---|---:|---:|---:|
| 2M | 1.246 | 1.214 | 1.058 |
| 8M | 3.123 | 3.119 | 3.704 |
| 100M | 23.132 | 23.576 | 31.071 |

The preceding separate-path 100M baseline was 33.942/35.384 ms for
LazyExpr-NumPy/evaluate. These are separate benchmark invocations, not an
interleaved same-buffer microbenchmark. Reported 100M int64 application-owned
temporary buffers are 11,028,032 bytes: 8,000,000 shared driver bytes,
3,000,000 reusable input bytes, and 28,032 private placeholder-buffer bytes.
This excludes the required 800 MB final output and native context internals;
the application buffers remain within the unchanged 32 MiB cap. Each invocation
fetches 200 input chunks and reuses 24 decoders, with no gather indices, output
copying, or output zero-fill.

Companion measurements:

- int16/cc, LZ4 level 1, 100M: 9.912/10.534 ms for LazyExpr-NumPy/evaluate,
  versus the prior roughly 10 ms path; no wide-integer improvement is claimed
  for this already-fast narrow-integer case.
- float64/cc, LZ4 level 1, 100M: 25.123/23.908 ms.
- int16/TCC, LZ4 level 1, 100M: 20.241/20.405 ms; correctness passed.
- int32/cc, Zstd level 5, 100M: 16.519/16.414 ms. Extending the measured
  high-thread crossover to eligible int32/int64 maps also reduces 8M int32
  from 9.016/9.380 to 2.101/2.316 ms; reductions keep their previous floor.
- A probe overriding source blocks to 128 items and chunks to 1,048,576 items
  at 8M float64 measured 112.136/112.670 ms before grouping, then
  5.359/5.539 ms after grouping. The retained bounded-reader baseline measured
  45.454/44.446 ms. This is an opt-in layout probe, not the standard benchmark
  layout; compressed output intentionally retains its tiny block grid.

Diagnostics and reproduction drivers are retained under `opencode/`:
`prefilter-coarse-int64-cc.txt`, `prefilter-coarse-int32-zstd-cc.txt`,
`prefilter-coarse-report.txt`, and `bench_prefilter_layout.py`.

Validation: 840 targeted tests passed across native graph lowering, sources,
stages, math/weak-capture semantics, storage readers, resident execution,
Menudet/public evaluate dispatch, output binding, caching, and benchmarks.
Binding coverage includes mixed source order, NumPy output alias rejection,
tail canaries, tiny chunks, interpreter/JIT operation, failure recovery, and
joining outstanding workers. Ruff lint/format and diff whitespace checks pass.
Builds/imports use the isolated test package; the shared editable installation
has not been overwritten and must be rebuilt before testing this implementation.

### Resident inputs / uncompressed stored-output allocation profile

The reproducible 100M int64 NumPy-input to Blosc2-output anomaly (LZ4 level 0
around 63 ms versus level 1 around 33 ms) was profiled separately. This default
implicit stored-output route still uses `slices_eval` and 50 NumExpr calls;
it does not enter the shared native prefilter. Both levels choose identical
2,000,000-item chunks and 16,000-item blocks. Their output payload/storage sizes
are 800,003,200 bytes at level 0 and 5,673,032 bytes at level 1.

Shuffled warm timings on the same operands reproduce 62.27 versus 33.56 ms.
The Python profile misleadingly charges the excess to NumExpr `re_evaluate`
(roughly 49 versus 16 ms), while chunk assignment takes roughly 10--14 ms.
Per-phase `getrusage(RUSAGE_SELF).ru_minflt` counters establish the actual
allocation/first-touch sensitivity:

| control | level 0 warm total | level 1 warm total | level 0 minor faults |
|---|---:|---:|---:|
| fresh output, ordinary NumExpr buffers | 63.5--64.0 ms | 34.3--34.5 ms | about 34,200, in evaluation |
| fresh output, one reused NumExpr buffer | 60.8--61.3 ms | 34.9--35.9 ms | about 33,200, in chunk storage |
| reused Blosc2 destination, original fallback held constant | 31.5--33.1 ms | 33.0--33.5 ms | 0--1 |

Level 1 incurs fewer than 100 warm faults in these runs. NumExpr result pointers
cycle through 50 distinct buffers per level-0 invocation versus two at level 1;
with explicit scratch reuse both levels use one buffer. Reusing only that
scratch moves the page faults and timing to the storage phase, rather than
eliminating them. Reusing the stored destination recycles already-backed raw
output pages and eliminates the warm allocation penalty. The latter control
explicitly disables native stored-output admission to avoid comparing different
evaluators when an `out=` argument is supplied.

This is a real fresh-output allocation cost, not slower int64 arithmetic, a
chunk-size change, or a compression-backend regression. Compression keeps this
particular output about 141 times smaller and avoids most of the physical-page
commit/first-touch work and allocator pressure. Once the destination's pages
are reusable, level 0 is slightly faster or comparable, as expected. Merely
reusing a temporary buffer is therefore not a fix. Extending prefilter-driven
stored-output execution to all-resident implicit destinations is a separate
dispatch opportunity; it was not changed during this profiling task.

Diagnostics under `opencode/`: `profile_numpy_stored_levels.py`,
`profile_numpy_stored_levels.out`, `profile_stored_faults.py`,
`stored-faults-fresh.txt`, `stored-faults-reuse-result.txt`, and
`stored-faults-reuse-output.txt`. Initial qualification checked the complete
result at both levels; phase-counter runs check sampled values without creating
an additional full-size NumPy expected result between timed invocations.
