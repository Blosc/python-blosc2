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
