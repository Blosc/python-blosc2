# Native linspace optimization qualification

> Subsequent ablation: the full-write/skip-zeroing optimization and its API have
> been removed entirely. The measurements below describe
> the original all-optimizations-enabled qualification. See the separate
> [zero-fill ablation report](linspace-zero-fill-ablation.md) for its isolated contribution; other optimizations
> and partition defaults remain unchanged.

## Scope

Keep general and constructor-specific partition defaults unchanged. Optimize
generation/compression overhead without imposing larger persistent layouts on
subsequent expressions or reads. No changes to filter/codec selection, numerical
policy or storage format.

Implemented in Python-Blosc2 and the owning native checkouts:

- C-Blosc2: reuse worker-local LZ4 state with the independent-block fast-reset
  compressor when LZ4 is bundled statically. External LZ4 keeps its original
  public entry point. Dictionary streams remain independently decodable.
- C-Blosc2: combine dynamic block claiming with compressed-output reservation
  under the existing job mutex. Regular successful dynamic blocks take one lock
  per iteration instead of separate cancellation/claim/output locks. Error and
  static-scheduling paths keep synchronized cancellation; Windows retains its
  existing scheduler.
- C-Blosc2: add `blosc2_set_prefilter_full_write()` on opaque compression
  contexts. It suppresses initial destination clearing only for callbacks that
  promise to initialize every byte. It does **not** mark compressed output as
  disposable, change callback return semantics, or alter public structure layout.
- Python-Blosc2: opt Menudet prefilters into that contract. Clear partial edge
  blocks before scattering compact values, and explicitly initialize skipped
  blocks/padding. Older system C-Blosc2 retains the original zero-fill path.
- miniexpr: record nested `while` presence during compilation; avoid per-block
  environment/loop-cap lookups for loop-free JIT kernels. Kernels with loops keep
  invocation-time cap changes and existing error behavior.
- Python-Blosc2: prepare/import constructor ramps once rather than exporting a
  validated artifact and immediately importing it again. No shared mutable kernel
  cache or relaxed arithmetic was introduced.

The retained native changes are integrated in the owning repositories and pinned
directly by commit in `CMakeLists.txt`; local backports and patch hooks have been
removed. The full-write items above describe only the historical experiment.

## Measurements

Apple M4 Pro, 8 performance + 4 efficiency cores, 24 GB; conda `blosc2`, Python
3.14.4. Billion-element float64 `(1000, 1000, 1000)` ramps, LZ4 + SHUFFLE +
BYTEDELTA, 12 threads. Seven warm calls following one warmup; separate sequential
processes for baseline and candidate, no concurrent builds/tests during timings.
Timing excludes validation and result destruction. Validation reads seeded random
coordinates and endpoints and checks a slab against independently computed values.

The baseline uses the original installed extension in an isolated snapshot. The
candidate uses an incremental Release build with native source overrides. Both
use **LZ4 1.11.1**. An initially reused build cache selected 1.10.0, giving roughly
half the compression ratio; those mismatched measurements were discarded.

### Final matched-build, interleaved comparison

A pristine reference was additionally compiled from the original Python sources
and the unmodified pinned C-Blosc2 3.6.2 / miniexpr 0.3.1 sources. It was compared
with the independently built patched wheel using the same compiler and dependency
versions. Both persistent worker processes retain their inputs, but only one
computes at a time. Execution order alternates for seven measured pairs per
workload, after discarding one warmup pair. The timing driver validates a
representative coordinate after each result; separate pipeline checks cover
endpoints, seeded coordinates and a slab. The production paired utility validates
34 endpoint/seeded probes per result.

| Operation | Reference median | Patched median | Output ratio, both builds |
| --- | ---: | ---: | ---: |
| `linspace(0, 1, ...)` | **200.98 ms** | **169.65 ms** | 68.7704x |
| `(a + b).compute()` | 1667.97 ms | 1656.47 ms | 22.1777x |
| `((a + b) * c).compute()` | 4876.79 ms | 4863.48 ms | 5.7503x |

Thus the constructor saves **15.6% latency (1.185x throughput)** without changing
partitions or compression ratio. Its median process CPU time drops from 1.847 s
to 1.572 s. The expression differences are below 1%, not claimed as meaningful
speedups or regressions. The final wheel additionally corrects skipped-diagonal
zero-fill diagnostics; that branch is not used by these measured workloads.

This paired check matters: long sequential runs sometimes suggested 5–8%
expression regressions, while other runs showed none. Alternating warmed
processes avoids assigning sustained-load drift to the second build. The final
no-regression conclusion is based on the paired measurements, not a selected
favorable sequential run.

Individual samples and provenance are retained in
`plans/linspace-native-qualification.json`. The production paired benchmark also
checks matching partitions, LZ4 version and thread count before proceeding.

### Preliminary sequential layout comparison (default execution policy)

| Layout / operation | Baseline | Candidate |
| --- | ---: | ---: |
| Default: `linspace(0, 1, ...)` | 198.35 ms | **171.58 ms** |
| Default: `(a + b).compute()` | 1673.62 ms | 1658.63 ms |
| Default: `((a + b) * c).compute()` | 4858.95 ms | 4852.88 ms |
| Default: 10-row slab read | 17.3 µs | 16.8 µs |
| Explicit large: `linspace(0, 1, ...)` | 144.16 ms | 137.99 ms |
| Explicit large: `(a + b).compute()` | 856.36 ms | 855.91 ms |
| Explicit large: `((a + b) * c).compute()` | 3256.25 ms | 3272.17 ms |
| Explicit large: 10-row slab read | 60.7 µs | 62.0 µs |

Default partitions remain `(2, 1000, 1000)` chunks / `(1, 10, 1000)` blocks
(16 MB / 80 KB), with **68.7704x compression before and after**. These sequential
results initially indicated a 13.5% constructor latency reduction; the final
paired comparison above is the stronger default-layout qualification.

The explicit large comparison retains `(4, 1000, 1000)` / `(1, 50, 1000)`
(32 MB / 400 KB), and 91.0524x compression. It is **not** a new default.
It helps these streaming expressions but makes the small slab read roughly
3.5x slower, illustrating why construction time alone is not a layout policy.

## Tests and packaging

- C-Blosc2: 1,734 CTest cases passed, including new full-write/partial-write,
  callback-error, LZ4 state reuse/failure recovery, split-stream, partial-block,
  dictionary, threading and shared-pool checks.
- miniexpr: **458/458 CTest cases passed** on the final complete run, including
  TCC and CC parity gates, invocation-time loop-cap changes and nested loops.
  An initial run hit the outer command's 120-second limit; the final run used
  an adequate outer timeout and completed in 118.62 seconds.
- Incremental isolated package: 635 targeted Python tests passed.
- Independently built wheel from default pinned dependencies: **865 targeted
  Python tests passed**, including constructor semantics, unchanged partitions,
  artifact-import count, physical padding bytes, prefilter failure cleanup,
  storage reopening, expressions, callbacks and compression round trips.
- Wheel build succeeded without native source overrides. Sdist was built and
  checked for both native patches, the patch hook and the compatibility header.
- The shared conda editable installation was not overwritten.

Builds retain pre-existing Cython qualifier/absolute-value/unreachable-code,
Accelerate deprecation, and archive/linker warnings; no new warnings were
observed in changed native code. Qualification here is macOS/ARM64, not a claim
of measured x86/Windows/WASM performance or full optional/network Python coverage.

## Reproduction

`scripts/benchmark_linspace_pipeline.py` accepts an isolated package path and
checks both package and extension locations before measuring:

```sh
conda run --no-capture-output -n blosc2 python -S scripts/benchmark_linspace_pipeline.py \
  --package build_py314/linspace-candidate --layout both --output candidate.json
conda run --no-capture-output -n blosc2 python -S scripts/benchmark_linspace_pipeline.py \
  --package build_py314/linspace-wheel-test --jit on --layout both --output wheel-jit.json
```

For paired measurements (recommended when comparing small performance changes):

```sh
conda run --no-capture-output -n blosc2 python -S scripts/benchmark_linspace_interleaved.py \
  --reference build_py314/linspace-reference \
  --candidate build_py314/linspace-final-wheel-test --output paired.json
```

The scripts record individual wall/process-CPU samples, partitions, compression
ratio and import provenance. The pipeline benchmark also times constructing all
three operands and records a lifecycle sum of the measured stage medians; it is
not an independently timed end-to-end latency. Do not run other builds/tests
concurrently with either benchmark.
