# Clang Menudet compressed-output profile

## Finding

The extra output **staging/readback** is expensive, but there is no output codec
compression to blame: the temporary result uses `clevel=0` and occupies
64,000,512 stored bytes for 64,000,000 result bytes. Input decoding and especially
decoder-context disposal are also substantial. The original hypothesis is
therefore only partly right: bypassing temporary output storage is promising,
but input handling must not be dismissed.

## Workload and wall timings

8M float64 values, `x + (y - 1)`, two live resident compressed inputs, eight 1M
chunks each, 15,625-value blocks, 12 threads, Clang portable JIT. Inputs use
ZSTD level 5 and shuffle, with stored sizes 2,399,062 / 2,334,768 bytes.
Compilation, input creation and correctness checks are excluded from timing.

The uninstrumented extension measured 17.06 ms compressed versus 2.81 ms resident
(50 warm samples); standalone full-chunk input decoding took 4.35 ms. Timings
vary from the earlier 13.24/2.66 ms comparison, but the gap persists.

A separate isolated extension adds monotonic native timestamps. It reuses the
same native archives and production scheduling. Four modes are interleaved in a
deterministically shuffled order for 100 warm samples:

| Profiling mode | End-to-end ms | Prefilter/chunk updates ms | NumPy output readback ms | Output storage release ms |
|---|---:|---:|---:|---:|
| Normal | 19.20 | 11.66 | 5.61 | 1.37 |
| Skip evaluation | 18.75 | 10.70 | 5.61 | 1.41 |
| Skip input decoding; evaluate zero buffers | 12.90 | 5.16 | 5.66 | 1.31 |
| Skip both; retain allocation/output scheduling | 11.28 | 3.95 | 5.57 | 1.30 |

Normal output allocation and prefilter setup measure about 0.075 and 0.005 ms.
Column medians need not sum exactly to the median total. The ablations deliberately
change numerical execution and output values; they are diagnostic controls, not
alternative implementations or semantics-preserving production optimizations.
The last mode includes input buffer allocation/zeroing and synchronization, so
its 11.28 ms must **not** be attributed exclusively to output copying.

Nevertheless, simply removing input decoding leaves a large cost, and output
readback/release alone account for roughly 7 ms in the normal instrumented run.
Removing input handling saves about 6.3 ms in this control; it is also important.

## Callback work and native sample

Normal callbacks consume 66.74 summed worker milliseconds:

| Callback region | Summed worker ms | Approximate share |
|---|---:|---:|
| Input allocation, chunk access, decoder creation/decode/disposal | 56.39 | 84% |
| Portable binding/evaluation | 9.17 | 14% |
| Remaining callback work | 1.18 | 2% |

These totals overlap across workers and are **not additive wall-clock phases**.
Standalone parallel full-chunk decode measured 4.57 ms in this instrumented run;
the prefilter instead creates and destroys a private decoding context for every
input block: 512 callbacks × 2 inputs = 1,024 contexts per read.

A five-second macOS `sample` trace of the uninstrumented normal path corroborates
this: worker stacks repeatedly show
`portable_menudet_prefilter → blosc2_free_ctx → ZSTD_freeDCtx → free_medium → madvise`.
ZSTD sequence decoding, unshuffle, JIT evaluation and output memmoves also appear.
No ZSTD compression stack appears. Idle-thread and synchronization samples must
not be mistaken for active CPU percentages.

## Consequences

The two strongest targets are:

1. Publish worker results directly into the final NumPy output, eliminating the
   temporary SChunk storage, copy-back and subsequent storage release.
2. Reuse decoder contexts and input staging per worker rather than per block,
   preserving live storage/dictionary/postfilter constraints and FP isolation.

This profiling task does not change production dispatch or the shared install.

## Reproduction

Run `bench/menudet-prefilter-instrument.py --build-dir build-portable-python
--destination <external-profile-directory>` using conda environment `blosc2`.
It generates modified Cython sources and builds only an isolated profiling
extension, leaving the production extension and source unchanged. A persistent
Ninja build of the intended native checkout is required.

Select `<external-profile-directory>/package` explicitly with conda Python `-S`,
plus the intended numexpr and conda dependency paths, then run
`bench/menudet-prefilter-profile.py --samples 100` with
`ME_DSL_JIT_COMPILER=cc CC=clang`. For an uninstrumented control, select the normal
isolated package and pass `--modes 0`; `--native-sample <path>` captures macOS stacks.
Modes 1–3 require the instrumented extension and verify their expected altered
outputs before timing.
