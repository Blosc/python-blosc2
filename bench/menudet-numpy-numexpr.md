# Menudet versus NumPy and numexpr

## Parallel resident execution

Large resident full reads now use reusable worker threads for conservatively
proven lane-local kernels. Threads share the prepared artifact and write directly
to disjoint views of one unpublished output; inputs are not copied. Each read
waits for every submitted worker before returning or propagating a failure.
Floating flags are aggregated. No execution failure is replayed serially.

The current policy requires aligned, host-endian, C-contiguous floating arrays
with matching shapes. It excludes broadcasting, partial selections, context,
reductions, scalar captures, constant-only weak arithmetic and final conversions.
The worker count is `min(blosc2.nthreads, 32, size // 131072)`; fewer than two
workers keeps the existing serial path. In this benchmark 100k remains serial,
1M uses seven workers and 8M uses twelve. Standalone `evaluate_block` remains
serial. `--samples 50` now also measures the same lazy resident read with
`blosc2.nthreads=1`, for a same-run serial baseline.

The original pre-parallel measurements below are retained for comparison.

Updated resident comparison, warm medians of 50 samples (milliseconds):

| Values | Workers | NumPy TCC/Clang run | numexpr 12t TCC/Clang run | Menudet TCC serial → auto | Menudet Clang serial → auto |
|---|---:|---:|---:|---:|---:|
| 100k | 1 | 0.031 / 0.027 | 0.140 / 0.126 | 0.211 → 0.210 | 0.160 → 0.067 |
| 1M | 7 | 0.653 / 0.607 | 0.287 / 0.289 | 2.435 → 0.593 | 1.291 → 0.498 |
| 8M | 12 | 6.600 / 6.516 | 1.878 / 1.916 | 19.789 → 3.728 | 11.006 → 2.661 |

At 8M, the new resident path is 5.31x faster than same-run serial TCC and 4.14x
faster than same-run serial Clang. Clang Menudet is 2.45x faster than NumPy, but
12-thread numexpr is still 1.39x faster than Clang Menudet. At 100k both Menudet
columns execute the same serial path; their difference is measurement/cache
variability, not parallel speedup. Separate compiler runs must not be interpreted
as simultaneous measurements. A preceding run measured 0.561/0.561 ms at 1M and
3.807/2.803 ms at 8M for TCC/Clang automatic execution, illustrating variability.

Compressed dispatch is unchanged. Updated end-to-end compressed readings were:

| Values | NumPy TCC/Clang run | numexpr 12t TCC/Clang run | Menudet TCC | Menudet Clang |
|---|---:|---:|---:|---:|
| 100k | 0.234 / 0.237 | 0.348 / 0.306 | 0.425 | 0.339 |
| 1M | 3.000 / 2.554 | 2.006 / 2.032 | 3.596 | 2.515 |
| 8M | 20.277 / 20.017 | 15.693 / 15.466 | 17.911 | 13.243 |

The configured suite passed 13,214 tests (76 skipped, 12,605 deselected) before
the final metadata-tightening and submission-failure test additions.
Final focused qualification passed 107 tests, covering reduction fallback, partial,
strided and broadcast reads, float32/float64, interpreter/JIT, Boolean output,
FP aggregation, concurrent shared artifacts, ordered failure propagation,
submission-failure joins and private output alias rejection. The instrumented
Cython binding ASan gate passed 83 tests; native static archives were not instrumented
and leak detection was disabled.

Expression: `x + (y - 1)` on two float64 linspace arrays. Warm median of 50
samples; results checked against NumPy. macOS arm64, NumPy 2.5.3, numexpr
2.14.2. Blosc2 uses 12 threads; numexpr is measured with both 1 and 12 threads.
Both Menudet specializations confirmed JIT availability. Compilation is excluded.
Every operation allocates a new result; NumPy also allocates an intermediate.

## Resident NumPy inputs (milliseconds)

Baseline columns come from the TCC run; the separate Clang run measured similar
baselines except NumPy at 1M (0.759 rather than 0.593 ms).

| Values | NumPy | numexpr 1t | numexpr 12t | Menudet TCC | Menudet Clang |
|---|---:|---:|---:|---:|---:|
| 100k | 0.031 | 0.060 | 0.133 | 0.209 | 0.097 |
| 1M | 0.593 | 0.957 | 0.301 | 2.423 | 1.500 |
| 8M | 6.554 | 8.319 | 1.934 | 20.010 | 11.289 |

Resident Menudet evaluates one portable invocation; it does not use the parallel
compressed prefilter. NumPy wins at 100k; parallel numexpr wins at 1M and 8M.

## Compressed inputs, including decoding (milliseconds)

NumPy and numexpr first materialize each input through public `NDArray[:]`.
Menudet uses automatic single-chunk dispatch for 100k/1M and its native portable
prefilter for 8M (eight 1M chunks, 15,625-value blocks). These are end-to-end
comparisons of public reads, not identical decoding pipelines. Full materialization
uses 128 MB of input buffers at 8M; the Menudet prefilter uses 3 MB of input block
staging plus an 8 MB scheduling buffer. Result/storage/codec scratch is additional.

| Values | NumPy TCC/Clang run | numexpr 1t TCC/Clang run | numexpr 12t TCC/Clang run | Menudet TCC | Menudet Clang |
|---|---:|---:|---:|---:|---:|
| 100k | 0.244 / 0.239 | 0.263 / 0.257 | 0.328 / 0.320 | 0.427 | 0.356 |
| 1M | 3.053 / 2.332 | 2.894 / 2.824 | 2.022 / 2.057 | 3.641 | 2.523 |
| 8M | 20.931 / 20.442 | 23.080 / 22.758 | 16.082 / 15.910 | 16.458 | 16.340 |

At 8M, automatic Menudet is about 1.25–1.27x faster than decoding then using
NumPy, and within 2–3% of decoding then using 12-thread numexpr, with much smaller
input staging. At smaller sizes NumPy/numexpr remain faster overall. Compiler
choice strongly affects resident evaluation but has little effect on this 8M
compressed workload. These observations concern this simple expression only.

Reproduce with `bench/menudet-numpy-numexpr.py --samples 50`, selecting the rebuilt
isolated package and conda `blosc2` dependencies. Use `ME_DSL_JIT_COMPILER=tcc` or
`ME_DSL_JIT_COMPILER=cc CC=clang`. The shared editable extension was not changed.
