# Menudet versus NumPy and numexpr

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
