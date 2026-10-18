# Integrated Menudet math backend and FP-mode performance

2026-10-18, local macOS arm64 (Apple M4 Pro), Python 3.14, NumPy 2.5.4,
integrated `PortableKernel` artifacts from this checkout. Warm `evaluate()`
medians, 2,000,000 float64 items, 9 samples after warmup. Values are ns/item
(lower is better). `jit=off` exercises the bounded vector interpreter; `jit=on`
requests optional JIT and therefore falls back to that interpreter for explicit
SLEEF/Accelerate backends (see below). `fast` has no effect on interpreted
arithmetic; it changes `cc`-generated kernels only.

A post-review rebuild and rerun kept the shape and absolute level of these
results: median absolute change was 1.3% and the largest change was 9.8% on the
fastest 0.95 ns/item row, i.e. within run-to-run noise.

## JIT off (bounded vector interpreter)

| Expression | System strict | System fast | SLEEF u10 strict | SLEEF u10 fast | SLEEF u35 strict | SLEEF u35 fast | Accelerate strict | Accelerate fast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sin(x)+cos(x) | 10.47 | 10.47 | 6.58 | 6.61 | 5.07 | 5.20 | 3.60 | 3.64 |
| exp(sin(x))+log1p(x+1.5) | 10.41 | 10.88 | 8.69 | 8.70 | 8.24 | 8.09 | 5.73 | 5.79 |
| sin(x)*cos(x)+tan(0.25x) | 12.72 | 12.78 | 10.54 | 10.53 | 7.71 | 7.66 | 5.36 | 5.34 |
| sqrt(x*x+1) | 3.39 | 3.38 | 2.84 | 2.85 | 2.86 | 2.87 | 2.90 | 2.94 |

## JIT on (optional JIT requested)

| Expression | System strict | System fast | SLEEF u10 strict | SLEEF u10 fast | SLEEF u35 strict | SLEEF u35 fast | Accelerate strict | Accelerate fast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sin(x)+cos(x) | 5.87 | 5.79 | 6.86 | 6.85 | 5.40 | 5.39 | 4.10 | 4.12 |
| exp(sin(x))+log1p(x+1.5) | 8.17 | 8.20 | 8.67 | 8.81 | 8.07 | 8.09 | 5.76 | 5.86 |
| sin(x)*cos(x)+tan(0.25x) | 7.50 | 7.43 | 10.75 | 10.63 | 7.68 | 7.86 | 5.47 | 5.41 |
| sqrt(x*x+1) | 0.95 | 0.99 | 2.97 | 2.96 | 2.97 | 3.02 | 3.01 | 2.91 |

SLEEF/Accelerate rows are vector-interpreter fallbacks because accelerated
backends are intentionally not fused into the per-lane generated JIT. System
math retains generated-JIT support. Fast-math does not improve these
transcendental maps measurably here; its purpose is permission, not an expected
speedup. Accelerate is consistently fastest for transcendental maps. SLEEF u35
is faster than u10 and still within the audit's documented accuracy bounds for
these inputs; the u10/u35 split is observable rather than masked by the
process-global default. The pure arithmetic `sqrt` reduction favors generated
system JIT.

Reproduce with `bench/mathlib-performance.py`; raw rows are in
[mathlib-performance.json](mathlib-performance.json).
