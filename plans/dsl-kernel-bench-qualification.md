# Menudet kernel benchmark qualification

## Scope and reproduction

`bench/ndarray/dsl-kernel-bench.py` now authors all ten workloads with public
`@blosc2.menudet`, specializes explicitly, and binds with `MenudetKernel.lazy`.
No private authoring, `.dsl_source`, or portable objects passed to `lazyudf`
remain. Float32 input/return annotations preserve the former explicit output
dtype; otherwise loop accumulator inference produced float64.

Commands (2026-11-01, from the repository root):

```sh
conda run -n blosc2 python bench/ndarray/dsl-kernel-bench.py --threads 4 --output plans/dsl-kernel-bench-tcc.json
conda run -n blosc2 python bench/ndarray/dsl-kernel-bench.py --threads 4 --compiler cc --output plans/dsl-kernel-bench-cc.json
conda run -n blosc2 python bench/ndarray/dsl-kernel-bench.py --size 33 --repeat 1 --threads 2 --no-jit
conda run -n blosc2 pytest -n 0 tests/test_bench_dsl_kernel.py tests/test_menudet_decorator.py
```

Host: Apple M4 Pro, macOS arm64; Python 3.14.4, NumPy 2.5.4,
Python-Blosc2 4.15.0.dev0, C-Blosc2 3.4.2.dev (2026-10-30).
System `cc`: Apple clang 21.0.0 (clang-2100.3.34.2).
Python checkout base: `361f87f492f7d4e843eec2eb94b1897ae05806f6` plus this migration.
The checkout pins miniexpr to `6413c800984ff1b8c63953e336b3b8b245b1c3a5`;
the loaded extension does not expose its embedded native revision, so that pin
is not independent proof of the binary's provenance. Package and actual extension
paths are recorded in both JSON files (editable Python sources, conda-environment
extension). These are host measurements, not independent-wheel qualification.

## Measurement contract and correctness

Both JSON files contain raw warm samples, best times, preparation and first-compute
times, execution options/environment, package paths, storage and input metadata.
Preparation includes specialization/possible compilation; decoration happens at
module import, outside timing. First compute is **not** cache-isolated cold
compilation. Every route allocates/compresses a fresh NDArray using the same
BLOSCLZ level-1 chunks/blocks. NumPy conversion and validation are excluded.
Throughput counts logical input/output bytes, not compressed bytes or measured
memory traffic. This benchmark does not measure peak memory.

Inputs are deterministic `(1000, 1000)` float32 linspaces on `[0, 1]` and
`[1, 0]`, chunks `(1000, 1000)`, blocks `(25, 1000)`, four Blosc threads.
One first computation precedes three warm repetitions for each route, in fixed
baseline/expression/kernel order. All routes check shape, dtype and values against
an independent NumPy evaluation before being timed. Tolerance is `rtol=1e-5`,
`atol=32 * float32.eps`: typed native intermediates differ from NumPy weak-scalar
rounding, and the heavy loop cancels near zero. At 1M, the former `atol=2e-6`
failed on 11 heavy-loop elements, with maximum violating absolute error 2.27e-6.
No Python-body fallback is used for the Menudet kernels.

All ten Menudet specializations report `has_jit=True` in both full runs;
all ten interpreter smoke cases report `False`. Backend configuration is recorded
as requested, not inferred from speed. The expression arms preserve the original
miniexpr-disabled/enabled switch, **including fallback**. Their execution reports
are unavailable (`null`) here; enabling miniexpr is not proof of native execution.
Do not describe these measurements as a verified pure-native expression comparison
or as NumExpr results. Lightweight tests cover lazy binding, float32 output,
non-divisible multidimensional storage partitions, representative three-route
execution and restoration of the expression switch. All 50 selected tests pass.

## Observations (warm best milliseconds)

| Case | Base TCC run | Expr TCC run | Menudet TCC | Base CC run | Expr CC run | Menudet CC |
|---|---:|---:|---:|---:|---:|---:|
| loop1 | 114.95 | 114.50 | 18.89 | 113.83 | 113.42 | 4.64 |
| loop2 | 248.51 | 249.31 | 32.03 | 246.62 | 244.13 | 5.52 |
| loop4 | 551.92 | 551.02 | 58.46 | 535.73 | 527.14 | 7.03 |
| loop4_heavy | 1657.33 | 1650.05 | 59.32 | 1647.19 | 1667.73 | 6.27 |
| nested2 | 532.40 | 536.31 | 80.03 | 536.60 | 551.10 | 4.73 |
| transcend1 | 3.43 | 3.42 | 24.27 | 3.69 | 3.89 | 24.50 |
| transcend2 | 2.27 | 2.31 | 23.05 | 2.66 | 2.55 | 23.22 |
| transcend3 | 3.41 | 3.50 | 23.66 | 3.70 | 3.91 | 23.77 |
| transcend4 | 10.15 | 10.04 | 31.96 | 9.94 | 10.27 | 29.15 |
| sincos_id | 4.91 | 4.98 | 29.42 | 5.05 | 4.90 | 24.74 |

CC's loop kernels outperform TCC on this host, but this is not a blanket kernel
speedup recommendation: both Menudet configurations lose to the expression routes
for every transcendental workload. Expression-route times are nearly identical;
their backend use remains unqualified. These fixed-order, short host runs do not
establish other-platform performance or cache-isolated compilation cost. There
are no pre-existing plots/notebook outputs to regenerate for this script.
