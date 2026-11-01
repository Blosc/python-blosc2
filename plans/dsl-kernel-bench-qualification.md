# Menudet kernel benchmark qualification

## Current warm/cold CLI

The benchmark now follows `bench/menudet-engines.py` timing modes:

```sh
python bench/ndarray/dsl-kernel-bench.py --jit cc --warmups 3 --samples 3
python bench/ndarray/dsl-kernel-bench.py --jit tcc --warmups 0 --samples 3
python bench/ndarray/dsl-kernel-bench.py --no-jit --warmups 0 --samples 1
```

`--jit tcc|cc` replaces `--compiler`; omission inherits environment/default JIT
policy rather than forcing TCC. `--warmups` replaces the old `--repeat` interface
but now counts **untimed** repetitions. Separate `--samples` counts timed
repetitions; their **median**, not their minimum, is reported. Both default to 3.
`--no-jit` and `--jit` are mutually exclusive.

`--warmups 0` measures preparation-cold samples. Before each timed sample, Python
expression/graph/kernel caches are cleared, the native positive JIT cache is
disabled, and disk compilation uses a new temporary directory. The timer includes
fresh LazyExpr construction or Menudet specialization/binding, computation and
fresh output allocation/compression. Operand construction, module imports,
cache isolation/cleanup and correctness validation are outside the timer. CPU/OS
caches are not flushed. A preliminary correctness-qualified run is still made;
it cannot warm a cold sample's isolated preparation caches. JSON records mode,
warmups, sample count, median statistic, and raw samples. `preparation_s` and
`first_compute_s` remain qualification diagnostics, not parts of warm timing.

The profiling wrapper `profile-lazyexpr-prefilter.py` uses the same option names.
Historical JSON, profiles and measurements below retain their original warm-best
protocol; they are not relabeled as medians or preparation-cold runs.

### Math backend selection

`--math-backend system|sleef|accelerate` mirrors `bench/menudet-engines.py` and
applies to **both** routes: the value is placed in the `blosc2.jit_options`
context that wraps preparation and evaluation. Graph preparation, Menudet
artifact loading and the interpreter's math dispatch all read the task-local
policy there, so LazyExpr and Menudet use the same library. Omitting the option
inherits the current Python/native policy instead of forcing a default.

Availability is checked once, before any table is printed: a `float64` probe
kernel is specialized under the requested policy, so a build without SLEEF or
Accelerate exits with status 2 and the native explanation rather than silently
substituting another library. The probe uses `float64` so it never shares a
signature or cache entry with the measured `float32` kernels, and it is
untimed in both warm and preparation-cold modes. The request is recorded in
JSON as `math_backend_requested` and echoed on stdout as
`Miniexpr math backend request: ...`.

Effect on warm medians (arm64, 1000 × 1000, jit inherited, ms; LazyExpr/Menudet):

| Case | system | accelerate |
|---|---:|---:|
| transcend1 | 0.81 / 0.97 | 0.63 / 0.67 |
| transcend2 | 1.16 / 0.68 | 0.62 / 0.96 |
| transcend3 | 0.99 / 0.88 | 0.82 / 0.62 |
| transcend4 | 1.97 / 1.92 | 1.74 / 1.93 |
| sincos_id | 1.83 / 1.72 | 1.02 / 0.97 |

These are host samples for checking that the option takes effect, not a
portable ranking. `sleef` is unavailable in this build and correctly aborts.

#### Accelerated math no longer disables JIT for arithmetic-only kernels

SLEEF/Accelerate previously disabled JIT for every program, because generated
kernels would bypass the accelerated dispatch and pay a per-lane packing
callback. That rule was too broad: it also disabled JIT for programs that never
call a math function, where the math library cannot change any result. Combined
with the profile authoring a loop, `--math-backend accelerate --jit cc` ran the
loop cases in the interpreter and showed 10–47 ms.

JIT eligibility is now declined only when the program actually contains a math
call: a node with a Menudet math name, or a `**`/`pow` node whose semantics can
resolve through the backend. Programs that are pure arithmetic, comparison and
`where` keep JIT under accelerated backends; transcendental kernels still keep
the batched interpreter so accelerated semantics are preserved.

Measured on the same host with `--math-backend accelerate --jit cc` (warm
medians, ms):

| Case | LazyExpr | Menudet | Menudet mode before | Menudet mode after |
|---|---:|---:|---|---|
| loop1 | 1.22 | 1.58 | interp | jit |
| loop4 | 2.52 | 2.08 | interp | jit |
| loop4_heavy | 3.15 | 2.94 | interp | jit |
| nested2 | 1.99 | 3.99 | interp | jit |
| sincos_id | 1.44 | 1.29 | interp | interp |

The transcendental row is intentionally unchanged: its results are owned by the
accelerated library. Native coverage asserts both halves of the rule — a
`sin(x) + y` control is JIT-eligible under system math, `x * 2 + y - 1` stays
JIT-eligible under accelerated math, and `sin(x) + y` does not.

### Reported engine per route

The table prints the engine each route **actually selected**, not the requested
policy: `jit`, `interp`, or `other` when neither native Menudet path ran. JSON
records the same mapping under `execution_modes`, and the per-route
`execution_reports` retain the underlying backend string.

This matters because auto policy resolves differently by kernel shape. With no
`--jit` or `--no-jit`, an expanded LazyExpr expression stays on the interpreter
while a multi-statement Menudet loop auto-enables JIT. A preparation-cold sample
therefore charges Menudet for TCC compilation that LazyExpr never pays, which
reads as Menudet being slower:

| Case | LazyExpr mode | LazyExpr ms | Menudet mode | Menudet ms |
|---|---|---:|---|---:|
| loop1 | interp | 1.64 | jit | 4.94 |
| loop4 | interp | 4.53 | jit | 7.84 |
| nested2 | interp | 4.71 | jit | 8.07 |

With `--jit cc` both routes report `jit`; with `--no-jit` both report `interp`.
Under a shared policy the cold comparison is meaningful: with `--jit tcc`,
Menudet is faster cold because the authored loop compiles less code than the
expanded expression. With `--no-jit`, the remaining Menudet gap is interpreter
control-flow bookkeeping, described in `menudet-interpreter-batching.md`, not
compilation.

## Benchmark layout update

The benchmark now compares only normal `blosc2.lazyexpr` execution and public
Menudet kernels. The benchmark-local `miniexpr_enabled()` helper and redundant
Base route were removed: the legacy flag does not control newer native dispatch
and therefore cannot establish a NumPy-only baseline. The console labels are
`LazyExpr` and `Menudet`, with `Menudet speedup = LazyExpr time / Menudet time`.
NumPy remains an independent, untimed correctness reference. JSON now contains
only `lazyexpr` and `menudet` route timings and reports, plus explicit route and
speedup metadata. Both routes retain matched output storage and warm timing rules.

The measurements and original qualification below are historical; existing
JSON files are preserved rather than relabeled as results of this updated layout.
See `menudet-loops-qualification.md` for the later loop fast-path qualification.

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
