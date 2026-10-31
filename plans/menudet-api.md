# Separate Menudet/JIT contracts and tip 11 at 200M (2026-10-31)

## Chosen public interfaces

Keep **`@blosc2.menudet` for validated lazy kernels** and **`@blosc2.jit` for
NumPy-style eager/tracing convenience**. The proposed portable flag on `@jit`
was prototyped and then rejected: it selected substantially different authoring
semantics and return types, not just an execution option. The prototype code,
tests and option restrictions have been removed. Its
[decision record](portable-jit-api.md) and raw measurements remain explicitly
historical, not guidance for a supported API.

`@menudet` retains its existing `MenudetFunction`/`MenudetKernel` types,
specialization and lazy results, typed annotations/scalar captures, logical
domains/partitions, native errors and validated artifact persistence. Its Python
body is never executed, including after native failures. `@jit` retains its
existing tracing/control-flow dispatch, eager NumPy results (or NDArray with
storage options), `out=`, `disable=`, `strict=` and pandas integration. There is
no artifact/schema/native ABI rename and no restoration of retired DSL authoring
or historical source-backed recipes.

Native interpretation/JIT is separate from authoring. `@menudet(jit=False)` uses
the native interpreter; `jit=True` prefers compilation and `require_jit=True`
requires it. Select backend/compiler/cache controls through `jit_options` when
specializing/calling a Menudet function. A prepared kernel retains its policy;
call/specialize again to select another. `jit_backend` selects TCC or system CC;
`compiler` specifies the CC command. These are not new decorator options.

The Menudet **backend-binding fix is retained**: scoped backend defaults reach a
runtime-only artifact copy without injecting machine-specific compiler pragmas
into saved author source. Original author pragmas/environment precedence and
native validation remain authoritative. Tests cover nested contexts, cache
identity, serialization neutrality, malformed/duplicate-key/non-UTF-8 payloads,
and actual selection of an unavailable CC executable with require-JIT enabled.

The performance changes are retained independently: contiguous partition
scheduling, aligned native generator materialization, native bitwise lowering
and optimized integer locals. Existing examples/tutorials/benchmarks using
`@menudet` stay on that interface; they do not need a decorator rename.

## Fresh 200M @menudet qualification

The manually selected **200M default is preserved**. The benchmark has been rerun
after restoring `@blosc2.menudet(ndim=1, require_jit=True)`, and the local PNG and
guide claims are regenerated from this run—not relabeled prototype output.

Apple M4 Pro, 12 threads, system CC, int32, clevel=0, chunks=4,194,304,
blocks/logical partitions=262,144; medians of seven warm runs:

| Route | Time | Whole-process peak RSS |
|---|---:|---:|
| NumPy PCG64 + asarray | 0.3716s | 1617.4 MiB |
| True legacy engine | 0.0444s | 849.2 MiB |
| Legacy compatibility adapter | 0.1726s | 871.1 MiB |
| Menudet serial control | 0.2932s | 883.7 MiB |
| Menudet native prefilter | **0.0456s** | **837.1 MiB** |

The hash is about **8.2x faster than NumPy staging**, uses **1.9x lower peak RSS**,
and is 6.4x faster than the same-build serial control. It is within about 3% of
legacy execution time on this configuration. The hash and PCG64 have different
sampling semantics: the hash is a uint32 permutation, not independent draws.
These are **not `blosc2.random` measurements**. Descriptive quality checks use a
separate 20M sample; sampled exact hash values are checked before warm timings
are accepted. No timing thresholds were added to tests.

Raw measurements and provenance: [tip11-menudet-200m-cc.json](tip11-menudet-200m-cc.json).
Worker package/extension paths identify the active editable install. Binary hash,
revisions, backend, actual output grids, execution reports and individual warm
samples are included. Timing and memory runs are separate subprocesses.
Preparation/first compute are reported separately; the persistent JIT cache was
not cleared, so these are not guaranteed cold-compilation measurements.

Reproduction in the `blosc2` conda environment:

```sh
python bench/optim_tips/tip_11_dsl_random.py --compiler cc --repeat 7 --quality-items 20000000 --output plans/tip11-menudet-200m-cc.json --plot
```

The CLI still selects the same explicit backend for all internal control arms,
including the direct legacy compilation probe. Standalone Menudet functions also
honor scoped backend settings without environment mutation. This API separation
change is Python-only; it does not require another native rebuild when the
previously qualified native optimizations are already installed.

## Remaining scope

Other legacy workload migrations and their numerical/performance qualifications
remain separate tasks. Existing modern Menudet examples need auditing, not
entry-point alignment. Independent wheels, other machines/backends at 200M and
non-default FP-accuracy policy remain outside this qualification. Menudet's
native FP policy is distinct from the expression evaluator's `fp_accuracy` knob.
