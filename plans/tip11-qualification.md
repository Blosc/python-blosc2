# Tip 11: first qualification (2026-10-31)

**Historical first-stage results below.** The follow-up
[native-generator qualification](tip11-parallel-qualification.md) reaches
legacy-level system-CC throughput at 20M/12 threads. The chosen separate
`@menudet`/`@jit` contracts and current 200M figure are recorded in
[menudet-api.md](menudet-api.md); the original JSON files below remain unchanged.

**Status at this first stage: provisionally ported to public `@blosc2.menudet`.**
Hash qualification preceded the decorator-design review. That review ultimately
retained `@menudet` as the chosen interface, so no further entry-point migration
is required. No other benchmarks were migrated at this stage.

## Results

20,000,000 int32 elements, Apple M4 Pro, macOS arm64, 12 Blosc2 threads.
Python 3.14.4, NumPy 2.5.4, Python-Blosc2 4.15.0.dev0, C-Blosc2 3.4.2.dev.
All arms allocate a new NDArray, `clevel=0`, chunks=4,194,304,
blocks=262,144; Menudet logical partitions equal blocks. Seed=42.

| Route | System CC, warm median | TCC, warm median | CC peak RSS / growth |
|---|---:|---:|---:|
| NumPy PCG64 + asarray | 0.0380s | 0.0373s | 243.5 / 175.3 MiB |
| True legacy engine (internal control) | 0.00534s | 0.0105s | 155.9 / 90.0 MiB |
| Original source via Menudet compatibility adapter | 0.0430s | 0.7427s | 181.4 / 114.1 MiB |
| Public Menudet hash, typed captures | 0.0458s | 0.4380s | 183.3 / 116.3 MiB |

The migrated hash is **8.6x slower than true legacy CC JIT**, and **1.21x slower
than NumPy staging**, while reducing whole-process peak RSS by about 25%.
With TCC it is 42x slower than legacy and 11.7x slower than NumPy. Do not claim
the historical 3.6x speedup. These numbers do not measure `blosc2.random`.

Raw samples, first-process compute/preparation timings, package/extension paths,
binary SHA256, revisions and execution environment:

- [System CC](tip11-qualification-cc.json), used for the regenerated guide PNG.
- [TCC](tip11-qualification-tcc.json).

Imports are excluded. Each route has a separate timing process and a separate
memory process. Preparation/specialization is separate from the first compute,
followed by three warm runs. The disk JIT cache was **not cleared**: preparation
may include loading cached code, not cold compilation. Legacy adapter preparation
also occurs inside compute, unlike explicit Menudet specialization. Memory is OS
whole-process high-water RSS measured immediately after first compute; growth is
relative to the post-preparation high-water mark. It is not the old helper's
tracemalloc/RSS hybrid, so do not compare the memory numbers directly with old plots.

## Why the multi-second regression occurred

1. `SourceKernel` + `lazyudf` now uses the Menudet compatibility adapter. It is
   **not the old engine**. The explicitly labeled `legacy-engine` arm bypasses
   that adapter via an internal flag solely as a benchmark control. Its legacy
   native JIT compilation probe must succeed; no recipe is loaded or persisted.
2. The Menudet default is one logical partition spanning the array. Its semantics
   require evaluating complete partitions for slices, so output chunks can
   repeatedly evaluate the entire array. Explicit bounded partitions avoid that.
3. The generic partition scheduler builds per-lane coordinate arrays and advanced
   indexing grids even for contiguous slices. `menudet_lazy.py` now intersects
   rectangular slices with the **original whole partitions**, without those grids.
   Reductions, context coordinates and partial-group evaluation stay unchanged.
4. Native typed bitwise operations were replaying scalar interpreter helpers
   for every XOR/AND. `../miniexpr/src/dsl_menudet_jit.c` now emits direct
   same-width typed bitwise operations, retaining the checked/mixed-domain paths.
   Both native and Python changes are required for these results.
5. The current Menudet artifact binding does not forward the `jit_backend` Python
   option. The benchmark explicitly sets `ME_DSL_JIT_COMPILER` in its own process
   and workers. `compiler=` means the CC executable, **not** the compiler backend.
   This binding limitation is recorded, not silently treated as qualified API behavior.

Exploratory 20M runs before the fixes took about 4.7s through the original adapter.
With explicit partitions and a typed uint32 kernel, native bitwise lowering alone
reduced about 2.2s to 0.35s; eliminating scheduler index grids reduced about 0.35s
to 0.045s. These exploratory numbers used different intermediate authoring/layout
choices and are **not a controlled speedup series**. The paired table above is
the reproducible comparison. The remaining legacy gap includes serial partition
evaluation/copying in Menudet versus native chunk/block-parallel legacy generation.
Parallel native context-generator materialization remains follow-up work.

## Correctness and interpretation

- Public hash authoring uses typed int64 captures and native `_flat_idx` context
  with declared `ndim=1`; this remains the current documented context spelling.
- For counters in `[0, 2**32)` each multiplication is below `2**59`. Explicit masks
  truncate the positive int64 state to 32 bits; `int32(...)` defines the signed
  output conversion. Do not rely on the legacy output dtype selecting int64
  arithmetic for the whole function.
- Checked against an independent NumPy uint64 implementation, including partition
  boundaries, partial slices, output rechunking, interpreter parity and small
  blocks near `2**31` and `2**32` without allocating a huge array.
- All measured hash routes check shape, dtype and 257 evenly spaced values before
  reporting warm results. NumPy PCG64 is a different workload, not a value-parity arm.
- The hash is bijective on uint32 counters (invertible XOR shifts and an odd
  multiplier). Zero duplicates are expected by construction, not evidence of a
  better RNG. At 20M samples: mean=18,477.9, scaled std=0.577356,
  chi-square=192.54, lag1=0.000204; duplicate count in the first 1M is zero.
  PCG64 has 124 duplicates there. These are descriptive checks, not certification.

## Reproduction and build provenance

Base revisions: Python-Blosc2 `6a6027457aa6693fcd7d9d2a6c4ff50ef8297453`,
miniexpr `7ba6d40fdb5084526fc1750d7a30c4125caa332f`, plus this task's changes.
The extension was incrementally rebuilt in `build-portable-python`, retaining
`FETCHCONTENT_SOURCE_DIR_MINIEXPR=/Users/faltet/blosc/miniexpr`, then copied into
an isolated package with the Python sources. **The shared editable extension was
not overwritten.** Rebuilding without refreshing the extension actually imported
by Python does not activate the native fix. A native dependency version/pin
carrying this change is also needed for future independent wheels.

Once using the matching package and rebuilt extension in the `blosc2` environment:

```sh
python bench/optim_tips/tip_11_dsl_random.py --items 20000000 --compiler cc --quality-items 20000000 --output plans/tip11-qualification-cc.json --plot
python bench/optim_tips/tip_11_dsl_random.py --items 20000000 --compiler tcc --output plans/tip11-qualification-tcc.json
```

For the isolated run, conda Python `-S` was used with the isolated package and
the conda dependency site-packages explicitly on `sys.path`. Workers inherit
those paths and `-S`; each raw row records its actual package/extension paths.

Validation: 40 Python tests passed (`test_menudet_lazy`, `test_tip11_hash`,
default-selected cases from `test_menudet_jit_integration` and
`test_menudet_jit_parity`; 374 deselected by repository defaults). Native
`menudet_jit_parity_tcc` and `menudet_jit_parity_cc_gate` passed. Existing native
bitwise dtype/boundary/mask matrices own numerical coverage; a composed bitwise
expression extends that suite rather than adding Python dtype cross-products.
The full native CC matrix, full Python suite, independent wheel and documentation
build were not run. The initial reused build needed its stale pip-isolation NumPy
include path refreshed. Build warnings were existing dependency CMake policies,
CBLAS deprecations, generated-C const/abs warnings, and duplicate-library linking;
the native bitwise incremental rebuild introduced no compiler warning.

## Remaining gates

- The later API review retained `@menudet`; the proposed decorator retargeting
  is no longer a gate. Current follow-up qualification is linked above.
- Keep true-legacy authoring explicitly internal; it is not supported migration advice.
- Fix/qualify public backend option propagation separately.
- Improve native context-generator scheduling if legacy-level throughput is required.
- Qualify `blosc2.random` separately before attaching performance claims to the
  guide's RNG recommendation; do not substitute hash timings.
- 200M, other CPUs/OSes and wheels remain unqualified.
