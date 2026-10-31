# Tip 11: native parallel generator follow-up (2026-10-31)

**Update:** the fresh 200M `@menudet` qualification now supersedes the guide plot
below; see [menudet-api.md](menudet-api.md). The 20M JSON
measurements here remain unchanged as investigation history. The documented
`jit_backend` binding limitation below has now been fixed.

## Outcome

**Legacy-level throughput is recoverable on the qualified 20M/12-thread CC
configuration.** The first-stage gap was not an inherent cost of portable
artifacts: Python-side serial materialization and conservative generated integer
locals were substantial avoidable costs.

Final paired results, medians of seven warm runs, allocating int32 NDArrays:

| Route | CC, 12 threads | CC, 1 thread | TCC, 12 threads |
|---|---:|---:|---:|
| NumPy PCG64 + asarray | 0.0364s | 0.0367s | 0.0360s |
| True legacy engine, internal control | 0.0047s | 0.0131s | 0.0105s |
| Legacy source via compatibility adapter | 0.0197s | 0.0201s | 0.7053s |
| Menudet serial materialization control | 0.0323s | 0.0317s | 0.4207s |
| Menudet native prefilter | **0.0046s** | **0.0265s** | **0.0615s** |

CC/12-thread Menudet is 7.0x faster than the same-build serial control and within
measurement variation of legacy (0.98x legacy execution time). Whole-process peak
RSS is 148.4 MiB, versus 188.3 MiB serial and 241.8 MiB NumPy staging; peak growth
over post-preparation RSS is 80.8 MiB for a 76.3 MiB logical result. These are not
claims of parity at one thread (still about 2x slower), on TCC (5.9x slower), or
on other machines. The hash and PCG64 have **different sampling semantics**.

Raw final results, including execution reports, binary hashes and revisions:

- [CC, 12 threads](tip11-integer-locals-cc.json), used for the previous 20M guide plot.
- [CC, 1 thread](tip11-integer-locals-cc-1thread.json).
- [TCC, 12 threads](tip11-integer-locals-tcc.json).

## Investigation and changes

The existing native Menudet prefilter already supports input-free 1-D context
kernels. Constructors use it, but general `MenudetLazyArray.compute()` did not.
No new native execution ABI was needed.

`src/blosc2/storage_readers.py` now selects that prefilter for **numeric,
input-free, elementwise 1-D kernels** when output blocks equal the original
logical partitions, chunks contain whole blocks, and the bounded scratch/runtime
eligibility checks pass. It uses `uninit` storage and lets C-Blosc2's block workers
evaluate directly into the output payload. There is no Python result array or
copy per partition. Context-free input-free kernels can use the same route.

This is deliberately **not arbitrary repartitioning**. Group reductions are
allowed because each callback evaluates exactly one original whole partition,
including a final short group. `_flat_idx`, `_i0` and `_n0` see the original domain
and correct block origin. Partial selections, mismatched blocks, chunks splitting
partitions, N-D generators, complex/string outputs and unsupported runtimes keep
the existing implementation. A compatible preallocated `out=` can also be used.
The existing prefilter owns worker joins, FP aggregation, error propagation and
native context cleanup; tests exercise these contracts with the new dispatch.

With **only this Python dispatch change**, the installed extension produces:

| CC threads | Serial control | Native prefilter | True legacy |
|---|---:|---:|---:|
| 1 | 0.0445s | 0.0381s | 0.0126s |
| 12 | 0.0434s | 0.0061s | 0.0045s |

Sources: [12-thread CC](tip11-native-prefilter-cc.json),
[1-thread CC](tip11-native-prefilter-cc-1thread.json),
[TCC](tip11-native-prefilter-tcc.json). Parallelism, rather than eliminating
Python call overhead alone, explains most of this improvement.

The residual per-core gap prompted inspection of native generated C.
`../miniexpr/src/dsl_menudet_jit.c` declared every host local `volatile`.
Those observation/rounding barriers are unnecessary for integer/Boolean locals
and force stores/loads between hash rounds. The JIT now keeps them only for
floating-point locals, preserving the existing FP policy. This improves CC
single-thread native materialization from about 0.038s to 0.0265s and 12-thread
materialization from about 0.0061s to 0.0046s. TCC sees essentially no improvement
from this qualifier change. Checked operations still issue and check their native
status even when a local result is unused; a native parity case covers this.

The final execution report confirms 77 native block evaluations on 12 workers,
zero input-buffer allocations/fetches, and zero Python output-copy bytes. Only
padding is zero-filled. The same kernel remains available through the explicitly
labeled `menudet-serial` control, so a legacy-versus-new comparison cannot
accidentally compare the same implementation twice.

The compatibility adapter is intentionally not retuned: LazyUDF exposes its
auto-selected source blocks (32,000 items on this run), while the requested
output blocks are 262,144. Its logical groups therefore do not meet the new exact
alignment rule. The hash is group-independent, but the adapter cannot assume this
for arbitrary kernels. Final raw rows record both output grids and source blocks
where available. All materialized output grids are identical across routes.

## Correctness and qualification

- 222 selected Python tests passed: generator scheduling, existing storage readers,
  Menudet lazy arrays, hash workload, decorators, array bindings, policy, and
  default-selected JIT integration/parity cases. Repository defaults deselected
  374 cases; no timing thresholds were added.
- Generator tests cover full original groups with reductions, logical coordinates,
  short final partitions, persistent output, clevel 0/5, multiple thread counts,
  mismatched layouts, partial selection, FP status aggregation, native failures
  and subsequent output-context reuse.
- Existing exact hash checks cover partial slices, rechunking, interpreter parity
  and high-origin blocks near uint32 boundaries. All timed hash routes check
  sampled exact values; descriptive quality statistics at 20M match stage one.
- Native `menudet_jit_parity_tcc` (full) and `menudet_jit_parity_cc_gate` (compact)
  passed, including the new unused checked integer-local regression. An attempted
  full CC run exceeded a 120-second execution budget; it is **not qualified as a
  completed full CC matrix**. No assertion failure was reported before timeout.
- The full Python suite, independent wheel and full documentation build were not
  run; 200M and other platforms remain outside this qualification.
- Ruff and diff whitespace checks are part of the final validation. No new native
  compiler warnings were introduced. Reusing the pip build directory required
  refreshing its expired build-isolation NumPy include path; rebuilding also
  printed the existing CMake policy/deprecation, CBLAS, generated-C const/abs and
  duplicate-library warnings noted in stage one.

## Reproduction and activation

Hardware/protocol remain Apple M4 Pro, macOS arm64, 20,000,000 items, seed 42,
chunks=4,194,304, blocks/partitions=262,144, clevel=0, and separate timing/memory
processes. System CC is Apple clang 21.0.0, native JIT default `-O3`; no custom
CFLAGS or FP policy were supplied. Preparation and first compute are reported
separately; the persistent JIT cache was not cleared, so they are **not guaranteed
cold-compilation timings**. Python 3.14.4, NumPy 2.5.4, Python-Blosc2 4.15.0.dev0,
C-Blosc2 3.4.2.dev; base revisions are retained in every raw JSON.

The dispatch-only results use the actual shared editable extension. The final
integer-local optimization uses an incrementally rebuilt extension in
`tip11-parallel-isolated`, imported with conda Python `-S` and explicit dependency
paths. **The shared editable extension was not overwritten with that new native
optimization.** Thus the current checkout can gain most of the acceleration
immediately (~0.006s CC on this machine); matching the final ~0.0046s result needs
the additional native rebuild/install from the local miniexpr source.

For example, in this checkout with the `blosc2` environment:

```sh
python -m pip install -e . --config-settings=build-dir=build-portable-python --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=/Users/faltet/blosc/miniexpr
python bench/optim_tips/tip_11_dsl_random.py --compiler cc --repeat 7 --quality-items 20000000 --output plans/tip11-integer-locals-cc.json --plot
python bench/optim_tips/tip_11_dsl_random.py --compiler cc --threads 1 --repeat 7 --quality-items 0
python bench/optim_tips/tip_11_dsl_random.py --compiler tcc --repeat 7 --quality-items 0
```

At this stage Menudet's artifact binding did not yet forward Python `jit_backend`,
so the standalone benchmark explicitly selected `ME_DSL_JIT_COMPILER`. That binding
gap has since been fixed. The subsequent API review retained separate `@menudet`
and `@jit` decorators, and a fresh 200M run is recorded in
[menudet-api.md](menudet-api.md). A native dependency pin carrying the changes,
independent wheels and other platforms remain follow-up qualifications. None of
these measurements qualify `blosc2.random`.
