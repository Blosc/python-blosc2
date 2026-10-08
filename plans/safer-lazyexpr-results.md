# Safer LazyExpr experiment: implementation and initial results

Date: 2026-10-08. Branch: `safer-lazyexpr`.

**Status: narrowed phase-1 experiment. Ordinary in-memory defaults are restored;
safe loading uses graph execution. Exit gates remain incomplete. Not ready to merge.**

## Implemented

- `expression_graph.py` parses bounded ASTs into immutable tuple graphs with an
  explicit numerical-operation allowlist. Registered namespace spellings are
  syntax aliases, not arbitrary namespace traversal.
- Numerical source execution in `lazyexpr.py` now routes through the graph bridge;
  the only Python `eval()` bridge is gated by the trusted evaluation context.
- Structured recipes validate syntax before reference resolution. Recursive
  recipe expansion is bounded, and dependency validation rejects cycles while
  permitting shared dependencies.
- Ordinary construction/evaluation retains the trusted legacy default. Safe loading
  establishes its own graph boundary. Explicit `evaluation="safe"` or
  `with blosc2.expression_evaluation("safe"):` exercises graph execution in memory.
  These are experiment APIs, not an approved public API freeze. No new `open()`
  parameter was introduced.
- Deferred dtype, shape and compute routes restore expression policy and
  revalidate changed text/bindings. Parsing plans are cached, never operand data.
- Full loading prefers safe numerical reconstruction when eligible. Existing
  legacy UDF loading gates remain; source-reconstructed UDF dependencies are
  rejected by safe graph admission.
- Existing NumExpr/miniexpr and chunk/reduction schedulers remain in use. No
  Menudet lowering or reinterpretation of LazyExpr semantics was added.
- Added disk/frame/in-memory round trips for the five prototype expressions,
  hostile operand/subclass tests, forbidden recipes before reference resolution,
  graph mutation and cycle checks, caller-function shadowing and parser bounds.

## Current capability boundary

| Category | Experimental behavior |
| --- | --- |
| Registered numerical calls, reductions, indexing, constructors and shape operations | Validated graph dispatch; existing schedulers/inference reused. |
| Exact non-object NumPy arrays/scalars | Admitted; ndarray subclasses and object dtype reject. |
| Concrete Blosc2 arrays, fields, columns, remote/storage adapters | Explicit types admitted; complete adapter/provenance audit still outstanding. |
| Numeric concrete pandas Series | Adapted via its concrete NumPy storage; extension arrays reject. |
| Caller-authored in-memory LazyUDF | Trusted caller capability, not constructible by saved graph text; source-reconstructed legacy UDF rejects. |
| Arbitrary array-protocol objects | Ordinary in-memory compatibility preserved; reject under the safe graph policy before metadata/protocol hooks. |
| Unregistered calls, arbitrary attributes, lambdas, comprehensions, starred arguments and file-writing constructor keywords | Reject before operand resolution for structured recipes. |

The safety claim is limited to expression-text-driven execution. Native-library
bugs, large allocations, costly computation, permitted external references and
explicit caller-supplied UDF capabilities are not sandboxed. This experiment does
not establish a new security guarantee for all admitted adapter implementations.

## Verification and benchmarks

Initial two-phase local run: **11,334 passed, 55 skipped** (includes 43 new graph
tests). The narrowed phase-1 follow-up passed **11,338 tests, 55 skipped**. Ruff
lint/format and tracked whitespace checks passed for the follow-up changes.

Benchmark driver: `bench/ndarray/safer_lazyexpr.py`. The historical control uses
an immutable `git archive HEAD` source snapshot and the same compiled extensions,
not the modified branch's full-mode evaluator. Workers verify their import path,
use one computation thread, check outputs, and record construction, first/repeated
execution, reopening and process peak RSS. Inputs are compressible linspaces;
this is preliminary, not the full plan's workload matrix.

| Measurement | Initial observation |
| --- | --- |
| Float64, 1M elements, seven families, 9 repeats | Execution geometric mean ratio 1.043. Math/reopening initially approximately 1.29/1.30; these did not reproduce in the follow-up. |
| Float64, 1M elements, follow-up, 15 repeats | Execution geometric mean ratio 1.001; individual ratios 0.913–1.059. |
| Float32, 1M elements, 9 repeats | Execution geometric mean ratio 0.971; individual ratios 0.874–1.040. |
| Float64, 1K elements | Several execution regressions approximately 13–21%; construction overhead also visible. |
| Peak RSS, float64 follow-up | Ratios 0.744–1.571; math/indexing memory increases need dedicated measurement and explanation. |

Ratios are experimental/current divided by historical baseline; below 1 is
faster. These short local runs are noisy and do not establish statistical
significance or pass the complete performance gate.

## Outstanding completion gates

1. Finish the route and adapter inventory, including nested proxy/reference policy,
   deferred CTable materialized-expression reconstruction and all composition
   routes. Exact wrapper type admission alone does not establish closure safety.
2. Replace provisional signature families and reused dummy inference with precise
   per-operation schemas, keyword-value contracts and typed/shape-aware graph
   rules. Method/function normalization is not yet complete.
3. Prove once-per-evaluation reduction reuse and bounded-memory execution across
   chunks, partial reads and fallback routes; graph-local caching alone is not a
   scheduler-wide proof. Add allocation/materialization instrumentation.
4. Expand differential/property tests and bridge instrumentation across the full
   graph lifetime and all persistence containers. Validate WASM, platform and
   free-threaded behavior; only local macOS verification was performed here.
5. Complete the benchmark matrix: compression distributions, shapes, dtype
   families, partial reads, selection, strings, datetime/complex, persisted CTable
   recipes and multi-threaded workloads. Investigate RSS and small-array overhead.
6. Review the phase-2 API and compatibility exclusions before changing production
   defaults. Phase 2 is no longer enabled by default in this experiment.

Recommendation: retain as an experiment until those gates pass. The initial
numerical compatibility and large-array timings support further investigation,
but do not justify a merge or a completed two-phase claim.

## Follow-up: narrow phase 1 and close deferred-policy gaps

Following review, ordinary in-memory construction is trusted again. Safe graph
execution is established by safe loading or explicit experimental opt-in. A
safe ambient context takes precedence over a dependency's trusted construction
mode; safe-loaded expressions cannot be downgraded by entering trusted mode.
Programmatic constructors propagate a safe operand's boundary before coercion,
and validate the resulting dependency closure. Functionally numerical expressions
constructed in trusted mode can participate without making that context a
requirement for legacy execution.

Additional closure checks inspect NDField owners, ProxyNDField/Proxy sources,
SimpleProxy sources, portable inputs and Column dependencies. Legacy DSL table
dependencies reject; caller-authored UDF inputs use the same cycle guard. These
checks are a concrete improvement, not a completed audit of every storage adapter.

Loaded CTable computed recipes reconstruct with the effective loading policy.
Generated-expression metadata gets a runtime-only policy derived from storage,
not a persisted safety flag. Append/refill/refresh reconstruction uses that policy.
Safe loading validates both computed and materialized expression syntax. Reduction
and selection methods restore graph context; selection dependencies are admitted
before coercion and revalidated on later computation.

New regressions instrument the legacy eval bridge while reopening a graph outside
the safe test context, composing/selecting/reducing it, mutating it, and reopening
and appending to a table with computed and stored expression columns. The existing
minimal-protocol compatibility test again uses ordinary construction without an
explicit permission override.

No Menudet lowering was attempted: completing the graph boundary remains the
prerequisite. Precise operation contracts, metadata invalidation, full adapter
inventory, scheduler-wide reuse proofs and platform checks are still outstanding.

### Follow-up performance

Two sequential float64 runs (1K/1M elements, seven families, 15 repetitions) used
the same historical source snapshot and matching compiled extensions. Safe graph
execution versus historical execution at 1M elements had geometric mean ratios
**1.021** and **1.048**. These aggregates conceal reproducible family regressions:
centering ratios **1.232/1.273**, and persisted reopening/execution workload ratios
**1.196/1.388**. In the final run, repeated execution for centering was 5.70 →
7.26 ms and for the persisted workload 3.76 → 5.22 ms. The latter times measure
execution after reopening; the driver records open time separately.

Small-array safe execution also retains approximately 14–26% overhead in several
families. Ordinary construction now uses the compatibility route, but these
measurements exercise explicit safe graphs, not that restored ordinary default.
Peak RSS remains variable; no claim of a memory improvement is justified.

The repeated >10% regressions are an outstanding performance gate, not excused by
the geometric mean. Further profiling should separate recursive closure validation,
graph/dummy inference and scheduler reconstruction before adding another backend.
