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

## Profiling checkpoint (after `5b582dfb`)

The phase-1 checkpoint was committed as `5b582dfb`. The subsequent profiling work
does not alter the runtime or weaken any validation.

The benchmark now supports `--families`, independent `--rounds` with alternating
baseline/safe process order, and worker-only `--profile-output`. Profiling is
enabled only around repeated evaluation, excluding construction, opening, output
checking and imports. Profiled timing results must not be used as uninstrumented
performance evidence.

### Isolated repeated-execution profiles

101 evaluations of each million-element workload, historical source control with
matching extensions:

| Profile | Total profiled seconds | Native reduction seconds | Compressed-output update seconds | Operand validation cumulative seconds |
| --- | ---: | ---: | ---: | ---: |
| Centering, safe | 0.699 | 0.263 | 0.263 | 0.010 |
| Centering, historical | 0.775 | 0.300 | 0.316 | n/a |
| Persisted expression, safe | 0.489 | n/a | 0.387 | 0.004 |
| Persisted expression, historical | 0.433 | n/a | 0.339 | n/a |

These profiles do not explain the previous large ratios as validation overhead:
native work dominates, and its timings vary between processes. Cumulative decorator
and dispatcher times include numerical execution; they are not boundary overhead.
The centering profile contains exactly 101 whole-array reductions for 101 repeated
evaluations, in both implementations. A new multi-chunk regression also asserts
one reduction per evaluation and recomputation after operand data changes.

### Longer, uninstrumented repeated-process measurements

Five sequential rounds, alternating process order, 50 repetitions per worker,
float64, 1M elements, the same historical control:

| Family | Median of per-round execution ratios | Per-round range |
| --- | ---: | ---: |
| Centering | 0.955 | 0.859–1.000 |
| Persisted expression execution | 0.948 | 0.811–1.007 |

The earlier >20% regressions did not reproduce under this longer protocol. This
does not prove an improvement: native timing variation, allocator/compression state
and process order remain confounders. Construction is still slower: round ratios
are 1.018–1.153 for centering and 1.176–1.273 for the persisted workload. The
small-expression overhead and RSS question remain open.

Conclusion: there is not enough evidence to justify a more complex validation
cache or a new engine as a performance fix. Retain the boundary, use the improved
measurement protocol for future changes, and prioritize precise operation/metadata
contracts and the remaining adapter/lifetime audit. No Menudet lowering yet.

Profiling follow-up verification: **11,339 passed, 55 skipped**; Ruff lint/format
and whitespace checks passed. Profiling artifacts remain in the approved temporary
directory; the runtime itself is unchanged from the committed checkpoint.

## Metadata validity follow-up (after `0c92a0d5`)

The profiling checkpoint was committed as `0c92a0d5`. The next implementation
step addresses cache correctness, not performance shortcuts or Menudet lowering.

Previously, safe construction could cache `_shape`/`_dtype`, then return those
values after rebinding a public operand, resizing an admitted NumPy operand, or
editing expression text. Persistence could also select `expression_tosave` and
`operands_tosave` from the original construction instead of the current recipe.

Safe graphs now record an input-metadata signature at construction and check it
at deferred entry points, including chunks/blocks and recipe export. The signature
includes expression text, bindings, admitted shape/dtype/partition metadata,
nested expression metadata and selection dependencies. It contains no array data
or computed results. Data-only changes therefore reevaluate without metadata
reconstruction. Changed signatures trigger inference through the existing validated
constructor before replacement caches become visible. Failed inference cannot
return old metadata. Trusted numerical dependencies entering a safe graph also
refresh potentially stale constructor caches.

After a metadata change, derived caches and the original persistence recipe are
invalidated. Disk, frame and nested structured encoding select the current recipe.
The ordinary trusted path's existing mutation behavior is unchanged.

New regressions cover rebinding before first computation, shape/text changes,
selection dtype changes, data-only updates, disk/frame/structured round trips,
trusted dependency metadata and incompatible rebinding. This is **not** a new
typed inference engine: precise per-operation rules and complete synchronization
of wrapper/source metadata are still outstanding. In particular, wrapper-owned
cached metadata and graph-dependent constructor caches need further audit. No
transactional snapshot or concurrency guarantee was introduced.

Verification: **11,348 passed, 55 skipped**. A repeated-execution centering profile
spent approximately 0.007 seconds in metadata refresh across 101 evaluations
(0.747 seconds total profiled execution). This is diagnostic, not a timing gate.
Three uninstrumented alternating-order historical-control rounds at 1M float64
elements, 50 repetitions each, gave centering median ratio 1.089 (range
0.896–1.125) and persisted-execution median 0.960 (0.836–1.029). Timing variability
remains material; no performance improvement or complete gate pass is claimed.

## Reduction/cast contracts follow-up (after `2158e7e0`)

The metadata checkpoint was committed as `2158e7e0`. The next slice replaces the
single broad reduction keyword set with explicit per-operation contracts shared
by parsing and direct dispatch. Contracts cover axis forms, duplicate axes and
arguments, keepdims, accumulation dtype descriptors, variance degrees of freedom,
and mutually exclusive ddof/correction. Known malformed literals reject before
operand/reference resolution. Bound argument values are checked before direct
backend dispatch. Parser-time checking reads literals only; it never executes an
expression or folds potentially expensive arithmetic to validate an argument.

NumPy `astype` contracts validate dtype, order, casting and boolean flags. They
are currently implemented only for admitted NumPy arrays; there is no registered
streaming Blosc2 cast, and no whole-array materialization fallback was introduced.
Safe validation now uses the graph parser rather than requiring agreement with
the legacy broad method-name filter. Ordinary trusted validation remains unchanged.

These are initial per-operation argument contracts, **not** complete type/shape
rules or a canonicalized reduction API. Positional layouts beyond common prefixes
remain receiver-dependent; NumPy and Blosc2 differ, so blindly normalizing them
would change existing semantics. Remaining keyword families, backend-specific
signatures, rank-dependent axis validation and metadata rules still need review.
Boolean flags are admitted as concrete Python/NumPy booleans, not arbitrary
truth-coercible objects. Object-dtype cast/accumulation targets remain excluded.

### A correctness bug exposed by the new contracts

Graph nodes previously keyed literal intermediates by `(literal, value)`. Python
considers `True`, `1`, `1.0` and `1+0j` equal as dictionary keys. Consequently a
cached integer could replace a boolean keepdims flag or a floating literal,
changing numerical promotion or making an otherwise valid reduction fail.

Literal nodes now include their concrete type in their immutable identity. Tests
verify exact scalar types, mixed-type promotion and keepdims round trips, while
also confirming that equivalent reduction nodes still reuse one intermediate
within an evaluation. No cross-evaluation result cache was added.

Verification: **11,377 passed, 55 skipped**. New tests cover malformed recipe
contracts before resolution, runtime bound-axis rejection before dispatch,
positive axes/keepdims/casts, persisted reduction round trips and typed literal
cache identity. No performance or cross-platform gate pass is claimed for this
slice; the remaining adapter/lifetime and operation-metadata audits remain open.

## Follow-up: backend-specific reduction binding and root metadata

Reviewed reduction layouts now distinguish NumPy, Blosc2 functions, array
methods and LazyExpr methods. In particular, NumPy's positional `out` is not
Blosc2's `keepdims`, and array versus LazyExpr `std`/`var` methods have different
`ddof` positions. Duplicate arguments, unsupported reviewed backend keywords and
rank-dependent axis bounds reject before dispatch. Explicit `np`/`numpy` and
`blosc2` reduction namespaces retain their selected backend rather than becoming
interchangeable aliases.

Safe construction normalizes positional arguments for root reductions on direct
named operands. Data-free shape and a limited dtype-rule subset bypass executing
reductions on tiny inference dummies, including otherwise valid `ddof` cases.
Normalization preserves typed arguments and leaves non-reduction grouping alone.
Full-permission safe-eligibility checks also consider reviewed root binding.

This is not complete graph-wide type inference: nested expressions, table-specific
receivers, cumulative operations, casts and other operation signatures still need
review. Half-precision fallback promotion, boolean extrema, complex statistical
reductions and integer statistical dtype overrides retain existing inference where
the independently specified rules are incomplete. Existing complex-extrema warning
failures are preserved, not silently redirected to NumPy.

Verification: **11,498 passed, 55 skipped**. Differential tests cover receiver
layouts, reduction dtype/shape families, namespace selection, axis validation,
output positions, persistence and data-free inference instrumentation. Ruff lint,
format and whitespace checks passed. No new benchmark or platform gate is claimed;
the experiment remains incomplete and is not ready to merge.

## Follow-up: cumulative-operation contracts and metadata

Reviewed `cumsum`/`cumprod` and `cumulative_sum`/`cumulative_prod` separately.
NumPy's newer cumulative functions accept keyword-only options; Blosc2 functions
and array methods accept positional dtype/include-initial, while LazyExpr methods
use positional include-initial without a dtype slot. The safe binder now preserves
those layouts and validates concrete boolean `include_initial` values, duplicate
arguments and scalar axes. Blosc2 cumulative `out` is not registered; NumPy's
supported output slot remains available.

Direct named-input root metadata now describes legacy flattening for `axis=None`,
the newer APIs' one-dimensional requirement when axis is omitted, and the
one-element shape extension for `include_initial`. Supported dtype/shape inference
does not execute cumulative work on dummy arrays. NumPy-qualified operations also
use NumPy dtype rules even when their input is a Blosc2 operand.

The current Blosc2 cumulative backend does not consistently honor explicit dtype
overrides. Those cases retain existing inference and execution semantics, covered
by differential tests; this change does not fix or silently reinterpret that
backend behavior. Nested graph metadata, other signatures, half-precision fallback
rules and complete streaming/allocation audits remain outstanding.

Verification: **11,580 passed, 55 skipped**. Added differential receiver/layout
tests, flattening and output-position checks, include-initial persistence, invalid
contract rejection, data-free inference instrumentation and operand-rebinding
metadata coverage. Ruff lint/format and whitespace checks passed. No new benchmark
or cross-platform completion claim is made.

## Follow-up: nested shape propagation and wrapper/source synchronization

Added data-free graph shape propagation for scalar/array inputs, elementwise
broadcasting, unary operations, comparisons and basic indexing. Reviewed
reductions now contribute shapes inside enclosing arithmetic/functions; function
reductions can also consume reviewed intermediate shapes. Positional binding
uses the selected backend rather than the old shape inferencer's universal slots.
Incompatible broadcasting rejects before dummy execution. Data-dependent indexing,
matrix multiplication, unreviewed receiver layouts and other unsupported rules
explicitly return unknown and retain existing inference. This is not graph-wide
dtype inference: nested dtype promotion and dummy-reduction removal remain open.

The initial adapter audit found stale source metadata in SimpleProxy and NDField.
After recursively admitting the source, safe validation refreshes SimpleProxy's
shape/dtype and rank-dependent partitions. NDField validation refreshes field
dtype, offset and partitions after parent rebinding, or rejects a removed field.
Hostile rebound sources and cycles reject before metadata or compute hooks,
including under an ambient full evaluation context. Ordinary trusted-only
construction has not acquired this synchronization policy.

The scheduler's indexing-to-`.slice` rewrite also exposed a NumPy adapter mismatch:
NumPy has no NDArray-style slice method. Safe dispatch now uses direct indexing
for exact admitted NumPy arrays, with one index and no constructor keywords.
This introduces no whole-array conversion or streaming Blosc2 cast fallback.

Verification: **11,598 passed, 55 skipped**. Tests instrument shape inference
against data reads, compare nested reductions/indexing against NumPy and exercise
wrapper resizing, rebinding, changed field layouts, removed fields and cyclic or
hostile sources. Ruff lint/format and whitespace checks passed. Caching proxies,
table/storage/remote adapters, full persistence reachability and allocation bounds
still require audit; this slice does not establish complete adapter lifetime safety.
Construction overhead and cross-platform behavior were not newly measured.

## Follow-up: nested unary dtypes and caching-proxy admission

Added a narrow data-free dtype traversal for numeric named inputs, basic indexing,
reviewed direct-input reductions and selected unary numerical calls. Unary output
types use NumPy ufunc dtype resolution, not evaluation of synthetic values. With
reviewed shape and dtype rules, construction now skips both dummy execution and
slice-placeholder creation. Nested cases such as `sqrt(x.std(axis=0, ddof=2))`
therefore no longer perform invalid reductions on tiny all-one dummies. Unsupported
binary promotion, weak scalar rules, intermediate reduction receivers, casts and
backend-specific exceptions retain existing inference. This remains a subset, not
a complete independent graph type system.

Proxy construction records the source owning its cache. Safe admission rejects
subsequent source rebinding, including equal-metadata replacements and rebinding
before first expression admission. Source/cache shape, dtype, chunks and blocks
must agree; the cache must be an admitted concrete NDArray. Failures occur before
fetching, with no cache writes, discard or automatic reconstruction. The diagnostic
asks callers to construct a new proxy. ProxyNDField metadata now refreshes after
rebinding to a valid proxy and rejects missing fields.

These checks establish neither cached-data freshness nor transactional snapshots.
They do not replace the proxy's existing stamp/refresh policy, certify adopted
persistent-cache contents, or complete the remote-source refresh/lifetime audit.
Table/storage/remote adapters and full persistence/container reachability remain
open. Ordinary trusted-only evaluation retains its compatibility behavior.

Verification: **11,660 passed, 55 skipped**. Differential unary/reduction dtype
tests cover NumPy and Blosc2 inputs, boolean/integer/float/complex families, metadata
rebinding and persistence. Instrumentation rejects dummy execution and slice
placeholder creation for reviewed cases. Proxy tests cover source identity,
resizing, incompatible partitions/dtypes, hostile caches, parent-field rebinding
and pre-fetch rejection under ambient full mode. Ruff lint/format and whitespace
checks passed. No new performance, allocation-bound or platform gate is claimed.

## Follow-up: NumPy binary promotion, numeric casts and deferred column closure

Added data-free binary dtype resolution for reviewed NumPy-only arithmetic and
comparison trees, including basic indexed inputs and numeric casts. Python int,
float and complex operands retain weak scalar promotion, while concrete NumPy
scalars retain their dtype. Weak integers are checked against resolved input
ranges; finite float/complex scalar overflow retains existing warning/fallback
inference rather than silently changing promotion. Blosc2 binary trees still use
existing inference: notably int32 division remains float32 there versus NumPy's
float64. Mixed/backend-changing call trees remain unreviewed, not normalized to
one universal promotion API.

NumPy astype now binds its exact dtype/order/casting/subok/copy layout, rejects
duplicate positional/keyword options, and validates numeric output dtype and
can-cast rules without synthetic execution. Fixed-shape numeric casts preserve
shape. Flexible strings, structured/subarray casts and other incomplete output
rules retain existing inference. No streaming Blosc2 cast was registered, and
whole-array conversion is not a fallback for unsupported Blosc2 receivers.

Column admission now checks validity and selection-mask dependencies plus the
actual cached computed LazyExpr and its input closure, not only the descriptor's
expression text and named column dependencies. Hostile cached objects reject
before dtype hooks. Remote HDF5 field admission recursively validates the records
source before synchronization; rebinding updates inherited metadata while
preserving explicitly supplied logical dtype overrides, including overrides that
initially equal the physical storage dtype.

Verification: **11,713 passed, 55 skipped**. Differential tests cover binary dtype
pairs, broadcasting, weak/concrete scalars, division backend differences, casts
and invalid options. Instrumentation verifies reviewed metadata needs no dummy
execution. Adversarial tests cover cached column expressions, their inputs,
visibility dependencies and remote-field source rebinding. Ruff lint/format and
whitespace checks passed.

This is not completion of binary/cast contracts or table/storage/remote audits.
Backend-changing intermediate types, remaining casts, persisted operand-type
semantics, table recipe/cache consistency, Parquet owner/dependency lifetime,
remote refresh behavior and bounded-memory execution still need review. No new
performance or cross-platform gate is claimed.

## Follow-up: field-reference persistence and Parquet lifetime admission

Persistence review found that NDField operands were previously written as references
to their entire structured parent, losing the field selector. The experimental
operand recipe now has an explicit `ndfield` version-1 record containing the field
name and the ordinary parent reference. Disk, frame and structured reconstruction
restore an NDField rather than an NDArray. Relative parent references support
relocation and relative source/carrier paths; missing parents retain the standard
MissingOperands diagnostic. Invalid selectors reject before resolving their parent,
reference decoding is bounded, and nested field parents reject.

This is a new experimental operand-recipe spelling: older readers will not know
it. It does not migrate historical files that already lost their selector, alter
the parent reference's safe/full policy, or bypass legacy proxy loading gates.
NumPy-only operand persistence is already unsupported and rejects before writing
the destination; there was no silent NumPy-to-Blosc2 promotion conversion to fix.

Parquet column admission now verifies concrete storage/discovery/cache ownership,
original storage identity, generation/closed-state checks and schema metadata
agreement before reads. Scalar Parquet-backed public Column views are also admitted
through their checked RemoteCTable storage dependency. Rebound storage, hostile
owners/caches and closed or stale handles reject before row-group reads. Variable-
length and ndarray-row columns explicitly reject until their numerical expression
adapter and row-shape semantics are reviewed; they are not treated as scalar rows.

Verification: **11,735 passed, 55 skipped**. Added field round trips under safe and
full permission, receiver/dtype preservation, relative-path relocation, malformed
selectors, missing parents and legacy parent-policy tests. Parquet tests cover
data-free admission of raw/public scalar columns, closed/stale/changed metadata,
hostile ownership, equal-metadata storage rebinding and unsupported ndarray rows.
Ruff lint/format and whitespace checks passed.

Remaining gates include intermediate/mixed-backend type rules, other casts and
wrapper persistence, table recipe/cache synchronization, remaining remote adapter
closures, refresh races and allocation/streaming proofs. This slice does not certify
Parquet cache contents, resource bounds, external-reference authorization or
concurrent snapshots. No new performance or cross-platform gate is claimed.

## Local experiment assessment: remote closures, cache lifetime and differential probes

The field/Parquet checkpoint is committed as `aa2f3372`. Checkpoint `b96cc2be` checks
RemoteArray's concrete transport, store owner, cache proxy and carrier closure
before metadata or data access. Equal-metadata source rebinding rejects; supported
standalone refresh replaces the source provenance together with its cache, and an
existing safe expression remains usable. Closed/stale handles keep their existing
diagnostics. Admission tests cover NONE/MEMORY/DISK caching, repeated evaluation,
blocked transport chunk reads and hostile metadata hooks.
These checks do not authorize external references or guarantee atomic refresh.

Constructor review uncovered operand-dependent results retained in `cons_cache`
across evaluations. Safe evaluation now reconstructs those values each time while
retaining constant-only constructor caching. Cached constant results are admitted
again before use. A NumPy operand mutation regression covers explicit `asarray`
conversion; its simple shape rule now avoids the legacy constructor shape parser's
unsupported case. This is explicit conversion requested by the expression, not a
materializing fallback for an unsupported cast. Trusted constructor caching is
unchanged. Unregistered computed-column recipe kinds now reject explicitly.

The deterministic differential corpus adds 48 combinations of seeded nested
expressions, float32/float64 and disk/frame/structured reconstruction. It exercises
shape/dtype, values, partial slices, NumPy ufunc composition, source value mutation
and ambient full mode with the Python text bridge disabled. Dtype comparisons use
the existing trusted backend contract, not NumPy's potentially different reduction
accumulation dtype. Another 24 forbidden-syntax variations reject before operand
resolution. Three allocation probes reject whole-source NumPy coercion and bound
Python-visible chunk reads for partial arithmetic, reductions and centering. They
are not a proof of native peak memory, every schedule or every expression family.

Final local verification for `b96cc2be`: **11,837 passed, 55 skipped** in the
`blosc2` conda environment. Ruff lint/format and whitespace checks passed. This
is the default suite, not heavy/network or cross-platform certification.

### Historical-control performance assessment

Five alternating-order rounds per workload, 31 repeated evaluations per process,
one thread, 1,000 and 1,000,000 elements, float32 and float64, macOS ARM64. Control:
the same immutable pre-experiment source snapshot and matching compiled extension
used in earlier measurements. Raw artifacts are `safer-final-f64.json` and
`safer-final-f32.json` in the approved OpenCode temporary directory. These runs
precede the final RemoteArray/constructor changes; measured workloads do not use
those routes. Ratios are safe/historical; lower is faster. Family figures below
are medians of five paired process ratios, not confidence intervals.

| 1M-element family | float64 execution ratio | float32 execution ratio |
| --- | ---: | ---: |
| Arithmetic | 0.987 | 0.996 |
| Math | 1.037 | 1.037 |
| Reduction | 1.022 | 1.027 |
| Centering | 0.922 | 1.037 |
| Axis reduction | 1.018 | 1.067 |
| Indexing | 0.986 | 0.961 |
| Persisted expression | 1.008 | 0.961 |

The geometric mean of all paired large-array ratios is **1.003 for float64** and
**1.026 for float32**, within the provisional 5% budget for this limited matrix.
Individual rounds remain noisy (some exceed 10%); no statistical speedup or
complete family/platform gate is claimed. Tiny-array median execution ratios are
1.048–1.330 for float64 and 1.053–1.338 for float32, adding roughly **26–77 us**
and **27–71 us** respectively. Construction is generally slower, except reviewed
axis metadata; it is not hidden in execution timings.

Peak RSS remains a failed/unresolved review gate: median paired indexing ratios
are **1.285 float64** and **1.272 float32** at 1M elements, and about **1.21** at
1,000 elements. Float32 arithmetic is **1.153**. Whole-process high-water RSS does
not locate the allocation or establish asymptotic growth, but these differences
cannot be dismissed from the small Python-visible allocation probes.

### Verdict and remaining gates

**The experiment works as a bounded proof of concept, not as a completed phase-1
security certification or a production-default replacement.** It demonstrates
eval-free reconstruction/execution for admitted numerical graphs, boundary
retention across deferred work and composition, useful rejection diagnostics,
backend-specific numerical compatibility and promising large-array throughput.
Ordinary construction remains trusted; `deserialize='safe'|'full'` is retained.
No default switch or Menudet lowering is justified by these measurements.

| Gate | Local outcome |
| --- | --- |
| Supported graph lifetime and numerical regressions | Covered by targeted and seeded tests; not exhaustive |
| Safe text bridge isolation | Instrumented on supported routes, not whole-project sandboxing |
| Metadata/signature coverage | Partial; mixed/backend-changing intermediates, weak scalars and casts remain |
| Persistence/adapters | Field/Parquet/RemoteArray slices covered; remaining wrappers and container reachability need audit |
| Table consistency | Unknown recipes reject; broader recipe/cache synchronization remains |
| Throughput | Limited large-array matrix meets geometric-mean target; tiny-expression overhead remains |
| Memory | Not passed; indexing RSS requires explanation and broader native allocation instrumentation |
| Concurrency and platforms | Refresh races, free-threaded, WASM and cross-platform safe-graph verification remain unverified |

Finishing those gates requires additional implementation, audit and measurements;
passing the local suite cannot substitute for them. Keep this branch experimental
and phase 2 deferred. Menudet publication and compatibility freeze remain separate.

## Follow-up assessment: release cycles, mixed contracts and native-kernel closures

Checkpoint `e84afd70` resolves a concrete evaluation lifetime defect: the recursive
`ExpressionGraph.evaluate` visitor closed over itself, its operands and its local
array cache. Repeated evaluations therefore retained arrays until cyclic GC ran.
The evaluator now clears the cache and breaks the recursive closure in `finally`,
on both success and failure. Weak-reference regressions with cyclic GC disabled
verify that operands are released while a successful independent result remains
usable. Existing once-per-evaluation common-subexpression/reduction reuse is kept.
This is deterministic release of graph-owned references, not a general guarantee
about allocator high-water RSS or caller-retained exception tracebacks/views.

Computed expression columns now compare descriptor and cached graph roots, operand
names and actual stored-column identities after admitting the cached dependency.
Unilateral expression/dependency/cache replacement rejects with a recipe/cache
mismatch instead of using stale numerical semantics. Supported append/delete
operations keep an existing safe graph live with current values and shape. Explicit
computed-column dtype overrides remain permitted; no universal dtype normalization
or automatic materializing rebuild was introduced.

Added differential contracts cover 72 combinations of six nested/reduction
expressions, int32/float32/float64 and all NumPy/Blosc2 receiver pairs, plus 40
weak-scalar cases over five dtypes and four literals. Safe/trusted values, shape,
dtype and integer-overflow outcomes agree in these cases. This extends empirical
coverage; it does not complete the mixed-intermediate static metadata rule set.
Exploratory probes also confirm existing unsupported shape-changing methods on
computed LazyExpr receivers and the trusted NumPy `.slice` rewrite limitation.
Those are not repaired by whole-array materialization or treated as parity successes.

### Updated memory and throughput measurements

Same historical source control, one thread, five alternating-order rounds and 31
repeated evaluations per process. Artifacts in the approved OpenCode temporary
directory: `safer-cycle-index.json`, `safer-cycle-f32.json`, `safer-cycle-f64.json`.
No other suite/benchmark was run concurrently with the measurement workers.

The dedicated float64 indexing scaling sweep now has median paired RSS ratios
**1.007, 1.015, 1.017 and 1.015** at 1K, 100K, 1M and 4M elements. Median execution
ratios are **1.025, 0.996, 1.008 and 1.001** respectively. The previous reproducible
indexing excess is no longer present in this matrix after the closure cleanup.
These are medians of paired ratios, not ratios of unpaired median RSS values.

| 1M-element family | float64 execution | float32 execution | float64 RSS | float32 RSS |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic | 1.064 | 1.054 | 1.142 | 0.938 |
| Math | 1.033 | 1.004 | 0.991 | 0.960 |
| Reduction | 1.015 | 1.034 | 0.992 | 1.064 |
| Centering | 1.082 | 0.976 | 1.101 | 1.065 |
| Axis reduction | 1.008 | 1.054 | 0.949 | 0.964 |
| Indexing | 0.993 | 0.992 | 0.883 | 0.978 |
| Persisted expression | 1.013 | 0.982 | 0.976 | 1.012 |

The geometric mean over all paired 1M-element execution ratios is **1.037 float64**
and **1.018 float32**, still within the provisional 5% budget for this matrix.
Tiny-array median added latency is roughly **15–79 us float64**, **12–66 us float32**;
indexing overhead fell, but the other small-expression costs remain. No statistically
significant speedup is claimed. Whole-process RSS is still variable: the float64
arithmetic/centering families are higher in this run, despite lower or near-equal
float32 figures. The indexing defect is explained and addressed, but a complete
native-allocation/resource-bound gate is still not passed.

### Portable operand audit

Checkpoint `a144fc6f` implements the following native-operand checks.

Safe graph admission previously checked PortableLazyArray inputs without checking
its actual kernel dependency; portable computed-column caches had the same gap.
Admission now checks exact PortableKernel/native-handle types, original artifact
and handle provenance, and cached metadata against fresh native-handle metadata.
Metadata comparison rejects foreign types before equality hooks. Portable table
cached kernels must also agree with their recipe artifact. Tests cover hostile
kernel/handle/metadata objects, same-type native-handle rebinding, artifact mutation
and cached table kernel/artifact disagreement, before hooks or evaluation.

PortableLazyArray admission also checks concrete input mappings, numeric
domain/partition extents, lane limits, context rank, cached output/grid geometry,
native input names/dtypes and broadcasting against the declared domain. Portable
column bindings/row metadata are checked before native descriptor helpers; current
stored-column dependencies and scalar row contracts are revalidated. Validated
cached artifacts reuse their owned native handle instead of recompiling during
admission. Instrumentation rejects accidental native artifact reconstruction on
first/repeated cached-column evaluations.

Final verification: **11,973 passed, 55 skipped**, default suite in the `blosc2`
conda environment. Ruff lint/format, pre-commit checks and whitespace validation
passed. The performance workers above measured the cycle/cache checkpoint; their
ordinary numerical workloads do not exercise the subsequent portable-wrapper
checks. Native-wrapper admission overhead has not been benchmarked separately.

This adds no Python-source reconstruction, native lowering, execution fallback or
persisted-format change. Native artifacts remain native-only. Basic portable
domain/partition/binding consistency is now covered; full wrapper/container
reachability, broader table-cache consistency, allocation bounds and concurrent
mutation guarantees remain unverified.

### Revised verdict

**The bounded experiment continues to work, with a materially stronger memory and
dependency-lifetime result. It remains incomplete against the full phase-1 exit
criteria.** Keep the experimental opt-in and safe/full permission distinction.
Remaining work is now: complete backend/weak-scalar/cast rule coverage and explicit
capability review; finish wrapper/container and broader table-cache audits;
profile small-expression overhead and native allocations; verify refresh/lifetime
races under the documented non-transactional contract; and certify additional
platforms, free-threaded Python, WASM and the wider benchmark matrix. Local green
tests cannot stand in for those unverified gates. No default switch, merge or
Menudet publication decision follows from this checkpoint.

## Follow-up: intermediate reductions, containers and concurrency/platform evidence

Implementation checkpoint: `e0c36357`. Final default-suite verification:
**12,017 passed, 55 skipped** in the `blosc2` conda environment. Ruff lint/format,
workflow YAML parsing, pre-commit checks and whitespace validation passed.

Reviewed NumPy binary/index/cast intermediates now provide a metadata-only receiver
for reductions. This preserves NumPy-method versus Blosc2-function positional
layouts and accumulator dtype rules without constructing an array or executing a
dummy numerical operation. Direct named receivers retain their existing path.
Mixed/Blosc2-changing intermediates still defer to the existing reviewed execution
and inference where their static rule is unknown, rather than adopting NumPy's
semantics. Tests cover nested sum/mean, explicit NumPy std with ddof and cast-then-sum.
The latter two forms are checked against NumPy directly: the trusted validator or
dummy inference does not support them, so they are not called parity successes.

The platform review identified NumPy 1.x's value-dependent scalar/0-D promotion.
Static binary dtype rules now defer those cases to backend inference on NumPy 1.x,
instead of applying NumPy 2.x's NEP 50 rules or raising an incorrect early integer
overflow. The guard branch is tested locally; actual NumPy 1.26 verification is
pending its existing CI matrix entry. Array/array dtype rules are retained.

### Container reachability

DictStore/TreeStore assignment of a LazyExpr failed in the size estimator because
LazyExpr has no `nbytes` property. Logical shape and normalized dtype now estimate
the storage tier without evaluating the expression. Dtype normalization also
preserves existing LazyUDF recipes that specify dtype as a string or scalar type.
No generic array conversion, operand materialization or UDF loading-policy bypass
was introduced.

Disk/ZIP DictStore and TreeStore, plus EmbedStore, now have explicit graph-lifetime
tests under both safe/full loading permissions, including embedded and threshold-
externalized leaves. Writer instrumentation rejects expression reads/materialization;
reader instrumentation rejects the Python text bridge. Safe graphs remain safe
through partial reads and ufunc composition under ambient full mode. Malformed
expression carriers in all five container configurations reject before operand
reference resolution. This covers those leaf routes, not every heterogeneous,
recursive container/reference arrangement or crash-recovery behavior.

### Concurrency contract and probes

RemoteArray admission now uses the existing owner-lock then operation-lock order
to check its transport/cache closure coherently with supported refresh. Concrete
native RLocks are checked before context hooks; replaced locks reject. Refresh
already preserves its operation-lock identity while swapping source state, so no
new global lock or transactional snapshot mechanism was added.

Controlled Event-based tests block an active safe read and overlap standalone
refresh for NONE/MEMORY/DISK caching, then verify read completion and subsequent
safe evaluation. A shared, safe-loaded expression also runs on four threads under
mixed ambient policies; composition remains safe and thread-local policy is restored.
These are deterministic ordinary-thread probes on GIL-enabled CPython, not an
exhaustive race proof or permission to mutate expression mappings/table recipes
concurrently. An evaluation may observe permitted operand changes between reads;
there is still no transactional snapshot guarantee. Native thread tests explicitly
skip on single-threaded WASM runtimes; the skip is reported, not counted as evidence.

### Platform verification harness

`scripts/verify_safe_lazyexpr.py` runs the bounded graph/property/container/portable
corpus without xdist and emits Python/NumPy/native versions, actual package path,
free-threaded-build and effective GIL state, test counts and skip/failure reasons.
It neither forces `PYTHON_GIL=0` nor equates a free-threaded build with no-GIL safety.
The native Linux/Windows/macOS/NumPy-1.26 workflow uploads its JSON report; wheel and
WASM/Pyodide workflows also run the harness. Those jobs are prepared, not executed
for this unpublished checkpoint.

Local harness evidence: **710 passed, no skips**, macOS ARM64, CPython 3.14.4,
NumPy 2.5.3, C-Blosc2 3.3.5; ordinary build, GIL enabled. Artifact:
`safer-platform-local.json` in the approved OpenCode temporary directory.

| Requested verification | Current status |
| --- | --- |
| Local numerical/metadata and container corpus | Tested; static compatibility coverage remains partial |
| Shared readonly graphs and RemoteArray read/refresh overlap | Tested with ordinary threads and three cache policies |
| Concurrent table/Parquet refresh, expression mutation and arbitrary adapter races | Not certified |
| Linux/Windows/macOS and NumPy 1.26 | CI harness wired; only local macOS/NumPy 2.x executed here |
| Free-threaded interpreter builds | Wheel harness records actual GIL fallback; external results pending |
| Actual no-GIL execution | Unsupported certification claim; never forced or inferred |
| WASM/Pyodide | Harness wired; external results pending; unavailable thread probes reported as skips |

This advances all four requested areas but does not close the full experiment gate.
The remaining static/backend rules, complete heterogeneous reachability review,
native resource bounds and external platform/race evidence must still be reviewed.
Defaults and the existing safe/full distinction remain unchanged.
