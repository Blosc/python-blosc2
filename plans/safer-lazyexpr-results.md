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
