# Make typed expression evaluation the default and retire uniform execution

## Decision

Move ordinary expressions permanently to typed evaluation. Reuse the existing
Menudet graph/artifact engine and storage schedulers; do not repair or maintain a
second numerical type system in the uniform-dtype evaluator.

Switch the default after a focused qualification pass, not after achieving native
coverage for every layout. Retain a temporary legacy escape hatch during the
transition, then delete ordinary-expression uniform execution and the selector.

Explicit compatibility DSL kernels have a separate language contract and are not
silently migrated or removed. Python UDF execution contracts remain unchanged.

This plan supersedes the rollout sequence in
`plans/retire-uniform-expression-evaluator.md`; that document retains useful
implementation details, historical measurements, and regression inventory.

## Starting point

- The typed route exists behind `BLOSC2_EXPRESSION_EVALUATOR=typed`.
- Typed artifact prefilters support admitted flat C-order N-D layouts, native
  block reductions, candidate-block skipping, and whole-chunk pruning.
- Unsupported signatures/layouts use bounded typed graph or NumPy chunk execution,
  without replay through the uniform evaluator.
- Local full suites pass in both modes. Cross-platform native, NumPy 1.26,
  Pyodide, wheel/ABI, and pre-commit checks passed for the migration changes.
- Most platform CI still used the legacy default; targeted typed tests do not
  establish full-suite typed-mode platform qualification.
- Representative maps, constructors, and reductions show performance parity.
  Some use shared automatic native dispatch in both modes, so these measurements
  do not independently qualify every replacement path.

## Numerical and execution policy

### NumPy 2 and newer

Use typed native execution when admitted, otherwise bounded typed graph/NumPy
evaluation. Preserve intermediate operation dtypes independently of output and
reduction accumulator dtypes, strong NumPy scalar types, and exact integers.

### NumPy older than 2

Evaluate ordinary numerical expressions with the installed NumPy, retaining
Blosc2's bounded storage traversal and reduction scheduling. Do not implement
NumPy 1.x value-dependent promotion in the native compiler, and do not silently
substitute NumPy 2 semantics under NumPy 1.x.

Audit all automatic ordinary-expression entry points—including `LazyExpr._new_expr`,
`_prepare_cached_evaluate`, `prepare_resident`, `compute_resident_if_eligible`,
`compute_native_prefilter_if_eligible`, `prepare_native_first`,
`_prepare_typed_map_kernel`, and `evaluate_chunk`—so they decline native execution
when a shared NumPy < 2 check (`_NUMPY_VALUE_PROMOTION`) is active. Existing
prefilter declines alone are not a complete guarantee of this policy. Leave
`lower_native_graph` available for explicit native-only APIs (`LazyExpr.native_kernel()`
and `LazyExpr.compute(_require_native=True)`), which carry an explicit Menudet/NumPy-2.x
semantic contract. Dtype/shape metadata probes must agree with execution without
reading full operands. Preserve documented Blosc2 exceptions and test them explicitly.

This policy does not require decompressing entire arrays. Reduced native
acceleration on NumPy 1.x is acceptable and must be documented. Keep a compatible
NumPy 1.26 CI runtime to test the integration contract rather than native parity.

### Pyodide

Qualify the supported modern NumPy runtime using actual Pyodide wheels. Use the
typed interpreter/qualified JIT or NumPy chunk fallback as available. Typed WASM
compression prefilters are not a prerequisite for the default switch.

Keep platform-specific dtype policies explicit. Do not assume desktop threading,
JIT, address-space, or allocation behavior applies to WASM. Prioritize bounded
memory and measure extra buffers/copies in fallback paths.

### Shared contracts

- Ordinary expressions must never reach the uniform evaluator in typed mode.
- Preparation may decline before numerical execution; execution failures must
  propagate without replay after partial writes.
- Preserve output identity, supported aliases/strides, slices, broadcasting,
  empty identities, reduction dtypes, and reader cleanup on success/failure.
- Numerical results should follow installed NumPy semantics, subject to existing
  documented policies and justified floating-point reduction-order differences.

## Phase 1: Close the focused qualification gap

1. Implement/audit the NumPy < 2 policy consistently across automatic
   ordinary-expression dispatch while preserving explicit `native_kernel()` /
   `compute(_require_native=True)` entry points. Add tests that prove automatic
   native numerical execution is bypassed on NumPy < 2 while storage scheduling
   remains bounded and explicit native-only APIs continue to work.
2. Run full typed-mode native CI across Linux, macOS, Windows, current NumPy, and
   the supported NumPy 1.26 runtime. Exercise interpreter and supported JIT
   configurations with representative integration tests, not redundant exhaustive
   Python numerical matrices.
3. Run the complete applicable suite in actual Pyodide with typed globally
   selected, not only host-simulated WASM restrictions or targeted typed tests.
4. Keep explicit compatibility DSL coverage separate. Retain tests that prohibit
   ordinary typed expressions from calling the uniform prefilter/reducer.
5. Verify installed wheel imports and extension capability handling; do not rely
   solely on an editable checkout's native library.

Native operator/dtype matrices belong in miniexpr. Python tests own bindings,
dispatch, scheduling, metadata, output/storage contracts, and representative
end-to-end numerical regressions.

**Exit:** full typed-mode platform runs have no unexplained failures; NumPy 1.x
fallback and real Pyodide integration are demonstrated. Do not weaken numerical
or storage assertions merely to make the suite green.

## Phase 2: Switch the development default

Originally gated on Phase 1 passing; the user subsequently selected a
switch-first development rollout (see the status below). Make typed the default
and qualify that revision in CI. Keep
`BLOSC2_EXPRESSION_EVALUATOR=legacy` temporarily for diagnosis and rollback.

- Run ordinary CI under the new default; retain a limited explicit legacy job
  while the escape hatch exists.
- Update migration documentation and release notes with the default change,
  NumPy < 2 fallback policy, and temporary escape hatch.
- Exercise examples, tutorials, downstream workloads, and release-candidate wheels.
- Investigate differences as bugs or explicit policy decisions, not by adding
  automatic uniform-evaluator fallback.

Do not wait for universal native layout coverage or an acceleration win on every
benchmark. Correct bounded fallbacks are valid implementation choices.

## Phase 3: Target performance and memory risk during qualification

Run a focused report alongside the default-switch qualification period. Start
with paths most likely to differ, rather than exhaustively benchmarking shared
native fast paths:

| Workload | Main risk |
| --- | --- |
| Non-flat N-D maps and edge chunks | Lost fusion, copies, chunk scratch |
| Persisted arrays and nested evaluation | Reader lifetime, IO, prefetch buffers |
| Broadcasting, partial slices, C/F/strided outputs | Scheduling and temporary allocations |
| Scalar/axis reductions, std/var, cumulative operations | Map/accumulator typing, partials, scheduling |
| Masks, selections, indexed queries, candidate pruning | Work skipping and cardinality handling |
| Strings, complex, datetime, float16 | Fallback performance and dtype metadata |
| Actual Pyodide maps/reductions | Extra copies, memory pressure, interpreter costs |
| NumPy 1.26 ordinary expressions | Correct bounded fallback, documented slowdown |

Use interleaved legacy/typed measurements with identical inputs/storage settings.
Record actual route/admission and capability information, not just `jit` versus
`interp`. Separate preparation-cold from warm measurements; report elapsed time,
peak memory, scratch buffers, thread settings, and correctness checks. Distinguish
shared-route parity from replacement-route evidence.

Agree on workload-specific acceptable time/memory limits before accepting the
report. Treat repeatable large regressions, unbounded temporaries, and broken
contracts as blockers; do not block on timing noise or small isolated slowdowns.
Optimize demonstrated gaps, or explicitly document acceptable tradeoffs.

**Exit:** no unresolved major performance/memory regressions in representative
replacement paths, and no correctness/lifetime regressions during a short
development/release-candidate qualification period. Universal native acceleration
is not an exit requirement.

## Phase 4: Delete ordinary-expression legacy execution

After the typed default holds up under Phases 2–3:

1. Remove ordinary-expression uniform map/prefilter and reduction branches.
2. Remove guards, literal specialization, caches, and helpers needed only by those
   branches. Audit shared callers before deleting code used by explicit DSL
   compatibility or string-width metadata.
3. Remove legacy implementation-detail assertions while retaining numerical,
   storage, no-replay, scalar-promotion, boolean reduction, and reader-lifetime
   regression tests.
4. Remove `BLOSC2_EXPRESSION_EVALUATOR`, selector-dependent branches, and obsolete
   legacy CI. Document removal of the temporary setting.
5. Rerun native/Pyodide suites, wheel checks, and the focused performance/memory
   report against the pre-deletion typed default.

**Exit:** ordinary expressions have no reachable uniform numerical execution path;
NumPy < 2 uses the explicit bounded NumPy policy; all qualification checks pass.

## Phase 5: Separate compatibility cleanup

Inventory any remaining uniform-evaluator dependencies in explicit DSL kernels
and metadata queries. Migrate or deprecate those through their own documented,
tested contracts. Ordinary-expression retirement is complete without pretending
that every internal uniform-evaluator dependency has also disappeared.

## Immediate next action

Audit NumPy < 2 dispatch and add full typed-mode native/Pyodide CI. If green,
switch the development default and collect targeted fallback-path performance and
memory evidence during the qualification period. Keep deletion separate from the
default switch so failures remain easy to isolate.

## Implementation progress (2026-11-04)

### Phase 1 implementation

- Centralized pre-2 promotion decisions on `_NUMPY_VALUE_PROMOTION`.
- Automatic ordinary-expression metadata preparation, resident execution,
  native-first dispatch, native prefilters, and chunk evaluation now decline
  native execution on NumPy < 2. Ordinary uniform prefilters/reducers also decline
  there, even when the temporary selector requests legacy.
- Kept bounded metadata recipe caching: NumPy fallback does not require disabling
  reusable, operand-free frontend metadata.
- Preserved explicit `native_kernel()` / `_require_native=True` entry points and
  explicit Menudet function/kernel execution. The shared resident preparer has an
  explicit-kernel opt-in so automatic admission policy does not disable those APIs.
- Added policy tests for resident/cached evaluation, persisted and nonpersisted
  chunk traversal, reductions, partial slices, strided outputs, scalar promotion,
  data-free metadata, and explicit native-only execution.
- Separated automatic-native implementation tests from NumPy-fallback numerical
  contracts. NumPy 1.x no longer has to satisfy native route/cache/JIT assertions;
  explicit native APIs and ordinary numerical/storage contracts remain tested.
- Native CI now selects typed globally, with a limited legacy escape-hatch job.
  Actual Pyodide and wheel qualification also select typed; the dedicated Pyodide
  workflow asserts that selection inside the test runtime.

Local NumPy 2.5.4 validation: full typed and legacy suites each **12,675 passed,
98 skipped** using four workers. The full single-process fallback-policy probe
passed **12,048 tests, 725 skipped** with pre-2 admission flags enabled. That
probe requires a single process to retain the injected flags and runs on modern
NumPy: it validates dispatch/contracts, not actual NumPy 1.26 numerical
qualification. Automatic-native route assertions account for the additional
policy skips; ordinary numerical and explicit native-only tests still run.

**At the Phase 1 checkpoint, exit remained pending:** new actual cross-platform,
NumPy 1.26, Pyodide, and installed-wheel CI results had not been collected for
these changes. The default was still legacy at that checkpoint; the subsequent
switch-first decision and Phase 2 status are recorded below. Phase 4 deletion
has not started.

### Performance qualification tooling

`bench/ndarray/evaluator_fallbacks.py` adds interleaved legacy/typed workloads for
non-flat N-D maps, persisted inputs, broadcasting, slices, strided outputs,
scalar/axis reductions, variance, selections, complex, and float16. It validates
results before timing, observes uniform/typed prefilters and NumPy/chunk calls,
and records native reports where available. Allocation probes are separate from
timing and explicitly report only tracemalloc-visible peak bytes—not complete
native scratch or process RSS. A small smoke run and its integration test pass.

Remaining performance evidence includes broader workload/layout runs,
native/process-memory measurements, strings/indexed queries, and actual Pyodide
memory/performance. Acceptance thresholds still need
agreement before accepting a qualification report.

### Measured non-flat-layout regression: initial evidence

Idle-host interleaved measurements (macOS arm64, conda `blosc2`, NumPy 2.5.4,
12 threads, seven rounds) now demonstrate that shared-fast-path parity does not
extend to all fallback layouts. All measured results passed NumPy comparisons.

For a `(2048, 2048)` float32 input, chunks `(127, 191)`, blocks `(31, 47)`:

| Operation | Legacy median | Typed median | Typed / legacy |
| --- | ---: | ---: | ---: |
| Non-flat `.compute()` | 51.74 ms | 87.02 ms | 1.68x |
| Persisted `.compute()` | 65.26 ms | 99.07 ms | 1.52x |
| Scalar sum | 12.92 ms | 27.94 ms | 2.16x |
| Variance | 25.67 ms | 69.68 ms | 2.71x |

The smaller `(512, 512)` case with chunks `(31, 47)`, blocks `(7, 11)` showed
about 3.0x for non-flat materialization, 2.6x persisted materialization, and
1.6–1.7x for scalar sum/variance. Broadcasting, partial slices, strided outputs,
axis sums, and selections were near parity in that run; complex fallback was
faster. Tracked peaks remained bounded in these probes, but native/process peak
memory has not been fully qualified.

Route observations show legacy using fused uniform prefilters while typed runs
per-chunk evaluation for the non-flat cases. An experiment forcing NumPy rather
than native chunk math recovered only a small part of the gap, so merely adding
a small-chunk native cutoff is not a sufficient fix and has not been done.

These were repeatable large regressions under the plan's default blocker rule.
The user subsequently requested profiling/optimization, explicitly accepting
that remaining gaps should be recorded rather than indefinitely blocking the
migration. The follow-up below records that work. Correctness/platform gates
remain separate from this performance decision.

### Profile-guided fallback optimization

Ten-operation cProfile runs identified distinct costs rather than a single slow
numerical compiler:

- Materialization repeatedly entered general `NDArray.__setitem__`, constructing
  SimpleProxy wrappers, inferring storage policy, refreshing metadata, and checking
  index invalidation for every result chunk—even newly allocated private outputs.
- Persisted prefetch already returned valid-shaped edge chunks, but the consumer
  read those edge slices again.
- Nonpersisted edge reads repeated public indexing/metadata processing despite
  already having validated scheduler coordinates.
- Native lowering and schedule specialization were repeated for every chunk.
- Small floating arithmetic chunks paid scalar interpreter costs under the
  default unspecified JIT policy. Variance incurred these costs on both passes.

Implemented bounded, Python-only improvements:

1. Fresh private in-memory outputs use the existing native slice writer directly
   when result shape/dtype/contiguity exactly match the scheduled slice. User
   outputs and persisted destinations retain general assignment contracts.
2. Reuse prefetched edge chunks, including scalar-zero reduction chunks, instead
   of decoding them again. Read other scheduled edges directly into valid-shaped
   NumPy buffers.
3. Cache native plans/schedules per invocation and metadata signature, capped at
   64 entries. Cached entries retain no operand arrays, rebind current chunks on
   every execution, and clear on success/failure. Nested evaluations have separate
   invocation state.
4. For small (<65,536 items) floating arithmetic chunks under the default execution
   policy, prefer vectorized NumPy to scalar native interpretation. Explicit JIT,
   compiler, math/accuracy policies, nondefault optimization, integer inputs, and
   transcendental calls retain native admission. In particular, this does not
   relax tests to hide backend-dependent transcendental rounding changes.

Regression tests cover edge reads occurring exactly once, zero-edge reduction
extent, private versus user-output assignment, native schedule reuse/rebinding,
operand release and cache cleanup, and arithmetic-only NumPy dispatch.

Nine-round interleaved follow-up on the same `(2048, 2048)` geometry, without
competing test/build processes:

| Operation | Legacy median | Optimized typed median | Typed / legacy |
| --- | ---: | ---: | ---: |
| Non-flat `.compute()` | 51.70 ms | 70.50 ms | 1.36x |
| Persisted `.compute()` | 64.88 ms | 85.14 ms | 1.31x |
| Scalar sum | 12.78 ms | 19.82 ms | 1.55x |
| Variance | 25.21 ms | 44.54 ms | 1.77x |

Typed times improved roughly 19%, 14%, 29%, and 36% respectively relative to the
initial measurements. These layouts still lose legacy's fused block
decode/evaluate/reduce/compress pipeline. Reaching parity likely requires broader
typed native block-geometry integration, not more frontend micro-optimizations.
The remaining gap is recorded as an accepted initial-rollout limitation per the
user's instruction; it is not claimed to be solved or used to justify uniform
fallback. Peak tracked allocations remain chunk-scale; complete native/process
memory and real Pyodide performance qualification remain pending.

Post-optimization typed full suite: **12,681 passed, 98 skipped** (four workers;
two additional focused dispatch regressions were added afterward and passed).
The final legacy full suite passed **12,683 tests, 98 skipped**. The focused
fallback-policy run after optimization passed **44 tests, 26 skipped** on the
modern runtime with injected pre-2 flags.

An independent final wheel was built and installed in an isolated target,
leaving the shared editable installation untouched; both package and extension
import paths were verified before its integration run, including LazyExpr and
reduction tests (**1,767 passed, 33 skipped**, conda Python with `-S` and pytest
`-n 0`). Build warnings came from existing native sources/toolchain policies;
the profile-guided optimizations introduce no native source changes.

The performance decision is resolved: proceed with the recorded residual gaps.
The user subsequently chose option 2: switch the development default now and
qualify that revision in CI, rather than waiting for a separate pre-switch run.

### Phase 2: Development default switched

- The unset selector now selects typed. Explicit `typed` and the temporary
  `legacy` escape hatch remain accepted; invalid values still raise.
- Added regressions for the unset default, explicit selectors, and ordinary
  evaluation with the selector absent and uniform execution prohibited.
- Native, wheel, and actual Pyodide CI now exercise the unset default rather than
  masking it with an explicit typed override. Native CI retains one explicit
  legacy job; its artifact names remain distinct from the default jobs.
- Updated the migration guide and development release notes, including the
  NumPy < 2 policy and recorded non-flat-layout performance limitations.

Fresh full platform/NumPy 1.26/Pyodide results remain pending for this revision.
Switching the development default does not claim that CI has already passed.
Ordinary-expression legacy deletion and selector removal remain deferred until
the new default has completed qualification; no compatibility DSL removal is
part of this switch.

Local validation with `BLOSC2_EXPRESSION_EVALUATOR` removed from the environment:
full default suite **12,687 passed, 98 skipped** using four workers. Ruff lint and
format checks and workflow YAML parsing pass. Fresh platform CI is still pending;
these local results do not substitute for actual NumPy 1.26 or Pyodide runs.

#### First default-mode CI follow-up

The first switch-first CI run exposed platform assumptions rather than numerical
dtype failures:

- Summary-index OR queries could be pruned with one column's candidate-block
  bitmap. That is sound for a conjunction, not an OR: a block matching only the
  other branch was skipped. OR now uses the plan's combined candidate segments;
  conjunctions keep the combined per-column bitmap optimization. This also makes
  the documented segment path the single owner of OR correctness.
- NumPy 1 has no string ufuncs, so `np.add` cannot concatenate Unicode/bytes.
  The pre-2 installed-NumPy policy now applies only to numerical expressions;
  string/bytes expressions keep their existing miniexpr-backed route.
- Typed-mode test expectations were corrected where they assumed host native
  prefilters (none on WASM), native-only square rewriting/disk-cache tracing
  (unavailable under the NumPy 1 fallback), or compared a `pow` result across two
  engines rather than against an explicit NumPy reference.

These fixes preserve the fallback/retirement policy; they do not reintroduce
uniform-dtype numerical execution. Fresh CI results for the follow-up revision
remain required.
