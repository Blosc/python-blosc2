# Remove NumExpr from Python-Blosc2

Status: runtime removal implemented on `improved-miniexpr`; performance and
platform qualification remain release decision points. See
[miniexpr-implementation-results.md](miniexpr-implementation-results.md).

Current completion audit, compatibility/coverage inventory and follow-up
qualification: [remove-numexpr-status.md](remove-numexpr-status.md). The earlier
starting-point descriptions and timing snapshots below are historical, not a
claim that the current package still imports NumExpr.

### Implemented removal

- Removed the runtime import, required dependency, NumExpr program cache,
  initialization/thread coupling and version diagnostics.
- Internal chunk evaluation now selects native graph execution or approved
  NumPy evaluation before execution; numerical failures are not replayed.
- Native interpreter admission no longer requires JIT. Native destination
  descriptors and host task partitioning are integrated into Python-Blosc2.
- Complex construction can fill an existing NumPy complex destination directly;
  unsupported operations retain explicit NumPy semantics rather than NumExpr.
- Compatibility keyword spellings remain validated, but `optimization` does not
  reproduce NumExpr algebraic rewrites. Nullable `numexpr=None` attributes and
  the legacy `ne_evaluate` helper name remain non-executing compatibility surfaces.
- Added an isolated subprocess that rejects NumExpr imports and qualifies
  resident, compressed, complex, prepared and repeated evaluation. Full-suite
  absence qualification is performed with a NumExpr import blocker as well.

Functional independence is not universal performance parity: genuine no-JIT
arithmetic is substantially slower than the previous NumExpr fallback. Windows
and remote exact-revision evidence are also not yet qualified. These must remain
explicit when deciding whether to release the removal.

### Allocation-policy prerequisite implemented

This subsection records the earlier prerequisite. Native destination descriptors
now replace the resident NumPy layout fallback for admitted numerical graphs;
unsupported operations still have explicit NumPy fallbacks.

`evaluate()` now consumes `order` as a Blosc2 allocation policy instead of
forwarding it in NumExpr options. Explicit `order="C"` therefore no longer blocks
native-first admission. `A`/`K` follow the allocated destination's actual layout;
an explicit destination retains its own layout regardless of `order`.

For resident expressions with non-C-contiguous NumPy destinations, the fallback
evaluates the approved expression graph with NumPy and copies into the supplied
output. It does not select NumExpr because of layout. This supports Fortran and
strided outputs, but may allocate a full intermediate result and is not native
strided-output acceleration. Compressed/mixed inputs still need the bounded
fallback migration described below. NumExpr remains in other runtime paths.

Tests block NumExpr entry points for representative layout cases and verify that
explicit C order retains native admission. Future interpreter/JIT strided sinks
should replace this NumPy fallback without changing the allocation contract.

## 1. Goal and definition of completion

Remove NumExpr completely from Python-Blosc2's runtime implementation and required
dependencies. Use miniexpr (Menudet interpreter and optional JIT) for admitted
numerical expressions, with explicit NumPy implementations for unsupported
operations. Preserve the expression API, bounded-memory execution and fused
evaluation/compression wherever their contracts permit.

Completion means:

- No package imports, calls, compilation, validation, dtype inference, thread
  configuration or version reporting depend on NumExpr.
- Installation and the mandatory test suite work with NumExpr absent. Installing
  NumExpr alongside Blosc2 does not change Blosc2's behavior or chosen backend.
- JIT-disabled and JIT-unavailable execution are supported first-class modes,
  rather than reasons to choose an unrelated expression engine.
- Resident, compressed, mixed and explicit-output workloads retain their public
  contracts. Any intentional compatibility changes are documented and reviewed.
- Optional comparison tools may import NumExpr **as an external reference only**.
  They must not be imported by the package or required by normal tests/install.

Removing the dependency does not require native complex/string implementation,
nor does it imply universal performance superiority over NumExpr. These are
separate objectives; remaining performance gaps must be measured and visible.

## 2. Starting point and evidence

See [evaluate-numpy-parity.md](evaluate-numpy-parity.md) and
[menudet-performance-parity.md](menudet-performance-parity.md).

The current checkout already has native graph preparation, specialization,
interpreter execution, JIT execution, bounded storage readers, resident workers,
and shared compression-prefilter execution. Eligible real arithmetic, math,
selection and reductions use these paths. However, several automatic admissions
require a JIT artifact even when the native interpreter can evaluate the graph.

The recent 72-case NumPy-output audit closed sampled API gaps; it is not an
exhaustive backend/dtype qualification. The direct-output NumExpr program cache
added during that work improves fallback performance but is explicitly temporary
in the architecture proposed here.

Recent copy measurements on M4 Pro, float64, eight threads, warmed explicit
destination, median of 31 shuffled samples:

| Elements | NumPy | NumExpr | Blosc2 JIT | Blosc2 with JIT disabled |
| --- | ---: | ---: | ---: | ---: |
| 8M | 0.922 ms | 0.801 ms | 0.762 ms | 0.902 ms |
| 100M | 11.917 ms | 10.143 ms | 7.020 ms | 10.277 ms |

The JIT-disabled measurements are from a separate run and **used the NumExpr
compatibility backend**. They are not Menudet interpreter measurements. The
corresponding NumPy/NumExpr controls in that run were 0.951/0.803 ms and
11.889/10.151 ms. Establish a genuine interpreter baseline before predicting
whether it will be faster.

Known performance gaps include small-array preparation/scheduling and conservative
parallel admission for integer-input math. Native complex support has not been
established by adding the expression-level `complex(real, imag)` constructor.

## 3. Dependency and behavior inventory

Start with a repository-wide search, including docs, examples, packaging,
benchmarks, CI scripts and tests. Classify every occurrence as runtime behavior,
public compatibility, test oracle, benchmark comparison or historical prose.
Record the replacement and owning test for each runtime call site.

| Location | Current responsibility | Planned treatment |
| --- | --- | --- |
| `pyproject.toml` | Required host dependency | Remove after replacement paths qualify. |
| `src/blosc2/__init__.py` | Optional import, `blosc2.numexpr`, initialization of NumExpr threads | Remove runtime integration; decide exposed-attribute migration. |
| `src/blosc2/core.py` | Thread documentation and version diagnostics | Report Blosc2/native capabilities; remove NumExpr coupling. |
| `src/blosc2/lazyexpr.py:ne_evaluate` | Safe/full evaluation, operand discovery, NumExpr validation, datetime and NumPy fallbacks | Replace with backend-neutral dispatch preserving scope and evaluation policy. |
| `_resident_numexpr_program`, `_resident_numexpr_output` | Compiled compatibility cache and direct NumPy output | Replace with native direct-output execution or explicit NumPy fallback; delete program cache. |
| Lazy expression chunk/reduction/selection paths | Repeated `ne_evaluate` calls and fused expressions | Route through shared native preparation/execution, not a second scheduler. |
| `_numpy_eval_expr`, scalar expression construction | Metadata inference and eager scalar evaluation | Use native metadata or approved NumPy semantics without NumExpr. |
| `expression_graph.py`, `utils.py`, `ndarray.py` | Function registry, shape inference, ufunc dispatch and aliases | Audit semantics and remove backend-dependent assumptions/comments. |
| `ctable.py`, indexing and query consumers | Chunked predicates, strings, fields and layout assumptions | Verify shared expression behavior and bounded memory transitively. |
| `tests/test_numexpr_threads.py` and NumExpr-based tests | Thread integration and reference results | Replace integration assertions; separate optional reference tests. |

Do not assume that replacing the public `evaluate()` path removes the dependency:
scalar construction, metadata probes and internal chunk calls are also consumers.

## 4. Intended execution architecture

```text
public expression / LazyExpr / internal chunk evaluation
    -> existing evaluation-policy and operand validation
    -> normalized expression + dtype/shape/output contract
    -> immutable native preparation when supported
       -> JIT implementation when enabled and available
       -> Menudet interpreter otherwise
    -> explicit NumPy fallback when native preparation declares unsupported
    -> shared scheduling and destination handling
```

### Admission and failure rules

- Separate semantic capability, backend availability and parallel-safety proofs.
  `has_jit == False` alone must not mean “unsupported expression.”
- Keep strict-native requests strict. An interpreter is native execution; NumPy
  is not. An explicit require-JIT request must fail if JIT is unavailable.
- Fallback is a preparation decision. A kernel execution error must propagate;
  never replay a partially written operation using another backend.
- Keep checked scalar conversions, integer failure domains, context-dependent
  recipes and reduction ordering intact. Do not loosen admission merely to
  increase benchmark coverage.
- Preserve safe-mode graph validation. Removing `numexpr.validate()` must not
  introduce Python `eval` into safe mode. Preserve the existing full-mode policy
  separately; changing the trust model is not a prerequisite for this removal.

### Shared execution and storage

- Extend existing graph/artifact APIs as needed so interpreter and JIT can bind
  an external NumPy destination, including a serial direct-output path.
- Keep allocation policies such as `order` separate from execution-engine
  options. Preserve the implemented native-eligible C / NumPy strided fallback
  split until native strided sinks qualify; never restore layout-driven NumExpr
  dispatch. Account for the NumPy fallback's intermediates in performance reports.
- Verify whether borrowed interpreter artifacts support the same binding and
  lifetime contracts; do not assume the current JIT-only resident helper does.
- Reuse existing resident worker and prefilter machinery. Compressed output must
  retain fused evaluate/filter/compress execution, not materialize a full NumPy
  result before compression.
- Use capability-based thread admission, including artifact reentrancy and
  invocation-local scratch. Join all workers before exposing errors or releasing
  inputs/output, including submission failures.
- Support broadcasting/strides through existing native schedules/readers where
  possible. Otherwise use a documented bounded fallback or explicit temporary
  dictated by the operation's semantics. Report full-result copies when needed.
- Keep caches bounded and operand-free: expression, signature, semantic/backend
  configuration and immutable preparation only. Rebind arrays, output, thread
  policy and runtime state on each call. Test concurrent reuse and eviction.

## 5. API and semantic decisions

Create a compatibility table before implementation, distinguishing documented
Blosc2 behavior from incidental NumExpr behavior. NumPy-style promotion remains
the starting policy; exact NumExpr quirks are not the universal oracle.

### Expression coverage

Inventory arithmetic, comparison/Boolean/bitwise operations, shifts, integer
division/modulo/power, weak scalar captures, transcendental functions, `where`,
`copy`, `ones_like`, predicates, complex functions and string operations.
Include nested forms, namespaced calls, constant-only and scalar expressions.

For reductions include axis/tuple axes, keepdims, empty inputs, initial/where,
accumulator dtype and staged/nested forms. Classify each as native interpreter,
native JIT, explicit NumPy fallback or unsupported by the public contract.

### Output contract

- Preserve output identity, exact shape requirements, 0-D outputs, read-only
  rejection, order and supported casting policies.
- Preflight aliasing/overlap, reversed/strided/Fortran arrays, alignment,
  byte order, empty domains and integer overflow-sensitive cases.
- Prefer direct writes. Any necessary temporary must have a stated reason,
  bounded size where possible, and observable allocation accounting.
- `copy(x)` may lower to an identity graph node but must still copy into the
  supplied destination or return independent storage when no output is supplied.

### Keyword and exposed-API migration

Audit `evaluate(**kwargs)`, `lazyexpr(ne_args=...)`, output/storage options and
all documented forwarding to NumExpr. Decide each option explicitly:

| Option/surface | Proposed policy to review |
| --- | --- |
| `local_dict`, `global_dict`, internal frame depth | Preserve name resolution and caller scope; do not cache frames or mappings. |
| `out`, `casting`, `order` | Implement as Blosc2 contracts independent of backend; validate before writes. |
| `optimization`, `truediv` | Inventory accepted values and observable effects; implement supported semantics or document a staged deprecation. Never silently ignore arithmetic-changing options. |
| `sanitize` | Define compatibility with existing safe/full policy; it cannot disable mandatory safe-mode validation. |
| `disable_cache` | Define which preparation caches it bypasses consistently; retain if promised. |
| `ne_args` | Preserve as a compatibility spelling initially if needed, normalize internally, and propose a backend-neutral replacement only with an API migration decision. |
| `blosc2.numexpr` | Audit external-facing documentation/tests; remove or temporarily expose a deprecated `None` placeholder, never import NumExpr through it. |
| `NUMEXPR_*` environment variables | No effect on Blosc2 after removal; explain native thread/JIT controls instead. |

Document release policy for removed options; avoid silently claiming a fully
drop-in NumExpr implementation when promotion or option semantics differ.

### NumPy fallback boundaries

- Complex construction/arithmetic: use approved NumPy operations where native
  support is absent. Preserve infinities by assigning real/imaginary components
  separately. Native complex kernels are a later optimization, not a removal gate.
- Strings: cover fixed-width bytes, Unicode and supported string dtypes explicitly;
  test `contains`, comparisons and query selection without NumExpr.
- Datetime/timedelta: preserve unit promotion, `NaT` and comparison/arithmetic
  distinctions. Raw-int64 comparison shortcuts require the existing semantic proof.
- Unsupported linalg and shape-changing operations: retain explicit NumPy/Blosc2
  implementations and their existing materialization contracts.
- NumPy expression evaluation may create intermediates. For compressed data,
  preserve tile-local evaluation where valid; do not advertise fused execution
  or bounded memory for inherently global fallbacks without evidence.

## 6. Implementation phases and exit gates

### Phase A — Freeze contracts and capture baselines

1. Complete the dependency inventory and API option table.
2. Audit existing miniexpr tests before proposing additional native tests.
3. Capture representative output/dtype/error behavior and current route reports.
4. Run reproducible JIT, JIT-disabled and actual interpreter measurements;
   distinguish API cost, preparation, execution, output allocation and copies.

Exit: every runtime NumExpr use has an owner/replacement, and unresolved semantic
choices are recorded rather than hidden in implementation defaults.

### Phase B — Interpreter/direct-output foundation

1. Audit native interpreter binding, external output, status reporting, artifact
   ownership, reentrancy and scratch requirements.
2. Add missing primitives in miniexpr or C-Blosc2 at the owning layer; expose only
   necessary binding changes in `blosc2_ext.pyx`.
3. Qualify serial and parallel interpreter execution against the existing native
   semantics, including direct NumPy sinks and prefilter invocation.
4. Refactor JIT-specific admission checks into semantic/backend capability checks.

Exit: supported real maps execute without JIT or NumExpr, with direct output and
verified native route reports. Compressed sinks remain compression-driven.

### Phase C — Replace internal evaluation and inference

1. Introduce a backend-neutral replacement for `ne_evaluate` using existing
   expression/native modules; avoid duplicating public and internal compilers.
2. Migrate chunk maps, selection, reductions, scalar construction and synthetic
   metadata probes. Preserve caller scope and evaluation mode across wrappers.
3. Prefer native metadata inference over executing data; keep reviewed tiny NumPy
   probes only for unsupported inference with documented warning/error behavior.
4. Replace the resident NumExpr program cache with shared native preparation or
   the explicit NumPy fallback, then remove its locking/program machinery.

Exit: runtime import blocking shows zero NumExpr accesses for the covered numeric
and fallback families, including LazyExpr and table/query consumers.

### Phase D — Fallback coverage and API migration

1. Complete complex/string/datetime/layout fallback contracts from section 5.
2. Implement reviewed keyword policies and any compatibility aliases.
3. Make dtype inference, casting and actual execution agree across all routes.
4. Convert essential tests from NumExpr-dependent expectations to NumPy/manual
   contract references. Keep optional differential tests separate.

Exit: full public behavior coverage runs with NumExpr absent; intentional changes
are enumerated in migration/release documentation.

### Phase E — Remove dependency and platform integration

1. Remove runtime imports, package dependency, initialization/thread handling,
   diagnostics, helper names and obsolete comments.
2. Update docs, tutorials, examples and benchmark tooling; external comparison
   tools should clearly report an unavailable optional reference engine.
3. Remove obsolete thread-coupling tests and add absence/independence tests.
4. Verify wheel metadata and install/test in clean environments without NumExpr.

Exit: runtime source and package metadata contain no executable NumExpr
dependency; platform qualification passes with and without optional NumExpr
installed, producing the same Blosc2 dispatch decisions.

### Phase F — Performance qualification and release decision

1. Compare final performance against Phase A on matched hardware/configurations.
2. Profile remaining gaps rather than routing back to NumExpr. Prioritize small
   preparation overhead, integer-input math admission and interpreter copy/maps.
3. Separate scheduling changes from kernel changes; validate error/lifetime
   semantics for each broader parallel admission.
4. Publish native interpreter/JIT/NumPy-fallback coverage and residual regressions.

Exit: review functional completeness and performance evidence independently.
Universal NumExpr parity is not claimed merely because dependency removal passes.

Dependency order: A -> B -> C -> D -> E -> F. Inventory, fallback research and
benchmark design can inform B early; performance measurements run throughout.

## 7. Testing and platform qualification

### Tests at the owning layer

- miniexpr: exhaustive numerical semantics, dtype/operator matrices, interpreter
  versus JIT, FP status, overflow and native complex support if added later.
- C-Blosc2: worker/prefilter lifecycle, storage and compression integration.
- Python-Blosc2: binding contracts, dispatch, name resolution, output identity,
  casting/shape validation, metadata, bounded scheduling, lifetime and a small
  representative end-to-end semantic set. Avoid duplicating native matrices.

Test cache rebinding and garbage collection, simultaneous callers, overlapping
buffers, failed submissions, failure propagation without replay, and unsupported
preparation falling back before writes. Test safe/full evaluation explicitly.

### Dependency absence and isolation

- Run mandatory suites in an environment with NumExpr genuinely uninstalled.
- Add an isolated subprocess test with an import blocker that fails on attempts
  to import NumExpr, catching optional-import regressions as well as hard ones.
- Repeat representative operations with NumExpr installed and instrumented to
  fail on use; installed optional dependencies must not affect dispatch.
- Ensure test fixtures do not silently skip core contracts through
  `pytest.importorskip("numexpr")`; reserve that for external comparisons.
- Check wheel requirements/imports, documentation builds and supported examples.

### Platform/backend matrix

Qualify Linux x86-64 and available ARM64, macOS ARM64, Windows, and actual wasm32.
On hosts exercise interpreter-only, available JIT backends and unavailable-JIT
fallback. WASM uses its supported single-thread mode; do not demand host worker
speedups. Cover supported NumPy/Python versions via the existing CI matrix.

Windows complex fallback must not introduce C99 `_Complex` assumptions. Actual
WASM tests must cover complex/strings and dtype widths; setting `IS_WASM=True`
on a host is only a dispatch probe, not platform qualification.

For native changes, retain exact-revision qualification evidence and aligned
Python/native pins. Build isolated test packages where necessary, verify imported
extension paths, and do not overwrite another checkout's editable installation.
Use the `blosc2` conda environment for Python/build/test commands.

## 8. Performance methodology and acceptance

Extend the existing resident parity benchmark with explicit backend selection
and verified route reports; do not label “JIT off” as “interpreter” without proof.
Provide NumPy and optional NumExpr reference columns and retain `N/A` for failed
correctness qualification, without a verbose failure trailer in the main table.

Benchmark dimensions:

- Sizes: scalar/empty, small arrays (including 1k and 100k), 8M and 100M where
  memory permits; separate cold preparation from warmed API calls.
- Inputs: float32, float64, representative integer/Boolean, plus fallback complex,
  string and datetime cases; include non-contiguous/broadcast layouts.
- Recipes: copy/identity, arithmetic chains, division/power, transcendental math,
  predicates/where and reductions. Keep overflow/accumulation differences explicit.
- Storage: resident, compressed and mixed inputs; NumPy and compressed outputs,
  including explicit destinations. Preserve integer/arange/compression controls.
- Threads: serial and representative physical-core/SMT counts; avoid claiming
  portability from a single M4 Pro result.

Use randomized engine order, warmups and repeated samples. Publish medians and
variability. Keep fresh-output and reused-output experiments separate; report
logical read/write bandwidth as such, not measured DRAM traffic. Record native
revision, backend, compiler, versions, shapes/dtypes, geometry and thread counts.

Track full-result temporary bytes, bounded scratch, copies, compressed bytes,
preparation/cache cost and allocation/first-touch effects when relevant.
Native copy must not hide a full-result allocation/copy behind a fast kernel time.

Suggested review gates (to ratify after baseline collection):

- No unexplained repeatable regression above 10% on established large-array JIT
  paths; use variability to distinguish noise from regressions.
- Interpreter-only paths must be compared against the prior no-JIT behavior,
  which may have used NumExpr. Any material regression needs an explicit decision
  and an actionable profile; do not silently substitute NumPy to make a claim.
- Small-array cases report absolute overhead as well as ratios. Set budgets from
  actual baselines rather than arbitrary thresholds.
- No new unbounded/full-result materialization on formerly bounded compressed
  operations without an explicit documented semantic requirement.

## 9. Principal risks and decision points

| Risk | Required evidence or mitigation |
| --- | --- |
| Interpreter is correct but slower than the current fallback | Measure direct-output serial/parallel routes before removing NumExpr; profile native loops and scheduling independently. |
| NumExpr promotion was implicitly part of user-visible behavior | Inventory dtype/casting differences and decide them before routing changes. |
| Parallel admission changes integer/error semantics | Keep native proofs and error-domain tests; do not infer safety from successful ordinary values. |
| NumPy fallback creates large intermediates | Preserve tile evaluation and instrument memory; identify inherently global operations explicitly. |
| Native artifact/scratch cannot be shared concurrently | Qualify ownership and reentrancy; use invocation-local state or bounded worker contexts. |
| Removing validation changes safe-mode behavior | Keep graph/operand validation authoritative and test malformed/unapproved expressions. |
| Metadata probes disagree with execution | Prefer shared native inference; test fallback shape/dtype contracts and scalar promotion. |
| Platform-specific gaps surface late | Run actual Windows/WASM interpreter-only jobs during B–D, not just at dependency removal. |
| Thread controls change workload behavior | Document Blosc2 thread ownership and eliminate nested oversubscription. |

Open decisions for review: public keyword deprecations, `blosc2.numexpr` migration,
required platform/backend coverage, performance regression budgets, and whether
native complex/string acceleration is scheduled after dependency removal.

## 10. Final deliverables checklist

- [x] Complete runtime-call inventory and compatibility/coverage table.
- [x] Native interpreter direct-output and shared scheduling qualification (local).
- [x] Backend-neutral internal evaluator and metadata/scalar migration.
- [x] Explicit NumPy fallback contracts and representative tests.
- [x] Removal of executable NumExpr integration, imports, thread controls and dependency; documented compatibility names remain.
- [ ] Clean-environment and import-blocking CI; actual Windows/WASM results.
- [x] Updated API docs, diagnostics, examples and migration notes.
- [x] Reproducible interpreter/JIT/fallback performance and bounded-memory report (local; release budgets still open).
- [x] Documented remaining gaps, intentional behavior changes and release decision point.
