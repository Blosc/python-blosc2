# Safer LazyExpr evaluation: two-phase implementation plan

Date: 2026-10-08. Branch: `safer-lazyexpr`.
Status: experimental implementation in progress; exit gates are not complete.
See `plans/safer-lazyexpr-results.md` for implemented scope, measurements and gaps.
Current direction: narrow phase 1 first; ordinary in-memory defaults stay compatible.
Phase 2 and Menudet lowering remain opt-in experiments, not production defaults.

## 1. Goal and recommendation

Replace Python evaluation of expression text with a validated computation graph,
while preserving ordinary LazyExpr numerical behavior. First enforce the boundary
for safe deserialization, then consider making it the default for all LazyExpr
construction and execution after compatibility and performance measurements.

Use Menudet as an optional backend for semantically compatible graph segments,
not as the definition of LazyExpr semantics. A trusted NumExpr/NumPy/Blosc2
implementation is also a valid backend when invoked through explicit operations,
not Python evaluation of user text. Security and backend selection are separate.

The separate branch is appropriate: do not make this a prerequisite for the
Menudet draft release, and do not assume that interpreter-only Menudet will
outperform existing expression engines. Advance each phase only after its gates
pass; retain the option to ship phase 1 without phase 2 or Menudet acceleration.

## 2. Agreed policy and non-goals

- Keep `deserialize="safe"` as the default. No new `safe_expr` switch on `open()`.
- Safe loading accepts supported numerical calls, reductions, methods and
  indexing; it is **not** an arithmetic-only grammar.
- `deserialize="full"` is the explicit trusted-data escape hatch for legacy
  expression reconstruction using the existing evaluator, where supported.
  It does not promise recovery of arbitrary saved Python callbacks.
- Legacy DSL/Python UDF and table-kernel opt-in behavior remains unchanged.
  Full-only nested payloads cannot become safe merely because their parent is a
  validated expression.
- No silent semantic conversion to Menudet's checked arithmetic, operand-typed
  promotion, logical block reductions, or independent table-row grouping.
- No new whole-LazyExpr portable format, migration tooling, native language
  expansion, JIT expansion or universal libm accuracy promise in this work.
- Phase 1 does not restrict unrelated in-memory expressions. Phase 2 is a
  separate compatibility decision, not an incidental side effect of phase 1.
- Phase 1 covers current structured LazyExpr recipes and persisted CTable
  expression columns. Historical formats already requiring `"full"` retain that
  gate; proxy/reference policy is unchanged. No historical-format migration.

## 3. Threat model and precise safety claim

Treat saved expression strings and their metadata as untrusted. Safe evaluation
must not resolve arbitrary Python names, call arbitrary functions or methods,
traverse arbitrary attributes, import modules, or execute Python source supplied
by the file. Allowed operations dispatch through explicit trusted implementations.

An expression such as `sqrt(x)` is not inherently an exploit. The existing
regex validator alone does not establish this stronger boundary; no specific
exploit is asserted by this plan. The current broad Python fallback is the
execution surface being removed from safe-expression paths.

The claim is limited to expression-text-driven Python execution. It is not a
process sandbox and does not eliminate native-library vulnerabilities, expensive
computations, large allocations, permitted filesystem/network reference access,
or behavior of custom operand objects deliberately supplied by trusted callers.
Reference resolution retains its existing policy; any stricter URL/path policy
must be specified separately. Resource bounds below limit parsing/graph abuse,
not all runtime resource consumption.

## 4. Initial code inventory

Inventory and classify all reachable routes before implementation:

| Location | Current concern / integration point |
| --- | --- |
| `src/blosc2/b2objects.py` | `decode_structured_lazyexpr()` validates text, resolves operands, then calls the ordinary constructor; nested recipes recurse here. |
| `src/blosc2/lazyexpr.py` | `ne_evaluate()` merges caller locals and can fall back to `eval()` with broad Blosc2 globals; WASM uses a NumPy evaluation route. |
| `src/blosc2/lazyexpr.py` | Constructor handling, shape/dtype inference, slice rewriting, `compute()` and indexing contain additional text-evaluation routes. |
| `src/blosc2/lazyexpr.py` | `fast_eval`, `slices_eval`, `chunked_eval`, `reduce_slices` and reduction dtype helpers own scheduling and existing array semantics. |
| `src/blosc2/utils.py` | `infer_shape()` and NumPy namespace population must be checked for indirect evaluation and dynamic name resolution. |
| `src/blosc2/deserialization.py` | Immutable effective loading policy; currently missing-policy lookups can default to full, which must not be used to infer safe graph provenance. |
| `src/blosc2/msgpack_utils.py`, `core.py`, `schunk.py` | Frame/object dispatch, safe decoding and historical LazyArray gates. |
| `src/blosc2/ref.py`, stores and object containers | Recursive policy propagation and operand resolution. |
| `src/blosc2/ctable.py` | Saved expression columns are in phase-1 scope; filtering and deferred expression reconstruction must preserve their safe execution boundary. |
| `src/blosc2/portable_kernel.py`, `portable_lazy.py` | Menudet signature validation, native import and logical scheduling, reusable only under proven semantic eligibility. |

Also trace UTF8 expressions, structured fields, `ColExpr`, CTable expressions,
RemoteArray/Proxy operands, NumPy ufunc construction and chained saved expressions.
Record each route as safe-graph, full-only legacy, trusted programmatic execution,
or out of scope. Do not claim whole-LazyExpr safety while leaving an undocumented
string-evaluation route reachable from a safe-loaded expression.

## 5. Shared architecture

### 5.1 AST parser and explicit operation registry

Create a small package module, provisionally `src/blosc2/expression_graph.py`,
following existing naming/style conventions. Parse expression text with `ast`
and lower supported syntax to immutable graph nodes. Do not use `eval()`,
`exec()`, compiled Python expression code, or dynamic namespace population to
construct, infer or execute the safe graph.

Each registered operation declares canonical name/aliases, argument and keyword
schema, type/shape rules, implementation dispatcher and backend eligibility.
Populate the registry from reviewed explicit entries, not all public callables
found in NumPy or Blosc2. Initially keep it internal; do not introduce an
arbitrary user-callback registration mechanism under the safe policy.

Node families: operand, literal, unary/binary operator, comparison, approved
elementwise call, reduction, index/slice, supported constructor and shape operation.
Add selection, cumulative and linear-algebra nodes according to the compatibility
inventory, rather than treating them as unrestricted calls.

Rules:

- Operand names bind only to the explicitly resolved operand mapping.
- Calls resolve only to registry operations, never operand values or caller locals.
- Normalize supported spellings such as `sum(x)` and `x.sum()` to the same node.
  Supported `np.sqrt(x)` / `blosc2.sqrt(x)` spellings, if retained, are syntactic
  aliases for a reviewed operation, not namespace-object attribute lookup.
- `x.mean()` is a registered reduction, not generic `getattr(x, "mean")`.
- Validate positional/keyword arity and keyword types. Reject `*args`, `**kwargs`
  and keywords that introduce arbitrary callbacks or file access.
- Reject arbitrary attribute chains, private/dunder names, lambdas,
  comprehensions, assignment expressions and imports. Classify any presently
  supported conditional-expression syntax explicitly before deciding its status.
- Indexing must specify supported integers, slices, ellipsis, new axes, field
  names and explicit index operands. Build slices directly; never evaluate `np.s_`
  text. Reject executable expressions disguised as indices.
- Scalar literals, typed scalar formatting, strings, `nan` and `inf` need explicit
  rules; dtype specifications use validated descriptors, not imported classes.
- Preserve source spans for actionable diagnostics.
- Bound text size, AST node count/depth and literal sizes before costly lowering;
  bound generated graph size and catch parser recursion failures predictably.

### 5.2 Graph semantics and execution

Keep a typed/shape-aware graph separate from backend compilation. Derive shape
and dtype from operation rules and operand metadata, without evaluating expression
text or computing the full input merely to guess metadata. Operations requiring
data-dependent output shapes must be explicitly represented or rejected with a
compatibility explanation; do not quietly infer them by Python evaluation.

Reuse existing array scheduling and reduction implementations where possible,
through vetted dispatchers. NumExpr may remain a fast backend using text generated
from validated nodes and an explicit operand environment. User text is not passed
to the current broad `ne_evaluate()` fallback. Unsupported NumExpr operations
use a registered direct dispatcher or fail, never a Python-text fallback.

Preserve broadcast shape, dtype promotion, scalar/array return conventions,
`out`, `where`, slicing, errors and warning behavior. Backend selection precedes
execution: arithmetic failures do not trigger a retry under different semantics.

### 5.2.1 Operand admission and trusted adapters

Define an explicit table of approved operand types and adapters before enabling
safe loading. An approved mathematical operation can still invoke arbitrary
Python through object-dtype arithmetic, ndarray subclasses or protocols such as
`__array__`, `__array_ufunc__` and `__array_function__`.

Admission must precede coercion or metadata inspection that could invoke such
hooks. Do not accept an object merely because it has `shape` and `dtype`, and do
not treat `np.asarray()` as a safe admission check. Establish trusted concrete
types/adapters first, then obtain metadata or buffers through their reviewed
interfaces. Classify subclasses explicitly rather than inheriting trust from a
base class automatically. Approved container adapters must preserve nested
deserialization policy during reads as well as during initial resolution.

The inventory must cover numeric and non-object NumPy arrays/scalars, Blosc2
containers, structured fields and reference-backed operands actually supported
under the existing safe policy. Object arrays and custom protocol-bearing objects
need explicit classification; unsupported objects reject before invoking hooks.
This does not authorize broadening proxy/reference permissions. Phase 2 must
document separately how explicitly supplied trusted programmatic objects interact
with the safe expression boundary.

### 5.2.2 Permissions, requirements and evaluation lifetime

Separate caller permission from a node's execution requirements:

- `deserialize="full"` permits legacy reconstruction; it does not make an
  otherwise validated `sqrt(x)` graph inherently full-only. Prefer validated
  reconstruction when compatible, even when full permission was supplied.
- Mark actual legacy-execution requirements on nodes/dependencies. Validated
  graphs can compose regardless of their loading permission, provided their
  entire dependency closure meets safe execution requirements.
- A graph with a legacy-only dependency cannot execute under safe permission.
  Require an explicit trusted route; never infer elevated permission just because
  one operand was opened with `"full"` or because a carrier is absent.
- Syntax alone is insufficient: an ordinary expression with a full-only proxy or
  legacy-kernel operand still has that dependency's requirements. Safe eligibility
  is derived by validation, not by trusting persisted flags or silently changing
  an object's immutable deserialization policy.

Use an evaluation context for intermediate reuse. For `x - mean(x)`, compute the
mean once per evaluation and reuse it across output chunks. Repeated reductions
may share intermediates within that evaluation when semantically equivalent.
Cache immutable parsing/compilation plans across evaluations, not computed values
that could become stale when operand data changes. Validate changed expression
text, bindings and relevant metadata before reusing a plan.

Keep execution bounded-memory: reuse existing fusion/streaming, release temporary
buffers when their consumers finish, and avoid full-array materialization merely
to execute a graph node. Detect cycles in nested recipes and dependency graphs
before evaluation, while allowing legitimate shared dependencies. Bound recursive
recipe expansion separately from the size of each expression's AST.

Preserve existing behavior under concurrent operand mutation; this work does not
add transactional snapshots or promise consistency across multiple reads. State
this explicitly when describing per-evaluation reuse and multi-pass reductions.

### 5.3 Reductions are graph/scheduler operations

Menudet already supports reductions over a declared logical group. That does not
make a per-chunk kernel equivalent to a LazyExpr global or axis reduction.

| Expression | Intended graph plan |
| --- | --- |
| `sqrt(x) + y` | Eligible elementwise segment, then normal output scheduling. |
| `sum(sqrt(x))` | Elementwise segment followed by the existing whole-array reduction contract. |
| `x.mean(axis=0)` | Explicit axis reduction with its shape and promotion rules. |
| `x[0]` | Validated indexing node, with existing dimension-dropping rules. |
| `x - mean(x)` | Reduction dependency computed first, then broadcast into the elementwise stage. |

Specify axes (including tuples/negative axes), `keepdims`, accumulation dtype,
empty inputs, identities, NaNs, masks and `ddof` where currently supported.
Inventory sum/prod/min/max/any/all/mean/variance/std separately; preserve the current
floating accumulation behavior rather than replacing it with an accidental block
combine order. A partial read of a result cannot redefine a global reduction's
input domain. Only introduce fused partial reductions after differential proof
and a documented combination algorithm. No whole-array-to-one-block workaround
that destroys bounded-memory evaluation.

### 5.4 Menudet lowering is optional and semantics-gated

Start with a narrow audited elementwise subset. Compare LazyExpr and Menudet
semantics for each operator/type combination, especially integer overflow, weak
literals, signed/unsigned mixtures, float promotion, casts, NaNs, short-circuiting
and warnings. Do not assume matching output dtype implies matching intermediates.

Use explicit casts or typed temporaries only when they provably preserve the
LazyExpr contract; otherwise choose another approved backend. Complex, datetime,
object and other unsupported Menudet computations must not become blanket
LazyExpr exclusions. Object operations require a separate safety classification:
arbitrary Python operator dispatch is not safe just because it is called numeric.

Draft Menudet is interpreter-only. Keep NumExpr/miniexpr expression fast paths
when they satisfy the graph boundary and outperform kernel lowering. Cache graph
parsing and backend plans using immutable graph content plus relevant dtype/shape,
backend/version and policy keys; never reuse a full-only plan as a safe plan.
Do not persist caches or use pickle as a new trust boundary.

## 6. Phase 1 — safe deserialization end to end

### P1.0 Establish the compatibility and reachability baseline

- [ ] Inventory all evaluator routes and supported expression forms in section 4.
- [ ] Capture a versioned corpus from current tests/docs/examples, including
  ordinary default-loading function/method/indexing round trips.
- [ ] Cover current structured LazyExpr recipes and persisted CTable expression
  columns. Document historical formats that remain behind their existing full-only
  gate, including applicable old LazyArray metadata. Never relax UDF/proxy gates.
- [ ] Define the approved operand/adaptor matrix and reject unsupported objects
  before coercion, protocol dispatch or untrusted metadata access.
- [ ] Record baseline outputs, metadata, exceptions and representative timings.

### P1.1 Build a thin end-to-end prototype first

- [ ] Implement only the graph/registry operations needed for `sqrt(x) + y`,
  `sum(x)`, `x.mean(axis=0)`, `x[0]` and `x - mean(x)` initially.
- [ ] Exercise save → safe reopen → inference → evaluation, plus composition and
  partial reads. Keep the prototype isolated from production default routing
  until the compatibility gate is met; these five expressions are not the final
  supported subset.
- [ ] Prove existing schedulers can execute these nodes without returning to a
  Python-text evaluator. Demonstrate global reduction scope and once-per-evaluation
  intermediate reuse without cross-evaluation result caching.
- [ ] Benchmark the prototype against the existing engine immediately: opening,
  planning, first/repeated execution, fusion, peak memory and partial reads.
  Record results by elementwise/reduction/indexing family, not just an aggregate.
- [ ] Review integration feasibility before expanding the registry. If the design
  forces full intermediate materialization or defeats existing scheduling, revise
  the graph-to-scheduler interface first. Menudet lowering is not needed here.

### P1.2 Expand the parser, registry and trusted graph executor

- [ ] Implement shared architecture without changing ordinary in-memory defaults.
- [ ] Provide direct approved execution for common math, reductions and indexing
  before enforcing the loader boundary; do not reintroduce the arithmetic-only
  persistence regression.
- [ ] Add exhaustive positive/negative parser and signature tests.
- [ ] Add a capability matrix: safely supported, full-only legacy or unsupported.

### P1.3 Integrate safe loaders and preserve graph provenance

- [ ] Parse/validate recipe syntax and call signatures **before** resolving its
  operands, so an invalid parent cannot trigger needless reference access.
- [ ] After permitted reference resolution, validate operand types, shape and
  graph contracts; construct the graph without the ordinary text evaluator.
- [ ] Propagate `deserialize` unchanged through disk, frames, MessagePack,
  nested containers, stores, references and CTable expression recipes.
- [ ] Safe-loaded expressions retain a safe execution graph through `compute()`,
  indexing, reductions, ufunc composition, `where`, copies and nested expressions.
  No deferred fallback to Python text evaluation after opening.
- [ ] Handle mutable public expression text/operand mappings: invalidate and
  revalidate against the safe policy, or reject the mutation explicitly. Never
  use a stale graph after mutation or downgrade because a carrier is absent.
- [ ] Track actual execution requirements separately from loading permission.
  Validated graphs opened with `"full"` can compose with safe graphs; dependencies
  requiring legacy execution must reject or use an explicitly trusted route.
- [ ] Preserve existing recipe format where feasible: compile its text on load.
  Do not trust a persisted "safe" flag or prevalidated graph claim. Version any
  unavoidable format change explicitly; revalidate every safe load.
- [ ] Full loading prefers compatible validated reconstruction and permits the
  existing supported legacy route where necessary. Mark actual legacy requirements
  after composition; existing structural validation still applies. Numerical
  execution failures never trigger legacy retry.
- [ ] Implement per-evaluation intermediate reuse, bounded-memory scheduling and
  dependency-cycle detection; cache plans, not operand-dependent results.
- [ ] Errors identify the disallowed operation/location and explain the trusted
  `deserialize="full"` alternative; do not suggest that full recovers any callback.

### P1.4 Security and compatibility verification

- [ ] Test approved numerical expressions through every supported persistence
  entry point and ordinary default reopening.
- [ ] Test global/axis reductions, fixed strings, datetime/complex approved routes,
  constructors, broadcasting, indexing, fields and partial reads from the corpus.
- [ ] Test forbidden callables, attribute traversal, namespace aliases, caller-local
  shadowing, malformed keyword arguments and hostile index/constructor arguments.
- [ ] Use harmless sentinel operand objects/subclasses to prove unsupported types
  reject without invoking coercion, ufunc, array-function or metadata hooks.
- [ ] Test safe/full-opened validated graph composition separately from graphs
  with actual full-only dependencies, including proxies and legacy kernels.
- [ ] Test shared dependencies versus cycles, recursive recipe expansion bounds,
  reduction reuse across chunks and reevaluation after operand data changes.
- [ ] Assert forbidden recipes fail before operand resolution and before callbacks
  or side effects. Use harmless sentinels, never destructive exploit payloads.
- [ ] Instrument each Python-text evaluation bridge reachable from safe graphs to
  fail if invoked. Exercise construction/inference, first and repeated execution,
  mutation/composition, fallback, WASM and nested recipes, not only loader calls.
- [ ] Retain tests proving legacy kernels, proxies and other full-only nested
  payloads still require full. Verify the explicit full compatibility route.
- [ ] Add bounded parser fuzz/property tests and differential graph tests.
- [ ] Document the limited safety claim and known resource/reference boundaries.

### Phase-1 exit gate

All currently supported ordinary numerical persistence cases in the agreed corpus
must work with safe graph reconstruction. Any deliberate exclusion requires an
explicit capability entry and user review, not changing tests to `"full"` merely
to regain green CI. No Python-text evaluator may be reachable from a safe-loaded
graph's lifetime. Approved operand adapters and execution requirements must enforce
the same boundary. Platform suites and phase-1 performance checks must pass:
preserve fusion/streaming and review time and memory regressions by workload family,
using the provisional budgets below. Menudet lowering is optional at this gate.

## 7. Phase 2 — all LazyExpr construction and execution

Begin only after reviewing phase-1 results. This phase changes the default boundary
for string-based construction and replaces internal text-based reconstruction in
programmatically built expressions as well.

### P2.0 Decide the public compatibility escape hatch

- [ ] Propose and obtain approval for a construction/evaluation policy on
  `blosc2.lazyexpr()` and relevant constructors/context APIs. Naming is not yet
  decided; prefer one explicit safe/full policy, not conflicting Boolean switches.
- [ ] Distinguish this execution policy from `deserialize`, which applies to
  loading. No new parameter on `open()` is necessary.
- [ ] Preserve ordinary Python functions that **build** graphs, e.g.
  `expr = helper(x, y)`. Their trusted Python execution is outside expression-text
  parsing; `"helper(x, y)"` is not an approved safe call unless it names a builtin
  reviewed operation. LazyUDF remains a separate explicit Python execution API.
- [ ] Preserve convenient operand discovery from caller locals where compatible,
  but discover only validated operand names. It must not expose callable names,
  modules or globals as an execution namespace.

### P2.1 Make graphs the canonical in-memory representation

- [ ] Build graph nodes directly for overloaded operators, ufuncs, methods,
  `ColExpr` and table expressions rather than stringify-and-evaluate cycles.
- [ ] Treat expression text as display/serialization output, with explicit
  round-trip rules; preserve documented `.expression`/`.operands` behavior or
  publish a reviewed compatibility change.
- [ ] Apply the same registry to string constructors and later compute paths.
- [ ] Cover shape/dtype inference, constructors, UTF8 dispatch and operations
  currently delegated to NumPy. Unsupported operations fail with precise errors.
- [ ] Keep trusted legacy execution only through the approved explicit policy.
  Never choose it because a fast backend lacks a feature.
- [ ] Expand Menudet eligibility only with differential semantic evidence and
  measured benefit; do not silently replace numerical contracts.

### P2.2 Benchmark before changing defaults

Start a reproducible driver under `bench/ndarray/` during the phase-1 prototype,
provisionally `safer_lazyexpr.py`, writing machine-readable results. Expand it
here to the complete workload matrix. Compare the pre-change baseline, safe graph
with established backends, and eligible Menudet segments. The first performance
goal is preserving established fusion and scheduling, not replacing the backend.

Measure separately:

- Parse/construct time, shape/dtype inference, cold compilation, cached planning.
- First execution, repeated execution and amortized end-to-end cost.
- Persistence/open cost and first evaluation after opening.
- Peak resident memory, intermediate allocation size and bytes read/written.
- Backend choice and reason, so speedups cannot hide changed coverage/semantics.

Workload matrix:

- Tiny arrays (dispatch overhead), cache-resident arrays, and compressed streaming
  arrays exceeding cache; in-memory and disk-backed variants.
- Simple arithmetic; `sqrt`/trig/exp composites; comparisons and selection;
  multiple operands and scalar broadcasting; rank/broadcast variations.
- Global and axis reductions, `x - mean(x)`, empty and scalar outputs, cumulative
  operations where supported, slices/strides and indexing-heavy workloads.
- Float32/64, supported integer widths/mixes, fixed strings, complex and datetime
  trusted-backend routes. Report unsupported combinations rather than omit them.
- Aligned/misaligned chunk and block shapes, compressibility levels, thread counts,
  and repeated graph reuse with changed operands.
- Representative CTable expression filters/computed columns, without conflating
  their semantics with independent-row Menudet kernel registration.

Run with pinned dependency versions, fixed seeds and documented hardware/thread
settings. Use warmups, repeated randomized-order trials, medians and dispersion;
validate outputs before comparing timing. Avoid timing assertions in ordinary CI.
Use the required `blosc2` conda environment for local Python/tests/builds.
Report each workload and family separately (elementwise, reductions, indexing,
constructors and table operations), including worst regressions and absolute
latencies. An overall geometric mean must not conceal a failing family.

Proposed acceptance targets, subject to review after baseline collection:
no more than 5% geometric-mean regression on established large-array workloads,
investigate every reproducible regression over 10%, and no unexplained material
memory-growth regression. Report tiny-array absolute latency separately (relative
ratios are misleading there). These are review gates, not reasons to weaken safety
or skip workloads. If targets fail, optimize planning/caching or retain the faster
approved backend; defer phase 2 when necessary.

### Phase-2 exit gate

The compatibility matrix and explicit trusted escape hatch are approved. All safe
construction/evaluation routes are graph-based; platform correctness/security
tests pass. Benchmark reports support the default change, with every remaining
regression and limitation reviewed. Do not infer this gate from phase-1 CI success.

## 8. Delivery sequence and checkpoints

1. Inventory/corpus/threat-model review within the defined phase-1 scope; specify
   trusted operand adapters and permission/requirement rules.
2. Thin five-expression end-to-end prototype and early time/memory benchmarks;
   review scheduler integration before building the full registry.
3. Expand parser/registry and direct executor with isolated tests; no default changes.
4. Safe loader integration and graph lifetime/policy propagation; retain full route.
5. Phase-1 differential/security/platform tests and per-family benchmark report.
6. Review whether to release phase 1, continue experimentally, or stop.
7. Phase-2 API decision, direct graph construction and wider route coverage.
8. Optional narrow Menudet lowering and plan caching, measured independently.
9. Complete benchmark/platform review before approving phase-2 default expansion.

Maintain a progress log with completed gates, excluded forms, semantic differences,
benchmark artifacts and exact CI revisions. Run Ruff/format, doctests, the default
suite, relevant opt-in tests, native/WASM and installed-wheel tests. If Menudet
lowering changes, also run portable conformance and the independent math corpus.
No release publication or unrelated API changes are authorized by this plan.

## 9. Decisions to review before coding

- The operation inventory and any forms requiring explicit full mode.
- The route inventory implementing the defined phase-1 scope: current structured
  LazyExpr and CTable expression recipes, with historical full-only formats and
  nested legacy/proxy gates retained.
- Parser limits and policy for object/custom operands and data-dependent shapes.
- Phase-2 construction-policy API and mutable expression compatibility.
- Final performance budgets after reproducible baseline collection.
- Which Menudet segments are both semantically equivalent and worth accelerating.

The recommended initial experiment is **safe graph reconstruction with existing
numerical backends**. It provides the safety benefit directly; Menudet integration
can then be evaluated without confusing safety with a performance/backend rewrite.
