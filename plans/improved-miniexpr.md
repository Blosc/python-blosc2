# A reusable native execution layer for miniexpr

Status: implementation proposal for review. This document does not authorize
implementation or native API changes.

## 1. Objective

Make miniexpr sufficiently capable and flexible to serve independent C projects
and bindings for multiple languages. Move numerical execution machinery out of
Python-Blosc2 into the native layer that owns its semantics.

The target is more than adding strided output: provide reusable preparation,
validation, layout traversal, bounded buffering, safe work partitioning and
execution for both the interpreter and JIT. Python-Blosc2 becomes primarily an
API, object-lifetime and storage adapter.

Related plans:

- [remove-numexpr.md](remove-numexpr.md): removing the NumExpr runtime dependency.
- [evaluate-numpy-parity.md](evaluate-numpy-parity.md): current compatibility and
  performance evidence, including resident NumPy fallback limitations.
- [menudet-performance-parity.md](menudet-performance-parity.md): existing fused
  execution, storage-reader and scheduling work.

This work enables NumExpr removal but need not block that entire project. A
native serial interpreter plus explicit NumPy fallbacks can support the removal
milestone before every scheduling optimization is migrated.

## 2. Architectural ownership

| Responsibility | Target owner |
| --- | --- |
| Restricted expression parsing, graph construction, dtype inference | miniexpr |
| Immutable preparation and shape specialization | miniexpr |
| Numerical semantics, checked conversions, FP diagnostics | miniexpr |
| Array bounds/layout validation, broadcasting, gather/scatter | miniexpr |
| Tiling, scratch estimates, parallel eligibility and partitioning | miniexpr |
| Interpreter/JIT selection and execution reports | miniexpr |
| Scheduling tasks onto an application's existing worker pool | Host executor adapter |
| Optional standalone worker-pool implementation | Separate optional miniexpr component, if justified |
| Compression, decompression, storage caches and persistent containers | C-Blosc2 or another storage host |
| Python object inspection, references and exception translation | Python bindings |
| Public NumPy allocation policy and unsupported NumPy operations | Python-Blosc2 |

Do not move Python-specific APIs, NumPy objects, Blosc2 compression policy or a
particular storage format into miniexpr. Move the reusable execution rules.

### Two supported execution models

1. **Whole-domain execution:** the host supplies descriptors, optional readers
   and an execution policy. miniexpr prepares and partitions work, then runs it
   serially or through a host executor.
2. **Caller-driven block execution:** a host such as C-Blosc2 already controls
   workers and block lifetimes. It invokes prepared miniexpr work for a supplied
   block without creating another parallel region.

Both models must share preparation, validation, kernels, scratch rules and
diagnostics. Caller-driven execution is essential for fused
evaluate/filter/compress behavior; it is not a legacy path to replace with
full-array materialization.

## 3. Source assessment and existing foundations

Paths below refer to `../miniexpr` unless prefixed with `python-blosc2`.
Line references are approximate and describe the checkout inspected for this plan.

### Native array traversal

`src/dsl_array.c` already contains:

- `a_add()` / `a_valid()` (around lines 40–59): checked signed-stride arithmetic
  and allocation bounds, including negative strides.
- `a_offset()` / `a_coordinates()` (around lines 83–97): logical coordinates and
  byte offsets.
- `a_run()` (around line 232): shared preflight, bounded input gathering,
  storage-reader calls, map evaluation and serial reduction.
- A shared `me_artifact_eval_ex()` tile invocation (around line 384).
- Scratch and input-copy reporting through `me_array_report`.

Output handling is currently contiguous:

- Map destination: `output + begin * source_width` when no scratch is used.
- Reduction destination: `output + group * width`.

This is a good place to add bounded output scatter without changing every kernel.
Serial reduction output layout can change independently of accumulation order.

### Native graph execution

`src/dsl_graph.c:g_execute()` and `g_execute_staged()` validate raw output
pointers/capacities and assume C-order layout. The output contract must propagate
through final conversions and the final stage, while internal intermediates can
remain contiguous.

`me_graph_execute_alias()` is a separate transactional adapter: it allocates a
full contiguous result within a staging budget and publishes it only on success.
It does not provide bounded-memory arbitrary alias handling. Preserve that
distinction rather than silently changing its guarantees.

### Public interfaces and adapters

- `src/miniexpr_artifact.h`: `me_array_view` already describes input allocation
  bounds, offsets, shapes, signed strides and byte order. Its documented output
  contract is currently disjoint, aligned C-order storage.
- `src/miniexpr_graph.h`: ordinary and storage-source execution expose a raw
  output pointer/capacity. Plans and schedules have explicit ownership APIs.
- `python-blosc2/src/blosc2/dsl_graph_bridge.h`: adapts native graph APIs and
  capability availability.
- `python-blosc2/src/blosc2/blosc2_ext.pyx`: graph and block output admission
  currently requires writable, aligned C-contiguous destinations.
- `python-blosc2/src/blosc2/resident_execution.py`: resident worker pool,
  partitioning, lane-local safety proofs and worker joining.
- `python-blosc2/src/blosc2/storage_readers.py` and `native_graph.py`: storage and
  resident eligibility, scheduling heuristics and native dispatch.

Existing native tests such as `tests/numpy-compat/arrays.c` cover input reversal,
broadcasting, bounds/overlap rejection, tile-size-independent reductions and
scratch accounting. Extend this coverage instead of reproducing its cross-products
in Python.

## 4. Design invariants

1. **No language-runtime dependency.** Core execution is plain C, with explicit
   buffers, callbacks, status codes and ownership rules.
2. **One semantic owner.** Native metadata determines parallel safety and legal
   traversal changes. Python must not maintain another AST whitelist for the
   same decision after migration.
3. **Interpreter and JIT share contracts.** Lack of JIT is a backend availability
   decision, not evidence that an expression is unsupported.
4. **Contiguous fast paths stay fast.** New descriptors and scatter support must
   not add per-element branching to existing contiguous kernels.
5. **Bounded workspace.** Temporary memory scales with tile/workers except for
   explicitly declared staged intermediates or transactional alias staging.
6. **Preflight before execution.** Validate all invocation geometry, output and
   policy constraints before reading through callbacks or writing results.
7. **No replay after failure.** Execution errors never trigger another backend
   or repeat reader requests as a fallback strategy.
8. **Immutable preparation, invocation-local execution state.** Concurrent calls
   with independent buffers do not mutate cached plans or shared scratch.
9. **Preserve numerical domains.** Checked conversions, masks, reduction ordering
   and context-sensitive recipes cannot be arbitrarily retiled or reordered.
10. **No hidden nested parallelism.** One component owns each parallel region.

## 5. Public C API evolution

Names in this section are conceptual; choose final names after an API review.
Prefer additive, versioned entry points and compatibility wrappers over breaking
the existing raw-pointer APIs.

### 5.1 Writable destination descriptor

Introduce a writable view corresponding to `me_array_view`, with:

- Element dtype and rank/shape.
- Allocation base, accessible capacity and logical-origin byte offset.
- Signed byte strides.
- Explicit byte-order policy.
- Structure size/version and reserved fields where appropriate for ABI evolution.

Use a mutable base pointer rather than repurposing an input's `const void *` by
casting throughout the implementation. Public output descriptors must describe
the full accessible allocation, not confuse `array.nbytes` with the byte span of
a strided view.

First version: native-endian numeric destinations with exact result dtype and
shape. Support empty dimensions, scalar output, Fortran arrays, transposes,
stepped slices and reversals when non-overlap can be proven. Keep byte-swapped
outputs, arbitrary output casts and exotic overlapping layouts separate until
their contracts are qualified.

Existing pointer/capacity execution APIs synthesize contiguous descriptors and
continue to preserve their behavior.

### 5.2 Destination validation

Reuse checked bounds arithmetic, but add output-specific rules:

- Rank/shape/dtype must match the specialized result.
- Every reachable write must be inside the described allocation; account for
  negative strides and integer/pointer overflow on 32- and 64-bit hosts.
- Require appropriate alignment in the initial API. Later unaligned-store
  support can use byte copies if explicitly qualified.
- Prove distinct output elements occupy disjoint byte ranges. Input broadcasting
  rules are insufficient for writable destinations.
- A practical initial proof sorts nontrivial axes by absolute stride and checks
  that each axis step spans the complete footprint of faster axes. This is a
  sufficient condition, not a complete test for every possible legal layout.
- Reject ambiguous layouts rather than perform unbounded overlap searches.
- Preserve conservative input/output disjointness initially; document whether
  overlap tests use allocation bounds or reachable footprints.
- Validate empty outputs without executing kernels or dereferencing data.

Wrappers still own host permissions: C cannot discover that a Python array is
read-only, so the Python binding must reject it before making the native call.

### 5.3 Execution policy and workspace

Define a versioned policy distinguishing:

- Backend preference: interpreter-only, JIT-preferred, JIT-required.
- Thread/task budget and optional host executor.
- Tile/workspace budget, including aggregate parallel scratch.
- Floating status requests and existing numerical policies.
- Supported alias mode, initially disjoint only for the new bounded-output path.

Expose metadata-only workspace estimation before execution. Clearly separate:
iterator scratch, backend scratch, staged intermediates, host storage buffers,
output storage and alias staging. An optional caller-provided workspace API is
useful for embedding, but must not permit concurrent reuse of mutable workspace.

Avoid a single global mutable thread count or executor in the core. Policies
belong to invocations or immutable execution contexts with explicit lifetime.

### 5.4 Reporting and capabilities

Reports should distinguish:

- Selected interpreter/JIT implementation and fallback/rejection reason.
- Actual worker count, tile/task counts and execution mode.
- Gathered and scattered bytes, direct-write tiles and peak temporary bytes.
- Intermediate/normalization/full-output-copy bytes.
- FP flags and availability, plus deterministic failure location where supported.

Add fields through versioned report interfaces rather than silently changing ABI
layouts. Capability discovery lets older bindings retain a supported fallback.

## 6. Serial strided-output implementation

### Shared array loop

Extend `a_run()` to receive an internal validated destination abstraction:

1. Preserve the contiguous pointer fast path.
2. For a strided map destination, allocate one bounded contiguous output tile.
3. Evaluate the tile with the existing artifact call.
4. On successful tile evaluation, scatter values using logical coordinates and
   checked destination offsets.
5. For reductions, keep the accumulator loop and logical input order unchanged;
   write each completed group to its destination offset.
6. Account for output scratch and scattered bytes, including short final tiles.

Do not scatter a failed tile. Earlier successfully completed tiles may remain
visible; ordinary execution is not transactional. Document this behavior and
preserve the stronger guarantee only for explicit transactional APIs.

### Graph integration

- Propagate the destination through ordinary execution and source-backed maps.
- In conversion graphs, keep conversion inputs contiguous and use the supplied
  destination for the final conversion.
- In staged graphs, preserve contiguous intermediate storage and apply the user
  layout only to the final stage.
- Preflight the final destination before any stage or storage read begins.
- Keep alias staging behavior separate; any new strided transactional adapter
  needs its own budget and publish semantics.

This delivers functional output flexibility for interpreter and JIT without
adding stride parameters to every generated kernel.

## 7. Move scheduling and safety metadata into miniexpr

### Capability analysis

Native preparation should expose sufficient information to decide:

- Is the operation elementwise and independent across lanes?
- Does it contain checked conversions or weak-scalar evaluation whose error
  domain must be preserved?
- Does execution depend on global indices, block context, captures or masks?
- Are reductions present, and which grouping/order policies are permitted?
- Is it legal to partition, change tile size or reorder traversal axes?

Derive these properties from native IR and numerical semantics. Start with the
same conservative admitted set as current Python proofs, then expand using native
tests. Do not interpret a generic “elementwise” label as permission to reorder
errors or perform unchecked casts.

### Work partitioning

miniexpr produces work descriptions with logical ranges, input/output bindings,
scratch requirements and independent status slots. Partition decisions must
remain independent of Python array objects.

- Maps: assign disjoint logical regions after proving destination injectivity.
- Reductions: preserve existing serial semantics by default. Parallel reduction
  policies require explicit compatibility with the established native contract.
- Storage readers: preserve bounded requests and callback participation rules.
- Staged graphs: respect dependencies and intermediate lifetimes; avoid treating
  every stage as an independent concurrent operation.

First parallel release can restrict layouts and graph shapes while retaining a
general serial path. Unsupported parallel execution should choose serial native
execution, not require a Python numerical fallback.

## 8. Host executor interface

### Recommended first model

Provide a synchronous fork/join executor callback: miniexpr gives the host a task
count, a task-entry callback and invocation state; the host returns only after
every dispatched task has completed or been safely joined.

Specify these details in the public header and a standalone example:

- Executor/user-data lifetime and which thread invokes callbacks.
- Tasks may run concurrently and out of order, but each task index executes at
  most once; define how undispatched tasks are reported on submission failure.
- The caller thread may participate, subject to the worker budget.
- Per-task scratch/status ownership; no shared mutable kernel buffers.
- Submission failures must join in-flight work before returning an error.
- Completed work is never replayed, and failure selection is deterministic by
  logical task order rather than wall-clock completion.
- No callback may escape through a language exception or longjmp; wrappers
  translate exceptions into status and retain objects until all tasks finish.
- Reentrancy and nested calls must be documented. Caller-driven block execution
  disables child parallel regions to avoid deadlock and oversubscription.
- FP environment/status handling belongs to each executing worker; aggregate
  native flags after joins and restore worker environments as required.

Serial execution is the default when no executor is supplied. Add a standalone
thread-pool adapter only if embedding users need it; do not make pthreads or a
particular pool an unconditional dependency of the core.

### Storage callback concurrency

Existing synchronous reader APIs do not automatically promise concurrent calls.
Define reader capabilities explicitly:

- Serial-only reader: retain serial execution or serialize its calls according
  to a reviewed bounded strategy.
- Concurrent reader: host guarantees independent state or internal locking.
- Worker-local reader state: optional construction/destruction hooks if needed
  for private decompression contexts.

No reader or executor pointer survives its documented lifetime. Language wrappers
that need a runtime lock must acquire it only around their callbacks; native
compute should remain able to run without that lock.

## 9. C-Blosc2 integration and fused compression

Implement a C-Blosc2 executor adapter using its existing workers where the public
interfaces permit it. Audit the available worker API first; do not couple
miniexpr directly to private C-Blosc2 structs.

Keep compression-driven execution:

- A compression worker receives its block from C-Blosc2.
- Prepared miniexpr execution evaluates into the worker's block buffer.
- Filtering/compression proceeds immediately while data is cache-resident.
- Outer compression scheduling owns concurrency; miniexpr's block call is serial
  within that worker.

For NumPy/resident output, whole-domain miniexpr scheduling can use the host
executor. For compressed/mixed inputs, source callbacks supply bounded tiles.
Keep compressed-output byte emission in C-Blosc2: a generic output callback for
all storage formats is not needed for the first strided-buffer release.

Choose one owner for tile/chunk partitioning at each entry point. Avoid a second
whole-domain pass, redundant decode, nested pools or full-result normalization.

## 10. Python-Blosc2 migration

1. Extend bridge capability checks and Cython descriptor binding. Recover the
   actual owning allocation, offset and strides rather than passing only the
   view's logical `nbytes`.
2. Retain references to inputs, output, executor and callback state until native
   fork/join completion. Release the GIL around native work where supported.
3. Route eligible resident strided outputs to native execution; retain NumPy for
   unsupported dtypes/operations and older native runtimes.
4. Replace Python AST-based parallel proofs with native capability queries.
5. Move worker partitioning, tile sizing and workspace admission into native
   scheduling as each corresponding native path qualifies.
6. Remove superseded Python schedulers/helpers only after route and lifetime
   tests demonstrate equivalent behavior. Avoid retaining two default schedulers.
7. Keep `order` as a binding-level allocation policy. It must not select NumExpr;
   the destination descriptor conveys the resulting actual layout.
8. Preserve API-level shape/casting/read-only validation and exception types,
   translating native diagnostics consistently.

Do not promise native complex or arbitrary string support as part of this work.
Current real-numeric output support and the approved NumPy fallbacks remain
explicitly classified.

## 11. Performance optimization after correctness

Bounded scatter avoids full-result allocation but does not guarantee fast Fortran
output. Separate functional support from the following optimizations:

- Detect contiguous destination inner spans and write directly into them.
- Coalesce adjacent compatible axes and avoid repeated division/modulo per lane.
- For proven lane-local maps, choose traversal order considering all input and
  output strides, not only destination order.
- When inputs/output share a dense axis permutation, evaluate contiguous spans
  without gather/scatter where semantic admission permits.
- Choose tile sizes from item widths, active inputs, cache behavior and the
  aggregate workspace budget.
- Avoid false sharing at worker boundaries and excessive tiny tasks.
- Keep logical error reporting stable when execution order changes, or decline
  reordering for recipes that cannot support that guarantee.
- Preserve reduction traversal/rounding rules; do not reorder reductions simply
  because a different physical layout would be faster.

Gather/scatter should be implemented once in the native traversal layer and used
by both interpreter and JIT. Specialized stride-aware generated kernels may be
considered later if profiling justifies their maintenance cost.

## 12. Implementation phases and exit gates

### Phase A — Contracts and API design

- Audit current array/graph/block interfaces, native capability metadata and
  C-Blosc2 worker integration.
- Freeze descriptor, failure, overlap, workspace and executor contracts.
- Inventory Python scheduling decisions and map each to a native owner.
- Capture current correctness, route, memory and performance baselines.

Exit: reviewed additive API proposal and a precise first-release support table.

### Phase B — Serial output descriptors and scatter

- Implement writable descriptors and validation.
- Add bounded map scatter and reduction final-store addressing.
- Preserve raw-pointer API wrappers and the contiguous fast path.
- Extend ordinary/source-backed graph execution, conversions and staged finals.
- Add native tests, including interpreter/JIT variants and invalid descriptors.

Exit: standalone C programs execute supported strided-output workloads with
bounded scratch and without Python or C-Blosc2 dependencies.

### Phase C — Native capability and partition planning

- Produce conservative lane-local/partitionable metadata from native IR.
- Implement metadata-only work partitioning and aggregate workspace estimation.
- Establish deterministic status/error collection and task lifetimes.
- Compare admission decisions with current Python proofs on representative cases.

Exit: a serial test executor can run all planned tasks with equivalent results,
failure domains and memory bounds; unsupported parallel cases stay serial.

### Phase D — Executor integration

- Add synchronous host fork/join API and an example host executor.
- Qualify concurrent calls, submission failures, joins, reentrancy and FP state.
- Implement C-Blosc2 adapter and reader-concurrency capabilities.
- Preserve caller-driven block execution and reject/avoid nested parallelism.

Exit: real parallel execution works in a standalone C host and C-Blosc2, including
strided destinations, without Python scheduling.

### Phase E — Bindings and migration

- Expose new capabilities/descriptors in Python-Blosc2's bridge and Cython layer.
- Route strided resident outputs to native execution.
- Replace Python eligibility/partitioning machinery incrementally.
- Retain bounded compressed/mixed paths and fused compression.
- Update NumExpr-removal plan with completed native capabilities.

Exit: Python API tests prove direct destination handling, no NumExpr dependency
for migrated routes, and removal of the corresponding duplicated Python logic.

### Phase F — Layout-aware optimization and qualification

- Optimize spans, axis ordering and task sizing using profiles.
- Compare interpreter/JIT against NumPy and optional NumExpr references.
- Qualify platforms, memory bounds, sanitizer results and exact native revisions.
- Publish support/limitation tables and standalone integration examples.

Exit: documented correctness and performance results, explicit remaining gaps,
and approved release/integration decision.

Phases B and E's binding work may be integrated early for serial functionality;
the parallel machinery does not have to land before useful strided-output support.

## 13. Verification strategy

### Native tests

Extend existing array and graph tests for:

- C/F layouts, axis permutations, stepped/reversed views, scalar/empty shapes,
  degenerate axes and broadcast inputs.
- Canary bytes outside the logical destination, bounds overflow, insufficient
  capacity, misalignment, invalid dtype/shape and self-overlapping strides.
- Input/output overlap rejection and unchanged transactional alias guarantees.
- Tile tails, multiple tile sizes, mask participation, final conversions,
  reduction result layouts and staged final outputs.
- Identical serial semantics under interpreter/JIT, including FP/error status.
- Scratch accounting and budgets independent of total map output size.
- Concurrent invocations and deterministic task failures; no writes after return.
- Serial-only/concurrent readers, failed callbacks and failed executor submission.

Use sanitizers where supported. Keep exhaustive numeric/backend matrices in
miniexpr; do not duplicate them in Python-Blosc2.

### Integration tests

- Python: output identity/layout, read-only views, allocation-owner lifetimes,
  rebinding, native route selection, exceptions and no fallback replay.
- C-Blosc2: compression-worker ownership, bounded decompression, mixed sources,
  compression ratios/results and absence of nested parallel regions.
- Standalone C: build/install and execute with no Python/NumPy/Blosc2 runtime.
- An FFI-style example: opaque handles, descriptor construction, callback lifetime
  and status translation. A full additional language binding is not required to
  prove the first release's C ABI usability.

### Platforms and builds

Exercise Linux x86-64 and available ARM64, macOS ARM64, Windows and actual wasm32.
WASM must have a working serial/no-thread configuration. Test JIT-disabled,
available-JIT and unavailable-JIT policies explicitly. Verify 32-bit arithmetic
and avoid assuming POSIX threading or C99 complex support.

For Python/build commands use the `blosc2` conda environment. Use persistent
isolated builds and verify package/extension paths. Native changes need their
own qualification; preserve exact-revision evidence gates and aligned native pins
when integrating into Python-Blosc2.

## 14. Benchmarks and acceptance criteria

Benchmark copy, arithmetic chains, predicates/selection, transcendental maps and
representative reductions. Include contiguous and mixed-layout operands with
C/F/transposed/stepped/reversed outputs, interpreter and JIT, serial and parallel.
Use small inputs, 8M elements and 100M where memory permits; retain compressed and
mixed-input cases with realistic chunk/block geometry.

Measure separately:

- Preparation/specialization and cold startup.
- Warm native execution and complete public API cost.
- Fresh allocation versus warmed reused destinations.
- Logical throughput, scratch, gathered/scattered bytes and full-result copies.
- Compression-driven output throughput and memory use.

Use correctness qualification before timing, randomized engine order and repeated
samples. Report variability and actual routes; JIT disabled is not proof that a
native interpreter ran. NumExpr remains an optional external reference.

Acceptance gates:

- Existing C-output paths have no unexplained repeatable material slowdown; agree
  on numerical thresholds after collecting stable baselines.
- Strided maps do not allocate full-output staging unless explicitly requested
  by an alias/transactional contract.
- Serial native support works independently of any host executor.
- Parallel scratch respects the aggregate policy and task counts respect budget.
- Every migrated Python scheduling decision has native coverage and a clear owner.
- Supported compressed-output paths remain fused; no accidental full NumPy
  materialization or nested pool appears.
- Publish residual layout-performance gaps rather than claiming universal parity.

## 15. Risks and scope controls

| Risk | Mitigation |
| --- | --- |
| Public ABI becomes difficult to evolve | Add versioned descriptors/APIs and keep old contiguous wrappers. |
| Safe writable-layout proof rejects unusual valid views | Document a conservative subset; broaden only with evidence. |
| Parallel scheduling changes error/reduction semantics | Native IR capability proofs and deterministic task reporting. |
| Host callback lifetimes cause use-after-free | Synchronous join contract and explicit ownership examples/tests. |
| Host pool deadlock or oversubscription | Caller-driven serial block mode and documented nesting rules. |
| Scatter is correct but slower than NumPy | Preserve fallback visibility; optimize spans/traversal after profiling. |
| Graph staging adds unexpected memory | Keep intermediate budgets separate from iterator scratch. |
| Native/Python decisions remain duplicated | Track migration ownership and remove superseded helpers per phase. |
| Project expands into unrelated features | Defer native complex/string acceleration, general alias solvers and a mandatory built-in pool. |

## 16. Effort and review checkpoints

Source-based rough estimate for the narrower serial strided-output feature:
3–5 engineering days for native implementation/tests plus 2–4 for bindings and
platform qualification, assuming familiarity with these repositories. This is an
estimate, not a delivery commitment.

The broader native execution-layer migration adds API design, capability analysis,
host executor integration and ownership testing. Treat it as a multi-week staged
project; estimate Phases C–F after Phase A resolves executor and semantic scope.
Do not extrapolate a fixed deadline from the simple scatter-loop change.

Review checkpoints:

- [ ] Descriptor/API ownership and backward compatibility approved.
- [ ] First-release layout/alias/dtype subset approved.
- [ ] Serial native implementation and standalone example qualified.
- [ ] Partitioning/error-domain metadata approved.
- [ ] Executor and reader-concurrency contracts approved.
- [ ] C-Blosc2 fusion and Python lifetime tests passed.
- [ ] Duplicated Python scheduling removed for migrated paths.
- [ ] Platform, memory and performance evidence published.
- [ ] Remaining NumExpr-removal tasks reconciled with the new capabilities.
