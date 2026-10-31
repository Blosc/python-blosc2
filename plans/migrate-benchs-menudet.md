# Migrate tutorials, examples and benchmarks to portable computation

## Direction and release gate

Keep **`@blosc2.menudet` for validated lazy kernel authoring** and **`@blosc2.jit`
for its original NumPy-style eager/tracing convenience contract**. These are
different interfaces, not two spellings of the same operation. Use `@menudet`
in kernel examples, tutorials and benchmarks; retain `@jit` in material that
teaches eager results, `out=`, tracing/control-flow dispatch or pandas engines.

The proposed portable flag on `@jit` was prototyped and rejected after review:
combining different return types/authoring contracts behind a flag was less clear
than retaining separate decorators. The prototype was removed. The historical
[decision record](portable-jit-api.md) and its raw measurements remain labeled as
history, not supported API guidance. See [menudet-api.md](menudet-api.md) for the
chosen separation and fresh 200M measurements. Artifact identifiers, native ABI
and retirement/deserialization invariants are unchanged.

Menudet must provide:

- Lazy results, explicit `.specialize(...)`, and kernel binding/evaluation.
- Defined input/output types, scalar annotations and capture semantics.
- Explicit logical domain, partitions, context indices and result cardinality.
- Validated artifacts usable in persistent recipes and table columns.
- No execution of the Python body, including on errors. Native interpreter
  fallback is distinct from Python fallback and may remain supported.
- Execution controls independent of authoring semantics: interpreter, JIT policy,
  compiler backend, FP policy, and an explicit require-JIT option where supported.

Preserve existing `@jit` tracing/control-flow dispatch, eager return types, `out=`,
storage kwargs, `disable=`, and pandas engine behavior. Existing `strict=True` is
not an alias for Menudet. Execution policy (interpreter/JIT, compiler/backend) is
independent of the authoring decorator; native interpretation is not Python fallback.

## Scope and inventory

Baseline: tracked files audited on 2026-10-31. Include Python scripts, notebook
sources/outputs, and Python embedded in JavaScript/HTML. Exclude untracked local
experiments from the release count. Do not bulk-replace every `lazyudf` call:
ordinary Python/Numba/Cython UDFs remain valid and serve different purposes.

### A. Seven benchmark workload migrations

| File | Migration and qualification | Effort |
|---|---|---|
| `bench/optim_tips/tip_11_dsl_random.py` | Migrated to `@menudet`; explicit domain, casts and partitions; 20M/200M host CC measurements. Legacy authoring retained only as labeled internal controls; current `_flat_idx` syntax is native Menudet context. | High |
| `bench/chicago-taxi/string-ops.py` | Port string kernel, declare output dtype/width, validate conditional/string behavior, rerun engine and codec comparisons. | High |
| `bench/ndarray/compute_expr_udf.ipynb` | Port three predicates and bindings, restore correctness assertions, regenerate timings and plots. | High |
| `bench/ndarray/dsl-kernel-bench.py` | Port ten kernels and execution calls; replace `.dsl_source` assumptions; preserve expression/kernel baselines and backend selection. | Medium |
| `bench/ndarray/jit-dsl.py` | Port four kernels, annotate scalar parameters, separate specialization/cold/warm timings and retain backend comparisons. | Medium |
| `bench/ndarray/linear-constructor.py` | Replace legacy context indices and scalar-only domain binding; validate endpoint/rounding semantics against NumPy and `blosc2.linspace`. | Medium |
| `bench/ndarray/sum-linear-idx.py` | Port index-based generation; preserve and check lazy reduction versus materialize-then-reduce comparisons. | Medium |

These are workload benchmarks even where comments currently call their
`SourceKernel` imports “internal normalization” or “internal constructor” uses.
Changing the comment is not a public-API migration.

### B. Two artifact-authoring cleanup candidates

Both already execute Menudet artifacts; they are not legacy execution migrations.

- `bench/mathlib-performance.py`: replace private source/export authoring with
  the chosen public portable authoring interface, preserving math backend,
  accuracy and FP settings. Do not lose pragma/policy configuration in the move.
- `bench/jit-coverage.py`: decide whether to retain a documented internal fixture
  builder. Its generated typed sources exercise native artifact coverage and
  export is outside reported timings. Public authoring is desirable only if it
  does not change the coverage target. Keep native numerical cross-products in
  miniexpr rather than duplicating them in Python tests.

### C. Three intentional old-versus-new comparisons

Do not replace the legacy control arm with the same modern implementation:

- `bench/menudet-integer-increment.py`: retain the explicitly labeled legacy
  control and its modern public `@menudet` arm.
- `bench/menudet-profile-parity.py`: retain the legacy control and distinguish
  shared artifact-fixture construction from public authoring coverage. Consider
  adding a public-authoring route without relabeling the old measurements.
- `bench/ndarray/native-graph-preparation.py`: keep old export-twice preparation
  as an internal baseline; do not confuse it with supported user authoring.

Keep these comparisons clearly internal, or archive them if no longer useful.
Internal compatibility must not restore retired public authoring or source-file
reconstruction.

### D. Two archived JavaScript demos: separate revival decision

- `bench/js-transpiler/dsl-js-node.mjs`
- `bench/js-transpiler/newton-dsl-js.html`

`bench/js-transpiler/README.md` already marks them historical. Leave them archived
unless explicitly reviving this benchmark family. Revival requires matching
Pyodide wheels, removal/replacement of unsafe version-skew overlays, current
portable authoring/transpiler integration, backend/fallback correctness checks,
and fresh Node/browser measurements. Do not imply that swapping the decorator
qualifies this platform or that legacy JS backend assumptions hold for Menudet.
The pure-JavaScript `worker-pool-bench.mjs` needs no Python authoring migration.

Thus the benchmark inventory is **14 code files: 7 workload migrations, 2
artifact-authoring candidates, 3 intentional comparisons, 2 archived demos**.

### E. Three legacy example notebooks

| File | Work |
|---|---|
| `examples/ndarray/black-scholes_hist-dsl.ipynb` | Port pricing and bin-index kernels; replace context symbols; annotate scalar/string configuration as applicable; retain the generic histogram UDF unless intentionally redesigning it. Check bin boundaries, counts, pricing accuracy and memory. |
| `examples/ndarray/mandelbrot-dsl.ipynb` | Port portable-kernel arms, retain the ordinary Numba UDF baseline, verify escape/iteration semantics, rerun comparisons and regenerate images/outputs. |
| `examples/ndarray/mandelbrot-pure-dsl.ipynb` | Port authoring and binding/backend configuration; verify output dtype, iteration counts and geometry; rerun comparisons and regenerate outputs. |

All three contain published execution output, so changing source cells alone is
insufficient. The Black–Scholes notebook also contains obsolete prose claiming
DSL kernels are necessarily elementwise: explain current cardinality contracts
without implying that a histogram is automatically supported by block reduction.

### F. Existing Menudet material to retain and audit, not re-port

One tutorial:

- `doc/tutorials/03.lazyarray-udf-kernels.ipynb`

Eight example scripts:

- `examples/menudet.py`
- `examples/menudet_decorator.py`
- `examples/ctable/udf-computed-col.py`
- `examples/embedded-expr-udf-b2z.py`
- `examples/ndarray/dsl_save.py`
- `examples/ndarray/malformed_dsl.py`
- `examples/ndarray/portable_artifact.py`
- `examples/vlstore-lazyudf.py`

Keep the existing `@menudet` decorators and public type names. Audit prose,
expected errors and measured outputs without renaming the entry point. Preserve
the validated-kernel examples and durable-reference
persistence contract. Decide whether low-level JSON import/export remains an
advanced example rather than silently removing functionality. Rename files only
where useful, updating links; a historical filename is not itself a runtime bug.

Together with the modern arm of `menudet-integer-increment.py`, this is **10
existing modern files already using the chosen interface**, not 10 additional
legacy workload migrations. The benchmark is already counted in section C.

## Documentation synchronized with the code

- `doc/guides/menudet.rst`: teach `@menudet` and distinguish its validated lazy
  authoring contract from ordinary eager/tracing `@jit`.
- `doc/guides/migrating_to_menudet.rst`: update destination API and guidance while
  retaining explicitly labeled legacy “before” examples and migration warnings.
- `doc/guides/optimization_tips.md`: replace tip 11's historical results with a
  coherent measured recommendation; revalidate tip 15's `@jit` descriptions and
  figures where dispatch changes affect them.
- `bench/chicago-taxi/README.md`: update code-path explanations, tables, figure
  and conclusions together with the string benchmark.
- `bench/optim_tips/README.md`: preserve the script/plot synchronization rule.
- Audit reference pages, release notes, errors and tests that link to these
  tutorials; retain `@menudet` guidance and do not publish a conflicting `@jit` mode.

For tip 11, choose **one measured story** before implementing it:

1. A portable integer hash as a kernel-authoring illustration, clearly not a
   statistically rigorous RNG; or
2. `blosc2.random` versus NumPy staging as the recommended random-generation tip.

Do not show `blosc2.random` code with the hash kernel's speedup. If retaining
both stories, give them separate benchmark labels, measurements and claims.

## Execution phases

### 0. Qualify the existing Menudet API

- Keep separate decorator contracts and consistent type names/options/error guidance.
- Test lazy returns, specialization, captures, scalar annotations, broadcasting,
  context/domain handling, partition/cardinality rules and native errors.
- Verify no Python body execution/fallback and preserve ordinary `@jit` and
  pandas contracts. Keep tests at the API/binding layer.
- Preserve rejection of historical source recipes under safe and full loading.
- Resolve domain/context or backend gaps before promising a mechanical migration.

### 1. Migrate correctness-first, at small sizes

- Audit the modern Menudet tutorial/examples first to establish canonical idioms.
- Port simple predicates/arithmetic, then loops/scalars, then context-dependent
  generators, strings and mixed histogram workflows.
- Compare against independent NumPy/Python references and, where necessary,
  materialized results from an isolated old release. Never overwrite originals.
- Check dtype, shape, row order, string width/truncation, integer conversions,
  endpoint behavior and grouping, not just approximate values.
- Test table independent-row execution and recipe round-trips with durable
  operands. Do not send portable functions/kernels to `lazyudf()`.
- Unsupported workloads must be documented as blocked or redesigned explicitly,
  not routed through private authoring and declared migrated.

### 2. Rerun performance experiments

Use the `blosc2` conda environment for all Python/build/test commands. Record the
actual package and extension paths, Python/C-Blosc2/miniexpr revisions, platform,
CPU, threads, compiler/backend, FP settings, seeds, shapes, dtypes, chunks, blocks,
compression, input layout and measurement protocol.

- Separate decoration/specialization, cold compilation/load, warm execution,
  allocation, compression and IO where relevant. State exactly what is timed.
- Match eager/lazy materialization and output types across competing routes.
- Verify correctness before timing. Do not rank mismatching results as faster.
- Check the backend actually used; a requested JIT is not proof of JIT execution.
- Use repeatable warmups/repetitions; isolate peak-memory runs as appropriate.
- Rerun relevant baselines on the same machine/configuration, not only the new
  kernel arm against old recorded competitor numbers.
- Start with small smoke runs, then representative published sizes. Full Chicago
  Taxi, large notebooks and multi-backend matrices require explicit time/data
  budgets. Qualification of one platform does not qualify all platforms.

### 3. Regenerate artifacts and interpret again

- Update tip 11's PNG and Chicago Taxi's `string-ops.png`, plus every affected
  notebook's executed outputs/embedded plots. Retain raw timing data and metadata
  alongside or linked from the results, with reproducible commands.
- Recompute ratios from the new data. Reassess the recommendation itself:
  performance rankings, memory benefits and bottleneck explanations may change.
- Remove obsolete claims rather than preserving their wording with new numbers.
- Keep historical results only if clearly versioned and separated from current
  recommendations. Do not relabel prototype or legacy output as current Menudet measurements.
- If tip 15 is rerun, regenerate both its control-flow and elementwise plots and
  their discussion; this is requalification, not legacy decorator migration.

### 4. Final validation and completion criteria

- Run affected scripts and executable documentation snippets in isolated output
  directories. Execute notebooks with matching dependencies and clear provenance;
  expensive full runs are separate from routine CI smoke tests.
- Add lightweight tests for the supported authoring/binding contracts and small
  representative workload cases. Do not add timing thresholds to normal tests.
- Build documentation and check renamed-file links, labels and error guidance.
- Search tracked tutorials/examples/benchmarks for retired decorators,
  `DSLKernel`/`SourceKernel`, `.export(...)`, legacy context symbols and portable
  objects passed to `lazyudf`; inspect notebook outputs as well as source cells.
- Maintain an explicit allowlist of historical migration examples, archived JS
  demos and intentional internal comparison/fixture builders. A search hit is not
  automatically a defect, but every retained authoring hit needs a reason.
- Mark each item migrated, intentionally retained, archived or blocked. Complete
  only when code, correctness evidence, figures and prose tell the same story.

## Suggested priority

### First workload status (2026-10-31)

Tip 11 is **ported to public `@menudet` and host-qualified at 200M**.
A temporary decorator-consolidation experiment was rejected; the current plot is
regenerated from a fresh `@menudet` run, not relabeled prototype measurements. See
[chosen API/200M qualification](menudet-api.md), [first qualification](tip11-qualification.md)
and [parallel-generator follow-up](tip11-parallel-qualification.md) with raw results.
Exact integer checks, a regenerated hash-only figure, updated guide caveats,
contiguous-partition scheduling, aligned native generator materialization and
native miniexpr bitwise/integer-local lowering accompany it. The system-CC hash
now matches true legacy throughput at 20M and 200M/12 threads on this host; single-thread
and TCC performance remain distinct qualifications, not implied parity.
This does not qualify `blosc2.random` or 200M on other platforms/backends. The retained
`SourceKernel`/`lazyudf` arm is an explicitly internal old-versus-new control,
not supported user authoring. Both repositories' fixes are required.

1. Keep the existing modern tutorial/examples on `@menudet` and audit their remaining contracts.
2. Tip 11's hash recommendation/figure is host-qualified; keep any future `blosc2.random` study separate.
3. Port the standalone arithmetic/loop/generator benchmarks and validate timing
   boundaries before collecting representative results.
4. Port Chicago Taxi and the four legacy notebooks (one benchmark, three
   examples), budgeting separately for full runs and interpretation.
5. Clean up artifact authoring where justified; document intentional controls.
6. Leave archived JS revival out of the main migration unless separately funded.

The core legacy workload scope is **10 files: 7 benchmarks and 3 example
notebooks**. There is no active legacy-authoring tutorial to port, but the modern
kernel tutorial already uses the chosen public entry point. Figure regeneration and
interpretation are explicit deliverables, not follow-up polish.
