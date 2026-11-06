# Replace legacy DSL with Menudet

## Release decision

Menudet replaces the unqualified alpha DSL interface in the next release.
Legacy public authoring **and historical source-recipe execution are retired**.
This supersedes the earlier proposal to retain trusted legacy execution under
`deserialize="full"`: that would create an indefinite compatibility obligation
and leave users with an ambiguous, partially retired authoring model.

Menudet is the supported computation language going forward. Menudet 0.1 remains
beta: API and artifact-format changes may occur and will be documented with
migration guidance. This is not a frozen 1.0 compatibility promise or a claim
that all platforms/backends have identical qualification status.

## Scope and invariants

- Retire `@blosc2.dsl_kernel`, direct `blosc2.DSLKernel(...)`, and
  `DSLKernel.from_source(...)`. Retain importable diagnostic stubs, not a warning
  or an escape hatch for new public legacy authoring.
- Reject persisted legacy source-backed DSL/Python UDF recipes under **both**
  safe and full deserialization. Full permission cannot enable retired formats.
- Reject historical computed/generated table columns carrying legacy source
  metadata; never silently discard columns or reinterpret old block/batch groups.
- Preserve ordinary arrays, supported LazyExpr recipes, generic Python UDF
  execution, and validated Menudet artifacts. In particular, MenudetFunction is
  a kernel author, not a replacement for LazyExpr.
- Reject MenudetFunction and MenudetKernel arguments to `lazyudf()` explicitly,
  with guidance toward direct invocation or `kernel.lazy(...)` respectively.
- Keep graph planning internal. Do not expose `NativeGraph` or add MenudetGraph.
- Retain only shared implementation infrastructure required by Menudet,
  numerical constructors, expression lifting and other internal dispatch.
  Internal source normalization is not a supported legacy authoring API.
- Do not change native ABI, serialized graph identifiers or artifact versions
  merely to simplify the public Python interface.

## Migration strategy: act before upgrading

The release is deliberately incompatible with historical source recipes. The
migration must be performed in an isolated installation of a previous release
that supports the user's particular format, not by re-enabling it in this release.

1. Back up source recipes, tables, containers and their durable operands. Keep
   the old installation and dependency versions available until comparison ends.
2. In that old installation, open only trusted historical data, using full
   permission if that release requires it.
3. Evaluate recipes with their original numerical/grouping behavior and write
   ordinary materialized arrays to new destinations. Do not copy a recipe carrier
   or preserve source-recipe metadata as though that were a materialized result.
4. For tables, export the values of all required stored/computed/generated columns
   into ordinary arrays or a new plain physical table. Generated columns can
   retain transformer metadata even though values are already stored; ensure the
   migrated output has no legacy computed/transformer recipe metadata. Do not
   assume saving the existing table removes those recipes.
5. Preserve relevant shape, dtype, row order, validity/null information and operand
   identities needed to interpret the snapshot. Use suitable supported export
   paths for nullable, variable-length or nested table data rather than blindly
   treating every column as a fixed-width NumPy array.
6. Reauthor the computation using Menudet. Explicitly choose types, scalar
   captures, logical partitions and table independent-row semantics.
7. In a separate candidate environment, compare Menudet values against the old
   materialized baseline using exact or appropriate numerical comparisons. Cover
   representative data, boundary cases and reductions whose grouping can change.
8. Complete the production upgrade only after comparison passes. Persist new
   Menudet recipes with durable operands, or retain snapshots if recomputation is
   unnecessary. Never overwrite originals during validation.

No automatic in-place conversion and no unlimited historical execution promise
are introduced. Keep lightweight format detection and actionable rejection errors.

## Implementation phases

### 1. Audit and separate public authoring from shared internals

- Inventory production/test/benchmark uses of DSLKernel and the old decorator.
- Keep the public names as stubs raising an immediate RuntimeError with:
  `Use @blosc2.menudet instead. See the migration guide:
  https://blosc.org/python-blosc2/guides/migrating_to_menudet.html`.
- Rename the implementation object to internal SourceKernel and import it
  explicitly where source normalization remains necessary. Do not implement
  caller inspection, environment opt-outs, or monkeypatch public API in tests.
- Update internal tests and benchmarks to use explicit implementation imports;
  those imports are not public migration advice. Existing native numerical
  coverage should not be duplicated by new Python cross-product matrices.
- Existing syntax validators may remain diagnostic helpers; they must not be
  advertised as a route to restore public legacy authoring.

### 2. Retire historical reconstruction

- Introduce UnsupportedLegacyRecipeError, separate from safe-policy permission
  errors. Its message must say that full permission does not help and point to
  previous-release materialization and the migration guide.
- Detect historical standalone LazyArray UDF markers, structured `lazyudf`
  payloads, MessagePack extensions and nested recipe operands.
- Reject before Python source parsing/execution, kernel reconstruction, operand
  resolution where metadata permits preflight, or destination writes.
- Apply the same rejection to high-level file/cframe/container loads and table
  computed/generated metadata, under safe and full policies.
- Remove the Python source reconstruction implementations. Any retained old
  helper name is only a rejection stub, never an executable loader.
- Do not reject historical expression-only recipes simply because the old marker
  name also contains LazyArray. Preserve their existing expression policy.
- Keep raw low-level carrier access distinct from loading an executable recipe;
  it does not make legacy execution supported.

### 3. Make the lazyudf boundary explicit

Before dtype inference, allocation or execution:

- MenudetFunction: TypeError recommending `function(*inputs)`.
- MenudetKernel: TypeError recommending `kernel.lazy({name: operand, ...})`.
- Include the migration-guide link in both messages.
- Ordinary Python UDFs retain their existing signature and runtime behavior.
- Do not add an adapter translating dtype, shape, block/chunk grouping or JIT
  settings; doing so would create a second potentially conflicting kernel API.

### 4. Documentation and public API

- Keep `doc/guides/menudet.rst` for walkthroughs and
  `doc/reference/menudet.rst` for supported API directives and concise contracts.
- Update `doc/guides/migrating_to_menudet.rst` with the pre-upgrade snapshot and
  reauthor/compare workflow, including table recipe metadata and backup caveats.
- Show migration of the **whole** old lazyudf construction, not just replacement
  of the decorator. Explain the separate roles of LazyExpr, Python LazyUDFs,
  MenudetFunction, MenudetKernel and their lazy results.
- Replace every promise of trusted legacy loading with the new unsupported-format
  policy. `deserialize="full"` remains meaningful for other supported features,
  not this retired format.
- Reconcile syntax/JIT/persistence/table references, public docstrings, tutorials,
  examples and release notes. Do not recommend internal SourceKernel authoring.
- Keep old benchmark/platform results clearly historical rather than representing
  them as Menudet qualification. Port representative runnable examples.
- Keep the public graph export removed and its formats unchanged.
- Publish and verify the migration guide at
  https://blosc.org/python-blosc2/guides/migrating_to_menudet.html alongside release.
- Use explicit valid upstream specification links pinned to the bundled miniexpr
  revision; the spec files belong to miniexpr, not this repository.

### 5. miniexpr boundary

This Python policy does not authorize a blanket removal of native APIs or shared
compiler infrastructure. Menudet is the recommended native language and legacy
full DSL is a separate semantic profile. Inventory native consumers and API
replacements before any upstream deprecation/removal. Do not silently change
existing native entry-point semantics. Keep exhaustive numerical/backend
qualification upstream. No sibling-checkout or dependency-pin changes are
required merely to reject Python's historical source recipes.

## Validation and completion criteria

- Public decorator/constructor/from_source use fails before inspecting or running
  user code; imports still provide actionable retirement diagnostics.
- MenudetFunction/MenudetKernel passed to lazyudf fail with the correct alternative,
  whether dtype is supplied or omitted. Ordinary Python UDFs still execute.
- Menudet authoring, specialization, native interpreter/JIT dispatch, persistence,
  expression lifting and numerical constructors remain functional.
- Genuine historical fixtures and crafted malformed/side-effect-bearing recipes
  reject under safe/full policies, standalone and nested, without reconstruction.
- Historical computed/generated table recipes reject rather than losing columns;
  ordinary physical tables and new Menudet columns still round-trip.
- Nested inline legacy descriptors preflight before sibling references resolve,
  including under full permission. Ordinary full-permission expressions retain
  their supported behavior.
- Plain-array snapshots and supported LazyExpr recipes load normally; expression
  dependencies that point to retired recipes fail with migration guidance.
- Execute current migration-guide examples, the Menudet tutorial and migrated
  standalone examples. Clearly distinguish previous-release-only snippets.
- Run focused tests and the full default suite in the blosc2 conda environment;
  verify package and extension paths. Rebuild only if native/Cython code changed.
- Build Sphinx and check links. Report unrelated baseline warnings honestly.
- Run lint/format/diff checks without rewriting unrelated user work.
- Heavy, network, optional native semantic matrices and cross-platform release
  qualification remain their separate gates, not implied by a local suite pass.

Completion means there is one supported portable-kernel authoring interface,
retired recipes cannot execute under any deserialization permission, migration
guidance provides an old-release escape route, and supported arrays/expressions/
Python UDFs/Menudet computations continue to work.
