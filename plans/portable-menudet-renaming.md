# Menudet naming, 0.1 beta, and documentation plan

## Goals and decisions

- Menudet is the language; miniexpr is its native implementation. Portable is a
  descriptive adjective, not a separate language or API family.
- Publish the language and artifact specifications as **Menudet 0.1 beta**.
  Beta signals a usable design expected to remain relatively stable, while allowing
  changes before a 1.0 compatibility commitment.
- Menudet has never been released. Remove the draft 1.0 artifact contract and old
  Menudet-specific `Portable` API and storage names directly; do not add aliases,
  migration paths, or legacy readers for them.
- Keep unrelated uses of portable, such as portable remote archives, unchanged.
- `DSLKernel` and `@blosc2.dsl_kernel` remain separate, broader authoring APIs.

## Implementation status

Implemented in Python-Blosc2 and the sibling miniexpr checkout:

- Public author/executable/error/validation names, internal modules and native
  bindings, lazy/table storage discriminants, and artifact language identifiers
  use Menudet. Old unreleased public names have no aliases.
- Language and artifact versions are 0.1 beta. Old draft versions reject. The
  graph numerical-policy tag is `menudet-numpy-0.1`; graph wire-format revisions
  and the independent numeric-table schema revisions remain separate contracts.
- The user guide is now `doc/reference/menudet.rst`; syntax, accuracy, API,
  release notes, examples, tools and native specification links are aligned.
- `CMakeLists.txt` pins the published native revision
  `d8b01fb0f8db817b7b4b8929cb03cfb1c13e98e5`, including the beta renaming and
  captured-scalar reduction fix. Source overrides must supply the same contract.

Validation on macOS ARM64:

- Default Python suite and doctests: **12,002 passed**, 88 skipped.
- Native Menudet/NumPy/graph tests: **392 passed**.
- Native build without the artifact loader succeeds; four focused core typing,
  interpretation and validation tests pass.
- Independent clean wheel built from the previously patched pin: **270 focused
  integration tests passed**. The editable install was also rebuilt/refreshed.
- Ruff checks and documentation builds pass. Sphinx still reports unrelated
  existing documentation warnings; the new reference has no page-specific
  warnings.
- The initial extended run exposed two pre-existing interpreter/JIT mismatches
  in uniform captured-scalar reductions. These are now fixed: the interpreter's
  contiguous-input reduction shortcut excludes uniform captures and delegates
  broadcasting to the existing evaluator. The published native pin includes the
  fix. Native regression tests cover all seven block reductions, weak/typed
  captures, float32/float64/integer/Boolean values, empty/single/multiple lanes,
  and absent/all-true/partial/all-false masks, with independently checked results.
  The regression failed before the fix and passes afterward. Native TCC/system
  compiler parity and ASan/UBSan checks pass. After rebuilding the editable
  package, the broader opt-in Python run passes **2,318 tests**, with 34 skipped
  and one deselected by the default marker policy; the original eight captured-extrema/
  truth cases and all 374 tests in their parity module pass.
- Windows and WASM execution were not requalified locally. Their complex support
  is documented from the native implementation and platform tests, independently
  of the beta stability label.

## Starting point and naming collision

The latest Python-Blosc2 commit, `1d065fc7` ("Consume native reduction metadata
and inferred artifact outputs"), does not change these goals, but modifies the
affected code. Preserve its native output inference and reduction metadata paths,
including native bridges, graph scheduling, and their tests; do not reintroduce
the former JSON round trips or inference probe as part of this cleanup. Check the
current working tree and both repository revisions before implementation.

The decorator wrapper currently occupies `MenudetKernel`, while `PortableKernel`
represents the executable specialized object. Rename the wrapper to
`MenudetFunction` and reserve `MenudetKernel` for the executable object. An
artifact is its serialized JSON. Rename `PortableArtifactError` to `MenudetError`
and `validate_portable_dsl` to `validate_menudet`. Rename the observable returned
lazy-array type to `MenudetLazyArray`.

## Work plan

### 1. Establish the implementation baseline

- Inspect the pinned and local miniexpr revisions, build configuration, existing
  tests, specification versions, artifact and graph formats, and storage paths.
- Verify complex support across macOS, Linux, Windows with clang-cl, and WASM32
  against implementation and platform test evidence. Identify JIT and floating
  status limitations separately; do not equate existing CI with release-wheel
  qualification.
- Inventory affected public APIs, imports, docstrings, errors, examples, tests,
  native bindings, metadata tags, and documentation links.

### 2. Rename APIs and Menudet-specific implementation identifiers

- Resolve and apply the two distinct kernel/author class names above; update
  exports, imports, return types, annotations, docstrings, exceptions, diagnostics,
  examples, and tests. Remove old public names without aliases.
- Rename Menudet-specific modules, helper names, and native binding identifiers
  where practical, updating Python, Cython, C bridges, and native callers together.
  Preserve the behavior of the latest native metadata and inference changes.
- Rename storage and table metadata identifiers such as `kind: portable` and
  `transformer_kind: portable` to `menudet`, updating every writer, reader,
  validation path, and test together. Remove old Menudet draft tags and fixtures.
- Coordinate changing the artifact language identifier from `miniexpr` to
  `menudet` with the native specification, producers, and loaders. Miniexpr
  remains the name of the implementation. Check other format identifiers,
  including graph identifiers, independently; change only as needed for a
  coherent contract, not by blind text replacement.

### 3. Establish Menudet 0.1 beta

- Replace the draft 1.0 language and artifact specifications with 0.1 beta in
  miniexpr; align the Python bindings, defaults, validation, export, import,
  examples, fixtures, and tests with version `0.1`.
- Remove acceptance of obsolete draft 1.0 artifacts; no migration or re-export
  guidance is needed for a public release. Treat graph format and numerical-policy
  identifiers as separate contracts and coordinate any required updates with the
  native implementation rather than assuming they share the artifact version.
- Document the beta stability expectations and the eventual purpose of a 1.0
  compatibility commitment.

### 4. Rewrite the user-facing reference

- Rename `doc/reference/portable_dsl.rst` to `doc/reference/menudet.rst` and
  update the index, links, API references, and examples.
- Lead with what Menudet is, its beta status, and a short `@blosc2.menudet`
  example. Cover artifact export/import and execution, types and numerical
  semantics, block reductions and logical partitions, persistence, table
  integration, validation, errors, and the security boundary.
- Present JIT and platform restrictions as user-visible limitations. Link to
  syntax, numerical accuracy, and API references rather than duplicating them.
- Remove incremental development history, CI qualification ownership, benchmark
  notes, backend internals, and dispatch-threshold inventories from the user
  guide. Preserve important behavior and numerical caveats.
- Reconcile stale and contradictory claims, including Windows complex support,
  JIT availability, wrapping versus checked integer arithmetic, and the old
  Python inference probe description. Update the native specification and feature
  matrix as well as Python documentation.

### 5. Validate

- Test public exports and removal of old names; authoring, specialization,
  inferred output, artifacts, validation, exceptions, lazy execution, graphs,
  persistence, and computed/generated table columns.
- Confirm 0.1 artifact round trips and rejection of old draft versions and tags;
  check graph and storage metadata separately. Cover native reduction metadata
  and inferred output regressions introduced by the latest commit.
- Run focused native and Python tests in their appropriate environments, build the
  docs, and check examples and links. Verify actual imported extension paths.
- Consult or run platform-specific tests for Windows and WASM; report any missing
  evidence instead of asserting unverified qualification.
