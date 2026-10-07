# Portable DSL 1.0 implementation progress

## Published release certification — green platform matrix

The user authorized publishing changes and monitoring/fixing CI until green.
Native Menudet documentation was published as miniexpr
`36770f87e10f1b92bfe0f9f92a7039eba2e976ea`; all seven native jobs passed in
[37618199852](https://github.com/Blosc/miniexpr/actions/runs/37618199852).
Python still pins the already-green runtime revision `3418cdce...`: the subsequent
native changes are documentation/matrix metadata only, not runtime changes.

Python release-hardening and certification fixes are published on `portable-dsls`.
Certified implementation/workflow revision:
`e768bc538cb096c7f5c669046ed3aed5e55d9c6b`. Successful runs:

- [Tests 37622410469](https://github.com/Blosc/python-blosc2/actions/runs/37622410469)
  — all standard platform jobs green, including independent math reports.
- [WASM 37622410514](https://github.com/Blosc/python-blosc2/actions/runs/37622410514)
  — Python/Pyodide suite plus the independent corpus green.
- [Python wheels 37622420247](https://github.com/Blosc/python-blosc2/actions/runs/37622420247)
  — **27 successful jobs**: seven native/WASM build jobs and twenty abi3 test
  jobs across Python 3.11–3.15. Includes Linux x64/ARM64, Windows x64/ARM64,
  macOS x64/ARM64, Pyodide 3.13/3.14 and free-threaded builds. PyPI upload is
  intentionally skipped (no tag/publication). The pre-existing skip of testing
  macOS Intel cp315/cp315t wheels is retained for unavailable pre-release dependency
  wheels; do not claim those specific wheels were tested.

The first two wheel runs (`37618308296`, `37620050340`) failed because importing
the extension safely enables the GIL on free-threaded Python, emitting a
RuntimeWarning that pytest promoted to an error. A command-line filter initially
failed because pytest treats CLI warning messages literally, not as regexes.
Moved the **exact extension-specific regex** into `pytest.ini`; other runtime
warnings remain errors. A local synthetic check confirms both acceptance of the
expected import warning and rejection of an unrelated numeric RuntimeWarning.
No `PYTHON_GIL=0`, no no-GIL safety declaration, no tolerance changes or new
conformance skips were introduced.

Added `workflow_dispatch` to the wheel workflow to certify development revisions
without release tags. Both wheel Pyodide versions and each abi3 matrix job now
also run the independent corpus.

Downloaded and verified all five native Python math reports: every report has
**284 samples and zero failures**. Maximum observed ULP: Windows **2**, Linux
**2** (including NumPy 1.26 and Python 3.14 jobs), macOS ARM64 **3**. Reports are
under `<temp-root>/menudet-native-ci-math-reports`. Wheel/WASM corpus executions
are additionally certified by the successful corresponding job commands.

The earlier external-platform/published-branch blockers below are superseded.
Release certification for the tested matrix is now green; final release version,
publication, beta feedback and compatibility freeze remain separate decisions.

## Menudet / 4.15.0 release hardening — current handoff

The user selected **Menudet — a little language for portable computation on
arrays and tables**, and approved the release-hardening plan. Menudet is the
portable language; miniexpr implements it. Existing API/format identifiers remain
unchanged. Broader JIT acceleration is deliberately separate: the native guard
still forces portable interpretation, preserving checked/ordered semantics.

### Delivered in this increment

- Both version fields now read `4.15.0.dev0` (development, not final publication).
- Python reference, release notes and native spec introductions name Menudet,
  retain the draft warning, and distinguish full DSL from portable semantics.
- `examples/menudet.py` demonstrates the same array/table kernel plus explicit
  logical block-scalar reductions; assertions passed against the installed wheel.
- Added `scripts/certify_menudet_math.py` and its checked-in JSON corpus: **284
  samples, 40 canonical functions, float32 and float64**, generated with mpmath
  1.3.0 at 100/200 decimal digits. Regeneration equality and deliberate failure
  detection passed. No new runtime dependency; mpmath is needed only to regenerate.
- Function-specific finite sample budgets and coverage limitations are documented
  in `doc/reference/menudet_accuracy.md`. These are not universal ULP guarantees
  or certification of all possible inputs. Native exceptional/exact/alias fixtures
  remain required.
- Wired the corpus into native Python, WASM and installed-wheel CI. Native jobs
  upload platform reports; tested wheels also execute the seven portable modules.

### Verification

All Python/build commands used conda `blosc2`. Fresh stock build fetched native
`3418cdce4b5c11e1661c8b6453e3d31d94b0b2b4`; both source override cache entries
are empty. Isolated imports asserted the installed target path and excluded the
editable finder only within the test subprocess.

- Working-tree stock wheel: **11160 tests passed, 53 skips** (excluding heavy,
  network and TUI), plus **120 installed-module doctests passed, 2 skips** in a
  clean working directory. The serial combined invocation initially encountered
  one relative-filename save failure: installed doctest modules are outside the
  repository root fixture's isolation. All tests passed, and all doctests passed
  on the clean-directory rerun; no user's existing file was removed to fix it.
- Independent corpus on that wheel: **284 passed**, maximum observed error
  **3 ULP** (lgamma/float64); no tolerance changed after observing results.
- Built an **8,044,023-byte clean sdist**, verified inclusion of checker, corpus
  and example, built a wheel from the unpacked sdist, and isolated-installed it.
  This wheel passed **183 portable tests, 17 explicit external opt-in skips**,
  plus all **284 independent math samples**.
- Clean-sdist wheel SHA256:
  `c7fcc99d44fe12a7cdffc717eff4bc04450954c6d51eb98b282d09c2646b0d67`.
- Clean snapshot copied only tracked files and the four requested new files;
  unrelated untracked local work was neither packaged nor deleted. A direct
  dirty-checkout sdist attempt timed out and is not a release artifact.
- Ruff, format, workflow YAML, reference regeneration and whitespace checks pass.

### Documentation disposition

Fixed installation list-table indentation, stopped treating generated `doc/html`
and autosummary stash trees as sources, and documented NumPy's finfo alias without
reparsing its incompatible nested sections. Documented deserialization policy
types/errors and taught the public-API tripwire to recognize manual `py:class`.
Fresh whole-tree Sphinx now has **zero ERROR diagnostics**, and no Menudet page
diagnostic. It still has existing non-Menudet warnings: the fresh build reported
769 (639 autosummary stub, 100 duplicate object/label, 11 ambiguous references,
2 image, 17 other); the final follow-up with notebook execution disabled reports
772, with no undocumented-finfo warning. They remain visible, unsuppressed
documentation backlog, not a claim of a warnings-as-errors clean build. Menudet
pages and parser errors are the scope of this draft's documentation acceptance;
broad autosummary/reference cleanup is explicitly deferred.

### External gates still pending

The updated Python branch/workflows are not remotely available, so no new Python
platform job can certify these changes yet. Native miniexpr's seven-job matrix
is already green, but it does not certify Python artifacts/wheels. Windows,
Linux, Pyodide, free-threaded/ABI/platform wheel jobs must run the updated corpus
and integration tests and resolve any failures before releasing. No release or
beta publication occurred; final version metadata waits for those gates.

Temporary directories under the approved temp root: `menudet-release-build`,
`menudet-release-wheel`, `menudet-release-installed`, `menudet-release-snapshot`,
`menudet-clean-sdist`, `menudet-sdist-unpacked`, `menudet-sdist-wheel`,
`menudet-sdist-installed`, `menudet-release-docs-clean`, and
`menudet-release-doctrees-clean`. Reports: `menudet-math-macos.json` and
`menudet-sdist-math-macos.json`.

## Dependency pin updated after green native CI

At the user's request, `CMakeLists.txt` now pins the green native revision
`3418cdce4b5c11e1661c8b6453e3d31d94b0b2b4` from CI run `37612064807`.
Earlier notes saying the Python pin remains at `55c882b` are superseded.
Stock Python wheels have not yet been rebuilt/tested against this updated pin.

## Native CI green (supersedes investigation below)

At the user's request, published the fixes and monitored/repaired CI until green.
Revision `02791f338a853af517b6fedbd38c18d5a2b8da1a` fixed WASM and Windows
builtin recognition, but run `37611459450` exposed a second Windows issue:
identical-code folding merged the floating cast callback with real/conj identity
callbacks, misclassifying their output type. A unique volatile read in the cast
callback preserves its address identity without altering signed zero or NaNs.
The anchor fixture now reports the source and dtypes on a type mismatch.

Final published revision: `3418cdce4b5c11e1661c8b6453e3d31d94b0b2b4`.
CI run [37612064807](https://github.com/Blosc/miniexpr/actions/runs/37612064807)
completed **successfully: all seven jobs green**, including Windows x64/ARM64,
Linux x64, Linux ARM64 interpreter/JIT, macOS, and WASM (including side-module
helper and JIT trace checks). Targeted local portable interpreter test also passed
after the cast-identity repair. Python's dependency pin remains at `55c882b`;
this entry does not claim Python stock-wheel certification of the final fixes.

## Native CI failure investigation (2026-10-07)

CI run `37610312891` on native revision `55c882b` failed on Windows x64,
Windows ARM64, and WASM. Both Windows failures occur while compiling the portable
`hypot(x, 2.0)` fixture: libm address recognition fails. WASM compilation fails
because its fenv lacks `FE_INVALID`; the new test also incorrectly assumed host
rounding/thread facilities.

Local fixes in miniexpr:

- `src/functions.c`: typed builtin recognition also matches the exact builtin
  registration address, handling CRT inline/import aliases such as Windows hypot.
- `tests/test_dsl_portable_interp.c`: WASM retains numeric/literal/environment
  checks under its fixed nearest rounding, without requiring unavailable exception
  flags or pthreads. Native upward-rounding, exception and concurrency checks stay.

Verification: complete native interpreter suite **325 passed**; local Emscripten
interpreter-only suite **48 passed**; CI-style WASM with JIT enabled **50 passed**;
targeted native portable interpreter ASAN/UBSAN **passed**. Existing duplicate
library and warning-enabled unused-function warnings remain; no WASM build warning
was observed. Remote Windows jobs still require revalidation with these fixes;
the reported failed CI run is not claimed green. Published pin is unchanged.

Build directories: `<temp-root>/miniexpr-portable-1-interpreter`,
`<temp-root>/miniexpr-portable-1-wasm-fix`, and
`<temp-root>/miniexpr-portable-1-sanitized`. WASM was configured with `emcmake`,
Node as emulator, SLEEF OFF, then TCC JIT ON/bundled host TCC OFF/trace ON to match
the CI configuration. All commands used `conda run -n blosc2`.

## Published dependency update — latest status

The user authorized committing/pushing miniexpr and committing the Python
implementation plus dependency pin. Native implementation is published on
`Blosc/miniexpr` main at `55c882b238c4daf0523a52a44d5d84db56af24e4`.
Python-Blosc2 `CMakeLists.txt` pins that exact revision. Earlier notes saying the
published pin is unchanged are historical and superseded by this update.

A fresh stock wheel (no local source overrides) built successfully using that
revision. Both `FETCHCONTENT_SOURCE_DIR_MINIEXPR` and
`FETCHCONTENT_SOURCE_DIR_BLOSC2` are empty in its CMake cache, and the fetched
native checkout reports the pinned SHA. The isolated installed wheel passed all
seven portable test modules: **183 passed, 17 external opt-in skips** (200 cases).
The editable development installation was not replaced.

Verification commands used `conda run -n blosc2 python -m pip wheel . --no-deps
--no-build-isolation --wheel-dir <temp-root>/portable-1-published-pin-wheel
--config-settings=build-dir=<temp-root>/portable-1-published-pin-build`, followed
by a separate `pip install --no-deps --target <temp-root>/portable-1-published-pin-smoke`
and pytest with the editable finder excluded only in that subprocess. Wheel SHA256:
`434b4cb6fdb1e5fa11dd088a749a42359ed798242e8de05cef69710d642c9d87`.

This verifies the published-pin macOS ARM64 wheel, not Linux/Windows/WASM
certification or beta readiness. Unrelated user work remains unstaged.

## CURRENT STATUS — authoritative handoff (supersedes all older sections below)

### Unreleased portable 0.1 retired; draft 1.0 is the public default (2026-10-07)

The user's corrected release scope is implemented in both repositories. Portable
0.1 was never released and has no compatibility obligation. The public story is
experimental/draft portable 1.0 versus legacy Python-specific persistence, whose
active reconstruction requires explicit `deserialize='full'`. No automatic
migration or reinterpretation occurs. Ordinary full-DSL semantics are unchanged.

- `DSLKernel.export`, its authoring helper and `validate_portable_dsl` default to
  1.0. Removed the Python 0.1 dtype/call allowlists, scalar encoder and export
  branch. Old native builds report unavailable draft support instead of exposing
  their retired loader; the published native pin remains unchanged.
- Native `dsl_portable.c` now contains only typed draft validation. The artifact
  loader rejects unsupported schema/language versions before interpreting their
  envelope as 1.0, then strictly validates context/signature/capabilities/constants.
  Removed the old source filter, full-profile artifact compilation, tiled evaluator
  state and mask/ND rejection branch. `me_artifact_eval` remains a useful rank-zero
  numeric elementwise call adapter to descriptor evaluation, not format support.
- Portable compilation no longer inherits full-DSL `ME_DSL_FP_MODE` defaults;
  it uses the strict parsed draft default and still rejects explicit non-strict
  source. Ordinary full-DSL environment policy remains intact.
- Removed obsolete freeze/language/artifact draft documents and the 18-entry
  frozen-excluded manifest/category/rejection tests. Retargeted the native corpus
  and runners to interpreter-first 1.0. Separate full-DSL audit cases remain.
  Five corpus reference files now reflect operand-driven float32 arithmetic,
  explicit float64 casts and integral Boolean-negation zero, rather than output
  context typing. Native encoding, schema, ownership, binding and descriptor
  regressions remain and now load 1.0.
- The Python host suite retargets the former frozen boundary to actual arithmetic
  admission/results and checked overflow, preserving all 200 orthogonal cases.
  Mixed captures and NumPy normalization now execute; the old rejected string
  capture case now verifies a typed Unicode snapshot. Optional JIT requests always
  assert interpreter fallback, never pretend `has_jit=True`. External corpus/runner
  tests use explicit environment paths only and now exercise portable artifacts.
- User-facing Python/native API docs and specifications describe draft 1.0 only,
  with a prominent experimental notice and no invented beta/platform claim.
  Implementation plan/scope current decisions record retirement. Historical
  verification below and unrelated user files/logs are retained.

Verification so far, all commands in conda `blosc2`:

- Complete native CTest: **325 passed**; complete ASAN+UBSAN CTest: **325 passed**.
  The count drops from 343 solely because the 18 obsolete frozen rejection gates
  were removed; useful arithmetic/masking/encoding/validation fixtures remain.
- Seven portable Python modules: **183 passed, 17 external skips = 200 cases**.
  Explicit external native corpus/runner integration: **176 passed, zero skips**
  across `test_portable_artifact.py` and `test_dsl_portable.py`.
- Full default Python suite: **11280 passed, 55 skipped**, 36.35s, after the final
  bridge/validator changes and retained Unicode capture regression.
- Native warning-enabled rebuild retains existing unused helpers/aggregate
  initializers/unknown SIMD pragmas and duplicate-library linker warnings. No new
  warning in the retirement validator/artifact/runner changes or sanitizer finding.

Remaining release gates are unchanged: independent function/platform accuracy,
Windows/Linux/WASM and stock-published-pin wheel certification, authorized native
publication/pin update, and clean whole-tree documentation. Local development
override wheel results are not stock-published-pin evidence.

### Final retirement verification and file handoff

- Rebuilt editable extension with explicit local miniexpr and C-Blosc2 overrides.
  After isolated wheel installs, rechecked source import path and active descriptor
  support; the editable install remains in use. Published pin/dependencies unchanged.
- Isolated local-native-override wheel: all seven portable modules **183 passed,
  17 external skips**, 1.80s. Smoke subprocess removed only its own editable finder,
  asserted the imported package was beneath its isolated target and used `-n 0`.
  Wheel SHA256 `16a514a8063601c4a4eb1ce1ebfe1a0b13e1421c8e4cbb1e32126424127c290f`.
  This is not stock-published-pin, platform or published-wheel certification.
- Separate artifact-disabled local-override wheel built/installed only to another
  isolated target. Ordinary array arithmetic and draft numeric/string/ND source
  validation pass; retired source version rejects and artifact loading explicitly
  reports `NotImplementedError`. No installed editable replacement.
- Final native and ASAN+UBSAN suites: **325/325** each, 9.08s / 9.23s. Final
  incremental builds report only the existing duplicate `-lm`/static-library
  linker warning; no sanitizer error. Earlier warning-enabled full rebuild retains
  the pre-existing warnings documented above, not a warning-free whole-tree claim.
- Ruff lint and format checks pass on all four Python files changed by retirement;
  Cython compiled successfully in editable and both wheel builds. Both repositories
  pass `git diff --check`.
- Sphinx exits successfully; neither changed portable/native-syntax Python page has
  a diagnostic. Whole-tree output contains **931 WARNING / 4 ERROR lines**, including
  existing installation list-table, NumPy/reference/image/generated-stub issues.
  This remains an incomplete clean docs gate, not a release acceptance claim.
- Final callsite/version audit over source/tests/docs/examples/build/CI files finds
  no retired portable support/default/freeze links; explicit 0.1 rejection tests
  and numeric literals remain. Moved the prior numeric audit to
  `../miniexpr/plans/numeric-audit-0.1-history.md` as a factual archive, out of public
  specifications. Unrelated untracked user work and user logs are untouched.
- Final additive regression verifies an actual retired envelope (no `context`,
  `core-scalar`, `scalar-per-element`) rejects specifically as unsupported version,
  not as malformed 1.0. The containing diagnostics case reran **1 passed**;
  lint/format and both whitespace checks passed again. No extra test-count grid.

Retirement files (in addition to preserved earlier 1.0 implementation):

- Python: `src/blosc2/dsl_kernel.py`, `portable_kernel.py`, `blosc2_ext.pyx`,
  `dsl_artifact_bridge.h`; `tests/test_portable_artifact.py`, `test_dsl_portable.py`;
  `doc/reference/portable_dsl.rst`, `dsl_syntax.md`; current plan, scope and progress.
- Native: `src/dsl_portable.c`, `dsl_artifact.c`, `dsl_compile.c`, `miniexpr.h`,
  `miniexpr_artifact.h`, `dsl_portable_types.h`, `functions.c`;
  `tests/CMakeLists.txt`, `test_dsl_portable_validation.c`, `test_dsl_artifact.c`;
  both portable runner/README directories, affine JSON and five arithmetic/Boolean
  reference fixtures; draft 1.0 specs and native DSL syntax/usage docs.
  Removed `doc/dsl-spec/{0.1,artifact-0.1,draft-history-0.1}.md` and
  `tests/portable-dsl/frozen-excluded.txt`; preserved historical numeric audit as above.
- Reproducible logs/builds/wheels are under the approved temporary root with
  `retire-` prefix: editable, full/target/external Python, native/sanitized builds
  and tests, wheel build/install/tests, disabled build/install/smoke and docs.

## PRIOR IMPLEMENTATION HANDOFF — historical; superseded by retirement above

Final parent-session check: the seven portable Python modules passed again
(**183 passed, 17 external skips**, 200 cases); whitespace checks passed in both
repositories. Local implementation milestones 1–9 are complete as recorded in
the implementation checklist. Remaining milestone 10 gates are external release
certification and pre-existing whole-tree documentation issues, not a claim that
miniexpr has entered beta or portable 1.0 is frozen.

- Parent scope correction implemented: compliant newly authored DSLKernel-backed
  LazyUDF saves/frames/stores/MessagePack/nested references now automatically
  normalize and export validated native 1.0 recipes. Blanket rejection was too
  strict and is removed. Shared `portable_from_lazyudf` preflight preserves typed
  captures/scalar bindings, parameter-named array refs, output domain and existing
  uniform logical block partitions. Block-scalar return/elementwise shape conflicts
  reject with explicit `PortableKernel.lazy` guidance; chunk boundaries that split
  that uniform block grid reject rather than inventing grouping. Loaded legacy
  recipes are marked and remain nonmigrating, explicitly full-policy-only on load.
- CTable independent-row APIs accept DSLKernel authors directly with explicit
  output dtype and reduction cardinality; authoring validates before registration.
  Old vector-column APIs remain ambiguous about row-domain/grouping and now report
  this specific reason and the usable authored independent-row route. Restored
  add-square/loop/where/object/table roundtrips replace obsolete blanket-rejection
  tests. Unsupported effect/callback and legacy saves preserve destinations.
  Latest scope-correction verification: **11280 Python / 343 native / 343
  ASAN+UBSAN**, isolated development wheel **458 passed, 17 external skips**.
  The compact portable boundary is unchanged at **200 cases**. Details below.
- Public validation closure is now implemented: core `me_validate_portable_dsl`
  explicitly dispatches 1.0; new `me_validate_portable_dsl_ex` carries exact string
  output width, logical ND rank and asserted/inferred cardinality. It uses the
  existing profile parser/compiler with JIT OFF and no input buffers/callbacks,
  execution, Python reconstruction or optional yyjson/artifact dependency.
  Python `validate_portable_dsl(..., language_version='1.0', ndim=...,
  cardinality=...)` routes to that API, preserving frozen 0.1 and reporting native
  status/source locations. Actual local editable extension was rebuilt.
- The finite 1.0 matrix/spec now has normative draft arities/types, checked
  combinatoric domains/width limits, ~ semantics, rounding/ties, NaN/zero rules,
  ordered/empty/coherent reductions, Unicode 15.0.0 case data/exact whitespace
  sets and string index/width rules. Independent analytic/hexadecimal references,
  adjacent float steps, exact fused cancellation, all integer factorial limits,
  canonical/alias math and explicit exceptional-value fixtures supplement—not
  replace—the older same-host libm regression matrix. No universal accuracy
  bound or external platform certification is invented. Implementation checklist
  now distinguishes completed local milestones 1–9 from specific release gates.
- Independent correctness review found and fixed two boundary bugs (finished-loop
  lane iterator clobbering and NumPy signed-minimum capture conversion warnings).
  Post-review suites still pass: **11280 Python / 343 native / 343 ASAN+UBSAN**;
  portable boundary remains **200 cases (183 passed, 17 external skips)**.
  Both surfaced contract concerns are now **resolved**: typed structural validation
  rejects divergent block-scalar returns as ambiguous output, preserving coherent
  scalar control flow; MessagePack defaults to safe with explicit recursive caller
  policy propagation. Post-fix isolated native-override wheel also passes the
  entire 200-case portable boundary. See contract-follow-up addendum below.
- Installed conda `blosc2` editable extension uses the explicit local miniexpr
  development override. Published native pin is unchanged. Local 1.0 tests run,
  not skip. No commits, publishing or external platform certification performed.
- Native typed interpreter, draft schema/signature/constants, fixed strings,
  ND descriptors, masked reductions, logical lazy partitions, partial reads,
  streaming output/rechunking and safe portable persistence are connected.
- Added explicit independent-row CTable native computed/generated APIs:
  `add_portable_computed_column` and `add_portable_generated_column`, requiring
  `row_domain='independent'`. Each row is an original one-lane group; ND shape
  (1,), origin (0,) are row-local, not global table indices. Append, refresh,
  deletion and compacted save do not change these semantics. Fixed-shape ndarray
  row inputs now also support scalar native reductions, with scalar/array input
  broadcasting inside the complete row group, persisted immutable row shapes and
  zero row-local ND origin. Table records carry artifact JSON, bindings and
  row-domain metadata and load natively, never Python functions. Multi-row table
  partitions/vector-valued output modes are explicitly unsupported; the native
  standalone/lazy APIs still support the full logical-partition contract.
- Added explicit `DSLKernel.export(..., version='1.0', cardinality=..., ndim=...)`.
  Reuses existing static row/string/NumPy call normalization and capture token
  parameterization; typed signed/unsigned/float/byte/Unicode snapshots. Native
  load validates normalized source and output widths/cardinality before export
  succeeds. Default export remains frozen 0.1; no migration on import.
- Full default Python suite after resolving persistence regressions and a real
  carrierless RemoteArray policy bug: **11280 passed, 55 skipped**. The policy
  helper now handles objects whose `schunk` is None without losing effective
  recursive policy. Previously failing old new-save expectations now verify
   rejection of noncompliant kernels before destination writes; compliant authored
   save roundtrips were restored by the parent scope correction above. Old
   explicit-full load fixtures remain.
- Local native-development wheel built and isolated import/native reduction smoke
  passed, explicitly **not a stock published-pin wheel result**. Artifact-disabled
  development wheel built and isolated ordinary arithmetic/missing-loader smoke
  passed. Editable install was not replaced by either isolated wheel install.
- Added finite native float32/float64 operation result-type/domain fixtures for
  26 numeric libm operations and six exceptional domains. These compare host
  libm within explicit 8-epsilon relative bounds; independent cross-platform ULP
  certification is still an external release-matrix gate, not claimed complete.

Current changed inventory (including preserved earlier edits): native sources/tests
and draft specs listed in historical inventory below; Python bridge
`blosc2_ext.pyx`, `dsl_artifact_bridge.h`, `portable_kernel.py`, `portable_lazy.py`,
`dsl_kernel.py`, `ctable.py`, `deserialization.py`, `b2objects.py`, `lazyexpr.py`,
`schunk.py`, `core.py`, `ndarray.py`, `msgpack_utils.py`, `embed_store.py`,
`dict_store.py`, `tree_store.py`; `doc/reference/portable_dsl.rst`; portable tests,
legacy policy/persistence tests and affected table/ndarray/objectarray tests.
Most recent native closure: `src/dsl_portable.c`, `src/miniexpr.h`, and the
public-validation/independent-reference test functions. Keep unrelated user
files and all existing uncommitted changes.

Table-copy/materialization and native signature/shape/dependency revalidation on
save and load are now implemented. Executed fixtures include static row/NumPy
normalization, typed numeric/Unicode snapshots, scalar/ND constructors, scalar
and vector-row append/extend/refresh, fixed-string row outputs, compact-copy,
deletion, materialization, disk/frame round trips, no Python reconstruction, and
malformed-artifact rejection preserving an existing destination. The portable
Python boundary remains **200 cases: 183 passed, 17 opt-in external skips**.
Native default suite **343/343 passed** and complete ASAN/UBSAN suite
**343/343 passed**; the two targeted artifact/interpreter sanitizer tests also
passed separately. Added descriptor initializers produce no new test warnings;
the full sanitizer build exposes older aggregate-initializer warnings in existing
nonportable test files. No sanitizer error was reported.

Exact remaining gates: independent function-specific/platform accuracy matrix,
Windows/Linux/WASM and stock-published-pin wheel CI, publication/native pin
authorization, and a clean whole-tree docs gate (existing installation list-table
parsing errors, missing tutorial images, duplicate/generated documentation and
unrelated reference warnings remain). Development docs build succeeds and the
portable page has no reported diagnostics. Local draft integration is tested;
do not claim all beta/platform acceptance criteria complete. Final post-edit
Python/lint/docs/wheel results are recorded in the addendum below.

### Exact implementation file inventory (preserved earlier work included)

Python tracked changes:
`doc/reference/portable_dsl.rst`, `src/blosc2/b2objects.py`,
`src/blosc2/blosc2_ext.pyx`, `src/blosc2/core.py`, `src/blosc2/ctable.py`,
`src/blosc2/deserialization.py`, `src/blosc2/dict_store.py`,
`src/blosc2/dsl_artifact_bridge.h`, `src/blosc2/dsl_kernel.py`,
`src/blosc2/embed_store.py`, `src/blosc2/lazyexpr.py`,
`src/blosc2/msgpack_utils.py`, `src/blosc2/ndarray.py`,
`src/blosc2/portable_kernel.py`, `src/blosc2/schunk.py`,
`src/blosc2/tree_store.py`, `tests/ctable/test_ctable_computed_cols.py`,
`tests/ctable/test_ctable_dsl_columns.py`, `tests/ndarray/test_dsl_kernels.py`,
`tests/ndarray/test_lazyudf.py`, `tests/test_b2objects.py`,
`tests/test_deserialization.py`, `tests/test_dsl_portable.py`,
`tests/test_objectarray.py`.
New implementation files:
`src/blosc2/portable_lazy.py`, `tests/test_portable_descriptor.py`,
`tests/test_portable_descriptor_host.py`, `tests/test_portable_lazy.py`,
`tests/test_portable_policy.py`, `tests/test_portable_table_export.py`.
Durable task files: this progress document, the approved implementation plan and
scope inventory. Generated extension/build/wheel/log products are not source
implementation files; unrelated user untracked files remain untouched.

Miniexpr tracked changes:
`CMakeLists.txt`, `src/miniexpr.h`, `src/dsl_portable.c`,
`src/dsl_artifact.c`, `src/dsl_compare.c`,
`src/dsl_compile.c`, `src/dsl_compile_internal.h`,
`src/dsl_compile_support.c`, `src/dsl_eval.c`, `src/dsl_eval_internal.h`,
`src/dsl_jit_runtime_host.c`, `src/dsl_jit_runtime_internal.h`,
`src/dsl_jit_runtime_nonhost.c`, `src/dsl_parser.c`, `src/dsl_parser.h`,
`src/functions.c`, `src/functions.h`, `src/miniexpr.c`,
`src/miniexpr_artifact.h`, `src/miniexpr_internal.h`,
`tests/test_dsl_artifact.c`, `tests/test_dsl_portable_validation.c`.
New miniexpr implementation files:
`doc/dsl-spec/1.0.md`, `doc/dsl-spec/artifact-1.0.md`,
`doc/dsl-spec/features-1.0.json`, `src/dsl_portable_expr.c`,
`src/dsl_portable_expr.h`, `src/dsl_portable_fp.h`,
`src/dsl_portable_types.c`, `src/dsl_portable_types.h`,
`src/dsl_semantic_profile.h`, `tests/test_dsl_portable_checked.c`,
`tests/test_dsl_portable_interp.c`, `tests/test_dsl_portable_types.c`.

### Authored-save scope correction — latest verification (authoritative)

This section supersedes all verification records below. No native implementation
changes were needed for authored-save conversion; the existing native 1.0 parser,
compiler and artifact validation remain authoritative. All checks used conda
`blosc2` and explicit development overrides where building wheels.

- Full Python default suite: **11280 passed, 55 skipped**, 32.62s. Restored native
  roundtrips for add-square, static loops/where, slices, named bindings, DictStore
  operands, ObjectArray, captures/scalar constants, frames and nested arithmetic.
  Complete logical-block reductions plus ND coordinates and sliced reads are
  checked against independent expected values, retaining original partitions.
- Seven portable modules: **183 passed, 17 external skips = 200 cases**; tests
  extend existing orthogonal cases rather than introducing another parameter grid.
  Unsupported print/callback stores preserve old entries on Embed/Dict/Tree routes;
  unsupported disk saves and loaded legacy recipe saves preserve destinations.
- Complete native suite: **343 passed**, 8.34s. Complete ASAN/UBSAN suite:
  **343 passed**, 8.49s, `ASAN_OPTIONS=symbolize=0`. No sanitizer failures.
- Development-override wheel built and installed only to an isolated target;
  isolated subprocess asserted package path and removed only its editable finder.
  Portable modules plus b2objects/ObjectArray/ndarray DSL/CTable DSL/computed-column
  modules: **458 passed, 17 external skips, 1 network deselected**, 4.22s.
  Wheel `blosc2-4.14.2.dev0-cp311-abi3-macosx_26_0_arm64.whl`, SHA256
  `a4255253709584e7a1ccb69d585dc9ed6cc8a80d100b59a7bed0f6234f31e1c9`.
  This is not stock-pin/platform certification. Editable import path and descriptor
  availability were subsequently rechecked and preserved.
- Ruff check/format passed for the 10 Python files touched by this correction;
  repository whitespace check passed. Sphinx build succeeds with **391 whole-tree
  warnings**, including missing generated API stubs; the portable page itself has
  no reported diagnostic. The clean whole-tree docs gate remains incomplete.
- Authoring route: `portable_from_lazyudf` in `src/blosc2/portable_lazy.py`, shared
  through `b2objects.encode_b2object_payload` and LazyUDF save/frame. CTable APIs
  accept DSLKernel plus `dtype`, explicit `row_domain='independent'` and optional
  `cardinality='block_scalar'`; the approved plan and portable docs record this.
  No legacy source fallback/migration or changed elementwise-to-scalar shape.
- Logs under the approved temporary root: `author-save-python.log`,
  `author-save-portable.log`, `author-save-native.log`, `author-save-sanitized.log`,
  `author-save-wheel-build.log`, `author-save-wheel-tests.log`, `author-save-docs.log`.
  Wheel and isolated target directories: `author-save-wheel`, `author-save-wheel-smoke`.
- Local corrected scope is fulfilled. External function-specific accuracy,
  Windows/Linux/WASM, authorized publication/pin and stock-wheel certification,
  and clean whole-tree docs remain pending. No commits/push/publication occurred.

### Public validator and finite-reference closure — earlier verification (superseded)

These results supersede the earlier verification records below. All Python,
build and test commands used conda `blosc2`; native overrides were explicit.

- Rebuilt the installed editable extension with the documented local miniexpr
  and C-Blosc2 overrides. Rechecked `blosc2.__file__` resolves to this repository's
  `src/blosc2/`, descriptor support is available, and 1.0 Unicode source validation
  succeeds after both isolated wheel installations. Published pin unchanged.
- Full default Python suite: **11280 passed, 55 skipped**, 42.87s.
- Final complete native CTest: **343 passed**, 8.73s. Final complete ASAN/UBSAN
  CTest: **343 passed**, 8.86s, `ASAN_OPTIONS=symbolize=0`. These include the
  extended validator and independent analytic/exact reference fixtures, including
  the final floating-power NaN exceptions. No sanitizer errors or new warnings
  in the modified validator/interpreter tests were reported.
- Isolated development-override wheel: **183 passed, 17 external skips** across
  the seven portable modules = **200 cases**. No new parametrized cases or local
  1.0 skips; validation assertions extend the existing authoring test. Wheel:
  `blosc2-4.14.2.dev0-cp311-abi3-macosx_26_0_arm64.whl`, SHA256
  `39dc5fcd00867991a06dc14426829a305bccc3fa581224e5bb8dfc71b0e096bf`.
  Isolated subprocess removed only its own editable finder, asserted wheel import
  path, and ran pytest with `-n 0`. This is **not stock-pin certification**.
- Separately built/installed isolated artifact-disabled wheel: core frozen 0.1
  and draft 1.0 numeric/string/ND source validation passed, missing ND context
  rejected, ordinary arithmetic passed, optional artifact import explicitly
  rejected. Plain source validation does not require the optional JSON loader.
- Ruff check and format pass for all **27 changed/new Python files**; both
  repositories' `git diff --check` pass. Finite feature JSON parses successfully.
- Sphinx HTML build succeeds with **142 whole-tree warnings**; no diagnostic
  against `doc/reference/portable_dsl.rst` or the updated validator docstring.
  Existing generated/reference/installation/image diagnostics are not a clean
  documentation gate and remain recorded as outstanding.
- Logs under `/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/`:
  `closure-editable-build.log`, `closure-python.log`, `closure-native-build.log`,
  `closure-native-ctest.log`, `closure-sanitized-build.log`,
  `closure-sanitized-ctest.log`, `closure-wheel-build.log`,
  `closure-wheel-tests.log`, `closure-disabled-build.log`, `closure-docs.log`.
  Isolated wheel directories are `closure-wheel`, `closure-wheel-smoke`,
  `closure-disabled-wheel`, and `closure-disabled-smoke`.
- Local requested implementation closure is complete. Remaining release gates:
  independent function-specific/platform accuracy acceptance, Windows/Linux/WASM
  execution, authorized native publication/pin and stock-pin wheel matrix, and
  clean whole-tree docs. No beta-ready claim, publication, dependency change or
  destructive repository operation was performed.

### Earlier local verification addendum (superseded by closure results above)

- `conda run -n blosc2 pytest -q`: **11280 passed, 55 skipped**, 38.69s,
  after final row-shape/broadcast/width fixes. Subsequent test-helper/comment
  cleanup reran ndarray DSL and CTable DSL tests: **138 passed**, 1.88s.
- Seven compact portable test modules: **183 passed, 17 skipped** = **200**;
  the 17 skips are 16 external frozen-0.1 corpus cases and one external artifact
  runner. No local draft 1.0 test skips. Frozen 0.1 self-contained artifact tests
  continue to execute, with default export still 0.1.
- Complete native interpreter CTest: **343 passed**. Complete ASAN/UBSAN CTest:
  **343 passed**, 43.90s, `ASAN_OPTIONS=symbolize=0`. Targeted artifact and
  portable interpreter sanitizer pair: **2 passed**, 0.87s. No test failure or
  sanitizer error; the existing nonportable aggregate initializer warnings noted
  above do not indicate a new implementation warning.
- Ruff lint and format: all **27 changed/new Python files passed**. Cython is
  validated by actual wheel compilation, not Ruff's Python parser. Both repos'
  `git diff --check` passed.
- Final native-development override wheel:
  `blosc2-4.14.2.dev0-cp311-abi3-macosx_26_0_arm64.whl`, SHA256
  `1d88d169e4a2af3df0747bdd5002deca019fab77aeac9b32aa3d40a1bb0f05a9`.
  Built with explicit local miniexpr/C-Blosc2 source overrides and artifacts ON.
  Installed into a separate temporary target; removed the editable import finder
  only inside the smoke subprocess and asserted the loaded wheel package path.
  The **entire 200-case portable suite** ran against that isolated wheel:
  **183 passed, 17 external opt-in skipped**, 1.08s, including typed authoring,
  table vector/string/constructor persistence and lazy descriptor tests.
  This is **not** a published-pin/stock-wheel platform certification result.
- Separate artifact-disabled development wheel built and isolated smoke passed:
  ordinary array arithmetic works, availability query is false, portable loader
  explicitly raises `NotImplementedError`. Editable development install remained
  untouched and was rechecked for the active descriptor ABI afterwards.
- Final incremental Sphinx HTML build succeeded with **117 whole-tree warnings**;
  no diagnostic names `doc/reference/portable_dsl.rst` after correcting its new
  heading underline. Earlier clean full build exposed existing installation
  list-table parsing errors and unrelated missing-image/reference warnings.
  A successful non-`-W` build is not a clean whole-tree documentation gate.

Logs/builds/wheels remain under the approved temporary root
`/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/`:
`portable-1-full-python-final.log`, `portable-1-comments-regression.log`,
`portable-1-native-ctest.log`, `portable-1-sanitized-all-ctest.log`,
`portable-1-sanitized-all-build.log`, `portable-1-docs-final.log`,
`portable-1-final-wheel/`, `portable-1-final-wheel-smoke/`,
`portable-1-disabled-wheel/`, `portable-1-disabled-smoke/`.
The supported local draft integration is implemented and tested. External
platform/independent-accuracy/stock-pin gates remain unexecuted, and no dependency
pin, publishing or repository state-changing git action was performed.

### Independent correctness review addendum (2026-10-07)

Scope: independently read repository instructions, approved scope/plan and current
handoff; inspected profile-aware expression typing/conversions, reduction masks,
scalar returns, constant decoding, artifact width/capability/cardinality and
descriptor checks, Python typed capture/export, lazy original-group reads and
persistence/policy propagation. This is focused local review, **not complete
assurance**, a whole-JIT audit or external platform certification.

Concrete fixes:

- `../miniexpr/src/dsl_eval.c`: portable `for` evaluation no longer zeroes the
  iterator local in inactive lanes while another lane continues. Before the fix,
  `for i in range(x): ...; return i` on `[2, 4]` returned `[0, 3]`, rather than
  `[1, 3]`. Preserve full-DSL behavior outside the versioned portable profile.
  Native regressions cover exhausted ranges, lane-specific break, and runtime
  rejection of undefined iterators for empty ranges.
- `src/blosc2/portable_kernel.py`: determine integral capture magnitude with a
  Python integer rather than `abs()` in the NumPy fixed-width dtype. Capturing
  `np.int64(INT64_MIN)` as float64 formerly raised an overflow RuntimeWarning
  (an exception under the suite's warning policy). Captured conversion and
  immutable snapshot now have an authoring regression.
- Merged host boundary checks into existing portable cases: masked scalar return
  through a reduction local with the first lane invalid, all-invalid sum identity,
  tampered scalar cardinality/string output width rejection, and reverse/empty
  elementwise partial reads of `x + sum(x)` preserving exactly the original group.
  No test counts expanded, tests suppressed or failures relabeled.

Findings from the initial review — resolved by the contract follow-up below:

1. **Divergent block-scalar returns:** native import accepts
   `s = sum(x); if x < 0: return s + 1; return s` as `block_scalar` (normalized
   multiline DSL). Input `[-1, 2]` returns `1`: the first subset wrote `2`, then
   the remaining subset overwrote the one scalar output with `1`.
   `dsl_compile.c` recognizes the reduced local as uniform without accounting for
   varying return control; `dsl_eval.c` writes each scalar return to the same slot.
   The follow-up now rejects divergent scalar returns structurally; it neither
   invents a selection rule nor reinterprets cardinality.
2. **Implicit legacy permission at the low-level MessagePack boundary:**
   `msgpack_unpackb(payload)` in `msgpack_utils.py` deliberately defaults to FULL.
   A handcrafted extension-43 legacy `lazyudf` record reconstructed a `LazyUDF`
   without passing `deserialize='full'`. Safe-mode paths retain propagation and
   rejection; this is not a demonstrated safe-mode bypass. The helper's documented
   compatibility exception conflicted with the plan's explicit-opt-in wording.
   The follow-up now defaults this boundary to safe and propagates explicit-full
   permission through nested values, without a legacy/default-full fallback.

Exact post-fix verification (conda `blosc2`, local native development override):

- Rebuilt native interpreter, sanitizer build and Python editable extension using
  the existing local miniexpr/C-Blosc2 FetchContent overrides; published pin unchanged.
- Full Python `pytest -q`: **11280 passed, 55 skipped**, **31.17s**, after final
  boundary-test edits. Seven portable modules: **183 passed, 17 external skipped**,
  **1.79s**, still exactly **200 cases**.
- Full native CTest: **343/343 passed**, **8.75s**. Full sanitizer CTest with
  `ASAN_OPTIONS=symbolize=0`: **343/343 passed**, **8.91s**. No sanitizer error;
  incremental review builds reported no warnings. A new exploratory test initially
  assumed preassigned loop iterators were admissible; native compilation rejects
  this existing syntax restriction. Replaced that assumption with the actual
  supported empty-range/undefined-local boundary, not a change in language policy.
- Ruff lint and format checks pass for all four Python files changed by review;
  both repos' `git diff --check` passes. No post-review wheel/docs rebuild claimed.
- Logs: approved temporary root, `portable-1-review-{native-build,native-ctest,
  sanitized-build,sanitized-ctest,editable,full-python,portable-python}.log`.

External beta gates remain independent function/platform accuracy certification,
Linux/Windows/WASM and stock-published-pin wheel CI, publication/pin authorization,
and clean whole-tree docs. The semantic/policy findings above were additional
local acceptance concerns and have now been resolved, as recorded below.
No commits, pushes, pin changes or publishing performed; unrelated files preserved.

### Contract follow-up — both independent-review concerns resolved (2026-10-07)

- `../miniexpr/src/dsl_compile.c` validates block-scalar return coherence after
  static type/provenance joins converge. Reject returns under typed lane-varying
  branches or loop bounds/conditions, including scalar returns in loops whose
  lane-varying break/continue can split the return mask. Diagnostic:
  `ambiguous block-scalar output: return under lane-varying control or loop flow`.
  No first/last scalar selection and no output-cardinality reinterpretation.
  Uniform reduced-scalar conditions/coherent scalar loops still compile; varying
  loops may rejoin before a single top-level scalar return. Nested loop flow
  rejoins locally rather than tainting unrelated outer returns.
- This intentionally conservative check also rejects matching scalar return
  expressions inside a varying branch/loop: equality alone does not establish
  coherent group control flow. The exact excluded patterns are documented in
  `../miniexpr/doc/dsl-spec/1.0.md` and `doc/reference/portable_dsl.rst`.
  Native and merged Python cases exercise divergent reduced-local returns,
  varying range/while/elif, lane-varying break/continue, and accepted reduction-
  controlled/coherent-loop and post-rejoin scalar returns.
- `src/blosc2/msgpack_utils.py`: public and private decoding boundaries default
  safe. Explicit-full extension decoding passes its effective policy recursively
  into frames, structured references, NumPy object values and sets. Audited all
  source callsites: SChunk VLMeta/getall, EmbedStore metadata, ObjectArray and
  BatchArray already pass their owning carrier's policy; proxy reads explicitly
  pass safe. No internal call inherits the newly changed public default as a
  substitute for caller policy. Locally constructed/full containers retain
  their existing effective policy; safe-opened containers remain safe.
- Merged legacy MessagePack checks into the existing portable policy case:
  default/safe handcrafted extension-43 legacy payloads, both direct and nested
  in extension-46 object arrays, reject with a monkeypatched source-reconstruction
  failure hook never called. Explicit-full decoding reconstructs those same
  values intentionally. Existing local/full metadata and container tests pass
  without adding default-full fallback or weakening any safe assertion.

Final exact local verification:

- `conda run -n blosc2 pytest -q`: **11280 passed, 55 skipped**, **34.60s**, after
  final boundary checks. Seven portable modules: **183 passed, 17 external
  skipped**, **3.10s**, exactly **200 cases**. Targeted descriptor/policy pair:
  **10 passed**, **2.15s**.
- Rebuilt interpreter and sanitizer native builds. Complete CTest:
  **343/343 passed**, **45.29s**; complete ASAN/UBSAN CTest with
  `ASAN_OPTIONS=symbolize=0`: **343/343 passed**, **45.59s**. No sanitizer error or
  new compiler warning; both incremental links reported the existing duplicate
  `-lm`/`libminiexpr.a` warning.
- Rebuilt editable install with explicit local miniexpr/C-Blosc2 overrides;
  rechecked installed source import and descriptor ABI after isolated wheel work.
- Built native-override wheel
  `blosc2-4.14.2.dev0-cp311-abi3-macosx_26_0_arm64.whl`, SHA256
  `a0874703e46863b67040fbc096aa007c734e56f5f1b26da1fb903caa670cf649`.
  Installed to a separate temporary target; removed the editable finder only in
  the smoke subprocess and asserted the package path. Complete portable suite
  against the isolated wheel: **183 passed, 17 external skipped**, **1.52s**,
  using `-n 0` so worker startup cannot reintroduce editable imports. This is
  local-development override evidence, **not stock-published-pin certification**.
- Ruff lint/format pass for all six Python implementation/test files changed by
  independent review/follow-up; both repos' `git diff --check` passes.
- Sphinx command exits successfully; portable page has no diagnostic. Whole-tree
  output includes **818 WARNING / 4 ERROR diagnostic lines**, including existing
  installation list-table errors and unrelated reference/image issues. This is
  not a clean docs gate, and no unrelated docs cleanup was attempted.
- Approved temporary-root logs use `portable-1-contract-` prefix: native/sanitized
  build and CTest, editable, full/portable Python, wheel build/install/test and
  docs. Wheel and target directories: `portable-1-contract-wheel/` and
  `portable-1-contract-wheel-smoke/`.

No remaining local semantic question from these two findings. External beta
gates remain independent function/platform accuracy certification, Linux/Windows/
WASM and stock-published-pin wheel CI, authorized publication/native pin change,
and clean whole-tree docs. No backend expansion, commits, pushes or publication.

## HISTORICAL RECORDS — superseded status/next-actions, retained verification history

All inventory, pending-work and skip statements below describe earlier sessions.
They are not current handoff instructions. Use CURRENT STATUS above and its final
verification addendum for resume decisions.

## Target and approved decisions

Implement `portable-DSL-1.0-implemenation.md` through beta readiness. All Q1–Q3
are approved: standard signed/unsigned integer widths and operand promotion;
explicit block-scalar cardinality/grid output; explicit `deserialize="full"`
legacy opt-in, without migration tooling. Interpreter correctness comes first.
No final contract freeze or publication is implied by completing implementation.

## Starting state

- Python-Blosc2: `23a2a91a`; no tracked modifications on entry.
- Miniexpr: `98b2bf9`; no tracked modifications on entry.
- Numerous unrelated untracked files exist in Python-Blosc2; preserve them.
- Native repository instructions: `../miniexpr/AGENTS.md` (K&R C style, minimal
  changes, relevant tests, no new dependencies).
- Python commands must run through `conda run -n blosc2`.

## Verification log

- Baseline: `conda run -n blosc2 pytest tests/test_portable_artifact.py
  tests/test_dsl_portable.py -q`: **159 passed, 41 skipped**. Skips include opt-in
  external corpus/runner tests; this does not certify portable 1.0.
- Native build directory:
  `/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/miniexpr-portable-1-interpreter`.
  Configured with `conda run -n blosc2 cmake -S /Users/faltet/blosc/miniexpr -B
  <build-dir> -DMINIEXPR_ENABLE_TCC_JIT=OFF -DMINIEXPR_USE_SLEEF=OFF
  -DMINIEXPR_USE_ACCELERATE=OFF -DMINIEXPR_BUILD_SHARED=OFF
  -DMINIEXPR_BUILD_EXAMPLES=OFF -DMINIEXPR_BUILD_BENCH=OFF
  -DMINIEXPR_BUILD_ARTIFACT=ON`; built with `--parallel 8`.
- First native suite with promotion helper: **341/341 passed**. Latest suite
  adds checked-arithmetic tests: `conda run -n blosc2 ctest --test-dir <build-dir>
  --parallel 8 --quiet --output-log <temp-dir>/portable-1-native-ctest.log`
  exited successfully. Current suite has 342 tests.
- Checked arithmetic compiled independently with `conda run -n blosc2 cc
  -std=c99 -Wall -Wextra -Werror -fsanitize=address,undefined
  -fno-sanitize-recover=all -I /Users/faltet/blosc/miniexpr/src
  /Users/faltet/blosc/miniexpr/src/dsl_portable_types.c
  /Users/faltet/blosc/miniexpr/tests/test_dsl_portable_checked.c
  -o <temp-dir>/portable-1-checked-sanitized`; executing it **passed**.
  No compiler/sanitizer warnings. Full CMake build emits its existing duplicate
  static-library/`-lm` linker warning; no new compiler warning was observed.
- Draft matrix audit: all **87** distinct registered `functions.c` builtin names
  are covered, plus four DSL intrinsics (`int`, `float`, `bool`, `range`); no
  duplicate matrix names. This is coverage of the inventory, not certification.
- `git diff --check` passes in both repositories.

## Milestones

1. In progress: implemented draft schema and finite builtin inventory; concrete
   platform/domain/tolerance certification remains (not a schema design blocker).
2. Integrated: profile-aware typed interpreter and draft artifact 1.0 acceptance
   for validated implemented capabilities; standalone 0.1 validator stays frozen.
3. Integrated: numeric operations/casts/comparisons/lazy operands and static local
   joins. Remaining work is the finite conformance/platform matrix.
4. Integrated: ordered masked reductions, padded lanes, scalar cardinality,
   empty standalone blocks and evaluation-local caches/definition masks; expand
   returned-lane/control-flow/concurrency certification fixtures.
5. Integrated: extended artifact descriptor ABI, typed signatures/constants and
   immutable width/cardinality/capability/context validation.
6. Integrated: fixed strings, dynamic integral indices/bounded replacement, and
   logical ND coordinates; platform/endian/Unicode conformance remains.
7. In progress: conditional Python bridge and standalone descriptor routing done;
   logical lazy scheduling, partial reads and compiled Python 1.0 tests remain.
8. Pending: all persistence routes and legacy policy enforcement.
9. Integrated: interpreter-only optional-JIT fallback and explicit required-JIT
   unsupported diagnostic; frozen/full profiles preserve their original behavior.
10. Pending: complete conformance/platform/wheel/docs verification.

## Findings to retain

- `functions.c` registers `log` differently depending on `ME_NAT_LOG`; the
  portable profile must bind natural log explicitly, independent of build flags.
- Existing artifact eval assumes elementwise `nitems` output and skips empty
  calls. Extended descriptors are required for block-scalar and empty identities.
- Existing `b2objects.py` internally hard-codes `deserialize="full"` on operand
  opens. Effective user policy must propagate instead.
- Full-DSL compilation uses shared expression helpers; approved operand typing
  must be profile-aware, not an unversioned global change.

## Next action

Continue with static local joins, remaining numeric functions and mask conformance,
then extended artifact descriptors/cardinality/context.
Do not enable 1.0 acceptance until the typed-tree/evaluator actually enforces the
contract. Current public runtime APIs still accept only 0.1; staged semantics are
isolated from frozen/full-DSL execution. No semantic blocker identified yet.

## Continuation increment: typed interpreter integration

- Added internal `dsl_compile_program_profile()` and `private_compile_profile_ex()`;
  the original entry points retain full-DSL behavior. Profile metadata is per
  program/parser state, not global. The 1.0 raw-tree path bypasses full-DSL numeric
  promotion, optimization, bytecode and output-context node retyping.
- Parser leaves preserve exact decimal integer magnitudes (through uint64), weak
  literal identity and direct float32 literal rounding. Unary literal signs are
  resolved without executing arithmetic. Typed pass inserts checked conversion
  nodes, including final output conversion only after operand-driven computation.
- Added `dsl_portable_expr.[ch]` operating on the existing `me_expr` tree and
  checked helpers. Core arithmetic, shifts/bitwise, powers, exact mixed integer
  comparisons, int/float/bool casts and lazy truth/where now execute natively.
  Unsupported operations still reject in this **internal staged** path.
- DSL assignment/return and condition masks route through the checked evaluator;
  skipped lanes do not execute invalid arithmetic. New locals infer independently
  of output dtype. Existing locals still use their first inferred type; approved
  static join inference remains to implement.
- Optional JIT requests are forced to interpreter before IR construction in 1.0.
  Print is rejected. No public 1.0 dispatch or persistence changed yet.
- Reconfigured/rebuilt the native build above; `tests/test_dsl_portable_interp`
  passed. `conda run -n blosc2 ctest --test-dir <build-dir> --parallel 8 --quiet
  --output-log <temp-dir>/portable-1-native-ctest.log` passed (343 tests).
- Additional changed native files: `functions.[ch]`, `miniexpr.c`,
  `dsl_compile.c`, `dsl_compile_internal.h`, `dsl_jit_runtime_internal.h`,
  `dsl_eval.c`, `dsl_semantic_profile.h`, `dsl_portable_expr.[ch]`,
  `tests/test_dsl_portable_interp.c`, and the CMake source list.
- This initial increment's tests were subsequently expanded and explicitly made
  assertion-enabled even in Release builds; use the latest verification below.

## Continuation increment: ordered masked reductions and numeric coverage

- Profile-aware DSL parsing retains `//=` instead of lowering through floating
  `floor(a / b)`. Expression tokenization and comparison lowering recognize `//`;
  full-DSL tokenization/compound-assignment behavior stays unchanged. Checked
  integer floor division/remainder, float floor division/divisor-sign remainder,
  and distinct truncation-sign builtin `fmod` now execute through typed nodes.
- Added explicit builtin identity dispatch for the staged unary libm family,
  aliases, selected binary libm functions, abs/square/sign and real-only
  conj/real/imag. Calls use float32 libm entry points for float32 computation and
  float64 otherwise; no external callback is executed. Natural-log binding is
  independent of `ME_NAT_LOG`. `rint` ties-to-even does not use host rounding mode;
  `round` keeps native half-away ties. Scalar fmin/fmax zero ties are explicit.
- sum/prod/mean/min/max/any/all typed nodes now reduce current participating lanes
  in logical order, with checked integer accumulation and float32 operation
  boundaries. Empty identities/errors, numeric NaN propagation and min/max zero
  ties are implemented. A per-expression evaluation-local reduction cache
  broadcasts without quadratic repeated reduction work for normal expressions;
  it is never stored on a compiled/shared handle.
- Expanded native integration coverage: exact int64/uint64 literals and
  comparisons, contextual literal errors, arithmetic/shift/power extrema,
  operand-vs-output widths, narrowing, NaN truth/comparisons, libm typing,
  conditional invalid arithmetic, nested masked reductions, loop early exit,
  returned lanes, continue, empty identities and ordered cancellation/overflow.
- Added profile guards to `dsl_jit_runtime_host.c`,
  `dsl_jit_runtime_nonhost.c`, and both execution paths in `dsl_eval.c` so no
  uncertified cached/backend function can execute for a 1.0 program.
- Staged compilation rejects external callbacks, print, non-strict FP, ND symbols
  without the unfinished explicit descriptor, unsupported calls/strings, nested
  reductions and reduction-valued short-circuit/where alternatives needing
  additional expression masks. These are **temporary implementation diagnostics**,
  not approved scope removals. Local dtype-changing reassignment rejects clearly
  rather than silently narrowing to the first assignment. Static joins remain
  required. Inconsistent return cardinality now rejects.
- Additional changed native files: `dsl_parser.[ch]`, `dsl_compare.c`,
  `dsl_jit_runtime_host.c`, `dsl_jit_runtime_nonhost.c`, and updated draft `1.0.md`.

### Latest reproducible verification

Let `T=/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode`.
All commands ran through the required `blosc2` environment.

- `conda run -n blosc2 cmake --build $T/miniexpr-portable-1-interpreter --parallel 8`
  passed; log `$T/portable-1-build.log`. Normal build has only the previously
  observed duplicate-library linker warning.
- `conda run -n blosc2 ctest --test-dir $T/miniexpr-portable-1-interpreter
  --parallel 8 --quiet --output-log $T/portable-1-native-ctest.log`: **343 passed**.
- Integrated sanitizer build configured with:
  `conda run -n blosc2 cmake -S /Users/faltet/blosc/miniexpr
  -B $T/miniexpr-portable-1-sanitized -DMINIEXPR_ENABLE_TCC_JIT=OFF
  -DMINIEXPR_USE_SLEEF=OFF -DMINIEXPR_USE_ACCELERATE=OFF
  -DMINIEXPR_BUILD_SHARED=OFF -DMINIEXPR_BUILD_EXAMPLES=OFF
  -DMINIEXPR_BUILD_BENCH=OFF -DMINIEXPR_BUILD_ARTIFACT=ON
  -DCMAKE_BUILD_TYPE=Debug
  '-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -Wall -Wextra'`.
  `conda run -n blosc2 cmake --build $T/miniexpr-portable-1-sanitized
  --target test_dsl_portable_interp --parallel 8` and
  `conda run -n blosc2 $T/miniexpr-portable-1-sanitized/tests/test_dsl_portable_interp`
  **passed**, with no sanitizer findings. The warning-enabled shared-source build
  emits numerous existing unused-function/initializer warnings; no warnings came
  from the new expression module. Log `$T/portable-1-sanitized-build.log`.
- Independently: `conda run -n blosc2 cc -std=c99 -Wall -Wextra -Werror -I src
  -c src/dsl_portable_expr.c -o $T/portable-1-expr-warning-check.o` **passed**.
- Python regression command from the baseline rerun: **159 passed, 41 skipped**.
  It uses the installed/pinned native dependency, not sibling-checkout inference;
  it does not verify staged 1.0 or newly edited native code.
- Latest `git diff --check` passes in both repositories.

### Handoff and true remaining gates

No unresolved approved-semantic conflict or external blocker was encountered.
The implementation is coherent/tested but **not beta-ready**. No public 1.0
validator/artifact acceptance, Python native bridge/persistence transition or pin
update was made; no release/publication/platform-CI result is claimed.

Next concrete work:

1. Add static numeric local joins (do not reintroduce output-context typing),
   finish remaining matrix operations (`fma`, `ldexp`, `logaddexp`, e/pi and
   combinatorics), per-function domains/tolerances and bound/range conversion
   conformance. Review the FP environment/nearest-rounding requirement explicitly.
2. Extend expression masks for reduction-valued lazy operands if admitted;
   complete Mandelbrot/partition/padding/concurrent-handle fixtures and cardinality
   joins. Existing program eval still broadcasts scalar outputs into nitems slots
   and does not execute an empty standalone block; do not certify that old ABI.
3. Implement extended native artifact schema/API with capacities, valid lanes,
   logical context and block-scalar outputs. Only then admit strings/ND and public
   1.0 validation/loading, retaining frozen 0.1 dispatch.
4. Continue Python bridge, immutable logical block scheduling, all persistence
   routes and effective explicit legacy opt-in, then the remaining beta gates.
   Do not update the published native pin to unpublished local code.

## Further continuation: static joins and strict FP environment

- Implemented bounded fixed-point numeric local inference using discovery trees
  that are discarded, followed by rebuilding against stable slot dtypes. Numeric
  return paths also join when output dtype is inferred. Storage conversion is
  inserted after RHS operand-driven computation, not before it. Frozen/full DSL
  continues its original single-pass compilation. For-variable collision checks
  are preserved while permitting discovery-pass rebuilds.
- Added `dsl_portable_fp.h`: staged compilation and expression evaluation save
  the caller thread's floating environment, establish nontrapping IEEE nearest/
  ties-even, then restore rounding and exception flags on success/failure. No
  shared handle or global semantic switch stores this state.
- Added local reassignment/loop/return-join tests and compilation/evaluation under
  FE_UPWARD with pre-existing FE_INVALID, checking nearest arithmetic/literals and
  exact caller environment restoration. Target `test_dsl_portable_interp` rebuilt
  and **passed** in the ordinary native build. Integrated sanitizers/full suite
  will be rerun after the next dependent increment.
- Supersedes the earlier temporary "local type joins not implemented" diagnostic
  and nearest-rounding TODO. Remaining operation, masks, descriptor and host gates
  are unchanged; public 1.0 acceptance remains closed.

## Further continuation: remaining numeric operations and lazy reduction masks

- Added typed fma/fmaf (explicit fusion), integral-exponent ldexp with checked int
  bounds, stable logaddexp including equal infinities, exact e/pi constants, and
  checked integral fac/ncr/npr. Combinatorics retains the promoted integral dtype;
  negative operands or k>n error. Binomial factors cancel before multiplication,
  so e.g. uint64 ncr(67,33) succeeds without spurious intermediate overflow.
- Found native pure FUNCTION0 token value 40 collides with logical-not token 40.
  Portable tokenization preserves zero-argument builtin identity without that
  token flag; frozen/full behavior is not changed.
- Reduction-valued where alternatives and short-circuit RHS now have explicitly
  derived participating masks, with evaluation-local child contexts/caches and
  ownership cleanup. Discarded arithmetic/reductions are not executed. Generated
  synthetic short-circuit assignments are admitted structurally while retaining
  the prohibition on user reduction assignments/returns in control-flow bodies.
- Compact integration cases cover checked factorial, exact large binomial,
  explicit fused cancellation, ldexp, infinite logaddexp, constants, masked
  where/and reductions and nested alternative masks. Ordinary native target
  `test_dsl_portable_interp` rebuilt and **passed**; sanitizer/full rerun follows.
- Supersedes the temporary reduction-valued lazy-operand diagnostic. All numeric
  builtin names now have staged implementations (complete domain/accuracy corpus
  and finite per-operation matrix details are still certification work).

## Further continuation: descriptors, empty groups, definition state and fixtures

- Added internal `dsl_eval_program_portable()` with a byte-capacity descriptor,
  valid-lane mask, single-item block-scalar output and explicit empty scalar-block
  execution. Reduction provenance propagates through locals; lane-dependent
  condition/range/loop-flow assignments demote uniformity. Empty reductions use
  the real zero-lane group, with a separate dependency slot for scalar locals.
  Elementwise zero-item calls still skip execution. Legacy eval ABI is unchanged.
- Added per-call/per-lane local initialization masks to staged numeric evaluation;
  participating uninitialized reads error, lazy discarded reads do not. Loop
  variable writes define only active lanes. Range arguments now use checked int64
  conversions instead of legacy unchecked float/uint64 casts.
- Added public extended **descriptor ABI** (not language/schema 1.0 loading):
  `me_artifact_eval_ex`, buffer widths/capacities, logical-context/mask fields,
  cardinality and itemsize queries. Frozen 0.1 rejects unavailable context/masks.
  Product/address extent/alignment/overlap checks precede execution; old ABI is
  preserved. Tests cover undersized/misaligned/overlapping buffers, bad descriptor
  versions, overflow and empty calls with no data/output buffers.
- FP environment setup now also verifies gradual float32/float64 underflow,
  rejecting flush-to-zero/denormals-are-zero environments without arch-specific
  or global mutable switches. Updated native language/artifact drafts.
- Compact fixtures added for scalar output sentinels, empty sum/mean/min, local
  scalar provenance, definition masks, invalid range bounds, lane-dependent
  cardinality, Mandelbrot-style break/partition/padding, and concurrent shared
  handle calls with distinct caller rounding/exception flags.
- Descriptor/artifact and initial empty-group increment passed all **343** native
  tests; targeted interpreter/artifact ASAN/UBSAN tests both passed. Subsequent
  definition/range/FP/fixture/cardinality edits are being verified in the next
  full/sanitizer rerun. No public 1.0 acceptance or native pin update occurred.

- Latest integrated rerun after definition/range/gradual-underflow fixtures and
  lane-dependent cardinality demotion: full **343/343 native tests passed**;
  targeted artifact/interpreter ASAN/UBSAN **2/2 passed**; the typed expression
  module still compiles with `-Wall -Wextra -Werror`; Python baseline rerun is
  **159 passed, 41 skipped**; both repository `git diff --check` calls passed.
  Ordinary native build retains only the observed duplicate-library linker
  warning. Follow-up concurrent error-path FP restoration assertions and the
  sinpi/cospi increment below are also covered by the final integrated rerun.

## Further continuation: pi-scaled math accuracy

- Replaced staged sinpi/cospi's large `pi*x` intermediate with exact modulo-two
  reduction and reflection in pi units before libm evaluation. Integer and
  half-integer anchors are exact, including large float32/float64 inputs; near
  half-integer cospi avoids cancellation. Documented signed-zero anchors and
  non-finite NaN behavior; frozen/full DSL is untouched.
- Added integer/half-integer/large/negative-zero/near-half/infinite input fixtures
  for both float precisions. Concurrent shared-handle fixtures now also assert
  FP environment restoration after checked-arithmetic failure.
- Final verification: **343/343 native tests passed**, targeted ASAN/UBSAN
  interpreter/artifact **2/2 passed**, typed expression source warning check
  **passed** with `-Wall -Wextra -Werror`. Python regression remains the latest
  **159 passed / 41 skipped** run above. No genuine blocker occurred; beta gates
  remain implementation work, not a scope reduction or public 1.0 certification.

### Precise remaining work after these increments

1. Finish static cardinality/placement conformance (including returned lanes and
   reduction-valued nested flow), per-operation domain/tolerance/type matrix, and
   platform-specific strict-FP certification. Extended internal evaluator has not
   been certified as a complete artifact execution contract.
2. Connect descriptor-based numeric 1.0 artifact compilation/evaluation only after
   validation, typed constants and immutable signature/cardinality are enforced;
   finish actual schema 1.0 decoding and required-context/capability queries.
3. Add strings and ND using existing machinery with profile-aware semantics;
   verify byte widths, capacities, coordinates and empty logical blocks.
4. Then complete Python bridge/logical scheduling/persistence/legacy policy and
   platform/wheel/documentation gates. Keep published pin and public version gates.

## Continuation: native schema/signatures, strings, ND, and staged Python bridge

- `me_artifact_load` now dispatches schema/language 1.0 to profile-aware typed
  interpreter compilation. Immutable signatures enforce all eleven numeric types,
  exact width-checked signed/unsigned constants, IEEE bits, fixed-string widths,
  Unicode32BE/byte hex constants, result cardinality and context rank. Unknown,
  missing and unimplemented capabilities/operations/signatures explicitly reject.
  Source tree scanning requires used numeric/control/reduction/string features.
  Draft acceptance is not full certification; 0.1 validator/ABI remain frozen.
- Added internal uniform-input signature marker for strong captured constants;
  constants are broadcast in call-local buffers without mutating shared handles.
  Descriptor evaluation bypasses legacy tile execution/JIT; legacy eval rejects
  1.0 handles. Added native schema/capability/rank queries.
- Reused existing fixed-string validators, width inference, Unicode 15.0.0 case
  tables and kernels through stack-local materialized operand adapters, not a new
  parser/engine. Typed nodes handle string values, nested operations, predicates,
  concat, lazy where and string-width fixed-point joins. Non-string numeric ops
  reject string operands; explicit bool/string truth is defined by first code unit.
  Active Unicode input scalars validate; masked padded inputs are not read.
- ND coordinates come only from explicit descriptor shape/origin/extent: checked
  row-major logical `_flat_idx`, `_iN`, logical `_nN`, rank. Extent product equals
  nitems; out-of-domain padded coordinates require invalid masks. Context buffers
  and masks cannot overlap output. There is no implicit physical/global synthesis.
- End-to-end native artifact fixtures cover masked scalar/empty results, exact
  uint64 max constants, missing capabilities/cardinality/rank, thirteen byte-string
  operations (nested/where), Unicode expansion/truncation/constant endian encoding,
  invalid Unicode, and padded 2D coordinates. Target ordinary native tests pass.
- ASAN found an adapter header copy over-read of zero-arity expression allocations;
  fixed by copying only `offsetof(me_expr, parameters)` into owned stack headers.
  Sanitized artifact executable now passes with `ASAN_OPTIONS=symbolize=0` (the
  local symbolizer stalled when reporting the now-fixed finding). Interpreter
  sanitizer target also passed. Final full rerun follows below.
- Started conditional Python native bridge: descriptor availability, authoritative
  descriptor_info, and explicit evaluate_block (scalar shape(), fixed widths,
  masks and logical ND context). Cython generation passed; new optional installed-
  native tests never search a sibling checkout. No native pin/build replacement.

### Exact next task at session limit

1. Add returned-lane/reduction-local/string concurrent fixtures and finish finite
   operation domain/type matrix certification (special domains, float32 result
   types, literal/strong operand combinations, endian and width bounds). Dynamic
   integral substr/split indices, replace bounds for typed constants/dynamic
   strings, and explicit required-JIT rejection were completed subsequently.
2. Complete logical block-grid scheduling and partial reads in the lazy host.
   `PortableKernel.from_json`, signature/cardinality/rank metadata, standalone
   evaluate and explicit evaluate_block now route to native descriptors when
   the installed ABI supports them, without Python reconstruction. Authoring 1.0,
   lazy scheduling and persistence remain unchanged. Preserve pin independence;
   actual compiled Python/native 1.0 execution tests currently require an opt-in
   runtime and skip normally. Low-level Cython generation and bridge header compile
   pass; host routing is tested with an explicit test double, not a language engine.
3. Complete all safe persistence routes and explicit legacy policy propagation;
   then platform/wheel/docs gates. No genuine blocker or scope conflict occurred.

- Subsequent native signature increment: integral source/captured/per-lane string
  indices materialize into existing kernels with safe bounded clamping;
  substring dynamic lengths retain subject width. Replace derives a checked
  literal-aware/worst-case bound from capacities, rejecting widths over one MiB
  and empty participating needles. Added negative index, UINT64_MAX length,
  typed byte constants/replacement and runtime-empty-needle fixtures. Ordinary
  and ASAN/UBSAN artifact executables passed. Added explicit `jit-required`
  unsupported diagnostic; final integrated rerun follows.

## Latest integrated verification and resume boundary

### Current continuation: logical partitions and persistence (local native build)

- Rebuilt the editable extension in conda `blosc2` using supported CMake
  `FETCHCONTENT_SOURCE_DIR_MINIEXPR=/Users/faltet/blosc/miniexpr` and
  `FETCHCONTENT_SOURCE_DIR_BLOSC2=.../build-portable-python/_deps/blosc2-src`
  development overrides, `pip install -e . --no-build-isolation`. Published
  FetchContent pin is unchanged. All three installed-native descriptor tests
  now execute and pass; they do not skip in local verification.
- Added `src/blosc2/portable_lazy.py` native-only `PortableLazyArray`, constructed
  via `PortableKernel.lazy`. Logical domain/partition tuples are independent of
  operand storage grids. Scalar returns allocate the C-order partition grid;
  elementwise returns allocate the original domain. Edge groups use actual valid
  extents (no padded storage lanes synthesized). ND calls preserve original
  logical shape/origin. Native partition lane limits validate at construction.
- Basic slices/integers/ellipsis/negative strides select intersecting original
  groups, evaluate each whole group, and only then gather selected output lanes.
  Partial scalar reads never repartition a reduction. Fancy indexing conservatively
  evaluates the full original result before indexing. `compute`/`rechunk` preserve
  partitions while choosing output storage chunks/blocks. Full compute streams
  output storage chunks using selected original groups, avoiding full-result
  NumPy allocation. Groups crossing output chunk boundaries may be evaluated
  repeatedly but are never repartitioned. Partial/fancy reads return NumPy values.
  Broadcast operands materialize against each group's extent while preserving
  output-domain ND coordinates, with a partial-read regression fixture.
- Portable disk carriers, cframes, MessagePack structured objects, EmbedStore,
  DictStore, TreeStore and nested arithmetic LazyExpr payloads roundtrip with
  native artifact JSON and logical partitions, no originating function/source
  reconstruction. Native artifact load precedes operand binding. Normal tests
  never infer a sibling directory; local overrides are development-only.
- Removed new full/Python LazyUDF persistence and the old source fallback.
  Shared recursive preflight runs before store overwrite/delete, and LazyUDF
  save/frame rejects before touching a destination. Raw legacy UDF carriers
  reject on NDArray save/frame too. Existing source-backed files still load only
  under explicit effective `deserialize='full'`; policy propagates recursively
  through structured operand refs and old LazyArray refs. No migration added.
- Safe arithmetic LazyExpr envelopes now have an AST whitelist (no calls,
  attributes, subscriptions or ambient names) before host construction, with
  safe policy propagated to every nested operand. Legacy nested DSL cannot hide
  behind an arithmetic or portable carrier; explicit full remains required.
- CTable computed/generated legacy DSL metadata is policy-detected before
  reconstruction; persistent add/save/frame/copy/physical-pack routes reject
  full-DSL source descriptors before output writes. In-memory full DSL computation
  remains supported. **Native portable CTable computed/generated transformer
  descriptors are still unfinished**; do not claim table integration complete.
- Current new tests cover original-group reads, physical grid mismatch, partial
  slices/rechunking, block-scalar grid shapes, ND/Unicode/mixed disk roundtrips,
  all five container routes, native-only guard, preserved overwrite destinations,
  effective nested legacy policy and CTable policy/preflight. Existing legacy
  roundtrip tests now assert the new save rejection, while frozen legacy load
  fixtures remain tested with explicit full. Removed 22 redundant external-corpus
  Python parameter cases; exhaustive conversion coverage remains native.

Changed Python inventory added this continuation: `src/blosc2/portable_lazy.py`,
`portable_kernel.py`, `b2objects.py`, `lazyexpr.py`, `schunk.py`, `core.py`,
`ndarray.py`, `msgpack_utils.py`, `embed_store.py`, `dict_store.py`, `tree_store.py`,
`ctable.py`; tests `test_portable_lazy.py`, `test_portable_policy.py`,
`test_b2objects.py`, `test_deserialization.py`, `test_dsl_portable.py`, and
`ctable/test_ctable_computed_cols.py`. Native edits from previous increments
remain preserved; no publishing, pin change or git state operations occurred.
User documentation updated in `doc/reference/portable_dsl.rst` for draft logical
partitions, partial reads, persistence routes and explicit legacy policy.

Exact next implementation: portable CTable computed/generated descriptors and
row-domain/append/refresh semantics. Full DSL table sources now reject new
persistence but native portable table transformers are not connected yet. Then
finish authoring 1.0 records and the finite native conformance/platform gates.

Latest verification for this continuation:

- Combined relevant Python portable/persistence/CTable suite: **315 passed,
  19 skipped**. All actual 1.0 descriptor/lazy/ND/string runtime tests execute,
  none skip. The 19 skips are 18 opt-in external 0.1 corpus fixtures and one
  opt-in standalone native artifact runner; stock tests still do not discover
  sibling sources. Portable-only coverage remains compact after replacing
  redundant conversion/corpus parameter cases with orthogonal host boundaries:
  **181 passed + 19 skipped = 200 portable Python cases** in the focused run.
- Full native CTest rerun passed (**343 tests**); targeted ASAN/UBSAN artifact and
  interpreter rerun **2/2 passed** with symbolization disabled as documented.
- Ruff lint/format for all 18 affected Python source/test files passed; both repo
  `git diff --check` calls passed. Local editable build completed successfully;
  the three previously skipped installed-native descriptor fixtures passed.
- CTable computed/generated tests now **96 passed**, keeping full DSL local
  computation while asserting new source persistence rejection. Frozen legacy
  file load remains tested under explicit full. No external platform CI or wheel
  matrix executed; do not claim beta readiness. No semantic blocker encountered.

- Final native build and full CTest run: **343/343 passed**, including all new
  schema/signature/string/ND/index/replacement fixtures. Build log retains only
  the existing duplicate-library linker warning.
- Final targeted ASAN/UBSAN interpreter/artifact CTest run: **2/2 passed** with
  `ASAN_OPTIONS=symbolize=0` to avoid the local symbolizer hang. The real adapter
  over-read was fixed before this passing run; it is not being suppressed.
- `dsl_portable_expr.c` and `dsl_artifact.c` independently compiled with
  `-Wall -Wextra -Werror`. Cython generation and bridge header syntax checks for
  both descriptor-enabled and unavailable branches passed. This is not yet an
  end-to-end compiled Python 1.0 runtime certification.
- Python relevant tests: **160 passed, 44 skipped**; three additional skips are
  intentional installed-runtime 1.0 descriptor fixtures. The extra passing host
  fixture checks native-authoritative metadata, byte-order/stride normalization
  and explicit block routing without a source interpreter/test engine.
- Ruff lint/format and both repository diff whitespace checks passed. Source
  byte-string literals now contextualize safely in where/output families, with
  ASCII checks and bounded UTF32-to-byte materialization; native fixture passes.
- First resumed implementation task: connect `PortableKernel` standalone block
  routing to lazy **logical partition scheduling** (immutable C-order block grid,
  block-scalar grid result shape, full-group evaluation before partial selection,
  explicit ND origin/extent and padding masks), then persistence/legacy policy.
  Keep old installed runtime tests independent. In parallel with those dependency
  milestones, finish the finite conformance/platform checklist above, not another
  numerical approximation polishing loop. No genuine blocker encountered.

## Files changed in this implementation session (current inventory)

Python-Blosc2:

- `plans/portable-DSL-1.0-progress.md`: durable handoff and verification.
- `src/blosc2/dsl_artifact_bridge.h`: version-guarded descriptor ABI/fallbacks.
- `src/blosc2/blosc2_ext.pyx`: low-level signature and block-evaluation bridge.
- `src/blosc2/portable_kernel.py`: authoritative descriptor metadata/standalone
  block routing with explicit ND context, no Python source reconstruction.
- `tests/test_portable_descriptor.py`: optional installed-native 1.0 fixtures.
- `tests/test_portable_descriptor_host.py`: pin-independent host routing fixture.
- Previously authored scope/implementation plans and unrelated untracked work
  remain preserved. No native pin, public authoring or persistence changes yet.

Miniexpr:

- `CMakeLists.txt`: add typed helper/expression sources and native integration tests.
- `src/dsl_semantic_profile.h`: per-program full/portable profile identities.
- `src/dsl_portable_expr.h`, `.c`, `dsl_portable_fp.h`: typed tree validation,
  checked interpreter, libm/reductions/masks/string values and local FP policy.
- `src/dsl_compile.c`, `dsl_compile_internal.h`, `dsl_compile_support.c`:
  profile compilation, fixed-point local joins/widths/cardinality, ownership.
- `src/dsl_eval.c`, `dsl_eval_internal.h`: masked statement integration,
  initialization tracking, scalar/empty/descriptor/ND evaluation.
- `src/dsl_parser.c`, `dsl_parser.h`: profile token/parser integration.
- `src/functions.c`, `functions.h`, `src/miniexpr.c`, `miniexpr_internal.h`:
  raw typed expression profile, operator identities, string-kernel adapters.
- `src/dsl_compare.c`: profile-aware integral comparison integration.
- `src/dsl_jit_runtime_internal.h`, `dsl_jit_runtime_host.c`,
  `dsl_jit_runtime_nonhost.c`: staged interpreter/JIT guards and program metadata.
- `src/dsl_artifact.c`, `miniexpr_artifact.h`: draft schema/signatures/constants,
  capabilities/context/cardinality queries and checked descriptor API.
- `tests/test_dsl_artifact.c`, `test_dsl_portable_interp.c`: end-to-end fixtures.
- `src/dsl_portable_types.h`, `src/dsl_portable_types.c`: internal operand
  promotion and division dtypes; checked signed/unsigned integer arithmetic,
  floor division/modulo, shifts, nonnegative powers, fixed-width bitwise
  operations, float-to-integer and exact integer conversions/comparisons.
- `tests/test_dsl_portable_types.c`: explicit 121-pair expected promotion matrix.
- `tests/test_dsl_portable_checked.c`: exhaustive small-domain and extrema tests.
- `doc/dsl-spec/1.0.md`, `artifact-1.0.md`, `features-1.0.json`: native drafts;
  not a claim of implemented 1.0 or a frozen specification.

Helpers define left shift as checked multiplication, signed right shift as floor
division by a power of two, and nonnegative integer powers (`0**0=1`). Fractional
weak floating literals next to integers/Booleans remain floating rather than
being truncated. These are implementation-led details documented in the draft.
