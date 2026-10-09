# Menudet CI qualification — 2026-10-09

This is CI evidence, not an M7/top-level status update. The parent owns
`plans/menudet-numpy-compat.md` and `plans/menudet-m7-scope-review.md`; neither was
edited or staged here. The initial local diagnosis below is retained as historical
evidence; the publication/remote qualification continuation supersedes its earlier
authorization and unpublished-revision statements.

## Authorized publication and remote qualification continuation

The user explicitly authorized task-scoped commits, pushes and workflow dispatch.
Only the four native fix/regression files were staged. Native fixes are published
as `7fa177713112561c363206c97f7faf1223ab4230` (parent `25a4aa4`). Python's existing
task baseline `6c863ebacc0d47359c9f5890dda2bef0dce7b293` was also pushed; its remote
branch had still pointed at `cf6f25c1`. No dependency default or pin was changed.

Runs launched at these exact revisions:

| Workflow | Run | Revision / selection | Status |
| --- | --- | --- | --- |
| Native CI | [37952303800](https://github.com/Blosc/miniexpr/actions/runs/37952303800) | native `7fa177713112561c363206c97f7faf1223ab4230` | Success, seven jobs |
| Explicit paired qualification | [37952373294](https://github.com/Blosc/python-blosc2/actions/runs/37952373294) | Python `6c863eba`; dispatch `native_ref=7fa177713112561c363206c97f7faf1223ab4230` | Failed; findings below |
| Python Tests | [37952354685](https://github.com/Blosc/python-blosc2/actions/runs/37952354685) | Python `6c863eba`, unchanged release dependency | Failed; draft 1.0 dependency lacks 1.1 |
| Python Tests (WASM) | [37952354689](https://github.com/Blosc/python-blosc2/actions/runs/37952354689) | Python `6c863eba`, unchanged release dependency | Failed; draft 1.0 dependency lacks 1.1 |

The push-triggered paired run `37952354858` selects a branch, not an explicit SHA;
the dispatched run above is the authoritative exact-selection qualification.
Initial dispatch returned 404 because the Python workflow was not yet published;
dispatch succeeded after pushing the existing Python baseline.

### Inspected supported matrices and evidence boundaries

- Native `ci.yml`: Linux x64, macOS ARM64, Windows x64; Linux ARM64 TCC enabled
  and disabled; Windows ARM64 with bundled TCC/TCC JIT disabled; standalone wasm32
  with side-module JIT trace assertion. Disabled-TCC jobs preserve required host-CC
  corpus tests where supported. Their existing exclusion of the full-DSL TCC
  runtime-cache test is configuration-specific, not a new qualification skip.
- Paired `menudet-paired.yml`: Linux x64, macOS ARM64, Windows x64; conda `blosc2`,
  Python 3.12, NumPy 2.5.3; native artifact build and full CTest plus five explicit
  Menudet Python modules, all four corpora and required array capabilities.
- Python `build.yml`: default-dependency full suite on the three platforms at
  Python 3.12, plus Linux Python 3.12 / NumPy 1.26 and Linux Python 3.14. The 1.26
  job builds with NumPy 2 headers then pins the compatible runtime and Zarr range.
  These are host/default-dependency checks, **not** NumPy 1.26 qualification of the
  explicit experimental native pair.
- Python `wasm-tests.yml`: Pyodide `314.0.0`, Python 3.14, whole wheel suite and
  certification scripts; unchanged release dependency, **not** explicit-pair WASM.
- Release `cibuildwheels.yml` inspected, not dispatched: Linux x64/ARM64, Windows
  x64/ARM64, macOS x64/ARM64; cp311-abi3 and cp314t/cp315t wheels; abi3 install
  matrix Python 3.11–3.15; Pyodide cp313/cp314. Existing cp315 macOS x64 test
  exclusion reflects missing dependency wheels; free-threaded import enables the
  GIL and is not no-GIL qualification. None constitutes current-pair wheel evidence.

### First remote iteration and follow-up fixes

Native `7fa1777` passed all seven jobs in run `37952303800`. Explicit-pair run
`37952373294` failed honestly: Windows Ninja generated shared import and static
libraries with the same `miniexpr.lib` path; macOS optional arithmetic compilation
exceeded 60 seconds; Linux's NumPy infinity-to-int64 sentinel was signed minimum,
not the corpus generator's ARM64 signed maximum.

Follow-up native commit `216c603ef64d7e49bbbb8fc6001f51c74bfb3d0b` passed all seven
jobs in [37954150929](https://github.com/Blosc/miniexpr/actions/runs/37954150929).
It raises arithmetic corpus **time budgets** to 180 seconds (no numeric policy
change) and records reviewed **exact** x64/ARM64 NumPy observations for the already
platform-qualified nonfinite cast case. The original expected value, divergent
classification and native `evaluation_error` baseline are unchanged. The Python
drift checker selects the explicit machine observation; unlisted machines retain
the original reference and fail on drift. New regressions require exact comparison
and forbid machine variants without a platform qualification.

Python follow-ups select static-only native linkage in the paired workflow, use
Windows-native source paths, and require actual TCC/CC/Clang portable JIT execution
on Linux/macOS as a separate test gate. The JIT test's three old assertions that
division/shifts must fall back were obsolete after expanded lowering. They are
replaced by **required compilation**, bit-exact interpreter/value/status parity,
zero/overflow/nonfinite/invalid-shift guard checks (nine new backend cases); mixed
uint64/signed-weak comparison and fast-math fail-closed assertions remain intact.

Broader Tests `37952354685` and Tests (WASM) `37952354689` failed because their
unchanged release dependency accepts draft 1.0 only while new tests exercise 1.1.
CI now checks out `.github/menudet-native-ref` (exact native `216c603`) and sets an
explicit source override **only on `refs/heads/numpy-compat`** in those workflows.
Main/release defaults and native/Pyodide dependency pins remain unchanged. Release
wheel workflows are not modified or dispatched. The explicit-pair dispatch still
selects a full native SHA independently of that branch-specific CI file.

Follow-up local checks (all `conda run -n blosc2`): native **430/430** (86.59 s),
five-module explicit pair **487 passed / zero skipped** (66.98 s), actual-JIT
**207 passed / zero skipped** (142.34 s), Ruff lint/format and `git diff --check`
passed. No compilation source changed in the follow-up native commit; no new
compiler warnings were emitted by the incremental native build.

### Second remote iteration, isolation and remaining failures

Python `65a25b8a5036134057ce79477e7db070d0d54bae` was published with the initial
workflow/reference/JIT-test fixes. Run
[37955282982](https://github.com/Blosc/python-blosc2/actions/runs/37955282982)
passed macOS but exposed Windows temporary-file sharing and Linux TCC headers.
Python `36023f571c6f2c4d59fb68415840b72b35aefdc6` closes the corpus input before
launching the native subprocess (new regression also checks cleanup), adds JIT
trace evidence, and fixes CI-only Pandas/Arrow test runtime requirements.
The release dependency pins remain unchanged. The NumPy 1.26 job alone uses
PyArrow 21.0.0: PyArrow 26 rejected NumPy 1.26 at import without expressing that
requirement in its PyPI `Requires-Dist` metadata. Pandas is an existing dev
dependency needed by Arrow's dictionary conversions; it is installed in CI rather
than adding a new shipping dependency.

At Python `36023f57` / native `216c603`, explicit-pair run
[37957000054](https://github.com/Blosc/python-blosc2/actions/runs/37957000054)
passed Windows and macOS. Linux's **required** TCC test failed (not skipped):
`/usr/include/stdint.h:26: error: include file 'bits/libc-header-start.h' not found`.
Native follow-up `b2c88e9a22e8121024c28991763cfcf22687fbeb` discovers existing
architecture-specific Linux system include directories through libtcc's optional
system-include API, complementing the existing multiarch library discovery.
No dependency was added. New native `numpy_compat_portable_tcc` requires actual
compilation on supported bundled Linux/macOS TCC builds. It passed locally in a
fresh static native build; Linux needs its own remote result.

The parent independently committed alias cleanup at native `b0541bb` and Python
`cf9cf359` **without publishing it**. Continued CI fixes use separate worktrees
rooted at native `216c603` / Python `36023f57`, with task branch
`menudet-ci-qualification`. Only task descendants are pushed explicitly as
`HEAD:refs/heads/numpy-compat`. Parent alias commits/files, M7 plans and
`graph-preparation.md` are not staged, changed, reset or pushed by this task.
The final pair does **not** qualify the parent's alias-removal corpus or adapters.

Concrete broader release gates still open (not removed from CI):

- [Tests 37957000018](https://github.com/Blosc/python-blosc2/actions/runs/37957000018),
  Python `36023f57` / native `216c603`: Linux Python 3.12/3.14, macOS and Windows
  passed. Linux NumPy 1.26 failed **14 tests** in
  `tests/ctable/test_pandas_arrow_import.py`: those fixtures request NumPy 2
  `StringDType` UTF-8 storage, which the package correctly rejects on 1.26.
  Other results: 12,116 passed / 423 skips. No unrelated CTable implementation or
  test was changed to conceal these errors.
- [Pyodide 37955282922](https://github.com/Blosc/python-blosc2/actions/runs/37955282922),
  Python `65a25b8a` / native `216c603`: **80 failed, 10,479 passed, 1,011 skipped**
  (758.40 s). Failures are 70 reduction tests assuming host `intp` rather than
  the documented portable int64/uint64 accumulator defaults, nine tests demanding
  unavailable WASM FP flags/threading, and one subprocess test on a runtime without
  processes. These failures remain visible; no WASM job/test was hidden or numerical
  policy weakened. Standalone native WASM success is not paired Python WASM success.
- GitHub core API quota temporarily exhausted at 16:11 UTC; reset header was
  16:17:53 UTC. Work waited for reset and resumed. Watches now use a 60-second
  interval to avoid unnecessary quota consumption; no result is inferred from a
  watch interrupted by API errors.

Native `216c603` run `37954150929` counts: Linux x64 430, Linux ARM64 TCC 430,
Linux ARM64 TCC-disabled 338, macOS ARM64 430, Windows x64 335, Windows ARM64 244,
WASM 63. All passed, including the WASM side-module trace assertion. Native logs
also contain pre-existing SLEEF macro/always-inline/posix_memalign warnings and
`dsl_jit_backend_libtcc.c` diagnostic format-truncation warnings; green CI is not
claimed to be warning-free.

### Required TCC runtime asset qualification

Python `e44c0044f904bdc0ebee16ea4737e273d63949b9` selected native `b2c88e9`.
[Paired 37958957787](https://github.com/Blosc/python-blosc2/actions/runs/37958957787)
passed Windows/macOS but the new **required native TCC** test failed on Linux.
Native `b2c88e9` [37958780104](https://github.com/Blosc/miniexpr/actions/runs/37958780104)
also failed Linux x64/ARM64 TCC while the other five jobs passed. The gate was
not removed or converted back to optional fallback.

Native diagnostic commit `4aabfe873d044e64583247c7322d53c6b06ff9de` enabled trace
on that test. [37960125294](https://github.com/Blosc/miniexpr/actions/runs/37960125294)
identified the second missing asset exactly:
`/usr/include/string.h:33: error: include file 'stddef.h' not found`.
The multiarch glibc header fix was working; compiler-provided TinyCC headers were
not staged or installed beside the relocated runtime library.

Native `aefe19dfc893877eb76818d0d4c57adb78fbacdf` stages/installs the existing
bundled TinyCC `include/` tree beside `libtcc`. No new dependency, native pin,
numeric assertion or tolerance is involved. A fresh static local native build
passed **431/431** (103.08 s), including required TCC. Installation to a private
prefix succeeded, and required TCC also passed with `ME_DSL_JIT_LIBTCC_PATH`
pointing at that installed runtime (0.41 s), rather than the staged build copy.
Local static linkage emitted duplicate `-lm` linker warnings, recorded separately
from the prior shared-build no-warning result. Remote Linux/ARM64 must confirm
the runtime fix; local Apple results alone are not Linux evidence.

Remaining release gates include experimental-pair NumPy 1.26 and other interpreter
versions, paired Python ARM64 Windows/Linux and WASM integration, release wheel/
sdist qualification against the selected native revision, clean-install optional
packaging, sanitizer/stress coverage and M7 scope/docs/performance acceptance.
No skipped corpus, increased tolerance or removed platform is accepted as a CI fix.

## Revision boundaries

- Native revision supplied at delegation:
  `2567eb6c9aca059c0f19277a09d6496b65b86d97`.
- Native HEAD observed during this task:
  `25a4aa4e2afb05f350afa50a928d5478ff05fa7a`.
- Python HEAD:
  `6c863ebacc0d47359c9f5890dda2bef0dce7b293`.
- Local tests below use native `25a4aa4` **plus the working-tree fixes below**.
  They do not qualify the clean native SHA or an unpublished final revision.
- No state-changing Git or GitHub operation was performed by this task.

## Published failures inspected with read-only `gh`

| Run | Exact native SHA | Result |
| --- | --- | --- |
| [37901442377](https://github.com/Blosc/miniexpr/actions/runs/37901442377) | `5d6b1a37e34cddb9f194a10a3157168a5fa92526` | Linux x64/ARM64 builds, Windows x64/ARM64 function tests, WASM artifact test failed; macOS passed |
| [37934733524](https://github.com/Blosc/miniexpr/actions/runs/37934733524) | `25a4aa4e2afb05f350afa50a928d5478ff05fa7a` | Same platform failures; Windows also exposed Boolean `real()` result inference |

The newer run was discovered during the task: its base native revision is now
published, contrary to the initial unpublished assumption. The working-tree fixes
are not published and have no CI result.

Python's latest observed published branch runs were Tests `37896228314` and
Tests (WASM) `37896228386`, both successful at
`cf6f25c1020c13864413e97b39e466c893b77a7b`. Those historical successes do **not**
qualify Python `6c863eba` or the current pair. No paired workflow run was listed.

## Targeted fixes

In `/Users/faltet/blosc/miniexpr`:

1. `CMakeLists.txt`: make `m` a public adapter dependency on non-Windows.
   Linux logs show `fesetround` missing for the adapter consumer and `fegetenv`
   missing for the conformance runner. A shared miniexpr library's private `m`
   dependency cannot satisfy these direct consumer references. The local runner
   link command now contains `libminiexpr_artifact.a`, miniexpr, and `-lm`.
2. `tests/CMakeLists.txt`: embed the affine JSON fixture into the WASM virtual
   filesystem at the exact compiled-in path. `test_dsl_artifact` previously aborted
   at `read_fixture`'s `fopen` assertion. This preserves the test instead of skipping it.
3. `src/functions.c`: recognize both the direct CRT `remainder` symbol and the
   registered builtin address when selecting NumPy modulo. Windows CRT inline/import
   aliases otherwise select C remainder semantics, reject narrow integer signatures,
   and produce wrong exceptional floating results.
4. `CMakeLists.txt`: propagate Windows `/OPT:NOICF` to the library and its consumers.
   Builtin function-pointer identity is semantic; Release identical-code folding
   must not merge `real`/`conj` callbacks with different Boolean inferred dtypes.
5. `tests/test_dsl_numpy_builtin_identity.c`: exercise Boolean `real` versus `conj`
   inference and signed int8 NumPy remainder inference/values. This runs on native
   and WASM and is included in Windows Release CTest builds.

No dependency pins, workflow files, unrelated plans, or user files were changed.

## Local qualification

All builds, installation and tests used `conda run -n blosc2`. The local environment
is macOS ARM64, Python 3.14.4, NumPy 2.5.3, AppleClang 21; WASM uses Emscripten 5.0.0.
Build directories and logs are under
`/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/`.

- Native Release/artifact build: succeeded. Final full suite:
  **430/430 passed** (31.13 s), including both required-CC
  arithmetic/function corpus tests.
- WASM Release/artifact build with the CI options: succeeded. Final full suite:
  **63/63 passed** (28.66 s), including the previously
  failing fixture test.
- New builtin-identity regression: passed separately on native and WASM.
- WASM side-module trace gate: passed and emitted
  `[me-dsl] jit runtime built: fp=strict compiler=tcc`.
- Python editable extension rebuilt successfully with
  `--config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=/Users/faltet/blosc/miniexpr`.
  This selects the patched native source rather than the release pin.
- `git diff --check`: passed.
- Compiler warnings: none in the standalone native/WASM builds. Native bundled
  tinycc emitted one GNU make `jobserver unavailable: using -j1` warning.

- Python explicit-pair run: **483 passed, zero skipped** (71.32 s). Selected
  `tests/test_menudet_arrays.py`, `tests/test_menudet_native_graph.py`,
  `tests/test_menudet_functions.py`, `tests/test_menudet_arithmetic.py`, and
  `tests/test_menudet_conformance.py`, with `-q -n 0` and these environment variables:

  ```text
  MENUDET_REQUIRE_ARRAY_RUNTIME=1
  MENUDET_NUMPY_FUNCTION_CORPUS=/Users/faltet/blosc/miniexpr/tests/numpy-compat/functions-v1.1.json
  MENUDET_NUMPY_ARITHMETIC_CORPUS=/Users/faltet/blosc/miniexpr/tests/numpy-compat/arithmetic-v1.1.json
  MENUDET_NUMPY_CORPUS=/Users/faltet/blosc/miniexpr/tests/numpy-compat/vectors.json
  MENUDET_NUMPY_CORPUS_V2=/Users/faltet/blosc/miniexpr/tests/numpy-compat/vectors-v2.json
  MENUDET_NUMPY_RUNNER=/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/menudet-ci-native/tests/numpy_compat_runner
  ```

  The preliminary unconfigured Python run (467 passed, 16 skipped) is not treated
  as full corpus qualification. Python build output contained no warning/error.

## Remaining gates / blockers

- The patches need actual Linux x64, Linux ARM64 (TCC on/off), Windows x64,
  Windows ARM64 and GitHub WASM execution. Local macOS/WASM success does not prove
  Windows alias/folding behavior or Linux linker behavior. No local Windows/Linux
  qualification was performed.
- Python `6c863eba` is not represented by the historical Python green runs.
  The current paired workflow builds a native source override and explicitly
  requires array capabilities and all four corpora; dispatch can select an exact
  native SHA via `native_ref` and uploads both checked-out SHAs.
- Final native fixes and the Python revision must be remotely available for
  GitHub CI to qualify the final pair. Publication/workflow dispatch was not
  authorized in this task, so no new CI run was launched.
- Current-pair NumPy 1.26, Linux/Windows Python paired results, Python WASM integration,
  and the broader Python test suite remain separate gates; no claim of current-pair
  coverage is made from historical green results.
