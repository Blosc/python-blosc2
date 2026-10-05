# Safer JIT implementation review

Status: implementation completed and integrated on `safer-jit`; macOS validation
completed with the environment limitations recorded below. Linux policy/platform
verification remains a release gate. No dependency revisions were pushed.

## Implemented

- Linux minicc uses anonymous memfd-backed RX/RW aliases. It requests `MFD_EXEC`,
  retries without that flag only for `EINVAL`, and reports allocation failures
  without a filesystem fallback. Reservations, partial aliases and descriptors
  are cleaned up. Section sizes are checked before narrowing; allocation sizes
  before doubling. ARM/RISC-V publication synchronizes both aliases.
- TCC dispatch precedes all miniexpr filesystem-cache operations. Its states are
  program-owned, not disk-cached. Failures keep the interpreter path; libtcc
  diagnostics are captured on the owning program. API discovery and compile/delete
  are serialized. Explicit libtcc path overrides do not substitute another library.
- Explicit CC compilation/loading and persistent cache reuse remain available.
  Native fallback tracing names the interpreter. Generated allocation declarations
  use the target size type and the code-generation cache version is bumped.
- Python dependency pins, subprocess regression tests, JIT reference documentation,
  constructor/computation docstrings and release notes are updated.

## Implementation work by repository

### minicc: executable allocation and lifetime

- `tcc_memfd.h` implements Linux anonymous backing storage via the syscall interface,
  avoiding a dependency on the newer glibc `memfd_create` symbol. It closes the
  descriptor after mapping and preserves the original failure's `errno` on rollback.
- `tccrun.c` selects the allocator for native Linux in both CMake and configure
  builds, retains both aliases until state destruction, checks section/alignment
  sizes and publishes generated instructions through both cache addresses.
  Calling `tcc_relocate()` twice now returns an error instead of exiting the host.
- `tests/memfd_test.c` injects allocation failures without public runtime switches;
  `tests/libtcc_api_test.c` exercises executable code and mutable globals/BSS.
  `CMakeLists.txt` registers the tests and `README` explains the allocation model.

### miniexpr: dispatch, diagnostics and generated C

- `src/dsl_jit_runtime_host.c` dispatches TCC before cache-directory creation or
  cache probes. `src/dsl_jit_runtime_nonhost.c` also uses the captured failure
  reason and explicitly identifies interpreter fallback in traces.
- `src/dsl_jit_backend_libtcc.c` loads the error-callback API, stores diagnostics
  on the owning program, checks extra-option failures and serializes compiler
  discovery/compilation/destruction with POSIX or Windows locks. It honors
  `ME_DSL_JIT_LIBTCC_PATH` as an exclusive override.
- `src/dsl_jit_cgen.c` replaces `malloc(unsigned long long)` with a target-size
  declaration; `src/dsl_jit_runtime_internal.h` advances the cache version to 9.
- `tests/test_dsl_jit_libtcc.c` covers real compiler diagnostics and retained-state
  lifetime; `tests/test_dsl_jit_codegen.c` checks the allocation declaration.
- CMake links thread support, pins the updated minicc revision and tracks allocator
  sources as dependencies of the staged library. `README.md` documents fallback,
  TCC's filesystem independence and CC's separate persistent-cache requirements.

### Python-Blosc2: integration, regressions and API reference

- `CMakeLists.txt` pins the updated miniexpr revision. Integrated builds use the
  sibling sources through FetchContent overrides while the revisions remain local.
- `tests/ndarray/test_safer_jit.py` runs fresh-process checks using
  `safer_jit_probe.py` and the test-only `safer_jit_libtcc.c` relocation-denial
  double. The probes cover constructors, expressions, DSL control flow and
  reductions under successful JIT and interpreter fallback.
- `doc/reference/jit.rst` is the canonical JIT options reference. Constructor,
  lazy-array and DSL reference pages link to it; `src/blosc2/ndarray.py` and
  `src/blosc2/lazyexpr.py` expose accurate defaults and best-effort semantics.
- Reduction guidance is updated in both `doc/conf.py` and the generated
  `doc/reference/reduction_functions.rst`, so Sphinx does not overwrite it.
  `RELEASE_NOTES.md` records the behavior changes, and
  `plans/safer-jit-execution.md` records the implementation status.

### Implementation revisions

| Repository | Revision | Work |
| --- | --- | --- |
| minicc | `ffbb8ebf` | Anonymous memfd aliases and allocator/API tests |
| miniexpr | `19200c6` | In-memory TCC dispatch, diagnostic capture and minicc integration |
| miniexpr | `d704f89` | Target-size allocation declarations and cache-version update |
| Python-Blosc2 | `d9b5ca15` | Dependency integration and fallback regressions |
| Python-Blosc2 | `c10a0cd8` | Independent external-compiler diagnostic opt-in test |
| Python-Blosc2 | `8c92213c` | JIT reference, docstrings, release notes and review |
| Python-Blosc2 | `d756211b` | Preserve reduction JIT guidance through documentation generation |

## Validation on macOS ARM64

All Python, test and build commands used the `blosc2` conda environment.

- minicc CTest: **11 passed**, including mutable data/BSS, 100 execution/destruction
  cycles, repeated relocation errors and allocation fault injection.
- The allocator harness covers create/resize/reservation/RX/RW failures, cleanup,
  unsupported flags, unavailable syscall, denial, bounds and absence of RWX
  requests. On macOS it uses test-only unlinked-file backing and strips executable
  permission because shared executable file mappings are denied there. **This does
  not validate Linux memfd or ARM64 alias execution.**
- miniexpr CTest: **36 passed**, excluding the unrelated Node smoke test. The
  unfiltered run failed only because Homebrew Node cannot load `libllhttp.9.3.dylib`.
  A separate miniexpr build with TCC JIT disabled succeeds.
- Integrated editable Python builds succeed using local FetchContent overrides
  for miniexpr/minicc and the existing sibling SLEEF/C-Blosc2 sources.
- Focused Python JIT/DSL suites: **116 passed**, including the initial **18** new
  regressions. A final run of the expanded new suite reports **19 passed**.
  These check real TCC builds without a compiler or usable TMPDIR, no TCC cache
  files, quiet relocation/compile/load fallback with correct results, opt-in
  diagnostics (including the independent compiler-output opt-in),
  disabling/precedence and fresh-process CC disk-cache reuse.
- Default Python suite: **10,613 passed, 36 skipped, 4 failed**. All four failures
  are Node-dependent JavaScript tests with the same missing Homebrew library.
  No environment-wide repair was attempted.
- Ruff passes. Sphinx HTML generation succeeds, constructor HTML exposes both JIT
  kwargs, and the JIT Python examples execute successfully. Existing documentation
  warnings remain (duplicate objects, autosummary stubs, themes and unrelated
  references); the build is not warning-clean. The new cross-reference was fixed.

## Findings addressed in review

- Check ELF-sized fields before conversion to `unsigned`, not just the final size.
- Never apply `MAP_FIXED` after a failed reservation.
- Keep the writable alias alive for mutable data/BSS, not only code emission.
- Keep error-callback contexts alive until retained TCC states are destroyed.
- Dispatch TCC before any filesystem-cache creation or probing.
- Return recoverable failures instead of terminating the host or printing default
  compiler diagnostics.
- Document `BLOSC_ME_JIT` at its actual `LazyExpr.compute()` scope; `0` is not a
  disabling value. Use `ME_DSL_JIT=0` for miniexpr runtime JIT.
- Update the reduction-reference generator as well as its output; otherwise a
  Sphinx build removes the newly added JIT guidance.

## Remaining weak points and validation gaps

1. **Linux remains a release gate.** No configured Linux VM was available locally.
   Test the installed wheel on enforcing RHEL 10, Linux x86_64 and ARM64, allowed
   and denied memfd mappings, and `noexec` temporary storage. Syscall tracing should
   confirm anonymous storage and no generated files. Fault injection cannot
   establish SELinux compatibility.
2. **Dual aliases are not immutable code.** Neither mapping is RWX, but physical
   backing remains writable through one alias and executable through the other;
   the RX mapping also covers the allocation's data region. This is the selected
   allocation model, not a sandbox or strict physical-page W^X. Stricter policies
   may deny it; interpreter fallback is intentional.
3. **Windows/WebAssembly were not runtime-tested.** Their allocation paths are
    preserved. Windows gains callback capture/serialization. Homebrew Node was
    repaired by the user; the JavaScript glue smoke test now passes locally.
4. **Concurrency coverage is limited.** The new lock protects libtcc discovery and
   compile/delete, not all pre-existing miniexpr global positive/negative caches.
   A comprehensive threaded/sanitizer audit remains separate work.
5. **Compiler-library discovery remains process-scoped.** Its initial failure is
   sticky and loaded libraries are retained. Set path/environment controls before
   Python starts. Negative-cache suppression may give a generic recent-failure
   reason rather than repeat the original diagnostic.
6. **CC cache security is unchanged.** Ownership, races, symlinks and cache trust
   were not hardened. Choose suitably protected storage; TCC bypasses that cache.
7. **No Linux performance comparison was possible.** Correctness checks do not
   establish memfd overhead or throughput. Cold compilation, first library loading,
   process startup and computation need separate timing.
8. **Dependency publication is not part of local validation.** Pins reference local
   dependency revisions. Remote builds cannot fetch those revisions until they are
   published; local builds used explicit source overrides.

Issue #730 is not platform-verified until item 1 is completed.

## Follow-up: Python configuration and native call-local options

Implemented process defaults (`set_jit_options`, `get_jit_options`) and nested
`contextvars` overrides (`jit_options`) for JIT enablement/backend, accuracy,
tracing, compiler command/flags, exact CC cache directories and compiler output.
Evaluation takes a settings snapshot without modifying the environment. Explicit
per-call and LazyUDF settings retain precedence over Python defaults; existing
environment overrides retain precedence where supported. See the consolidated
parameter/precedence documentation in `doc/reference/jit.rst`.

Miniexpr accepts borrowed call-local settings through
`me_compile_nd_jit_options`; compiling-thread settings restore on success and
failure. Compiler-affecting settings now enter process-cache keys. CC-generated
file paths are shell-quoted, including spaces, apostrophes and dollar signs.
TCC remains filesystem-independent and ignores CC-specific settings.

Validation: the full Python suite passed with **10,635 passed, 36 skipped**.
After the final compiler-path parsing improvement, the 36 focused configuration
and fallback tests and all 38 miniexpr tests passed; Ruff checks passed. Sphinx
builds successfully, with unrelated existing warnings and no JIT-page warnings
after fixing heading underlines. The TCC-disabled native build also succeeds.
Linux enforcing SELinux and Windows/WebAssembly runtime validation remain open.

Current dependency integration pins published miniexpr revision
`3f4db93a628aedc662b7d27a5f35c97ffaeb0424`, matching `CMakeLists.txt`. It includes
the native options API and compiler-command parser fix, plus the subsequent native
DSL syntax extensions and chained-comparison support. The configuration-specific
validation above originally used `a1c950522b6da2b8f812c8fd89f21b7ce5446c51` through
the sibling source override. The later integrated syntax validation passed with
**10,737 Python tests passed, 36 skipped**, and **42 native tests passed**.
