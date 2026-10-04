# Safer JIT implementation review

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
   preserved. Windows gains callback capture/serialization. Broken local Node
   prevents the existing JavaScript smoke test from running.
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
