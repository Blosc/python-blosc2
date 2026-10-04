# Safer, filesystem-independent TCC JIT execution

Status: implementation and macOS integration validation completed; Linux policy
and platform validation remain open. See [the review](safer-jit-review.md) for
test results, fixes, remaining weak points and release gates. Checklists below
retain the original validation scope and are not a claim of full platform coverage.

Issue: [Support for SELinux, python-blosc2 #730](https://github.com/Blosc/python-blosc2/issues/730).

## Objective

Make the default JIT work transparently on Linux installations with SELinux,
without requiring a system C compiler or a writable/executable temporary
directory. When the operating system denies JIT execution, valid computations
should continue through the miniexpr interpreter, quietly by default.

Keep bundled TCC as the default native compiler. Keep `cc` explicitly selectable
for optimized execution and persistent shared-library caching. The measurements
behind this decision show that compiling a simple kernel can take only about
30 ms on Linux, but that is still a meaningful fixed cost for small operations.
On macOS, first loading a newly generated library can cost substantially more
than compiling it. Backend selection should remain predictable rather than
introducing workload thresholds as part of this work.

## Agreed execution contract

| Situation | Expected behavior |
| --- | --- |
| Default native JIT, supported target | Try bundled TCC; no system compiler required |
| Linux TCC executable allocation | Use anonymous memfd-backed RW/RX mappings |
| TCC compilation and execution | Keep generated code and its bookkeeping in memory |
| TCC unavailable or executable allocation denied | Clean up and use the interpreter |
| Explicit `jit_backend="cc"` | Try the system compiler and existing persistent cache |
| CC unavailable, compilation fails, or library loading is denied | Use the interpreter |
| `jit=False`, absent an applicable environment override | Skip JIT and use the existing non-JIT route |
| `jit=True` | Prefer JIT; successful JIT compilation is not required |
| Native TCC/CC backend explicitly selected | Select the attempted backend; retain interpreter fallback |
| Ordinary execution without debug settings | No raw compiler diagnostics or fallback warnings |
| Opt-in tracing | Report the attempted backend, failure reason, and interpreter fallback |
| Invalid expressions, arguments, or unsupported backend names | Preserve normal validation errors |

The filesystem contract concerns JIT artifacts: TCC must neither create nor
require source files, binaries, metadata files, or cache directories. Normal
loading of installed libraries and OS paging remain ordinary platform behavior.
Persistent user arrays requested with `urlpath=` are independent of this contract.

Fallback preserves the supported computation and its numerical-accuracy
contract; it may be slower. It does not promise identical floating-point bit
patterns across all backends or successful execution after unrelated failures.
Native macOS and Windows retain their platform-specific in-memory allocation
mechanisms. WebAssembly retains its existing backend selection and helpers.

## Starting points in the code

Paths prefixed with `minicc/` and `miniexpr/` refer to those separate repositories;
they are available locally as `../minicc` and `../miniexpr`.

| Area | Main files |
| --- | --- |
| TCC runtime allocation, relocation, and release | `minicc/tccrun.c`, `minicc/tcc.h` |
| TCC build configuration | `minicc/CMakeLists.txt`, `minicc/cmake/config.h.in`, `minicc/configure` |
| Bundled TCC build and revision | `miniexpr/cmake/MiniexprTinyccTargets.cmake`, `miniexpr/cmake/MiniexprFetchDeps.cmake` |
| Dynamic libtcc API and compilation | `miniexpr/src/dsl_jit_backend_libtcc.c` |
| Native JIT dispatch and caches | `miniexpr/src/dsl_jit_runtime_host.c`, `miniexpr/src/dsl_jit_runtime_cache.c` |
| Other platform dispatch | `miniexpr/src/dsl_jit_runtime_nonhost.c` |
| System compiler and shared-library loading | `miniexpr/src/dsl_jit_backend_cc.c` |
| Diagnostics and generated C | `miniexpr/src/dsl_config.h`, `miniexpr/src/dsl_jit_cgen.c` |
| Python dispatch and constructors | `src/blosc2/lazyexpr.py`, `src/blosc2/ndarray.py`, `src/blosc2/blosc2_ext.pyx` |
| Python DSL validation helpers | `src/blosc2/dsl_kernel.py` |
| Bundled miniexpr revision | `CMakeLists.txt` |

Current behavior that motivates the changes:

- Linux TCC normally uses malloc-backed storage and changes code-page permissions
  with `mprotect()`. The reported SELinux denial occurs on that path.
- The optional `CONFIG_SELINUX` path uses a hardcoded `/tmp/.tccrunXXXXXX` backing
  file. Its creation/resizing and partial mapping failures need better handling.
- Miniexpr builds minicc through CMake, rather than its `configure` script.
  Merely supplying `--with-selinux` elsewhere does not configure the wheel's libtcc.
- TCC already compiles in memory and retains its `TCCState`; it does not write
  reusable compiled binaries. However, host dispatch prepares and probes the disk
  cache before reaching the TCC branch, introducing an unnecessary dependency.
- A relocation failure is already checked and normally permits interpreter
  fallback. The missing libtcc diagnostic callback allows raw stderr messages to
  make a recoverable failure look fatal.
- CC builds position-independent shared libraries and loads them with `dlopen()`.
  SELinux file permissions or a `noexec` cache mount can still deny that route.

## Task A: implement Linux memfd-backed allocation in minicc

### A1. Make the build select the intended allocator

- [ ] Make memfd-backed executable allocation the default for native Linux TCC
  builds, including the shared libtcc shipped by miniexpr.
- [ ] Use target-platform configuration, including cross-compilation settings;
  selection must not depend on whether SELinux is enabled on the build machine.
- [ ] Wire the allocator selection through CMake and keep the supported configure
  build consistent. Ensure legacy `CONFIG_SELINUX` handling cannot accidentally
  select the temporary-file allocator for the bundled Linux configuration.
- [ ] Keep the implementation self-contained: no libselinux dependency or runtime
  SELinux detection is needed. Allocation success determines availability.
- [ ] Support the wheel's libc baseline. Where necessary, use a guarded syscall
  wrapper instead of requiring a newer `memfd_create` libc symbol or newer headers.
- [ ] Forward any required build setting through miniexpr's minicc sub-build.

### A2. Allocate one backing object and two aliases

- [ ] Create an anonymous memfd with close-on-exec semantics. Request executable
  memfd capability explicitly on kernels supporting the relevant flags.
- [ ] Handle older kernels deliberately: an unsupported new flag may justify
  retrying with the compatible flag set. An actual permission denial must remain
  a denial. Distinguish `EINVAL`, `ENOSYS`, `EACCES`, and `EPERM` where possible.
- [ ] Validate page alignment, nonzero allocation sizes, multiplication overflow,
  and the representable range of TCC's existing size/offset fields.
- [ ] Check `ftruncate()` before mapping or accessing the object. A successful
  mapping beyond the backing object's real length can otherwise lead to SIGBUS.
- [ ] Reserve a contiguous virtual-address region large enough for both aliases,
  then establish an RX mapping and an RW mapping of the same memfd contents.
  Check the reservation before using `MAP_FIXED` within that owned region.
- [ ] Preserve TCC's relocation model: executable symbols refer to the RX alias,
  emitted code is copied through the RW alias, and writable data/BSS use their
  intended writable addresses. Preserve the required fixed-distance offset.
- [ ] Close the descriptor after the mappings have been established. The mappings
  keep the backing object alive without a filesystem name.
- [ ] Retain the required mappings for the compiled state's lifetime. In
  particular, do not unmap the entire RW alias after code generation: TCC may
  still need that alias for mutable data.

No individual mapping should request simultaneous write and execute permission.
Separate aliases do not guarantee that every security policy permits execution;
mapping failure remains a normal reason to use the interpreter.

### A3. Finish code publication and handle every failure

- [ ] Perform architecture-correct instruction-cache synchronization before
  exposing executable entry points, accounting for the RW and RX aliases.
  The existing SELinux branch skips the helper where the ordinary path performs
  ARM/AArch64 cache maintenance, so this needs explicit attention.
- [ ] Publish `run_ptr`, allocation size, and ownership metadata only after the
  allocation is complete. Track enough information for deterministic cleanup.
- [ ] Use a single, complete failure path for descriptors, reservations, and
  partially installed mappings. Preserve the original operation and `errno`
  before cleanup can overwrite them.
- [ ] Report executable-allocation failures through a recoverable libtcc error
  result. Audit this path for `exit()`/`abort()` calls; installing a diagnostic
  callback alone does not make a fatal path recoverable.
- [ ] Update `tcc_run_free()` to release the allocation correctly on success and
  on failures later in relocation or symbol lookup.
- [ ] On unsupported or denied memfd allocation, return failure directly. The
  default Linux path must not create a temporary file or launch another compiler.

## Task B: capture JIT diagnostics and preserve quiet fallback in miniexpr

- [ ] Load and register `tcc_set_error_func()` alongside the other libtcc API
  symbols, immediately after creating a state and before options, library lookup,
  compilation, or relocation can produce diagnostics.
- [ ] Capture messages in a bounded diagnostic buffer owned by the appropriate
  compilation/state. Do not redirect process-wide stderr.
- [ ] Keep callback data valid for the entire period in which libtcc may use it,
  including state cleanup. A successful compilation must not retain a pointer to
  an expired stack-local diagnostic buffer.
- [ ] Preserve useful details in the miniexpr error record: backend, operation or
  phase, and the allocator/compiler/loader explanation. Avoid replacing a useful
  diagnostic with only `tcc_relocate failed` or `compilation failed`.
- [ ] Leave the runtime kernel pointer unset on failure and release the failed
  TCC state. Do not retry relocation on an already partially relocated state.
- [ ] Emit captured diagnostics only through opt-in tracing. Include an explicit
  interpreter-fallback indication, rather than requiring users to infer it from
  a generic JIT-skip message. Existing validation helpers should retain access to
  an intelligible failure reason.
- [ ] Keep the same recoverable behavior for CC compiler absence, nonzero compiler
  exit, cache access failures, and `dlopen()` failure. Compiler output remains
  opt-in through `ME_DSL_JIT_DEBUG_CC`.
- [ ] Preserve validation and interpreter exceptions. Backend unavailability is
  recoverable; invalid Python/DSL input must still be diagnosed normally.
- [ ] Follow existing locking and state-ownership conventions. Avoid introducing
  races or cross-talk through shared diagnostic buffers during concurrent use.
- [ ] Preserve per-kernel negative caching. Add bounded backend/allocator-level
  suppression for reliably identified environmental failures if needed to avoid
  retrying the same denied operation for every new expression. Transient resource
  errors and kernel-specific compilation errors must not permanently disable JIT.
  Use structured status where available rather than parsing human-readable text.

The public meaning of `jit=True` and explicit TCC/CC selection remains best effort.
Neither requests a new strict "JIT must succeed" execution mode.

## Task C: remove TCC's dependency on filesystem caching

- [ ] Refactor `dsl_try_prepare_jit_runtime()` so the native TCC branch runs before
  all CC-specific disk-cache preparation and probing.
- [ ] Keep common eligibility checks, the runtime-disable setting, and applicable
  in-memory negative-cache handling ahead of backend dispatch.
- [ ] Ensure the TCC branch does not call `dsl_jit_get_cache_dir()`, build artifact
  paths, inspect disk metadata, read cached shared libraries, or create directories.
- [ ] Retain successful TCC states for their compiled program's lifetime and
  continue releasing them through the normal program cleanup path.
- [ ] Keep CC's existing cache keys, metadata validation, shared-library loading,
  process-local handle reuse, and persistent artifacts on the CC route.
- [ ] Ensure a denied TCC attempt proceeds directly to the interpreter, including
  when a system compiler happens to be installed.
- [ ] Apply consistent diagnostic handling to the existing Windows/nonhost TCC
  dispatcher without introducing host filesystem caching into that path.
- [ ] Confirm the macOS TCC path also bypasses disk-cache setup; memfd allocation
  itself is Linux-specific.

## Task D: integrate dependencies and verify Python dispatch

- [ ] Add the fixed minicc revision to `miniexpr/cmake/MiniexprFetchDeps.cmake`
  once available, and verify the sub-build produces the intended libtcc artifact.
- [ ] Add the updated miniexpr revision to Python-Blosc2's `CMakeLists.txt`.
  Use existing local source overrides for development across the repositories.
- [ ] Exercise `arange`, `linspace`, lazy expressions, DSL `lazyudf`, and supported
  reductions through their real Python/Cython dispatch paths.
- [ ] Confirm constructor `jit` and `jit_backend` kwargs reach evaluation rather
  than storage constructors. Validate both omitted settings and explicit settings.
- [ ] Verify normal Python operations do not raise or issue warnings solely
  because the requested native JIT backend is unavailable.
- [ ] Confirm wheels contain the updated libtcc and introduce no new system
  compiler, libselinux, or newer-libc requirement.
- [ ] Add a release-note entry linking #730 and describing filesystem-independent
  TCC execution and quiet interpreter fallback.

## Task E: correct the generated allocation declaration

The investigation exposed an Apple Clang warning for generated
`extern void *malloc(unsigned long long)`: that target's `size_t` is
`unsigned long`.

- [ ] Replace the hardcoded declaration in miniexpr's C generator with one using
  the actual target size type. Use compiler-provided type information or the
  established target-aware generation mechanism.
- [ ] Preserve header-independent TCC compilation and verify LP64, LLP64, and
  wasm32 type choices. Matching width alone does not make C prototypes compatible.
- [ ] Check related generated allocation declarations and casts for the same
  narrow issue, and advance the code-generation cache version if required.
- [ ] Verify generated code compiles without the reported warning under Clang and
  GCC, and still compiles through bundled TCC.

## Task F: document JIT options in the API reference

This is a required deliverable of #730. Users should be able to discover JIT
controls from the constructor and computation reference pages, including options
currently accepted through `**kwargs`.

### F1. Add a canonical reference page and entry-point links

- [ ] Add `doc/reference/jit.rst` with a stable cross-reference label and include
  it in `doc/reference/index.rst`.
- [ ] Link it from `doc/reference/ndarray.rst`, `doc/reference/lazyarray.rst`,
  `doc/reference/dsl_syntax.md`, and applicable reduction documentation.
- [ ] Explicitly document `jit` and `jit_backend` in the `arange()` and
  `linspace()` docstrings in `src/blosc2/ndarray.py`; their `**kwargs` descriptions
  must not imply that only `empty()` storage parameters are accepted.
- [ ] Audit and align the `LazyArray`/`LazyExpr`/`LazyUDF` compute and `lazyudf()`
  documentation, plus relevant reduction entry points, with their actual support.
- [ ] Correct the DSL backend overview to include CC and automatic native
  interpreter fallback. Show backend settings on supported constructors or
  `compute()` calls; do not suggest passing arbitrary JIT kwargs to `__getitem__`.

### F2. Document API semantics precisely

The reference must cover:

- `jit=None`: the API's default policy. Distinguish DSL kernels and constructors
  implemented using them from plain expressions; do not promise every default
  expression evaluation is JIT-compiled.
- `jit=True`: request/prefer JIT, including eligible plain-expression auto-lifting;
  interpreter fallback remains available when compilation cannot be used.
- `jit=False`: request the supported non-JIT route, subject to documented
  environment overrides on the entry points that implement them.
- `jit_backend=None` and `"tcc"`: bundled compiler, low startup cost, no persistent
  TCC binary cache, and Linux memfd-backed execution without JIT artifact files.
- `jit_backend="cc"`: system compiler and optimized generated code, persistent
  shared-library caching, filesystem requirements, and best-effort fallback.
- `jit_backend="js"`: existing WebAssembly/Pyodide-only support and eligibility
  rules. Native selection of this backend remains an API error.
- The distinction between backend selection, `fp_accuracy`, and
  `strict_miniexpr`. The latter is not a "require successful native JIT" flag.
- Current backend/target limitations, including TCC floating-point-mode support
  and targets that use the interpreter because a native compiler backend is absent.
- Normal fallback is quiet and may reduce performance. Successful computation by
  itself is not proof that JIT ran; use native tracing or `validate_dsl_jit()` on a
  supported DSL kernel to inspect availability.

### F3. Document environment controls and their scope

Audit actual call paths and value parsing before writing the final table. In
particular, `_jit_from_env()` currently applies `BLOSC_ME_JIT` through
`LazyExpr.compute()`; do not describe it as a universal constructor override.
Document established precedence accurately and add focused checks where settings
interact. Any consistency fixes must be explicit, tested behavior changes.

| Control | Required reference explanation |
| --- | --- |
| `BLOSC_ME_JIT` | Recognized enabling values and backend values; entry-point scope and precedence over supported per-call settings |
| `ME_DSL_JIT=0` | Disable miniexpr runtime JIT, including when Python requested it |
| `ME_DSL_TRACE=1` | Native codegen/runtime diagnostics, actual builds, cache hits, and fallback reasons on stderr |
| `BLOSC_ME_JIT_TRACE=1` | Python compute-engine routing on supported paths; distinct from native JIT success reporting |
| `ME_DSL_JIT_DEBUG_CC=1` | Opt into the external compiler's output |
| `CC`, `CFLAGS` | Select/configure the explicitly requested system compiler backend |
| `TMPDIR` | CC cache root and its default per-user location; no TCC artifact/cache dependency after this work |
| `ME_DSL_JIT_TCC_OPTIONS` | Options for generated-code compilation, not build flags that reconfigure libtcc's allocator |
| `ME_DSL_JIT_POS_CACHE` | Applicable process-local cache control; disabling it does not disable persistent CC cache reuse |

Do not document `BLOSC_ME_JIT=0` as a global disabling switch: its current parser
does not implement that meaning. Include other miniexpr-specific controls only
after verifying their scope, precedence, and supported status.

### F4. Add concise examples and troubleshooting guidance

- [ ] Show default constructor use, `jit=False`, explicit TCC, and explicit CC:

  ```python
  a = blosc2.linspace(0, 10, 10_000)
  a = blosc2.linspace(0, 10, 10_000, jit=False)
  a = blosc2.linspace(0, 10, 10_000, jit=True, jit_backend="tcc")
  a = blosc2.linspace(0, 10, 10_000, jit=True, jit_backend="cc")
  ```

- [ ] Show a plain lazy expression using `compute(jit=True)` and a DSL kernel
  configured through `lazyudf()`. Follow the repository's result-type conventions:
  `expr[:]` for values and `expr.compute()` for an `NDArray`.
- [ ] Show one trace-enabled shell command and representative messages for a
  successful TCC build, a CC disk-cache hit, and interpreter fallback.
- [ ] Explain that a fresh Python process can reuse a CC binary. Cache-cold
  benchmarking needs a fresh CC cache directory, and compilation, first library
  loading, Python startup, and array execution are separate costs.
- [ ] Explain SELinux and `noexec` outcomes in terms of supported execution and
  fallback. The ordinary solution must not require users to change system policy.
- [ ] Build the reference docs, check cross-references and generated signatures,
  and execute the new examples in the supported environment.

## Validation plan

### 1. Native allocator tests in minicc

Extend the existing libtcc API test coverage and its build integration. Use
test-local fault injection rather than adding public runtime failure switches.

- Compile, relocate, and execute a small function; verify its result and mutable
  global/data behavior through the RX/RW layout.
- Repeat allocation, execution, and destruction to detect descriptor and mapping
  leaks. Exercise independent states and existing multithreaded test coverage.
- Inject failure at memfd creation, resize, reservation, RX mapping, and RW
  mapping. Check recoverable status, original diagnostics, and complete cleanup.
- Test unsupported executable-memfd flags versus genuine permission denial, and
  an unavailable syscall. Verify that no ordinary temporary file is created.
- Exercise boundary/overflow checks without attempting enormous real allocations.
- Verify executable mappings are not RWX. Validate ARM64 code publication on a
  suitable Linux runner, rather than relying only on x86 cache coherence.

### 2. Backend and fallback tests in miniexpr

Extend `tests/test_dsl_jit_runtime_cache.c`, relevant DSL tests, and focused
libtcc-backend tests as needed.

- TCC succeeds with an unusable `TMPDIR` and without a system compiler on `PATH`.
  A regular file used as `TMPDIR` is a useful deterministic invalid-directory case.
- Instrument filesystem-cache helpers to confirm TCC neither creates nor probes
  cache artifacts, including when a directory containing old kernels exists.
- Capture default stdout/stderr on denied TCC relocation and denied CC loading;
  verify correct interpreter results and no unsolicited diagnostics.
- Enable tracing in a fresh process and verify backend, phase/reason, and explicit
  fallback reporting. Test the external compiler's debug-output opt-in separately.
- Verify `jit=False` makes no native JIT attempt and `jit=True` permits fallback.
- Test compiler absence, failed compilation, and failed loading independently for
  CC. Preserve successful cold-build and cross-process disk-cache reuse tests.
- Verify failed states and callback contexts are released; test bounded retry
  behavior without poisoning unrelated kernels or backends.
- Confirm malformed DSL and genuine interpreter errors still surface normally.

### 3. Python integration and reference examples

Use existing suites such as `tests/ndarray/test_jit.py`,
`tests/ndarray/test_jit_dsl_dispatch.py`, and
`tests/ndarray/test_dsl_kernels.py`, with focused new cases where necessary.

- Check `arange`, `linspace`, a plain expression, and a control-flow DSL kernel
  against expected results under successful JIT and forced backend failure.
- Cover omitted settings, explicit `jit=True`/`False`, and explicit TCC/CC backend
  selection. Verify documented environment precedence on each applicable API.
- Use subprocesses for loader/allocator failures so unintended process termination
  is detected. Disable incidental Python bytecode writes when checking JIT files.
- Assert actual TCC JIT success in the positive case, not just successful output
  that could have come from the interpreter. Test fallback explicitly as well.
- Run the documented API examples and relevant ndarray doctests. Check the
  generated `arange`/`linspace` reference displays both JIT controls.

### 4. Real platform and policy coverage

| Environment | Required observation |
| --- | --- |
| RHEL 10, SELinux enforcing, policy permitting executable memfd mappings | Installed wheel executes a real TCC kernel successfully |
| SELinux enforcing policy denying the requested executable mapping | Correct interpreter result, quiet default output, explanatory opt-in trace |
| Linux with a `noexec` temporary mount | TCC succeeds independently of that mount when memfd execution is allowed |
| Linux with unusable temporary/cache storage | TCC has no filesystem-cache dependency |
| Linux with memfd unavailable or denied by the environment | Quiet, recoverable interpreter fallback |
| Linux x86_64 and ARM64 | Correct relocation, execution, cache synchronization, and cleanup |
| macOS and supported Windows native targets | In-memory TCC execution and quiet fallback still work |
| Existing interpreter-only targets and WebAssembly | Existing backend policies and validation behavior remain supported |

Run SELinux-specific cases on an enforcing Linux host or VM. An ordinary container
on a host without enforcing SELinux cannot establish this behavior. Use targeted
syscall/mapping inspection to verify memfd use and absence of JIT artifact writes;
distinguish normal shared-library reads from generated-file activity.

### 5. Performance checks

- Measure small fresh kernels and repeated construction, plus a representative
  large array, to check memfd setup and code publication overhead.
- Compare TCC before/after and CC cache-cold/cache-warm runs separately.
- Include more than a scalar ramp: control flow, indexed kernels, and math calls
  can have different compile and execution costs.
- Measure repeated denied attempts to ensure negative caching bounds overhead.
- Report results without timing-based correctness assertions or changing the
  default backend based on this narrow benchmark set.

## Implementation order and completion checks

1. Implement and test minicc's Linux allocator and build configuration.
2. Implement diagnostic capture and filesystem-independent TCC dispatch in miniexpr.
3. Verify CC fallback behavior and fix the generated allocation declaration.
4. Update dependency pins and exercise the integrated Python paths.
5. Complete reference documentation, docstrings, examples, and release notes.
6. Validate the installed Linux wheel under enforcing SELinux and run platform
   regressions before considering the issue resolved.

Use the `blosc2` conda environment for all Python, test, and build/install commands.
Run relevant native C tests in each dependency, focused Python tests first, and
then the project's required default suite. Build docs with Sphinx and check links
and warnings.

### Acceptance checklist

- [ ] The default wheel needs neither a system compiler nor libselinux.
- [ ] Linux TCC uses memfd-backed RW/RX allocation with correct lifetime handling.
- [ ] TCC JIT creates no source, binary, metadata, or cache-directory artifacts.
- [ ] Invalid temporary/cache storage does not prevent a permitted TCC JIT.
- [ ] Policy-denied TCC and CC execution falls back quietly and computes correctly.
- [ ] Debug tracing explains why JIT was unavailable and identifies fallback.
- [ ] Real input errors retain their documented behavior.
- [ ] CC still supports explicit selection and persistent cache reuse.
- [ ] The reported generated `malloc` warning is resolved.
- [ ] `arange`, `linspace`, and computation reference pages expose the JIT options,
  defaults, backend requirements, environment precedence, and fallback semantics.
- [ ] Linux policy tests and native platform regressions cover actual JIT success
  as well as interpreter fallback.
