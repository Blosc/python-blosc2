# Portable DSL v0.1: first implementation plan

**Status:** Implementation plan for the `portable-dsls` branch. Proposed API names
and wire-format details below remain experimental until the first implementation
and conformance tests establish them.

### Implementation progress

The first raw-source slice is implemented across Python-Blosc2 and `../miniexpr`:

- Draft native profile: `../miniexpr/doc/dsl-spec/0.1.md`.
- Native-only Python constructor: `blosc2.DSLKernel.from_source(source)`, with no
  Python `exec`, captures, frontend rewriting, or Python-function fallback.
- Shared finite-float64 fixtures and standalone C runner:
  `../miniexpr/tests/portable-dsl/`, covering branches, chained comparisons,
  bounded loops, and reordered native input bindings.
- Native compiler selection now honors explicit source pragmas ahead of the
  `ME_DSL_JIT_COMPILER` environment default, with regression tests.
- Interpreter-only and bundled-TCC build configurations exercise the corpus;
  required-JIT tests reject silent fallback.

The reviewed native baseline is miniexpr revision
`a1f845b590582a85e5fac04184f3f4758e7ca9a5`. Python-Blosc2's existing dependency
pin is unchanged; its installed extension and the authoritative native checkout
are tested independently. Integrating the updated native dependency remains
necessary before claiming the precedence change in distributed Python builds.

This is **not yet the full v0.1 completion gate**: the wider dtype/semantic audit,
artifact packaging/integration and the remaining semantic audit
are still pending. The raw-source constructor accepts native DSL, not only the
draft portable subset. No new dependency or persisted artifact format has been
introduced.

Native profile validation is now implemented as `me_validate_portable_dsl()` in
`../miniexpr/src/dsl_portable.c`, with a structured status and optional diagnostic.
It filters native parsed source, checks an explicit homogeneous-input signature,
then compiles with JIT disabled without evaluating a kernel. The corpus runner
uses it before execution. `blosc2.validate_portable_dsl()` exposes the same check
through the extension, with no Python parser/rewriter fallback; older native builds
report `runtime_unsupported`. The local editable environment is rebuilt against
the authoritative checkout via `FETCHCONTENT_SOURCE_DIR_MINIEXPR`; the repository
dependency pin remains unchanged. The draft spec now documents the exact initial
feature filter and distinguishes validation from runtime constraints, backend
compatibility, numerical-accuracy certification, and trust/resource policies.

Validation for this slice passed: 137 full native tests, 70 conformance/validation
tests with bundled TCC disabled, 24 AddressSanitizer checks, and 210 focused Python
tests with standalone native-runner comparisons enabled. Ruff and diff whitespace
checks passed; the native builds emitted no new compiler warnings.

P3 now has a candidate artifact schema in `../miniexpr/doc/dsl-spec/artifact-0.1.md`
and a separate optional `miniexpr_artifact` C adapter. With user approval, it uses
yyjson 0.12.0 pinned to `8b4a38dc994a110abaec8a400615567bd996105f`; the raw compiler
does not fetch/link yyjson when `MINIEXPR_BUILD_ARTIFACT` is OFF (the default).
The adapter validates strict JSON structure and duplicate keys, versions and
capabilities, typed scalar encodings, entry point and exact binding coverage,
and portable-profile compilation. It owns source/names/scalars, binds runtime
buffers by name, broadcasts constants in bounded tiles, and separates load,
binding, native evaluation, and allocation failures. Required strict FP semantics
are made explicit on an internal source copy when no FP pragma is supplied;
retained compiler pragmas remain preferences rather than compatibility requirements.
The hand-authored affine artifact runs in the standalone C runner without Python.
The schema/profile remain experimental; native adapter packaging and the remaining
semantic audit are still pending. P4 integration is described below.

P3 validation passed: 143 full native tests, 75 conformance/validation/artifact
checks with bundled TCC disabled, 29 AddressSanitizer checks, and 210 focused
Python regression tests. A fresh static-only raw-compiler build without artifact,
SLEEF, TCC, or Accelerate dependencies passed 70 conformance/validation checks;
yyjson was not fetched. The adapter and tests also passed `-Wall -Wextra -Werror`
syntax checks. A test-only deprecated `sprintf` warning was fixed; final native
builds emitted no new compiler warnings.

P4 now exposes `DSLKernel.export()` and `PortableKernel.from_json()` /
`evaluate()` in Python. Export uses the unspecialized author's DSL source and
snapshots exact supported scalar globals/closure values into collision-free
parameters and typed constants. Explicit parameter constants and capture dtype
overrides are supported; no user conversion hooks or kernel calls are used.
Input/output dtypes are explicit and portable native validation plus the C loader
validate every exported artifact. Import calls that same loader through an owned
Cython handle, without Python source/JSON parsing or frontend preparation.
Runtime input names/dtypes/shapes are checked, host storage is adapted, and native
constant broadcast/evaluation is reused. NumPy-alias rewriting now excludes local
and closure shadowing so export cannot silently reinterpret a local attribute call.
The extension links the adapter only if the optional native target is available;
disabled/older builds report unsupported artifact operations. Local builds use
the source override and `MINIEXPR_BUILD_ARTIFACT=ON`; the repository dependency pin
is unchanged because native dependency publication/integration is still pending.
Artifact execution is currently eager NumPy output, not lazyudf/container integration.

P4 validation passed: 240 focused Python tests with the adapter enabled, including
Python export -> standalone C execution, fresh-process import without the author,
typed scalar boundaries/NaN payloads, snapshot/collision/closure handling, storage
adaptation, native diagnostics, and concurrent interpreter/TCC evaluations.
An actual adapter-disabled editable build passed 212 tests with 28 artifact-only
cases skipped and verified `NotImplementedError` without a Python fallback.
The environment was restored to the adapter-enabled local build. The native
conformance/profile/artifact selection passed 99 checks; Ruff, the standalone
example, and whitespace checks also passed.

The next conformance slice expands the shared corpus with explicit
input/output dtypes and expected compile/evaluation outcomes. It covers all five
candidate dtypes, exact int64 boundaries and comparisons above `2**53`, signed
zero and non-finite identity, Boolean-result semantics, bounded loop control,
small finite casts, and sampled `sin` results. Integer fixtures are decoded and
compared without a floating-point intermediate. CTest also generates CC variants
that change only the compiler preference, uses build-local JIT caches, and applies
timeouts. Missing-return kernels now have runtime JIT support: successful paths,
mixed successful/failing elements, loop returns, and hybrid vector cleanup are
tested under required-JIT policies. Semantic missing-return errors propagate
without interpreter retry; generic backend failures retain existing fallback.
This advances P2 coverage but does not freeze
overflow, general promotions/conversions, or global transcendental accuracy rules.

The numeric stabilization slice adds 25 shared exact output-conversion fixtures
(the full supported 5×5 dtype matrix), including representable cast boundaries,
int64 rounding, signed zero, subnormals, and non-finite identity/truth values.
Native integer-literal validation now checks normalized digits exactly so
`2**53 + 1` cannot bypass the profile limit through double rounding.
The audit found unresolved integer-valued division promotion and float32 math
intermediate/condition discrepancies across JIT and linked interpreter paths.
Shared reproducers and strict expected-failure artifact tests preserve the gaps;
they block freezing rather than redefining portable semantics. Details are in
`../miniexpr/doc/dsl-spec/numeric-audit-0.1.md`. Native publication, dependency
integration, and the rest of the semantic matrix remain pending.

This slice passed 328 focused Python tests with 10 strict expected JIT failures,
202 native portable/profile checks in the TCC-enabled build, and 55 selected
interpreter-only portable/profile/artifact checks. Ruff and whitespace checks
passed, and the native builds emitted no new warnings.

## 1. Goal and scope

Establish a small, versioned miniexpr kernel language that executes independently
of Python, then demonstrate Python export and standalone C import of a portable
kernel artifact. Deliver this vertical slice before deciding how far to extend
the language, its packaging, or other language bindings.

This implements the first stage of [the broader proposal](portable-DSLs.md), with
two refinements:

- Native execution and cross-language conformance come before artifact design.
- Compiler pragmas remain in portable source and override local compiler defaults,
  while retaining best-effort fallback.

Python-like syntax is compatible with a language-independent standard. The
contract is a documented grammar and semantics, implemented without Python, not
the absence of syntax familiar to Python users.

### Two independently useful deliverables

1. **Portable source:** identical source plus an explicit signature can execute
   through miniexpr's C API and Python-Blosc2.
2. **Portable artifact:** a small manifest carries that source, signature, typed
   constants, and semantic requirements between environments.

The first deliverable must not depend on completing the second.

### Initial required scope

- One named kernel, scalar-per-element execution, one scalar result per element.
- An explicit ordered input signature and output dtype.
- A documented subset of existing native numeric and Boolean syntax.
- Native interpreter execution as the mandatory baseline.
- TCC execution where supported, with tests proving which backend actually ran.
- Raw-source construction from Python without reconstructing a Python function.
- Python export of a kernel with a captured numeric scalar constant.
- A standalone C loader that consumes the exported artifact without Python-side
  preparation at load time.
- Version checks, typed binding validation, and useful unsupported-feature errors.

### Deferred beyond the initial completion gate

- Complete string/bytes support and normalization of every authoring convenience.
- New method syntax, membership syntax, or split unpacking in miniexpr.
- Reduction kernels, index symbols, arbitrary rank constraints, and broadcasting.
- Complex, structured, object, datetime, nullable, and variable-length dtypes.
- External C functions, closures, and host callbacks in portable artifacts.
- `print` and other observable block-level side effects.
- Whole lazy computation graphs, array storage references, and remote access.
- Integration into Blosc2 persistence containers and generated-code caches.
- Mandatory JavaScript, Rust, or Julia implementations.

Existing functionality outside this subset continues to work through existing
APIs; it simply is not certified by the initial portable profile.

### Baseline after reviewing the updated miniexpr checkout

The authoritative checkout is `../miniexpr`. Its existing `doc/dsl-syntax.md`
is already titled **miniexpr DSL Syntax (Canonical Reference)** and documents
native syntax and several exact semantic/error rules. `doc/dsl-usage.md` already
demonstrates standalone C compilation/evaluation and documents JIT controls.
This work standardizes a portable subset of that language; it does not introduce
native DSL execution from scratch or replace the existing reference.

In particular, the full language already supports reductions, reserved ND symbols,
`print`, and string-valued kernels. Those features need explicit profile exclusions
in v0.1 rather than descriptions implying they are absent from miniexpr.

## 2. Ownership and implementation boundaries

### miniexpr owns the portable language

Keep the normative specification, native validation, interpreter behavior, and
language conformance corpus with miniexpr. Its implementation is the initial
implementation of the specification, not an implicit substitute for one.

Keep the kernel DSL distinct from miniexpr's classic expression API. Do not
silently expand or change that API when implementing portable kernel support.

### Documentation layout and versioning

Use these paths in the authoritative miniexpr repository:

- `doc/dsl-syntax.md`: retain the current canonical reference for the full native
  DSL. Extend or clarify it where the semantic audit finds omissions; preserve
  its existing role and links from other documentation.
- `doc/dsl-usage.md`: retain the practical C API and runtime configuration guide.
- `doc/dsl-spec/0.1.md`: add the normative **portable profile** specification,
  identifying the admitted grammar, dtype/function matrix, semantic requirements,
  and explicit exclusions from the full DSL.
- `doc/dsl-spec/artifact-0.1.md`: add the versioned manifest and typed-binding
  specification once the portable-source milestone works.

During drafting, derive rules from the canonical reference rather than inventing
a second grammar. At publication, make the versioned profile's normative rules
self-contained or pin incorporated material to an immutable miniexpr revision.
A moving `doc/dsl-syntax.md` link must not silently change the meaning of a
published v0.1 artifact. Link from the current canonical reference to the profile
and explain full-language versus portable-profile support.

Native work must be made in the authoritative miniexpr checkout and integrated
through the dependency mechanism used by Python-Blosc2. Build-tree copies under
`build*/_deps/` are inspection/build artifacts, not the source of record.

### Python-Blosc2 owns authoring and integration

Use `src/blosc2/dsl_kernel.py` as the integration point for current kernel
authoring, normalization, and capture handling. Inspect the existing extension
bridge before adding new native entry points. If artifact handling merits its
own module, use a non-underscored name such as `dsl_portable.py`.

Separate the native specification from Python guidance in
`doc/reference/dsl_syntax.md`. Link to the normative miniexpr reference and clearly
identify decorator behavior and frontend-only conveniences.

### Keep the layers independently testable

1. Parse and validate native source against the portable profile.
2. Validate signature and typed bindings.
3. Check target semantic capabilities.
4. Compile or interpret using the selected execution preference.
5. Execute on caller-supplied buffers.

Artifact decoding should feed these same operations. It should not introduce a
second language validator or execute frontend rewriting at import time.

## 3. Define the initial language profile

Use separate identifiers for the language version and artifact schema version.
Suggested initial values are `0.1` for each; these are experimental compatibility
identifiers, not miniexpr package versions.

### Candidate core, to freeze after the native audit

- One top-level `def`, positional named parameters, indentation-based blocks.
- Local assignments, supported compound assignments, and `return`.
- `if`/`elif`/`else`, `for ... in range(...)`, bounded test cases for `while`,
  `break`, `continue`, and `pass`.
- Native numeric literals, arithmetic, comparisons, chained comparisons, and
  Boolean operators.
- A short, explicit whitelist of native casts and mathematical functions.
- Comments, docstrings, semicolons, and multiline syntax already accepted natively.
- Initially target `bool`, `int32`, `int64`, `float32`, and `float64`; include only
  types and operations whose behavior is specified and verified across the
  interpreter and supported JIT paths. Do not claim unsigned or mixed-type support
  merely because a host API accepts the dtype.

Start the first executable fixture with `float64`; expand to the frozen core
before declaring v0.1 complete. Record the final operator/function/type matrix in
the normative specification rather than relying on this candidate list.

### Semantic decisions required before freezing the profile

Document and test:

- Parameter binding, duplicate names, local assignment and type inference rules.
- Literal typing, promotions, return conversion, and explicit casts.
- Integer overflow, narrowing, signed division, remainder, and division by zero.
- Floating-point rounding assumptions, non-finite values, and signed zero.
- Evaluation order, single evaluation of chained operands, and short-circuiting.
- Boolean-result rules for `and`/`or`, without implying Python operand-return rules.
- Loop bounds, step handling, and errors such as a zero `range` step.
- Missing returns and use of uninitialized locals.
- The scalar-per-element execution model and independence from chunk boundaries.
- Error categories and when errors occur: validation, compilation, or execution.

Carry forward the rules already documented in the native reference, verifying
them rather than treating them as undecided design choices:

- Native parameters match variable names by set membership; caller variable order
  may differ from source parameter order.
- Incompatible local reassignment and inconsistent return dtypes are compile-time
  errors; an executed path without a return fails at runtime.
- A zero `range` step and exceeding the `while` iteration cap are runtime errors.
  Specify the cap and its configuration/portability implications for this profile.
- `//=` lowers to `floor(a / b)`; do not assume Python integer floor-division
  precision without auditing intermediate typing.
- Reserved ND names remain reserved even though index-aware kernels are excluded.
- The full DSL permits reductions at top level and in conditions, but not inside
  `if`/`for`/`while` bodies. The v0.1 elementwise profile excludes reductions in
  all positions, including conditions, to retain chunk-independent semantics.

Audit existing behavior before promising a rule. If interpreter and JIT disagree,
either fix the discrepancy with regression coverage or exclude that operation/type
combination from v0.1. Do not standardize accidental C undefined behavior.

Use a small core capability identifier rather than a capability for every token.
Add separate capabilities only for meaningful optional semantic features. Backend
names such as `tcc` are execution preferences, not language capabilities.

## 4. Compiler preferences and interpreter baseline

Miniexpr includes an interpreter. It also builds bundled libtcc by default on
supported targets, so ordinary users do not need a separately installed compiler.
TCC may be disabled in custom builds or unavailable on a target; Python-Blosc2
currently disables it on Windows ARM64.

For this implementation:

1. Preserve `# me:compiler=tcc|cc` in portable source.
2. An explicit source pragma wins over the local compiler-selection default.
3. With no pragma, use the runtime's existing default selection policy.
4. Preserve existing best-effort fallback when the preferred backend is unavailable
   or cannot compile the kernel. Use the interpreter when it satisfies the required
   semantics.
5. Keep ordinary fallback quiet, while making the actual execution backend
   observable to diagnostics and conformance tests.

Audit the current selection path and codify its fallback order; do not invent a
new fallback chain as a side effect of artifact support. A deliberate local
interpreter-only execution mode remains useful for testing and deployments; it
is distinct from choosing a default compiler.

Reuse the documented `ME_JIT_DEFAULT`, `ME_JIT_ON`, and `ME_JIT_OFF` policies,
including `me_eval_params.jit_mode`, rather than adding a parallel JIT toggle.
Test that explicit `ME_JIT_OFF` prevents JIT even with a compiler pragma: choosing
whether to JIT and choosing which compiler to prefer are separate controls.

Floating-point requirements are independent of backend preferences. Define how
`# me:fp=strict|contract|fast` maps to the artifact's semantic requirements and
which interpreter/JIT modes satisfy them. Reject conflicting declarations rather
than silently allowing one to override the other. Do not promise identical bits
across all mathematical libraries or imply a stronger meaning of `strict` than
the specification defines.

Compiler paths, cache directories, and diagnostic settings remain local. The
artifact contains a compiler preference, never instructions to locate or load an
arbitrary executable or library.

## 5. Raw-source execution before packaging

Build the first vertical slice around source such as:

```text
# me:compiler=tcc
def affine(x, scale, offset):
    y = x * scale + offset
    if y < 0:
        return 0.0
    return y
```

At this stage all three names can be ordinary explicitly typed inputs. Execute
the exact same source through a standalone C program and Python, using the same
input values and declared output type.

Expose or reuse a Python raw-source construction path that accepts source and a
signature. Loading it must not use Python `exec`, fabricate a Python function, or
inspect globals. Final public names should follow the existing kernel API after
inspection; do not add competing constructors unnecessarily.

The C executable must link miniexpr without libpython. Python may launch it as a
test subprocess, but the executable must also run directly from a shell with
prepared fixtures. Native execution requires no Python-generated temporary code.

Start from the existing `me_compile()` / `me_eval()` / `me_free()` flow documented
in `doc/dsl-usage.md`; use `me_compile_nd_jit()` where its explicit JIT controls
are needed. Audit existing parser diagnostics and backend test helpers before
introducing any new native API. The new work here is a reusable conformance runner
and profile validation, not a replacement execution engine.

## 6. Minimal artifact design

After raw-source conformance works, use a standalone UTF-8 JSON manifest as the
first interchange format. Keep it small enough for a native loader and independent
of Blosc2 storage. Reassess container integration after v0.1 works end to end.

### Proposed fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Artifact encoding/structure version |
| `language` | Language identifier and version |
| `requires` | Required semantic capability identifiers |
| `source` | Complete canonical native DSL source, including retained pragmas |
| `entry_point` | Name of the single kernel in the source |
| `inputs` | Ordered runtime input names and language-neutral dtypes |
| `constants` | Explicit named, typed scalar bindings |
| `output` | Output dtype and scalar-per-element execution contract |
| `semantics` | Required floating-point profile and other supported requirements |
| `metadata` | Optional non-semantic producer information |

Freeze required fields and unknown-field handling in a small schema document.
Reject duplicate JSON keys, duplicate bindings, and unknown semantic fields.
Permit explicitly designated informational metadata without interpreting it.
Do not introduce a runtime schema-validation dependency just for this manifest.

### Signature and constants

Use the source's parameter list as the full native parameter order. Every
parameter must be supplied exactly once by either `inputs` or `constants`.
Disallow overlapping names, missing bindings, and unused binding entries.
`inputs` follows source order with constant parameters omitted.

This ordering is an artifact convention, not a new native-language restriction.
The adapter must map names to the `me_variable` array and supply evaluation
pointers in that array's order, as the current native API requires. Include a
reordered-variable fixture to prevent accidental positional-only binding.

For captured constants, export a collision-free explicit parameter and record its
typed value in `constants`. This avoids reliance on implicit native globals or
Python literal formatting. Reuse native scalar binding support if available;
otherwise provide an adapter with defined broadcast and lifetime behavior rather
than exposing an array-sized copy requirement in the artifact contract.

### Typed scalar encoding

- Boolean: JSON Boolean with dtype `bool`.
- Integer: decimal string with explicit fixed-width dtype and range validation.
- Float: exact IEEE-754 bit-pattern hex string with explicit `float32` or
  `float64` dtype and fixed digit count, written most-significant byte first.
- Decode floats into the host representation explicitly; do not rely on native
  endianness or JSON numeric precision.

This preserves large integers, signed zero, infinities, and NaNs in transport.
Transporting a NaN bit pattern does not promise that arithmetic preserves its
payload. Strings/bytes get a separate encoding decision when that profile is added.

### Compatibility

For the experimental implementation, accept only explicitly supported language
and schema versions. Reject unknown versions clearly instead of guessing forward
compatibility. Keep frozen example artifacts as regression fixtures. Once a v0.1
contract is published, change semantics under a new version rather than silently
reinterpreting existing artifacts.

## 7. Python export and import

### Export pipeline

1. Obtain the existing author's source and explicit specialization signature.
2. Normalize supported frontend conveniences using the existing frontend machinery.
3. Discover scalar captures without calling the kernel or arbitrary user callbacks.
4. Resolve each supported capture to an explicit dtype and snapshot its value at
   export time. Require an explicit type when inference would be ambiguous.
5. Replace captures with collision-free bound parameters.
6. Run native portable-profile validation on the resulting source and signature.
7. Derive required capabilities and emit the manifest deterministically.

Do not treat `kernel.dsl_source` alone as the finished artifact. Audit whether
existing normalization or scalar specialization has already discarded type or
capture information; retain that information before lowering if necessary.

The first required example captures a numeric scale constant. After export,
changing the Python global must not change execution of the saved artifact.
Reject unresolved names, arrays/objects as captures, external functions, and
unsupported constructs with the relevant name and source location where possible.

### Import pipeline

Decode and validate the artifact, construct the typed bindings, then use the same
native source path as C. Import must not require the original module, function,
decorators, globals, or frontend rewriters.

Provide a raw-source portable validator first. An optional strict authoring mode
that rejects frontend sugar and implicit captures can build on it later; it need
not block the first export/import milestone.

## 8. Native artifact loader

Implement the loader as a thin adapter adjacent to miniexpr's native API, with
clear ownership of decoded source, binding buffers, compiled handles, and errors.
The language compiler must remain usable without JSON artifact support.

Before choosing JSON machinery, inspect existing dependency policy and available
parsers. Reuse an appropriate existing facility where possible; any new dependency
requires an explicit decision. Avoid a hand-written partial JSON parser that only
accepts the Python exporter's preferred formatting.

Loader stages:

1. Decode JSON with bounded lengths and validate field structure.
2. Check schema/language versions and required capabilities.
3. Decode and range-check typed constants and signature information.
4. Validate entry point and exact binding coverage against native source.
5. Check semantic requirements independently of the compiler preference.
6. Construct an executable handle using the existing interpreter/JIT machinery.
7. Bind caller-provided data and execute; free all owned resources on every path.

Define logical buffers as same-length inputs and output for the initial profile,
with scalar constants broadcast to each element. Specify empty-input behavior,
host dtype conversion, buffer ownership, and binding lifetime. Keep physical
strides, endian conversion, and storage layout in adapters rather than in source.

Distinguish malformed artifacts, unsupported requirements, invalid source,
binding errors, and evaluation errors. Lack of the preferred compiler alone is
not an invalid artifact.

The loader does not provide a sandbox. It must not execute Python, follow artifact
paths to load libraries, or use artifact data as shell commands. General resource
isolation remains a host responsibility rather than a new v0.1 subsystem.

## 9. Conformance corpus and verification

Create language-neutral fixtures containing source, signature, inputs, expected
outputs or error category, and numeric comparison rules. Keep native language
fixtures authoritative in miniexpr and consume the same cases from Python tests;
avoid independently maintained copies with diverging expectations.

Seed the corpus from existing native tests, especially `test_dsl_syntax.c`,
`test_dsl_python_syntax.c`, `test_dsl_literals.c`, `test_dsl_comparisons.c`,
`test_dsl_bool_locals.c`, `test_dsl_guards.c`, and `test_dsl_jit_options.c`.
Reuse their helpers where appropriate and preserve implementation-level coverage;
only promote profile-relevant cases into the shared language-neutral corpus.

Required groups:

- Basic arithmetic, branches, loops, casts, and every admitted dtype/function.
- Chained comparisons, short-circuit guards, and operand evaluation order/count.
- Integer boundaries, promotions, narrowing, and defined arithmetic errors.
- Floating-point boundaries, signed zero, NaNs, infinities, and profile-specific
  tolerances for mathematical functions.
- Signature ordering, captured constants, name collisions, and empty buffers.
- Invalid syntax, missing returns/bindings, unsupported types and capabilities.
- Malformed JSON, duplicate fields, invalid scalar encodings, unknown versions,
  and inconsistent semantic declarations.
- Compiler pragma precedence, no-pragma defaults, and interpreter fallback.
- Frozen artifact import without its originating Python module.

Use expected values derived from the specification, not only agreement between
two implementations. For single-evaluation tests, use native test instrumentation
where pure numeric outputs cannot reveal repeated evaluation; do not require
portable user callbacks just to make the tests observable.

Run all core cases explicitly through the interpreter. Run the applicable corpus
through TCC, and through CC where available. JIT tests must inspect the actual
backend or use an existing no-fallback test mode so an interpreter fallback cannot
masquerade as JIT conformance. Fallback itself has separate tests.

Include at least one configuration with TCC disabled and one normal bundled-TCC
configuration. Exercise representative supported platforms in CI as available.
JavaScript is a later capability-specific target, not a first-version gate.

Python, build, and test commands in this repository must use the `blosc2` conda
environment. Run focused native tests and Python DSL tests as each layer changes,
then the repository's required checks before completion. Keep unrelated tests and
benchmarks outside the implementation loop unless failures justify investigation.

## 10. Ordered work packages and completion gates

### P0 — Audit and freeze a small contract

- Locate authoritative miniexpr sources and dependency integration points.
- Record the reviewed miniexpr revision and use its existing canonical reference,
  usage guide, and native DSL tests as the specification/conformance baseline.
- Inventory native parsing, dtype rules, execution APIs, capture specialization,
  scalar bindings, and backend selection.
- Write the initial normative grammar/semantics and supported operation matrix.
- Record exclusions and resolve interpreter/JIT discrepancies for admitted cases.

**Gate:** a precise candidate profile and first fixtures, with no dependence on
Python behavior to explain native semantics.

### P1 — Execute identical raw source in C and Python

- Add the standalone C runner and Python raw-source integration.
- Execute the affine example and control-flow cases with explicit signatures.
- Add interpreter/TCC backend observability and selection-policy tests.

**Gate:** the same source executes without Python preparation in C and through
Python, and interpreter execution is verified explicitly.

### P2 — Establish the shared core conformance suite

- Expand fixtures across the frozen core syntax and dtype matrix.
- Document numerical comparison rules and runtime error categories.
- Verify interpreter/TCC parity and applicable CC coverage.

**Gate:** all claimed v0.1 core semantics have representative positive, boundary,
and rejection coverage. The portable-source milestone is independently usable.

### P3 — Implement the minimal artifact and C loader

- Freeze the initial manifest and scalar encodings.
- Implement native decoding, binding validation, ownership, and error handling.
- Load a hand-authored artifact and reuse the P1 execution path.

**Gate:** a standalone C executable loads and runs a fixture artifact without
Python and rejects malformed or unsupported artifacts explicitly.

### P4 — Export and re-import a Python-authored kernel

- Implement explicit-signature export and captured scalar parameterization.
- Implement Python import through the native portable-source path.
- Export in Python, execute in C, and compare against specified expected results.
- Verify snapshot semantics, deterministic export, and independence from the
  original Python module.

**Gate:** the complete Python-to-C artifact round trip works for the numeric core.

### P5 — Document, stabilize, and review the first version

- Publish the native reference and short Python/C examples.
- Document pragma precedence, fallback, version handling, and supported dtypes.
- Preserve frozen artifacts and complete focused regression/platform checks.
- Record practical limitations and review the next priorities with the user.

**Gate:** v0.1 meets the checklist below. Further expansion is a separate decision.

## 11. Follow-up candidates after v0.1

Extend the same corpus and manifest rather than redesigning the architecture.

1. **Strings and bytes:** define encodings, lengths, case behavior, substring rules,
   output storage, and dtype descriptions before claiming portable-profile support.
   Start from existing `doc/strings.md` and `doc/dsl-syntax.md`: native string
   locals and returns already work, widths have documented constraints, shorter
   returns are NUL-padded, and string kernels execute on the interpreter rather
   than the JIT. A string profile must not require TCC support to be portable.
2. **Native string methods:** consider a fixed whitelist lowering to existing
   functions, without general object dispatch.
3. **Native membership:** consider `in`/`not in` for explicitly supported
   string/bytes operands, with defined evaluation order.
4. **Frontend adapters:** NumPy aliases and pandas row access remain authoring and
   binding conveniences; add portable column mappings when needed.
5. **Split unpacking:** decide missing-separator, empty-separator, result-count,
   and evaluation-count semantics before lowering to `split_part` calls. Two
   extraction calls do not automatically reproduce Python unpacking behavior.
6. **Additional execution profiles:** reductions, index-aware kernels, other
   backends, and foreign-language bindings, each with matching conformance cases.

Promotion of useful syntax into miniexpr is welcome when it benefits authors in
multiple environments. Eliminating a Python rewrite alone is not a sufficient
reason to enlarge the language.

## 12. First-version acceptance checklist

- [ ] Native versioned specification defines the admitted language subset.
- [ ] Same raw source and signature execute from C and Python.
- [ ] Interpreter passes the full core conformance corpus.
- [ ] Supported JIT paths pass applicable cases with actual backend verification.
- [ ] Compiler pragmas survive export and override local compiler defaults.
- [ ] Best-effort fallback preserves required semantics and remains quiet by default.
- [ ] Explicit typed constants survive a Python export/C import round trip.
- [ ] Saved artifacts execute without the original Python module or Python runtime.
- [ ] Loaders reject unsupported versions/capabilities and malformed bindings.
- [ ] Dtype, numeric, error, and buffer contracts are documented and tested.
- [ ] Existing Python authoring and evaluation APIs retain their behavior.
- [ ] Frozen artifacts and shared fixtures guard the published v0.1 contract.
- [ ] Follow-up scope is reviewed after this slice is working, not assumed upfront.
