# Portable DSL kernels and language-specific authoring frontends

**Status:** Proposal for later consideration; no implementation or API commitment.

## Motivation

Python authoring conveniences make DSL kernels pleasant to write, but their source
may depend on Python-specific rewriting, captured globals, or pandas conventions.
When other languages support Blosc2 lazy arrays, they should not need Python to
load or execute a kernel authored in Python.

The guiding principle is **portable execution artifacts, flexible authoring
frontends**. Separate the canonical DSL understood by miniexpr from the authoring
dialects offered by Python and future Rust, Julia, JavaScript, or other frontends.

## Current boundary

Chained comparisons now belong to the native miniexpr DSL front end. C callers can
pass their raw syntax inside DSL kernels, with single operand evaluation and
short-circuiting. Docstrings, semicolons, multiline expressions/comments, `pass`,
and numeric literal conveniences are also native syntax.

Five convenience families still depend on Python rewriting:

| Python authoring syntax | Canonical native representation |
| --- | --- |
| String methods such as `s.lower()` | `lower(s)` |
| String membership: `x in s`, `x not in s` | `contains(s, x)`, optionally negated |
| Two-part split unpacking: `a, b = s.split(sep, 1)` | Two `split_part(...)` assignments |
| Pandas row access: `row["column"]` | Named kernel parameters and a column mapping |
| NumPy calls such as `np.sin(x)` and `np.maximum(a, b)` | Native calls such as `sin(x)` and `fmax(a, b)` |

Python also specializes captured scalar constants. This is another preprocessing
dependency, rather than a syntax feature. JavaScript emission has its own lowering
for comparison chains; that is a backend adapter, not a Python dependency of native
execution.

## Proposed architecture

### 1. Specify a canonical, language-independent DSL

Use native miniexpr DSL syntax and function semantics as the portable contract.
Document the boundary explicitly: an authoring frontend may accept more syntax,
but exported artifacts must conform to the canonical language.

Specify evaluation order, short-circuiting, dtype conversions, numeric precision,
string/bytes behavior, and reduction restrictions. Portability must not imply
Python's arbitrary-precision integers or promise identical floating-point bits
across backends and accuracy modes.

Keep the DSL kernel language distinct from miniexpr's classic expression API.

### 2. Normalize at export time, not at load time

Provide a future export step that resolves frontend-specific syntax and implicit
dependencies into a portable kernel. A foreign-language loader must not import
Python, reconstruct a Python function, inspect its globals, or execute frontend
rewriters.

Normalization must preserve supported semantics, including operand evaluation
count and order. For example, lowering split unpacking must not evaluate a
nontrivial subject or separator twice merely because two parts are extracted.
Unsupported or ambiguous constructs should fail export with actionable errors.

The existing `kernel.dsl_source` is useful source material, but should not be
treated as a complete portable artifact: bindings and frontend metadata may still
be needed.

### 3. Make the portable kernel self-describing

An artifact should contain, or explicitly reference:

- Canonical DSL source and its entry point.
- Ordered input names and language-neutral dtype descriptions.
- Output type and any required shape, rank, or index-symbol constraints.
- Captured scalar constants as explicit, typed bindings.
- Any column-to-parameter mapping, without requiring pandas objects.
- A DSL language version, an artifact schema version, and required capabilities.
- Semantic execution requirements, such as floating-point accuracy expectations.

Separate required semantics from optional execution preferences. Compiler paths,
cache directories, diagnostic toggles, and host-specific JIT settings should remain
local runtime configuration rather than requirements embedded in portable kernels.

The container format is intentionally undecided. Evaluate a standalone manifest
and integration with existing Blosc2 persistence rather than introducing a new
storage format prematurely. Do not store generated machine code as the portable
representation.

Typed bindings need a defined encoding for integer widths, floating-point values
(including non-finite values), strings, and bytes. Do not rely on Python `repr`,
NumPy dtype objects, or pickle as the cross-language contract.

### 4. Offer an optional strict portability mode

A future authoring/validation option could reject frontend-only syntax and implicit
captures rather than normalize them. For example:

> Use `lower(s)` instead of `s.lower()` for portable DSL source.

Strict mode would help users write source that can be copied directly into another
language's runtime. It should be opt-in; existing convenient Python authoring must
remain supported.

No particular option name or API signature is proposed yet.

### 5. Validate independently of Python

Treat export success and target compatibility as separate checks. An artifact can
be valid canonical DSL while requiring capabilities absent from a particular
runtime or backend.

Validate schema, language version, input bindings, and required capabilities before
execution. Report unsupported requirements explicitly. Where an interpreter can
honor the required semantics, lack of native JIT should retain the existing
best-effort fallback policy.

External C functions or closures need explicit handling: either reject them for a
self-contained export or describe stable symbolic function requirements that a
loader must register. Never serialize process-local pointers or Python closures.

Portable does not mean safe to execute from an untrusted source. Define resource
limits and trust requirements separately; an artifact must not authorize loading
arbitrary libraries, running shell commands, or executing Python during loading.

## Which conveniences should become native?

- **NumPy aliases and pandas row access:** keep frontend-only. They are ecosystem
  integrations, not necessary parts of the language-independent DSL.
- **String methods, membership, and split unpacking:** possible future native
  syntax extensions, but not required for portability if export normalizes them.
  Decide based on usefulness across languages and maintenance cost.
- **Captured constants:** represent them as explicit bindings in the artifact.
- **General language constructs:** consider native implementation when they carry
  broadly useful semantics, as with chained comparisons.

Avoid moving conveniences into miniexpr solely to eliminate Python rewrites. The
goal is a stable portable contract, not reproduction of the entire Python surface.

## Phased work

1. **Inventory and specification:** document native syntax, frontend extensions,
   implicit dependencies, and the initial portable capability set.
2. **Artifact design:** agree on versions, schemas, typed bindings, and whether
   portable kernels are standalone or attached to persisted lazy arrays.
3. **Python export prototype:** normalize conveniences and resolve scalar captures
   without changing existing evaluation APIs. Add optional strict validation.
4. **Native loading and execution:** implement a C-facing validation/loading path
   that accepts the artifact without Python preparation.
5. **Cross-language conformance:** establish a reusable corpus before adding further
   language frontends and backend capability profiles.

Exporting an entire persisted lazy computation graph is a separate follow-up. A
portable kernel contract alone does not specify array references, storage access,
graph execution, or remote authentication.

## Validation and acceptance criteria

- Export from Python, then load and execute from a standalone C process without
  importing Python or applying Python-side preparation.
- Cover all five convenience families, explicit scalar bindings, column mappings,
  and kernels already written in canonical syntax.
- Check evaluation order, single evaluation, short-circuiting, numeric boundaries,
  string/bytes behavior, and documented reduction rules.
- Compare interpreter, TCC/CC, and applicable JavaScript results under declared
  capability and accuracy profiles; do not require unavailable backend features.
- Reject unresolved globals, malformed bindings, unsupported versions or
  capabilities, missing external functions, and unsafe loading instructions.
- Verify that frontend implementation changes do not break previously exported
  artifacts within the promised compatibility policy.
- Preserve current Python authoring and quiet best-effort JIT fallback behavior.

## Open decisions

- Minimum initial dtype/function capability set and capability naming.
- Canonical source normalization and whether source maps accompany diagnostics.
- Artifact encoding, storage location, and typed scalar representation.
- Version compatibility and migration policy.
- Whether runtime shape constraints are fixed or parameterized.
- Registration and versioning of optional external functions.
- Semantic compatibility rules for floating-point accuracy and backend differences.

These decisions should be reviewed before implementation; this plan records the
direction without expanding the current DSL API or persistence contract.
