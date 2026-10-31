# Portable DSL 1.0: agreed scope and specification work

Status: feature-level scope agreed on 2026-10-07. Version 1.0 is the intended
first public compatibility contract; its detailed specification remains draft
until the acceptance gates below pass. Draft 1.0 is the sole public portable
format/default; the unreleased 0.1 path is retired with unsupported-version rejection.

## Goal

Persist DSL kernels only when their normalized source, typed bindings, and
execution requirements have a language-independent contract. Validate on save
and independently on load. Preserve convenient authoring syntax when it can be
lowered without retaining Python execution or live Python state.

Interpreter-first: missing JIT support is not grounds for excluding a feature.
Select fallback before execution. A valid artifact can require capabilities
which a particular host lacks; that host must reject it explicitly.

## Agreed inclusions

- Real/integer/Boolean arithmetic, mixed numeric inputs, casts, predicates,
  and an explicitly enumerated math-function set, including elementwise `where`.
- Existing branches, loops, local assignments, and loop control.
- ND index/shape symbols with explicit host-supplied logical context.
- Fixed-width Unicode strings/bytes and statically bounded string returns.
- Supported authoring rewrites and immutable typed scalar captures; static
  column references become explicit named bindings.
- Limited block-local reductions: top-level reductions and supported
  control-flow conditions, including Mandelbrot-style `all(...)` early exit.
- Portable kernel persistence with validation after normalization on save and
  independent native validation on load. Unsupported kernels must fail saving;
  no silent Python-specific persistence fallback for this path.
- Interpreter-first semantics; JIT is optional and may fall back before execution.

## Inventory: prioritize semantic difficulty

| Feature | Current implementation evidence | Portability work / difficulty | Agreed feature-level disposition |
| --- | --- | --- | --- |
| Block reductions inside kernels | Native DSL admits top-level reductions and reductions in conditions; rejects reductions inside control-flow bodies (`src/dsl_compile.c`). | **High.** Explicit block grouping, masking, scalar outputs/broadcast, accumulation order, empty blocks and padding. | Include limited reductions over the current evaluation block, including Mandelbrot `all` conditions. Exclude unsupported reductions inside control-flow bodies; no implicit full-array reduction. |
| External functions and closures | Native `me_variable` supports registered C functions/closures. Python authoring allows named calls syntactically; native compilation determines availability. | **Very high.** Function implementation identity/version, state, ABI, registration, and reproducibility cannot be represented by process-local pointers. | Exclude externally registered functions, callbacks, and opaque captured objects. Allow supported builtin functions and typed constant snapshots. |
| Complex arithmetic | Native complex types exist, but Windows rejects them because of complex ABI incompatibilities (`doc/data-types.md`, `doc/windows-limitations.md`). | **High.** Portable representation, branch cuts, non-finite behavior, promotion, and a functioning interpreter across target platforms. | Exclude complex inputs, intermediates, outputs, and constants for 1.0. This is a real reduction from the full native language. |
| Observable side effects | Native DSL implements `print`, restricted to uniform/scalar values. Arbitrary Python I/O is not native DSL. | **High for deterministic behavior.** Ordering and repetition depend on block scheduling, parallelism, partial evaluation, and repeated lazy evaluation. | Exclude `print` and external effects from portable kernels. Do not silently strip them. Host diagnostics remain execution policy. |
| Numeric promotion and conversions | Current DSL supports arithmetic, casts, mixed operands, predicates, and math more broadly than portable 0.1. | **High but essential.** Define intermediate/result types, literal typing, division, casts, predicate rounding, and a bounded supported function list. | Include ordinary real/integer/Boolean numerical use cases; interpreter-first. Do not promise all native numeric combinations automatically. |
| ND index/shape symbols | `_flat_idx`, `_i<d>`, `_n<d>`, `_ndim` already have native synthesis and Python constructor tests. | **Medium-high.** Explicit logical shape, evaluation region, origin/coordinates, slicing/view interpretation, broadcasting and padding rules. | Include a defined ND-context capability; do not expose physical chunk layout as logical semantics. Unsupported host contexts must reject. |
| Fixed-width strings and bytes | Native string locals/returns, width inference, comparisons, concat, transforms, substring and split operations; Python string DSL tests. Interpreter-only today. | **Medium-high.** Typed widths, endianness/code units, constants, first-NUL semantics, truncation, Unicode mapping version and whitespace tables. | Include a bounded fixed-width string/bytes capability. Existing operations are candidates, not all certified yet. |
| Variable-length string representation | Native `me_eval_varlen()` packs evaluated fixed-width values into Arrow-style offsets/data. Intermediates stay fixed-width. | **Medium for an adapter; high for a new language type.** Layout, capacity, offsets and ownership versus dynamically sized intermediates. | Defer a portable varlen ABI/type. Do not reject a fixed-width kernel merely because its host stores the result as varlen. |
| Authoring conveniences | `_NumpyAttrCallRewriter`, `_StringSyntaxRewriter`, `_RowSubscriptRewriter` in `dsl_kernel.py`. | **Low-medium.** Correct normalization, name resolution, evaluation order, typed captures and explicit column mappings. | Include supported normalization: `np.sin(x)`, string methods/membership/split sugar, static `row["col"]`. Dynamic DataFrame operations are not an existing native DSL feature. |
| Persisted graph and storage bindings | `b2objects.py` and `lazyexpr.py` preserve sources and operand references; reconstruction still uses `kernel_from_source`. CTable has DSL-backed column round trips. | **High integration scope.** Kernel portability does not define remote access, authentication, recursive graph execution, or storage identity. | Integrate portable kernel payloads and explicit bindings in the host container. Exclude arbitrary Python-object persistence; do not claim all host graphs/storage protocols are portable. |

## Agreed exclusion list

These are the kernel exclusions for the first public release:

1. **Unsupported reductions inside control-flow bodies and implicit global
   collectives.** Permit top-level reductions and supported reduction conditions,
   including an `if all(...)` condition within a loop. This is distinct from a
   reduction assignment/return inside the body. No implicit reinterpretation of
   block reductions as full-array reductions. Classify operations by semantics:
   elementwise `fmin`/`fmax` are not block reductions.
2. **Externally registered functions, closures, and runtime object dependencies.**
   No function pointers, Python callbacks, opaque object captures, or implicit
   module/global state. Supported scalar captures become immutable typed values;
   arrays/columns become explicit host bindings rather than hidden captures.
3. **Complex-valued computation.** Defer until there is a portable cross-platform
   interpreter contract, independently of JIT availability.
4. **Observable kernel side effects**, including native `print`. Host tracing is
   not a persisted kernel effect.
5. **Non-strict FP semantic modes** (`fast`/`contract`) for the initial contract.
   Strict arithmetic and explicitly documented math-function accuracy are the
   baseline; this does not imply bitwise-identical transcendental results across
   all platforms. Persist semantic requirements, not compiler paths or caches.
6. **Dynamically sized intermediate/output types and arbitrary memory access.**
   Retain statically bounded string widths and explicit output cardinality
   (elementwise or supported block-scalar reduction).
   General gathers/scatters, mutation of inputs, arbitrary array subscripting,
   and dynamically selected columns are not new 1.0 features. Several of these
   are already outside today's DSL; distinguish them from lost capabilities.

Separate exclusions from the *artifact/host scope*: generated machine code,
Python source execution on import, pickled runtime objects, external function
registration protocols, and universal graph/remote-storage execution. Saving a
portable kernel in a LazyArray does not itself make every referenced data source
available in C or another language.

## Block-local reduction contract to specify

The reduction group is the **current evaluation block**, not the full logical
array. Partition-dependent results are permitted; reproducing those results
requires preserving the evaluation partition. The persistence/host integration
must record or unambiguously reconstruct that partition, not silently retile it.
Logical ND coordinates and physical evaluation grouping remain distinct concepts.

The mandatory motivating case is `if all(escape_iter != limit): break` in the
Mandelbrot loop. It is a block-level early-exit optimization, not an array-wide
synchronization primitive. Conformance must test different partitions and show
both this use case and a deliberately partition-dependent reduction result.

The approved policies below settle valid-element/active-mask participation,
empty identities, scalar broadcast, partition preservation, and ordered sums.
Before freezing, complete output-cardinality and per-operation details.
Masked/nested conditions that current implementations cannot
define consistently must be identified explicitly, not accidentally admitted.

## Hard numeric domains: not syntactic exclusions

Overflow, non-finite/out-of-range integer casts, division by zero, and invalid
shift counts depend on runtime values. They cannot generally be detected by a
save-time validator. Do not claim that successful persistence proves their
absence. The agreed rules below require runtime errors for the specified invalid
integer operations and conversions, and IEEE-style floating behavior. Implement
those checks rather than relying on caller preconditions for these cases.

Likewise, unsigned integers, narrower widths, and each math builtin need an
explicit type/function matrix. They are open inventory decisions, not covered
by a blanket claim of full miniexpr compatibility. Real arithmetic remains a
major specification task even with limited reductions and no complex numbers.

## What should not be excluded merely to save JIT work

- Integer-input division, supported mixed numeric inputs, nested casts and
  arithmetic predicates, once their interpreter semantics are defined.
- An agreed set of math builtins and elementwise `where`.
- ND symbols supplied through a defined logical context.
- Fixed-width Unicode/bytes kernels, including bounded string returns.
- Static column access and supported NumPy/string authoring sugar after lowering.

For example, the existing row rewrite accepts a single row parameter used only
as `row[<string literal>]` and records parameter-to-column mappings. It is not a
general translation of arbitrary `df[expr]` indexing.

## Evidence and documentation gaps

- `../miniexpr/doc/dsl-syntax.md`: statements, ND context, string locals/returns,
  registered functions, reductions and print.
- `../miniexpr/doc/strings.md`: UCS4 versus bytes, NUL termination, width bounds,
  Unicode truncation, and the existing varlen output adapter.
- `../miniexpr/src/dsl_compile.c`: actual reduction restrictions; existing
  unsupported combinations are not an additional portable-only limitation.
- `src/blosc2/dsl_kernel.py`: authoring validator and normalization boundaries.
- `tests/ndarray/test_dsl_kernels.py`, `tests/ndarray/test_string_output.py`, and
  `tests/ctable/test_ctable_dsl_columns.py`: ND, string, and persistence use cases.
- `src/blosc2/b2objects.py`, `src/blosc2/lazyexpr.py`: existing persistence paths.

Some references disagree: Python's DSL page still describes reduction-in-body
behavior as silently incorrect, while current native compilation rejects it;
the strings guide's scalar-condition wording also needs reconciliation with
masked string DSL behavior. Resolve against implementation/tests before freezing
normative text. This inventory is a source review, not new feature certification.

## Agreed semantic and compatibility decisions

The following choices were approved after the feature-level scope discussion.
They govern portable 1.0; they are not claims about existing full-DSL behavior.

### Numeric computation

- Operand types and explicit casts determine intermediate computation; output
  conversion happens last. An output dtype must not change the calculation.
  In particular, float32 `(x + 1.0) - x` computes in float32 even when the output
  is float64. Apply the new rules through the versioned portable path without
  silently redefining ordinary full-DSL execution.
- Literals adopt the operand type where representable; otherwise use explicit
  documented promotion rules or reject unrepresentable literals. Mixed typed
  operands follow one enumerated promotion table.
- Integer `/` produces float64; homogeneous float32 division stays float32.
- `int(expr)` evaluates its argument first, then truncates toward zero.
- Boolean arithmetic uses numeric zero/one. Explicit `bool()` and Boolean output
  perform truth conversion.
- Integer overflow, out-of-range integer narrowing, non-finite/out-of-range
  float-to-integer conversion, integer floor-division/remainder by zero, and
  invalid shift counts are runtime errors.
- Floating arithmetic retains IEEE-style NaNs, infinities and signed zero,
  including floating division by zero. Enumerate math-function domain behavior
  and accuracy guarantees separately; strict does not imply universally
  bitwise-identical transcendental results.
- JIT must implement the same checks and semantics or fall back before execution.

### Block reductions

- Reductions consume valid, currently active lanes of the evaluation block.
  Top-level reductions see all valid block elements. Supported nested conditions
  see their participating lanes; exclude padding and lanes that have returned
  or left the relevant loop.
- `any(empty)` is false, `all(empty)` is true, and `sum(empty)` is zero.
  Empty `min`/`max` is a runtime error. Specify other admitted reductions in the
  operation matrix before freezing.
- A reduced scalar controls or broadcasts only to participating lanes.
- Persist the logical evaluation partition independently of physical storage.
  Rechunking must not silently change reduction groups. Partial reads evaluate
  the original groups needed for the slice, then select requested results.
  Explicit repartitioning may change results and must be described as such.
- Specify accumulation dtype and logical lane order for floating sums; initially
  exclude reassociation into a different reduction tree. Integer accumulation
  obeys checked-overflow rules. Relaxed reductions are a later capability.

### Persistence and legacy formats

- New DSL-kernel saves normalize authoring syntax/captures, validate portable
  1.0, and persist the artifact, bindings and required execution context.
- Unsupported kernels fail saving with actionable diagnostics; there is no
  silent fallback to Python-specific persistence.
- Detect legacy persisted kernel formats and require explicit opt-in before
  using them, through the applicable deserialization policy. Never execute
  legacy Python reconstruction merely to detect the format.
- **No migration tooling or automatic conversion path is in scope.** Users who
  want a portable 1.0 artifact must follow the guidelines, manually adapt their
  kernels/bindings, and save through the validated 1.0 path. Legacy opt-in is
  permission to use the old representation, not a claim of portable compliance.
- Publication guidance must document supported operations, the numeric behavior
  change, reduction grouping, and legacy opt-in/manual adaptation.

## Remaining specification drafting

Implementation-led drafting can cover schema fields, capability negotiation,
typed encodings, explicit column bindings, existing ND coordinate conventions,
string width/Unicode rules, diagnostics, and compact conformance scenarios.
Use verified interpreter behavior as evidence, not inconsistent documentation or
an accidental backend behavior as the normative contract.

The policy choices above are settled. Draft the exact promotion table, literal
representability rules, function/type matrix, reduction NaN behavior and
accumulation dtypes, mask transitions, partition encoding, and legacy format
discriminators/opt-in API against them. Surface contradictions or additional
compatibility consequences for review rather than changing approved policy.

Draft an explicit dtype/function matrix based on current use cases; surface any
proposed omission of an existing useful operation for review. Pin string mapping
behavior and document truncation rather than silently redesigning it. Do not
promise bitwise-identical transcendental results without an implementation that
can deliver them.

## Bounded acceptance gates

1. Draft and review the detailed 1.0 semantics, type/function matrix, capability
   requirements, and detailed rules implementing the agreed decisions above.
2. Implement native validation and interpreter conformance for those contracts;
   reject unsupported kernels with actionable diagnostics.
3. Integrate portable payloads in kernel persistence, preserving typed bindings,
   ND context and reduction partition requirements. Verify standalone C loading
   without Python reconstruction, plus host save/load round trips.
4. Certify supported platforms. JIT must preserve the contract or select fallback
   before execution; additional acceleration is not a release gate.

Keep tests orthogonal and retain native ownership of exhaustive language fixtures.
Scope changes require explicit agreement; do not restart a whole-language JIT audit.
