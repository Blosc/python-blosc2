# Parallel execution of lane-local while loops

## Status and goal

Proposal for later implementation. No native or Python execution policy is changed
by this document.

Allow eligible Menudet `while` kernels to run across existing native block workers
without rewriting them as captured-bound `for` loops. Eligibility must be proved
from native semantics, not asserted by Python or inferred from benchmark results.
This is not a promise to parallelize arbitrary loops.

## Motivation and existing evidence

The original `mandelbrot_menudet(cr, ci, max_iter)` in
`bench/ndarray/jit-dsl.py` used a runtime `int64` iteration bound and a `while`
loop. Its artifact was JIT-capable but not natively partitionable, so materializing
compressed output executed one serial numerical block even with multiple threads.

Local macOS ARM measurements on 1,048,576 float64 elements, 50 iterations, three
warmed samples:

| Diagnostic control | One worker | Twelve workers | Speedup |
| --- | ---: | ---: | ---: |
| Unchanged while kernel, manually scheduled decoded NumPy blocks | 644 ms | 73 ms | 8.8x |
| Captured-bound for kernel, native compressed-storage prefilter | 512 ms | 61 ms | 8.4x |

These are different workloads/storage paths, not interchangeable timing arms.
The manual while experiment matched serial values (including NaNs) and merged
floating-point flags. A small lane-varying-bound check also matched. This supports
feasibility for that recurrence, not a general safety proof or performance claim.
The benchmark now uses the second approach as a benchmark-only workaround.

## Current implementation boundaries

- In miniexpr `src/dsl_artifact.c`, `artifact_proof_block()` handles a conservative
  bounded `for` subset but rejects `while`. `me_artifact_is_partitionable()` also
  rejects runtime integer inputs in its current floating-map proof.
- miniexpr already executes/JIT-compiles while loops. The missing feature is
  parallel-admission proof and binding, not basic while code generation.
- `src/blosc2/storage_readers.py:storage_reader_eligible()` relies on the native
  proof. The dense compressed prefilter additionally requires matching NDArray
  operands; a scalar bound represented as a zero-dimensional NumPy operand fails
  that layout gate independently of the proof.
- `MenudetLazyArray` preserves original logical groups when fast-path admission
  fails. Merely increasing the array size, setting threads, or shrinking output
  blocks does not make an unproven kernel partitionable.

## Supported subset and non-goals

Start with a canonical loop such as:

```python
i = 0
while i < max_iter:
    # Total, lane-local numerical operations on this element's state.
    i = i + 1
```

Initially require a recognized integer induction variable, monotonic safe update,
read-only uniform scalar bound, and a body without cross-lane effects. Prove
counter overflow safety, initialization, and bound invariance. Preserve existing
behavior for negative/zero bounds rather than silently normalizing arguments.

Later consider lane-local data-dependent conditions, including a bounded escape
condition. Different lanes may take different iteration counts; uniform iteration
counts are not a prerequisite for numerical independence.

Initially exclude reductions in conditions/bodies, cross-element reads/writes,
context/group-dependent operations, unsupported checked conversions, early
returns, break/continue, nested while loops, and mutations of loop bounds. Extend
these only with dedicated semantic analysis. Retain existing serial execution for
everything unproven; do not weaken authoritative graph rejection.

## Implementation phases

### 1. Specify semantics and native admission

- Audit interpreter and JIT while semantics, including active masks, condition
  prefixes, iteration caps, empty domains, floating status and failure ordering.
- Separate two questions: is the body lane-local, and can this invocation be
  safely retiled given runtime bounds/cap/error policy?
- Decide how native metadata distinguishes unconditional partitionability from
  conditional invocation-time eligibility. An artifact-wide Boolean must not
  certify properties that hold only for a particular runtime bound.
- Define compatibility behavior for older extensions and serialized artifacts.
  Native validation must reconstruct/verify proof; do not trust an editable JSON
  flag or a user-provided promise.

### 2. Prove canonical while loops in miniexpr

- Add bounded-cost induction/body analysis to the native proof infrastructure;
  do not enumerate arbitrary runtime iteration counts during analysis.
- Recognize safe integer bounds/counter uses without broadly admitting integer
  arithmetic that can produce checked errors or overflow.
- Add invocation-time guards where static proof is insufficient. A rejected guard
  declines parallel admission before execution, retaining the original serial
  semantics; it must not introduce a new public exception or numerical fallback.
- Keep iteration-cap behavior intact. In particular, do not replace while with
  for when that would bypass the configured cap. Define handling of cap violations
  before admitting them, including partial writes and floating status on errors.

### 3. Bind runtime scalar controls to existing block workers

- Carry a validated read-only uniform scalar control to every worker without
  allocating an array-sized copy or changing its dtype/promotion semantics.
- Extend native source/prefilter bindings as needed, then the Python compressed
  admission gate; do not simply remove the all-NDArray check.
- Reuse existing worker pools, bounded scratch, FP isolation/status merging and
  lifetime management. Keep numerical execution outside the Python GIL.
- Preserve logical coordinates, tail handling and alias/output validation.
  Never replay execution on another route once native work has started.
- Do not add an unconditional user-visible "parallel-safe" override.

### 4. Broaden eligibility only after canonical-loop qualification

- Analyze lane-local data-dependent termination and masked state updates.
- Check whether interpreter batching and JIT have equivalent observable failure
  domains when lanes terminate differently.
- Evaluate dynamic block scheduling for uneven iteration counts before adding
  complexity. Treat lane-varying bounds, nested loops and early flow as separate
  extensions, not automatic consequences of uniform scalar support.

## Validation ownership

Native numerical/proof semantics belong in miniexpr tests. Cover representative
interpreter/JIT parity, eligible and rejected loops, zero/negative/large bounds,
counter overflow, cap boundaries and cap changes, masked/empty/tail blocks,
floating flags, nested dependencies and deterministic failures. Include concurrent
calls and ensure invalid/untrusted artifact metadata cannot bypass admission.

Python-Blosc2 tests should focus on scalar binding, capability negotiation,
dispatch, serialization compatibility, output/alias contracts and a few end-to-end
serial/parallel cases. Avoid duplicating exhaustive native dtype/backend matrices.
Test failure propagation and absence of replay, not just successful values.

## Performance qualification and acceptance

- Compare the same while artifact with one and multiple workers, separately for
  decoded NumPy execution and compressed-input/output execution.
- Use the original runtime-bound Mandelbrot recurrence plus a numerical recurrence
  with uneven per-lane work when that subset becomes supported.
- Record preparation separately; use warmed, interleaved repetitions, verified
  package/extension identities, actual JIT state, native route, blocks/workers,
  CPU versus wall time, FP status and memory/scratch costs.
- Include small/large arrays and block geometry controls. Select any scheduling
  threshold from evidence, not element count alone; computational intensity varies
  substantially with iteration count.
- Require correct semantics, native proof/admission and demonstrated concurrency
  on a sufficiently large eligible case. No wall-clock thresholds in pytest and
  no timing gates inferred from the local observations above.
- Keep unproven loops and older runtimes working through existing serial routes.
  Document the admitted subset and limitations before expanding it.
