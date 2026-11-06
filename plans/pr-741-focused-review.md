# PR #741 focused final review

Reviewed Python `bd6fab7327451472435fc0d24cb4fa34093ab1de` with the bundled
miniexpr `12bf4dad7acb0e5599474e03724ef1243986f1b6`.

The branch advanced from `f638c28b` during this review. The intervening changes
are outside the reviewed implementation; all 88 installed Python modules were
compared byte-for-byte with `bd6fab73` and match. Native/Cython sources did not
change between those revisions.

Scope: typed dispatch/fallback/no-replay, scalar/reduction dtype preservation,
persistence retirement/preflight ordering, native partitionability and guarded
selects. This is a focused source review with local checks, not exhaustive
assurance or final cross-platform qualification.

## Finding: mixed-input partitionability can bypass weak-scalar range checks

**P2 — fixed locally; published native pin still needs the fix.** The original
native `src/dsl_graph.c:g_partitionable()` combines
input integer dtypes (lines 347–358) and checks weak integer constants against
that aggregate width (lines 373–382). A literal can fit the aggregate type but
not the narrower intermediate expression that consumes it.

Reproducer against the independently built wheel:

```python
import blosc2
import numpy as np
from blosc2.graph_plan import NativeGraph

text = "(a + 300) + b"
signature = {"a": "uint8", "b": "int16"}
plan = NativeGraph.from_expression(text, signature, jit=False)
assert plan.info()["partitionable"] is True  # Incorrect admission.

inputs = {name: np.ones(513, dtype=dtype) for name, dtype in signature.items()}
expr = blosc2.lazyexpr(
    text, {name: blosc2.asarray(value) for name, value in inputs.items()}
)
print(expr.dtype)  # int16: construction incorrectly passes the range check.
expr[:]  # MenudetError: evaluation_error, rather than preparation-time OverflowError.
```

NumPy rejects `a + 300` with `OverflowError: Python integer 300 out of bounds
for uint8`. Also reproduced with `(a + -1) + b` for uint8/int16 and
`(a + 128) + b` for int8/int16. Ordinary mixed `a + b` remains correct.

Python `src/blosc2/native_graph.py:native_weak_scalars_eligible()` (lines 322–325)
trusts a positive native partitionability result and skips its dtype/range
inference. This makes the native proof error observable through the public
ordinary-expression API, not just through internal graph inspection.

This is not evidence of silently wrong values or replay. It is an unsound
admission decision and a range-check/error-phase regression. Do not fix it by
catching execution failures and retrying NumPy.

Recommended closure:

- Prove weak constant conversions at their actual consuming nodes, or decline
  partitionability conservatively when that proof is unavailable.
- Add native positive/negative proof tests for mixed widths and subtree-local
  literal bounds; add one representative Python preparation-time regression.
- Retain ordinary safe mixed-width acceleration and no-replay behavior.

### Local fix

Native fix revision: `d69c3c8` (pushed to `Blosc/miniexpr`).

`g_partitionable()` now intersects the integer input ranges and requires weak
integer literals to fit that conservative range as well as the common promoted
type. Boolean inputs do not narrow that range. This avoids introducing a second
per-node dtype system into the graph admission proof. Safe mixed-width maps and
in-range literals remain eligible; an expression such as `(uint8_x + int16_y) +
300` may conservatively stay nonpartitionable even though it is valid.

Native `tests/graph/preparation.c` covers signed/unsigned boundaries, both subtree
positions, safe mixed/Boolean maps, and revalidation after JSON round trips.
Python `tests/test_typed_expression_evaluator.py` covers the representative public
error contract with native-prefilter execution forbidden. The new Python test
failed against the original wheel and passes with the local native override.

Expression construction remains lazy: the public guarantee tested here is
`OverflowError` on evaluation without entering native-prefilter execution, not a
new requirement that constructing every LazyExpr eagerly raises.

The local-override wheel is isolated from the shared editable install. Its SHA256
is `ef3615b26c5e6c64c6cf57407eea13709b4c0f44d283b66c301c5b47dfd0de1d`.
The shipping pin is updated to `d69c3c8` to publish the native fix for
cross-platform qualification.

Fix verification: **455/455 native tests passed**, including interpreter,
TCC and system-CC gates; **114 focused Python tests passed**, and the final added
valid-promotion assertion passed separately. Installed-module doctests passed
**121 cases, 2 skipped**. Ruff lint/format and both repository whitespace checks
passed. Native build output retains the existing duplicate-library linker and
Make jobserver warnings; no new source warning was reported.

The full default Python test-directory run against the isolated local-override
wheel also passed: **13,815 passed, 217 skipped, 12,907 deselected, 3 xfailed**
in 205.40 seconds. Installed-module doctests were run separately as recorded above
to avoid collecting checkout modules against the isolated installed package.

## Other reviewed boundaries

- **Dispatch/no-replay:** preparation catches are separated from execution in
  `expression_evaluator.py`, `native_graph.py` and `lazyexpr.evaluate_chunk`.
  Typed prefilter and reduction execution failures propagate. The dynamic worker
  scheduler continues only with unclaimed chunks after a failure; it does not
  replay failed chunks, and ultimately raises the selected failure.
- **Dtypes:** typed map output and reduction accumulator types are separate.
  Strong NumPy scalars/0-D operands are not specialized into weak literals.
  Boolean sum/product use any/all semantics when the requested accumulator is
  bool. No further dtype defect was reproduced in the selected checks.
- **Persistence:** direct b2object legacy markers reject before VLMeta payload
  decoding; inline operand closures preflight before sibling references resolve.
  Legacy source reconstruction is a rejection stub, and both safe/full modes
  reject historical standalone, embedded and table recipes. No reconstruction
  bypass was found. This does not claim every malformed payload produces the same
  diagnostic or that external references can be inspected without opening them.
- **Guarded selects:** float math retains participation guards; plain leaf copies
  can be eager without changing numerical status. Uniform casts are deferred until
  a lane participates, then cached within that evaluation. Existing native tests
  cover never-active/late-active guards, nested selects, signaling NaNs,
  conversion overflow, and 513-lane tile boundaries. No additional guarded-select
  defect was reproduced.

## Local verification

All commands used conda `blosc2`. Built a new stock-pin wheel, without sibling
source overrides, and installed it only into an isolated temporary package.
Verified both package and extension import paths before tests; the shared editable
installation was not replaced.

- Wheel SHA256: `16aa270235537651b4dd553a25e07ceb66e130cbda9019833e139329cdc09541`.
- macOS ARM64, Python 3.14.4, NumPy 2.5.4.
- Focused integration suite: **448 passed, 374 native-semantics cases deselected**.
- Original reported expressions plus exact-int64 comparison, mixed-width addition
  and Boolean OR: **84 value-and-dtype checks passed** across `[:]`, partial
  slices, `.compute()`, `.sum()`, `.min()` and `.max()`.
- Separate opt-in Python numerical/JIT suite: **645 passed** (189.80 seconds).
- Native gates: **4/4 passed** (116.16 seconds): required TCC parity, required
  system-CC compact parity, interpreter/custom-fast-flags protection, and graph
  preparation. The parity executable also runs the guarded-batch participation,
  uniform-conversion, integer-promotion and int32-modulo regression checks.
- Initial broader JIT invocations reached the 120-second command limit without
  reported assertion failures; these interrupted runs are not counted as passes.

Temporary build, wheel, package and reproducer scripts:
`/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/final-focused-review/`.

## Draft response to Fabian — not posted

Thanks @FabianIsensee for the detailed reproductions and follow-up PRs. Here is
where the original report stands after integration:

- The ordinary-expression default now uses typed native execution where admitted,
  with explicit typed/NumPy fallback selected before execution. Intermediate dtypes
  are no longer chosen from the final output/reduction accumulator dtype.
- In a newly built wheel matching Python `bd6fab73` / miniexpr `12bf4dad`, I reran your
  reported expressions: negative constants, integer true division/hypot, fractional
  comparisons, signed modulo, power precedence/associativity, narrow integer
  arithmetic and Boolean division. Values and result dtypes matched NumPy across
  full/partial reads, materialization and scalar reductions; the narrow-integer
  and Boolean-division reproducers did not crash.
- The mean-dtype, Boolean-extrema, native arithmetic/complex and storage-path
  follow-ups are integrated. Focused fault-injection tests also confirm that an
  execution failure is propagated rather than replayed through another evaluator,
  including after a partial destination write.
- Local verification passed 448 focused integration tests, a separate 645-test
  opt-in numerical/JIT selection, and four native interpreter/TCC/CC/preparation
  gates. This is fresh macOS ARM64 evidence, not a replacement for final-head
  Linux/Windows/WASM and distribution qualification.
- This final review found an additional corner: mixed-width partitionability
  wrongly admitted an out-of-range weak scalar in a narrower subtree, e.g.
  `(a + 300) + b` for uint8/int16. A local native admission-proof fix and native/
  Python regressions now preserve the `OverflowError` without native-prefilter
  execution. The published dependency pin still needs this fix; no execution-error
  fallback/replay was introduced.
- On performance, std/var/where-related changes and subsequent typed-path
  optimizations are integrated, but I have **not** rerun your original
  300×512×512 / Ryzen / 28-thread comparison in this review. We should not claim
  that slowdown is closed from integration or correctness results alone.
  Arbitrary non-flat N-D layouts retain documented performance limitations;
  further acceleration remains separate from numerical correctness.

Thanks again for helping expose and fix these issues. After the remaining
admission corner and final-head qualification are closed, a rerun on your original
workload would be especially useful.
