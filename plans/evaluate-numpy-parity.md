# NumPy-input / explicit-NumPy-output evaluate parity

Current implementation status: the later `improved-miniexpr` work removes
NumExpr's runtime role and adds native strided destinations, host executors and
safe promoted-integer parallelism. See
[miniexpr-implementation-results.md](miniexpr-implementation-results.md) for
current results and the significant interpreter-only performance limitation.
The initial timings and NumExpr-backed fallback notes below are historical.

Scope: compare the public `numexpr.evaluate()` and `blosc2.evaluate()` APIs with
resident NumPy operands and an explicit NumPy `out`. No compression or fresh
full-result allocation is intentionally included in the performance comparison.

## Functional audit

Initial exploratory audit: 72 cases, including NumExpr's elementwise function
families, complex and byte-string operations, reductions, constant expressions,
broadcasting, reversed inputs, strided/Fortran destinations, and API options.
After the fixes, 70 cases agree and two reject the same prohibited output casts.
This is sampled API coverage, **not** exhaustive dtype/backend equivalence.

| Previously missing/broken behavior | Change |
| --- | --- |
| `copy(x)` | Preserve operand shape; dispatch NumPy receivers to NumPy; native adapter elides the value-copy node while retaining destination writes. |
| `fmod(x, y)` | Admit the approved elementwise function and its shape/arity contract. |
| Top-level `where(condition, x, y)` | Use NumPy for synthetic NumPy metadata operands; admit the native three-argument selection. |
| `complex(real, imag)` | Approved broadcasting array constructor; assign components separately to preserve infinities. |
| `sum`, `prod`, `min`, `max` into scalar `out` | Use ellipsis assignment instead of indexing a 0-D destination with `[:]`. |
| Constant-only expressions | Admit approved parsed graphs without operand names. |
| `casting="no"` / `"safe"` | Check the result dtype before destination writes, rather than silently performing a narrowing assignment. |

Representative regressions are in `tests/test_evaluate_numpy_compat.py`.
The existing recipe-cache tests now allow explicit destinations, which are
rebound per invocation and never retained by the cache.

Validation: 809 focused dispatch/API/storage tests plus 1,576 additional native,
lifetime, resident-execution and ndarray expression regressions passed (2,385
total); Ruff lint/formatting and whitespace checks passed.

## Dispatch/performance changes

### Follow-up: output identity, layout and keyword audit

- Empty explicit NumPy destinations now retain their identity, including shapes
  such as `(2, 0, 3)`. Evaluation still runs its normal validation path.
- Newly allocated results honor `order="F"`, `"A"` and `"K"`, including the
  default `"K"` with Fortran inputs. Elementwise allocation uses NumPy's broadcast
  iterator to choose layout, including permuted axes, without reading input data.
  Explicit destinations keep their existing layout. Invalid orders are rejected.
- `order` is consumed by Blosc2 rather than forwarded to NumExpr. Explicit C
  order retains native-first admission. Resident non-C-contiguous destinations
  use the approved NumPy graph fallback, followed by `np.copyto`; layout alone
  does not select NumExpr. This can allocate full-size intermediates and remains
  a performance opportunity for native strided-output support. Compressed/mixed
  fallback migration remains part of the removal plan.
- Probed `optimization="moderate"`/`"aggressive"`, `truediv=True`,
  `sanitize=True` and `disable_cache=True` in safe/full modes with fresh and
  supplied outputs. Invalid optimization and unknown keywords were rejected.
  These explicit options decline automatic native-first dispatch and reach the
  compatibility backend; accepting them does not mean native support.
- A regression test verifies that `disable_cache=True` rebuilds resident
  compatibility preparation on each call and populates neither the evaluate
  recipe cache nor the resident NumExpr program cache. This does not promise
  removal of parser caches or every lower-level library cache.
- `prepare_evaluate()` now provides an immutable operand-free evaluator with
  explicit per-call bindings, output rebinding, option overrides and lazy
  signature preparation. It reuses the existing bounded preparation caches;
  there is no global last-expression state. See the expression utilities docs.
- `re_evaluate()` additionally offers thread-local last-expression convenience
  after a successful public `evaluate()`. It rebinds operands from explicit
  dictionaries or caller scope, reuses only a still-live explicit output through
  a weak reference, and retains validation. Failed/nested evaluations and explicit
  prepared calls do not overwrite the saved expression. This is not a claim of
  matching NumExpr's unchecked-loop performance.
- Public option validation is now backend-independent, including invalid options
  on NumPy strided-output paths. Optimization is a backend request, not a promise
  of identical algebraic rewrites. Cache bypass scope and sanitize policy are
  documented explicitly.
- Overlapping NumPy operands are snapshotted before destination writes. Empty
  resident reductions use NumPy identities/errors, including axis reductions.
  This fixes empty sums/products incorrectly returning empty arrays and shifted
  copy destinations overwriting unread inputs.

### Initial parity improvements

- Fix native text ingestion's reduction flag: a root call is not necessarily a
  reduction. Previously `sqrt`, `hypot`, `fmod`, and `where` were misclassified.
- Extend the conservative lane-local floating-function admission list, including
  `log1p`, `expm1`, rounding, predicates and binary math. Previously several large
  arrays unnecessarily ran serially.
- Permit eligible resident reductions with explicit NumPy output, retaining the
  existing native parallel-reduction machinery.
- Reuse operand-free preparation recipes for reviewed math calls and explicit
  destinations, not just basic arithmetic and newly allocated outputs.
- For resident expressions that already need the optional NumExpr compatibility
  backend (notably complex construction), write directly into `out` instead of
  producing chunk temporaries. Cache unbound programs, with a per-program lock;
  neither operand nor destination arrays are retained. Execution failures are
  propagated without fallback/replay. This is **not a native performance win**.

Experiments lowering the Python-worker threshold and using a private prefilter
scheduler for small arrays did not reach parity. They are not retained; there is
no new scheduler, persistent native context or extension change in this patch.

## Reproduction

Use the `blosc2` conda environment and an extension matching this checkout:

```sh
conda run -n blosc2 python bench/evaluate-numpy-parity.py --threads 8 --samples 7 --json parity.json
conda run -n blosc2 python bench/evaluate-numpy-parity.py --dtype float32 --threads 8
conda run -n blosc2 python bench/evaluate-numpy-parity.py --dtype int64 --threads 8
```

The script qualifies output identity and numerical agreement before timing,
warms one shared destination, randomizes engine order, and reports medians.
Integer/bool comparisons are exact; floating/complex comparisons use explicit
tolerances. Unsupported or non-equivalent cases remain `N/A`, not timings.
Route instrumentation is outside the timed section. `numexpr-fallback` results
must not be presented as native parity. Compilation/warmup are excluded, while
public API preparation/rebinding is included on every timed call.

## Local measurements and remaining work

M4 Pro, eight threads, 8M float64 elements, seven timed samples, JIT `cc`:

| Expression | NumExpr ms | Blosc2 ms | Route |
| --- | ---: | ---: | --- |
| Linear arithmetic | 1.162 | 1.229 | native |
| Polynomial | 1.569 | 1.215 | native |
| Division | 1.101 | 1.209 | native |
| Power | 4.866 | 5.109 | native |
| Square-root norm | 1.489 | 1.181 | native |
| Sine + cosine | 3.794 | 4.264 | native |
| `log1p + expm1` | 4.667 | 4.531 | native |
| `hypot` | 1.299 | 1.424 | native |
| `fmod` | 3.465 | 3.712 | native |
| Rounding | 2.207 | 1.677 | native |
| Predicate | 1.469 | 1.116 | native |
| Selection | 1.387 | 1.232 | native |
| Copy | 0.802 | 0.738 | native |
| Sum of products | 6.247 | 1.293 | native |
| Complex construction | 1.407 | 1.476 | NumExpr compatibility |

The first baseline had `log1p + expm1` at approximately 37 ms; correcting
parallel admission removes most of that gap. Measurements are local warm runs,
not cross-platform guarantees; small cases are particularly sensitive to noise.

**Parity is not complete:**

- At 100k elements, serial native transcendental evaluation and Python dispatch
  still lose to NumExpr's worker scheduling (roughly 3–4x for some math recipes).
- Several float64 8M cases remain about 5–12% slower.
- Large int64 arithmetic/copy/reduction cases are competitive, but integer-input
  math returning floats still often misses conservative parallel admission and
  is about 5–7x slower. Do not simply broaden integer lane-local admission:
  checked conversions and integer error-domain semantics must remain intact.
- Large float32 sums fail the strict cross-engine numerical qualification
  (different accumulation/order behavior). Integer powers with overflow also
  fail agreement; NumExpr rejects integer complex construction. These remain
  `N/A`; no tolerance was relaxed to produce a performance comparison.
- This audit does not redefine Blosc2's NumPy-style dtype promotion to match
  NumExpr's different promotion rules.
