# NumExpr removal: completion audit

Date: 2026-10-22. Baseline: Python-Blosc2 `77e21443`, miniexpr
`ed437bd7b84f0e90e967664504e5d031169f4fb2`. This document describes the follow-up
working-tree changes as well; it is not evidence of remote qualification of them.

## Runtime inventory and ownership

The original runtime removal already existed before this audit. A source/metadata
audit found no package import or executable call to the NumExpr package. Historical
release notes and external comparison tools are not runtime integrations.

| Location | Replacement / retained contract | Owning checks |
| --- | --- | --- |
| `pyproject.toml` | No required NumExpr dependency | Wheel metadata audit; `qualify_without_numexpr.py` |
| `__init__.py`, `core.py` | No optional import, initialization, thread coupling or version probe; `numexpr=None` is compatibility only | `test_no_numexpr.py`, `test_expression_threads.py` |
| `lazyexpr.evaluate_chunk` | Metadata-only native graph admission; otherwise explicit NumPy semantics | `test_no_numexpr.py`, `test_evaluate_numpy_compat.py` |
| `lazyexpr.ne_evaluate` | Non-executing legacy name delegating to `evaluate_chunk`; preserves caller depth | Caller-scope checks in `test_no_numexpr.py` |
| Chunk maps, filters, reductions | Existing native prefilters or tile-local NumPy execution; no execution-error replay | `test_evaluate_numpy_prefilter.py`, ndarray reduction/selection tests |
| Resident program cache | Removed; bounded native preparation caches contain no operand arrays | `test_prepared_evaluate.py`, `test_re_evaluate.py`, native graph tests |
| `_numpy_eval_expr`, scalar construction | Native metadata or approved NumPy probes; no NumExpr validation/inference | LazyExpr, expression-policy and dtype tests |
| `expression_graph.py`, `utils.py`, `ndarray.py` | Approved expression grammar, NumPy function registry and native numerical contracts | Safe/full evaluation tests; elementwise function tests |
| `ctable.py`, `indexing.py` | Shared expression helpers, no direct NumExpr calls | Existing table/index/query suites |
| Benchmark/reference tests | Optional external NumExpr only; essential behavior does not skip when it is absent | Full absence run; benchmark tests without NumExpr |

The old minimum/maximum NaN assertion skips have been removed: those tests now
enforce the actual NumPy contract instead of hiding disagreements attributed to a
backend that is no longer used. Historical WASM/feature exclusions unrelated to
dependency removal have not been broadened or silently reclassified as native.

## Compatibility and coverage decisions

| Surface | Policy |
| --- | --- |
| `local_dict`, `global_dict`, caller lookup | Preserve public lookup and internal wrapper depth; do not cache frames |
| `out`, `casting`, `order` | Blosc2-owned shape/identity/layout/casting contracts, validated before writes |
| `optimization` | Validated compatibility hint; no NumExpr algebraic rewrites |
| `truediv` | Removed from `evaluate`/prepared evaluators; Python 3 / NumPy division, `//` for floor division |
| `sanitize` | Removed; mandatory safe-mode validation and the separate full-mode trust policy remain |
| `disable_cache` | Removed; operand-free preparation caches are automatic, diagnostic tooling owns cold-cache controls |
| `ne_args` | Legacy internal compatibility container; not a public `evaluate` option or engine selector |
| `blosc2.numexpr`, internal `lazyexpr.numexpr` | `None` placeholders, never a runtime import |
| `NUMEXPR_*` | No effect on Blosc2 workers or native execution policy |
| `evaluate` default | Native interpretation, also for prepared/repeated evaluators; explicit global/context/environment JIT policy remains available; other Menudet API defaults are unchanged |
| `jit=False` / unavailable compiler | Native interpretation where supported, not an unrelated expression engine |
| Strict native / require-JIT | Unsupported native signatures reject; explicitly required JIT still rejects when unavailable |
| Numerical/runtime errors | Propagate; preparation-only unsupported signatures may choose NumPy before execution |

Admitted real numeric maps, selection and supported reductions use existing native
graphs/schedulers. Complex portable kernels and qualified expression paths use
native interpretation, including the newly batched arithmetic; unsupported
complex construction/functions keep explicit NumPy semantics. Strings and
datetime/timedelta operations use approved NumPy semantics when native admission
declines, preserving units, NaT and component-safe complex construction. Unsupported
linalg and shape-changing operations retain their materialization contracts.

WASM's scalar string-expression reduction prefilter remains disabled for **all**
dtypes. Its chunked fallback is not a NumExpr dependency, and portable Menudet block
reductions remain supported. No host-worker speedup is claimed for WASM.

## Gaps fixed in this audit

- Native map and scalar-reduction prefilter exceptions after successful preparation
  now propagate, with prefilter cleanup, rather than replaying through NumPy.
  Injection tests preserve a partial caller-output write and prohibit replay.
- The existing serial bounded stored-output schedule no longer requires a JIT
  artifact. Its 262,144-element tile limit, output identity, broadcasting and
  interpreter route are tested; no new scheduler or parallel-safety rule is added.
- Direct `evaluate_chunk` caller lookup and fallback context now work independently
  of the legacy helper name.
- Absence tests cover resident/compressed/mixed, layout, complex, strings, NaT,
  prepared/repeated calls and genuine no-JIT execution. A hostile installed-module
  stand-in rejects NumExpr backend/thread accesses.
- The qualification runner blocks NumExpr imports, records swallowed package import
  attempts, checks requirements and optionally requires both module/distribution
  absence. CI explicitly uninstalls the optional reference before its full suite.
- The resident benchmark always provides a NumPy reference. NumExpr is optional;
  unavailable or non-equivalent reference results remain `N/A`, with reasons in
  JSON rather than a verbose table trailer. Native route probes observe successful
  execution/reporting, not merely an attempted lowering.
- Repeated resident native maps reuse bounded lowered plans keyed by metadata
  only; arrays and outputs are rebound per call. Syntax admission is cached and
  small maps bypass parallel-admission work. Warmed 1K `evaluate(out=...)`
  improved from ~66/74 µs (copy/linear) to ~26/30 µs. The cache is operand-free,
  thread-safe, size-bounded and invalidated by dtype/shape/mode/policy changes;
  aliased or incompatible outputs retain the general validated path.

## Qualification and release boundaries

The published baseline already passed native Linux/macOS/Windows x64/ARM64/WASM
CI and Python Linux/macOS/Windows/WASM CI. That evidence predates this audit and
must not be presented as qualification of its new changes.

Local qualification (macOS arm64, Python 3.14.4, NumPy 2.5.4, Python-Blosc2
`77e21443` plus this audit, native `ed437bd7...`):

- Shared-test-package full suite with NumExpr installed: **11,825 passed,
  145 skipped**.
- Independently built/extracted wheel with both NumExpr module and distribution
  metadata absent: **11,825 passed, 145 skipped**, and zero recorded package
  NumExpr import attempts.
- **85** packaged Python modules audited: no NumExpr import and no wheel
  `Requires-Dist: numexpr`.
- The old thread test was replaced by an independence test: `NUMEXPR_*`
  variables do not control Blosc2 workers and NumExpr remains unimported.
- Focused absence/replay/dispatch/benchmark suites passed, including the static
  package import/metadata audit and optional-tool unavailability paths.
- Sphinx documentation builds with only pre-existing warnings.
- After the `evaluate()` interpretation-default/API cleanup, the full local suite
  passed **11,828 tests, 145 skipped**. Focused cleanup coverage also passed with
  NumExpr absent and zero recorded runtime imports. Pre-push cache review added
  policy-invalidation, exact-type admission, eviction, strided-binding and
  concurrent-rebinding tests; all **44** focused policy/cache/absence tests pass.
- These counts are local evidence only; the new CI gates and follow-up changes
  have not yet run on remote platforms.

The shared conda installation is never uninstalled or overwritten: a
dependency-only search path excludes both NumExpr's module and metadata, and an
independently built wheel supplies the tested native extension.

### Refreshed performance

M4 Pro, eight workers, shared warmed NumPy destination, resident float64 inputs,
median of seven shuffled samples after three warmups. NumExpr 2.14.2 is an
external reference. TCC and system-CC JIT use separate runs. Routes come from the
native execution report; `copy` may use the identity/copy path when eligible.
Chained NumPy recipes allocate intermediates, while copy/complex write directly
to `out`.

| Recipe | NumPy ms | NumExpr ms | Interpreter ms | CC JIT ms | Route |
| --- | ---: | ---: | ---: | ---: | --- |
| copy | 0.918 | 0.803 | **0.667** | **0.746** | native identity/copy |
| linear | 7.367 | 1.220 | 1.474 | 1.230 | native |
| polynomial | 17.396 | 1.607 | 1.369 | 1.186 | native |
| sqrt norm | 15.409 | 1.504 | 1.569 | 1.171 | native |
| sine + cosine | 36.628 | 3.854 | 4.441 | 4.221 | native |
| complex(x, y) | 4.521 | 1.438 | 4.563 | 4.500 | NumPy fallback |

The interpreter results contain `jit_stages=0` and `interpreter_stages=1`,
`temporary_bytes=0`, no full-input materialization and eight map tasks; the route
probe verifies actual execution, not merely a lowering attempt. At 100K elements
interpreter maps take 0.10–0.24 ms versus NumExpr 0.09–0.14 ms; at 1K elements
Blosc2 API/preparation overhead is now about 0.04–0.05 ms versus NumExpr
0.008–0.012 ms after the prepared-plan cache.

Raw medians, spreads, routes and native execution reports:
[interpreter](remove-numexpr-interpreter-results.json),
[CC JIT](remove-numexpr-cc-results.json),
[TCC JIT](remove-numexpr-jit-results.json). Reproduce with:

```sh
conda run -n blosc2 python bench/evaluate-numpy-parity.py \
  --items 1000 100000 8000000 --backend interpreter --threads 8 \
  --samples 7 --warmups 3 --cases copy linear polynomial sqrt trig complex
```

Performance is not an absence gate. Existing before-removal baselines and residual
no-JIT regressions remain in [miniexpr-implementation-results.md](miniexpr-implementation-results.md)
and [no-jit-and-small-math.md](no-jit-and-small-math.md). The latter supersedes the
earlier scalar-interpreter snapshots. Native complex arithmetic batching improves
native complex expression throughput, but the benchmark's complex constructor
remains an explicit NumPy fallback and is not universal NumExpr parity.

Release approval, acceptable regression budgets, 100M/full mixed-layout memory
qualification, and remote CI for these follow-up changes remain explicit decision
points. They are not silently marked complete by a local functional pass.
