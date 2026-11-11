# Compatibility test sampling

The default suite uses `orthogonal_cases` (in `conftest.py`) for independent
compatibility dimensions rather than multiplying every dtype, operation, axis,
layout, tile size, and backend. It keeps a baseline, varies each dimension
independently, and adds mixed settings along a rotating diagonal. Every supplied
parameter value remains covered; exhaustive pairwise coverage is not promised.

Use explicit case tables for interactions that matter numerically. Python tests
retain representative cancellation, floating status, NaN/signed-zero, mask and
empty-input/error-boundary checks. Exhaustive dtype/operator and overflow/underflow
matrices belong in miniexpr. Compressed extrema retain every special-value case
for both min and max, with representative compiler, dtype, and root/method combinations.

Focused regressions for security/admission, metadata, persistence, masks,
ownership, concurrency, execution routes, and integer overflow are not replaced
by generic samples. The existing opt-in `--native-semantics` qualification
matrices are unchanged.

For the initial reduction, local default collection fell from 14,247 to 12,989
units. The largest modules changed as follows:

| Module | Before | After |
| --- | ---: | ---: |
| `test_menudet_arrays.py` | 408 | 67 |
| `test_native_graph_preparation.py` | 472 | 217 |
| `test_expression_graph_properties.py` | 347 | 107 |
| `test_expression_graph.py` | 467 | 342 |

Counts depend on installed optional dependencies. The initial pass affected only
tests added since `main`.

## General suite pass

A subsequent pass reduced local default collection from 12,989 to 12,002 units:

| Module | Before | After |
| --- | ---: | ---: |
| `ndarray/test_elementwise_funcs.py` | 903 | 459 |
| `test_schunk_insert.py` | 256 | 66 |
| `test_schunk_update.py` | 196 | 64 |
| `ndarray/test_auto_parts.py` | 116 | 33 |
| `ndarray/test_ndarray.py` | 314 | 269 |
| `ndarray/test_reductions.py` | 637 | 589 |
| `test_compressors.py` | 90 | 45 |

Elementwise coverage retains every function/dtype pair, rotating 1-D and 2-D
layouts instead of crossing both with every pair. Compressor tests retain every
codec/filter pair and alternate GIL modes. Eye construction retains every
geometry/offset/dtype combination, with representative chunk layouts. Automatic
partition sanity checks sample independent compression/shape/dtype axes.

SChunk mutations retain every storage format and all three meaningful API modes
(precompressed chunks, copied data, and non-copied data); the unused copy flag
is no longer varied for precompressed chunks. Storage/GIL/chunk-count settings
are sampled. Beginning, end, and middle positions are deterministic, and final
NumPy chunk contents are verified after insertion and replacement. Replacements
now differ from the original data rather than rewriting identical values.

Broadcast reductions retain one of two equivalent small geometries and the
existing opt-in large geometry. Heavy elementwise matrices and focused regression
tests remain unchanged.

`conftest.py` also isolates the process-global `set_releasegil` setting for each
test. Sampling must not make a callback test depend on which GIL parameter ran
last in an earlier module on the same xdist worker. Two isolation regression
cases bring the default local total to 12,004.

## Local HTTP diagnostics

Unhandled request errors from the standard-library test servers print the
exception type/message first, followed by the current test and complete
traceback. The same details are saved under
`.pytest_cache/d/http-server-errors/<worker>.log` (`master.log` without xdist).
With `-p no:cacheprovider`, logs go into a pytest temporary directory instead.
Each worker resets its log at session startup. Request errors are still logged,
not suppressed or automatically turned into test failures; inspect these files
if parallel progress output obscures a traceback from an otherwise passing run.

## Shared fixtures and duplicate cases

A final focused pass reduced default collection from 12,006 to 11,588 units:

| Module | Before | After |
| --- | ---: | ---: |
| `ndarray/test_lazyexpr.py` | 898 | 742 |
| `ndarray/test_lazyexpr_fields.py` | 442 | 242 |
| `ndarray/test_reductions.py` | 589 | 563 |
| `ndarray/test_stringarrays.py` | 76 | 40 |

Shared array fixtures now use explicit scenarios: both floating precisions still
see every original size/rank, and each default mixed-field dtype pair still sees
both sizes and both default partition relationships. Standalone dtype/shape
fixtures are unchanged. Remaining shared-fixture combinations retain their full
original coverage under `heavy`, including the NumPy 1.x short-integer skip.

Unicode roundtrips retain all 12 distinct effective partition configurations;
the former 48-case grid repeatedly normalized wrong-rank chunk/block tuples to
`None`. Reduction-expression tests collect only valid shape/axis/operation
combinations instead of counting early returns as passes. All cases that
previously performed numerical comparisons remain covered.

## PR #741 final orthogonality pass

This pass removes redundant combinations rather than changing default selection,
adding skips, or reclassifying tests as `native_semantics`. Local test-directory
collection fell from **14,033 to 12,372** cases (1,661 fewer). The 123 configured
module-doctest cases are separate. Counts vary with optional dependencies.

| Module | Before | After |
| --- | ---: | ---: |
| `ndarray/test_setitem_chunkwise.py` | 818 | 218 |
| `ndarray/test_lazyexpr.py` | 1,073 | 737 |
| `ndarray/test_reductions.py` | 799 | 432 |
| `ndarray/test_elementwise_funcs.py` | 459 | 325 |
| `test_native_graph_preparation.py` | 217 | 107 |
| `test_menudet_arrays.py` | 67 | 29 |
| `test_native_graph_extrema.py` | 76 | 52 |
| `test_expression_graph.py` | 342 | 320 |
| `test_native_graph_truth.py` | 66 | 48 |
| `test_menudet_artifact.py` | 146 | 134 |

- Chunkwise assignment retains **every scalar-value/region and source/region
  pair**. Disk placement and buffer budgets rotate across those pairs; each
  scalar value still sees all four disk/budget settings, and each array source
  sees all six source/destination placement/budget settings. Dedicated special
  chunks, padding, aliasing, conversion failure and scratch-memory checks remain.
- Ordinary-expression tests retain each expression and both authoring interfaces.
  Bool's operator-specific result dtypes retain both interfaces for every supported
  expression; other integer widths rotate. Shared fixtures keep both floating
  precisions at each rank, dropping an equivalent second small 1-D size. Function
  wrappers retain both precisions; rank, constructors and output slicing are
  sampled independently instead of multiplying each syntax form.
- Reductions retain each operation and both precisions where computation types
  matter. Offset/cancellation, dtype, axis and mask cases remain explicit; ddof,
  keepdims, disk placement and layout rotate where independent. Fast-path tests
  retain every reducer/layout/axis combination, and every persisted syntax form
  retains every reducer in memory and on disk. Already-heavy scenarios stay heavy;
  no former default case was moved into that marker.
- NumPy ufunc aliases share their canonical implementation's numerical cases;
  separate identity checks protect all alias bindings. Unsupported NumPy signatures
  that formerly returned before exercising Blosc2 are not generated. The broad
  wrapper sweep uses representative integer widths, retaining int64 true division
  because its policy differs from int32; native and focused Python tests retain
  exact wide-integer coverage.
- Native integration tables cover binding, traversal, status/error translation
  and route selection, not a second numerical conformance suite. The authoritative
  suites remain `miniexpr/tests/graph/preparation.c`, `tests/numpy-compat/arrays.c`
  and `tests/test_menudet_jit_parity.c`. Both floating widths and both extrema
  operations retain the special-value cases, while compiler/keepdims choices are
  sampled independently. Existing Python opt-in native-semantic matrices remain
  available unchanged.
- Security/admission regressions, persistence retirement/preflight, no-replay
  injection, concurrency, lifetime, explicit unsupported-route and the new
  mixed-width weak-literal regression tests are unchanged.

Validation in conda `blosc2`, using the configured parallel workers:
**12,402 passed, 94 skipped, 3 xfailed in 29.76 seconds**, including module doctests.
The legacy escape-hatch run also passed: **12,364 passed, 94 skipped, 41 xfailed
in 29.56 seconds**. No numerical assertion or tolerance was relaxed.

## Post-4.15 test-cost pass

Default collection (including configured doctests) drops from **13,205 to
12,977** cases in the local `blosc2` environment. This supersedes the chunkwise
assignment sampling described above:

- Scalar/source assignment cases sweep each value and region independently,
  adding mixed settings with `orthogonal_cases`, rather than retaining their
  full product. Dedicated padding, special-chunk, aliasing and buffer-budget
  regressions are unchanged.
- Cframe tests sample maker/storage combinations. The offsets-boundary fixture
  uses 2,049 tiny chunks instead of 3,000 larger ones. Byte-for-byte reserialization
  no longer builds that fixture only to skip it. Its directory-storage stress
  case remains available with `pytest -m heavy tests/ndarray/test_cframe.py`;
  memory and contiguous-file offsets-boundary roundtrips remain in the default suite.
- `read_into` retains every key, dtype and storage kind without crossing every
  key/dtype with every storage. Strided output and resize regressions are unchanged.
- Special-value SChunk construction keeps every value with empty, full and
  partial final chunks, rotating storage/compression settings and using smaller
  chunks. The int64 chunkwise-cast fixture is halved but still exceeds the
  16 MiB dispatch threshold.
- NumExpr import checks prefilter source text before parsing ASTs. Nested pytest
  disables third-party plugin autoload, non-TUI sessions avoid the Textual idle
  patch import, and thread-environment checks avoid redundant `strtol` spellings.

Crash/deadlock subprocess isolation, GIL/thread combinations, stress repetitions,
numerical assertions and tolerances are unchanged.

Validation in conda `blosc2`: **12,865 passed, 109 skipped, 3 xfailed in
32.74 seconds**, versus 34.81 seconds before this pass on the same machine.
The opt-in directory cframe stress case also passes. Ruff lint/format checks pass.
