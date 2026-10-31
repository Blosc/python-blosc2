# Compatibility test sampling

The default suite uses `orthogonal_cases` (in `conftest.py`) for independent
compatibility dimensions rather than multiplying every dtype, operation, axis,
layout, tile size, and backend. It keeps a baseline, varies each dimension
independently, and adds mixed settings along a rotating diagonal. Every supplied
parameter value remains covered; exhaustive pairwise coverage is not promised.

Use explicit case tables for interactions that matter numerically. In particular,
the reduction tests retain both floating widths for cancellation, product
overflow/underflow, extrema NaNs and signed zero, and empty-input identities or
errors. Compressed extrema retain every special-value case for both min and max,
with representative compiler, dtype, and root/method combinations.

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
