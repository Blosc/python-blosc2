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

Counts depend on installed optional dependencies. These reductions affect tests
added since `main`; pre-existing test matrices are preserved.
