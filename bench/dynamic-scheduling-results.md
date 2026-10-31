# Dynamic map scheduling: local comparison

Measured on the 8-performance-core/4-efficiency-core Mac, October 29, 2026.
Both versions use the `blosc2` conda environment (Python 3.14.4), NumPy 2.5.4,
numexpr 2.14.2, float64, 12 threads, LZ4 level 5, fresh outputs, and default
system math. No JIT backend is explicitly requested. Static and dynamic builds
were loaded in isolated packages; the installed extension was not replaced.
The static baseline includes batched comparisons (`miniexpr` `c2d3324` and
Python-Blosc2 `1000ccfc`); these comparisons isolate the subsequent scheduling work.

Each entry is a preparation-cold median of five interleaved engine samples,
in milliseconds. Before/after programs ran sequentially, without concurrent
builds or tests. Operand construction and cache isolation are outside timing.
Small differences, especially below one millisecond, may be run-to-run noise.

```sh
python bench/menudet-engines.py --dtype float64 --codec lz4 --warmups 0 --samples 5 --color never
python bench/menudet-engines.py --dtype float64 --codec lz4 --warmups 0 --samples 5 --color never --expr 'sin(x + 1) * cos(x - 2)'
python bench/menudet-engines.py --dtype float64 --codec lz4 --warmups 0 --samples 5 --color never --expr '(sin(x + 1) * cos(x - 2)) > 1'
```

## Arithmetic: `x + (y - 1)`

| Items | Inputs | evaluate static | evaluate dynamic | Menudet static | Menudet dynamic |
|------:|:-------|----------------:|-----------------:|---------------:|----------------:|
| 100K  | NumPy  |           0.113 |            0.120 |          0.129 |           0.126 |
| 100K  | Blosc2 |           0.420 |            0.405 |          0.263 |           0.287 |
| 1M    | NumPy  |           0.397 |            0.335 |          0.380 |           0.311 |
| 1M    | Blosc2 |           1.145 |            1.052 |          0.915 |           0.983 |
| 8M    | NumPy  |           2.000 |            1.824 |          1.954 |           1.832 |
| 8M    | Blosc2 |           3.464 |            3.643 |          4.235 |           4.512 |
| 100M  | NumPy  |          18.173 |           17.570 |         17.717 |          17.782 |
| 100M  | Blosc2 |          25.119 |           25.850 |         34.475 |          35.525 |

## Trigonometric float output: `sin(x + 1) * cos(x - 2)`

| Items | Inputs | evaluate static | evaluate dynamic | Menudet static | Menudet dynamic |
|------:|:-------|----------------:|-----------------:|---------------:|----------------:|
| 100K  | NumPy  |           0.599 |            0.591 |          0.573 |           0.596 |
| 100K  | Blosc2 |           0.643 |            0.589 |          0.444 |           0.440 |
| 1M    | NumPy  |           1.236 |            1.100 |          1.662 |           1.492 |
| 1M    | Blosc2 |           2.402 |            2.404 |          1.595 |           1.871 |
| 8M    | NumPy  |           9.924 |            7.752 |         10.217 |           8.236 |
| 8M    | Blosc2 |          10.810 |           11.053 |         10.739 |          10.416 |
| 100M  | NumPy  |         102.857 |           96.940 |        102.791 |          95.702 |
| 100M  | Blosc2 |         117.521 |          105.617 |        121.230 |         120.683 |

## Trigonometric boolean output: `(sin(x + 1) * cos(x - 2)) > 1`

| Items | Inputs | evaluate static | evaluate dynamic | Menudet static | Menudet dynamic |
|------:|:-------|----------------:|-----------------:|---------------:|----------------:|
| 100K  | NumPy  |           0.656 |            0.676 |          0.622 |           0.666 |
| 100K  | Blosc2 |           0.574 |            0.621 |          0.408 |           0.463 |
| 1M    | NumPy  |           1.276 |            1.162 |          1.666 |           1.553 |
| 1M    | Blosc2 |           2.218 |            2.422 |          1.675 |           1.652 |
| 8M    | NumPy  |          10.341 |            8.153 |         10.579 |           8.545 |
| 8M    | Blosc2 |          10.972 |           11.163 |         10.227 |          10.347 |
| 100M  | NumPy  |         104.406 |          101.737 |        103.310 |          99.147 |
| 100M  | Blosc2 |         115.301 |          109.867 |        112.839 |         114.497 |

## Observations

- Resident 8M trigonometric evaluation takes about 21–22% less time. In the
  dynamic runs numexpr took 8.216 ms (float output) and 7.941 ms (boolean output),
  versus evaluate's 7.752 ms and 8.153 ms.
- Arithmetic gains are small or absent at large sizes. Compressed 8M arithmetic
  regresses about 5–7%; compressed 100M arithmetic about 3%.
- Blosc2-output columns also include compression scheduling and are not expected
  to follow resident NumPy-output gains. They remained approximately unchanged
  in the large cases (100M arithmetic LazyExpr: 35.055 to 34.610 ms).
- The initial adjacent-job queue regressed fresh 100M resident arithmetic to
  45–56 ms, while reused destinations remained near 14 ms. Interleaving initial
  claims across the allocation removed that first-touch regression without
  pre-populating or reusing outputs and without adding a static scheduler.
- Numerical qualification remains enabled in every benchmark. The boolean
  predicate is not constant-folded; both transcendental subtrees still execute.

## Validation

Native graph preparation/execution, allocation-failure tests, and ASan graph
tests pass. The combined Python scheduling, output, storage-reader, decorator,
dispatch, route, and benchmark qualification now passes all 388 tests with a
clean exit. Initial runs exposed a pre-existing ownership/GC crash, also
reproducible with the saved static build: GC cleared a borrowed artifact's owner
before its destructor, causing a second native free. A follow-up separates the
native ownership flag from that clearable Python reference and adds an explicit
unreachable-cycle regression. It does not change scheduling or benchmark timings.

Build warnings are the existing Cython shared-declarator, CBLAS deprecation,
generated-C qualifier/`abs`, dependency CMake policy, and duplicate-library
warnings. The scheduling C code adds no compiler warnings.
