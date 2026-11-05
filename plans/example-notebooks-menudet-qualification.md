# Legacy example notebooks: Menudet qualification

Date: 2026-11-05

Scope: section E of [migrate-benchs-menudet.md](migrate-benchs-menudet.md). These
were the last three legacy example notebooks using retired `@dsl_kernel`
authoring.

Files migrated and executed:

- `examples/ndarray/black-scholes_hist-dsl.ipynb`
- `examples/ndarray/mandelbrot-dsl.ipynb`
- `examples/ndarray/mandelbrot-pure-dsl.ipynb`

## Environment

- Apple Mac mini, Apple M4 Pro, 12 cores, 24 GiB RAM, macOS/Darwin 25.6.0 arm64.
- Python 3.14.4; NumPy 2.5.4; Numba 0.68.0; Matplotlib 3.10.8.
- Editable Python-Blosc2 4.15.0.dev0 from this checkout:
  `src/blosc2/__init__.py`.
- C-Blosc2 3.5.0 (`$Date:: 2026-11-04 #$`); extension
  `blosc2_ext.abi3.so` from the `blosc2` conda environment.
- 12 execution threads; default FP policy; system `cc` is Apple clang 21.0.0.
- Notebook execution used `jupyter nbconvert --execute`, so all published outputs
  below come from the current implementation.

## Notebook changes

### `black-scholes_hist-dsl.ipynb`

- Ported both kernels to `@blosc2.menudet`.
- Black-Scholes captures `X`, `risk_rate` and `sigma` at decoration instead of
  passing them as scalar arguments. Scalar arguments are currently represented as
  0-d operands, which prevents the dense storage prefilter for this kernel. A
  1e7-element probe measured about 0.5 s with scalar arguments versus about 0.03 s
  with captures. The notebook explains the trade-off; annotated scalar parameters
  remain supported when configuration must vary per call.
- The bin-edge kernel uses `ndim=1`, `_i0`/`_n0` coordinates and annotated scalar
  arguments. Edges are `float64`, matching NumPy exactly.
- The bin-index kernel captures the data range and bin count so it can use
  `menudet-prefilter`. Its output is `float32`, exact for every index below
  `2**24` (1000 bins) and the same size as the old `int32` result. The counting
  UDF converts indices to integers.
- The generic Numba counting UDF remains a `blosc2.lazyudf` step. The prose now
  explains elementwise versus block-scalar cardinality and why a histogram is not
  a partition reduction.
- Added seed 0 for the random histogram input and explicit correctness checks.
- Regenerated all timings, plots and prose.

### `mandelbrot-dsl.ipynb`

- Ported the kernel to `@blosc2.menudet` with a captured iteration bound.
- Seeds `zr`/`zi` with explicit `np.float32(0.0)`. Using a Python `0.0` literal
  promoted intermediates and produced 1576 differing pixels at 1200x800x200; the
  explicit float32 seeds match Numba exactly.
- Retained the ordinary Numba `lazyudf` arm and the native Numba baseline.
- Added exact-equality checks, escape-range reporting, a results discussion and
  regenerated images/timings.

### `mandelbrot-pure-dsl.ipynb`

- Ported authoring to `@blosc2.menudet`; execution policy is selected with
  `blosc2.jit_options`.
- Replaced the old `cparams.nthreads=1` arm with interpreter / `tcc` / `cc` arms.
  `CParams(nthreads=...)` and `blosc2.set_nthreads(1)` do not partition Menudet
  execution, so the old "1 thread" comparison was not meaningful for this kernel.
- Added exact-equality checks and a results discussion; regenerated plots and
  timings.

## Correctness

- Black-Scholes: `max |native Numba - Menudet| == 0`; the first 4096 prices match
  the independent NumPy reference (`atol=1e-12`); the `cc` and `tcc` results also
  agree exactly.
- Histogram: counts and bin edges match the Numba reference exactly for seed 0,
  1e8 samples and 1000 bins. The `float32` bin indices are exact in this range.
- Mandelbrot (`mandelbrot-dsl`): all four routes are bit-identical; escape range
  1..200; geometry 800x1200 `float32`.
- Mandelbrot (`mandelbrot-pure-dsl`): interpreter, `tcc` and `cc` all match the
  NumPy UDF exactly; escape range 1..200.

## Measurements (best of 3 after warmup)

All times are seconds on the host above. These are one fresh run, not universal
constants.

### Black-Scholes, N = 1e8

| Route | First run | Best |
|---|---:|---:|
| NumPy baseline | 2.907 | 2.907 |
| Native Numba | 0.505 | 0.152 |
| Menudet `cc` | 0.291 | 0.275 |
| Menudet `tcc` | 0.960 | 0.933 |

Steady state: Numba is fastest; Menudet `cc` is 1.80x slower than Numba and about
10.6x faster than NumPy. `tcc` is 6.14x slower than Numba. Menudet `cc` has the
lower first-run cost (0.291 s versus 0.505 s for Numba).

### Histogram, N = 1e8, 1000 bins

| Route | First run | Best |
|---|---:|---:|
| NumPy baseline | 0.435 | 0.435 |
| Native Numba | 0.828 | 0.086 |
| Menudet pipeline `cc` | 0.263 | 0.138 |
| Menudet pipeline `tcc` | 0.218 | 0.205 |

Steady state: Numba's dedicated histogram is fastest; the Menudet pipeline is
1.61x slower (`cc`) and 2.39x slower (`tcc`), but about 3.1x faster than NumPy.
The remaining cost is the separate bin-index materialization and serial counting,
not the compiled edge/index kernels.

### Mandelbrot DSL vs Numba, 1200x800, max_iter = 200

| Route | First run | Best |
|---|---:|---:|
| Native Numba | 0.372 | 0.020 |
| Blosc2+Numba UDF | 0.187 | 0.021 |
| Menudet `cc` | 0.039 | 0.036 |
| Menudet `tcc` | 0.073 | 0.065 |

With early-exit loops admitted to the partition proof, Menudet partitions across
storage worker threads: Menudet `cc` is now only 1.85x slower than native Numba
(down from 15x slower previously), and `tcc` is 3.31x slower (down from 27x
slower). Menudet `cc` also has a very low first-run cost (0.039 s vs 0.372 s for
native Numba and 0.187 s for Blosc2+Numba).

### Mandelbrot pure Blosc2, 1200x800, max_iter = 200

| Route | First run | Best |
|---|---:|---:|
| Blosc2+NumPy UDF | 1.584 | 1.599 |
| Menudet interpreter (`jit=False`) | 1.692 | 1.630 |
| Menudet `tcc` | 0.082 | 0.067 |
| Menudet `cc` | 0.041 | 0.038 |

The JIT backends are 23.7x (`tcc`) and 41.7x (`cc`) faster than the NumPy UDF
(up from 3.1x and 6.0x previously). Partitioned interpretation is also ~10x
faster than before (1.63 s vs 15.6 s), running neck-and-neck with the vectorized
NumPy UDF.

## Discussion changes and open limitations

- The retired DSL prose in the Black-Scholes notebook was removed. Menudet
  supports elementwise results and block-scalar reductions; a histogram is a
  scatter/binning operation and still needs a generic counting UDF.
- The Black-Scholes prose now distinguishes authoring from execution policy and
  reports the current host-specific rankings instead of the old DSL ratios.
- Scalar *arguments* currently prevent the dense storage prefilter. Captures are
  used in these notebooks. If scalar arguments become admissible, the capture
  workaround can be simplified and re-measured.
- Early-exit loops (`break`) are now admitted by Menudet's native partition proof
  (`native_partitionable=True`), allowing the 2-D kernel to partition across
  storage workers. This restores multi-threaded throughput and brings Menudet `cc`
  within 1.85x of native Numba (with a much faster cold start).
- The old saved DSL timings and the old "1 thread" comparison were not reused.

## Reproduction

```sh
conda run -n blosc2 jupyter nbconvert --to notebook --execute --inplace \
  examples/ndarray/black-scholes_hist-dsl.ipynb
conda run -n blosc2 jupyter nbconvert --to notebook --execute --inplace \
  examples/ndarray/mandelbrot-dsl.ipynb
conda run -n blosc2 jupyter nbconvert --to notebook --execute --inplace \
  examples/ndarray/mandelbrot-pure-dsl.ipynb
```
