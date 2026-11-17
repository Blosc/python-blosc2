# Prefilter reduction qualification

## Scope and native dependency

Floating extrema and index reductions use miniexpr's `ME_EXTREMA_REDUCE_API`
from `miniexpr_reduce.h`. Python-Blosc2 owns dispatch, scheduling and bounded
buffers; miniexpr owns value comparison, NaN selection and index tie-breaking.
There is no local Python-Blosc2 numerical fallback for floating extrema.
Runtimes without that native capability keep the existing graph-reader or
chunked index-reduction path.

The bundled native revision, `c720b6970f00b6b57c1aad48ff6b31d4d4023a62`,
exposes this capability. Initial qualification used its parent revision
(`efcaf3bf236cc04ee6dfd3644ef2a03f78dcad4f`) with the same native reduction
changes overlaid via `FETCHCONTENT_SOURCE_DIR_MINIEXPR`. A wheel using the older
revision without that overlay was checked separately for fallback compatibility.

## Reproduce

Use the `blosc2` conda environment and a native source tree containing the new
extrema API as well as the existing fixed-leaf pairwise API:

```sh
conda run -n blosc2 pip install -e . --no-build-isolation \
  --config-settings=build-dir=build-reduction-qualification \
  --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=/path/to/native-source
conda run -n blosc2 python bench/ndarray/prefilter_reductions.py \
  --size 16000003 --repeats 3 --threads 1 4 12 \
  --output doc/development/prefilter_reductions_qualification.json
```

The benchmark fixes chunk/block sizes at 262144/8192 items and rotates route
order after warmup. It covers BLOSCLZ and ZSTD, float32/float64, plain inputs and
`a + a`, and six operators. Every timed result must be bit-identical across
routes. The JSON stores timings, actual routes, worker counts and environment.
This is a local crossover study, not multi-platform performance certification.

The `prefilter` timing is the **automatic** choice: a conservative scheduling
gate may retain the fallback. Check `prefilter_selected` rather than assuming
every case runs a prefilter. The baseline disables prefilter admission; for
`argmin/argmax` it is the existing chunked implementation, not a graph reducer.

## Local results

144 cases, 16,000,003 items, three timed repetitions per route. Median speedups
across codecs, dtypes and expressions:

| Operator | 1 thread | 4 threads | 12 threads |
|---|---:|---:|---:|
| min | 1.34x | 1.18x | fallback |
| max | 2.81x | 2.38x | 1.44x |
| any | 1.86x | 1.48x | fallback |
| all | 1.56x | 1.30x | fallback |
| argmin | fallback | 1.91x | 2.45x |
| argmax | fallback | 1.92x | 2.49x |

Two qualification findings changed the implementation:

* Dispatching the arg-extrema dtype once per span rather than per lane removed
  a severe scan-loop regression.
* Each prefilter invocation schedules one chunk, not the whole input. Cheap
  reducers retain independent chunk-region readers when the requested worker
  count exceeds the work available in one chunk. Max retains its prefilter
  with a chunk-sized worker budget. Single-thread arg-extrema keeps the
  existing implementation.

All partial buffers are capped at 1 MiB. Index reductions count both value and
index storage against that limit.

## Validation layers

Native tests cover typed comparisons, integer extremes, NaN payloads,
signaling-NaN quieting, signed-zero rules, unaligned input, partition merges,
completion-order independence and noncanonical boolean storage. The standalone
native primitive tests also run with address/undefined-behavior sanitizers.

Python tests cover binding validation, direct-buffer dispatch, tie positions
across chunks, keepdims, integer binding, thread scheduling, fallback admission,
scratch limits and execution errors that must not replay another route.
Run the default suite and the heavy reduction module in the conda environment.
Independent wheels with and without the new native capability are imported in
isolated processes to ensure source tests do not accidentally load an editable
extension.

Local validation results:

* Default parallel suite: **13,237 passed**, 109 skipped, 3 xfailed.
* Heavy reduction module: **5,108 passed**.
* Focused native reduction/graph CTest selection: **12 passed**; new primitive
  tests also pass under ASan/UBSan with `-Wall -Wextra -Werror`.
* Native-capability wheel: the full isolated API test run passed; its save
  doctest initially encountered an existing `array.b2frame` in the checkout.
  Re-running packaged doctests in a fresh temporary directory passed
  **122 tests**, with 2 skipped, without touching that file.
* Previous-pin wheel: affected API suites passed with the capability correctly
  disabled and unsupported fast-path cases skipped.

An unfiltered native CTest run exceeded the 120-second command limit; the
focused native selection above is the completed native regression check.
