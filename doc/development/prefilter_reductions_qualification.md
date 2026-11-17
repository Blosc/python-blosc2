# Developing prefilter reductions

Python-Blosc2 owns dispatch, scheduling and bounded buffers. Miniexpr's
`ME_EXTREMA_REDUCE_API` in `miniexpr_reduce.h` owns floating-extrema rules and
index tie-breaking. Older runtimes retain the graph-reader or chunked path.
Numerical tests belong in miniexpr; Python tests cover bindings, API contracts,
scheduling, fallback admission and execution errors that must not replay.

## Statistical reductions

`mean()` reuses the fixed-leaf pairwise sum path and preserves its output dtype.
With miniexpr's optional `ME_MOMENTS_REDUCE_API`, full-domain 1-D float32/float64
`var()` and `std()` can decode/map each block once. Miniexpr computes centered
block moments, merges them in logical order, and applies `ddof` and square root.
Float32 moments accumulate in double precision before casting the final result.
The origin and mean offset remain separate during merging to avoid rounding
away small variation on large-offset inputs.

This is not bit-identical to the existing two-pass reduction. Explicit
accuracy/backend policies, dtype changes, masks, slices, unsupported axes,
invalid degrees of freedom, and missing native capabilities retain the existing
implementation. Each block uses four doubles of partial storage, subject to
the same 1 MiB admission cap.

## Benchmarking

Run in the `blosc2` conda environment. Keep optional per-run output outside the
checkout, in temporary storage or CI artifacts:

```sh
results_dir="$(mktemp -d)"
conda run -n blosc2 python bench/ndarray/prefilter_reductions.py \
  --size 16000003 --repeats 3 --threads 1 4 12 \
  --output "$results_dir/prefilter-reductions.json"
```

The script rotates route order after warmup and requires bit-identical results
for extrema/truth/index reducers. Use `--ops mean var std` to benchmark statistics
with dtype-appropriate tolerances (including absolute error for near-zero means).
It covers codecs, float dtypes, plain/computed expressions and thread counts.
The `prefilter` timing uses automatic dispatch: inspect `prefilter_selected` and
the actual route before interpreting speedups. The baseline disables prefilter
admission; index reductions then use the chunked implementation.
For `var()`/`std()`, the baseline disables only the moments capability, retaining
the existing accelerated two-pass implementation.

Prefilters schedule one chunk at a time, so compare worker counts against work
per chunk, not just total array size. Partial storage is capped at 1 MiB,
including both values and indices for index reductions.

## Validation

Run the default suite and relevant heavy reduction tests. Check native
primitives under sanitizers, and independently test wheels with and without
the optional native capability. For isolated wheel imports, use conda Python
with `-S`, explicit package/dependency paths and `pytest -n 0`. Run doctests in
a fresh temporary working directory to avoid collisions with existing files.
