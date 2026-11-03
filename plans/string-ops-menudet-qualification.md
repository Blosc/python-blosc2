# Chicago Taxi string benchmark: public Menudet migration

Date: 2026-11-03. This is a bounded local correctness/performance observation,
not full-table or cross-platform qualification.

**Historical stage:** the measurements below describe the initial serialized
migration before the native prefilter/arena fixes. Updated full-table kernel
controls and six-engine smoke results are in the
[profiling/fix report](string-ops-menudet-prefilter.md). Keep these original
samples separate from the later implementation.

## What changed

- Replaced internal `SourceKernel`/`lazyudf` authoring with a public
  `@blosc2.menudet(jit=False)` kernel and explicit Unicode input/output types.
- The longest branch reserves `company width + payment width + 7` characters;
  specialization follows the actual loaded widths, not a hard-coded table width.
- Logical partitions, stored output chunks and input chunks match. The public
  string route uses general partition-preserving materialization, not the numeric
  native-prefilter shortcut. There is no Python-function execution fallback.
- Preserved the filter/transform expressions, codec/filter settings, raw arm and
  NumPy/pandas/polars/DuckDB baselines. Fixed the driver's source-lifetime issue
  when `--apply` follows the raw arm.
- Added optional JSON output with raw samples, fingerprints and runtime/storage
  metadata. The default figure now has a distinct Menudet filename. Historical
  full-table artifacts and measurements were not overwritten or relabeled.

## Correctness and scope

Small tests exercise 4,103 rows across chunks, blocks and a short final chunk:
suffix/interior cab matches, empty inputs, a full-width result, and a non-ASCII
one-to-one lowercase example. Both compressed/raw arms exactly match independent
Python and NumPy values; filter/transform results are also checked. Tests cover
explicit output width, author-cache reuse and rejected input types/widths.

The real-data comparison uses the first 100,000 rows of the existing local
`chicago-taxi-flat.parquet`. Loaded dtypes are `U31`/`U11`, with kernel output
`U49`; these are not the historical full-table input widths. All six variants
pass the existing cross-engine fingerprints: all row lengths plus sampled string
content, and all Boolean output values. That fingerprint is not an exhaustive
comparison of every character. The small synthetic tests do compare full values.

This does not qualify every Unicode case-expansion rule or byte-string input.
The benchmark explicitly accepts fixed-width Unicode, matching its dataset loader.

## Reproduction and observations

From the repository root, with the matching compiled extension:

```sh
conda run -n blosc2 python bench/chicago-taxi/string-ops.py \
  --parquet bench/chicago-taxi/chicago-taxi-flat.parquet \
  --nrows 100000 --nruns 3 \
  --engines 'blosc2,blosc2 (raw),numpy,pandas,polars,duckdb' \
  --plot bench/chicago-taxi/string-ops-menudet-100k.png \
  --output bench/chicago-taxi/string-ops-menudet-100k.json
```

macOS ARM, 12 configured Blosc2 threads, Python 3.14.4 in the `blosc2` conda
environment. Package and extension identities are recorded in the JSON.
One untimed pass followed by best-of-three observations, including output
allocation/compression; preparation is not reported as a separate measurement.
These are single-process, fixed-order observations, not independent allocations.

| Engine | Kernel best time | Kernel result footprint |
| --- | ---: | ---: |
| Blosc2 LZ4-5 | 83.0 ms | 0.274 MiB |
| Blosc2 raw | 82.5 ms | 24.5 MiB |
| NumPy | 36.5 ms | 18.7 MiB |
| pandas | 15.4 ms | 4.4 MiB |
| polars | 9.3 ms | 4.4 MiB |
| DuckDB | 8.7 ms | 4.0 MiB |

The current general string-kernel path is materially slower than the other
engines in this bounded comparison. Do not carry over historical claims about
legacy-kernel throughput, rankings or compression costs. This run does not isolate
the cost of authoring versus execution/materialization, or prove a regression
against a matching historical build.

Current evidence:

- [Raw timing samples and metadata](../bench/chicago-taxi/string-ops-menudet-100k.json)
- [Current 100k-row figure](../bench/chicago-taxi/string-ops-menudet-100k.png)
- A separate 1,000-row `--apply` smoke run with both Blosc2 arms also passed.

## Remaining work

- Profile and, if justified, improve the public string execution/materialization
  path; do not bypass native safety gates or restore private authoring for speed.
- Budget a full 24.3M-row comparison and new codec/filter sweeps before replacing
  the historical full-table figure, tables and recommendations.
- Record additional dependency/build identities and isolated resource costs for
  publishable performance qualification. The current JSON is smoke-run evidence,
  not a complete native build manifest or input-data archive.
- Preserve fixed-width string semantics and independent reference checks in any
  subsequent optimization.
