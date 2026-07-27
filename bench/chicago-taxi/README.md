# Chicago taxi: a selective-query benchmark (Blosc2 `.b2z` vs Parquet)

One highly selective query (filter + projection + sort; 67 matches out of
24.3 M rows) against the flat [Chicago Taxi](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
dataset, stored in two on-disk formats (Parquet and Blosc2 `.b2z`) and answered
by five tools: DuckDB, PyArrow, pandas, polars, and Blosc2's `CTable.where()`.

The full write-up, methodology, and results live in
[`compare-query-methods.ipynb`](compare-query-methods.ipynb).

## Requirements

```bash
pip install "blosc2>=4.4.3" pyarrow duckdb polars pandas matplotlib jupyter
```

`blosc2` provides both the `CTable` container and the `parquet-to-blosc2`
CLI used to build the `.b2z` input. macOS or Linux only (the driver relies on
`/usr/bin/time`).

## Quick start

Open the notebook and run all cells:

```bash
jupyter lab compare-query-methods.ipynb
```

The notebook downloads the dataset on first run (~654 MB parquet, from
[cat2.cloud](https://cat2.cloud/demo)) and builds the `.b2z` from it
(~670 MB, a few seconds). Everything is re-runnable; existing files are
reused, not re-downloaded.

Or run the driver directly from a terminal:

```bash
# warm cache, best of 7
python compare-query-methods.py --nruns 7

# cold cache (flushes the OS file cache before every run; needs sudo)
sudo -v && python compare-query-methods.py --nruns 1 --purge
```

## Measuring a *cold* cache properly

Two gotchas the driver's `--purge` flag takes care of (and that you must handle
yourself if flushing manually):

1. **Flush before every timed run** — `sudo purge` (macOS) or
   `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` (Linux).
2. **Wake the disk before timing.** After a flush plus a few idle seconds, the
   first read pays the storage device's idle-state exit latency (tens of ms on
   NVMe drives with power management) — and it lands on whichever process
   touches the disk first, not on the engine you meant to measure. `--purge`
   reads a few MB of the *other* input file after each flush; manually, a
   `head -c 4000000 <some file> > /dev/null` right before the run does the job.

## Files

| File | Role |
|------|------|
| `compare-query-methods.ipynb` | the benchmark notebook: dataset download, `.b2z` build, cold + warm runs, plots, analysis |
| `compare-query-methods.py` | driver: runs each select script in a fresh subprocess under `/usr/bin/time`, checks row counts, writes the summary table and plots |
| `select-duckdb-flat.py` | the query in DuckDB SQL over parquet |
| `select-arrow-flat.py` | the query via PyArrow dataset scan over parquet |
| `select-pandas-flat.py` | the query via pandas (parquet read + NumPy filter/sort) |
| `select-polars-flat.py` | the query via polars lazy scan over parquet |
| `select-blosc2.py` | the query via `blosc2.open()` + `CTable.where()` over `.b2z` |
| `string-ops.py` | a separate benchmark: *string* kernels over the same dataset (see below) |

Each `select-*.py` prints the result, then `open:`/`compute:`/`print:`/`total:`
timings; the driver parses the `total:` line (query time, excluding interpreter
and import startup) alongside `/usr/bin/time`'s wall clock and peak memory.

## String ops (`string-ops.py`)

The numeric benchmark above is I/O-bound. `string-ops.py` is the opposite: it
loads the two *string* columns — `company` (`<U44`) and `payment.type`
(`<U11`) — into memory once and times three string workloads on each engine.
All engines must produce identical results or the run fails.

```bash
python string-ops.py                      # whole 24.3 M-row table, best of 3
python string-ops.py --nrows 1000000 --apply
python string-ops.py --engines blosc2,numpy --nrows 1000000
```

| task | expression |
|---|---|
| `filter` | `startswith(company, 'Taxi') & (payment_type != 'Cash')` → bool |
| `transform` | `'co=' + company + '\|pay=' + lower(payment_type)` → str |
| `kernel` | the same, branching on whether the company is a cab company |

All three are timed; only `kernel` is plotted. It is the point of the exercise:
the shape of the
[pandas-3 blog kernel](https://datapythonista.me/blog/whats-new-in-pandas-3),
row-wise control flow rather than one expression. blosc2 runs it as a
`@blosc2.dsl_kernel`; the other engines have to rewrite it as a mask plus two
fully-evaluated branches. `--apply` adds the row-wise pandas spelling, which is
what you would write first and is ~70x slower than everything else.

**blosc2 uses LZ4 at `clevel=5` with no filters**, rather than the stock
ZSTD-5 + SHUFFLE. Both are throughput-for-ratio trades and both are explained
under the results; the result is still 12x smaller than what the Arrow-backed
engines hold. **`blosc2 (raw)` is the identical path at
`clevel=0`** — same container, same kernel, same (empty) filter pipeline,
operands and result both uncompressed. Compression is the only variable between
the two blosc2 bars.

Results on an Apple M-series laptop (8 cores, 24 GB), full table, warm
(see `string-ops.png`):

| | filter | transform | kernel | kernel result |
|---|---|---|---|---|
| **blosc2** | **167 ms** | **1.10 s** | 3.11 s | **68 MB** |
| blosc2 (raw) | 224 ms | 1.21 s | 3.40 s | 5 766 MB |
| pandas | 193 ms | 2.07 s | 5.28 s | 932 MB |
| polars | 92 ms | 1.74 s | 3.83 s | 932 MB |
| duckdb | 337 ms | 1.98 s | 2.94 s | 842 MB |

blosc2 is **fastest of all five on `transform`** (1.80x DuckDB), **2.0x DuckDB
on `filter`**, and within 1.06x on `kernel` — while holding the result in **12x
less memory** than any of them. Only polars' `filter` is faster.

**Compression is free, and then some.** Compare the two blosc2 rows: the
compressed run is *faster* than the uncompressed one on all three tasks,
because a compressed block is less memory traffic than a 5.8 GB uncompressed
result. It also stores 85x smaller.

### Why no filters — a time/ratio tradeoff, not a fix

blosc2's default for `<U` is SHUFFLE with `filters_meta` 4: shuffle by the UCS4
code unit, which separates the ASCII payload byte from the three mostly-zero
high bytes. That is a good default and it wins on **ratio**. It costs time,
which is what this benchmark optimizes for. 1 M rows, blosc2 alone in the
process:

| codec | filters | filter | transform | kernel | result | operand cratio |
|---|---|---|---|---|---|---|
| ZSTD-5 | none | 10.0 ms | 80.3 ms | 149.3 ms | 1.08 MB | 860x |
| ZSTD-5 | **SHUFFLE meta=4** (default) | 15.9 | 80.4 | 155.5 | **0.75 MB** | **1321x** |
| LZ4-5 | none | 7.2 | 43.9 | 118.1 | 2.76 MB | 198x |
| LZ4-5 | SHUFFLE meta=4 | 12.4 | 50.5 | 124.8 | 2.70 MB | 225x |

Shuffle buys 1.4x ratio under ZSTD and ~2 % under LZ4, where the codec already
handles the zero runs. `filter` pays the most for it — that task writes 1 byte
per row, so the un-shuffle on the operand side has nothing on the output side
to offset it. **Keep the default when ratio matters; drop filters for
throughput**, which is the call made here because 2.8 MB against DuckDB's 34 is
an overwhelming memory win either way.

One trap regardless of which you pick: `filters_meta` is SHUFFLE's element
width, and a `<U` container picks 4 for itself only when you *don't* build a
`CParams`. Constructing one for any reason resets it to 0 — "shuffle by the
whole item" — which is strictly worse than both rows above: 6.6 MB and 75.8 ms
on the LZ4 `transform`.

Note the numbers above are lower than the table's: timing blosc2 in a process
that also runs the other engines costs it ~40 % even though it goes first. The
comparison table keeps every engine in one process, as it always has; use
`--engines blosc2` when tuning.

Four things got this from an earlier 8.49 s `kernel`, and two were bugs rather
than tuning:

1. **`upper`/`lower` stopped reserving a 3x/2x case-expansion bound** (miniexpr
   `5a7de4f`). NumPy does not reserve either — it truncates — so the result went
   `<U101` → `<U54`, halving every byte moved.
2. **Expression results were losing SHUFFLE's code-unit width.** Constructing a
   `CParams` defaults `filters_meta` to all zeros, i.e. "shuffle by the whole
   item", which scatters characters across the slot; left alone the container
   picks 4 for `<U` (the UCS4 code unit). Identical bytes compressed 3.2x worse
   on the expression path than through `asarray()`.
3. **Blocks are now sized for the result, not the operands** (see `BLOCKS` in
   the script). The result inherits the operands' block shape in *rows* and is
   much wider per row, so a row count tuned for `<U36` operands gave 1.7 MB
   blocks for the `<U54` result — out of cache on every task.

4. **LZ4-5 and no filters instead of the ZSTD-5 + SHUFFLE default**, as
   described below.

### The `<U` dtype used to cost 3.2x here (mostly fixed — see above)

The same kernel over the *same* blosc2 code path, with `S` (bytes) operands
instead of `<U`, at 1 M rows:

| | time | output |
|---|---|---|
| blosc2 `<U` | 300 ms | 404 B/row |
| **blosc2 `S`** | **114 ms** | **54 B/row** |
| duckdb | 111 ms | 35.9 B/row |
| polars | 137 ms | 39.7 B/row |

On `S`, blosc2 is at DuckDB parity and ahead of polars — with the result still
compressed to 2 MB. Three multiplicative factors inflate `<U`:

1. **UCS4 — 4 bytes per codepoint.** The others hold UTF-8, ~1 B/char here.
2. **The `lower()` width bound — 2x.** On `<U` it must reserve for Unicode
   full-case expansion (`ß`→`SS`), so `<U36`.lower() → `<U72` and the result is
   `<U101` where 54 suffices. On `S`, case mapping is ASCII-only and 1:1, so the
   bound is exact — that is most of the `S` win.
3. **Fixed-width padding.** Mean result length is 31.7 chars in a 101-char slot.

Even on `S`, blosc2 holds 54 B/row (the compile-time max, on every row) against
DuckDB's 35.9 (31.7 data + 4 offset + 0.1 validity) — they pay the mean plus an
offset. That residual 1.4x is what native variable-width output would remove.

Everything else measured small: operand decompression 23 ms; per-op interpreter
cost ~10 ns/row/op (~50 ms of the 300 for this 5-op kernel); `lazyudf`
construction ~0. Thread scaling is 3.9x on 8 cores, consistent with being
bandwidth-bound on the wide output.

**Practical advice: use `S` for ASCII/Latin-1 string columns.** It is available
today and already reaches DuckDB parity on this workload.

NumPy is implemented but **off by default**: its `kernel` builds five full-width
`<U` temporaries, ~10 GB each at 24 M rows. Run it with `--engines` at a smaller
`--nrows` — at 1 M rows it is 718 ms and 417 MB, losing on both counts.
