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

Each `select-*.py` prints the result, then `open:`/`compute:`/`print:`/`total:`
timings; the driver parses the `total:` line (query time, excluding interpreter
and import startup) alongside `/usr/bin/time`'s wall clock and peak memory.
