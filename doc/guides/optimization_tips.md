# Optimization tips

Seven small idioms that make a measurable difference in either speed or memory
(often both), pulled from recent `blosc2` release notes. Each one is backed by a
small benchmark in
[`bench/optim_tips/`](https://github.com/Blosc/python-blosc2/tree/main/bench/optim_tips)
that times a naive approach against the recommended one and plots peak time and
memory for both. Run any of them yourself — see the
[bench/optim_tips README](https://github.com/Blosc/python-blosc2/tree/main/bench/optim_tips/README.md).

Numbers below were measured on an Apple M2 laptop (macOS, Python 3.14); absolute
values will differ on your machine, but the direction and rough magnitude of each
effect should not.

## 1. Build large arrays with blosc2's own constructors

`blosc2.arange()`, `blosc2.linspace()` and `blosc2.fromiter()` fill an `NDArray`
chunk-by-chunk. Building the same array in NumPy first and compressing it via
`asarray()` means holding the whole thing uncompressed in memory at once.

```python
# Avoid: materializes the full array in NumPy first
a = blosc2.asarray(np.linspace(0, 1, N))

# Prefer: fills the NDArray chunk-by-chunk
a = blosc2.linspace(0, 1, N)
```

![blosc2.linspace() vs asarray(np.linspace())](optim_tips/tip_01_constructors.png)

At 200M float64 elements, `blosc2.linspace()` was modestly faster (0.56s vs
0.76s) and used **~25x less peak memory** (63 MiB vs 1.5 GiB) — the gap widens
with array size, since the naive path's memory is O(N) while the constructor's
stays roughly O(chunk size). This applies equally to `arange()` and `fromiter()`.

## 2. Align slices with the chunk grid

`NDArray.slice()` has a fast path when a slice's boundaries land exactly on
chunk boundaries: whole chunks are copied directly, with no
decompress/recompress. A slice that starts or ends mid-chunk falls back to the
general path.

```python
arr = blosc2.asarray(data, chunks=(4000, 2000))

# Avoid: mid-chunk boundaries force decompress + recompress
arr.slice((slice(500, 12500), slice(None)))

# Prefer: boundaries match the chunk grid -> whole chunks copied as-is
arr.slice((slice(4000, 16000), slice(None)))
```

![NDArray.slice(): chunk-aligned vs unaligned](optim_tips/tip_02_chunk_aligned_slicing.png)

The aligned slice was **~48x faster** (1ms vs 57ms) on a 16000x2000 float64
array. Memory use was comparable between the two — this tip is purely about
avoiding wasted decompression/recompression work, not about avoiding
materialization (neither path produces a NumPy array). If you control chunk
sizes, pick slice boundaries — or a `chunks=` shape — that line up with how you
plan to read the array.

## 3. Sorted top-k via `sort_by(view=True)`

`CTable.sort_by(view=True)` returns a lightweight sorted *view* that gathers
rows from the parent on demand, instead of materializing a whole sorted copy of
the table. On a column with a `FULL` index it streams straight from the index,
so the table is never actually sorted.

```python
t.create_index("temperature", kind=blosc2.IndexKind.FULL)

# Avoid: sorts (and copies) the whole table just to keep 10 rows
top10 = t.sort_by("temperature")[:10]

# Prefer: zero-copy view, streamed from the index
top10 = t.sort_by("temperature", view=True)[:10]
```

![CTable.sort_by(view=True) top-10](optim_tips/tip_03_sort_by_view.png)

On a 2M-row table, the view form took **15ms vs 733ms** for a full sort — a
**~50x** speedup — while also using about 25% less peak memory (34 MiB vs 46
MiB). The larger the table relative to *k*, the bigger this gap gets, since the
naive path's cost is dominated by sorting rows you're about to discard.

## 4. Let SUMMARY indexes answer `min()`/`max()` directly

Closing a `CTable` auto-builds `SUMMARY` indexes (per-block min/max) for its
eligible scalar columns. `Column.min()`/`max()` (and `argmin()`/`argmax()`
inside `group_by()`) then answer from those precomputed summaries instead of
decompressing the column at all.

```python
# create_summary_index=True is the default; closing the table builds the index
with blosc2.CTable(Row, urlpath="t.b2d", mode="w") as t:
    t.extend(data)

t = blosc2.CTable.open("t.b2d")
hottest = t["temperature"].max()  # answered from the SUMMARY index
```

![Column.max() with vs without a SUMMARY index](optim_tips/tip_04_summary_index_where.png)

On a 10M-row column, the indexed `max()` took **5ms vs 26ms** without an index
(~5x) and needed essentially no extra memory (56 KiB vs 30.5 MiB) — it never
touches the compressed column data at all. The same SUMMARY indexes can also
let a selective `where()` query skip whole blocks, but only when the column's
values are ordered or clustered enough that a block's min/max range excludes
it entirely; with independently random data every block spans nearly the full
value range and there's nothing to skip, so we don't show that case here — the
`min()`/`max()` win is the reliable one.

## 5. Compute on compressed columns via `Column.raw`

`Column.__getitem__` (`t["col"][:]`) always materializes a full NumPy array,
with null-sentinel processing applied. `Column.raw` returns the underlying
compressed `NDArray` directly; its own reduction methods (`sum()`, `mean()`,
...) work chunk-by-chunk and never hold the whole column decompressed at once.

```python
# Avoid: decompresses the whole column into one NumPy array first
total = t["val"][:].sum()

# Prefer: chunk-wise reduction straight over the compressed column
total = t["val"].raw.sum()
```

![Column.raw.sum() vs col[:].sum()](optim_tips/tip_05_column_raw.png)

On a 50M-row column, `raw.sum()` was modestly faster (88ms vs 126ms) and used
**~12x less peak memory** (31 MiB vs 382 MiB). `Column.raw` has no
null-processing, and physically-stored fixed-width columns can be
over-allocated to chunk capacity — slice to `len(t)` if you need exactly the
logical rows, and it raises `AttributeError` for computed columns, which have
no backing storage.

## 6. Memory-map read-only opens

`blosc2.open(path, mmap_mode="r")` memory-maps a file instead of going through
regular file I/O for every chunk access. For workloads that touch many
scattered chunks, mapping once avoids repeated open/seek/read syscalls per
access.

```python
# Avoid (for read-heavy, scattered access): regular I/O per chunk
arr = blosc2.open(path)

# Prefer: map the file once, read pages directly
arr = blosc2.open(path, mmap_mode="r")
```

![open(mmap_mode='r') vs plain open()](optim_tips/tip_06_mmap_read.png)

Across 8,000 scattered slice reads, `mmap_mode="r"` was **~1.2x faster** (2.8s
vs 3.2s); peak memory was essentially identical for this single-process,
single-open workload. The bigger real-world payoff of `mmap_mode` shows up
with cold OS caches and with multiple readers/processes sharing one file,
where mapped pages are shared rather than each reader paying its own I/O and
buffer-copy cost — a single warm-cache process, as benchmarked here, is the
worst case for showing it off. `mmap_mode="r"` only works with `mode="r"` on an
existing file; see the
[Sharing containers across processes](sharing_across_processes.rst) guide for
the multi-reader/NFS/Windows caveats.

## 7. Skip the transient decompress in chunked `extend()`

Passing a `blosc2.NDArray` as a column value to `CTable.extend()` writes it
chunk-by-chunk rather than fully decompressing it upfront — except that, by
default, `extend()` still does one transient full decompression to run
constraint/nullability checks. `validate=False` skips that check, for a column
you already know is valid.

```python
# Avoid: extend() still fully decompresses src once for validation
t.extend({"val": src})

# Prefer: skip the validation pass on a known-good NDArray column
t.extend({"val": src}, validate=False)
```

![CTable.extend(validate=False)](optim_tips/tip_07_chunked_writes.png)

Extending a table with a 20M-row `NDArray` column, `validate=False` was
**~1.3x faster** (99ms vs 128ms) and used **~2.7x less peak memory** (63 MiB vs
168 MiB) by skipping that transient decompression. The matching `col[:] =
ndarray` fast path on `Column.__setitem__` gives the same chunk-by-chunk write
behavior for updating an existing column.
