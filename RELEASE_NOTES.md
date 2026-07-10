# Release notes

## Changes from 4.7.0 to 4.8.0

### Sharing containers across processes

- New `locking` storage parameter (and the `BLOSC_LOCKING` environment
  variable to enable it fleet-wide) serializes accesses to an on-disk
  `SChunk`/`NDArray`/`EmbedStore`/`DictStore` against other handles and other
  processes, via a small sidecar lock file (`.b2lock`). Advisory: every
  handle touching the container must opt in.
- `SChunk.holding_lock()` / `NDArray.holding_lock()`: a context manager to
  hold the exclusive lock across several operations, making a multi-step
  mutation atomic to other locked handles.
- New `SChunk.refresh()`, mirroring the existing `NDArray.refresh()`.
- Fixed a data-loss bug in `NDArray.append()`: it read the cached,
  unrefreshed shape before computing the resize target, so under
  concurrent growth/shrink — even inside `holding_lock()` — another writer's
  just-appended data could be silently deleted.
- `EmbedStore` and `DictStore` (`.b2d`) now support cross-process writers
  under locking: transactional writes plus key-map re-sync, so readers
  follow keys added or removed by another process.
- `DictStore.to_b2z()` (and `TreeStore`, which inherits from it) now replaces
  the target file atomically, so concurrent readers always see either the old
  or the new archive, never a torn one.
- Growth-SWMR (single writer, multiple readers): a reader `NDArray` handle
  opened before a `resize()` made through another handle follows the new
  shape on its next data access, or via the new explicit `NDArray.refresh()`.
- New user guide page,
  [Sharing containers across processes](https://www.blosc.org/python-blosc2/getting_started/sharing_across_processes.html),
  covering all of the above plus the caveats (NFS, `mmap_mode`, Windows
  in-use-file rename).

### Bug fixes

- Fixed `detect_aligned_chunks()` (used internally to fast-path aligned
  slice reads/writes): a floor-division undercounted the chunk grid for
  arrays whose shape isn't a multiple of the chunk shape, which could
  silently return the wrong chunk's data for an otherwise-aligned slice
  with a nonzero start in an earlier dimension.

### Others

- Raised the manylinux wheel baseline from `manylinux2014` (CentOS 7, glibc
  2.17, GCC 10.2) to `manylinux_2_28` (AlmaLinux 8, glibc 2.28, GCC 12),
  fixing a build failure with NumPy >=2.5 which requires GCC >=10.3.

## Changes from 4.6.0 to 4.7.0

### DSL → JavaScript backend for WebAssembly (`jit_backend="js"`)

- Under WebAssembly/Pyodide, `@blosc2.dsl_kernel` kernels can now be transpiled
  to JavaScript and run via the browser's JIT.  It is the **default** there for
  transpilable floating-point kernels (silently falling back to miniexpr for
  anything it can't handle), and beats the WASM TinyCC JIT on compute-heavy
  kernels (e.g. ~2.8x on a Newton-fractal kernel).  Request it explicitly with
  `compute(jit_backend="js")`; outside WebAssembly that raises.
- Supports index/shape symbols (`_i0`/`_n0`/`_ndim`/`_flat_idx`) and integer inputs
  with a floating-point output.  Integer/complex *output*, reductions, and
  unsupported constructs stay on miniexpr.  Native builds are unaffected.

### New `blosc2.validate_dsl_jit()`

- A new introspection helper reports whether a DSL kernel actually JIT-compiles
  (vs. silently falling back to the interpreter) for a given set of operand and
  output dtypes, without running it on real data::

      status = blosc2.validate_dsl_jit(kernel, [np.float64, np.float64], np.float64)
      status["jit"]  # True if a runtime JIT kernel was produced

### miniexpr fixes

- `DSLValidator` now rejects `;`-joined sibling statements with a clear
  "one statement per line" error, and assigning to an input parameter raises a
  targeted error naming the param.
- Fixed (in miniexpr) a name collision where DSL variables named `out`, `idx`,
  `nitems`, `inputs` or `output` clashed with codegen-internal identifiers,
  causing the generated C to fail to compile and silently fall back to the
  interpreter.  Codegen identifiers are now namespaced under `__me`.

## Changes from 4.5.1 to 4.6.0

### `CTable.sort_by(view=True)`: zero-copy sorted views

- `CTable.sort_by()` now accepts **`view=True`**, returning a lightweight
  **sorted view** that shares the parent's column data and gathers rows on
  demand in sorted order — no whole-table copy.  This is ideal for reading a
  sorted slice of a large (possibly on-disk) table::

      t.sort_by("col", view=True)[:10]      # top-10 without materialising

  Sorting on a **fully indexed** column streams directly from the index, so the
  table is never materialised.  Multi-column sorts and dotted (nested) leaf
  names are supported (e.g. `t.sort_by(["trip.begin.lon", "payment.fare"],
  ascending=[True, False])`).

### `where` on dictionary (string) columns

- `where` expressions now work over **dictionary-encoded (string) columns**,
  including membership tests such as `'"Acme" in company'`, so categorical
  text columns can be filtered without decoding the whole column.

### `b2view` is now an opt-in extra

- The `b2view` terminal browser and its TUI stack (`textual`,
  `textual-plotext`) are no longer core dependencies: a plain
  `pip install blosc2` no longer pulls them, keeping the compression library
  lean (and dropping deps that are unusable under wasm32, which has no TTY).
  Install the viewer with `pip install "blosc2[tui]"`, or
  `pip install "blosc2[hires]"` to also get the high-res `h` view.  The
  `b2view` command prints this hint if the dependencies are missing.

### `group_by`: flexible aggregation naming

- `CTable.group_by(...).agg()` now accepts a **list of `(column, ops)` pairs**
  and **explicit output names** (pandas-style keyword arguments), alongside the
  existing auto-suffixed mapping; the forms can be combined::

      g.agg({"sales": ["sum", "mean"]})              # auto: sales_sum, sales_mean
      g.agg([(t.sales, ["sum", "mean"])])            # auto, but accepts Column objects
      g.agg(revenue=("sales", "sum"))                # explicit: revenue
      g.agg({"sales": "sum"}, n=("*", "size"))       # combined, with a named row count

  The list-of-pairs and named forms accept `Column` objects (`t.sales`), which
  the mapping form cannot because `Column` is unhashable and so cannot be a dict
  key.
- Aggregation ops may also be given as the matching blosc2 reduction *functions*
  (`blosc2.sum`, `mean`, `min`, `max`, `argmin`, `argmax`), matched **by
  identity** -- e.g. `g.agg([(t.sales, [blosc2.sum, "mean"])])`.  This is a
  naming shorthand only; arbitrary/UDF callables (and look-alikes such as
  `np.sum` or a user function named `sum`) are rejected rather than silently
  misinterpreted.

### `group_by` / `group_reduce`: tri-state `sort=`

- **Vectorized dictionary group ordering**: `group_by()` result building now
  batch-decodes dictionary (string) keys in one pass (`decode_batch`) instead of
  one `decode()` per group, making high-cardinality string group-bys dramatically
  faster (end-to-end `group_by().size()` dropped from seconds to milliseconds on
  ~100k-group workloads).
- **`sort=` is now a tri-state** (`None` / `True` / `False`) on both
  `CTable.group_by()` and `blosc2.group_reduce()`:
  - `True` — always return groups sorted by key.
  - `False` — never sort; deterministic but unspecified order.
  - `None` (the **new default**) — *auto*: sort only when cheap. Integer and
    dictionary keys are sorted (free / vectorized); float and multi-key results,
    whose only ordering is an O(G log G) Python sort over every distinct group,
    are left unsorted to avoid a cost that can rival the grouping itself on
    high-cardinality data.
- **Behavior changes** (the two APIs had different prior defaults, so they move
  in opposite directions):
  - `CTable.group_by()` previously returned results *always* sorted. Under the
    new `None` default, **float-key and multi-key group-bys are no longer
    key-sorted by default** — pass `sort=True` to restore sorted output. This is
    a deliberate divergence from pandas (which defaults to `sort=True`), suited
    to blosc2's large / on-disk datasets.
  - `blosc2.group_reduce()` previously defaulted to `sort=False` (unsorted).
    Under the new `None` default its **cheap kernels now sort by default** —
    most visibly float keys, which previously came out in hash order. Integer
    keys were already ascending; the generic Python fallback stays unsorted.
    Pass `sort=False` to opt out.

### Accelerated reductions from index summaries

- `min`/`max` on indexed `Column`s, and `argmin`/`argmax` inside `group_by`, are
  now **accelerated using the index's per-block min/max summaries**: when an
  index is available these reductions run from the precomputed summaries instead
  of decompressing the underlying data, which is dramatically faster on large
  columns.  A fast path also builds **min/max envelope plots** from any index.
- The **last `group_by` operation is memoized** and reused when the same
  grouping is requested again, avoiding recomputation in interactive / repeated
  workflows (e.g. `b2view`).

### b2view: group-by, sort, and richer plots

- **Interactive group-by (`G`)**: group a `CTable` by a column (integer, string,
  or now **float** keys) directly in the viewer, with a three-list / two-column
  menu; while grouped, `S`/`R` operate on the grouped result and the data
  panel's subtitle shows a `G(roup)` chip.  The last grouping is memoized for
  instant reuse.
- **Sort by column (`S`)**: sort a `CTable` by a **fully indexed** column via a
  dropdown (`R` toggles reverse) as a zero-copy `sort_by(view=True)` that streams
  from the index — the table is never materialised, `Esc` restores the original
  order, and a `SORTED` chip shows in the status bar.  Non-indexed columns can
  now be sorted too.  Sort and filter are mutually exclusive; a row window
  composes over a sort, and an `f`ilter is preserved across `S`ort / `G`roup.
- **Better plots of grouped/sorted views**: a grouped view plots **bars for a
  categorical key** and **lines for a numeric key**; numeric-key group plots
  render as **stem/impulse charts** rather than misleading connected lines.  Bar
  plots gain an `h`i-res counterpart mirroring the line/scatter plots, and `+`/`-`
  zoom about the view's left edge.
- **`--max`** maximizes the current panel, and `escape` is now the single,
  consistent way to back out of every modal.

### Other / bug fixes

- **C-Blosc2 upgraded to 3.1.5.**
- **Open-file cache correctness**: cached open handles are now validated against
  the file's fingerprint (`st_mtime_ns`, `st_size`) and cached index handles are
  released when a table closes, so a file changed underneath an open handle is no
  longer served stale.
- **NumPy 2.5 compatibility**: adjusted for deprecations in NumPy 2.5.
- Substantially **reduced test-suite runtime**, and emscripten builds no longer
  attempt to spawn subprocesses (unsupported there).

## Changes from 4.5.0 to 4.5.1

This follow-up release builds the `b2view` terminal viewer into a richer
data-exploration tool — a **scatter plot**, a **searchable column picker**, a
**one-shot demo download**, refreshed chrome, and several interaction fixes — and
upgrades the bundled **C-Blosc2 to 3.1.4**.  WASM/Pyodide is now a **fully
supported platform**, and `CTable.info` reports **per-column compressed sizes**.

### b2view: richer exploration

- **Scatter plots**: from a column plot, press `s` to scatter the current column
  (X) against another column (Y) chosen from a list, over the current (zoomed)
  row range; `h` then opens a high-resolution `matplotlib` scatter.
- **High-res for 1-D series is now an envelope plot** (matching the in-terminal
  view), and a new `r` key toggles between the min/max envelope and the raw
  values (strided-sampled when the range is wide).
- **Searchable column picker**: the `c` go-to-column key now opens a searchable,
  selectable list (type to filter, ↑/↓, Enter) for CTables, instead of a text
  field; N-D arrays still go by numeric index.
- **Show/hide columns**: `/` opens a searchable multi-select to pick which CTable
  columns are displayed.
- **Demo download**: `b2view --download` fetches a demo bundle
  (`chicago-taxi-flat.b2z` by default) into the current directory if it is not
  already there, then opens it.
- **Refreshed chrome**: a branded header, a left-docked filename label in the
  title, and clearer status chips.

### b2view: interaction fixes

- **Go-to-row/column pre-fill is now pre-selected**, so the first keystroke
  replaces the current index instead of appending to it (typing a column name no
  longer produced e.g. `0payment.fare`).
- **Escape keeps its layered exit while a panel is maximized**: with the data
  panel maximized, escape now unlocks a plot's locked row window (and clears
  filters) as documented, instead of being hijacked into restoring the panel —
  use `r` to restore (`ESCAPE_TO_MINIMIZE = False`).
- Test-suite robustness fixes (a timing flake and a Windows rendering glitch).

### Other

- **C-Blosc2 upgraded to 3.1.4.**
- **WASM/Pyodide is now a fully supported platform**, with more frequent CI runs.
- **`CTable.info` shows per-column compressed sizes** (`cbytes` and `cratio`),
  and `print_versions()` uses clearer `Python-Blosc2` / `C-Blosc2` labels.

## Changes from 4.4.5 to 4.5.0

This release teaches the `b2view` terminal viewer to **plot** — peak-preserving
envelope line plots of any series, with zoom, a row-window lock, and an optional
high-resolution matplotlib view — and gives `CTable` a **pandas-like display and
CSV** experience.  It also publishes **WASM/Pyodide wheels to PyPI** and adds
faster strided reads for `NDArray` and `Column`.

### b2view: plotting and data inspection

- **In-terminal plots**: press `p` on a numeric series (a CTable column or an
  array row) to draw a braille line plot.  Plots are **peak-preserving min/max
  envelopes by default**, so no spike or trough is hidden however large the
  series is; large local series stream their envelope *exactly* in bounded
  spans (only remote c2arrays fall back to a labeled strided sample).
- **Zoom and row-window lock**: zoom the plot into a row range and pan it; press
  `v` to **lock the data grid to the plotted range** so paging stays inside it
  (escape unlocks).  The plot and high-res views honor the locked window.
- **High-resolution view**: `h` opens a high-res `matplotlib` image of the
  plotted range (new optional `hires` extra: `matplotlib` + `textual-image`).
- **On-demand cell decode**: `enter` decodes a single skipped/expensive CTable
  cell, and SChunk nodes now preview as a paged hex dump.
- **Fixes and polish**: row paging re-aligns to the page grid after dim-mode
  single-row scrolls; the data panel now focuses correctly with
  `--path ... --panel data`; status chips are branded yellow.

### CTable display

- **`CTable.to_string()` now renders the whole table by default** (every row and
  every column), like `pandas`' `DataFrame.to_string()`.  New `max_rows` and
  `max_width` parameters truncate on demand.  *Behaviour change*: previously
  `to_string()` returned the truncated view; code that relied on that should
  pass `max_rows=`/`max_width=` (or use `str()`).
- **The `[N rows x M columns]` dimensions footer now follows pandas**: omitted by
  `to_string()` (pass `show_dimensions=True` to force it), and shown by
  `str`/`repr`/`print` only when the view is actually truncated.  Previously it
  was always appended.
- **`repr(ctable)` now shows the same truncated table as `str(ctable)`**
  (pandas/polars convention), instead of the one-line `CTable<…>` summary.  The
  compact summary remains available via `ctable.info`.
- **New display options** in `set_printoptions`: `display_width` controls the
  column-fitting width budget (`None` = auto-detect terminal, `-1` = show all
  columns, positive int = fixed budget), and `display_rows` now accepts `-1` to
  show all rows (`0` still shows none).
- **New `blosc2.printoptions(...)` context manager** temporarily sets the display
  options and restores them on exit, e.g.
  `with blosc2.printoptions(display_rows=-1, display_width=-1): print(t)`.

### CTable I/O

- **`CTable.to_csv()` now accepts no path**, returning the CSV as a string like
  `pandas`' `DataFrame.to_csv()`.  Passing a path still writes the file (and
  returns `None`); the returned string is byte-for-byte the same as the file.

### Performance

- **Faster strided reads**: `NDArray.__getitem__` gains a sparse-gather fast
  path for large strides, and `Column.__getitem__` short-circuits when the
  logical positions equal the physical ones.
- **Fix**: a negative `step` in `Column` getitem could return `[]`; it now
  returns the reversed selection.

### Indexing

- **Fix**: a sidecar-handle cache collision could return the wrong SUMMARY
  index for a compact-store column.
- **Cross-column index pruning** is now enabled for compact CTable queries, so
  more predicates prune blocks before any data is materialized.  The docs also
  note when summary indexes are *not* created automatically.

### Packaging

- **WASM/Pyodide wheels on PyPI**: the main wheel build now also produces
  `pyemscripten` wheels for CPython 3.13 (2025 ABI) and 3.14 (2026 ABI) and
  uploads them to PyPI, so `blosc2` is `micropip`-installable in Pyodide, and
  `b2view` prints a clear message instead of crashing when run under WASM.
  *Known limitation*: slicing an in-memory `SChunk` loaded from a frame fails on
  the Pyodide 0.29.x Emscripten toolchain (cp313); it works on Pyodide 314
  (cp314) and natively.  See issue #664.
- **cibuildwheel updated to 4.1.**

## Changes from 4.4.3 to 4.4.5

Note: 4.4.4 was skipped due to a failure during the release process.

This release promotes the `b2view` terminal viewer to a core feature —
installed by default, with new interactive row and column filtering — and
makes BatchArray block layouts (and hence compression ratios) reproducible
across CPUs.

### b2view, the terminal data viewer

- **Installed by default**: `textual` and `rich` are now regular
  dependencies, so the `b2view` CLI works out of the box (the `[tui]` extra
  is gone).  A getting-started walkthrough was added to the docs, and the
  README now lists the CLI tools.
- **Row filtering**: pressing `f` on a CTable node opens a modal that takes
  the same string expressions as `CTable.where()` (dotted nested names,
  `and`/`or`) and pages through the matching view.  Filters are remembered
  per node for the session, the data header shows the active filter plus
  the unfiltered total, and escape (or an empty expression) clears it.
- **Column filtering**: `/` narrows the visible columns by case-insensitive
  substring; column paging and the `c` goto-column modal then operate on
  that subset.  Combines freely with the row filter; escape clears one
  layer per press (rows first, then columns).
- **Mouse handling**: the terminal owns the mouse by default, so native
  text selection/copy works like in any CLI program; `--mouse` lets b2view
  capture it instead (click-to-focus, wheel scrolling by half a page,
  paging at the edges).
- **Navigation**: `?` opens a help screen listing all keys; `c` jumps to a
  column by index, exact name or unique name prefix; `s`/`e` jump to the
  first/last column window; row paging and jumps keep the cursor on its
  column; dim-mode index/viewport movements clamp at the boundaries instead
  of wrapping around.
- **Rendering**: column windows are fitted from measured rendered widths
  (and re-fitted on terminal resize and panel maximize/restore), and float
  columns use a uniform number of decimals so decimal points align down
  the column.
- **Test suite**: first automated tests for the TUI — Pilot-driven keyboard
  journeys against a deterministic generated store (marker `tui`), plus
  render unit tests.  Skipped on wasm, where Textual apps cannot start
  (no termios).

### BatchArray

- **Reproducible block layouts**: automatic variable-length block sizing
  now uses fixed byte budgets (1 MiB for clevel 1-3, 8 MiB for 4-6, 16 MiB
  for 7-8) instead of the CPU cache sizes, so the layout — and hence the
  compression ratio — no longer depends on the machine that created the
  array.

### Build and docs

- **Installing test dependencies**: the docs now use
  `pip install . --group test` (a PEP 735 dependency group); the stale
  `[test]` extra syntax was removed.
- **cibuildwheel updated to 4.0.**

## Changes from 4.4.2 to 4.4.3

This is a maintenance release focused on faster CTable cold-start, printing
and groupby performance, a lighter `import blosc2`, new raw-storage access
for columns, and support for the new J2K/HTJ2K codec plugins.

### CTable performance

- **Lazy column opening in views**: `select()` (and other view-producing
  operations) no longer open every projected column up front.  A column is
  only opened from storage when the view actually reads it, so selecting and
  then touching a subset of columns — or aggregating a single one — skips
  the cold-start cost of the rest.
- **Lazy index opening in queries**: query planning no longer opens every
  SUMMARY-indexed column on a wide persistent table; only indexes for
  columns actually referenced by the predicate are loaded.
- **Faster table printing**: `repr()`/`to_string()` now memoise per-column
  sparse gathers for the duration of a render and combine the head and tail
  rows into a single sparse read per column.  Each column is read from
  storage once instead of ~6 times (precision detection, width sizing and
  row rendering all hit the cache).
- **Groupby with integral float keys**: float key columns whose values are
  integral and fit a compact non-negative range (e.g. float32 id/second
  columns) now take the dense single-key fast path instead of the markedly
  slower generic float-hash path.  Fractional or non-finite keys fall back
  automatically.
- **No tempdir in read mode**: opening a `.b2z`/`.b2d` store in `'r'` mode
  no longer creates a temporary working directory, since nothing is ever
  written.

### Lighter imports and prefetcher rework

- **asyncio dependency dropped**: the on-disk chunk prefetcher used by the
  UDF and numexpr fallback engines now uses plain `concurrent.futures`
  instead of an asyncio event loop.  `import blosc2` no longer pulls in
  ~30 asyncio modules, saving ~3 MB of memory footprint at import time.
- **Prefetcher deadlock fixed**: an exception during evaluation could leave
  the generator finalizer blocked forever in `thread.join()` while the
  reader thread was stuck on a full prefetch queue.  A stop event now makes
  the producer bail out when its consumer goes away.

### New features

- **`Column.raw` accessor**: returns the underlying storage container of a
  column (`NDArray`, `ListArray`, `DictionaryColumn`, …) directly.  Unlike
  `Column.__getitem__`, which always materializes NumPy arrays, this is the
  column as a blosc2-native compressed object — usable as a lazy-expression
  operand without decompressing, and exposing storage details like `schunk`,
  `chunks` or `cparams`.  Note that this is a *physical* view: fixed-width
  containers are over-allocated to chunk capacity, so slice to `len(table)`
  to get just the live rows, and no validity-mask or null-sentinel
  processing is applied.  Raises `AttributeError` for computed columns,
  which have no backing storage.
- **J2K and HTJ2K codec IDs**: `blosc2.Codec.J2K` and `blosc2.Codec.HTJ2K`
  expose the IDs for the new JPEG 2000 codec plugins (installable with
  `pip install blosc2-j2k` and `pip install blosc2-htj2k`).

### Fixes

- **`--float-trunc-prec` and nested columns**: the precision-truncation
  filter of the `parquet_to_blosc2` CLI now propagates to float fields
  inside nested (struct/list) columns too.
- **Guard for unsupported computed-column expressions**: expressions that
  would serialize to an empty or non-round-trippable string are now rejected
  with an early, actionable `ValueError` at `add_computed_column()` time,
  instead of silently breaking on reload.

### Build

- **C-Blosc2 updated to 3.1.3.**

## Changes from 4.4.1 to 4.4.2

This is a feature and maintenance release that promotes DSL kernels to
first-class CTable computed columns, adds a new `CTable.__setitem__`
assignment idiom, optimises bulk NDArray writes, and fixes several
correctness issues.

### DSL kernels as first-class CTable columns

- **`add_computed_column()` accepts DSL kernels**: `@blosc2.dsl_kernel`-decorated
  functions can now back virtual computed columns directly, in addition to
  the existing string-expression form.  The column survives save/open
  round-trips via persisted `dsl_source`.
- **`add_generated_column()` accepts DSL kernels**: stored generated columns
  (written during `append`/`extend` and on `refresh_generated_column()`)
  now support DSL kernels as their transformer.
- **`CTable.where()` accepts UDF/DSL kernels**: filter predicates are no
  longer limited to expression strings — any DSL kernel can be passed directly.
- **`dtype` inference for DSL kernels**: when `dtype` is omitted,
  `lazyudf()` infers the output dtype via NumPy type promotion of the input
  column dtypes.  Pass `dtype` explicitly for type-changing kernels
  (comparisons, casts).
- **`kernel_from_source()` utility**: new `dsl_kernel.kernel_from_source()`
  reconstructs a `DSLKernel` from its stored source text, shared by the
  CTable DSL-column loaders and the persisted `LazyUDF` decoder.
- **Security note**: `.b2d` files from untrusted sources that contain DSL
  computed columns execute stored Python source on open.  A warning is now
  included in the documentation.

### New `CTable.__setitem__` column-assignment API

- **`t["col"] = arr`**: new shorthand equivalent to `t["col"][:] = arr`.
  Accepts any array-like including `blosc2.NDArray`.  Raises `KeyError` for
  unknown columns and `ValueError` for views or read-only tables.

### Chunked NDArray writes in `extend()` and `Column.__setitem__`

- **`extend({"col": ndarray})` decompresses chunk-by-chunk**: when a
  `blosc2.NDArray` is passed as a column value to `extend()`, it is now
  written in chunks instead of being fully decompressed upfront.  Pass
  `validate=False` to avoid a transient full decompression during constraint
  checking.
- **`col[:] = blosc2_ndarray` fast path**: a new no-holes fast path in
  `Column.__setitem__` skips the O(n) validity-mask gather and writes the
  NDArray one chunk at a time using contiguous slice writes.  Works for both
  scalar and fixed-shape ndarray columns.  Falls back to a chunked fancy-index
  path when deleted rows are present.

### `BLOSC_ME_JIT` environment variable override

- **Full CLI override**: `BLOSC_ME_JIT` now takes unconditional priority over
  both the `jit=` and `jit_backend=` keyword arguments, making it easy to
  switch JIT backends from the command line without modifying code.

### Correctness fixes

- **View corruption in `Column.__setitem__`**: a `None == None` guard
  evaluation on view-backed columns could fire the NDArray fast path,
  bypassing physical-position remapping and silently corrupting rows.  Fixed
  by explicitly checking `base is None` before activating the fast path.
- **`CTable.__setitem__` view guard**: the new `t["col"] = arr` API now
  raises `ValueError` on views, matching the contract of all other mutating
  CTable methods.
- **Fast path enabled for disk-opened tables**: the fast path previously
  remained dormant for tables opened from disk because `_last_pos` starts as
  `None`.  The guard now calls `_resolve_last_pos()` to lazily initialise it.
- **DSL column `jit_backend` preserved in `_empty_copy`**: the `jit_backend`
  setting was silently dropped during internal table copies; it is now
  retained.
- **`lazyexpr` Column unwrapping**: `convert_inputs()` now automatically
  unwraps `CTable.Column` objects to their backing NDArray so that shape and
  identity checks work correctly.

### Documentation and examples

- **Parquet-to-blosc2 walkthrough**: new step-by-step tutorial added to the
  getting-started section. Thanks to @SyedIshmumAhnaf.
- **CTable performance tips**: new section in the overview covering when to
  prefer computed vs. generated columns, chunk sizing, and query optimisation.
- **Simplified docstring examples**: examples throughout `ndarray.py` and
  `ctable.py` now use `blosc2.array()`, `blosc2.arange()`, and
  `blosc2.linspace()` directly instead of two-step numpy-then-`asarray`
  patterns.
- **`udf-computed-col.py` example**: new end-to-end example demonstrating DSL
  kernel computed and generated columns.

## Changes from 4.3.3 to 4.4.1

This is a feature release focused on a new interactive data viewer, automatic
SUMMARY indexes for fast WHERE queries, chunk-aligned Arrow/Parquet imports,
expanded `where()` acceleration via miniexpr, and a range of CTable ergonomics
and performance improvements.  Python 3.10 support has been dropped; Python
3.11 is now the minimum.

### b2view: interactive Text User Interface data viewer

- **New `b2view` command**: a terminal-based interactive viewer for all
  blosc2 containers — `NDArray`, `CTable`, `SChunk`, `BatchArray`, and more.
  Launch it with `b2view <file>` or as `blosc2.b2view()` from Python.
- **Full 1-D and 2-D browsing**: arrays with more than two dimensions are
  sliceable along any axis; 1-D arrays are shown as a single-column table.
- **CTable navigation**: scroll through rows with keyboard shortcuts; `t`/`b`
  jump to the top/bottom; `--panel` jumps straight to a named panel on launch.
- **`CTable.vlmeta` panel**: variable-length metadata is exposed in a dedicated
  panel.
- **New dim mode**: navigate along all dimensions freely for N-D arrays.

### SUMMARY indexes for fast WHERE queries

- **Automatic SUMMARY index creation**: when a `CTable` is closed after a
  write session, SUMMARY indexes (per-block min/max) are built by default for
  all eligible scalar columns with no extra configuration needed.
- **Incremental build during writes**: indexes are accumulated block-by-block
  during `extend()` and Arrow import, so closing the table costs almost nothing
  beyond the write already done.
- **Block-skip prefilter**: the miniexpr prefilter uses SUMMARY bitmaps to skip
  entire blocks whose min/max range cannot satisfy the WHERE predicate, reducing
  decompression work for selective queries.
- **Conjunction support**: per-column SUMMARY block masks are combined with
  bitwise AND so multi-column conjunctions prune blocks efficiently.
- **Cost gate**: a cost model guards index use; the SUMMARY path is skipped when
  block skipping is unlikely to help (e.g. very low selectivity).
- **`--no-summary-index`**: new CLI flag for `parquet-to-blosc2` to disable
  automatic index creation on import.

### CTable column grid alignment

- **Shared chunk/block grid for scalar columns**: fixed-size columns are now
  written on a shared chunk/block grid derived from the numeric column widths,
  so all columns have identical chunk boundaries.  This makes multi-column
  SUMMARY scans and chunk-parallel reads significantly faster.
- **Chunk-aligned Arrow import**: incoming Arrow/Parquet batches are buffered
  and flushed in exact chunk-sized blocks, so each chunk is compressed exactly
  once instead of being split across batch boundaries.
- **Vectorized dictionary-column import**: dictionary codes are now written in
  bulk at full chunk capacity rather than element by element.
- **Small fixed strings on the grid**: fixed-length string columns narrow enough
  to share the numeric grid are admitted to it, reducing the number of distinct
  chunk sizes.
- **`--reduce-mem`**: new CLI option for `parquet-to-blosc2` to cap the Arrow
  read-batch size on nested `list<struct>` imports, keeping peak RSS low at a
  modest speed cost.

### CTable.copy() enhancements

- **C-level bulk copy for `ListArray` and `BatchArray`**: a new `chunk_copy()`
  method transfers pre-compressed chunks directly at the C level, bypassing
  Python-level serialization and recompression.  `CTable.copy()` uses this path
  automatically.
- **`chunks=` / `blocks=` overrides in `CTable.copy()`**: callers can now
  specify target chunk and block sizes for the output copy.
- **`cparams` and `blocks` overrides**: `CTable.copy()` accepts `cparams` and
  `blocks` to recompress the copy with different settings.
- **`--chunks` / `--blocks`** added to the `parquet-to-blosc2` CLI.

### Take/gather APIs

- Added `NDArray.take()` following Array API `take` shape semantics, including
  `axis=None` flattening and N-dimensional integer indices.  One-dimensional
  gathers use a new sparse C-level path (`b2nd_get_sparse_cbuffer`) internally.
- Extended top-level `blosc2.take()` to dispatch to `NDArray.take()`,
  `CTable.take()`, and `Column.take()` while preserving the input container
  type.
- Added `CTable.take()` and `Column.take()` for logical row/value gathers that
  preserve order and duplicate indices, unlike mask-based views.
- For `ndim > 1` axis-based take, orthogonal selection is used internally for
  better performance.

### where() and miniexpr acceleration

- **`where(cond, x)` via miniexpr**: the single-argument `where` (fill-with-zero
  variant) is now handled directly by the miniexpr engine when the condition is
  a boolean array, avoiding a numexpr round-trip.
- **`where(cond, x, y)` via miniexpr**: the two-argument flavor is likewise
  dispatched to miniexpr for element-wise conditional selection.
- **Sparse boolean mask fast path**: when a boolean indexing result is very
  sparse (high selectivity), auto-detection switches to a fast gather path
  instead of a full-array scan.
- **Early boolean key check**: `NDArray.__getitem__` with a boolean array key
  now detects it before the general `process_key` / `nonzero` path, avoiding
  wasted work.
- **Compressed transient masks**: temporary boolean masks created during
  queries are now stored as LZ4-compressed blosc2 arrays, reducing memory
  pressure without measurable speed regression.
- **`BLOSC_ME_JIT` / `BLOSC_ME_JIT_TRACE`**: new environment variables to
  control and trace the miniexpr JIT backend at runtime.

### CTable views and lazy sorting

- **`sort_by()` on a view is now lazy**: calling `sort_by()` on a filtered view
  returns a position-reordered view without materializing data; the sort
  positions are cached and used directly on column access.
- **Lazy column materialization in filtered views**: `select()` on a view no
  longer materializes unneeded columns eagerly; columns are resolved only when
  accessed.

### NestedColumn and .info improvements

- **`NestedColumn` public class**: the previously internal
  `_NestedColumnNamespace` has been renamed and promoted to `NestedColumn`,
  providing aggregate metadata (`col_names`, `nrows`, `nbytes`, `cbytes`,
  `cratio`) and a structured `.info` report over a group of dotted columns.
- **Uniform `.info` across containers**: `Column.info`, `CTable.info`,
  `NestedColumn.info`, and related classes now follow a consistent field order
  (identity → shape/grid → sizes → content → compression params).

### Context manager support for blosc2.open()

- All objects returned by `blosc2.open()` — `NDArray`, `SChunk`, `CTable`,
  `BatchArray`, `ListArray`, and stores — now support the `with` statement.
  The `__exit__` method flushes and closes the underlying storage.

### Performance improvements and fixes

- **CTable.nrows stored persistently**: row counts are written to metadata on
  close and read back on open, avoiding a full column scan at startup.
- **Index sidecar loading from .b2z**: SUMMARY/BUCKET sidecars inside `.b2z`
  archives are now read in-place rather than extracted to a temporary directory,
  cutting open latency for indexed tables.
- **Compressed query cache**: the hot query-result cache is now stored
  LZ4-compressed, reducing its memory footprint with negligible overhead.
- **Query cache consistency fixes**: on-disk query cache side effects and a
  miniexpr chunk-cache race condition on Apple Silicon have been resolved.
- **macOS L2 floor for chunk sizing**: on macOS the full L2 cache is used as a
  floor for automatic chunk sizing, giving better compression/speed trade-offs.
- **Better Apple Silicon L3 handling**: missing L3 cache on Apple Silicon is
  handled more gracefully in the cache-size heuristic.
- **Table capacity management**: large CTables grow more conservatively, and
  capacity is trimmed on close and after Arrow import to reclaim over-allocated
  space.
- **Faster iteration with `iterchunks_info()`**: several hot loops switched to
  `iterchunks_info()` for lower overhead per chunk.
- **Cost-model index refinement threshold**: the previously hardcoded threshold
  for switching between index and scan has been replaced with a data-driven cost
  model.
- **Index prefetch reuse**: data already prefetched during an index lookup is
  reused in the refinement phase, avoiding redundant I/O.
- **Simplify index sidecar filenames** in _indexes/{col}/ directories.
- **`DictStore` embed disabled by default**: embedding a store inside a dict
  store is now opt-in (it was error-prone as the default).
- **Fixed wasm32 issue**: a 32-bit platform arithmetic fix for reduce operations.
- **Chunks never exceed array dimension**: `compute_chunks_blocks` now
  guarantees chunk dimensions are capped at the array shape dimension.
- **`max_rows` robust to older PyArrow**: truncation logic no longer depends on
  PyArrow APIs that are absent in older releases.
- **`cratio` display**: compression ratio is now shown with an explicit `x`
  suffix (e.g. `2.47x`) throughout `.info` output.
- Updated bundled C-Blosc2 to the latest release.

### Dropped Python 3.10 support

- Python 3.11 is now the minimum supported version.

## Changes from 4.3.1 to 4.3.3

note: 4.3.2 was an internal pre-release that was not published to PyPI.

This is a maintenance release focused on CTable display ergonomics, indexed-query
correctness, and query-planner performance.

### CTable display and print options

- **Pandas-like CTable display by default**: `str(table)` / `print(table)` now use
  a compact, pandas/DuckDB-style table representation, including a displayed
  logical row index, numeric alignment, compact spacing, and a trailing footer
  such as `[726017 rows x 5 columns]`.
- **Configurable display options**: added `blosc2.set_printoptions()` and
  `blosc2.get_printoptions()` for CTable rendering.  The supported options are
  `display_index`, `display_rows`, `display_precision`, and `fancy`.
- **`CTable.to_string()`**: added a one-off formatting API for producing CTable
  string representations without changing global print options.
- **Compact truncation for large tables**: when a table exceeds the configured
  `display_rows` threshold, only the first five and last five rows are shown,
  with an ellipsis row in between.
- **Float display refinements**: compact mode uses pandas-like fixed precision
  for floating-point columns, and integer-valued float columns are displayed
  with a single decimal place.
- **Fancy display preserved**: `set_printoptions(fancy=True)` restores the more
  decorated display with dtype rows, separator rules, and hidden row/column
  counts.

### Indexed queries and sorting

- **Cross-column exact index refinement**: multi-column conjunctions can now use
  exact positions from a selective indexed column (`FULL`, `PARTIAL`, or `OPSI`)
  as a compact pre-filter, then refine the remaining predicates on those
  positions instead of scanning the full table.
- **NaN-safe index boundary navigation**: fixed sorted-boundary navigation for
  floating-point indexes containing `NaN` values, so indexed results match scan
  results for bucket/full index lookups.
- **Better index-planner heuristics**: the planner now avoids low-value indexed
  paths when segment pruning is unlikely to help, and avoids expensive scalar
  specialization for non-scalar arrays.
- **Faster filtered sorting**: small filtered views can be materialized and
  sorted directly, avoiding an extra gather of sort keys.

### Performance and fixes

- Avoid full materialization of `valid_rows` in several CTable code paths.
- Keep row counts lazy for views and avoid unnecessary `nrows` calls in the
  query planner.
- Reduced overhead in root-column iteration and small query-planner operations.
- Fixed dictionary-column capacity handling during Arrow import and a regression
  affecting dictionary columns.
- Marked additional long-running tests as `heavy` to reduce default test-suite
  runtime.

### Documentation

- Updated the containers tutorial with dedicated `ListArray` and `CTable`
  sections, including CTable's columnar storage model and support for columns
  backed by `NDArray`, `BatchArray`, `ObjectArray`, `ListArray`, and related
  containers.

## Changes from 4.3.0 to 4.3.1

This is a maintenance release focused on CTable nested-column ergonomics,
grouped reductions, and API/documentation polish.

### CTable nested columns and grouped reductions

- **Nested column names in `group_by()` results**: grouped output columns can now
  preserve dotted/nested names such as `trip.sec` instead of requiring valid
  Python identifiers.
- **Column-object selectors**: `CTable.group_by()` and `CTable.sort_by()` now
  accept `Column` objects as well as string names, enabling idioms such as
  `t.group_by(t.trip.sec)` and `t.sort_by(t.trip.sec)`.
- **Grouped arg reductions**: `CTableGroupBy` now supports `argmin()` and
  `argmax()`, plus `agg({"col": "argmin"})` / `agg({"col": "argmax"})`.
  Results are logical row positions in the grouped table or view; groups with no
  non-null values return `-1`.

### NDArray constructor ergonomics

- **`blosc2.array()`**: added a NumPy-like constructor for NDArrays.  It mirrors
  `blosc2.asarray()` but defaults to `copy=True`, so passing an existing
  `NDArray` creates a copy unless `copy=False` or `copy=None` is requested.

### Documentation

- Expanded the CTable reference with `RowTransformer`, `Column.row_transformer`,
  and `CTableGroupBy.argmin` / `argmax` documentation.
- Added `blosc2.ndarray()`, `blosc2.dictionary()`, and related public schema
  factory functions to the Schema Specs reference.
- Moved `blosc2.group_reduce()` into the Reduction Functions reference and
  updated its example to use Blosc2 NDArrays.

## Changes from 4.2.0 to 4.3.0

### CTable: N-dimensional (ndarray) columns

- **Multidimensional columns**: CTable columns can now hold NDArray-backed cells, allowing
  each row of a column to contain a full n-dimensional compressed array.  This enables
  use cases such as embedding vectors, image patches, time-series windows, or any other
  multidimensional per-row payload.
- **CSV and DataFrame import/export**: Multidimensional column data can be imported and
  exported via CSV and pandas DataFrames, with automatic detection of array-valued cells.
- **Nullable ndarray columns**: Multidimensional columns fully support the nullable
  semantics (`null_count`, sentinel handling, `null_policy`) already available for scalar
  columns.
- **`from_pandas()` improvements**: `CTable.from_pandas()` now creates the correct
  specialized backing storage for `DictionarySpec`, `ListSpec`, `VLStringSpec`,
  `VLBytesSpec`, and other variable-length scalar specifications.
- **Improved schema coverage**: New CTable timestamp schema type and extended
  `Column.info` output with `shape`, `chunks`, and `blocks` descriptors.
- **Arg reductions**: Added `argmin()` and `argmax()` for scalar and ndarray
  CTable columns, plus row-transformer support for generated columns such as
  per-row peak-hour or dominant-embedding-dimension features.

### CTable: Group-by and filtered aggregation

- **`CTable.group_by()`**: The primary group-by interface.  Call
  ``t.group_by("city", sort=True).agg({"qty": "mean"})`` to produce a new
  :class:`CTable` with aggregated results.  Single-key and multi-key groupings are
  supported, along with convenience methods such as ``.size()``, ``.count()``,
  ``.sum()``, ``.mean()``, ``.min()`` and ``.max()``:

  .. code-block:: python

      by_city = t.group_by("city", sort=True)
      by_city.size()  # COUNT(*)
      by_city.sum("sales")  # SUM(sales) per city
      by_city.agg({"sales": ["sum", "mean"]})  # SUM(sales), AVG(sales) per city

- **Performance accelerators**: Dedicated Cython fast paths deliver significant speedups:
  ~25× for float32/64 group-by keys, ~8× for integer and dictionary-code keys, and a
  general-purpose hash table for arbitrary float keys.
- **Filtered aggregate pushdown**: The `where=` parameter is now accepted in aggregation
  methods, pushing the filter into the compute engine so that only matching rows are
  read and reduced.
- **Persistent grouped output**: Group-by results can be saved directly to persistent
  storage via the `urlpath=` parameter.
- **`blosc2.group_reduce()`**: New public function that performs group-by reduction over
  NDArray instances and CTable columns, with Cython-accelerated backends for common
  key/reduction combinations.

### CTable: Dictionary / categorical columns

- **`DictionarySpec` column type**: Introduced a new dictionary-encoded (categorical)
  column type that stores string or integer codes mapped to a shared dictionary, providing
  compact storage and accelerated equality and membership queries.
- **Dictionary types in `where` clauses**: Dictionary columns can be queried with the same
  `where=` expression syntax as other column types, including nested dotted-name access.
- **Improved display**: `CTable` printing now adapts to the terminal width, and dictionary
  values are shown in their decoded form.  `Column.info` has been extended with type
  details, shape, chunks, and blocks.

### CTable: Nested columns and field-name escaping

- **Dotted nested column access**: Columns whose names contain literal `.`
  (e.g., `"root.nested"`) are now fully addressable via the dotted accessor syntax in
  `where` expressions, `__getitem__`, and the public API.
- **Hierarchical `_cols` storage paths**: The internal column storage layout now preserves
  a hierarchical structure that mirrors the logical nesting, improving introspection
  and interop.
- **Nested-field pipeline**: A new flattened-storage pipeline with logical mapping
  preserves nested schema structure (field names, types, and hierarchy) through
  Arrow and Parquet import/export.  For unnamed top-level `list<struct<...>>` Parquet
  files, the logical schema round-trips faithfully, though the original physical row
  grouping is intentionally not preserved.
- **Field-name escaping**: Special characters (`.` and `/`) in column names are
  automatically escaped during schema construction and metadata round-trips.

### Parquet import/export improvements

- **Arrow serializer by default**: `CTable.from_parquet()` now defaults to the Arrow
  serializer, providing better schema fidelity and nested-type support.
- **Progress reporting**: A `--progress` flag and an ETA estimator have been added to
  the `parquet-to-blosc2` CLI for long-running imports.
- **`--max-rows` parameter**: `CTable.from_parquet()` and the CLI now accept `max_rows`
  to limit the number of imported rows.
- **`--timestamp-unit`**: New CLI option to control timestamp unit conversion on import.
- **`--float-trunc-prec`**: New CLI option to truncate floating-point precision on import.
- **Separated nested columns enabled by default**: The `separate_nested_cols` flag is now
  `True` by default for both the Python API and the CLI, ensuring nested Arrow structs
  are always expanded into flat columns.
- **`list_serializer` parameter**: New option to control how list-type columns are
  serialized, with sensible defaults for different list layouts.
- **Validation optimizations**: Arrow datetime values are validated only during import,
  reducing runtime overhead on subsequent operations.

### TreeStore: Inline CTable support

- **CTables inside TreeStore**: `CTable` objects can now be stored inline as items
  inside a `TreeStore`, enabling hierarchical storage that mixes arrays and tables in a
  single persistent container.
- **Cache hardening**: TreeStore cache assignments now use defensive copies and cache
  effective object roots to avoid aliasing and stale-cache errors.
- **Examples and tutorials**: New tutorials and docstring examples demonstrate how to
  store, retrieve, and query CTables within a TreeStore.

### Performance and usability enhancements

- **Faster open and import**: `blosc2.open()` and store constructors now assume valid
  file extensions and defer column metainfo loading, making `CTable.open()` and
  package import noticeably faster.
- **`CTable.nrows` is now lazy**: The row count is computed on demand rather than eagerly,
  speeding up open and schema-inspection workflows.
- **Accelerated scalar and small-slice access**: The batch/list path for reading scalar
  values or small column slices has been overhauled, eliminating internal placeholder
  materialization and yielding lower latency.
- **Late-import optimizations**: Heavy optional dependencies are imported lazily at the
  blosc2 package level, reducing the baseline `import blosc2` overhead.
- **`iter_arrow_batches()` optimization**: Avoids full Python object materialization of
  batches during iteration, reducing memory pressure.
- **`NDArray`-to-list conversion**: Small optimization when converting NDArray objects
  to Python lists.
- **`_last_pos` invalidation skipped**: Mid-table deletes no longer eagerly invalidate
  cached positional state, improving delete latency.

### Documentation, examples and benchmarks

- **API reference expanded**: `blosc2.group_reduce()` has been added to the Sphinx
  reference, along with updated CTable, Column, and TreeStore pages.
- **New tutorials and examples**: Added sections on CTable–TreeStore integration,
  nested fields, dictionary columns, aggregates, grouping and querying with `where=`.
- **New benchmarks**: Graph benchmarks for CTable insert time, column count, memory usage,
  and `where=` queries, plus dedicated group-by, nested-filter, and Parquet round-trip
  benchmarks.

### Fixes and compatibility

- **Null and NaN handling**: NumPy scalar null sentinels are now normalized to plain Python
  scalars, and floating-point NaN sentinels are treated consistently with Python
  `float('nan')`.
- **Empty aggregate results**: Filtered aggregations that produce no rows now handle the
  empty result gracefully.
- **Generated column safety**: Accessing a stalled (unfillable) generated column now raises
  a clear exception instead of producing undefined results.
- **Miniexpr bundling**: Miniexpr’s bundled `libtcc` and related runtime files are now
  kept inside the `blosc2` package, avoiding conflicts with other TCC installations.
- **Test improvements**: Torch-dependent tests are marked as `heavy`, PyArrow-optional
  tests are skipped when the library is absent, and parametrization matrices have been
  trimmed to reduce CI time.
- **Missing Cython validation**: Added validation guards for several Cython extension
  functions that previously lacked explicit error checking.
- **C-Blosc2 update**: Bundled C-Blosc2 has been updated to the latest version (3.0.3).
- **``blosc2.open()`` default mode changed from 'a' to 'r'**: Removed the FutureWarning that
  was added to prepare for this transition.

## Changes from 4.1.2 to 4.2.0

### CTable: columnar compressed tables

- Introduced `blosc2.CTable`, a new columnar table container for compressed, typed columns.  CTables support dataclass- and schema-based construction, row iteration, column access, table views, `head()` / `tail()` / `sample()`, sorting, selection and compact `where` expressions.
- Added persistent CTables backed by `TreeStore`, with support for `blosc2.open()`, `CTable.open()`, `CTable.load()`, `CTable.save()`, `CTable.to_b2d()` and `CTable.to_b2z()`.  CTable views can be saved too, and `.b2z`/`.b2d` path handling has been tightened.
- Added mutation operations for CTables, including `append()`, `extend()`, `delete()`, `compact()`, `add_column()`, `drop_column()`, `rename_column()` and related schema validation.
- Added computed columns, including virtual computed columns backed by lazy expressions, materialized computed columns and automatic filling of materialized computed columns during inserts.
- Added CTable indexing support, including persistent indexes, direct expression indexes, ordered index reuse, boolean `LazyExpr`/`NDArray` masks in `CTable.__getitem__`, `iter_sorted()` and indexing support for `.b2z` tables.
- Added nullable schema support and null policies for CTable scalar columns, preserving nullable scalar Parquet round-trips.
- Added variable-length CTable column support via `ListArray` / `ObjectArray`, including `vlstring` and `vlbytes` schema specs, fixed-length string/bytes import support and list/struct Arrow/Parquet round-trips.
- Added Arrow, Parquet and CSV interoperability for CTables, including batch-wise Arrow/Parquet import/export, Arrow schema metadata preservation, `CTable.from_arrow_batches()` improvements and a new `parquet-to-blosc2` CLI utility.
- Added CTable documentation, tutorials, examples and benchmarks covering schema definition, persistence, querying, indexing, mutations, nullable columns, computed columns and variable-length columns.

### Indexing and ordering

- Added a new indexing subsystem for NDArrays and CTables, including full, partial/bucket, light/medium and OPSI-style index kinds, out-of-core index builders and sidecar storage.
- Added `blosc2.Index` as the unified public index handle, plus APIs such as `create_index()`, `compact_index()`, `iter_sorted()`, `will_use_index()` and related query explanation support.
- Added materialized expression indexes for NDArrays and direct expression indexes for CTables.
- Added persistent query-result caching for indexed lookups, with FIFO pruning and cache accounting.
- Added `blosc2.argsort()` and refactored indexing APIs around explicit index enums and sorting helpers.
- Improved indexed query performance with Cython accelerators, threaded chunk batching, zero-copy/cached mmap reads, chunk-aware and reduced-order layouts and faster scattered row gathering.
- Reduced memory usage during index creation and lookup by avoiding full sidecar materialization, replacing memmap staging with Blosc2 scratch arrays and adding `tmpdir` support for full out-of-core indexes.

### Persistence, stores and serialization

- Added structured Blosc2 serialization based on b2object carriers, including persisted `C2Array`, `LazyExpr` and DSL `LazyUDF` objects.
- Added `blosc2.Ref` for serializing external references, plus examples for b2object bundles and persisted expressions/UDFs.
- Added `blosc2.load()` as a convenience loader.
- Added `vlmeta` support to `LazyArray` objects.
- Improved store handling by preserving lazy b2object carriers in `DictStore`, allowing reopened proxies to refill caches after read-only opens, relaxing `DictStore`/`TreeStore` suffix requirements and adding `DictStore.to_b2d()`.
- Accelerated `blosc2.open()` by trying standard opens first and warning on implicit append mode.

### Arrays, computation and containers

- Added `ObjectArray` for fully general object data and renamed the earlier `VLArray` work accordingly; added `ListArray` docstrings and Arrow integration improvements.
- Added schema helpers including numeric specs, `blosc2.struct()` and `blosc2.object()` for nested/fully general column declarations.
- Improved `fromiter()` with direct chunked construction and substantially lower peak memory use.
- Improved `asarray()` behavior for NDArray inputs when copy-inducing keyword arguments are supplied.
- Added `SChunk.reorder_offsets()`.
- Improved `BatchArray` defaults and documentation; the default compression level is now tuned for faster lookup/scan behavior.
- Continued matmul/linalg optimization work and shared-thread-pool integration.

### CLI, docs and examples

- Added the `parquet-to-blosc2` command with options such as `--max-rows`, `--parquet-batch-size`, `--blosc2-items-per-block` and `--use-dict`.
- Added new CTable, ObjectArray, BatchArray, containers, indexing and serialization tutorials and examples.
- Reorganized and expanded the API reference for CTable, Column, schema specs, Index, save/load helpers and miscellaneous APIs.
- Updated benchmark suites for CTables, indexing, Parquet import/export, BatchArray and NDArray construction/indexing.

### Fixes and compatibility

- Updated bundled C-Blosc2 to v3.0.2 and require C-Blosc2 >= 3.0.0 when building against a system library.
- Updated bundled C-Blosc2 and miniexpr sources multiple times.
- Restored compatibility with NumPy < 2.
- Fixed Windows and mmap/file-locking issues in index creation, rebuilds and temporary file cleanup.
- Fixed full-index query failures for large CTable columns and full out-of-core merge failures on systems with small `/tmp`.
- Fixed stale sidecar/cache reuse and targeted cache invalidation when persistent sidecars are replaced.
- Fixed `.b2z` double-open corruption caused by GC-triggered repacking and made temporary `.b2z` unpacking default to the source file directory.
- Fixed a regression when reopening persisted proxies in read-only mode.
- Fixed GC-induced thread hangs on macOS with Python 3.14 and hardened async chunk reading/cache cleanup paths.
- Fixed lazy-chunk source-size handling in decode/getitem callers.
- Fixed nullable validation, dictionary extend validation, CTable close propagation, print alignment and NumPy mask support.
- Fixed `arange()` regressions and several pre-existing `set_slice` error-handling issues.
- Clamped indexing/thread defaults for wasm32.

## Changes from 4.1.1 to 4.1.2

- A new fast path for src/blosc2/linalg.py that uses the matmul prefilter machinery in src/blosc2/blosc2_ext.pyx.
  - The fast path is only used for supported cases:
      - blosc2.NDArray inputs
      - 2-D only
      - floating-point only
      - matching dtypes
      - aligned chunk/block layouts that satisfy the current kernel assumptions
  - All other valid cases fall back to the existing chunk-by-chunk implementation in src/blosc2/linalg.py.
  - Some benchmarks for the supported cases show significant speedups over the chunked implementation:
    - aligned 400x400 float32: about 3.7x faster over chunked
    - aligned 400x400 float64: about 3.0x
    - aligned 800x800 float32: about 1.5x
    - misaligned case: auto correctly stays on chunked

## Changes from 4.1.0 to 4.1.1

- Update ``miniexpr`` to fix bug on ubuntu with ARM64.

## Changes from 4.0.0 to 4.1.0

- Add DSL kernel functionality for faster, compiled, user-defined functions which broadly respect python syntax and implement the `LazyArray` interface. See the introductory tutorial at: https://blosc.org/python-blosc2/getting_started/tutorials/03.lazyarray-udf-kernels.html
- Add read-only mmap support for store containers:
  `DictStore`, `TreeStore`, and `EmbedStore` now accept `mmap_mode="r"`
  when opened with `mode="r"` (including via `blosc2.open` for `.b2d`,
  `.b2z`, and `.b2e`).
- New .meta entry for store containers, allowing better store recognition at `blosc2.open()` time.  Fixes #546.
- Add `cumulative_sum` and `cumulative_prod` functions for Array API compliance.
- Add Unicode string arrays, support comparison operations with them, and optimised compression path.
- Add ``endswith`` and ``startswith`` and extend ``contains`` to support strings and offer `miniexpr` multithreaded computation when possible.
- Use DSL kernels to accelerate `arange`/`linspace` constructors by 6-10x.
- Improve documentation for `filters` and `filters_meta`.
- Fix edge case issues with `resize` and `constructors` so that `chunks` may be set independently of shape, and arrays may be extended from empty consistently.
- Continued work on `miniexpr` integration, interface, and support.
- Ruff fixes and implementation of PEP recommendations.

## Changes from 4.0.0-b1 to 4.0.0

- On Windows, miniexpr is temporarily disabled for integral outputs and mixed-dtype expressions.
  Set `BLOSC2_ENABLE_MINIEXPR_WINDOWS=1` to override this for testing.
- Handle thread workers for computation to ensure never exceeds NUMEXPR_MAX_THREADS. Thanks @skmendez!

## Changes from 3.12.2 to 4.0.0-b1

- PEP 427 compatibility changes to ensure C-blosc2 files and binaries are stored under blosc2/ subdirectories in shipped Python wheels
- Introduce miniexpr for hyper-fast multithreaded element-wise computations and reductions (on macOS and Linux). This justifies the major version number bump.
- Indexing with None for LazyExpr now matches Numpy behaviour (i.e. newaxis)
- Improvements to open and generally handle Treestore objects and b2z, .b2d, .b2e files. Thanks @bossbeagle1509!
- Minor changes to support new blosc2-openzl plugin

## Changes from 3.12.1 to 3.12.2

* Hotfix to change WASM wheel hosting to separate repo

## Changes from 3.12.0 to 3.12.1

* Hotfix for security - disallow ``import`` in (saved) ``LazyUDF`` objects
* Automate WASM wheel upload via YAML file

## Changes from 3.11.1 to 3.12.0

* `LazyUDF` objects can now be saved to disk
* Calls to ``__matmul__`` NumPy ufunc now passed to ``blosc2.matmul``
* Streamlined ``LazyUDF.compute`` is now much more robust and functional
* The ``get_chunk`` method for ``LazyExpr`` is more efficient and enabled for general ``LazyArray`` objects
* ``LazyExpr`` calculation can now be done even with expressions with pure scalar operands, e.g ``10 * 3 +1.``.

## Changes from 3.11.0 to 3.11.1

* Change the `NDArray.size` to return the number of elements in array,
  instead of the size of the array in bytes. This follows the array
  API, so it is considered a fix, and takes precedence over a possible
  backward incompatibility.
* Tweak automatic chunk sizing of results for certain (e.g. linalg) operations
  to enhance performance
* Bug fixes for lazy expressions to allow a wider range of functionality
* Small bug fix for slice indexing with step larger than chunksize
* Various cosmetic fixes and streamlining (thanks to the indefatigable @DimitriPapadopoulos)

## Changes from 3.10.2 to 3.11.0

* Small optimisation for chunking in lazy expressions
* Extend Blosc2 computation machinery to accept general array inputs (PR #510)
* Refactoring and streamlining of get/setitem for non-unit steps (PR #513)
* Remote array testing now performed with `cat2cloud` (PR #511)
* Added argmax/argmin functions (PR #514)
* Change `squeeze` to return view (rather than modify array in-place) (PR #518)
* Modify `setitem` to load general array inputs into NDArrays (PR #517)

## Changes from 3.10.1 to 3.10.2

* LazyExpr.compute() now honors the `out` parameter for regular expressions (and not only for reductions).  See PR #506.

## Changes from 3.10.0 to 3.10.1

* Bumped to numexpr 2.14.1 to improve overflow behaviour for complex arguments for ``tanh`` and ``tanh``
* Bug fixes for lazy expression calculation
* Optimised computation for non-blosc2 chunked array arguments (e.g. Zarr, HDF5)
* Various cleanups and most importantly shipping of python 3.14 wheels due to @DimitriPapadopoulos!
* Now able to use blosc2 in AWS Lambda

## Changes from 3.9.1 to 3.10.0

* Improved documentation on thread management (thanks to [@orena1](@orena1) in PR #495)
* Enabled direct ingestion of Zarr arrays, and added examples for xarray ingestion
* Extended string-based lazy expression computation using a shape parser and modified lazy expression machinery so that expressions like "matmul(a, b) + c" can now be handled (PR #496).
* Streamlined inheritance from ``Operand`` to ensure access to basic methods like ``__add__`` for all computable objects (``NDArray``, ``LazyExpr``, ``LazyArray`` etc.) (PR ##500).

## Changes from 3.9.0 to 3.9.1

* Bumped to numexpr 2.13.1 to incorporate new maximum/minimum NaN handling and +/* for booleans
  which matches NumPy behaviour.
* Refactoring in order to ensure Blosc2 functions with NumPy 1.26.
* Streamlined documentation by introducing Array Protocol

## Changes from 3.8.0 to 3.9.0
Most changes come from PR #467 relating to array-api compliance.

* C-Blosc2 internal library updated to latest 2.21.3, increasing MAX_DIMS from 8 to 16

* numexpr version requirement pushed to 2.13.0 to incorporate
``round``, ``sign``, ``signbit``, ``copysign``, ``nextafter``, ``hypot``,
``maximum``, ``minimum``, ``trunc``, ``log2`` functions, as well as allow
integer outputs for certain functions when integr arguments are passed.
We also add floor division (``//``) and full dual bitwise (logical) AND, OR, XOR, NOT
support for integer (bool) arrays.

* Extended linear algebra functionality, offering generalised matrix multiplication
for arrays of arbitrary dimension via ``tensordot`` and an improved ``matmul``. In addition,
introduced ``vecdot``, ``diagonal`` and ``outer``, as well as useful indexing and associated functions such as ``take``, ``take_along_axis``, ``meshgrid`` and ``broadcast_to``.

* Added many ufuncs and methods (around 60) to ``NDArray`` to bring the library into further alignment with the array-api. Introduced a chunkwise lazyudf paradigm which is very powerful in order to implement ``clip`` and ``logaddexp``.

* Fixed a subtle but important bug for ``expand_dims`` (PR #479, PR #483) relating to reference counting for views.

## Changes from 3.7.2 to 3.8.0

* C-Blosc2 internal library updated to latest 2.21.2.

* numexpr version requirement pushed to 2.12.1 to incorporate
``isnan``, ``isfinite``, ``isinf`` functions.

* Indexing is now supported extensively and reasonably optimally for slices
with negative steps and general boolean arrays, with both get/setitem having
equal functionality. In PR #459 we extended the 1D fast path to general N-D,
with consequent speedups. In PR # we allowed fancy indexing and general slicing
with negative steps for set and getitem, with a memory-optimised path for setitem.

* Various attributes and methods for the ``NDArray`` class, as well as functions, have
been added to increase compliance with the array-api standard. In addition,
linspace and arange functions have been made more numerically stable and now strictly
comply even with difficult floating-point edge cases.

## Changes from 3.7.1 to 3.7.2

* C-Blosc2 internal library updated to latest 2.21.1.

* Revert signature of `TreeStore.__init__` for making benchmarks to get back
  to normal performance.

## Changes from 3.7.0 to 3.7.1

* Added `C2Array.slice()` method and `C2Array.nbytes`, `C2Array.cbytes`, `C2Array.cratio`, `C2Array.vlmeta` and `C2Array.info` properties (PR #455).

* Many usability improvements to the `TreeStore` class and friends.

* New section about `TreeStore` in basics NDArray tutorial.

* New blog post about `TreeStore` usage and performance at: https://www.blosc.org/posts/new-treestore-blosc2

* C-Blosc2 internal library updated to latest 2.21.0.

## Changes from 3.6.1 to 3.7.0

* Overhaul of documentation (API reference and Tutorials)

* Improvements to lazy expression indexing and in particular much more efficient memory usage when applying non-unit steps (PR #446).

* Extended functionality of ``expand_dims`` to match that of NumPy (note that this breaks the previous API) (PR #453).

* The biggest change is in the form of three new data storage classes (``EmbedStore``, ``DictStore`` and ``TreeStore``) which allow for the efficient storage of heterogeneous array data (PR #451). ``EmbedStore`` is essentially an ``SChunk`` wrapper which can be stored on-disk or in-memory; ``DictStore`` allows for mixed storage across memory, disk or indeed remote; and ``TreeStore`` is a hieracrhically-formatted version of ``DictStore`` which mimics the HDF5 file format. Write, access and storage performance are all very competitive with other packages - see [plots here](https://github.com/Blosc/python-blosc2/pull/451#issuecomment-3178828765).

## Changes from 3.6.0 to 3.6.1

* C-Blosc2 internal library updated to latest 2.19.1.

## Changes from 3.5.1 to 3.6.0

* Expose the `oindex` C-level functionality in Blosc2 for `NDArray`.

* Implement fancy indexing which closely matches NumPy functionality, using
`ndindex` library. Includes a fast path for 1D arrays, based on Zarr's implementation.

* A major refactoring of slicing for lazy expressions using `ndindex`. We have also
added support for slices with non-unit steps for reduction expressions, which has introduced
improvements that could be incorporated into other lazy expression machinery in the future.
More complex slicing is now supported.

* Minor bug fixes to ensure that Blosc2 indexing does not introduce dummy dimensions when NumPy does not,
and a more comprehensive `squeeze` function which squeezes specified dimensions.

## Changes from 3.5.0 to 3.5.1

* Reduced memory usage when computing slices of lazy expressions.
  This is a significant improvement for large arrays (up to 20x less).
  Also, we have added a fast path for slices that are small and fit in
  memory, which can be up to 20x faster than the previous implementation.
  See PR #430.

* `blosc2.concatenate()` has been renamed to `blosc2.concat()`.
  This is in line with the [Array API](https://data-apis.org/array-api).
  The old name is still available for backward compatibility, but it will
  be removed in a future release.

* Improve mode handling for concatenating to disk. See PR #428.
  Useful for concatenating arrays that are stored in disk, and allows
  specifying the mode to use when concatenating.

## Changes from 3.4.0 to 3.5.0

* New `blosc2.stack()` function for stacking multiple arrays along a new axis.
  Useful for creating multi-dimensional arrays from multiple 1D arrays.
  See PR #427. Thanks to [Luke Shaw](@lshaw8317) for the implementation!
  Blog: https://www.blosc.org/posts/blosc2-new-concatenate/#stacking-arrays

* New `blosc2.expand_dims()` function for expanding the dimensions of an array.
  This is useful for adding a new axis to an array, similar to NumPy's `np.expand_dims()`.
  See PR #427. Thanks to [Luke Shaw](@lshaw8317) for the implementation!

## Changes from 3.3.4 to 3.4.0

* Added C-level ``concatenate`` function in response to community request. When possible, uses an optimised path which avoids decompression and recompression, giving a significant performance boost. See PR #423.

* Slicing has been added to string-based lazyexprs, so that one may use
  expressions like `expr[1:3] +1` to compute a slice of the expression. This is useful
  for getting a sub-expression of a larger expression, and it works with both
  string-based and lazy expressions. See PR #417.

* Relatedly, the behaviour of the `slice` parameter in the `compute()` method of `LazyExpr` has been made more consistent and is now better documented, so that results are as expected. See PR #419.

* UDF support for pandas has been added to allow for the use of ``blosc2.jit``. See PR #418. Thanks to [@datapythonista](https://github.com/datapythonista) for the implementation!

## Changes from 3.3.3 to 3.3.4

* Expand possibilities for chaining string-based lazy expressions to incorporate
  data types which do not have shape attribute, e.g. int, float etc.
  See #406 and PR #411.

* Enable slicing within string-based lazy expressions. See PR #414.

* Improved casting for string-based lazy expressions.

* Documentation improvements, see PR #410.

* Compatibility fixes for working with `h5py` files.

## Changes from 3.3.2 to 3.3.3

* Expand possibilities for chaining string-based lazy expressions to include
  main operand types (LazyExpr and NDArray). Still have to incorporate other
  data types (which do not have shape attribute, e.g. int, float etc.).
  See #406.

* Fix indexing for lazy expressions, and allow use of None in getitem.
  See PR #402.

* Fix incorrect appending of dim to computed reductions. See PR #404.

* Fix `blosc2.linspace()` for incompatible num/shape.  See PR #408.

* Add support for NumPy dtypes that are n-dimensional (e.g.
  `np.dtype(("<i4,>f4", (10,))),`).

* New MAX_DIM constant for the maximum number of dimensions supported.
  This is useful for checking if a given array is too large to be handled.

* More refinements on guessing cache sizes for Linux.

* Update to C-Blosc2 2.17.2.dev.  Now, we are forcing the flush of modified
  pages only in write mode for mmap files. This fixes mmap issues on Windows.
  Thanks to @JanSellner for the implementation.

## Changes from 3.3.1 to 3.3.2

* Fixed a bug in the determination of chunk shape for the `NDArray` constructor.
  This was causing problems when creating `NDArray` instances with a CPU that
  was reporting a L3 cache size close (or exceeding) 2 GB.  See PR #392.

* Fixed a bug preventing the correct chaining of *string* lazy expressions for
  logical operators (`&`, `|`, `^`...).  See PR #391.

* More performance optimization for `blosc2.permute_dims`.  Thanks to
  Ricardo Sales Piquer (@ricardosp4) for the implementation.

* Now, storage defaults (`blosc2.storage_dflts`) are honored, even if no
  `storage=` param is used in constructors.

* We are distributing Python 3.10 wheels now.

## Changes from 3.3.0 to 3.3.1

* In our effort to better adapt to better adapt to the array API
  (https://data-apis.org/array-api/latest/), we have introduced
  permute_dims() and matrix_transpose() functions, and the .T property.
  This replaces to previous transpose() function, which is now deprecated.
  See PR #384.  Thanks to Ricardo Sales Piquer (@ricardosp4).

* Constructors like `arange()`, `linspace()` and `fromiter()` now
  use far less memory when creating large arrays. As an example, a 5 TB
  array of 8-byte floats now uses less than 200 MB of memory instead of
  170 GB previously.  See PR #387.

* Now, when opening a lazy expression with `blosc2.open()`, and there is
  a missing operand, the open still works, but the dtype and shape
  attributes are None.  This is useful for lazy expressions that have
  lost some operands, but you still want to open them for inspection.
  See PR #385.

* Added an example of getting a slice out of a C2Array.

## Changes from 3.2.1 to 3.3.0

* New `blosc2.transpose()` function for transposing 2D NDArray instances
  natively. See PR #375 and docs at
  https://www.blosc.org/python-blosc2/reference/autofiles/operations_with_arrays/blosc2.transpose.html#blosc2.transpose
  Thanks to Ricardo Sales Piquer (@ricardosp4) for the implementation.

* New fast path for `NDArray.slice()` for getting slices that are aligned with
  underlying chunks. This is a common operation when working with NDArray
  instances, and now it is up to 40x faster in our benchmarks (see PR #380).

* Returned `NDArray` object in `NDarray.slice()` now defaults to original
  codec/clevel/filters. The previous behavior was to use the default
  codec/clevel/filters.  See PR #378.  Thanks to Luke Shaw (@lshaw8317).

* Several English edits in the documentation.  Thanks to Luke Shaw (@lshaw8317)
  for his help in this area.

## Changes from 3.2.0 to 3.2.1

* The array containers are now using the `__array_interface__` protocol to
  expose the data in the array.  This allows for better interoperability with
  other libraries that support the `__array_interface__` protocol, like NumPy,
  CuPy, etc.  Now, the range of functions that can be used within the `blosc2.jit`
  decorator is way larger, and essentially all NumPy functions should work now.

  See examples at: https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/jit-numpy-funcs.py
  See benchmarks at: https://github.com/Blosc/python-blosc2/blob/main/bench/ndarray/jit-numpy-funcs.py

* The performance of constructors like `arange()`, `linspace()` and `fromiter()`
  has been improved.  Now, they can be up to 3x faster, specially with large
  arrays.

* C-Blosc2 updated to 2.17.1.  This fixes various UB as well as compiler warnings.

## Changes from 3.1.1 to 3.2.0

* Structured arrays can be larger than 255 bytes now.  This was a limitation
  in the previous versions, but now it is gone (the new limit is ~512 MB,
  which I hope will be enough for some time).

* New `blosc2.matmul()` function for computing matrix multiplication on NDArray
  instances.  This allows for efficient computations on compressed data that
  can be in-memory, on-disk and in the network.  See
  [here](https://www.blosc.org/python-blosc2/reference/autofiles/operations_with_arrays/blosc2.matmul.html)
  for more information.

* Support for building WASM32 wheels.  This is a new feature that allows to
  build wheels for WebAssembly 32-bit platforms.  This is useful for running
  Python code in the browser.

* Tested support for NumPy<2 (at least 1.26 series).  Now, the library should
  work with NumPy 1.26 and up.

* C-Blosc2 updated to 2.17.0.

* httpx has replaced by requests library for the remote proxy.  This has been
  done to avoid the need of the `httpx` library, which is not supported by
  Pyodide.

## Changes from 3.1.0 to 3.1.1

* Quick release to fix an issue with version number in the package (was reporting 3.0.0
  instead of 3.1.0).


## Changes from 3.0.0 to 3.1.0

### Improvements

* Optimizations for the compute engine. Now, it is faster and uses less memory.
  In particular, careful attention has been paid to the memory handling, as
  this is the main bottleneck for the compute engine in many instances.

* Improved detection of CPU cache sizes for Linux and macOS.  In particular,
  support for multi-CCX (AMD EPYC) and multi-socket systems has been implemented.
  Now, the library should be able to detect the cache sizes for most of the
  CPUs out there (specially on Linux).

* Optimization on NDArray slicing when the slice is a single chunk.  This is a
  common operation when working with NDArray instances, and now it is faster.

### New API functions and decorators

* New `blosc2.evaluate()` function for evaluating expressions on NDArray/NumPy
  instances.  This a drop-in replacement of `numexpr.evaluate()`, but with the
  next improvements:
  - More functionality than numexpr (e.g. reductions).
  - Follow casting rules of NumPy more closely.
  - Use both NumPy arrays and Blosc2 NDArrays in the same expression.

  See [here](https://www.blosc.org/python-blosc2/reference/autofiles/utilities/blosc2.evaluate.html)
  for more information.

* New `blosc2.jit` decorator for allowing NumPy expressions to be computed
  using the Blosc2 compute engine.  This is a powerful feature that allows for
  efficient computations on compressed data, and supports advanced features like
  reductions, filters and broadcasting.  See
  [here](https://www.blosc.org/python-blosc2/reference/autofiles/utilities/blosc2.jit.html)
  for more information.

* Support `out=` in `blosc2.mean()`, `blosc2.std()` and `blosc2.var()` reductions
  (besides `blosc2.sum()` and `blosc2.prod()`).


### Others

* Bumped to use latest C-Blosc2 sources (2.16.0).

* The cache for cpuinfo is now stored in `${HOME}/.cache/python-blosc2/cpuinfo.json`
  instead of `${HOME}/.blosc2-cpuinfo.json`; you can get rid of the latter, as
  the former is more standard (see PR #360). Thanks to Jonas Lundholm Bertelsen
  (@jonaslb).

## Changes from 3.0.0-rc.3 to 3.0.0

* A persistent cache for cpuinfo (stored in `$HOME/.blosc2-cpuinfo.json`) is
  now used to avoid repeated calls to the cpuinfo library.  This accelerates
  the startup time of the library considerably (up to 5x on my box).

* We should be creating conda packages now.  Thanks to @hmaarrfk for his
  assistance in this area.


## Changes from 3.0.0-rc.2 to 3.0.0-rc.3

* Now you can get and set the whole values of VLMeta instances with the `vlmeta[:]` syntax.
  The get part is syntactic sugar for `vlmeta.getall()` actually.

* `blosc2.copy()` now honors `cparams=` parameter.

* Now, compiling the package with `USE_SYSTEM_BLOSC2` envar set to `1` will use the
  system-wide Blosc2 library.  This is useful for creating packages that do not want
  to bundle the Blosc2 library (e.g. conda).

* Several changes in the build process to enable conda-forge packaging.

* Now, `blosc2.pack_tensor()` can pack empty tensors/arrays.  Fixes #290.


## Changes from 3.0.0-rc.1 to 3.0.0-rc.2

* Improved docs, tutorials and examples.  Have a look at our new docs at: https://www.blosc.org/python-blosc2.

* `blosc2.save()` is using `contiguous=True` by default now.

* `vlmeta[:]` is syntactic sugar for vlmeta.getall() now.

* Add `NDArray.meta` property as a proxy to `NDArray.shunk.vlmeta`.

* Reductions over single fields in structured NDArrays are now supported.  For example, given an array `sarr` with fields 'a', 'b' and 'c', `sarr["a"]["b >= c"].std()` returns the standard deviation of the values in field 'a' for the rows that fulfills that values in fields in 'b' are larger than values in 'c' (`b >= c` above).

* As per discussion #337, the default of cparams.splitmode is now AUTO_SPLIT. See #338 though.


## Changes from 3.0.0-beta.4 to 3.0.0-rc.1

### General improvements

* New ufunc support for NDArray instances. Now, you can use NumPy ufuncs on NDArray instances, and mix them with other NumPy arrays. This is a powerful feature that allows for more interoperability with NumPy.

* Enhanced dtype inference, so that it mimics now more NumPy than the numexpr one. Although perfect adherence to NumPy casting conventions is not there yet, it is a big step forward towards better compatibility with NumPy.

* Fix dtype for sum and prod reductions. Now, the dtype of the result of a sum or prod reduction is the same as the input array, unless the dtype is not supported by the reduction, in which case the dtype is promoted to a supported one. It is more NumPy-like now.

* Many improvements on the computation of UDFs (User Defined Functions). Now, the lazy UDF computation is way more robust and efficient.

* Support reductions inside queries in structured NDArrays. For example, given an array `sarr` with fields 'a', 'b' and 'c', the next `farr = sarr["b >= c"].sum("a").compute()` puts in `farr` the sum of the values in field 'a' for the rows that fulfills that values in fields in 'b' are larger than values in 'c' (b >= c above).

* Implemented combining data filtering, as well as sorting, in structured NDArrays. For example, given an array `sarr` with fields 'a', 'b' and 'c', the next `farr = sarr["b >= c"].indices(order="c").compute()` puts in farr the indices of the rows that fulfills that values in fields in 'b' are larger than values in 'c' (`b >= c` above), ordered by column 'c'.

* Reductions can be stored in persistent lazy expressions. Now, if you have a lazy expression that contains a reduction, the result of the reduction is preserved in the expression, so that you can reuse it later on. See https://www.blosc.org/posts/persistent-reductions/ for more information.

* Many improvements in ruff linting and code style. Thanks to @DimitriPapadopoulos for the excellent work in this area.

### API changes

* `LazyArray.eval()` has been renamed to `LazyArray.compute()`. This avoids confusion with the `eval()` function in Python, and it is more in line with the Dask API.

This is the main change in the API that is not backward compatible with previous beta. If you have code that still uses `LazyArray.eval()`, you should change it to `LazyArray.compute()`.  Starting from this release, the API will be stable and backward compatibility will be maintained.

### New API calls

* New `reshape()` function and `NDArray.reshape()` method allow to do efficient reshaping between NDArrays that follows C order. Only 1-dim -> n-dim is currently supported though.

* `New NDArray.__iter__()` iterator following NumPy conventions.

* Now, `NDArray.__getitem__()` supports (n-dim) bool arrays or sequences of integers as indices (only 1-dim for now). This follows NumPy conventions.

* A new `NDField.__setitem__()` has been added to allow for setting values in a structured NDArray.

* `struct_ndarr['field']` now works as in NumPy, that is, it returns an array with the values in 'field' in the structured NDArray.

* Several new constructors are available for creating NDArray instances, like `arange()`, `linspace()` and `fromiter()`. These constructors leverage the internal `lazyudf()` function and make it easier to create NDArray instances from scratch. See e.g. https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/arange-constructor.py for an example.

* Structured LazyArrays received a new `.indices()` method that returns the indices of the elements that fulfill a condition. When combined with the new support of list of indices as key for `NDArray.__getitem__()`, this is useful for creating indexes for data.  See https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/filter_sort_fields.py for an example.

* LazyArrays received a new `.sort()` method that sorts the elements in the array.  For example, given an array `sarr` with fields 'a', 'b' and 'c', the next `farr = sarr["b >= c"].sort("c").compute()` puts in `farr` the rows that fulfills that values in fields in 'b' are larger than values in 'c' (`b >= c` above), ordered by column 'c'.

* New `expr_operands()` function for extracting operands from a string expression.

* New `validate_expr()` function for validating a string expression.

* New `CParams`, `DParams` and `Storage` dataclasses for better handling of parameters in the library. Now, you can use these dataclasses to pass parameters to the library, and get a better error handling. Thanks to @martaiborra for the excellent implementation and @omaech for revamping docs and examples to use them.  See e.g. https://www.blosc.org/python-blosc2/getting_started/tutorials/02.lazyarray-expressions.html.

### Documentation improvements

* Much improved documentation on how to efficiently compute with compressed NDArray data. Documentation updates highlight these features and improve usability for new users. Thanks to @omaech and @martaiborra for their excellent work on the documentation and examples, and to @NumFOCUS for their support in making this possible!  See https://www.blosc.org/python-blosc2/getting_started/tutorials/04.reductions.html for an example.

* New remote proxy tutorial. This tutorial shows how to use the Proxy class to access remote arrays, while providing caching. https://www.blosc.org/python-blosc2/getting_started/tutorials/06.remote_proxy.html . Thanks to @omaech for her work on this tutorial.

* New tutorial on "Mastering Persistent, Dynamic Reductions and Lazy Expressions". See https://www.blosc.org/posts/persistent-reductions/


## Changes from 3.0.0-beta.3 to 3.0.0-beta.4

* Many new examples in the documentation.  Now, the documentation is more complete and has a better structure.
 Have a look at our new docs at: https://www.blosc.org/python-blosc2/
 For a guide on using UDFs, check out: https://www.blosc.org/python-blosc2/reference/autofiles/lazyarray/blosc2.lazyudf.html
 If interested in asynchronously fetching parts of an array, take a look at: https://www.blosc.org/python-blosc2/reference/autofiles/proxy/blosc2.Proxy.afetch.html
 Finally, there is a new tutorial on optimizing reductions in large NDArray objects: https://www.blosc.org/python-blosc2/getting_started/tutorials/04.reductions.html
 Special thanks @omaech and @martaiborrar for the excellent work on the documentation and examples, and to @NumFOCUS for their support in making this possible!

* New CParams, DParams and Storage dataclasses for better handling of parameters in the library.  Now, you can use these dataclasses to pass parameters to the library, and get a better error handling.  See [here](https://www.blosc.org/python-blosc2/reference/storage.html).  Thanks to @martaiborra for the excellent implementation.

* Better support for CParams in Proxy and C2Array instances.  This allows to better propagate compression parameters from Caterva2 datasets to the Proxy and C2Array instances, improving the perception of codecs and filters used originally in datasets.  Thanks to @FrancescAlted for the implementation.

* Many improvements in ruff linting and code style.  Thanks to @DimitriPapadopoulos for the excellent work in this area.


## Changes from 3.0.0-beta.1 to 3.0.0-beta.3

* Revamped documentation.  Now, the documentation is more complete and has a better structure. See [here](https://www.blosc.org/python-blosc2/).  Thanks to Oumaima Ech Chdig (@omaech), our newcomer to the Blosc team.  Also, thanks to NumFOCUS for the support in this task.

* New `Proxy` class to access other arrays, while providing caching. This is useful for example when you have a big array, and you want to access a small part of it, but you want to cache the accessed data for later use.  See [its doc](https://www.blosc.org/python-blosc2/reference/proxy.html).

* Lazy expressions can accept proxies as operands.

* Read-ahead support for reading super-chunks from disk.  This allows for overlapping reads and computations, which can be a big performance boost for some workloads.

* New BLOSC_LOW_MEM envar for keeping memory under a minimum while evaluating expressions.  This makes it possible to evaluate expressions on very large arrays, even if the memory is limited (at the expense of performance).

* Fine tune block sizes for the internal compute engine.

* Better CPU cache size guessing for linux and macOS.

* Build tooling has been modernized and now uses `pyproject.toml` and `scikit-build-core` for managing dependencies and building the package.  Thanks to @LecrisUT for the excellent guidance in this area.

* Many code cleanup and syntax improvements in code.  Thanks to @DimitriPapadopoulos.


## Changes from 2.6.2 to 3.0.0-beta.1

* New evaluation engine (based on numexpr) for NDArray instances.  Now, you can evaluate expressions like `a + b + 1` where `a` and `b` are NDArray instances.  This is a powerful feature that allows for efficient computations on compressed data, and supports advanced features like reductions, filters, user-defined functions and broadcasting (still in beta).  See this [example](https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/eval_expr.py).

* As a consequence of the above, there are many new functions to operate with, and evaluate NDArray instances.  See the [function section docs](https://www.blosc.org/python-blosc2/reference/operations_with_arrays.html#functions) for more information.

* Support for NumPy 2.0.0 is here!  Now, the wheels are built with NumPy 2.0.0. If you want to use NumPy 1.x, you can still use it by installing NumPy 1.23 and up.

* Support for memory mapping in `SChunk` and `NDArray` instances.  This allows to map super-chunks stored in disk and access them as if they were in memory.  If curious, see  [some benchmarks here](https://github.com/Blosc/python-blosc2/blob/main/examples/ndarray/eval_expr.py).  Thanks to @JanSellner for the excellent implementation, both in the C and the Python libraries.

* Internal C-Blosc2 updated to 2.15.0.

* 32-bit platforms are officially unsupported now.  If you need support for 32-bit platforms, please use python-blosc 1.x series.

## Changes for 2.x series

* See the [release notes](https://github.com/Blosc/python-blosc2/blob/v2.x/RELEASE_NOTES.md) for the 2.x series.
