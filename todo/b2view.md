# b2view: improvements tracker

Running list of possible improvements for the `b2view` TUI.  The original
design document lives in `plans/b2view.md`; this file tracks incremental
work discovered while using and testing the viewer.

Tests live in `tests/b2view/` (marker `tui`); see the note at the top of
`tests/b2view/test_basics.py` before adding new ones.

## Pending

### Data panel

- [ ] CTable expensive columns (list/struct/object) show a `<...; skipped>`
      placeholder; offer on-demand decoding (e.g. a key to materialize the
      column, or decode just the cursor row).
- [ ] SChunk preview is not implemented (`model.preview` returns a message).
- [ ] `m` key in the plot modal: render a high-res matplotlib view of the
      *currently shown* range (the braille envelope stays the fast navigator;
      `m` is the explicit drill-down for the range you zoomed/panned to).
      Plot the *raw* values of the window read exactly (so "high-res" means more
      detail, not just more pixels of the same min/max buckets) — cap the window
      so an un-zoomed million-row series doesn't trigger a huge read (require
      zooming in, or fall back to the envelope above the cap).  Display via
      `textual-image` with a capability ladder: real image on kitty/iTerm2/sixel
      → half-blocks elsewhere → "terminal can't show images, staying in braille"
      message.  Push it as a screen on top of `PlotScreen` so `q` returns to the
      braille view with the zoom intact.  Deps: add `textual-image` and promote
      `matplotlib` from the dev group into the `plot` extra.  Gated on need:
      only worth it if the ~200-point braille resolution proves too coarse.
- [ ] Live mini-plot in the data panel that follows paging: a small,
      always-visible braille plot of the current row window (or cursor column),
      redrawn on paging — a sparkline companion to the table, vs. the one-shot
      `p` modal.  Reuses `plot_series`; the work is layout (find room in the
      data panel) and wiring the redraw to the paging events.
### Testing

- [ ] Visual regressions: consider `pytest-textual-snapshot` (SVG snapshots)
      if rendering glitches become a recurring theme.

## Done

- 2026-06-14: NDArray sources also support the `v` locked window, copy-free via
  the layout (not `NDArray.slice`, which copies).  `DataSliceLayout` gained a
  `row_window` field + `row_window_bounds`; `preview_array_from_layout` narrows
  the navigable row dim to `[w0, w1)` — it reports `nrows = w1 - w0` (so paging
  is bounded) and offsets every read by `w0` (so logical row 0 reads absolute
  `w0`).  `_sync_layout_scroll` clamps scroll to the window length.
  `_view_plot_range` routes ctable→slice-view / ndarray→layout, and
  `_enter`/`_exit_row_window` share a `_reload_row_window` helper.  Covered by
  the NDArray-leaf window assertions in the extended `test_plot_column`.
- 2026-06-14: `v` in the plot modal locks the data grid to the navigated row
  range (esc unlocks).  Backed by a new public `CTable.slice(start, stop=None,
  *, copy=True)` — `range`-style/`slice`-object bounds in live-row space,
  `copy=False` returns a zero-copy view (via `_view_from_positions`, like
  `head`/`tail`), `copy=True` a compact copy (via `take`), mirroring
  `NDArray.slice`.  b2view registers the `copy=False` view per-path in the
  model's `_window_views` (precedence over `_filter_views`, so it composes over
  an active filter); `len(view)` bounds paging for free.  App holds
  `self.row_window`; `_enter_row_window`/`_exit_row_window` reload in place, the
  header shows a `WINDOW a:b` chip, and `action_dim_exit` gained the unlock
  layer.  Tests: `CTable.slice` cases in `tests/ctable/test_ctable_take.py` and
  `test_plot_view_locks_ctable_window`.
- 2026-06-13: The `p` plot modal is now zoomable into a row range.
  `plot_series` gained `row_start`/`row_stop` (the whole series keeps the fast
  SUMMARY tier; a sub-range is read exactly, with `x` in absolute rows).
  `PlotScreen` holds a fetch closure + total `n` and re-queries on `+`/`-`
  (zoom about centre), `←`/`→` (pan), `0` (reset), `g` (type an exact
  `start:stop` via `PlotRangeScreen`); a key hint line and a `?`-help group
  advertise the keys.  Tests: `tests/b2view/test_plot_model.py`
  (sub-range exactness, clamping/ordering) and the extended `test_plot_column`
  Pilot journey.
- 2026-06-13: Row paging re-aligns to the page grid after dim-mode single-row
  scrolls.  `_scroll_navigable_viewport` shifts `start` by one row, which used
  to make every later page up/down carry that offset; `page_table` now takes
  `align=` and an explicit page up/down (only — cursor-edge paging stays
  contiguous) snaps `start` to the nearest page_size boundary, mirroring column
  paging's per-page re-fit.  Regression covered in `test_2d_paging`.
- 2026-06-13: Tier-2 plot envelope is no longer capped at
  `_PLOT_FULL_READ_MAX_BYTES` (~1 GB).  Above the ceiling, **local** objects
  (CTable columns, N-D arrays) are streamed in bounded spans
  (`_minmax_buckets_streaming`, ~`_PLOT_STREAM_BUFFER_BYTES` per read, aligned
  to native chunks) and the envelope stays **exact** (`method="reduce"`); only
  remote `c2array`s still fall back to the labeled strided `sample` (streaming
  would mean many round-trips).  Min/max are associative, so arbitrary span
  boundaries reproduce the single-read result bit-for-bit.  Unit tests in
  `tests/b2view/test_plot_model.py` (exactness vs full read, spike a sample
  would miss, all-NaN/int/edge cases, remote-stays-sample).
- 2026-06-12: Pilot-based test suite (`tests/b2view/test_basics.py`) with a
  deterministic store generator (`tests/b2view/tree_store_gen.py`); marker
  `tui`.
- 2026-06-12: CTable column paging (wide tables were unreachable past the
  first window); `preview_ctable` gained `col_start`/`ncols` bookkeeping.
- 2026-06-12: Viewport-consistency reload — the first page of a node was
  sized before layout settled (CLI fallbacks vs real viewport), making
  paging windows drift; also handles terminal resize.
- 2026-06-12: Column paging windows are aligned to page-size multiples
  (ragged last window no longer shifts subsequent pages); `end` jumps to
  the last aligned window, mirroring `b` for rows.
- 2026-06-12: Two-pass column fit — the preview fetches a generous candidate
  window, then trims to the columns whose *measured* rendered widths fit the
  pane (was a fixed ~11 chars/column estimate that wasted half the width on
  narrow bool/int columns).  Paging right starts at the first hidden column;
  paging left and `end` fit whole columns backward; windows are stable
  within a row buffer (widths measured over the buffer, not the visible
  page).  Superseded the fixed-multiple alignment policy above.
- 2026-06-12: Uniform decimals per float column — the decimal count is
  chosen once per column from its max magnitude in the buffer
  (`column_float_decimals` in render.py) instead of per value, so decimal
  points align down the column; zeros are formatted like their neighbors
  (all-zero columns still show plain 0.0).  Unit tests in
  `tests/b2view/test_render.py`.
- 2026-06-12: Row paging/jumps (page up/down, `t`/`b`, `g`oto, dim-mode
  changes) keep the cursor on its current column; only selecting a new node
  resets it to the first column.
- 2026-06-12: `s`/`e` keys jump to the start/end column window (aliases of
  Home/End, which were undiscoverable); the data panel subtitle now lists
  all jump keys: `rows: t/b/g | cols: s/e`.
- 2026-06-12: `p` plots the cursor column (or a 1-D leaf) of the loaded row
  buffer in a modal, via the optional `textual-plotext` package (new `plot`
  extra); braille scatter, NaN/inf filtered, non-numeric columns and a
  missing package just notify.  Works headless in Pilot tests.
- 2026-06-12: The `p` plot shows a downsampled overview of the *whole*
  series (`StoreBrowser.plot_series`); honors layout (fixed dims) for N-D
  arrays and active row filters for CTables.
- 2026-06-13: `p` plot is now a peak-preserving **min/max envelope** (was
  plain strided decimation, which aliased and hid extremes between samples).
  `plot_series` returns `{x, ymin, ymax, n, method}` and picks the cheapest
  tier: (1) `summary` — per-block min/max straight from the column's SUMMARY
  index, no decompression (~44x faster, the big win for large persisted /
  parquet columns; identity case only); (2) `reduce` — read + per-bucket
  min/max, bounded by `_PLOT_FULL_READ_MAX_BYTES` (~1 GB); (3) `sample` —
  labeled strided fallback above that ceiling.  `PlotScreen` draws the
  upper/lower envelope; the title states the method.
- 2026-06-12: `?` opens a help screen listing all keys grouped by area
  (panels, tree, grid rows/columns, dim mode); shown in the footer, closed
  with esc/`?`/`q`.
- 2026-06-12: `c` opens a go-to-column modal: accepts a column index, and
  for CTables also an exact column name or a unique name prefix; the target
  becomes the first visible column, keeping the row position.
- 2026-06-12: Resize Pilot test (`pilot.resize_terminal`) — it immediately
  caught that the resize handler lived on the App, which never receives
  Resize events; moved to BufferedDataTable.on_resize, so the windows now
  re-fit on terminal resize and panel maximize/restore for real.
- 2026-06-12: The terminal owns the mouse by default, so native text
  selection/copy works; `--mouse` lets b2view capture it instead
  (click-to-focus, wheel scrolls the data grid by half a page, paging at
  the edges via the same path as the arrow keys).
- 2026-06-12: Dim-mode index/viewport movements clamp at the boundaries
  instead of wrapping (left/right dimension *selection* still cycles);
  navigable viewports clamp to the last full page / whole-column window.
- 2026-06-12: CTable row filtering — `f` opens a modal that takes the same
  string expressions as `CTable.where()` (dotted nested names, and/or) and
  pages through the matching view; escape or an empty expression clears,
  filters are remembered per node (`StoreBrowser.set_filter`), and the data
  header shows the active filter plus the unfiltered total.
- 2026-06-12: CTable column filtering — `/` filters the visible columns by
  case-insensitive substring (`StoreBrowser.set_column_filter`); column
  paging, the two-pass width fit and the `c` goto-column modal all operate
  on the filtered subset (`preview_ctable` already took a `columns`
  universe).  Combines freely with the row filter; escape clears one layer
  per press (dim mode, then rows, then columns).
