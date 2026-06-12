# b2view: improvements tracker

Running list of possible improvements for the `b2view` TUI.  The original
design document lives in `plans/b2view.md`; this file tracks incremental
work discovered while using and testing the viewer.

Tests live in `tests/b2view/` (marker `tui`); see the note at the top of
`tests/b2view/test_basics.py` before adding new ones.

## Pending

### Navigation

- [ ] Column-name search/filter for wide CTables (e.g. `/` to filter the
      visible columns by substring).  Note: the `c` goto-column modal already
      resolves unique name prefixes; this item is about *filtering* the
      visible set, not jumping.
- [ ] Row paging can lose page alignment after dim-mode single-row scrolls
      (`_scroll_navigable_viewport` shifts by 1); consider re-aligning on the
      next page up/down, as column paging does now.

### Data panel

- [ ] CTable expensive columns (list/struct/object) show a `<...; skipped>`
      placeholder; offer on-demand decoding (e.g. a key to materialize the
      column, or decode just the cursor row).
- [ ] SChunk preview is not implemented (`model.preview` returns a message).

### Testing

- [ ] Visual regressions: consider `pytest-textual-snapshot` (SVG snapshots)
      if rendering glitches become a recurring theme.

## Done

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
