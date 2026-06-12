# b2view: improvements tracker

Running list of possible improvements for the `b2view` TUI.  The original
design document lives in `plans/b2view.md`; this file tracks incremental
work discovered while using and testing the viewer.

Tests live in `tests/b2view/` (marker `tui`); see the note at the top of
`tests/b2view/test_basics.py` before adding new ones.

## Pending

### Navigation

- [ ] Go-to-column: a column analogue of the `g`(oto row) modal, for jumping
      directly to a column index (arrays) or a column name (CTables).
- [ ] Column-name search/filter for wide CTables (e.g. `/` to filter the
      visible columns by substring).
- [ ] Row paging can lose page alignment after dim-mode single-row scrolls
      (`_scroll_navigable_viewport` shifts by 1); consider re-aligning on the
      next page up/down, as column paging does now.
- [ ] Discoverability: `home`/`end` column jumps and `t`/`b`/`g` row jumps
      are not visible in the footer or data panel subtitle; surface them.

### Data panel

- [ ] CTable expensive columns (list/struct/object) show a `<...; skipped>`
      placeholder; offer on-demand decoding (e.g. a key to materialize the
      column, or decode just the cursor row).
- [ ] SChunk preview is not implemented (`model.preview` returns a message).

### Testing

- [ ] Terminal-resize behavior: `on_resize` re-checks viewport consistency,
      but there is no Pilot test resizing the terminal mid-session.
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
