# b2view: TreeStore TUI Viewer Plan

## Goal

Create a read-only terminal user interface named `b2view` for browsing Blosc2 `TreeStore` hierarchies stored as `.b2d` directories or `.b2z` files. The viewer should allow users to navigate groups, arrays, and ctable/table-like objects, inspect metadata, and preview data without eagerly loading large datasets into memory.

## Primary Use Cases

- Open a `.b2d` or `.b2z` TreeStore from the command line.
- Browse the hierarchical structure interactively.
- Distinguish groups, arrays, and ctable/table objects visually.
- Inspect object metadata such as shape, dtype, chunks, compression, filters, and user attributes.
- Preview small slices of arrays.
- Preview rows and columns of ctables.
- Navigate large objects safely using paging/slicing controls.

## Proposed Command

```bash
b2view path/to/store.b2d
b2view path/to/store.b2z
```

Optional future flags:

```bash
b2view store.b2d --path /experiments/run_001
b2view store.b2d --readonly
b2view store.b2d --preview-rows 50
b2view store.b2d --theme dark
```

## Recommended Technology

Use **Textual** as the TUI framework, with **Rich** for rendering metadata, tables, and formatted values.

Reasons:

- Built-in tree widgets are suitable for TreeStore hierarchy browsing.
- Supports split-pane layouts, tabs, scrollable panels, modals, keybindings, and mouse interaction.
- Rich integration is excellent for tables, pretty-printed dicts, JSON-like metadata, and styled output.
- Easier to maintain than raw curses or urwid.
- Async/background task support is useful for lazy metadata/data loading.

Alternatives considered:

- `curses`: too low-level for this UI.
- `urwid`: mature, but more cumbersome for modern layouts.
- `prompt_toolkit`: excellent for prompts/REPLs, less ideal for a full-screen browser.

## High-Level UI Layout

Initial layout:

```text
┌──────────────────── TreeStore ────────────────────┬────────────────────────────┐
│ /                                                   │ Object info                 │
│ ├── experiments                                     │ path: /experiments/run_001  │
│ │   ├── run_001                                     │ type: NDArray               │
│ │   │   ├── signal                                  │ shape: (10000, 128)         │
│ │   │   └── events                                  │ dtype: float32              │
│ │   └── run_002                                     │ chunks: ...                 │
│ └── metadata                                        │ compression: zstd           │
├─────────────────────────────────────────────────────┴────────────────────────────┤
│ Data preview                                                                      │
│                                                                                   │
│ array/table contents here                                                          │
└───────────────────────────────────────────────────────────────────────────────────┘
```

Core panels:

1. **Hierarchy tree**
   - Shows groups and children.
   - Uses different icons/styles for groups, arrays, ctables, and unknown objects.
   - Loads children lazily when nodes are expanded.

2. **Metadata/details panel**
   - Updates when a node is selected.
   - Shows core metadata and storage/compression information.
   - Shows user metadata/attributes if present.

3. **Data preview panel**
   - Shows a small preview of the selected object.
   - For arrays, shows a bounded slice.
   - For ctables, shows the first page of rows and selected columns.
   - Should never materialize a large full object by default.

Potential future panels:

- Search/find panel.
- Slice/query input panel.
- Statistics panel.
- Histogram/summary visualization panel.
- Export dialog.

## Read-Only First

The first version should be strictly read-only.

Avoid:

- Editing metadata.
- Deleting nodes.
- Renaming nodes.
- Writing modified arrays/tables.

This keeps the first implementation safe and avoids accidental mutation of user stores.

## Lazy Loading Requirements

Lazy loading is central to the design.

Startup should:

1. Validate/open the store.
2. Populate only the root node and immediate children if cheap.
3. Avoid recursively scanning the entire tree.
4. Avoid loading array/table data.

On tree expansion:

- Load only the selected node's children.
- Cache child listings where appropriate.
- Provide a refresh command later if the underlying store changes.

On node selection:

- Load lightweight metadata.
- Render metadata immediately.
- Load data preview separately, ideally in a background task.

On data preview:

- Use small bounded reads.
- Provide paging or slicing controls.
- Catch and display errors without crashing the TUI.

## Suggested Package Structure

If included inside `python-blosc2`:

```text
src/blosc2/b2view/
    __init__.py
    cli.py          # console entry point
    app.py          # Textual App subclass and layout
    model.py        # TreeStore adapter / browser abstraction
    widgets.py      # custom widgets/panels
    render.py       # Rich renderables for metadata and previews
    keys.py         # keybinding constants/help text, optional
```

Potential tests:

```text
tests/test_b2view_model.py
tests/test_b2view_render.py
```

Console script:

```toml
[project.scripts]
b2view = "blosc2.b2view.cli:main"
```

If Textual is considered too heavy for the base install, make it an optional dependency:

```toml
[project.optional-dependencies]
tui = ["textual", "rich"]
```

Then document installation as:

```bash
pip install "blosc2[tui]"
```

## Backend Abstraction

The UI should not directly depend on many TreeStore internals. Add a small model layer that exposes a stable browsing API.

Example sketch:

```python
@dataclass
class NodeInfo:
    path: str
    name: str
    kind: str  # group, ndarray, ctable, unknown
    has_children: bool | None = None


@dataclass
class ObjectInfo:
    path: str
    kind: str
    metadata: dict
    user_attrs: dict | None = None


class StoreBrowser:
    def __init__(self, urlpath: str): ...

    def list_children(self, path: str) -> list[NodeInfo]: ...

    def get_info(self, path: str) -> ObjectInfo: ...

    def preview(
        self, path: str, *, start: int = 0, stop: int = 20, columns=None, slices=None
    ): ...
```

Benefits:

- Keeps Textual code clean.
- Makes unit testing easier.
- Allows later support for other stores/backends.
- Centralizes object kind detection and safe preview logic.

## Object Kind Detection

The browser layer should classify nodes as:

- `group`: hierarchy-only container.
- `ndarray`: Blosc2 array object.
- `ctable`: ctable/table-like object.
- `scalar` or `metadata`: optional future classification.
- `unknown`: fallback for unsupported objects.

Detection should be robust and avoid expensive reads. Prefer metadata/type information available from TreeStore before opening or materializing objects.

## Metadata Display

Metadata panel should group information into sections.

Suggested sections:

### General

- Path
- Name
- Object kind
- Shape
- Number of dimensions
- Dtype
- Number of rows, for tables
- Number of columns, for tables
- Logical size / nbytes when available

### Storage

- Store type: `.b2d` or `.b2z`
- Chunks/blockshape
- Chunk count if available cheaply
- Contiguity / urlpath details
- Compression codec
- Compression level
- Filters
- Split mode / special parameters if relevant

### Table Schema

For ctables:

- Column names
- Column dtypes
- Column shapes if nested or multidimensional columns are supported
- Nullable/missing-value information if applicable

### User Metadata

- Attributes
- Application metadata
- Any serialized user metadata stored with the object

Use Rich renderables:

- `rich.table.Table` for key/value metadata.
- `rich.tree.Tree` or nested tables for structured metadata.
- `rich.pretty.Pretty` for dict-like values.
- JSON syntax highlighting for JSON-compatible metadata.

## Data Preview Behavior

### NDArray Preview

Default behavior should depend on dimensionality:

- 0-D: show scalar value.
- 1-D: show `arr[:N]`.
- 2-D: show `arr[:R, :C]`.
- N-D: show a 2-D plane using default slices, e.g. first index for leading dimensions and bounded rows/columns for the last two dimensions.

Example defaults:

```python
max_rows = 20
max_cols = 10
```

For high-dimensional arrays, display the active slice spec:

```text
slice: 0, 0, :, :20
```

Future controls:

- Edit slice expression.
- Increment/decrement selected axis.
- Page through rows/columns.
- Toggle NumPy-like repr vs table view.

### CTable Preview

Default behavior:

- Show first N rows.
- Show all columns if the count is small.
- Truncate or horizontally scroll if many columns.
- Preserve column names and dtypes.

Controls:

- Page down/up by rows.
- Jump to start/end.
- Select visible columns.
- Show one row in detail view.

Future query support:

- Simple column projection.
- Row filtering expressions.
- Sorting if supported cheaply.
- Export current view.

## Keybindings

Initial keybindings:

```text
q          quit
?          show help
enter      expand/collapse tree node or open selected item
space      expand/collapse tree node
up/down    move selection
left/right collapse/expand or move focus
Tab        switch focus between tree, metadata, preview
r          refresh selected node metadata/preview
PgUp/PgDn  page preview rows
Home/End   jump within preview
/          search paths, future
s          edit slice/query, future
e          export selected preview, future
```

Keybindings should be shown in a help modal.

## Error Handling

The TUI should handle errors gracefully:

- Invalid path.
- Unsupported store format.
- Corrupt or partially missing nodes.
- Permission errors.
- Preview read failures.
- Unsupported object kinds.

Errors should appear in a status bar or modal panel, not as raw tracebacks unless debug mode is enabled.

Optional debug flag:

```bash
b2view store.b2d --debug
```

## Testing Strategy

Focus tests on non-UI logic first.

### Unit tests

- `StoreBrowser` opens `.b2d` and `.b2z` stores.
- Root children are listed correctly.
- Nested children are listed correctly.
- Object kind classification works for groups, arrays, and ctables.
- Metadata extraction returns expected keys.
- Array preview uses bounded slices.
- CTable preview uses bounded row ranges.
- Missing/invalid paths raise controlled exceptions.

### Rendering tests

- Metadata dicts render without crashing.
- Array previews render for 0-D, 1-D, 2-D, and N-D arrays.
- Table previews render with many columns and many rows.

### TUI smoke tests

If Textual testing utilities are available:

- App starts with a temporary TreeStore.
- Root node appears.
- Expanding a node loads children.
- Selecting an array updates metadata and preview panels.

## Implementation Milestones

### Milestone 1: Backend browser prototype

- Add `StoreBrowser` model.
- Implement opening `.b2d` and `.b2z` stores.
- Implement child listing.
- Implement object kind detection.
- Implement metadata extraction.
- Add unit tests.

### Milestone 2: Rendering helpers

- Add Rich renderers for metadata.
- Add array preview renderer.
- Add ctable preview renderer.
- Add tests for renderers.

### Milestone 3: Minimal Textual app

- Add CLI entry point.
- Build layout with tree, metadata panel, and preview panel.
- Populate root node.
- Update metadata and preview on selection.
- Add basic keybindings.

### Milestone 4: Lazy expansion and paging

- Load tree children on expansion.
- Add preview paging for arrays/tables.
- Add status bar and loading/error indicators.

### Milestone 5: Polish

- Add help modal.
- Add path search.
- Add configurable preview row/column limits.
- Improve style/theme.
- Document usage.

## Documentation

Add user documentation covering:

- Installation, including optional TUI dependency if applicable.
- Basic usage.
- Keybindings.
- Safety/read-only behavior.
- Preview limitations.
- Examples with `.b2d` and `.b2z` stores.

Possible locations:

```text
doc/b2view.rst
examples/b2view_create_sample_store.py
```

## Open Questions

- Should `textual` be a required dependency or optional extra?
- What is the exact public API for TreeStore child listing and object metadata?
- How should ctable objects be detected robustly?
- Should the first version live inside `blosc2` or as a separate package?
- Should `.b2z` random access limitations affect preview behavior?
- What object metadata should be considered stable/public versus implementation detail?
- Is write support ever desired, or should this remain permanently read-only?

## Recommendation

Start with a read-only, lazy-loading Textual app and a well-tested `StoreBrowser` abstraction. Keep the first version focused on safe hierarchy browsing, metadata inspection, and small bounded previews. Add richer querying, slicing controls, export, and statistics only after the core browser is reliable.
