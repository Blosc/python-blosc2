# CTable Persistency Plan

## Goal

Add persistent `CTable` support on top of `TreeStore` while keeping the public
API simple:

* in-memory tables when `urlpath is None`
* persistent tables when `urlpath` is provided

The first persistency iteration should support:

* creating a persistent table
* opening an existing persistent table
* reading rows, columns, and views from persisted tables
* appending rows

The first persistency iteration should **not** promise:

* full schema evolution
* dropping columns
* renaming columns
* transactional multi-entry updates

For now, the supported schema evolution story is:

* append rows only

---

## Storage layout

Each persisted `CTable` lives under a table root inside a `TreeStore`.

Confirmed layout:

* `table_root/_meta`
* `table_root/_valid_rows`
* `table_root/_cols/<name>`

Example:

* `people/_meta`
* `people/_valid_rows`
* `people/_cols/id`
* `people/_cols/score`
* `people/_cols/active`

Rationale:

* `_meta` holds mutable metadata in `vlmeta`
* `_valid_rows` is real table data and should be stored as a normal persisted array
* `_cols/<name>` stores one persisted NDArray per column

The underscore-prefixed names form the internal namespace for a table root and
must be treated as reserved.

---

## `_meta` entry

`_meta` should be a small serialized `SChunk` used primarily to hold mutable
`vlmeta`.

This is preferable to immutable metalayers because:

* we may want to evolve metadata over time
* multiple `CTable` objects may live in the same `TreeStore`
* schema and table metadata should be updateable without rewriting the entire table

For the first version:

* `tree_store["<table_root>/_meta"].vlmeta["kind"] = "ctable"`
* `tree_store["<table_root>/_meta"].vlmeta["version"] = 1`
* `tree_store["<table_root>/_meta"].vlmeta["schema"] = {...}`

This gives `open()` a minimal, reliable contract for introspection.

---

## Schema persistence format

The schema should be stored as JSON-compatible data in:

* `tree_store["<table_root>/_meta"].vlmeta["schema"]`

The schema document should be versioned and explicit.

Recommended shape:

```python
{
    "version": 1,
    "columns": [
        {
            "name": "id",
            "py_type": "int",
            "spec": {"kind": "int64", "ge": 0},
            "default": None,
        },
        {
            "name": "score",
            "py_type": "float",
            "spec": {"kind": "float64", "ge": 0, "le": 100},
            "default": None,
        },
        {
            "name": "active",
            "py_type": "bool",
            "spec": {"kind": "bool"},
            "default": True,
        },
    ],
}
```

Notes:

* `columns` must be an ordered list, not a dict.
* The order of the list is the source of truth for column order.
* Do not rely on dict ordering or TreeStore iteration order.
* The schema JSON should capture logical schema information only.

For the first version, do **not** duplicate:

* per-column `cparams`
* per-column `dparams`
* array chunk/block layout
* `expected_size`
* compaction settings

Those can be introspected directly from the stored arrays when needed.

---

## `_valid_rows` persistence

`_valid_rows` should be stored as a normal persisted boolean NDArray under:

* `table_root/_valid_rows`

This is the correct representation because `_valid_rows` is:

* table data, not metadata
* potentially large
* used in normal row visibility semantics
* already aligned with current delete/view/compaction logic

Do not encode `_valid_rows` into schema JSON or small metadata blobs.

---

## Column persistence

Each column should be stored as its own persisted NDArray under:

* `table_root/_cols/<name>`

This means:

* each column can be opened independently
* column-level array settings remain attached to the actual stored array
* persistence layout matches the internal columnar design cleanly

The schema JSON provides the logical order and type constraints; the arrays under
`_cols` provide the physical stored data.

---

## Constructor semantics

The recommended constructor shape is:

```python
table = b2.CTable(
    Row,
    urlpath=None,
    mode="a",
    expected_size=1_048_576,
    compact=False,
    validate=True,
)
```

Semantics:

* `urlpath is None`
  create an in-memory `CTable`
* `urlpath is not None`
  use persistent storage rooted at that path

Recommended `mode` meanings:

* `mode="w"`
  create a new persistent table, overwriting any existing table root if the API
  already supports that pattern elsewhere
* `mode="a"`
  open existing or create new
* `mode="r"`
  open existing read-only table

The important public signal is:

* `urlpath` chooses persistence
* `mode` chooses creation/open behavior

Users should not need to pass a `TreeStore` object explicitly for the common path.

---

## `open()` support

An explicit `open()` API should be supported.

Recommended shape:

```python
table = b2.open(urlpath)
```

or, if needed for clarity:

```python
table = b2.CTable.open(urlpath, mode="r")
```

For `open()` to detect a persisted `CTable`, it should inspect:

* `urlpath/_meta`
* `urlpath/_meta`.vlmeta["kind"]

If:

* `_meta` exists
* `vlmeta["kind"] == "ctable"`

then the object should be recognized as a persisted `CTable`.

This keeps `urlpath` simple: it points to the table root, and `_meta` provides
the type marker and schema.

---

## Multiple tables in one TreeStore

The design must support multiple `CTable` objects in the same `TreeStore`.

That is one reason `_meta` is a good choice:

* each table root has its own `_meta`
* each table root can be introspected independently
* schema metadata is naturally scoped to one table subtree

Example shared TreeStore:

* `users/_meta`
* `users/_valid_rows`
* `users/_cols/id`
* `orders/_meta`
* `orders/_valid_rows`
* `orders/_cols/order_id`

No additional global registry is required in the first version.

---

## Column name validation

Column name validation should be explicit and should be shared between:

* in-memory `CTable`
* persistent `CTable`

Reason:

* a schema should not be valid in memory and then fail only when persisted

Recommended first-rule constraints for column names:

* must be a non-empty string
* must not contain `/`
* must not start with `_`
* must not collide with reserved internal names

Reserved internal names for the table root layout:

* `_meta`
* `_valid_rows`
* `_cols`

This validation should happen during schema compilation, not only during
persistent-table creation.

---

## Column order

Column order should be preserved explicitly in the schema JSON.

The source of truth is:

* the order of `schema["columns"]`

Do not rely on:

* dict ordering as a persistence contract
* lexical ordering of `_cols/<name>`
* TreeStore iteration order

On load:

* reconstruct `table.col_names` from the schema list order
* rebuild any name-to-column map separately

---

## Read-only mode

When `mode="r"`:

Allowed:

* opening the table
* reading rows
* reading columns
* creating non-mutating views
* `head()`, `tail()`, filtering, and other read-only operations

Disallowed:

* `append()`
* `delete()`
* `compact()`
* any operation that mutates stored arrays or metadata

These should fail immediately with a clear error.

If some existing view path currently requires mutation internally, that should be
cleaned up rather than weakening the read-only contract.

---

## Failure model

The first persistency version does not need full transactional semantics.

Be explicit in the implementation and docs:

* updates touching multiple entries are not guaranteed to be atomic
* partial writes are possible if a failure occurs mid-update

That is acceptable for the first version as long as it is not hidden.

The initial goal is a correct and understandable persistent layout, not a full
transaction layer.

---

## Internal API sketch

This is a proposed internal storage split, not a final public API requirement.

Possible internal helpers:

```python
class TableStorage:
    def open_column(self, name: str): ...
    def create_column(
        self,
        name: str,
        *,
        dtype,
        shape,
        chunks=None,
        blocks=None,
        cparams=None,
        dparams=None
    ): ...
    def open_valid_rows(self): ...
    def create_valid_rows(
        self, *, shape, chunks=None, blocks=None, cparams=None, dparams=None
    ): ...
    def load_schema(self) -> dict: ...
    def save_schema(self, schema: dict) -> None: ...
    def exists(self) -> bool: ...
    def is_read_only(self) -> bool: ...


class InMemoryTableStorage(TableStorage): ...


class TreeStoreTableStorage(TableStorage): ...
```

Then `CTable` can route based on `urlpath`:

* `urlpath is None` -> `InMemoryTableStorage`
* `urlpath is not None` -> `TreeStoreTableStorage`

This keeps persistence a backend concern instead of scattering TreeStore logic
throughout all of `CTable`.

---

## Concrete implementation sequence

### Step 1: extend constructor/open signatures

Update `src/blosc2/ctable.py` to accept:

```python
class CTable:
    def __init__(
        self,
        row_type,
        new_data=None,
        *,
        urlpath: str | None = None,
        mode: str = "a",
        expected_size: int = 1_048_576,
        compact: bool = False,
        validate: bool = True,
    ) -> None: ...
```

And add:

```python
@classmethod
def open(cls, urlpath: str, *, mode: str = "r") -> "CTable": ...
```

### Step 2: add storage backend abstraction

Create a new module:

* `src/blosc2/ctable_storage.py`

Add:

* `TableStorage`
* `InMemoryTableStorage`
* `TreeStoreTableStorage`

### Step 3: implement TreeStore layout helpers

In `TreeStoreTableStorage`, add helpers for:

* `_meta` path
* `_valid_rows` path
* `_cols/<name>` paths
* reading/writing `vlmeta["kind"]`
* reading/writing `vlmeta["version"]`
* reading/writing `vlmeta["schema"]`

### Step 4: persist schema JSON

Connect compiled schema export/import to `_meta.vlmeta["schema"]`.

The schema compiler work should provide:

```python
def schema_to_dict(schema: CompiledSchema) -> dict: ...
def schema_from_dict(data: dict) -> CompiledSchema: ...
```

### Step 5: create/open persistent arrays

Wire `CTable` initialization so that:

* create path creates `_meta`, `_valid_rows`, and `_cols/<name>`
* open path loads schema first, then opens `_valid_rows` and columns

### Step 6: enforce read-only behavior

Add an internal read-only flag so mutating methods fail early when opened with
`mode="r"`.

Methods to guard first:

* `append`
* `extend`
* `delete`
* `compact`

### Step 7: test persistency layout and round-trips

Add tests covering:

* create persistent `CTable`
* reopen persistent `CTable`
* schema JSON present in `_meta.vlmeta`
* `_valid_rows` persisted correctly
* column order preserved after reopen
* multiple tables inside one TreeStore
* read-only mode errors on mutation

---

## Proposed tests

Suggested test file:

* `tests/ctable/test_persistency.py`

Suggested test cases:

* `test_create_persistent_ctable_layout`
* `test_open_persistent_ctable`
* `test_schema_saved_in_meta_vlmeta`
* `test_valid_rows_persisted`
* `test_column_order_roundtrip`
* `test_multiple_ctables_in_same_treestore`
* `test_read_only_mode_rejects_mutation`

---

## Recommendation

The recommended persistency design is:

1. use `urlpath` to switch between in-memory and persistent `CTable`
2. store one table per TreeStore subtree
3. use:
   * `_meta`
   * `_valid_rows`
   * `_cols/<name>`
4. store schema JSON in `_meta.vlmeta["schema"]`
5. store explicit markers in `_meta.vlmeta`:
   * `"kind": "ctable"`
   * `"version": 1`
6. preserve column order in the schema JSON as an ordered `columns` list
7. keep the first version limited to append-row persistence, not full schema evolution

This gives `CTable` a clear persistent layout, keeps `open()` introspection
simple, and stays consistent with the existing columnar design.
