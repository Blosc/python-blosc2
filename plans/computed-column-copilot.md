# Computed (Virtual) Columns for CTable

## Goal

Allow a `CTable` column to be backed by a `LazyExpr` whose operands are other
columns in the same table.  Such a column stores no data — values are computed
on-the-fly when read.  The column is read-only: writes, deletes, and schema
mutations that target it raise an error.

```python
@dataclass
class Row:
    price: float = blosc2.field(blosc2.float64())
    qty: int     = blosc2.field(blosc2.int64())

t = CTable(Row, data)

# Define a virtual column
t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])

# Read it — lazy, chunk-by-chunk
t["total"][0:100]          # np.ndarray of 100 values
t["total"][:]              # full materialisation
t.where(t["total"] > 500)  # filtering works
```

---

## 1  Data Model

### 1.1  New instance attribute: `_computed_cols`

Add a dict to `CTable`:

```python
self._computed_cols: dict[str, _ComputedColumnDef] = {}
```

where `_ComputedColumnDef` is a lightweight dataclass (or plain dict) holding:

| field         | type                          | purpose |
|---------------|-------------------------------|---------|
| `expression`  | `str`                         | The LazyExpr expression string (for display / serialization) |
| `col_deps`    | `list[str]`                   | Ordered list of column names the expression depends on |
| `lazy`        | `blosc2.LazyExpr`             | The live LazyExpr object, rebuilt on demand |
| `dtype`       | `np.dtype`                    | The result dtype (inferred or user-supplied) |

The `lazy` field always references the *physical* `NDArray` objects in
`self._cols` — so when the table grows, the expression automatically covers
the new rows.

### 1.2  Column-name registries

`col_names` already lists every column the table exposes.  Computed column
names are added to `col_names` just like regular ones, so display, iteration,
and `__getitem__`/`__getattr__` find them.

A helper property distinguishes the two kinds:

```python
@property
def _stored_col_names(self) -> list[str]:
    """Column names backed by physical NDArrays (excludes computed)."""
    return [n for n in self.col_names if n not in self._computed_cols]
```

This is used internally wherever only physical columns should be visited
(append, extend, compact, grow, save, etc.).

---

## 2  Public API Changes

### 2.1  `CTable.add_computed_column()`

```python
def add_computed_column(
    self,
    name: str,
    expr: callable | str,
    *,
    dtype: np.dtype | None = None,
) -> None:
```

*   `name` — must satisfy `_validate_column_name`; must not collide with
    existing stored or computed columns.
*   `expr` — either a **callable** `(cols: dict[str, NDArray]) -> LazyExpr`
    or an **expression string** such as `"price * qty"` that is evaluated with
    column NDArrays as local variables (via `blosc2.lazyexpr()`).
*   `dtype` — optional override; inferred from the `LazyExpr.dtype` otherwise.

Steps:

1.  Validate `name`.
2.  Build the `LazyExpr`:
    *   callable → call it with `self._cols`.
    *   string → call `blosc2.lazyexpr(expr, self._cols)` (exact API TBD).
3.  Verify all operands point to NDArrays owned by this table (no dangling
    refs).  Record dependency names in `col_deps`.
4.  Store the `_ComputedColumnDef`.
5.  Append `name` to `self.col_names`.
6.  Set `self._col_widths[name]`.
7.  **Do not** touch `_schema.columns` — computed columns live outside the
    schema that drives row validation and physical storage.
8.  If persistent, persist the computed-column metadata (see §6).

### 2.2  `CTable.drop_computed_column()`

```python
def drop_computed_column(self, name: str) -> None:
```

Remove from `_computed_cols`, `col_names`, `_col_widths`.  If persistent,
remove from saved metadata.

### 2.3  `CTable.computed_columns` (property)

```python
@property
def computed_columns(self) -> dict[str, _ComputedColumnDef]:
    """Return a read-only view of the computed-column definitions."""
```

---

## 3  `Column` class changes

### 3.1  `_raw_col` property

Currently:

```python
@property
def _raw_col(self):
    return self._table._cols[self._col_name]
```

Change to:

```python
@property
def _raw_col(self):
    cc = self._table._computed_cols.get(self._col_name)
    if cc is not None:
        return cc["lazy"]
    return self._table._cols[self._col_name]
```

When `_raw_col` returns a `LazyExpr`:

*   **`dtype`** — `LazyExpr.dtype` ✓
*   **`__getitem__`** — `LazyExpr.__getitem__` returns `np.ndarray` ✓
*   **comparisons** (`__lt__`, `__gt__`, etc.) — inherited from `Operand` ✓
*   **`shape`, `ndim`, `chunks`, `blocks`** — all available on `LazyExpr` ✓

### 3.2  `is_computed` property

```python
@property
def is_computed(self) -> bool:
    return self._col_name in self._table._computed_cols
```

### 3.3  Read-only guards

The following methods must check `is_computed` and raise
`ValueError("Column '<name>' is a computed column (read-only).")`:

*   `__setitem__`
*   `assign`

### 3.4  `__iter__` / `iter_chunks` / `to_numpy`

These currently iterate chunk-metadata of `_raw_col` (an `NDArray`) via
`iterchunks_info()` + fancy indexing with `_valid_rows`.

For computed columns `_raw_col` is a `LazyExpr`, which does **not** expose
`iterchunks_info`.  Two approaches:

**Option A (recommended):** Materialise the LazyExpr chunk-by-chunk using
slice-based `__getitem__`, then apply the valid-rows mask.

```python
def iter_chunks(self, size=65536):
    if self.is_computed:
        yield from self._iter_chunks_computed(size)
        return
    # ... existing code ...

def _iter_chunks_computed(self, size):
    raw = self._raw_col           # LazyExpr
    valid = self._valid_rows
    phys_len = len(valid)
    chunk_size = valid.chunks[0]
    pending = []
    pending_n = 0

    for start in range(0, phys_len, chunk_size):
        end = min(start + chunk_size, phys_len)
        mask = valid[start:end]
        n_live = int(np.count_nonzero(mask))
        if n_live == 0:
            continue
        data = raw[start:end]     # triggers LazyExpr evaluation for this slice
        if isinstance(data, blosc2.NDArray):
            data = data[:]
        segment = data[mask] if n_live < (end - start) else data
        pending.append(segment)
        pending_n += len(segment)
        while pending_n >= size:
            combined = np.concatenate(pending)
            yield combined[:size]
            rest = combined[size:]
            pending = [rest] if len(rest) > 0 else []
            pending_n = len(rest)
    if pending:
        yield np.concatenate(pending)
```

**Option B:** Call `lazy.compute()` once, then iterate the result NDArray
normally.  Simpler, but doubles memory for large tables.  Not recommended
as default.

`to_numpy()` can delegate to `iter_chunks()` — no extra changes needed.

`__iter__` delegates to `iter_chunks(size=chunk_size)` — same.

### 3.5  Aggregate helpers (`sum`, `min`, `max`, `mean`, `std`, etc.)

These already use `iter_chunks` / `_nonnull_chunks`, so they work without
changes once `iter_chunks` handles computed columns.

### 3.6  `null_value` / `is_null` / `notnull`

Computed columns have no `SchemaSpec`, so `null_value` returns `None` and
null-related helpers return all-False / all-True masks.  This is the correct
behaviour — the expression itself decides semantics.

---

## 4  CTable internal methods — skipping computed columns

Every place that writes physical data must skip computed columns.  These
methods iterate `self._cols` and/or `self.col_names`.  Each should use
`self._stored_col_names` (or guard with `if name in self._computed_cols:
continue`):

| Method                  | What to change |
|-------------------------|----------------|
| `_init_columns`         | No change (called at creation, before computed cols exist) |
| `_grow`                 | Iterate `self._cols.values()` only (already correct — computed cols are not in `_cols`) |
| `append`                | Skip computed cols when writing per-column values; do **not** require them in input rows |
| `extend`                | Same as `append` |
| `_normalize_row_input`  | Return only stored-column keys; ignore computed-column names |
| `_coerce_row_to_storage`| Iterate `self._schema.columns` (stored only) — no change needed |
| `compact`               | Iterate `self._cols.items()` only (already correct) |
| `sort_by`               | Use `self._cols.items()` for physical reorder (correct).  Allow sorting **by** a computed column by materialising it first into a temporary array for `np.lexsort` |
| `save`                  | Write only stored columns; persist computed-column metadata separately (see §6) |
| `load` / `open`         | Reconstruct `_computed_cols` from persisted metadata |
| `_empty_copy`           | Copy `_computed_cols` and rebuild LazyExprs for the new `_cols` |
| `_make_view`            | Share parent's `_computed_cols` (same NDArray operands) |
| `select`                | If a computed column is selected, include it in the view's `_computed_cols` |
| `cbytes` / `nbytes`     | Exclude computed columns — they use no storage |
| `__str__` display       | Include computed columns; fetch values via `LazyExpr.__getitem__` |
| `info_items`            | Show computed columns with a `(computed)` label |
| `describe`              | Include computed columns using `Column` aggregates |
| `cov`                   | Materialise computed columns via `to_numpy()` |
| `to_arrow`              | Materialise computed columns via `Column.to_numpy()` |
| `to_csv`                | Same |
| `from_arrow` / `from_csv` | No change — these create fresh tables without computed cols |
| `schema_dict`           | Exclude computed columns (they are not part of the row schema) |

### 4.1  `__getitem__` / `__getattr__`

Currently:

```python
def __getitem__(self, s: str):
    if s in self._cols:
        return Column(self, s)
    return None
```

Change to:

```python
def __getitem__(self, s: str):
    if s in self._cols or s in self._computed_cols:
        return Column(self, s)
    return None
```

Same for `__getattr__`.

### 4.2  `add_column` / `drop_column` / `rename_column`

*   `add_column`: disallow names that collide with `_computed_cols`.
*   `drop_column`: refuse if the column is a dependency of any computed column
    (raise `ValueError` listing the computed columns that depend on it).
*   `rename_column`: if a stored column is renamed and any computed column
    depends on it, rebuild that computed column's `LazyExpr` with the renamed
    operand.  Alternatively, refuse the rename and tell the user to drop the
    computed column first.  (The simpler "refuse" approach is fine for v1.)

### 4.3  `delete` (row deletion)

No change needed — `delete` only touches `_valid_rows`.  Computed columns
inherit the mask naturally.

---

## 5  `_Row` and `_RowIndexer`

`_Row.__getitem__` accesses `self._table._cols[col_name]`.  For computed
columns, this should go through `_computed_cols[col_name]["lazy"]` instead:

```python
def __getitem__(self, col_name: str):
    if self._real_pos is None:
        self._get_real_pos()
    cc = self._table._computed_cols.get(col_name)
    if cc is not None:
        return cc["lazy"][self._real_pos]
    return self._table._cols[col_name][self._real_pos]
```

---

## 6  Persistence

### 6.1  Metadata format

Store computed columns alongside the regular schema in the table metadata.
Add a new top-level key `"computed_columns"` to the schema dict:

```json
{
  "version": 1,
  "row_cls": "Row",
  "columns": [ ... ],
  "computed_columns": [
    {
      "name": "total",
      "expression": "(o0 * o1)",
      "col_deps": ["price", "qty"],
      "dtype": "float64"
    }
  ]
}
```

`expression` is the `LazyExpr.expression` string with operand placeholders
(`o0`, `o1`, …).  `col_deps` maps each placeholder back to a column name in
order (`o0` → `col_deps[0]`).

### 6.2  Save path

In `save()` and `schema_to_dict()` (or a parallel helper), serialize
`_computed_cols` into the above format.

### 6.3  Load / open path

In `load()`, `open()`, and the `__init__` existing-table branch, after
opening stored columns, check for `"computed_columns"` in the schema dict.
For each entry:

1.  Map `col_deps` names → `self._cols[name]` NDArrays.
2.  Build the operands dict: `{"o0": cols[dep0], "o1": cols[dep1], ...}`.
3.  Construct a `LazyExpr` using `blosc2.lazyexpr(expression, operands)` (or
    use `LazyExpr._new_expr` directly).
4.  Register in `_computed_cols`.

### 6.4  `FileTableStorage` / `InMemoryTableStorage`

No new storage methods are needed — computed columns are metadata-only.  The
existing `save_schema` / `load_schema` methods carry the extra key.

---

## 7  Indexing (CTableIndex)

Computed columns **cannot** be indexed (they have no persistent NDArray to
attach sidecars to).  `create_index` must reject computed column names:

```python
if col_name in self._computed_cols:
    raise ValueError(
        f"Cannot create an index on computed column {col_name!r}."
    )
```

However, `where()` filtering by a computed column expression still works via
the normal full-scan path (the `LazyExpr` is evaluated chunk by chunk).

---

## 8  `CTable.__str__` display

The display loop fetches column values via physical positions:

```python
col_data = {n: self._cols[n][positions] for n in self.col_names}
```

Change to a helper that routes through `_computed_cols` for virtual columns:

```python
def _fetch_col_slice(self, name, positions):
    cc = self._computed_cols.get(name)
    if cc is not None:
        return cc["lazy"][positions]
    return self._cols[name][positions]

col_data = {n: self._fetch_col_slice(n, positions) for n in self.col_names}
```

The dtype row also needs the same routing:

```python
def _col_dtype(self, name):
    cc = self._computed_cols.get(name)
    if cc is not None:
        return cc["dtype"]
    return self._cols[name].dtype
```

---

## 9  Sorting by a computed column

`sort_by` currently reads physical column data via `self._cols[name][live_pos]`.
For a computed column, materialise the required slice first:

```python
cc = self._computed_cols.get(name)
if cc is not None:
    raw = cc["lazy"][live_pos]
else:
    raw = self._cols[name][live_pos]
```

The result (`raw`) is a numpy array in both cases, so `np.lexsort` works
identically.

---

## 10  Expression-string API (convenience)

For a terser UX, support an expression string that references column names
directly:

```python
t.add_computed_column("total", expr="price * qty")
```

Implementation: build the operands dict by scanning the expression for column
names, then call `blosc2.lazyexpr(expr, operands)`.

This is secondary and can be deferred to a follow-up — the callable form is
sufficient for v1.

---

## 11  File-by-file change summary

| File | Changes |
|------|---------|
| `ctable.py` | `_ComputedColumnDef` dataclass; `_computed_cols` init in `__init__`, `open`, `load`, `_make_view`, `select`, `_empty_copy`; `add_computed_column`, `drop_computed_column`, `computed_columns` property, `_stored_col_names` property; guards in `append`, `extend`, `__getitem__`, `__getattr__`, `__str__`, `info_items`, `save`, `add_column`, `drop_column`, `rename_column`, `create_index`, `cbytes`, `nbytes`, `sort_by`, `describe`, `cov`, `to_arrow`, `to_csv`, `_fetch_col_slice`, `_col_dtype`; `Column._raw_col`, `Column.is_computed`, `Column.__setitem__`, `Column.assign`, `Column.iter_chunks`, `_Row.__getitem__` |
| `schema_compiler.py` | `schema_to_dict` — emit `"computed_columns"` when present; `schema_from_dict` — parse and return computed-column defs (as inert metadata; actual LazyExpr construction is done by the caller in `ctable.py`) |
| `tests/ctable/test_ctable_computed_cols.py` | New test file (see §12) |

---

## 12  Test plan

New file `tests/ctable/test_ctable_computed_cols.py`:

| Test | What it verifies |
|------|------------------|
| `test_add_computed_column_basic` | Create table, add computed column, read values match expected |
| `test_computed_column_dtype` | Inferred dtype correct; explicit dtype override works |
| `test_computed_column_read_slice` | Slice access `t["total"][2:5]` returns correct values |
| `test_computed_column_read_scalar` | Scalar access `t["total"][0]` returns a scalar |
| `test_computed_column_write_raises` | `t["total"][0] = 1` raises `ValueError` |
| `test_computed_column_assign_raises` | `t["total"].assign(...)` raises `ValueError` |
| `test_computed_column_in_where` | `t.where(t["total"] > 500)` returns correct view |
| `test_computed_column_compose` | Expression over computed + stored column works |
| `test_computed_column_after_append` | Append rows, computed column reflects new data |
| `test_computed_column_after_delete` | Delete rows, computed column respects valid_rows mask |
| `test_computed_column_iter` | `list(t["total"])` matches expected |
| `test_computed_column_iter_chunks` | `iter_chunks` yields correct chunks |
| `test_computed_column_to_numpy` | `t["total"].to_numpy()` returns full array |
| `test_computed_column_aggregates` | `sum`, `min`, `max`, `mean`, `std` work |
| `test_computed_column_display` | `str(t)` does not crash, includes computed column |
| `test_computed_column_info` | `t.info` includes computed column with `(computed)` label |
| `test_computed_column_describe` | `t.describe()` includes computed column stats |
| `test_computed_column_drop` | `drop_computed_column` removes it from all registries |
| `test_computed_column_name_collision` | Adding a computed col with existing name raises |
| `test_drop_dependency_raises` | Dropping a stored column used by a computed col raises |
| `test_computed_column_index_raises` | `create_index` on computed column raises |
| `test_computed_column_select` | `t.select(["price", "total"])` includes computed column |
| `test_computed_column_view` | Views inherit computed columns |
| `test_computed_column_sort_by` | Sorting by a computed column works |
| `test_computed_column_to_arrow` | `to_arrow` materialises computed columns |
| `test_computed_column_to_csv` | `to_csv` materialises computed columns |
| `test_computed_column_save_load` | Save, load, verify computed columns are restored |
| `test_computed_column_open` | Persistent table: open, verify computed columns |
| `test_computed_column_cov` | `cov()` includes computed column |
| `test_computed_column_exclude_nbytes` | `nbytes`/`cbytes` exclude computed columns |
| `test_computed_column_expr_string` | (v2) string-based expression form |
| `test_computed_column_append_skip` | `append` does not require computed-column value in input |
| `test_computed_column_extend_skip` | `extend` does not require computed-column value in input |
| `test_computed_column_compact` | `compact()` does not touch computed columns |

---

## 13  Implementation order

1.  **`_ComputedColumnDef`** dataclass + `_computed_cols` init in `__init__`.
2.  **`add_computed_column`** — callable form only.
3.  **`Column._raw_col`** routing + `is_computed` property.
4.  **Read-only guards** in `Column.__setitem__`, `Column.assign`.
5.  **`Column.iter_chunks`** — `_iter_chunks_computed` for LazyExpr.
6.  **`__getitem__` / `__getattr__`** on CTable — check `_computed_cols`.
7.  **`_Row.__getitem__`** — route through computed cols.
8.  **Skip computed cols** in `append`, `extend`, `_normalize_row_input`.
9.  **`__str__`** display — `_fetch_col_slice`, `_col_dtype`.
10. **`info_items`** — show `(computed)` label.
11. **`drop_computed_column`** + dependency guard in `drop_column`.
12. **`create_index`** guard.
13. **`sort_by`** — materialise computed column for lexsort.
14. **`cbytes` / `nbytes`** — exclude computed cols.
15. **`_make_view`, `select`, `_empty_copy`** — propagate `_computed_cols`.
16. **`to_arrow`, `to_csv`, `describe`, `cov`** — materialise.
17. **Persistence** — `save`, `load`, `open`, schema dict.
18. **`computed_columns`** property.
19. **Tests** — `test_ctable_computed_cols.py`.
20. **Expression-string API** (stretch goal).

---

## 14  Risks and open questions

1.  **Shape mismatch after `_grow`**: `_grow` doubles the physical NDArray
    capacity.  `LazyExpr.shape` is derived from its operands, so after a
    `_grow` the expression's shape automatically reflects the new physical
    size.  The valid-rows mask ensures only live rows are exposed.  **No
    risk** — but confirm with a test (`test_computed_column_after_append`).

2.  **Nested computed columns**: Can a computed column depend on another
    computed column?  The `LazyExpr` machinery supports nesting
    (`LazyExpr` as operand inside another `LazyExpr`).  However, for v1
    it is simpler to require all dependencies to be stored columns.  Add
    validation in `add_computed_column` and revisit if users request it.

3.  **Thread safety**: `LazyExpr` evaluation is not designed for concurrent
    access.  This is an existing limitation and not worsened by computed
    columns.

4.  **Expression-string parsing**: Robustly mapping column names in an
    expression string to operand placeholders requires either restricting
    column names to valid Python identifiers (already enforced by
    `_validate_column_name`) or using a custom parser.  Defer to v2.

5.  **Performance of `iter_chunks` for computed columns**: Evaluating the
    `LazyExpr` slice-by-slice inside `_iter_chunks_computed` incurs per-slice
    overhead.  For bulk reads, `to_numpy()` could short-circuit via a single
    `lazy.compute()` call.  Benchmark and optimise after the initial
    implementation.
