# CTable ndarray Column Type

Add support for **fixed-size multi-dimensional columns** to CTable, where each row
stores an array (1-D, 2-D, …) of uniform dtype instead of a scalar value.

---

## User story

```python
import numpy as np
import blosc2 as b2
from dataclasses import dataclass


@dataclass
class Row:
    id: int
    embedding: np.ndarray = b2.field(b2.ndarray(shape=(768,), dtype=b2.float32()))
    bbox: np.ndarray = b2.field(
        b2.ndarray(shape=(4,), dtype=b2.float64()),
        default_factory=lambda: np.full(4, np.nan, dtype=np.float64),
    )


t = b2.CTable(Row)
t.extend(
    {"id": np.arange(1000), "embedding": np.random.randn(1000, 768).astype(np.float32)}
)

# Filter rows where the max element of the embedding exceeds 0.5
view = t.where(t["embedding"].max() > 0.5)

# L2-norm per row — then filter
view = t.where(t["embedding"].norm() > 5.0)

# Slice into the inner array — first element of each embedding
first_elem = t["embedding"][:, 0]  # 1-D NDArray, can be used in t.where()

# Row access returns the full array
print(t[0].embedding)  # ndarray of shape (768,)
```

---

## 1. Schema — `NDArraySpec` class

**File:** `src/blosc2/schema.py`

```python
class NDArraySpec(SchemaSpec):
    """Fixed-size array column where each row stores a multi-dimensional array.

    Parameters
    ----------
    shape: tuple[int, ...]
        Per-row array shape, e.g. ``(3,)`` for a 3-element vector,
        ``(4, 4)`` for a 4×4 matrix.  Must be non-empty (scalar columns
        use ``int32``, ``float64``, etc.).
    dtype_spec: SchemaSpec
        The element type, e.g. ``float32()``, ``int32(ge=0)``, ``bool()``.
    nullable: bool, default False
        Whether the whole row-array can be ``None``.
    null_value: array-like, optional
        Explicit null sentinel pattern.  When omitted, the existing
        per-dtype sentinel convention is broadcast to ``shape`` (see §1.1).
    """

    python_type = np.ndarray  # each row value is an ndarray (or None)
    dtype: np.dtype  # from dtype_spec.dtype
    dtype_spec: SchemaSpec  # inner spec — constraints, serialization
    shape: tuple[int, ...]
    nullable: bool
    null_value: np.ndarray | None
```

**Factory function:**

```python
def ndarray(
    shape: tuple[int, ...],
    dtype: SchemaSpec,
    *,
    nullable: bool = False,
    null_value: np.ndarray | list | None = None,
) -> NDArraySpec: ...
```

**Dataclass usage:**

```python
@dataclass
class Row:
    v: np.ndarray = b2.field(b2.ndarray(shape=(3,), dtype=b2.float32()))
    # For nullable:
    opt: np.ndarray = b2.field(b2.ndarray(shape=(2,), dtype=b2.int32(), nullable=True))
    # Default with factory:
    mat: np.ndarray = b2.field(
        b2.ndarray(shape=(4, 4), dtype=b2.float64()),
        default_factory=lambda: np.eye(4, dtype=np.float64),
    )
```

Users write the annotation as `np.ndarray` (matches `NDArraySpec.python_type`).
The `default_factory` must produce an ndarray of the right shape + dtype.

**Important**: `np.ndarray` must **not** be added to `_ANNOTATION_TO_SPEC` —
plain `np.ndarray` annotations without `b2.field(b2.ndarray(...))` are
unsupported because the shape and dtype cannot be inferred.  A clear error
message must be raised.

**Naming note**: `b2.ndarray` (spec factory) vs `blosc2.NDArray` (array class)
differ by case only.  Document clearly: `b2.ndarray(...)` declares a schema
spec; `blosc2.NDArray` is the compressed array class.

### 1.1 Nullability

Uses the **existing sentinel convention** from scalar columns, broadcast to
`shape`.  The null sentinel for `ndarray(shape=(3,), dtype=int32())` is
`[int32.min, int32.min, int32.min]` — the scalar sentinel repeated in every
element position.

| Inner dtype | Default sentinel | Check |
|---|---|---|
| `float32` / `float64` | All-`NaN` | `np.isnan(arr).all(axis=inner)` |
| `int8` … `int64` | All-`min` (or all-`max` per policy) | `(arr == nv).all(axis=inner)` |
| `uint8` … `uint64` | All-`max` (or all-`min` per policy) | Same |
| `bool` | `[255, 255, …]` | Same |
| `string` / `bytes` | All-`"__BLOSC2_NULL__"` | Same |

`_null_mask_for` adaptation (in `Column`):

```python
if is_ndarray_spec:
    inner = tuple(range(1, arr.ndim))  # axes beyond row dim
    if np.issubdtype(self.dtype, np.floating):
        return np.isnan(arr).all(axis=inner)
    return (arr == null_value).all(axis=inner)
```

`_policy_null_value_for_spec` gets an `NDArraySpec` branch that calls itself
recursively on the inner spec and broadcasts to `shape`.

### 1.2 Serialization

```python
# to_metadata_dict → {"kind": "ndarray", "shape": [3], "dtype": {"kind": "float32", ...}, ...}
# Deserialization via spec_from_metadata_dict, which needs a new "ndarray" branch.
```

`_KIND_TO_SPEC["ndarray"]` must be added in `schema_compiler.py` (see §2).

### 1.3 `__init__.py` export

Add `ndarray` and `NDArraySpec` to the `from .schema import (...)` block
in `src/blosc2/__init__.py`.

### 1.4 Chunks / blocks interaction

`compute_chunks_blocks((expected_size, *per_row_shape))` already handles
multi-dimensional shapes.  For large per-row arrays (e.g. `(224, 224, 3)`)
users should provide per-column `chunks=` / `blocks=` hints via `b2.field()`.

---

## 2. Schema Compiler

**File:** `src/blosc2/schema_compiler.py`

- `CompiledColumn` gains `per_row_shape: tuple[int, ...]` (empty tuple = scalar).
- `compile_schema` populates it from `NDArraySpec.shape`.
- `validate_annotation_matches_spec` accepts `np.ndarray` (and `np.ndarray | None`
  for nullable) for `NDArraySpec` columns.  **`list[T]` is not accepted** —
  the annotation must be `np.ndarray` to signal fixed-size storage.
- `_policy_null_value_for_spec` and `_validate_null_value_for_spec` get
  `NDArraySpec` branches.
- `spec_from_metadata_dict` gets an `"ndarray"` branch.
- `_KIND_TO_SPEC` gets `"ndarray": NDArraySpec` (or a factory callable).

---

## 3. Column NDArray storage shape

Every place the physical NDArray backing a column is created must use
`(capacity, *per_row_shape)` instead of `(capacity,)`:

| Location | Current | New |
|---|---|---|
| `_init_columns` | `shape=(expected_size,)` | `shape=(expected_size, *per_row_shape)` |
| `_grow` | `resize((c*2,))` | `resize((c*2, *self._raw_col.shape[1:]))` |
| `add_column` | `shape=(capacity,)` | `shape=(capacity, *per_row_shape)` |
| `_create_empty_stored_column` | same | same |
| `materialize_computed_column` | same | same |
| `_add_physical_column_from_spec` | same | same |

**No changes needed** for `compact()` or `_sort_by_inplace` /
`_sorted_copy_from_positions`: both use `arr[pos_array]` and
`arr[start:end] = arr[other_pos]` patterns that work automatically for
multi-dimensional NDArrays.

---

## 4. Column class

**File:** `src/blosc2/ctable.py`

### 4.1 Metadata properties

```python
@property
def shape(self) -> tuple[int, ...]:
    return (len(self), *self._per_row_shape)


@property
def ndim(self) -> int:
    return 1 + len(self._per_row_shape)


@property
def size(self) -> int:
    return len(self) * math.prod(self._per_row_shape)  # math.prod(()) == 1 for scalar
```

`self._per_row_shape` is resolved from
`self._table._schema.columns_by_name[name].per_row_shape`.

### 4.2 Per-row aggregation methods

`max`, `min`, `sum`, `mean`, `std`, `var`, `any`, `all`, and **`norm`**
operate on ndarray columns by reducing **all inner axes** by default,
returning a **1-D NDArray** (one scalar per row):

```python
def max(self, axis=None, *, where=None):
    if self._is_ndarray_column():
        if axis is None:
            axis = tuple(range(1, self._raw_col.ndim))
        result = self._raw_col.max(axis=axis)
        # ensure result is NDArray for LazyExpr interop (see §9 risk #2)
        if not isinstance(result, blosc2.NDArray):
            result = blosc2.asarray(result)
        return result
    # existing scalar path (unchanged)
    ...


def norm(self, ord=2, *, where=None):
    """Per-row L-norm.  Returns a 1-D NDArray."""
    ...
```

The returned 1-D NDArray supports comparison operators (`>`, `<`, `==`, …)
which create a `LazyExpr` usable in `t.where(...)`.

An explicit `axis=` parameter allows partial reductions:

```python
t["mat"].max(axis=(-1,))  # reduce last dim only, keep others
t["mat"].sum(axis=None)  # global scalar across all rows+axes (explicit)
```

For **scalar columns** the current behavior is preserved.  Note that calling
`t["scalar_col"].max()` already returns a Python scalar today; there is no
risk of breakage.

**Implementation note:** the ndarray path bypasses `_lazy_aggregate_fastpath`
entirely and delegates to `self._raw_col`.  The `where=` parameter is not
supported in the ndarray branch for v1 (null masking is applied downstream in
`_null_mask_for`).

### 4.3 Element-slice access via `__getitem__`

`Column.__getitem__` accepts **tuple keys** for ndarray columns.  The first
element is the row selector (existing `int`/`slice`/`list`/`ndarray`/`bool`
logic), the remaining elements slice into the inner axes:

```text
t["emb"][:, 0]         → 1-D numpy array (first element per row)
t["emb"][:, :2]        → 2-D numpy array (first two elements per row)
t["emb"][:5, [0, 2]]   → 2-D numpy array (rows 0-4, inner positions 0 and 2)
```

For `(slice(None), *inner)` the inner slice is applied directly to
`self._raw_col`, which materializes eagerly as a numpy array via
`NDArray.__getitem__`.  For other row selectors, physical indices are resolved
first, then inner slicing is applied.

**Note:** The result is a numpy array, not an NDArray.  To use it in
`t.where(...)` as a predicate, the comparison (`> 0.5`) produces a boolean
numpy array which `where()` accepts via `blosc2.asarray()`.

### 4.4 Comparison operators — blocked for ndarray

`__gt__`, `__lt__`, `__eq__`, `__ne__`, `__ge__`, `__le__` raise a clear
`TypeError` for ndarray columns with an actionable message:

```python
t["emb"] > 0.5
# → TypeError: Cannot compare ndarray column 'emb' (shape=(768,)) with a
#   scalar directly.  Use an aggregation: t["emb"].max() > 0.5, or
#   an element slice: t["emb"][:, 0] > 0.5
```

### 4.5 `__setitem__`, `append`, `extend`

- **`append`**: `_coerce_row_to_storage` skips `.item()` for ndarray columns.
  Shape and dtype validation must be added:
  ```python
  arr = np.asarray(val, dtype=col.dtype)
  if arr.shape != col.per_row_shape:
      raise ValueError(
          f"Expected shape {col.per_row_shape} for column {col.name!r}, got {arr.shape}"
      )
  result[col.name] = arr
  ```
- **`extend`**: `np.ascontiguousarray(raw_columns[name], dtype=target_dtype)`
  preserves inner dimensions.  Explicit shape check: column batch must be
  `(n_rows, *per_row_shape)`.  A wrong shape gives a clear error early.
- **`__setitem__`**: assigns arrays at row positions.

### 4.6 Row access `t[i]`

`_physical_row_value` fetches `self._cols[col_name][pos]`, which for ndarray
columns returns the per-row array (e.g. shape `(768,)`) as a numpy array.
`_normalize_scalar_value` passes through non-0d arrays unmodified, so
`t[5].embedding` returns the correct ndarray.

**Fix needed (v1):** the computed-column branch of `_physical_row_value`
currently does `.ravel()[0]` (assumes scalars).  Since computed ndarray columns
are **deferred**, this fix is only needed once §D1 lands; mark it with a
`# TODO ndarray-computed` comment for now.

### 4.7 Display

- `Column.__repr__`: for narrow shapes (`len(per_row_shape) == 1` and
  `shape[0] <= 8`) show values inline: `[0.1, 0.2, …]`.  For wider arrays
  show only shape: `ndarray(768,)`.
- `CTable.__str__` header: show dtype and shape info (e.g. `float32[768]`)
  rather than individual values to avoid blowing up horizontal space.
- `_fetch_col_at_positions`: returns 2-D for ndarray columns; display
  formatter adapts by showing shape.

### 4.8 `_where_expression_operands`

This method currently includes all non-list/non-varlen/non-dictionary columns
as operands for string lazy expressions.  For ndarray columns it **must skip**
the column:

```python
if col is not None and not (
    self._is_list_column(col)
    or self._is_varlen_scalar_column(col)
    or self._is_dictionary_column(col)
    or self._is_ndarray_column(col)  # ← add this
):
    operands[name] = arr
```

Without this guard, a string expression like `"embedding > 0"` would pass a
2-D NDArray to `numexpr`, which does not handle multi-dimensional operands
in the expected row-wise manner.

### 4.9 Iteration

`__iter__` yields namedtuple rows with ndarray column values (full per-row
array).  Chunk decompression follows the normal NDArray access pattern — the
same chunk is reused for consecutive positions within it.

---

## 5. Interop

### 5.1 Arrow (`to_arrow` / `from_arrow`)

Arrow's `FixedSizeList` maps naturally for **1-D** shapes.  For **multi-dimensional**
shapes, the pragmatic approach is a **flat** `FixedSizeList` plus a `blosc2:ndarray_shape`
field-level metadata entry, rather than nested FixedSizeList (which has
limited support in the Arrow ecosystem):

```
ndarray(shape=(768,), dtype=float32())
    ⇄  pa.list_(pa.float32(), 768)
       + field metadata {"blosc2:ndarray_shape": "[768]"}

ndarray(shape=(2, 2), dtype=int64())
    ⇄  pa.list_(pa.int64(), 4)          # list_size = math.prod([2,2])
       + field metadata {"blosc2:ndarray_shape": "[2, 2]"}
```

`to_arrow` builds a flat `FixedSizeListArray` (C-contiguous reshape of the
physical values buffer) and embeds the original shape in field metadata.
`from_arrow` detects `FixedSizeList` fields with `blosc2:ndarray_shape` and
constructs `NDArraySpec`.  `FixedSizeList` fields without that metadata key are
treated as 1-D ndarray columns (shape = `(list_size,)`).

**New required code:** `_pa_type_from_spec` needs an `NDArraySpec` branch;
`_compiled_columns_from_arrow` needs to handle `FixedSizeList` and reconstruct
the spec.

Nullability: Arrow's validity bitmap maps to the sentinel pattern — null rows
are written as the sentinel pattern, and on import, the presence of a null
bitmap with `nullable=False` raises an import error.

### 5.2 Parquet (`to_parquet` / `from_parquet`)

Modern Parquet (via pyarrow) **does** write Arrow `FixedSizeList` arrays to
Parquet as a LIST with fixed-size metadata and reads them back correctly.
This means the Arrow round-trip (§5.1) is also a valid Parquet round-trip:
write via `to_arrow()` → `to_parquet()`, read via `from_parquet()` →
`from_arrow()`.

The earlier "flatten to N separate columns" strategy is **not recommended**:
for a `shape=(768,)` embedding, that would produce 768 Parquet columns,
which is impractical in all common Parquet tooling.

### 5.3 CSV (`to_csv` / `from_csv`)

- **`to_csv`**: ndarray cells are serialized as space-separated values
  (e.g. `"0.1 0.2 0.3"`).  Multi-dimensional arrays flatten in C order.
- **`from_csv`**: cannot infer ndarray columns — users must provide an
  explicit `row_type`.

### 5.4 DataFrame / NumPy

- **`to_dataframe`**: ndarray columns become DataFrame columns of dtype
  `object` (each cell is a numpy array).  Note: pandas has no native dtype
  for fixed-size arrays; `object` is the only option.
- **`from_dataframe`**: `object`-dtype columns of numpy arrays are **not**
  auto-detected as ndarray columns.  Users must provide an explicit `row_type`
  with `NDArraySpec` columns, same as for `from_csv`.  Auto-detection is
  fragile (requires inspecting the first row, may fail on empty DataFrames).
- **`__array__`**: the structured numpy array dtype uses `object` for
  ndarray columns (numpy structured arrays cannot nest fixed-size arrays natively).

---

## 6. Non-supported operations (v1)

These raise a clear `NotImplementedError` or `TypeError` with an actionable
message:

| Operation | Reason | Alternative |
|---|---|---|
| `t["emb"] > 0.5` | Array vs scalar comparison undefined | `t["emb"].max() > 0.5` |
| `t.sort_by("emb")` | No total order on arrays | Add scalar column `t.add_column("emb_max", t["emb"].max()[:])`, then `t.sort_by("emb_max")` |
| `t.groupby("emb")` | Grouping by array values undefined | Group by a scalar column |
| `t.describe()` on ndarray cols | No meaningful scalar stats | Shows `(stats not available for ndarray columns)` — same style as list columns |
| `t.cov()` / `t.corr()` | Columns must be scalar | Add scalar aggregation columns first |
| `create_index("emb")` | Index engine requires 1-D arrays | Add scalar aggregation column, then index it |
| `t["emb"] == t["emb2"]` | Element-wise equality → 2-D result, not a 1-D row mask | `(t["emb"] - t["emb2"]).norm() < eps` once element-wise ops are added; for v1, compare aggregations |

**Guards needed in code:**
- `_normalise_sort_keys`: add ndarray column check (the function currently
  allows any column with a dtype — without a guard, `_build_lex_keys` would
  pass a 2-D array to `np.lexsort`, which silently fails or crashes).
- `describe`: add ndarray branch showing "stats not available" (otherwise the
  numeric path tries `col.min()` / `col.max()` and returns arrays instead
  of scalars, breaking the display logic).

---

## 7. Computed columns + indexing (deferred — requires more infrastructure)

### 7.1 Problem

The natural workflow is:

```python
t.add_computed_column("emb_max", lambda cols: cols["embedding"].max(axis=-1))
t.create_index("emb_max")
t.where(t["emb_max"] > 0.5)  # index hit
```

But this can't work today because:
1. `cols["embedding"].max(axis=-1)` **materializes** the result (returns a
   numpy array or NDArray, not a `LazyExpr`).  `add_computed_column` requires
   a `LazyExpr` from the callable.
2. The lazy expression engine (`numexpr`/`lazyexpr`) has no **lazy reduction
   primitive** — reductions in `LazyExpr.max(axis=...)` eagerly call
   `compute(_reduce_args=...)` via `reduce_slices`.
3. Even with a LazyExpr, the query planner (`_find_indexed_columns`) matches
   operands by object identity — a computed column's operands are the *source*
   NDArrays, not the computed column's projected 1-D array.

### 7.2 Required infrastructure

1. **Lazy reduction expression** — a `LazyExpr` that carries a reduction
   descriptor (`op`, `axis`) without immediately calling `compute`.

2. **Computed column from lazy reduction** — the callable form returns
   the deferred expression (not the materialized result).

3. **Materialize-on-index** — `create_index` on a computed column:
   materializes it into `_cols`, creates a 1-D index, registers as
   `_materialized_cols` for auto-fill on future writes.

4. **Staleness and lazy rebuild** — `append` and `extend` mark the index
   `stale`; `_try_index_where` detects the flag, rebuilds inline, then uses
   the fresh index.

### 7.2.1 Why `delete` doesn't need stale-marking

When rows are deleted only `_valid_rows` changes; physical positions never
move.  The caller (`_try_index_where`) always intersects index results with
`_valid_rows`:

```
index returns  →  [pos_3, pos_7]
valid_rows     →  [T, T, F, T, T, T, F, T]
final result   →  [pos_3]          (pos_7 deleted → excluded)
```

A stale index can never "return deleted rows".  Staleness only matters when
**new** rows arrive (`append` / `extend`).

### 7.3 Fallback summary

Until §7.2 is built, users have two practical options:

| Option | API | Always in sync? | Performance |
|---|---|---|---|
| **Lazy scan** | `t["emb"].max() > 0.5` | ✅ yes | Chunked eval via `reduce_slices` — good for most workloads |
| **Manual materialize** | `t.add_column("emb_max", ...)` → `t.create_index("emb_max")` | ❌ user-managed | Index-accelerated |

The lazy scan is the recommended v1 path.

---

## 8. Implementation phases

### Phase 1 — Core (schema, storage, Column basics)

| # | Task | Files | Complexity |
|---|---|---|---|
| 1.1 | `NDArraySpec` class + `ndarray()` factory + `to_metadata_dict` / `from_metadata_dict` | `schema.py` | **Low** ~60 lines |
| 1.2 | `per_row_shape` in `CompiledColumn` / `compile_schema` | `schema_compiler.py` | **Low** ~40 lines |
| 1.3 | `_KIND_TO_SPEC["ndarray"]` + `spec_from_metadata_dict` branch | `schema_compiler.py` | **Low** ~15 lines |
| 1.4 | Nullability: `_policy_null_value_for_spec`, `_validate_null_value_for_spec`, `_null_mask_for` | `schema_compiler.py`, `ctable.py` | **Low** ~50 lines |
| 1.5 | Storage shape: `_init_columns`, `_grow`, `add_column`, etc. | `ctable.py` | **Low** ~30 lines (many 1-liners) |
| 1.6 | `Column.shape`, `ndim`, `size`, `_per_row_shape` helper | `ctable.py` | **Low** ~25 lines |
| 1.7 | `__init__.py` export of `ndarray` + `NDArraySpec` | `__init__.py` | **Low** ~3 lines |
| 1.8 | Tests for phase 1 | `tests/ctable/` | **Medium** ~150 lines |

### Phase 2 — Data path (append, extend, access)

| # | Task | Files | Complexity |
|---|---|---|---|
| 2.1 | `_coerce_row_to_storage`: skip `.item()` + shape validation | `ctable.py` | **Low** ~20 lines |
| 2.2 | `extend`: preserve inner dims + shape validation | `ctable.py` | **Medium** ~70 lines — existing scalar/list/dict/varlen branches, each needs ndarray handling |
| 2.3 | `__setitem__`: assign arrays at row positions | `ctable.py` | **Low** ~20 lines |
| 2.4 | Row access `_physical_row_value`: add `# TODO ndarray-computed` guard | `ctable.py` | **Low** ~5 lines |
| 2.5 | `_where_expression_operands`: skip ndarray columns | `ctable.py` | **Low** ~3 lines |
| 2.6 | Tests for phase 2 | `tests/ctable/` | **Medium** ~200 lines |

### Phase 3 — Query path (aggregation, slicing, display)

| # | Task | Files | Complexity |
|---|---|---|---|
| 3.1 | Per-row aggregation methods (`max`, `min`, `sum`, `mean`, `std`, `var`, `any`, `all`, `norm`) | `ctable.py` | **Medium** ~140 lines — ndarray branch + `blosc2.asarray()` wrap |
| 3.2 | `__getitem__` tuple-key support for inner-axis slicing | `ctable.py` | **Medium** ~80 lines |
| 3.3 | Comparison operators: `TypeError` with actionable message | `ctable.py` | **Low** ~15 lines |
| 3.4 | Display: `__repr__`, `__str__`, `_fetch_col_at_positions` | `ctable.py` | **Low** ~50 lines |
| 3.5 | Tests for phase 3 | `tests/ctable/` | **Medium** ~200 lines |

### Phase 4 — Interop

| # | Task | Files | Complexity |
|---|---|---|---|
| 4.1 | `to_arrow` / `from_arrow` — flat `FixedSizeList` + field metadata; `_pa_type_from_spec`; `_compiled_columns_from_arrow` | `ctable.py` | **Medium-High** ~120 lines |
| 4.2 | `to_parquet` / `from_parquet` — route through Arrow; verify round-trip | `ctable.py` | **Low** ~20 lines (inherits Arrow support) |
| 4.3 | `to_csv` / `from_csv` — space-separated serialization | `ctable.py` | **Low** ~40 lines |
| 4.4 | `to_dataframe` — `object` dtype cells; `from_dataframe` — requires explicit schema | `ctable.py` | **Low** ~30 lines |
| 4.5 | `__array__` — `object` dtype for ndarray fields | `ctable.py` | **Low** ~10 lines |
| 4.6 | Tests for phase 4 | `tests/ctable/` | **Medium** ~200 lines |

### Phase 5 — Polish and guards

| # | Task | Files | Complexity |
|---|---|---|---|
| 5.1 | `_normalise_sort_keys`: ndarray guard (prevents silent crash in `_build_lex_keys`) | `ctable.py` | **Low** ~8 lines |
| 5.2 | `describe`: ndarray branch showing "stats not available" | `ctable.py` | **Low** ~10 lines |
| 5.3 | `cov`, `corr`, `groupby` guards | `ctable.py` | **Low** ~15 lines |
| 5.4 | `info()` / `.info_items` report per-row shape | `ctable.py` | **Low** ~15 lines |
| 5.5 | Docs + doctests | `core.py`, `schema.py` | **Medium** ~100 lines |
| 5.6 | Edge cases: empty tables, all-null columns, `copy`, `rename_column` | `ctable.py` | **Medium** ~80 lines |

### Deferred (not in v1)

| # | Task | Reason |
|---|---|---|
| D1 | Computed column from ndarray aggregation | Requires lazy-reduction `LazyExpr` primitive |
| D2 | `create_index` on computed ndarray aggregation | Depends on D1 + materialize-on-index + stale tracking |
| D3 | `_physical_row_value` fix for computed ndarray columns | Not needed until D1 lands |
| D4 | Element-wise ops between ndarray columns (`+`, `-`, `*`) | Nice to have; needs element-wise LazyExpr on 2-D NDArray |

---

## 9. Complexity summary

| Phase | Code lines | Risk |
|---|---|---|
| 1 — Core | ~373 | **Very low** — new class + mechanical one-liner changes |
| 2 — Data path | ~318 | **Medium** — `extend` branches need care; shape validation is new |
| 3 — Query path | ~485 | **Medium** — aggregation return-type wrapping; tuple-key indexing |
| 4 — Interop | ~420 | **Medium-High** — Arrow `FixedSizeList` in `_pa_type_from_spec` and `_compiled_columns_from_arrow` is non-trivial |
| 5 — Polish | ~228 | **Low** — mostly guards and docs |
| **Total** | **~1,824** | — |

### Highest-risk areas

1. **`extend` ndarray path** (§2.2): the existing scalar branch does
   `np.ascontiguousarray(raw_columns[name], dtype=target_dtype)` followed by
   `self._cols[name][start_pos:end_pos] = values[:]`.  For ndarray columns the
   batch must have shape `(n_rows, *per_row_shape)`.  The main risk is
   correctly threading this through all dispatch branches (scalar/dict/list/varlen)
   without duplicating logic.

2. **Aggregation return type** (§3.1): `NDArray.max(axis=inner_axes)` goes
   through `LazyExpr` → `reduce_slices`, which returns a numpy array when
   `kwargs` is empty.  The Column method must wrap with
   `blosc2.asarray(result)` to get an NDArray that supports `__gt__` →
   `LazyExpr`.  This is a **one-liner fix** but easy to miss.

3. **Arrow interop** (§4.1): `_pa_type_from_spec` and
   `_compiled_columns_from_arrow` are moderately complex functions.
   `FixedSizeList` is a minority Arrow type with some rough edges in pyarrow.
   The flat-FixedSizeList + field-metadata convention needs to be correctly
   roundtripped through both Arrow and Parquet.

4. **`_where_expression_operands` guard** (§2.5 / §4.8): without this,
   string-expression queries like `t.where("embedding > 0")` silently pass
   a 2-D NDArray to numexpr, producing wrong results or a crash rather than
   a clear error.  Small fix, high consequence if missed.

---

## 10. Open questions

- **`b2.field()` default shorthand**: allow `default=0.0` to mean
  `default_factory=lambda: np.full(shape, 0.0, dtype=dtype)`?  Deferred —
  `default_factory` is explicit and unambiguous.

- **Nested ndarray in struct/list specs**: `ListSpec(item=NDArraySpec(...))`
  or `StructSpec({"a": NDArraySpec(...)})` not in v1 — the `BatchArray` /
  `ListArray` backends would need their own ndarray sub-path.  Users flatten
  nested structure into a single ndarray column.
