# CTable fixed-shape ndarray columns — follow-up plan

This plan assumes the current core implementation exists:

- `blosc2.ndarray(item_shape, dtype=...)` / `NDArraySpec`
- physical storage as `(nrows, *item_shape)`
- append / extend shape validation
- row access returning per-row NumPy arrays
- persistence round-trip
- `np.asarray(table)` using structured subarray dtypes

The remaining work below focuses on making ndarray columns safer, more queryable,
and easier to analyze from inside a `CTable`.

---

## 1. Immediate safety guards

These should land before advertising ndarray columns broadly.  Without them,
existing scalar-oriented table operations may accidentally receive 2-D/3-D arrays
and produce confusing errors.

### 1.1 String `where()` expressions

Skip ndarray columns in string-expression operands, or reject them with a clear
message.

Example rejected form:

```python
t.where("embedding > 0")
```

Suggested error:

```text
Column 'embedding' is a fixed-shape ndarray column. String expressions only
support scalar columns. Use an element projection or a row-wise reduction first.
```

Files likely involved:

- `src/blosc2/ctable.py`, `_where_expression_operands` / string where helpers

### 1.2 Sorting, grouping, indexing

Reject ndarray columns in operations requiring scalar comparable values:

- `sort_by("embedding")`
- `group_by("embedding")`
- `create_index("embedding")`
- scalar-only helpers used by `describe()`, `cov()`, `corr()`

Suggested alternatives in error messages:

```text
Cannot sort by ndarray column 'embedding' with per-row shape (768,).
Materialize a scalar feature first, e.g. embedding_norm or embedding_max.
```

---

## 2. Column metadata API

Add small, cheap introspection helpers to `Column`.

```python
t.embedding.shape  # (n_live_rows, 768)
t.embedding.item_shape  # (768,)
t.embedding.ndim  # 2
t.embedding.item_ndim  # 1
t.embedding.size  # n_live_rows * 768
t.embedding.item_size  # 768
```

For scalar columns:

```python
t.price.item_shape == ()
t.price.item_ndim == 0
```

These are useful for generic code that handles both scalar and fixed-shape array
columns.

---

## 3. Element projection / inner-axis slicing

Support tuple indexing on ndarray columns, where the first selector addresses
rows and remaining selectors address the per-row item axes.

Examples:

```python
t.embedding[:, 0]  # first coordinate for all live rows
t.embedding[:10, :4]  # first 4 coordinates for first 10 live rows
t.image[:, :, :, 0]  # channel 0 for all rows, if item_shape=(H, W, C)
t.matrix[5, :, :]  # one row's matrix
```

Return type can initially be NumPy arrays, matching current `Column.__getitem__`
materialization behavior.  Later, the common full-row-slice case may return a
Blosc2 `NDArray` view/projection if such support becomes available.

Important behavior:

- tuple indexing is only accepted for `NDArraySpec` columns
- scalar columns keep current indexing semantics
- boolean row masks still refer to live logical rows

---

## 4. Direct comparison guard

Block ambiguous comparisons on full ndarray columns:

```python
t.embedding > 0.5
```

A full comparison returns an N-D boolean array, not the 1-D row mask required by
`CTable.where()`.  Raise a clear error.

Suggested message:

```text
Cannot compare ndarray column 'embedding' directly; the result would not be a
1-D row mask. Use an element projection like t.embedding[:, 0] > 0.5 or a
row-wise feature like t.embedding.row_max() > 0.5.
```

---

## 5. Practical user-facing analysis helpers

This is an extra practical addition beyond the existing plans: provide explicit
row-wise feature extraction methods on ndarray columns.  These make it easy to
analyze fixed-shape columns without leaving `CTable` ergonomics.

### 5.1 Row-wise reductions

Add methods that reduce only the inner item axes and return one value per row:

```python
t.embedding.row_sum()
t.embedding.row_mean()
t.embedding.row_min()
t.embedding.row_max()
t.embedding.row_std()
t.embedding.row_var()
t.embedding.row_any()
t.embedding.row_all()
t.embedding.row_norm(ord=2)
```

For `item_shape=(768,)`, each returns shape `(nrows,)`.
For `item_shape=(H, W, C)`, each reduces axes `(1, 2, 3)` by default.

Optional `axis=` can reduce selected inner axes:

```python
t.image.row_mean(axis=(1, 2))  # mean over H,W, keep channel dimension
```

Naming these `row_*` avoids ambiguity with existing scalar-column `.sum()` /
`.max()` methods, which currently mean “reduce the whole column to a scalar”.

### 5.2 Materialize features as scalar columns

Add a convenience method for storing derived scalar/vector features:

```python
t.embedding.add_feature("embedding_norm", op="norm")
t.embedding.add_feature("embedding_max", op="max")
t.embedding.add_feature("embedding_0", element=0)
```

Equivalent manual workflow today would be:

```python
values = t.embedding.row_norm()[:]
t.add_column("embedding_norm", blosc2.field(blosc2.float64(), default=0.0))
t.embedding_norm[:] = values
```

The helper can:

- choose output dtype
- validate output shape is 1-D for scalar feature columns
- optionally create an index immediately:

```python
t.embedding.add_feature("embedding_norm", op="norm", create_index=True)
```

This is practical for workflows like similarity screening, image metadata,
embeddings, and sensor windows.

### 5.3 Quick summary method

Add:

```python
t.embedding.summary()
```

Potential output:

```text
ndarray column 'embedding'
  rows       : 1,000,000 live / 1,048,576 capacity
  item_shape : (768,)
  dtype      : float32
  storage    : NDArray shape=(1048576, 768), chunks=(..., ...), blocks=(..., ...)
  cbytes     : ...
  row stats  : min(row_norm)=..., mean(row_norm)=..., max(row_norm)=...
```

Keep this opt-in so normal table display stays compact.

---

## 6. Display improvements

Normal table display should not dump whole per-row arrays for wide shapes.

Suggested formatting:

- small 1-D item shapes, e.g. `(3,)`: show `[1.0, 2.0, 3.0]`
- larger shapes: show `ndarray(shape=(768,), dtype=float32)`
- multi-dimensional shapes: show `ndarray(shape=(32, 32, 3), dtype=uint8)`

`Column.__repr__` can use the same threshold.

---

## 7. Nullable ndarray columns

Later addition.  Current implementation should either reject nullable ndarray
specs or document that they are unsupported.

A robust design would use broadcast sentinels:

- float: all-NaN item means null
- signed int: all min or all max, following null policy
- unsigned int: all min or all max, following null policy
- bool: uint8 sentinel pattern

Null mask for a batch:

```python
inner_axes = tuple(range(1, arr.ndim))
mask = np.isnan(arr).all(axis=inner_axes)  # float
mask = (arr == null_value).all(axis=inner_axes)  # non-float
```

This requires changes in:

- null policy resolution
- null sentinel validation
- `Column._null_mask_for`
- append / extend coercion
- Arrow import/export null bitmap handling

---

## 8. Arrow / Parquet interop

Map ndarray columns to Arrow `FixedSizeList`.

Recommended representation:

- flatten each row in C order
- Arrow type: `fixed_size_list(value_type, prod(item_shape))`
- store original shape in field metadata, e.g.
  `b"blosc2:ndarray_shape": b"[2, 3]"`

Examples:

```text
item_shape=(768,), dtype=float32 -> fixed_size_list(float32, 768)
item_shape=(2, 3), dtype=float64 -> fixed_size_list(float64, 6) + shape metadata
```

Parquet can inherit this via Arrow.

---

## 9. CSV / DataFrame behavior

CSV:

- require explicit schema for import
- serialize ndarray cells as flattened values, preferably JSON arrays for
  readability and shape safety

DataFrame:

- export ndarray columns as object dtype cells containing NumPy arrays
- do not infer ndarray columns from object dtype on import unless an explicit
  row schema is provided

---

## 10. Structured NumPy materialization

Current implementation uses NumPy structured subarray dtypes:

```python
np.asarray(t).dtype["matrix"].shape == (2, 3)
```

Keep this.  It is more useful than object dtype for `np.asarray(table)` because
it preserves a compact homogeneous memory representation when all fields are
fixed-size.

Potential fallback: use object dtype only for columns that cannot be represented
as structured subarrays.

---

## 11. Tests to add next

- tuple inner slicing
- direct comparison guard
- sort/groupby/index guard messages
- display of small and large item shapes
- `Column.ndim`, `size`, `item_shape`
- row-wise feature helpers
- add-feature helper and optional indexing
- Arrow/Parquet roundtrip for `(n,)` and `(m, n)` shapes
- nullable ndarray columns once implemented

---

## Suggested next phase

A good next small PR would be:

1. safety guards for scalar-only operations
2. `Column.item_shape`, `ndim`, `size`
3. tuple inner slicing
4. direct comparison guard
5. tests for the above

This would make the existing core implementation much safer and substantially
more ergonomic without taking on nullable columns or Arrow interop yet.
