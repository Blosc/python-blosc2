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
Materialize a scalar generated column first, e.g. embedding_norm or embedding_max.
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
1-D row mask. Use an element projection like t.embedding[:, 0] > 0.5 or an
axis-aware reduction like t.embedding.max(axis=1) > 0.5.
```

---

## 5. Practical user-facing analysis helpers

Treat an ndarray column as a logical array with shape `(nrows, *item_shape)`.
Column reductions should follow NumPy / Blosc2 NDArray axis semantics over that
logical shape.

### 5.1 Axis-aware reductions

For scalar columns, existing behavior remains unchanged:

```python
t.price.shape  # (nrows,)
t.price.sum()  # scalar reduction over rows
```

For ndarray columns:

```python
t.embedding.shape  # (nrows, dim)
t.embedding.sum()  # scalar full reduction, same as axis=None
t.embedding.sum(axis=None)  # scalar full reduction
t.embedding.sum(axis=0)  # reduce rows -> shape (dim,)
t.embedding.sum(axis=1)  # reduce embedding coords -> shape (nrows,)
t.embedding.norm(axis=1)  # row-wise norm -> shape (nrows,)
```

For image-like columns:

```python
t.image.shape  # (nrows, H, W, C)
t.image.mean()  # scalar full reduction
t.image.mean(axis=0)  # mean image over rows -> shape (H, W, C)
t.image.mean(axis=(1, 2))  # per-row per-channel mean -> shape (nrows, C)
t.image.sum(axis=(1, 2, 3))  # per-row total -> shape (nrows,)
```

This avoids special `row_*` methods and minimizes surprise: the table row is
always the leading axis (`axis=0`), exactly as if the column were an NDArray of
shape `(nrows, *item_shape)`.

`CTable.where()` still requires a 1-D row mask.  For example,
`t.embedding.norm(axis=1) > 5` is valid, whereas
`t.image.mean(axis=(1, 2)) > 0.5` returns shape `(nrows, C)` and should be
rejected unless further reduced to `(nrows,)`.

### 5.2 Materialize reductions as generated columns

Use `CTable.add_generated_column()` as the canonical API for storing generated
scalar/vector columns.  This is consistent with the existing
`add_computed_column()` name while making the storage/maintenance semantics
explicit.

Generated columns are real stored columns declared from source columns and
maintained by the table.  They differ from virtual computed columns, which are
not stored and are evaluated lazily.  Generated columns are cheap to query
repeatedly and directly indexable, at the cost of storage and maintenance.

#### Row-oriented ndarray transformers

For ndarray columns, use a column-bound `row_transformer`.  It transforms each
row value independently and sees only the per-row `item_shape`, not the table's
leading `nrows` dimension.  Thus, for an embedding with item shape `(dim,)`,
`axis=0` means the embedding-coordinate axis.

```python
t.add_generated_column(
    "embedding_norm",
    values=t["embedding"].row_transformer.norm(axis=0, ord=2),
    dtype=blosc2.float64(),
    create_index=True,
)

t.add_generated_column(
    "embedding_max",
    values=t["embedding"].row_transformer.max(axis=0),
    dtype=blosc2.float32(),
)

t.add_generated_column(
    "embedding_0",
    values=t["embedding"].row_transformer[0],
    dtype=blosc2.float32(),
)
```

For image-like rows with item shape `(H, W, C)`:

```python
t.add_generated_column(
    "image_mean_rgb",
    values=t["image"].row_transformer.mean(axis=(0, 1)),
    dtype=blosc2.ndarray((3,), dtype=blosc2.float32()),
)

t.add_generated_column(
    "red_channel_mean",
    values=t["image"].row_transformer[:, :, 0].mean(axis=(0, 1)),
    dtype=blosc2.float32(),
)
```

#### Expression transformers

For scalar expressions, keep the same ergonomic forms as computed columns:

```python
t.add_generated_column(
    "total",
    values="price * qty",
    dtype=blosc2.float64(),
    create_index=True,
)

t.add_generated_column(
    "big_tip",
    values=(t.payment.tips > 100),
    dtype=blosc2.bool(),
    create_index=True,
)

t.add_generated_column(
    "price_with_tax",
    values=lambda cols: cols["price"] * 1.21,
    dtype=blosc2.float64(),
)
```

Equivalent manual workflow for the embedding norm today would be:

```python
values = np.linalg.norm(t.embedding[:], axis=1)
t.add_column("embedding_norm", blosc2.field(blosc2.float64(), default=0.0))
t.embedding_norm[:] = values
t.create_index("embedding_norm")
```

Suggested signatures:

```python
def add_generated_column(
    self,
    name: str,
    *,
    values: (
        str
        | blosc2.LazyExpr
        | Callable[[dict[str, blosc2.NDArray]], blosc2.LazyExpr]
        | RowTransformer
    ),
    dtype=None,
    create_index: bool = False,
) -> None: ...


def add_computed_column(
    self,
    name: str,
    expr: (
        str | blosc2.LazyExpr | Callable[[dict[str, blosc2.NDArray]], blosc2.LazyExpr]
    ),
    *,
    dtype=None,
) -> None: ...
```

`add_computed_column()` should not accept `RowTransformer` in v1.  A
`RowTransformer` often represents row-wise ndarray reductions/projections that
cannot yet be represented as a virtual LazyExpr.  If such support is added later,
`add_computed_column()` can accept only `RowTransformer` instances that implement
a lazy lowering path; otherwise it should raise a clear error recommending
`add_generated_column()`.

Validation rules for generated columns:

- string / LazyExpr / callable transformers follow the same dependency rules as computed columns
- `RowTransformer` is accepted only by `add_generated_column()`
- source columns are taken from the normalized transformer metadata
- generated columns are always stored and maintained; virtual columns remain the job of `add_computed_column()`
- generated columns intended for indexing must produce a 1-D result with length `nrows`

`RowTransformer` API sketch:

```python
t["embedding"].row_transformer.norm(axis=0, ord=2)
t["embedding"].row_transformer.max(axis=0)
t["image"].row_transformer.mean(axis=(0, 1))
t["embedding"].row_transformer[0]
t["image"].row_transformer[:, :, 0]
```

A `RowTransformer` object should expose at least:

```python
transformer.kind
transformer.source_columns
transformer.to_metadata()
transformer.evaluate_existing(table)
transformer.evaluate_batch(raw_columns)
```

The helper should:

- choose or validate output dtype
- validate output shape is compatible with the requested generated column spec
- write existing rows immediately
- register generated-column metadata so `append()` / `extend()` can auto-fill new rows
- mark a created index as stale, or rebuild/update it, when new rows arrive
- optionally create an index immediately

Suggested row-transformer metadata shape:

```python
{
    "kind": "generated_column",
    "output": "embedding_norm",
    "source_columns": ["embedding"],
    "materialized": True,
    "maintain_on_append": True,
    "stale_on_source_update": True,
    "dtype": "float64",
    "transformer": {
        "kind": "row_reduction",
        "source": "embedding",
        "op": "norm",
        "ord": 2,
        "axis": 0,
    },
    "index": {
        "create": True,
        "stale_policy": "mark_stale",
    },
}
```

Expression-generated columns use expression transformer metadata:

```python
{
    "kind": "generated_column",
    "output": "total",
    "source_columns": ["price", "qty"],
    "materialized": True,
    "maintain_on_append": True,
    "stale_on_source_update": True,
    "dtype": "float64",
    "transformer": {
        "kind": "expression",
        "expression": "price * qty",
    },
}
```

Expected maintenance behavior:

```python
t.add_generated_column(
    "embedding_norm",
    values=t["embedding"].row_transformer.norm(axis=0, ord=2),
    dtype=blosc2.float64(),
    create_index=True,
)
t.append((new_id, new_embedding))

# embedding_norm is automatically appended for the new row too
assert t.embedding_norm[-1] == np.linalg.norm(new_embedding)
```

Source-column mutation needs an explicit policy.  For v1, the simplest safe
policy is to mark dependent generated columns stale when source values are
assigned through `t.embedding[...] = ...`, and provide refresh methods:

```python
t.refresh_generated_column("embedding_norm")
t.refresh_generated_columns(source="embedding")
```

A later optimization can update only the affected generated rows during
`Column.__setitem__`, but correctness should come first.

This is practical for workflows like similarity screening, image metadata,
embeddings, sensor windows, and scalar business metrics that benefit from
materialization and indexing.

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
  row stats  : min(norm(axis=1))=..., mean(norm(axis=1))=..., max(norm(axis=1))=...
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
- axis-aware ndarray column reductions
- `add_generated_column()` with optional indexing
- generated-column auto-fill on `append()` / `extend()`
- generated-column staleness / refresh behavior after source-column mutation
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
