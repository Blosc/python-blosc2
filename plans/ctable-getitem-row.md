# Enhance `CTable` indexing and access semantics

## Motivation

Current `CTable.__getitem__` only supports column-name lookup:

```python
t["col"]
```

This leaves several surprising or missing behaviors:

- `t[1]` does not mean “row 1”
- `t[1:3]` does not mean a row slice
- `t[[1, 3]]` / `t[mask]` do not provide natural row selection through `[]`
- `t[["a", "b"]]` does not provide natural column projection
- `t["f1 > f2"]` is not available as shorthand for row filtering
- `t.row[...]` exists as a separate row API, which becomes redundant once `t[...]` gains proper row semantics

This would give `CTable` a much more natural and powerful access model while preserving efficient columnar behavior.

---

## Design goals

1. Make `CTable` indexing intuitive by key type.
2. Keep column access ergonomic.
3. Make scalar row access feel scalar-like.
4. Keep slices/masks/projections as table/view operations when possible.
5. Add convenient expression-based filtering.
6. Preserve efficient columnar behavior instead of materializing rows eagerly by default.
7. Provide explicit NumPy materialization hooks when desired.

---

## Proposed semantics

## 1. String key

### Column lookup first

If the string matches a column name:

```python
t["ingredients"]  # -> Column
```

### Otherwise: boolean expression filter

If the string is not a column name, interpret it as a boolean expression:

```python
t["f1 > f2"]
```

This should behave like:

```python
t.where("f1 > f2")
```

and return a filtered `CTable` view.

### Error behavior

If the string is neither:

- a known column name
- nor a valid boolean expression

raise a clear exception.

Recommended:

- parsing/evaluation failure -> propagate a meaningful expression error
- non-boolean expression result -> `TypeError` or `ValueError`

Example:

```python
t["f1 + f2"]  # error: expression does not evaluate to bool
```

---

## 2. Integer key

Integer indexing should return a single row object:

```python
t[1]
```

Recommended return type:

- a schema-derived **namedtuple row**

Example:

```python
row = t[1]
row.ingredients
row.id
```

### Why namedtuple

- scalar-like semantics for scalar indexing
- lightweight and familiar
- readable repr/printing
- works across mixed column types
- no need to force a NumPy structured scalar as the primary API

### Negative indexing

Support:

```python
t[-1]
```

with normal Python sequence semantics.

### Out-of-bounds

Raise `IndexError`.

---

## 3. Slice key

Slice indexing should return a row-selected `CTable` view:

```python
t[1:3]
```

This preserves columnar/view behavior and avoids eager row materialization.

### Important

Slices should **not** default to NumPy structured arrays. The view is more natural for Blosc2 because:

- it preserves table operations
- it avoids eager materialization
- it composes better with later filtering/projection

---

## 4. Integer-list / integer-array key

These should perform row gather and return a `CTable` view:

```python
t[[1, 3, 5]]
t[np.array([1, 3, 5])]
```

Return:

- row-selected `CTable` view

---

## 5. Boolean mask key

Boolean mask selection should return a filtered `CTable` view:

```python
t[mask]
```

Return:

- row-selected `CTable` view

Behavior should align with current `where(...)` / row-view behavior.

---

## 6. List of strings key

Column projection should be supported directly:

```python
t[["col1", "col2"]]
```

Return:

- a projected `CTable` view/subtable with only those columns

This is important and should be distinguished from row-gather lists by inspecting element types.

### Rules

- list of `str` -> column projection
- list/array of `int` -> row gather
- boolean array -> row filter

### Validation

Unknown column names should raise `KeyError`.

---

## 7. Tuple / multidimensional indexing

Do not add full multidimensional indexing in this change.

For now:

- unsupported tuple keys should raise `TypeError`

Possible future extension:

```python
t[rows, cols]
```

but that should not be introduced in the first pass.

---

## 8. Remove `t.row[...]`

Once `t[...]` gains proper row semantics, `t.row[...]` no longer adds much value.

Recommendation:

- deprecate `t.row[...]`
- then remove it in a follow-up cleanup

Rationale:

- `t[1]` is the natural spelling for row access
- keeping both `t[1]` and `t.row[1]` is redundant
- one obvious way is better here

### Transitional plan

Phase 1:

- keep `t.row[...]` working
- mark it as deprecated
- update docs/examples to prefer `t[...]`

Phase 2:

- remove `t.row[...]`

---

## 9. `where(..., columns=[...])`

The natural explicit filtered-projection API should be:

```python
t.where("x > 3", columns=["a", "b"])
```

not `query(...)`.

This should return a filtered `CTable` view projected to the selected columns.

### Relationship with shorthand `t["expr"]`

These should align:

```python
t["x > 3"]
```

should behave like:

```python
t.where("x > 3")
```

while:

```python
t.where("x > 3", columns=["a", "b"])
```

provides the explicit projected variant.

---

## 10. `__array__()` for `CTable` and views

It is useful to support explicit NumPy materialization via:

```python
np.asarray(t)
np.asarray(t[1:10])
```

This should produce a NumPy structured array.

### dtype mapping policy

When computing the structured dtype:

- fixed-width scalar columns -> native NumPy dtypes
- `vlstring` / `vlbytes` -> `object`
- list columns -> `object`
- nested/object-backed values -> `object`

Example structured dtype:

```python
[
    ("id", np.int64),
    ("score", np.float64),
    ("name", object),
    ("ingredients", object),
]
```

### Why this is useful

- interop with NumPy APIs
- explicit materialization for debugging/export
- compatibility with code expecting record arrays / structured arrays

### Why not use this as default `__getitem__` behavior

Because:

- slices/views should stay lazy-ish and table-oriented
- eager row-major materialization is expensive
- Blosc2 is fundamentally columnar

So `__array__()` should be an explicit materialization hook, not the default indexing return type for slices/masks.

---

## Summary of target API

Recommended access behavior:

```python
t["col"]  # -> Column
t["f1 > f2"]  # -> filtered CTable view
t[1]  # -> namedtuple row
t[-1]  # -> namedtuple row
t[1:3]  # -> row-sliced CTable view
t[[1, 3, 5]]  # -> gathered-row CTable view
t[mask]  # -> filtered CTable view
t[["a", "b"]]  # -> projected CTable view
np.asarray(t[1:3])  # -> structured ndarray
```

And explicit filtering/projection:

```python
t.where("x > 3")
t.where("x > 3", columns=["a", "b"])
```

---

## Current implementation areas to update

## File: `src/blosc2/ctable.py`

### `CTable.__getitem__`

This is the main dispatch point.

It should dispatch by key type and contents:

1. `str`
   - if exact column name -> `Column`
   - else -> expression filter
2. `int`
   - return namedtuple row
3. `slice`
   - return row-selected view
4. `list` / `np.ndarray`
   - if strings -> projected view
   - if bool mask -> filtered view
   - if ints -> gathered-row view
5. otherwise
   - raise `TypeError`

### Row object creation

Add a cached namedtuple type per schema.

Suggested internal helper:

- `_row_namedtuple_type()`
- `_materialize_row(index)`

The row object should be built from logical/live row semantics, not raw physical slots.

### Projection helpers

Add a helper for column subset views, e.g.:

- `_project_columns(names)`

This should preserve:

- schema subset
- computed column handling if applicable
- shared column storage when safe

### Expression dispatch helper

Add a helper for string-expression indexing, e.g.:

- `_getitem_expression(expr)`

This should likely reuse:

- existing `where(...)`
- existing expression parsing/evaluation

with the rule that expressions must evaluate to booleans.

### `where()` enhancement

Extend `where()` to accept:

```python
columns = [...]
```

so it can combine filtering and projection.

---

## Other files that may need updates

## Tests

Likely in:

- `tests/ctable/test_row_logic.py`
- `tests/ctable/test_ctable_dataclass_schema.py`
- new dedicated indexing tests if cleaner

## Documentation/examples

Any examples using:

- `t.row[...]`
- column-only `t[...]`

should be updated to reflect the new design.

---

## Testing plan

## New tests for `__getitem__`

### Column access

```python
assert isinstance(t["id"], Column)
```

### Row access by int

```python
row = t[1]
assert row.id == ...
assert row.text == ...
```

### Negative int

```python
row = t[-1]
```

### Slice access

```python
sub = t[1:4]
assert isinstance(sub, CTable)
assert len(sub) == 3
```

### Integer gather

```python
sub = t[[1, 3, 5]]
```

### Boolean mask

```python
sub = t[mask]
```

### Column projection

```python
sub = t[["a", "b"]]
assert sub.col_names == ["a", "b"]
```

### Expression string

```python
sub = t["a > b"]
```

### String precedence

If a column is literally named something expression-like, exact column-name match should win.

Example:

```python
t["a > b"]
```

returns the column if that exact column exists; otherwise it is treated as an expression.

### Error tests

- unknown projected column -> `KeyError`
- unsupported key type -> `TypeError`
- non-boolean expression -> error
- out-of-bounds integer -> `IndexError`

---

## Backward-compatibility notes

This is a meaningful behavior change.

### Old behavior

- `t["col"]` -> column
- `t[1]` -> not meaningful / confusing
- `t.row[1]` -> row selection path

### New behavior

- `t["col"]` -> column
- `t["expr"]` -> filter if not a column name
- `t[1]` -> namedtuple row
- `t[1:3]` -> ctable view
- `t[["a", "b"]]` -> projected view
- `t.row[...]` -> deprecated

This should be documented clearly.

---

## Implementation phases

## Phase 1: typed `__getitem__` dispatch

1. Update `CTable.__getitem__`
2. Add int/slice/list/mask/string-expression routing
3. Add clear exceptions for unsupported keys

## Phase 2: namedtuple rows

1. Add schema-cached row namedtuple type
2. Add logical-row materialization helper
3. Make `t[i]` return namedtuple rows

## Phase 3: projection support

1. Add `t[["a", "b"]]`
2. Add internal projected-view helper
3. Ensure schema/column metadata stays consistent

## Phase 4: filtering shorthand and `where(columns=...)`

1. Add `t["expr"]`
2. Add `t.where(..., columns=[...])`
3. Ensure behavior matches between both forms

## Phase 5: NumPy materialization hooks

1. Add `__array__()` for tables/views
2. Use structured dtype with `object` fallback for varlen/nested columns
3. Optionally add explicit helpers like `to_numpy()` / `to_records()`

## Phase 6: remove `t.row[...]`

1. Remove `t.row[...]`
2. Update tests/docs/examples
