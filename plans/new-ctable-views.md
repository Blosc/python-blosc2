# New CTable Column View Plan

## Goal

Revisit the `CTable` column access API so that value materialization and
subcolumn-view creation are separated cleanly.

The intended direction is:

- `t.price` returns a `Column`
- `t.price[:]` returns values
- `t.price[2:10]` returns values
- `t.price.view[2:10]` returns a logical column view/proxy

This document is for later consideration. It is intentionally decoupled from
the current computed-column implementation work.

## Motivation

The current `Column` API mixes two different ideas under `__getitem__`:

- data access
- creation of a masked/logical subcolumn view

Today:

- `t.price[3]` returns a scalar
- `t.price[2:10]` returns a `Column`
- `t.price[:]` returns a `Column`
- `t.price.to_numpy()` returns values

This makes examples and user intuition awkward because:

- users naturally expect `[:]` to mean "give me the values"
- computed columns already lean in that direction
- `to_numpy()` becomes mandatory for simple inspection/printing

## Proposed Direction

Split the API into:

### Behavior summary

| Expression | Current result | Proposed result |
|---|---|---|
| `t.price` | `Column` | `Column` |
| `t.price[0]` | scalar | scalar |
| `t.price[2:5]` | `Column` view | `np.ndarray` |
| `t.price[:]` | `Column` view | `np.ndarray` |
| `t.price[[0, 1, 2]]` | `np.ndarray` | `np.ndarray` |
| `t.price[mask]` | `np.ndarray` | `np.ndarray` |
| `t.price.view` | not present | `ColumnViewIndexer` |
| `t.price.view[:]` | not present | `Column` view |
| `t.price.view[2:5]` | not present | `Column` view |

The intended rule is:

- indexing a `Column` yields values
- `.view[...]` yields a sub-view for further chained operations

### 1. Value access through `__getitem__`

`Column.__getitem__` should return values for:

- integer indexing: scalar
- slice indexing: NumPy array
- list/tuple/integer-array indexing: NumPy array
- boolean-mask indexing: NumPy array

This would align `Column` with normal array-like expectations.

### 2. Explicit logical views through `.view[...]`

Introduce a dedicated view accessor:

```python
t.price.view[2:10]
t.price.view[[1, 5, 9]]
t.price.view[mask]
```

This would return a `Column`-like proxy that:

- preserves the logical row mask
- supports normal assignment via `view[:] = values`
- supports aggregates and iteration
- writes through to the underlying physical column

The key design benefit is that aliasing becomes explicit.

## Why `view[...]` Instead of `slice(...)`

The main alternatives are:

- `t.price.view[2:10]`
- `t.price.slice(2, 10)`

Recommendation: prefer `view[...]`.

Reasons:

1. `slice()` already implies extraction of data rather than creation of a proxy.
2. `NDArray.slice()` already exists with data-oriented semantics, so reusing the
   same name for a writable logical subview on `Column` would be confusing.
3. `.view[...]` makes aliasing explicit: the user is creating another handle on
   the same data, not materializing values.
4. The bracket form naturally supports all existing logical indexing shapes:
   slice, list, boolean mask, integer arrays.

## Desired Semantics

### Data access

```python
t.price[0]          # scalar
t.price[:10]        # NumPy array
t.price[[1, 3, 5]]  # NumPy array
t.price[mask]       # NumPy array
t.price[:]          # full NumPy array
```

### Logical views

```python
sub = t.price.view[2:10]
sub[:] = np.zeros(len(sub))

evens = t.price.view[[0, 2, 4, 6]]
evens[0] = 99
```

### Consistency with computed columns

Computed columns should follow the same front-end API:

```python
t.gross[:]          # values
t.gross[2:10]       # values
t.gross.view[2:10]  # logical read-only view or explicit unsupported operation
```

Open question:

- should computed columns support `.view[...]` at all, or should they remain
  read-only value accessors with no view concept?

My current inclination is:

- support `.view[...]` for read-only logical subviews if it comes for free
- otherwise defer and keep `.view[...]` initially only for physical columns

## Compatibility Impact

This is a breaking semantic change.

Code that currently relies on:

```python
t.price[2:10][:] = values
t.price[2:10].to_numpy()
```

would need to migrate to:

```python
t.price.view[2:10][:] = values
t.price[2:10]
```

This affects:

- tests
- examples
- documentation
- any downstream user code relying on slice-returned `Column` views

## Suggested Implementation Shape

### New helper: `ColumnViewIndexer`

Add a small helper object:

```python
class ColumnViewIndexer:
    def __init__(self, column): ...
    def __getitem__(self, item): ...
```

and expose it as:

```python
@property
def view(self):
    return ColumnViewIndexer(self)
```

Responsibilities:

- translate logical indexing into the existing masked `Column(...)` view model
- preserve current write-through behavior
- keep the main `Column` implementation simpler

A more concrete shape would be:

```python
class ColumnViewIndexer:
    def __init__(self, column: Column) -> None:
        self._column = column

    def __getitem__(self, item) -> Column:
        return self._column._view_from_key(item)

    def __repr__(self) -> str:
        return f"<ColumnViewIndexer col={self._column._col_name!r}>"
```

and then:

```python
@property
def view(self) -> ColumnViewIndexer:
    return ColumnViewIndexer(self)
```

### `Column.__getitem__`

Refactor to:

- int -> scalar
- everything else -> NumPy array

The old slice/list/bool logic can still be reused internally, but it should end
in value materialization rather than `Column(...)` construction.

For slices specifically, the implementation should move from "build a mask and
return `Column(...)`" to "translate logical slice positions into physical row
positions, then materialize values".

### `Column.__setitem__`

The canonical bulk-write idiom should be normal slice assignment, not
`assign(...)`.

That means the target API should support:

```python
t.price[:] = values
t.price.view[2:10][:] = values
```

This is more Pythonic than routing bulk writes through a dedicated method, and
it keeps views feeling like real writable proxies.

`assign(...)` can remain temporarily as a compatibility helper, but it should be
demoted in the design and documentation.

### Internal helpers

It may help to split current logic into:

- `_view_from_key(key)` -> `Column`
- `_values_from_key(key)` -> scalar or NumPy array

Then:

- `__getitem__` uses `_values_from_key`
- `view.__getitem__` uses `_view_from_key`

That avoids duplicating index normalization.

This split is preferable to a monolithic rewrite of `__getitem__`, because it
lets the existing subview semantics survive behind a more explicit entry point.

## Open Questions

### 1. Should `[:]` return NumPy or `NDArray`?

Recommendation:

- keep `Column[:]` returning NumPy for now

Reason:

- current `Column.to_numpy()` already materializes to NumPy
- this is the least surprising behavior for users
- returning `NDArray` would create a second value-materialization idiom

### 2. Should `.view[...]` exist on computed columns?

Options:

- `ComputedColumn.view[...]` returns a read-only computed subview
- `ComputedColumn.view[...]` raises `NotImplementedError`

Recommendation:

- defer this decision until the physical-column split lands

### 3. Do we need a deprecation phase?

Probably yes.

Possible path:

1. add `.view[...]` first while keeping old slice behavior
2. update examples/docs/tests to use `.view[...]` where they mean views
3. warn on non-scalar `Column.__getitem__` returning `Column`
4. switch `__getitem__` to return values

This may be more churn than the project wants, but it is the safest migration.

If `assign(...)` is still present at that point, it should be documented as a
compatibility convenience rather than the primary writable idiom.

## Edge Cases To Preserve

### Already-masked columns

If the `Column` already carries a mask, for example from:

```python
t.head(3)["price"]
```

then `.view[...]` should apply relative to the already-visible logical rows, not
restart from the root physical column.

### Empty slices

The target behavior should match NumPy:

```python
t.price[5:3]  # -> np.array([], dtype=t.price.dtype)
```

### Slice assignment

Changing `Column.__getitem__` should not affect:

```python
t.price[0:5] = values
```

because that is governed by `__setitem__`, not `__getitem__`.

The same should hold for view objects:

```python
view = t.price.view[0:5]
view[:] = values
```

This should be treated as the preferred writable bulk-assignment idiom.

### Computed columns

This plan should not assume too much here, but one likely consequence is:

- `ComputedColumn.__getitem__(slice)` should return values
- `ComputedColumn.view[...]` is either:
  - a read-only logical subview, or
  - explicitly unsupported in the first iteration

The important part is to avoid relying on unsupported fancy-indexing paths on
lazy expressions while implementing slice materialization.

## Migration Notes

Two main categories of downstream changes are expected.

### 1. Simplifications

Code that only wanted materialized values becomes shorter:

```python
# before
col[0:5].to_numpy()

# after
col[0:5]
```

### 2. Explicit view migration

Code that relied on slice-returned `Column` views must move to `.view[...]`:

```python
# before
view = col[0:5]
view[:] = values

# after
view = col.view[0:5]
view[:] = values
```

Likely affected areas:

- `tests/ctable/test_column.py`
- computed-column tests such as [tests/ctable/test_computed_column.py](/Users/faltet/blosc/python-blosc2-proves/tests/ctable/test_computed_column.py)
- examples that currently call `col[slice].to_numpy()`
- documentation/tutorial snippets that describe slice-returned `Column` objects

## Files Likely To Change

| File | Expected change |
|---|---|
| `src/blosc2/ctable.py` | add `Column.view`, add `ColumnViewIndexer`, split value access from view creation |
| `tests/ctable/test_column.py` | update slice semantics and move view-oriented assertions to `.view[...]` |
| `tests/ctable/test_computed_column.py` | align slice expectations for computed columns |
| `examples/ctable/computed-columns.py` | simplify materialization idioms where slice access now returns values |
| other examples/docs using `col[slice].to_numpy()` | simplify to `col[slice]` where appropriate |

## Recommended Next Step

When this is revisited, the first patch should be small:

1. add `Column.view[...]`
2. migrate internal tests/examples that truly want writable subviews
3. keep `Column.__getitem__` unchanged for that patch

Then a second patch can decide whether to flip slice/list/bool indexing to
return values by default.

That phased path reduces risk and gives the new API a chance to settle before
introducing a breaking semantic change.
