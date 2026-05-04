# CTable Nullable Scalar / Parquet Fidelity Plan

## Summary

Improve CTable scalar null handling so Parquet nullable scalar columns can round-trip with high fidelity:

```text
Parquet null -> CTable null sentinel -> Parquet null
```

This plan focuses on the remaining fidelity gaps after batch-wise Parquet import/export and Arrow schema metadata support:

- nullable numeric scalar columns;
- nullable string/bytes scalar columns;
- nullable bool scalar columns.

The guiding approach is to continue using CTable's existing in-band `null_value` sentinel model for scalar columns, while adding sensible default sentinel choices for Parquet imports and a special physical representation for nullable bools.

---

## Decisions

### 1. Automatic sentinel inference defaults to enabled for Parquet imports

`CTable.from_parquet()` should default to automatic scalar null sentinels:

```python
CTable.from_parquet(path, auto_null_sentinels=True)
```

That is, the public default should become effectively `True` for Parquet imports because preserving Parquet nulls is the expected behavior for interchange.

For lower-level Arrow batch imports, default can be considered separately, but the recommended behavior is also to make it available and likely default-on when importing Arrow schemas from Parquet.

### 2. String/bytes sentinels are configurable public options

Default string/bytes sentinels:

```python
string_null_value = "__BLOSC2_NULL__"
bytes_null_value = b"__BLOSC2_NULL__"
```

Expose them on import APIs:

```python
CTable.from_parquet(
    path,
    auto_null_sentinels=True,
    string_null_value="__BLOSC2_NULL__",
    bytes_null_value=b"__BLOSC2_NULL__",
)
```

and similarly for `from_arrow_batches()` where appropriate.

No full collision scan is performed by default. This avoids whole-column scans for large datasets. Users who know their data can override the sentinel values.

### 3. Nullable bool API supports both explicit sentinel and convenience nullable flag

Support both:

```python
blosc2.bool(null_value=255)
blosc2.bool(nullable=True)
```

`nullable=True` is shorthand for:

```python
null_value = 255
```

Physical representation:

```text
False = 0
True  = 1
Null  = 255
```

A nullable bool column is logically a bool column but physically stored as `uint8` so the sentinel is preserved.

### 4. Nullable bool expression rewrites use a conservative scope

Initial expression support should be:

- equality/inequality rewrites everywhere CTable column expressions are built;
- bare `flag` and `~flag` rewrites only in filter contexts.

For nullable bool encoded as `0/1/255`:

```text
flag == True   -> raw == 1
flag == False  -> raw == 0
flag != True   -> raw == 0
flag != False  -> raw == 1
```

In filter contexts:

```text
flag  -> raw == 1
~flag -> raw == 0
```

This excludes nulls from both true and false selections, matching common tabular filtering expectations.

---

## Goals

1. Preserve nullable numeric, string, bytes, and bool Parquet columns through CTable round-trips.
2. Avoid whole-column pre-scans for sentinel selection.
3. Keep using the current scalar sentinel null model for consistency with existing nullable columns.
4. Add general nullable bool column support to CTable, not just a Parquet-specific workaround.
5. Preserve batch-wise import/export behavior.
6. Maintain clear, predictable raw storage semantics.
7. Keep nullable bool expression/filter behavior useful without implementing full three-valued logic in V1.

---

## Non-goals

- Add separate validity bitmap storage for scalar CTable columns.
- Implement full SQL/Arrow/Pandas Kleene three-valued boolean algebra in V1.
- Guarantee no sentinel collision for string/bytes columns without user-provided sentinels.
- Hide all sentinels from raw column reads.
- Automatically infer collision-free string/bytes sentinels by scanning entire columns.

---

## Current behavior and gap

CTable already supports scalar nulls via `null_value` sentinels, e.g.:

```python
blosc2.int64(null_value=-1)
blosc2.float64(null_value=float("nan"))
blosc2.string(max_length=16, null_value="")
```

Arrow/Parquet export already maps sentinel values back to Arrow nulls in the scalar export path:

```text
sentinel in CTable -> Arrow null bitmap -> Parquet null
```

The missing pieces are:

1. During Parquet import, infer and attach appropriate sentinels automatically.
2. During batch writes, replace Arrow nulls with those sentinels before writing to CTable storage.
3. For bool columns, introduce a physical representation that can actually preserve a sentinel.
4. For strings/bytes, choose practical default sentinels without full scans.
5. Avoid lossy importer workarounds such as filling nulls with `0`, `NaN`, or `""` without recording them as actual `null_value` sentinels.

---

## Sentinel policy

### Numeric columns

For nullable Arrow/Parquet numeric fields, if `auto_null_sentinels=True`:

```text
int8    -> np.iinfo(np.int8).min
int16   -> np.iinfo(np.int16).min
int32   -> np.iinfo(np.int32).min
int64   -> np.iinfo(np.int64).min
uint8   -> np.iinfo(np.uint8).max
uint16  -> np.iinfo(np.uint16).max
uint32  -> np.iinfo(np.uint32).max
uint64  -> np.iinfo(np.uint64).max
float32 -> NaN
float64 -> NaN
```

These become the column spec's `null_value`.

Example:

```python
pa.field("score", pa.int32(), nullable=True)
```

maps to:

```python
blosc2.int32(null_value=np.iinfo(np.int32).min)
```

### String columns

For nullable Arrow string/large_string fields:

```python
blosc2.string(max_length=..., null_value="__BLOSC2_NULL__")
```

The selected `max_length` must be large enough to store both actual values and the sentinel. Therefore:

```python
max_length = max(inferred_or_configured_max_length, len(string_null_value))
```

No collision scan is performed by default.

### Bytes columns

For nullable Arrow binary/large_binary fields:

```python
blosc2.bytes(max_length=..., null_value=b"__BLOSC2_NULL__")
```

The selected `max_length` must be at least:

```python
len(bytes_null_value)
```

No collision scan is performed by default.

### Bool columns

For nullable Arrow bool fields:

```python
blosc2.bool(nullable=True)
```

or equivalently:

```python
blosc2.bool(null_value=255)
```

Physical storage dtype:

```python
np.uint8
```

Encoding:

```text
0   false
1   true
255 null
```

Non-null Arrow bool values are converted as:

```text
False -> 0
True  -> 1
```

Arrow nulls are converted as:

```text
Null -> 255
```

---

## Schema changes

### `blosc2.schema.bool`

Current `bool` spec has fixed dtype:

```python
dtype = np.dtype(np.bool_)
```

Change it to support nullable bools:

```python
class bool(SchemaSpec):
    python_type = builtins.bool

    def __init__(self, *, nullable: bool = False, null_value=None):
        if nullable and null_value is None:
            null_value = 255
        if null_value is not None and null_value != 255:
            raise ValueError("Nullable bool null_value must be 255")
        self.null_value = null_value
        self.nullable = null_value is not None
        self.dtype = np.dtype(np.uint8) if self.nullable else np.dtype(np.bool_)
```

Metadata:

```json
{"kind": "bool"}
```

or:

```json
{"kind": "bool", "nullable": true, "null_value": 255}
```

### Display/type labels

A nullable bool column should still display as a logical bool column, possibly with a nullable marker:

```text
bool?
```

or:

```text
bool(nullable)
```

Internally, physical dtype is `uint8`.

---

## Arrow/Parquet import changes

### API additions

Add/import parameters:

```python
@classmethod
def from_parquet(
    cls,
    path,
    *,
    columns=None,
    batch_size=65_536,
    urlpath=None,
    mode="w",
    cparams=None,
    dparams=None,
    validate=False,
    auto_null_sentinels=True,
    string_null_value="__BLOSC2_NULL__",
    bytes_null_value=b"__BLOSC2_NULL__",
    **kwargs,
): ...
```

For `from_arrow_batches()`:

```python
def from_arrow_batches(
    cls,
    schema,
    batches,
    *,
    urlpath=None,
    mode="w",
    auto_null_sentinels=True,
    string_null_value="__BLOSC2_NULL__",
    bytes_null_value=b"__BLOSC2_NULL__",
): ...
```

If there is concern about changing `from_arrow_batches()` defaults, keep that default as `False` initially but make `from_parquet()` pass `True`.

### Type mapping

Add helper:

```python
def _auto_null_sentinel(pa, pa_type, *, string_null_value, bytes_null_value): ...
```

Return appropriate sentinel for supported nullable scalar types.

When building specs from Arrow fields:

```python
if auto_null_sentinels and field.nullable:
    null_value = _auto_null_sentinel(...)
else:
    null_value = None
```

Then pass `null_value` into the corresponding SchemaSpec constructor.

For list and struct fields, do not use scalar sentinels; their nested nulls are handled in the ListArray/Python-object layer.

---

## Batch write behavior

When writing an Arrow column into CTable storage:

### List columns

Unchanged:

```python
list_col.extend(arrow_col.to_pylist())
```

### String/bytes columns

If Arrow nulls exist and the CTable spec has a `null_value`, replace `None` with the sentinel before building the NumPy array:

```python
values = arrow_col.to_pylist()
if null_value is not None:
    values = [null_value if v is None else v for v in values]
arr = np.array(values, dtype=col.dtype)
```

If Arrow nulls exist and no sentinel is configured, raise a clear error.

### Numeric columns

Use Arrow null mask:

```python
arr = arrow_col.to_numpy(zero_copy_only=False).astype(col.dtype)
if arrow_col.null_count:
    arr[np.asarray(arrow_col.is_null())] = null_value
```

For floats, Arrow may produce `NaN` for nulls already, but still explicitly applying the sentinel is clearer and consistent.

### Nullable bool columns

If physical dtype is `uint8`:

```python
values = arrow_col.to_numpy(zero_copy_only=False)
arr = values.astype(np.uint8)  # False -> 0, True -> 1
if arrow_col.null_count:
    arr[np.asarray(arrow_col.is_null())] = 255
```

Need care because Arrow `to_numpy()` on nullable bool may produce object arrays or fail in some versions. Fallback:

```python
py_values = arrow_col.to_pylist()
arr = np.array([255 if v is None else int(v) for v in py_values], dtype=np.uint8)
```

This fallback is acceptable for bool columns.

---

## Arrow/Parquet export behavior

The existing sentinel-to-Arrow-null logic should be extended to nullable bool physical `uint8` columns.

For scalar export:

```python
arr = col[:]
nv = col.null_value
null_mask = col._null_mask_for(arr) if nv is not None else None
```

For nullable bool:

```python
values = arr == 1
pa.array(values, mask=null_mask, type=pa.bool_())
```

Important: do not export nullable bool physical `uint8` as Arrow uint8. The logical Arrow type should be bool.

For non-nullable bool, keep current behavior:

```python
pa.array(arr)  # Arrow bool
```

---

## Nullable bool raw access semantics

Raw reads expose the physical sentinel representation:

```text
False -> 0
True  -> 1
Null  -> 255
```

Example:

```python
t["flag"][:]  # np.array([1, 0, 255], dtype=uint8)
```

This is consistent with CTable's existing nullable scalar model, where raw reads expose sentinel values.

`Column.is_null()`, `Column.notnull()`, and `Column.null_count()` should work normally.

---

## Nullable bool expression/filter semantics

### Equality/inequality rewrites

For nullable bool columns, rewrite these comparisons before they reach the generic LazyExpr mechanism:

```text
flag == True   -> raw == 1
flag == False  -> raw == 0
flag != True   -> raw == 0
flag != False  -> raw == 1
```

This ensures nulls do not match either true or false predicates.

These rewrites can apply everywhere CTable column expressions are built.

### Bare bool in filter context

In filter contexts:

```python
ct.where(ct.flag)
```

rewrite as:

```python
ct.where(ct.flag == 1)
```

### Negation in filter context

In filter contexts:

```python
ct.where(~ct.flag)
```

rewrite as:

```python
ct.where(ct.flag == 0)
```

This prevents nulls from being included by `~(raw == 1)` semantics.

### Temporary logical bool arrays

If needed for expression support, create temporary in-memory bool NDArrays scoped to the operation:

```python
flag_true = raw == 1
flag_false = raw == 0
```

These temporaries:

- are not persisted;
- are not added to `ct._cols`;
- live only during operations like `ct.where(...)`;
- can be compressed if materialized as Blosc2 NDArrays;
- should be avoided when a simple comparison rewrite is enough.

### Explicitly out of scope for V1

Full nullable boolean algebra, e.g. preserving nulls in value-producing expressions:

```text
~flag -> [False, True, Null]
flag & other_nullable_flag
flag | other_nullable_flag
```

can be deferred. V1 focuses on practical filtering semantics.

---

## Sorting and index interaction

Nullable scalar sentinels must not leak into user-visible sort/filter semantics.

### Sorting behavior

Nulls should participate in sorting but always sort last, regardless of sort direction:

```python
ct.sort_by("score", ascending=True)  # non-null ascending, nulls last
ct.sort_by("score", ascending=False)  # non-null descending, nulls last
```

The current non-indexed sort path already uses a null-indicator key:

```text
0 = non-null
1 = null
```

as a more significant lexsort key, which gives nulls-last behavior.

However, FULL-index sort fast paths can be wrong for nullable columns because they sort raw sentinels:

```text
signed int sentinel = dtype min     -> sorts first ascending
uint/bool sentinel = dtype max/255  -> order depends on direction
string sentinel = lexicographic     -> order depends on value
```

V1 rule:

> If the sort key column has `null_value`, do not use the FULL-index sort fast path unless the index is explicitly marked null-aware and nulls-last. Fall back to the null-aware lexsort path.

Future index metadata can include:

```json
{
  "null_aware": true,
  "null_order": "last"
}
```

so sorted index paths can safely support nullable columns.

### Indexed `where()` behavior

Normal comparisons should not match nulls:

```python
ct.where(ct.score > 10)  # null score rows excluded
ct.where(ct.score < 10)  # null score rows excluded
ct.where(ct.score == 10)  # null score rows excluded
ct.where(ct.score != 10)  # null score rows excluded
```

Explicit null selection should use:

```python
ct.where(ct.score.is_null())
```

For index-accelerated `where()`, raw sentinel values can otherwise produce incorrect matches. Examples:

```text
uint null sentinel = max_uint -> may match score > 10
int null sentinel = min_int   -> may match score < 0
string sentinel              -> may match lexicographic ranges
```

V1 rule:

> If an indexed query references nullable columns, post-filter index-produced candidate positions with the relevant `notnull` physical masks before returning them.

For simple comparisons and AND-only expressions:

```python
ct.where((ct.score > 10) & (ct.age < 20))
```

index result positions should be filtered as:

```python
positions = positions[score_notnull[positions] & age_notnull[positions]]
```

This is simple, safe, and avoids null sentinel false positives.

### OR expressions

Global null-mask post-filtering is not correct for OR expressions. Example:

```python
ct.where((ct.score > 10) | (ct.category == 3))
```

A row with `score == null` and `category == 3` should match. A global mask:

```python
score.notnull() & category.notnull()
```

would wrongly drop it.

V1 rule:

> For indexed expressions containing OR over nullable columns, fall back to full scan unless the expression has been rewritten with branch-local null checks.

Future branch-local AST rewrite:

```python
(score > 10) | (category == 3)
```

becomes:

```python
((score > 10) & score.notnull()) | ((category == 3) & category.notnull())
```

This lets the existing planner see a semantically correct predicate. The planner may still need post-filtering or index support for the temporary null-mask operands, but correctness no longer depends on a global mask.

### Does this remove the need for a nullable-aware planner?

Partly. Injecting/post-filtering null masks handles many cases without changing the planner:

- simple comparisons;
- range predicates;
- AND-only combinations.

A planner is still useful for:

- deciding when an expression is AND-only vs contains OR;
- identifying referenced nullable columns;
- combining indexed predicates efficiently;
- supporting future branch-local null rewrites.

So the V1 approach is not a full nullable-aware planner rewrite. It is a correctness layer around existing index results.

---

## Interaction with schema metadata

The CTable schema already stores each column's `null_value` in the column spec metadata.

For diagnostics and Parquet fidelity metadata, Arrow metadata can optionally record import-time sentinel decisions:

```json
{
  "metadata": {
    "arrow": {
      "fields": {
        "score": {
          "original_arrow_type": "int32",
          "null_sentinel": -2147483648
        },
        "label": {
          "original_arrow_type": "string",
          "null_sentinel": "__BLOSC2_NULL__"
        },
        "flag": {
          "original_arrow_type": "bool",
          "null_sentinel": 255,
          "physical_dtype": "uint8"
        }
      }
    }
  }
}
```

This is optional because the CTable schema itself is sufficient to export sentinels as nulls.

---

## Impact on `off/import-to-b2z-gpt.py`

Once nullable scalar support is implemented, the OFF importer should stop manually filling nullable scalar nulls.

Current workaround:

```text
nullable numeric/string -> fill with 0/NaN/""
nullable bool -> wrap as list<bool>
```

Future behavior:

```text
nullable numeric -> scalar with auto sentinel
nullable string  -> scalar with "__BLOSC2_NULL__" sentinel
nullable bytes   -> scalar with b"__BLOSC2_NULL__" sentinel
nullable bool    -> scalar nullable bool, physical uint8 sentinel 255
```

Long strings may still be wrapped as `list<string>` for variable-length storage, but that becomes a storage choice, not a null-preservation workaround.

Expected OFF roundtrip improvement:

- null-count differences for numeric/string/bool scalar columns should drop to zero;
- value differences caused by filled nulls should disappear;
- remaining differences should be due to intentional schema/storage transformations, if any.

---

## Tests

### Numeric nulls

1. Nullable int Parquet column round-trips null counts and values.
2. Nullable uint Parquet column round-trips null counts and values.
3. Nullable float Parquet column round-trips null counts and values using NaN sentinel.
4. Exported Parquet contains real nulls, not sentinel values.

### String/bytes nulls

1. Nullable string Parquet column imports with `null_value="__BLOSC2_NULL__"`.
2. Export maps sentinel back to Parquet nulls.
3. `max_length` is at least `len("__BLOSC2_NULL__")`.
4. Nullable bytes column behaves similarly with `b"__BLOSC2_NULL__"`.
5. User-provided `string_null_value` / `bytes_null_value` are respected.

### Bool nulls

1. `blosc2.bool(nullable=True)` uses physical `uint8` dtype.
2. `blosc2.bool(null_value=255)` is equivalent.
3. Nullable bool Parquet imports as `0/1/255` raw values.
4. Export maps `255` back to Parquet nulls and emits Arrow bool type.
5. `is_null()`, `notnull()`, `null_count()` work.

### Bool filtering

For raw values:

```text
[1, 0, 255]
```

Verify:

```python
ct.where(ct.flag == True)  # true row only
ct.where(ct.flag == False)  # false row only
ct.where(ct.flag != True)  # false row only
ct.where(ct.flag != False)  # true row only
ct.where(ct.flag)  # true row only
ct.where(~ct.flag)  # false row only
```

### OFF roundtrip

Run:

```bash
python off/import-to-b2z-gpt.py --roundtrip --overwrite
```

Expected:

- same row count;
- same column count;
- same Arrow types for imported/exported columns;
- zero null-count differences for scalar columns using auto sentinels;
- significantly fewer value differences than current baseline.

---

## Implementation milestones

### Milestone 1: Port numeric auto sentinel support

- Add `_auto_null_sentinel()` helper.
- Add `auto_null_sentinels` to `from_arrow_batches()` and `from_parquet()`.
- Replace Arrow nulls with numeric sentinels during batch writes.
- Add numeric nullable Parquet tests.

### Milestone 2: String/bytes sentinels

- Add `string_null_value` and `bytes_null_value` parameters.
- Include sentinel length in max length calculation.
- Replace Arrow nulls with configured sentinel during batch writes.
- Export sentinels as Arrow nulls.
- Add string/bytes nullable tests.

### Milestone 3: Nullable bool schema/storage

- Extend `blosc2.bool()` with `nullable=True` and `null_value=255`.
- Use `np.uint8` physical dtype for nullable bool.
- Adjust display/type helpers where needed.
- Add import/export conversion for nullable bool.
- Add nullable bool tests.

### Milestone 4: Nullable bool filter rewrites

- Detect nullable bool columns in expression-building paths.
- Rewrite equality/inequality comparisons.
- Rewrite bare nullable bool and `~nullable_bool` in filter contexts.
- Add filter tests.

### Milestone 5: Update OFF importer and assess

- Remove scalar null filling workaround.
- Stop wrapping nullable bools as `list<bool>`.
- Keep long string wrapping only when desired for storage efficiency.
- Run roundtrip assessment and document remaining differences.

---

## Open questions

1. Should `from_arrow_batches()` default `auto_null_sentinels=True`, or only `from_parquet()`?
2. Should `blosc2.bool(null_value=255)` allow any other sentinel in the future, or always enforce `255`?
3. Should raw nullable bool display show `True`/`False`/`NULL` even though raw reads expose `0/1/255`?
4. Should string/bytes sentinel collision checking be offered as an optional slow mode?
5. Should singleton-list long strings eventually be replaced by true variable-length scalar string storage?
