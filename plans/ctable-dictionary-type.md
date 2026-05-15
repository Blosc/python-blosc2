# Plan: CTable dictionary/categorical column type

## Motivation

Real-world Parquet files frequently contain Arrow dictionary-encoded columns, especially repeated string
columns. Arrow represents these as:

```text
dictionary<values=string, indices=int32, ordered=0>
```

Today, `CTable.from_arrow()` does not support Arrow dictionary types directly. The compatibility fallback is
to decode dictionaries to plain strings before import, but this loses the compact representation and prevents
fast integer-code indexing.

Add a CTable dictionary column type with Arrow-like semantics:

```python
blosc2.dictionary(
    index_type=blosc2.int32(), value_type=blosc2.vlstring(), ordered=False
)
```

For v1, keep the implementation intentionally narrow and optimized for the common Parquet case: string
categories represented by signed 32-bit codes.

## Goals for v1

- Add a public dictionary column spec.
- Support dictionary columns in CTable schemas and persistent metadata.
- Store dictionary columns as stable integer codes plus a dictionary of unique string values.
- Import Arrow/Parquet dictionary-encoded string columns without decoding to full strings.
- Export CTable dictionary columns back to Arrow dictionary arrays.
- Allow decoded reads by default while exposing codes and dictionary values for advanced users.
- Enable equality/membership filtering to operate on integer codes.
- Make dictionary columns indexable by indexing their codes.
- Ensure the real-world `~/Downloads/chicago-taxi.parquet` dataset can round-trip to/from Blosc2 format.

## Non-goals for v1

- General value types beyond `vlstring`.
- General index types beyond internal `int32`.
- Nested dictionary columns inside list/struct fields.
- Dictionary compaction/removal of unused categories.
- Ordered comparisons (`<`, `>`, sorting) beyond storing the `ordered` flag.
- Per-chunk or per-batch dictionaries.
- Schema-less/object fallback support for dictionaries.

## Public API

### Column spec

Add:

```python
blosc2.dictionary(
    index_type=blosc2.int32(),
    value_type=blosc2.vlstring(),
    ordered=False,
    nullable=True,
)
```

For v1:

- `index_type` must be `blosc2.int32()`.
- `value_type` must be `blosc2.vlstring()`.
- `ordered` is persisted and exported to Arrow, but ordered comparisons are not implemented initially.
- `nullable=True` means row slots may be null. Nulls are represented internally by code `-1`.
- `nullable=False` rejects null slots during writes/import.

Consider an alias later:

```python
blosc2.categorical(...)
```

but implement only `dictionary` first to match Arrow terminology.

### Example schema usage

```python
from dataclasses import dataclass
import blosc2


@dataclass
class Trip:
    vendor: str = blosc2.field(
        blosc2.dictionary(index_type=blosc2.int32(), value_type=blosc2.vlstring())
    )
    fare: float = blosc2.field(blosc2.float64())
```

### Column access

Default reads should return decoded values:

```python
ct["vendor"][:]  # ["Uber", "Lyft", None, "Uber"]
ct["vendor"][0]  # "Uber"
```

Expose internals explicitly:

```python
ct["vendor"].codes[:]  # np.ndarray(dtype=int32), e.g. [0, 1, -1, 0]
ct["vendor"].dictionary[:]  # ["Uber", "Lyft"]
```

Use `.dictionary` as the preferred public name for unique values because it matches Arrow terminology and the
`blosc2.dictionary(...)` spec name. A pandas-friendly `.categories` alias can be considered later, but should
not be part of the v1 API unless it falls out naturally.

Useful methods/properties:

```python
col.codes  # fixed-width NDArray-like codes storage
col.dictionary  # varlen string array of unique values
col.encode(values)  # values -> int32 codes, extending dictionary if allowed
col.decode(codes)  # codes -> values
col.value_to_code(value)  # single value lookup; KeyError if absent
col.code_to_value(code)  # single code lookup
```

For v1, keep mutation methods minimal and internal if needed. Public `.codes` and `.dictionary` are enough
for inspection and debugging.

Logical slice reads should follow existing `vlstring` behavior and return Python lists, not NumPy object arrays:

```python
ct["vendor"][:]  # ["Uber", "Lyft", None, "Uber"]
```

## Semantics

### Logical model

A dictionary column is logically:

```text
row slot -> int32 code -> dictionary value
```

Example:

```text
codes:       [0, 1, 0, -1]
dictionary: ["Uber", "Lyft"]
decoded:    ["Uber", "Lyft", "Uber", None]
```

### Nulls

Use reserved code `-1` for null row slots.

Rationale:

- `int32` codes give a simple, compact null representation.
- Code comparisons and indexes can include null slots naturally.
- This avoids a separate validity bitmap for v1 dictionary columns.

Rules:

- Valid category codes are `0 <= code < len(dictionary)`.
- `-1` means null slot.
- Codes `< -1` are invalid.
- If `nullable=False`, attempts to write/import null slots raise `ValueError`.
- Dictionary values themselves should not be null in v1. Null is represented only by slot code `-1`.

### Dictionary growth

Use an append-only global dictionary per column.

- New string values append to the dictionary and receive the next code.
- Existing values reuse their existing code.
- Deleting table rows does not remove dictionary values.
- Updating a row to a new value may append a new dictionary value.
- Codes are stable for the life of the column.

No automatic compaction in v1. A future explicit operation can be added:

```python
ct["vendor"].compact_dictionary()
```

but this requires recoding all codes and rebuilding any indexes, so defer it.

### Maximum cardinality

Because v1 uses signed `int32` and reserves `-1` for null, the maximum number of categories is:

```text
2_147_483_648
```

Practically, memory/storage constraints will be hit earlier. If appending a new category would exceed
`np.iinfo(np.int32).max`, raise `OverflowError`.

## Storage layout

Represent a dictionary column as a logical column object wrapping two persisted components:

```text
<ctable store>/
  _cols/
    vendor/
      codes        # int32 NDArray, one code per row
      dictionary   # variable-length string storage, unique values
```

Exact on-disk naming should match existing table storage conventions, but the logical layout should be
column-local. Do not store dictionary values as a separate user-visible CTable column.

### Codes storage

- Fixed-width `int32` NDArray.
- Shape grows with table rows.
- Uses the normal column compression parameters.
- Indexes operate on this codes array.

### Dictionary value storage

Use the existing variable-length scalar string machinery where possible:

- `vlstring` values.
- Append-only.
- Stored under the dictionary column directory.
- Maintains insertion order as category order.

### In-memory lookup cache

Maintain an in-memory mapping for fast encoding:

```python
_value_to_code: dict[str, int]
```

Build it lazily from persisted dictionary values when opening a table. Persist only dictionary values, not the
Python mapping.

## Schema metadata

Add a new spec kind, likely in `src/blosc2/schema.py`:

```json
{
  "kind": "dictionary",
  "index_type": {"kind": "int", "bits": 32, "signed": true, ...},
  "value_type": {"kind": "vlstring", ...},
  "ordered": false,
  "nullable": true,
  "null_code": -1
}
```

The compiler should produce a `CompiledColumn` with:

- logical type: dictionary;
- physical dtype for codes: `np.int32`;
- display width based on decoded strings, not codes where feasible.

Schema validation should reject unsupported v1 combinations early:

- non-`int32` index type;
- non-`vlstring` value type;
- null dictionary values;
- nullable policies incompatible with `-1` null code.

## Core implementation tasks

### 1. Add `DictionarySpec`

Implement in schema/spec layer:

- constructor helper `blosc2.dictionary(...)`;
- metadata serialization/deserialization;
- equality/repr/docs;
- validation of v1 constraints.

Potential fields:

```python
@dataclass(frozen=True)
class DictionarySpec(ColumnSpec):
    index_type: IntSpec
    value_type: VLStringSpec
    ordered: bool = False
    nullable: bool = True
    null_code: int = -1
```

### 2. Add dictionary column object

Implement a column class, for example:

```python
class DictionaryColumn:
    codes: blosc2.NDArray
    dictionary: _ScalarVarLenArray  # or existing vlstring backing type
```

Required operations:

- `__len__`
- `__getitem__` scalar/slice/list/boolean mask returning decoded values
- `__setitem__` scalar/slice/list values, encoding as needed
- `append` / `extend` for Arrow import and row appends
- `flush` if dictionary storage uses buffered batch machinery
- `close` if needed

For v1, prioritize the operations used by CTable append/import/read paths.

### 3. Extend table storage

Add storage factory methods analogous to existing list/varlen methods:

```python
storage.create_dictionary_column(name, spec, cparams=None, dparams=None)
storage.open_dictionary_column(name, spec, ...)
```

These create/open both physical components (`codes`, `dictionary`) under the logical column.

### 4. Extend CTable schema compilation and column creation

Update CTable creation paths to detect `DictionarySpec`:

- schema compiler;
- `_create_columns` / equivalent new-table creation;
- `_create_arrow_import_columns`;
- open-from-storage path;
- row append/update paths;
- column widths/display.

Dictionary columns should be logical `ct.col_names` entries just like ordinary columns.

### 5. Decoded read/write behavior

When assigning Python values:

```python
ct.append({"vendor": "Uber"})
ct["vendor"][3] = "Lyft"
ct["vendor"][4:6] = ["Uber", None]
```

Encoding behavior:

- If value is `None`: code `-1` if nullable, otherwise raise.
- If value is `str` and exists: use existing code.
- If value is `str` and missing: append dictionary value, assign new code.
- If value is not `str`/`None`: raise `TypeError`.

When assigning raw codes, require explicit codes API. Do not silently accept integers via logical column writes,
because integers could be real category values in future dictionary types.

## Arrow/Parquet interoperability

### Import from Arrow

Map Arrow dictionary columns as follows:

```text
dictionary<values=string, indices=int8|int16|int32|int64, ordered=X>
  -> blosc2.dictionary(index_type=blosc2.int32(), value_type=blosc2.vlstring(), ordered=X)
```

Accepted Arrow index types for v1:

- signed integer indices: `int8`, `int16`, `int32`, `int64`;
- unsigned integer indices: `uint8`, `uint16`, `uint32`, `uint64`, provided all values fit in signed
  `int32`;
- normalize internally to `int32`;
- reject if category count or any index value does not fit signed `int32`.

Accepted Arrow value types for v1:

- `string`, `large_string`, `utf8`, `large_utf8`;
- normalize internally to `vlstring`.

Rejected for v1:

- dictionary values of binary, numeric, struct, list, etc.;
- nested dictionary arrays inside list/struct;
- unsigned index arrays containing values that do not fit in signed `int32`.

### Chunked Arrow arrays and dictionary unification

Arrow chunked arrays and Parquet row groups may carry different dictionaries per chunk. CTable v1 should use
one global dictionary per column.

Import algorithm:

1. For each incoming Arrow dictionary array chunk:
   - read its dictionary values;
   - map chunk-local category values to global codes;
   - translate chunk indices to global int32 codes;
   - translate Arrow nulls to `-1`.
2. Append translated codes to the CTable codes storage.
3. Append new category values to the global dictionary as discovered.

Preserve first-seen category order. This is deterministic for a given input stream and works well for append-only
semantics.

If `ordered=True` and chunks have different dictionary orders, global first-seen order may not preserve the
semantic order. For v1:

- preserve and export `ordered=True` only when the importer can verify all chunk dictionaries have the same
  order for existing values;
- otherwise raise `ValueError`. Do not silently downgrade to `ordered=False`, because `ordered=True` carries
  semantic meaning and silently changing it could make comparisons/sorts incorrect later.

### Arrow schema inference

Update `_arrow_type_to_spec()`:

- recognize top-level Arrow dictionary type;
- return `DictionarySpec` for supported v1 string dictionaries;
- raise clear `TypeError` for unsupported dictionary variants.

Do not decode dictionary type to plain string inside core `CTable.from_arrow()` when dictionary support is
available. The CLI can later expose a flag to force decoding if desired.

### Arrow batch writing into CTable

Update `_write_arrow_batch()`:

- if compiled column is dictionary:
  - accept Arrow dictionary arrays and use the unification algorithm;
  - also optionally accept plain string arrays by encoding strings into the dictionary;
  - reject unsupported types.

This allows appending plain strings to an existing dictionary CTable column.

### Export to Arrow

When `iter_arrow_batches()` sees a dictionary column, emit Arrow dictionary arrays:

```text
dictionary<values=string, indices=int32, ordered=spec.ordered>
```

Implementation approach:

- Arrow dictionary values: `pa.array(dictionary_values, type=pa.string())` or `pa.large_string()`? Use `pa.string()`
  for v1 unless a value exceeds Arrow string limits, then use `large_string()`.
- Indices: `pa.array(codes, type=pa.int32())`, with null mask for `codes == -1`.
- Construct `pa.DictionaryArray.from_arrays(indices, dictionary, ordered=spec.ordered)`.

For slices/batches, reuse the full column dictionary rather than creating per-batch dictionaries. This preserves
stable codes and simplifies export.

### Parquet CLI behavior

Once core dictionary support exists:

- Default CLI import should preserve supported Arrow dictionary string columns as dictionary CTable columns.
- Add an escape hatch:

```bash
parquet-to-blosc2 --decode-dictionaries input.parquet output.b2d
```

or equivalent if users want plain `vlstring` columns.

The default should favor preserving dictionary encoding because it is compact and closer to the original Arrow
schema.

## Query and expression support

### Equality

For dictionary column `vendor`:

```python
ct["vendor"] == "Uber"
```

should translate to:

```python
ct["vendor"].codes == code_for("Uber")
```

If the value is absent from the dictionary, return an all-false boolean expression/selection without scanning.

Null equality:

```python
ct["vendor"] == None
```

maps to:

```python
codes == -1
```

Use whatever null comparison idiom is already preferred in CTable expressions; avoid encouraging `== None` in
user docs if there is an `is_null()` API.

### Membership

```python
ct["vendor"].isin(["Uber", "Lyft"])
```

maps to code membership:

```python
codes in [0, 1]
```

Values absent from the dictionary are ignored. If all requested values are absent, return all-false.

### Ordered comparisons

For v1:

- If `ordered=False`, `<`, `<=`, `>`, `>=` should raise `TypeError` for dictionary columns.
- If `ordered=True`, still defer implementation unless it is trivial to map to code comparisons. Document that
  ordered comparisons are not supported in v1 even though the flag is stored/exported.

This avoids ambiguous semantics between dictionary order and lexical string order.

## Indexing support

Dictionary columns should be indexed by codes.

### Index creation

User API should remain logical:

```python
ct.create_index("vendor")
```

Internally:

- detect `vendor` is dictionary;
- create the physical index on `vendor.codes`;
- store public index metadata under the logical column name `vendor`;
- mark the index as dictionary-aware so query planning maps values to codes before using it.

The public API should hide the code-index detail. On disk, index files may include an explicit `codes` suffix,
for example `__index__.vendor.codes...`, to avoid ambiguity and make debugging easier.

Avoid requiring users to write:

```python
ct["vendor"].codes.create_index()
```

though exposing code-level indexes for debugging is fine.

### Query planning with indexes

For equality:

1. Look up the queried string in the dictionary.
2. If present, query the integer index for that code.
3. If absent, return empty result immediately.

For membership:

1. Map present values to codes.
2. Query the integer index for those codes.
3. Ignore absent values.

For nulls:

- code `-1` can be included in the code index.
- `is_null()` queries use code `-1`.

### Index maintenance

Because dictionary values are append-only and codes are stable:

- existing index entries do not need recoding when new categories are appended;
- appending rows updates the code index just like appending rows to an integer column;
- deleting rows follows existing CTable valid-row semantics;
- dictionary compaction, if added later, must invalidate/rebuild indexes.

## Persistence and compatibility

### Opening existing tables

Existing tables do not contain dictionary specs, so no migration is needed.

### Versioning

Add a schema metadata version bump if the CTable schema format has one. Older versions of python-blosc2 will not
understand `kind: dictionary`; they should fail clearly when opening such tables.

### Robustness checks on open

When opening a persisted dictionary column:

- validate codes dtype is int32;
- validate dictionary storage exists;
- validate dictionary values are strings and contain no null entries;
- optionally validate codes are `-1` or within dictionary bounds. Full validation may be expensive; provide a
  debug/validation path rather than doing it unconditionally for huge tables.

## Testing plan

### Unit tests for spec/schema

- `blosc2.dictionary()` creates expected spec.
- Unsupported `index_type` raises.
- Unsupported `value_type` raises.
- Metadata roundtrip preserves `ordered`, `nullable`, `null_code`.
- Dataclass schema compilation supports dictionary fields.

### CTable behavior tests

- Create in-memory CTable with dictionary column.
- Append strings and nulls.
- Repeated strings reuse codes.
- New strings append dictionary values.
- Decoded scalar/slice reads work.
- `.codes[:]` and `.dictionary[:]` expose expected internals.
- `nullable=False` rejects nulls.
- Invalid value types raise.
- Persistent `.b2d`/`.b2z` tables reopen correctly.

### Arrow import/export tests

- Import `dictionary<values=string, indices=int8>`.
- Import `dictionary<values=string, indices=int16>`.
- Import `dictionary<values=string, indices=int32>`.
- Import `dictionary<values=string, indices=int64>` when values fit int32.
- Import unsigned dictionary indices when values fit signed int32.
- Reject too-large signed/unsigned dictionary indices or category counts.
- Import chunked arrays with different dictionaries and verify global unification.
- Preserve nulls as `-1` internally and Arrow nulls on export.
- Export emits Arrow dictionary type with int32 indices and string values.
- Parquet roundtrip preserves logical values.

### Query/index tests

- Equality filter on present value returns matching rows.
- Equality filter on absent value returns no rows without scanning if possible.
- Membership filter works.
- Null filter works.
- `ct.create_index("dict_col")` builds code index.
- Equality/membership use the code index.
- Appending rows after index creation maintains index correctness.

### CLI tests

- `parquet-to-blosc2` imports dictionary string column as dictionary column.
- Export produces Parquet/Arrow dictionary column.
- Optional dictionary-decoding flag imports as `vlstring` instead.
- Unsupported dictionary value type reports a clear error or decodes only if explicitly requested.
- Real-world acceptance test: `~/Downloads/chicago-taxi.parquet` imports to Blosc2, exports back to Parquet,
  and round-trip comparison succeeds for imported/exported columns.

## Suggested implementation order

1. Add `DictionarySpec` and public `blosc2.dictionary()` helper.
2. Implement dictionary column storage wrapper with codes + vlstring dictionary.
3. Integrate dictionary columns into CTable creation/open/read/write paths.
4. Add decoded reads and append/set encoding.
5. Add Arrow dictionary import with global dictionary unification.
6. Add Arrow export as `pa.DictionaryArray`.
7. Add equality/membership expression translation.
8. Add dictionary-aware index creation and query usage.
9. Add CLI preservation by default and optional decode flag.
10. Add docs/examples.

## Resolved design decisions

These decisions are part of the v1 plan:

1. Expose `nullable` on the dictionary spec, defaulting to `True`.
2. Accept Arrow unsigned dictionary indices if all values fit in signed `int32`; normalize internally to
   `int32`.
3. Raise for ordered Arrow dictionaries with incompatible/differing chunk dictionary order. Do not silently
   downgrade to unordered.
4. Make the Parquet CLI preserve supported dictionary columns by default. Provide an opt-out flag such as
   `--decode-dictionaries` for users who want plain `vlstring` columns.
5. Use `.dictionary` as the preferred public property for unique values. Consider `.categories` only as a
   future alias.
6. Return Python lists for logical slice reads, following existing `vlstring` behavior.
7. Keep `ct.create_index("vendor")` logical and hide code-index details from the public API. On-disk index
   artifacts may include a `codes` suffix for clarity.
