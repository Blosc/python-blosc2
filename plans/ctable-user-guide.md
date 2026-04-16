# CTable User Guide

This document explains how to use `CTable` as it currently stands.

---

## What is CTable?

`CTable` is a columnar compressed table built on top of `blosc2.NDArray`. Each
column is stored as a separate compressed array. Rows are never physically removed
on deletion — instead a boolean mask (`_valid_rows`) marks live rows, and
compaction can be triggered manually or automatically.

---

## Defining a schema

A schema is a Python `@dataclass` where each field uses `b2.field()` to declare
the column type and constraints.

```python
from dataclasses import dataclass
import blosc2 as b2


@dataclass
class Row:
    id: int = b2.field(b2.int64(ge=0))
    score: float = b2.field(b2.float64(ge=0, le=100), default=0.0)
    active: bool = b2.field(b2.bool(), default=True)
```

### Available spec types

| Spec | NumPy dtype | Constraints |
|---|---|---|
| `b2.int64(ge, gt, le, lt)` | `int64` | numeric bounds |
| `b2.float64(ge, gt, le, lt)` | `float64` | numeric bounds |
| `b2.bool()` | `bool_` | — |
| `b2.complex64()` | `complex64` | — |
| `b2.complex128()` | `complex128` | — |
| `b2.string(min_length, max_length, pattern)` | `U<n>` | length / regex |
| `b2.bytes(min_length, max_length)` | `S<n>` | length |

Constraints are enforced on every insert (see **Validation** below).

### Inferred shorthand

For columns with no constraints and no per-column storage options, you can omit
`b2.field()` entirely:

```python
@dataclass
class Row:
    id: int  # inferred as b2.int64()
    score: float  # inferred as b2.float64()
    flag: bool = True  # inferred as b2.bool(), default=True
```

### Dataclass field ordering rule

Python dataclasses require that fields **with defaults come after fields without
defaults**. Plan your schema accordingly:

```python
@dataclass
class Row:
    id: int = b2.field(b2.int64())  # required — no default
    score: float = b2.field(b2.float64(), default=0.0)  # optional
    active: bool = b2.field(b2.bool(), default=True)  # optional
```

---

## Creating a table

```python
import blosc2 as b2

# Empty table (in-memory)
t = b2.CTable(Row)

# Table pre-loaded with data
t = b2.CTable(Row, new_data=[(1, 95.0, True), (2, 80.0, False)])

# Reserve space upfront (avoids resizes)
t = b2.CTable(Row, expected_size=1_000_000)

# Disable constraint validation (faster for trusted data)
t = b2.CTable(Row, validate=False)

# Enable auto-compaction (fills gaps before resizing)
t = b2.CTable(Row, compact=True)

# Table-level compression settings (applied to all columns unless overridden)
t = b2.CTable(Row, cparams={"codec": b2.Codec.ZSTD, "clevel": 5})
```

### Persistent tables

Pass `urlpath` to store the table on disk. Persistent `CTable` is backed by a
`TreeStore`, and `blosc2.open(urlpath)` can materialize it directly from the
root `/_meta` manifest.

```python
# Create a new persistent table (overwrites any existing table at that path)
t = b2.CTable(Row, urlpath="people", mode="w", expected_size=1_000_000)
t.extend([(i, float(i % 100), True) for i in range(10_000)])

# Open an existing persistent table for reading and writing
t = b2.CTable(Row, urlpath="people", mode="a")
t.append((99999, 50.0, True))

# Open read-only (default for CTable.open)
t = b2.CTable.open("people")  # mode="r" by default
t = b2.CTable.open("people", mode="r")  # explicit

# Open read/write via the classmethod
t = b2.CTable.open("people", mode="a")

# Generic open() also materializes the richer object
t = b2.open("people")
```

`mode` values:

| mode | behaviour |
|---|---|
| `"w"` | create (overwrite if the path already exists) |
| `"a"` | open existing or create new |
| `"r"` | open existing read-only |

In-memory tables (`urlpath=None`, the default) behave exactly as before — no
`mode` or path handling is involved.

Recommended conventions:

- extensionless paths default to directory-backed stores
- `.b2d` and `.b2z` are still valid and useful conventions, but no longer required

### Store layout

```
people/                  ← TreeStore root (extensionless directory-backed example)
    embed.b2e            ← internal store metadata
    _meta.b2f            ← SChunk manifest with kind/version/schema in vlmeta
    _valid_rows.b2nd     ← tombstone mask
    _cols/
        id.b2nd
        score.b2nd
        active.b2nd
```

You can inspect the raw metadata:

```python
import blosc2, json

store = blosc2.TreeStore("people", mode="r")
meta = store["/_meta"]
print(meta.vlmeta["kind"])  # "ctable"
print(meta.vlmeta["version"])  # 1
schema = json.loads(meta.vlmeta["schema"])
```

### Per-column storage options

```python
@dataclass
class Row:
    id: int = b2.field(b2.int64(), cparams={"codec": b2.Codec.LZ4, "clevel": 1})
    score: float = b2.field(
        b2.float64(ge=0, le=100),
        cparams={"codec": b2.Codec.ZSTD, "clevel": 9},
        default=0.0,
    )
```

Column-level `cparams`/`dparams`/`chunks`/`blocks` override the table-level
defaults for that column only.

---

## Inserting data

### `append()` — one row at a time

Accepts a tuple, list, dict, or dataclass instance:

```python
t.append((1, 95.0, True))
t.append([2, 80.0, False])
t.append({"id": 3, "score": 50.0, "active": True})
```

Fields with defaults can be omitted:

```python
t.append((4,))  # score=0.0 and active=True filled from defaults
```

### `extend()` — bulk insert

Accepts a list of tuples, a NumPy structured array, or another `CTable`:

```python
# List of tuples
t.extend([(i, float(i), True) for i in range(1000)])

# NumPy structured array
import numpy as np

dtype = np.dtype([("id", np.int64), ("score", np.float64), ("active", np.bool_)])
arr = np.array([(1, 50.0, True), (2, 75.0, False)], dtype=dtype)
t.extend(arr)

# Another CTable
t.extend(other_table)
```

#### Per-call validation override

```python
# Skip validation for one trusted batch (even if table was built with validate=True)
t.extend(trusted_data, validate=False)

# Force validation for one batch (even if table was built with validate=False)
t.extend(external_data, validate=True)
```

---

## Validation

When `validate=True` (the default), constraints declared in the schema are
enforced on every insert:

```python
t.append((-1, 50.0, True))  # ValueError: id violates ge=0
t.append((1, 150.0, True))  # ValueError: score violates le=100
t.extend([(-1, 50.0, True)])  # ValueError: id violates ge=0
```

Boundary values are accepted:

```python
t.append((0, 0.0, True))  # ok — id=0 satisfies ge=0, score=0.0 satisfies ge=0
t.append((1, 100.0, False))  # ok — score=100.0 satisfies le=100
```

To skip validation entirely:

```python
t = b2.CTable(Row, validate=False)
```

---

## Reading data

### Row access

```python
t.row[0]  # first row → returns a single-row CTable view
t.row[-1]  # last row
t.row[2:5]  # slice → CTable view with rows 2, 3, 4
t.row[::2]  # every other row
t.row[[0, 5, 10]]  # specific rows by logical index
```

Row access always uses **logical indices** (i.e. index 0 is the first live row,
not the first physical slot).

### Column access

```python
t["id"]  # returns a Column object
t.score  # attribute-style access also works

# Iterate values
for val in t["score"]:
    print(val)

# Convert to NumPy array
arr = t["score"].to_numpy()

# Single value
val = t["id"][5]  # logical index 5
```

### Column slicing

```python
col_view = t["id"][0:10]  # returns a Column view (mask applied)
arr = col_view.to_numpy()  # materialise to NumPy
```

### head / tail

```python
t.head(10)  # CTable view of first 10 rows
t.tail(5)  # CTable view of last 5 rows
```

---

## Deleting rows

`delete()` marks rows as invalid in the tombstone mask — data is not physically
removed.

```python
t.delete(0)  # delete first live row
t.delete(-1)  # delete last live row
t.delete([0, 2, 4])  # delete multiple rows by logical index
t.delete(list(range(10)))  # delete first 10 live rows
```

Negative indices and mixed positive/negative lists are supported.

---

## Compaction

After many deletions, physical storage has gaps. Compaction moves all live rows
to the front and clears the rest.

```python
t.compact()  # manual compaction
```

Auto-compaction runs automatically before a resize when `compact=True`:

```python
t = b2.CTable(Row, compact=True)
```

---

## Read-only mode

When a table is opened with `mode="r"` (or via `CTable.open()` without specifying
mode), all mutating operations raise immediately:

```python
t = b2.CTable.open("people")  # read-only

t.append((1, 50.0, True))  # ValueError: Table is read-only
t.extend([(1, 50.0, True)])  # ValueError: Table is read-only
t.delete(0)  # ValueError: Table is read-only
t.compact()  # ValueError: Table is read-only
```

All read operations work normally: `row[]`, column access, `head()`, `tail()`,
`where()`, `len()`, `info()`, `schema_dict()`.

---

## Filtering

`where()` applies a boolean expression and returns a read-only view:

```python
view = t.where(t["score"] > 50)
view = t.where((t["id"] > 10) & (t["active"] == True))
```

Views share `_cols` with the parent table and cannot be mutated (no `append` or
`extend`).

---

## Table info

```python
len(t)  # number of live rows
t.nrows  # same
t.ncols  # number of columns
t.col_names  # list of column names

t.info()  # prints a formatted summary with dtypes and memory usage
print(t)  # prints the first rows in a table format
```

---

## Schema introspection

```python
t.schema  # CompiledSchema object
t.column_schema("id")  # CompiledColumn for column "id"
t.schema_dict()  # JSON-compatible dict of the full schema
```

`schema_dict()` example output:

```python
{
    "version": 1,
    "row_cls": "Row",
    "columns": [
        {"name": "id", "kind": "int64", "ge": 0, "default": None},
        {"name": "score", "kind": "float64", "ge": 0, "le": 100, "default": 0.0},
        {"name": "active", "kind": "bool", "default": True},
    ],
}
```

The dict can be restored to a `CompiledSchema` without the original Python class:

```python
from blosc2.schema_compiler import schema_from_dict

restored = schema_from_dict(t.schema_dict())
```

---

## Memory and compression

```python
# Compressed size of all columns + valid_rows mask
cbytes = sum(col.cbytes for col in t._cols.values()) + t._valid_rows.cbytes

# Uncompressed size
nbytes = sum(col.nbytes for col in t._cols.values()) + t._valid_rows.nbytes

print(f"Compression ratio: {nbytes / cbytes:.2f}x")
```

---

## Complete example

```python
from dataclasses import dataclass
import numpy as np
import blosc2 as b2


@dataclass
class Measurement:
    sensor_id: int = b2.field(b2.int64(ge=0))
    value: float = b2.field(b2.float64(ge=-1000, le=1000), default=0.0)
    valid: bool = b2.field(b2.bool(), default=True)


# Create and populate (in-memory)
t = b2.CTable(Measurement, expected_size=10_000)
t.extend([(i, float(i % 200 - 100), i % 3 != 0) for i in range(5000)])

# Query
hot = t.where(t["value"] > 50)
print(f"Hot readings: {len(hot)}")

# Delete invalid
invalid_indices = [i for i in range(len(t)) if not t.row[i].valid[0]]
if invalid_indices:
    t.delete(invalid_indices)

# Inspect
t.info()
print(t.schema_dict())
```

## Persistency example

```python
from dataclasses import dataclass
import blosc2 as b2


@dataclass
class Measurement:
    sensor_id: int = b2.field(b2.int64(ge=0))
    value: float = b2.field(b2.float64(ge=-1000, le=1000), default=0.0)
    valid: bool = b2.field(b2.bool(), default=True)


# --- Session 1: create and populate ---
t = b2.CTable(Measurement, urlpath="sensors", mode="w", expected_size=100_000)
t.extend([(i, float(i % 200 - 100), i % 3 != 0) for i in range(50_000)])
print(f"Saved {len(t)} rows to disk")
# Table is automatically persisted — no explicit save() needed.

# --- Session 2: reopen and query ---
t = b2.CTable.open("sensors")  # read-only by default
hot = t.where(t["value"] > 50)
print(f"Hot readings: {len(hot)}")
arr = t["sensor_id"].to_numpy()
print(f"First 5 sensor IDs: {arr[:5]}")

# --- Session 3: reopen and append more data ---
t = b2.CTable(Measurement, urlpath="sensors", mode="a")
t.extend([(50_000 + i, float(i), True) for i in range(1_000)])
print(f"Total rows: {len(t)}")
```
