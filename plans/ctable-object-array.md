# CTable container/schema naming cleanup and variable-length scalar string support

## Motivation

The current CTable-related container naming mixes two different concerns:

1. **Physical storage layout**
   - `VLArray`
   - `BatchArray`
2. **Logical row semantics**
   - `ListArray`

This has made the design harder to reason about as requirements have become clearer.

In particular, the recent Parquet/OFF import work exposed an important missing concept:

- we need to store **long scalar strings/bytes efficiently**
- we want to avoid fixed-width `NDArray` string dtypes for wildly variable-length payloads
- we do **not** want to change the logical type from scalar string to `list[string]`

The current workaround promotes long scalar strings to `list<string>` and stores them in a `ListArray`, wrapping each scalar value as a singleton list:

```python
before = "...json string..."
after = ["...json string..."]
```

This preserves bytes but changes logical shape, which is a departure from the source Parquet schema.

The root issue is that we are missing a clear separation between:

- **storage primitives** (how bytes/objects are packed)
- **logical column kinds** (what one row value means)

Since these APIs are not yet publicly released, this is a good time to simplify the model.

---

## Design principle

Adopt a strict split between:

### 1. Physical containers
Public low-level containers should describe **how objects are physically packed/stored**.

### 2. Logical schema kinds
CTable schema specs should describe **what one row value is**.

### 3. Internal adapters
Any row-wise wrapper needed to adapt a physical container to CTable column semantics should remain internal.

This prevents proliferation of public container types whose only purpose is to bridge CTable behavior.

---

## Proposed naming model

## Physical layer: public containers

### `ObjectArray`
Rename current `VLArray` to `ObjectArray`.

Semantics:
- one logical object per entry
- one serialized object per chunk
- row/item-oriented access

Why rename:
- `VLArray` emphasizes variable-length encoding but not the real abstraction
- the real concept is: an array of Python/Blosc2 objects
- `ObjectArray` pairs naturally with `BatchArray`

### `BatchArray`
Keep `BatchArray`.

Semantics:
- one entry is one batch
- each batch contains many objects/items
- optimized for packing many items per chunk

This gives a clear public pair:

- `ObjectArray`: unbatched object storage
- `BatchArray`: batched object storage

---

## Logical/schema layer: public CTable specs

### `list(...)`
Represents a list-valued row cell.

### `vlstring(...)`
Represents a scalar variable-length string column.

### `vlbytes(...)`
Represents a scalar variable-length bytes column.

These are **logical column kinds**, not low-level container types.

This is the right place to expose:
- nullability
- string/bytes validation constraints
- batching hints for storage
- serializer choices if needed

---

## Internal implementation layer

`ListArray` should no longer be treated as a fundamental public abstraction.
It is better understood as an **internal adapter** implementing row-wise list-column semantics on top of a physical storage primitive.

Similarly, the new scalar variable-length string/bytes support should be implemented via an internal row-wise adapter over `BatchArray`, rather than introducing additional public container classes like:

- `VLStringArray`
- `VLBytesArray`
- `VLScalarArray`

Those names add public API surface without adding a new true storage primitive.

### Recommended internal direction

- keep or rename `ListArray` internally if desired
- add an internal scalar adapter for `vlstring` / `vlbytes`
- both can be backed by `BatchArray`

This keeps the public container list minimal while preserving a clean implementation.

---

## Why this is better

## 1. Cleaner mental model

Users and maintainers can reason as follows:

### Low-level storage choice
- one object per slot/chunk -> `ObjectArray`
- many objects per chunk -> `BatchArray`

### CTable schema choice
- row is a list -> `list(...)`
- row is a long scalar string -> `vlstring(...)`
- row is a long scalar bytes value -> `vlbytes(...)`

This is much clearer than mixing storage and row semantics in container class names.

## 2. Avoids singleton-list hacks

Long scalar strings will no longer need to be represented as `list<string>` just to get efficient batched storage.

Instead:
- logical type remains scalar string
- physical storage uses batched object packing internally

## 3. Minimal public surface

Public containers remain few and conceptually crisp:
- `ObjectArray`
- `BatchArray`

No need to add more public wrapper classes just for CTable internals.

## 4. Better future extensibility

If later we need other logical variable-length scalar kinds, they can be added at the schema layer without inventing new public containers.

---

## Scope decision

This proposal focuses on supporting:

- `vlstring`
- `vlbytes`

for CTable scalar columns.

It does **not** aim to make a fully generic nullable object column type public right now.
That broader design space can be revisited later if needed.

---

## High-level implementation strategy

## A. Rename physical container `VLArray` -> `ObjectArray`

### Goals
- make public physical container names consistent
- preserve the existing storage behavior of current `VLArray`

### Notes
- because this machinery is not publicly released, we do not need to optimize for compatibility
- internal tags/metadata may remain versioned in a way that allows reopening existing local test artifacts if useful, but compatibility is not the primary constraint

### Tasks
- rename `src/blosc2/vlarray.py` to `src/blosc2/objectarray.py` and update the class name to `ObjectArray`
- update all imports/references across the repo
- keep on-disk metadata tag as `vlarray` for now; rename later if desired

#### Recommendation
- public class name: `ObjectArray`
- rename module file to `objectarray.py` in Phase 1 alongside the class rename — keeping the file named `vlarray.py` while the class is `ObjectArray` creates a persistent split-brain that adds noise during all subsequent phases; the cost is low since the APIs are not yet public
- metadata tag can remain `vlarray` initially to minimize internal breakage, then be renamed later if desired

---

## B. Reframe `ListArray` as internal adapter

### Goals
- stop treating `ListArray` as a fundamental public storage primitive
- make it clear it is a row-wise list-column adapter

### Tasks
- keep implementation in place initially
- reduce public-facing emphasis on `ListArray`
- optionally rename internally later (not required for phase 1)

This can be done incrementally; there is no need to rename the implementation immediately if that creates churn.

---

## C. Add new schema specs: `vlstring`, `vlbytes`

## File
- `src/blosc2/schema.py`

### `vlstring`
Properties should likely include:
- `nullable: bool = False`
- `serializer: str = "msgpack"`
- `batch_rows: int | None = 2048`
- `items_per_block: int | None = None`

### `vlbytes`
Properties should likely include:
- `nullable: bool = False`
- `serializer: str = "msgpack"`
- `batch_rows: int | None = 2048`
- `items_per_block: int | None = None`

### Dropped fields: `min_length`, `max_length`, `pattern`, `storage`
- `min_length` / `max_length` / `pattern` are application-layer validation concerns, not storage schema concerns. They do not affect how data is physically stored and would make `vlstring`/`vlbytes` the only place in the blosc2 schema system that enforces runtime business-logic constraints. They can be added later if a concrete use case emerges.
- `storage` is premature: there is only one storage backend (`batch`) and no second option is defined. Adding the field now would require CTable code to guard on it from day one while ignoring it. Add it when a real alternative exists.
- Note: `max_len` is intentionally kept on `string()` where it has a clear, unambiguous meaning — it sizes the fixed-width NDArray dtype (e.g. `dtype='<U64'`). With `vlstring` as an explicit opt-in, `string(max_len=N)` and `vlstring()` have non-overlapping jobs and `max_len` is no longer doing double duty as a storage-backend selector.

### Important design note
These specs are **logical scalar** specs and should not use fixed-width NumPy dtypes.

Recommended:
- `dtype = None`
- `python_type = str` / `bytes`

This forces explicit branching in CTable internals and avoids accidentally flowing into fixed-width NDArray code paths.

### Metadata serialization
Add metadata kinds:
- `"vlstring"`
- `"vlbytes"`

Example:

```python
{
    "kind": "vlstring",
    "nullable": True,
    "serializer": "msgpack",
    "batch_rows": 2048,
    "items_per_block": null,
}
```

---

## D. Schema compiler support

## File
- `src/blosc2/schema_compiler.py`

### Tasks
- add `vlstring` and `vlbytes` to `_KIND_TO_SPEC`
- update `compute_display_width()`
- update annotation/spec compatibility rules
- keep plain `str -> string()` and `bytes -> bytes()` inference unchanged
- require `vlstring`/`vlbytes` to be requested explicitly via `blosc2.field(...)`

### Reasoning
Automatic inference should continue to produce the simpler fixed-width scalar types. The variable-length scalar types are a deliberate storage choice.

---

## E. Internal scalar adapter over `BatchArray`

## Goal
Provide row-wise scalar column semantics on top of `BatchArray`.

This adapter should remain **internal**, not a new public container concept.

### Behavior required by CTable
- `append(value)` appends one scalar row
- `extend(values)` appends many scalar rows
- `flush()` writes pending values as batches
- `__len__()` returns number of rows
- `__getitem__(int)` returns one scalar value
- `__getitem__(slice/list/array)` returns row values
- `__setitem__(int, value)` updates one persisted/pending row
- nullable support via native `None`

### Storage backend
Use `BatchArray` physically.

#### Physical representation
A persisted chunk should contain many scalar items, e.g.:

```python
["a", "bbbb", None, "ccc"]
```

not singleton lists.

### Suggested implementation shape
- small internal adapter class in a new module, e.g. `src/blosc2/_scalar_array.py`
- one implementation parameterized by `py_type` (`str` or `bytes`)
- avoid creating separate public classes unless later justified

### Pending-buffer strategy
The adapter maintains a `_pending: list` that accumulates rows before they are flushed to a `BatchArray` chunk:
- on `append(value)`: append to `_pending`; if `len(_pending) >= batch_rows`, flush immediately
- on `extend(values)`: extend `_pending` in segments of `batch_rows`, flushing each full segment
- on `flush()`: serialize and write the remaining `_pending` entries as one chunk, then clear `_pending`
- on `__setitem__(i, value)`: if row `i` is still in `_pending` (index >= flushed row count), update it in-place there; otherwise re-read, repack, and rewrite the persisted chunk that contains row `i`

### Validation/coercion
For string mode:
- allow `str`
- optionally coerce to `str`
- allow `None` only if nullable

For bytes mode:
- allow `bytes`, `bytearray`, `memoryview`, optionally `str` -> encode if desired
- allow `None` only if nullable

### Null handling
Use native `None` in persisted batch payloads.
Do not use scalar null sentinels for `vlstring` / `vlbytes`.

### msgpack and None serialization
The `BatchArray`'s msgpack codec must be configured with `raw=False` so that msgpack strings round-trip as `str` (not `bytes`). Verify that `None` passes through the codec unchanged, since it is the native null representation for nullable columns. This is a common footgun with msgpack defaults.

---

## F. CTable storage backend support

## File
- `src/blosc2/ctable_storage.py`

### Current state
Storage knows how to create/open:
- NDArray scalar columns
- list columns

### Needed changes
Add support for scalar varlen string/bytes columns, e.g.:
- `create_varlen_scalar_column(...)`
- `open_varlen_scalar_column(...)`

The exact public/internal function names can be decided during implementation.

### Persistent layout
We can reuse the same file style as current list columns (`.b2b`).

Example:
- `/_cols/ingredients.b2b`

### Metadata tag
Tag the underlying stored object so reopen logic knows this `BatchArray` is serving a scalar varlen CTable column role.

Recommended additional metadata key:

```python
meta["vlscalar"] = {
    "version": 1,
    "py_type": "str",  # or "bytes"
    "nullable": True,
    "batch_rows": 2048,
}
```

Even if the public name is not `VLScalarArray`, `vlscalar` is an acceptable **internal metadata role name**. Alternatively, we may prefer a more direct tag like `vltext` or `ctable_varlen_scalar`. This can be finalized during implementation.

### Recommendation
Use a neutral internal role tag, for example:

```python
meta["ctable_varlen_scalar"] = {...}
```

This avoids tying metadata too strongly to a class name we do not want to expose publicly.

---

## G. CTable core integration

## File
- `src/blosc2/ctable.py`

This is the largest integration area.

### New spec/category checks
Today the code distinguishes mainly between:
- scalar NDArray-backed columns
- `ListSpec` columns

We need explicit branching for:
- fixed-width scalar columns
- list columns
- variable-length scalar string/bytes columns

Suggested helpers:
- `_is_list_column(col)`
- `_is_varlen_scalar_column(col)`
- `_is_varlen_column(col)` (optional shared helper)

### Query/expression guard
`vlstring` and `vlbytes` columns are **not queryable in phase 1**. Any code path that feeds a column into a lazy expression or index sidecar must raise `NotImplementedError` with a clear message when the column spec is `vlstring` or `vlbytes`. Without this explicit guard, partial code paths will silently produce wrong results (e.g. treating the column as a fixed-width NDArray).

### Areas to update
At minimum:
- nullable spec resolution
- column initialization
- grow logic
- flush logic
- open/load/save logic
- append / extend
- row coercion
- copy / compact / sort paths
- `Column.__getitem__`
- `Column.__setitem__`
- `Column.__iter__`
- Arrow schema export
- Arrow batch export
- Arrow import / Parquet import

### Null policy interaction
`vlstring` and `vlbytes` should bypass scalar sentinel logic.
Nullability is represented natively with `None`.

### Row coercion
Current scalar coercion path does:

```python
np.array(val, dtype=col.dtype).item()
```

That must not be used for `vlstring` / `vlbytes` because `dtype=None` and native nulls are expected.

### Grow behavior
Varlen scalar columns should behave like list columns:
- no NDArray resize needed
- only `_valid_rows` grows

### Flush behavior
A common flush helper should flush:
- list-backed columns
- varlen scalar columns

---

## H. Arrow / Parquet support

## Goal
Long scalar strings/bytes should round-trip as scalar Arrow/Parquet columns, not singleton lists.

## Export
When exporting:
- `vlstring` -> Arrow `string`
- `vlbytes` -> Arrow `large_binary`
- preserve `None` as Arrow nulls

## Import
Update Arrow/Parquet type inference so scalar string/bytes columns can map to:
- fixed-width `string` / `bytes` for short values
- `vlstring` / `vlbytes` for long values

### Important consequence
The OFF importer (`off/parquet-to-blosc2.py`) should stop promoting long scalar strings to `list<string>`.

Instead:
- keep the column logically scalar
- import it as `vlstring`
- store it physically through the new batched scalar mechanism

This fixes the current problem where:

```python
before = "json-string"
after = ["json-string"]
```

---

## I. Query/expression/indexing support

## Recommendation for phase 1
Support first:
- storage
- append/extend
- row access
- copy/load/save
- Arrow/Parquet import/export

Do **not** aim initially for full support in:
- lazy expressions over `vlstring`/`vlbytes`
- CTable indexing sidecars on these columns
- advanced vectorized operations

### Why
These features largely assume NDArray-like scalar columns and can be added later if needed.

### Sorting
Sorting may be implemented later either by:
- materializing values to Python lists / object arrays
- or adding targeted support

It does not need to block the initial storage redesign.

---

## J. Tests to add/update

## New tests
- schema round-trip for `vlstring` / `vlbytes`
- append/get/set/extend/flush behavior
- null handling with native `None`
- save/load persistent CTable containing `vlstring` / `vlbytes`
- Arrow export produces scalar string/binary columns
- Parquet import/export round-trips long scalar strings without singleton-list wrapping
- OFF-style regression test for `ingredients`

## Existing tests to revise
- any tests assuming long imported strings become `list<string>`
- list-column tests that may now share helper paths with varlen scalar columns

---

## K. Suggested implementation phases

## Phase 1: naming cleanup foundation
1. Rename public `VLArray` -> `ObjectArray`
2. Rename module file `vlarray.py` -> `objectarray.py`
3. Update all imports and references across the repo
4. Leave `BatchArray` unchanged
5. Keep `ListArray` working, but treat it as internal implementation machinery

## Phase 2: schema and storage primitives
1. Add `vlstring` / `vlbytes` specs
2. Add schema compiler support
3. Add internal scalar adapter over `BatchArray`
4. Add storage backend support for opening/creating these columns

## Phase 3: CTable integration
1. Add varlen scalar branching throughout `ctable.py`
2. Support append/extend/load/save/copy/basic access
3. Add flush handling and null handling

## Phase 4: Arrow/Parquet integration
1. Export `vlstring`/`vlbytes` as scalar Arrow columns
2. Import long scalar strings/bytes as `vlstring`/`vlbytes`
3. Update `off/parquet-to-blosc2.py` to stop wrapping long strings as singleton lists

## Phase 5: cleanup and polish
1. Update docs/examples/tests
2. Review whether any internal adapter renames are worthwhile
3. Reassess whether advanced query/index support is needed

---

## Open questions

### 1. Public name of the renamed `VLArray`
Recommendation: `ObjectArray`

Alternative possibilities:
- `ItemArray`
- `PackedObjectArray`

`ObjectArray` is the clearest.

### 2. File/module renaming
Rename `vlarray.py` -> `objectarray.py` in Phase 1 alongside the class rename.
Keeping the filename as `vlarray.py` while the exported class is `ObjectArray` creates a persistent split-brain that adds cognitive noise throughout all subsequent phases. The cost is low since the APIs are not yet public.

### 3. Internal metadata role name for varlen scalar column wrappers
Needs a final decision.

Recommendation:
- use a neutral internal key like `ctable_varlen_scalar`

### 4. Whether `vlbytes` should decode/encode `str` automatically
This is a policy choice.

Recommendation:
- keep behavior conservative initially
- accept bytes-like inputs explicitly
- only coerce from `str` if there is a strong existing precedent

### 5. Whether `serializer="arrow"` should be supported initially
Recommendation:
- start with `msgpack`
- add Arrow serializer support only if there is a demonstrated benefit

### 6. `min_length`, `max_length`, `pattern` on `vlstring`/`vlbytes`
Dropped from phase 1. These are application-layer validation concerns, not storage schema concerns. They can be revisited if a concrete use case is identified.

### 7. `storage` field on `vlstring`/`vlbytes`
Dropped from phase 1. There is only one storage backend (`batch`) and no second option is defined. Add when a real alternative exists.

---

## Final recommendation

Adopt the following conceptual model:

### Public physical containers
- `ObjectArray` (renamed from `VLArray`)
- `BatchArray`

### Public logical CTable specs
- `list(...)`
- `vlstring(...)`
- `vlbytes(...)`

### Internal adapters
- list-column adapter over physical containers
- varlen scalar string/bytes adapter over `BatchArray`

This gives a much cleaner and more scalable design than the current mix of storage and row-semantics names, and it directly solves the long-string import problem without distorting scalar columns into singleton lists.
