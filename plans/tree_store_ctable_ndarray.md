# TreeStore containing NDArrays and CTables

## Goal

Allow a persistent `TreeStore` to contain both ordinary Blosc2 leaves, such as
`NDArray`, and higher-level objects, specifically `CTable`, while keeping the
public API simple:

```python
with blosc2.TreeStore("bundle.b2z", mode="w") as ts:
    ts["/x"] = blosc2.arange(10)
    ts["/table"] = table

with blosc2.open("bundle.b2z", mode="r") as ts:
    x = ts["/x"]  # NDArray
    table = ts["/table"]  # CTable
```

The preferred design is **inline CTable subtree storage**, not nested `.b2z`
files.  This avoids the ZIP-inside-ZIP problem for read-only `.b2z` bundles and
keeps all CTable components directly addressable as normal Blosc2 frame leaves
inside the outer store.

## Non-goals for the first iteration

- Full recursive `DictStore` / `TreeStore` values.
- Arbitrary Python object storage.
- Public APIs like `CTable.save_to_store()` or `CTable.open_from_store()`.
- Zero-copy linking to an external CTable path.  Assignment should copy/materialize
  the CTable into the destination TreeStore subtree.

These may be considered later after the CTable use case is stable.

## Current situation

`CTable` is already persisted internally as a `TreeStore` with a layout like:

```text
table.b2d or table.b2z
  /_meta
  /_valid_rows
  /_cols/<column>
  /_indexes/...
```

The `/_meta` SChunk has metadata such as:

```python
meta.vlmeta["kind"] = "ctable"
meta.vlmeta["version"] = 1
meta.vlmeta["schema"] = ...
```

`blosc2.open("table.b2z")` already detects this root manifest and returns a
`CTable` instead of a raw `TreeStore`.

However, `TreeStore.__setitem__()` currently only supports array-like values via
`DictStore.__setitem__()`.  It does not support assigning a `CTable` as a leaf,
and if CTable internals were placed below `/table`, `TreeStore.__getitem__()`
would currently return a subtree view for `/table`, not a `CTable`.

## Proposed physical layout

Store CTable internals inline below the assigned key:

```text
bundle.b2z
  embed.b2e
  x.b2nd
  table/_meta.b2f
  table/_valid_rows.b2nd
  table/_cols/name.b2nd
  table/_cols/age.b2nd
  table/_indexes/...
```

For a directory-backed outer store:

```text
bundle.b2d/
  embed.b2e
  x.b2nd
  table/
    _meta.b2f
    _valid_rows.b2nd
    _cols/
      name.b2nd
      age.b2nd
    _indexes/
      ...
```

From the outer `TreeStore` point of view, `/table` is an object root.  Its
internal paths are implementation details.

## Proposed public semantics

### Write

```python
ts["/table"] = ctable
```

Materializes `ctable` into the subtree rooted at `/table` and registers `/table`
as a CTable object root.

### Read

```python
table = ts["/table"]
```

Returns a `CTable` when `/table` is registered as an object root or when
`/table/_meta` declares `kind == "ctable"`.

### Traversal

Normal high-level traversal should treat `/table` as one object:

```python
sorted(ts.keys())
# ['/table', '/x']
```

not:

```python
["/table/_meta", "/table/_valid_rows", "/table/_cols/name", ...]
```

A raw/internal view can be considered later if needed.

## Metadata design

Use two metadata layers:

1. **CTable internal manifest** at `/table/_meta`.
   - This remains authoritative for opening the CTable.
   - It already contains `kind == "ctable"`, version and schema.

2. **TreeStore object registry** in TreeStore-level metadata.
   - Used for efficient object-boundary detection and for hiding internals in
     `keys()`, `items()`, `walk()`, deletion and conflict checks.

Suggested TreeStore-level registry:

```python
tstore.vlmeta["objects"] = {
    "/table": {
        "kind": "ctable",
        "version": 1,
        "layout": "inline-tree-subtree",
    }
}
```

The registry is a convenience index.  If missing, TreeStore should be able to
fall back to probing `/table/_meta` for backward compatibility and robustness.

## Internal protocol

Do not expose public `CTable.save_to_store()` / `CTable.open_from_store()` APIs
initially.  Instead, implement private/internal hooks.

Possible hooks:

```text
CTable._save_to_treestore(store: TreeStore, key: str) -> None
CTable._open_from_treestore(store: TreeStore, key: str) -> CTable
```

or a more generic private object protocol later:

```python
obj.__blosc2_store_into__(store, key)
```

For the first implementation, CTable-specific private methods are likely simpler.

## Required implementation pieces

### 1. `TreeStoreTableStorage`

Add a new `TableStorage` backend in `src/blosc2/ctable_storage.py`:

```python
class TreeStoreTableStorage(TableStorage):
    def __init__(
        self,
        store: blosc2.TreeStore,
        root_key: str,
        mode: str,
        owns_store: bool = False,
    ): ...
```

It maps CTable logical storage keys:

```text
/_meta
/_valid_rows
/_cols/<name>
/_indexes/...
```

onto outer TreeStore keys/paths:

```text
<table-root>/_meta
<table-root>/_valid_rows
<table-root>/_cols/<name>
<table-root>/_indexes/...
```

For example:

```text
_table_key("/_meta")       -> "/table/_meta"
_table_key("/_valid_rows") -> "/table/_valid_rows"
_table_key("/_cols/x")     -> "/table/_cols/x"
```

It should implement the same `TableStorage` interface currently implemented by
`FileTableStorage`:

- `create_column()`
- `open_column()`
- `create_list_column()`
- `open_list_column()`
- `create_varlen_scalar_column()`
- `open_varlen_scalar_column()`
- `create_valid_rows()`
- `open_valid_rows()`
- `save_schema()`
- `load_schema()`
- `check_kind()`
- `table_exists()`
- `is_read_only()`
- `open_mode()`
- `delete_column()`
- `rename_column()`
- `close()`
- `discard()`
- index catalog / epoch helpers
- `index_anchor_path()`

Important lifecycle rule:

- If the backend is created from an existing outer `TreeStore`, it should not
  close/discard the outer store unless it explicitly owns it.
- Use `owns_store=False` for tables returned by `ts["/table"]`.

### 2. Refactor CTable open/save around `TableStorage`

Currently `CTable.__init__()`, `CTable.open()` and `CTable.save()` are strongly
oriented around either `FileTableStorage(urlpath, mode)` or `InMemoryTableStorage()`.

Add private helper paths so any `TableStorage` implementation can be used:

```text
CTable._open_from_storage(storage: TableStorage) -> CTable
CTable._save_to_storage(storage: TableStorage) -> None
```

Then:

- `CTable.open(urlpath, mode="r")` uses `FileTableStorage` and calls
  `_open_from_storage()`.
- `CTable.save(urlpath, overwrite=False)` uses `FileTableStorage` and calls
  `_save_to_storage()`.
- `CTable._open_from_treestore(store, key)` uses `TreeStoreTableStorage` and
  calls `_open_from_storage()`.
- `CTable._save_to_treestore(store, key)` uses `TreeStoreTableStorage` and
  calls `_save_to_storage()`.

This should reduce duplication and keep the public API unchanged.

### 3. TreeStore object registry helpers

Add private helpers in `src/blosc2/tree_store.py`:

```python
def _normalize_object_key(self, key: str) -> str: ...
def _objects_metadata(self) -> dict: ...
def _register_object(
    self, key: str, *, kind: str, version: int, layout: str
) -> None: ...
def _unregister_object(self, key: str) -> None: ...
def _object_info(self, key: str) -> dict | None: ...
def _object_roots(self) -> set[str]: ...
```

Fallback probing helper:

```text
def _probe_object_info(self, key: str) -> dict | None:
    # Look for key + "/_meta" and inspect vlmeta["kind"].
```

The registry should probably live in the TreeStore root vlmeta.  Subtree views
need to translate object keys correctly between full and subtree-relative paths.

### 4. TreeStore assignment integration

In `TreeStore.__setitem__()` before falling back to `DictStore.__setitem__()`:

```python
if isinstance(value, blosc2.CTable):
    self._set_ctable_object(key, value)
    return
```

`_set_ctable_object()` should:

1. Validate key and structural conflicts.
2. Reject assigning inside an existing object subtree unless this is an internal
   write performed by CTable storage.
3. Delete/overwrite an existing object root if overwrite semantics are allowed,
   or raise if the key exists.  This must be consistent with existing TreeStore
   assignment behavior.
4. Materialize the CTable into `key` via `CTable._save_to_treestore()`.
5. Register object metadata:

   ```python
   self._register_object(key, kind="ctable", version=1, layout="inline-tree-subtree")
   ```

Need an internal bypass flag/mechanism so `TreeStoreTableStorage` can write
`/table/_meta`, `/table/_cols/x`, etc. without being blocked by object-boundary
protection.

Possible approaches:

- `DictStore.__setitem__()` direct calls from `TreeStoreTableStorage` after full
  key translation.
- A private `TreeStore._set_internal(key, value)` method.
- A context manager `with store._raw_object_write(): ...`.

Prefer a small private method so the bypass is explicit and limited.

### 5. TreeStore retrieval integration

In `TreeStore.__getitem__()` after key validation and before returning subtree
views:

```python
info = self._object_info(key) or self._probe_object_info(key)
if info is not None:
    if info["kind"] == "ctable":
        return blosc2.CTable._open_from_treestore(self, key)
```

This ensures:

```python
ts["/table"]
```

returns `CTable` instead of a raw subtree.

### 6. Object-boundary protection

Prevent accidental user mutation of CTable internals through the outer TreeStore:

```python
ts["/table/_cols/x"] = arr  # should raise by default
del ts["/table/_meta"]  # should raise by default
```

Allowed operations:

```python
ts["/table"] = new_ctable  # replace whole object, if overwrite semantics permit
del ts["/table"]  # delete whole object subtree and registry entry
table = ts["/table"]  # object access
```

Private CTable storage code must be able to bypass this protection.

### 7. Collapse object internals in traversal

Update high-level `TreeStore` methods so object roots are treated as leaves:

- `keys()`
- `items()`
- `values()` if present/added
- `walk()`
- `get_children()`
- `get_descendants()`
- `get_subtree()` behavior around object roots

Suggested behavior:

- Include object root key, e.g. `/table`.
- Exclude descendants under registered object roots from normal high-level
  traversal.
- If a user asks for `get_subtree("/table")`, either:
  - return the CTable via `__getitem__()` and document that object roots are not
    normal subtrees, or
  - add an explicit raw/internal method later.

Avoid adding public raw APIs in the first iteration unless tests or development
needs require it.

### 8. Deletion semantics

`del ts["/table"]` should:

1. Detect `/table` as object root.
2. Delete all physical keys/files under `/table/...`.
3. Remove object registry entry.
4. Mark store modified.

Deleting normal subtrees should also remove object registry entries for any
objects inside the deleted subtree.

### 9. `.b2z` read-only behavior

This design avoids nested ZIPs.  For fixed-width columns and metadata, read-only
outer `.b2z` access can continue to use zip offsets for cframe leaves.

For list/varlen columns and index sidecars, mirror the existing
`FileTableStorage` logic:

- `.b2b` leaves can be opened by offset from the outer `.b2z` because they are
  Blosc2 cframes.
- Index sidecars that need filesystem paths may need extraction into the outer
  store working directory, as current `FileTableStorage` already does for
  `.b2z` tables.

### 10. Index catalog handling

`TreeStoreTableStorage` should store index sidecar paths consistently relative
to the outer store working directory, e.g.:

```text
table/_indexes/<col>/...
```

Then `DictStore.to_b2z()` naturally packs them into the outer `.b2z`.

Carefully port/adapt from `FileTableStorage`:

- `_walk_descriptor_paths()`
- `_relativize_descriptor()`
- `_absolutize_descriptor()`
- `_ensure_index_files_extracted()`
- `load_index_catalog()`
- `save_index_catalog()`
- `index_anchor_path()`

## Limitations of the design

- This addresses CTable-as-object, not arbitrary recursive stores.
- Object internals are physically present in the TreeStore and must be protected.
- High-level traversal becomes semantic rather than purely physical.
- Multiple mutable handles to the same inline CTable may conflict unless handled
  with caching or documented as unsupported.
- Assigning an in-memory CTable copies/materializes all columns.
- Assigning a persistent CTable should copy contents, not link to its source.
- Registry metadata and `/table/_meta` can get out of sync if manually edited;
  `/table/_meta` should remain authoritative for opening.
- Mutation of inline CTables inside append-mode outer `.b2z` requires careful
  flush/close ordering.

## Suggested implementation phases

### Phase 1: Storage refactor only

- Add `CTable._open_from_storage()`.
- Add `CTable._save_to_storage()`.
- Update existing `CTable.open()` / `save()` to use the helpers.
- Ensure all current CTable tests pass unchanged.

### Phase 2: Add `TreeStoreTableStorage`

- Implement the backend.
- Add private `CTable._save_to_treestore()` and `_open_from_treestore()`.
- Add focused tests using private methods initially if necessary.

### Phase 3: TreeStore object registry and dispatch

- Add object registry metadata helpers.
- Add `TreeStore.__setitem__()` support for `CTable`.
- Add `TreeStore.__getitem__()` dispatch to return `CTable` for object roots.

### Phase 4: Object-boundary traversal and deletion

- Hide object internals from `keys()`, `items()`, `walk()`, etc.
- Protect internals from direct mutation.
- Implement whole-object deletion.

### Phase 5: Full CTable feature coverage

Add/verify tests for:

- fixed-width columns
- list columns
- varlen scalar columns
- computed/materialized column metadata
- index catalogs and sidecars
- read-only `.b2z` bundles
- append-mode `.b2d` bundles
- append-mode `.b2z` bundles

## Test plan

### Basic TreeStore with NDArray and CTable

Parametrize over outer format `b2d` / `b2z`:

```python
with blosc2.TreeStore(path, mode="w") as ts:
    ts["/x"] = blosc2.arange(10)
    ts["/table"] = ctable

with blosc2.open(path, mode="r") as ts:
    assert isinstance(ts["/x"], blosc2.NDArray)
    assert isinstance(ts["/table"], blosc2.CTable)
```

### Traversal hides internals

```python
assert "/table" in ts.keys()
assert "/table/_meta" not in ts.keys()
assert not any(k.startswith("/table/_cols") for k in ts.keys())
```

### Raw physical persistence

For debugging-level checks, inspect the filesystem/zip entries and confirm
physical internals exist:

```text
table/_meta.b2f
table/_valid_rows.b2nd
table/_cols/...
```

### Structural conflict tests

```python
ts["/table"] = ctable
with pytest.raises(ValueError):
    ts["/table/_cols/x"] = arr
```

and reverse conflict:

```python
ts["/table/foo"] = arr
with pytest.raises(ValueError):
    ts["/table"] = ctable
```

### Deletion

```python
del ts["/table"]
assert "/table" not in ts
# assert no physical table/* entries remain after reopen
```

### CTable feature tests

- simple schema with numeric/string columns
- list columns
- nullable/varlen scalar columns
- indexes if applicable
- append/read after reopen

## Decisions on initially open questions

### Replacing an existing object root

Do **not** allow implicit replacement.  If `/table` already exists as an object
root, then:

```python
ts["/table"] = new_table
```

should raise.  Users must delete explicitly first:

```python
del ts["/table"]
ts["/table"] = new_table
```

Rationale: replacing a CTable subtree is destructive and can involve many
physical leaves.  Requiring an explicit delete avoids accidental data loss and
simplifies consistency handling.

### `get_subtree()` on object roots

`ts.get_subtree("/table")` should raise by default:

```python
ValueError("'/table' is a CTable object root, not a TreeStore subtree")
```

Use:

```python
ts["/table"]
```

to retrieve the `CTable` object.  Returning a raw subtree would expose internals;
returning a `CTable` from `get_subtree()` would make the method misleading.

### Public raw/internal inspection API

Do **not** add a public raw/internal API initially.  Keep object internals
private for the first implementation.  If a real debugging or advanced-use need
appears later, consider an explicit API such as:

```python
ts.get_subtree("/table", raw=True)
```

or:

```python
ts.get_object_storage("/table")
```

Avoid exposing this too early so the inline object layout can still evolve.

### Caching object handles

Do **not** cache returned object handles initially.  Multiple read-only handles
are fine.  Multiple mutable handles to the same inline object should be
documented as unsupported initially.

A weakref cache or writable-handle guard can be added later if practical issues
show up.

### Write ordering and close semantics

Returned inline CTable handles should be non-owning with respect to the outer
`TreeStore`, but the outer store should track inline handles it created so close
ordering is safe.

Recommended behavior:

- `TreeStore.__getitem__("/table")` returns a `CTable` backed by the outer store.
- The outer `TreeStore` keeps a private weak set/list of inline object handles it
  opened.
- `TreeStore.close()` closes any still-open inline object handles before packing
  an append/write-mode `.b2z` outer store.
- Then the outer store repacks as usual.

This makes the following safe:

```python
with blosc2.TreeStore("bundle.b2z", mode="a") as ts:
    table = ts["/table"]
    table.append(...)
# TreeStore.close() closes table first, then repacks bundle.b2z
```

Explicitly closing the table remains fine:

```python
with blosc2.TreeStore("bundle.b2d", mode="a") as ts:
    table = ts["/table"]
    table.append(...)
    table.close()
```
