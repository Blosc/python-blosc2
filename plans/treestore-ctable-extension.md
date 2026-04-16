# TreeStore Root-Level `CTable` Extension Plan

## Goal

Allow a `CTable` stored as the sole logical object inside a `TreeStore` to be
opened directly via:

```python
table = blosc2.open(urlpath)
```

That is, if a `TreeStore` at `urlpath` carries a recognized root manifest for
`CTable`, `blosc2.open(urlpath)` should return a `CTable` instance instead of a
raw `TreeStore`.

This plan intentionally covers only the simple first round:

- one `CTable` per `TreeStore`
- object root is the store root
- `/_meta` at store root is the manifest

Subtree object roots and multiple tables per store are deferred.

## Background

`TreeStore` now has persistent low-level container metadata through:

- `storage.meta["b2tree"] = {"version": 1}`

That is enough for `blosc2.open()` to recognize the path as a `TreeStore`, but
not enough to know whether the store should materialize as a richer object.

The generic extension contract in [tree_store_extensions.md](/Users/faltet/blosc/python-blosc2/tree_store_extensions.md)
introduces:

- `/_meta` as the logical-object manifest for store-backed objects

This plan applies that contract to `CTable`.

## Storage Layout

The persisted root-level `CTable` layout should be:

- `/_meta`
- `/_valid_rows`
- `/_cols/<name>`

Example:

- `/_meta`
- `/_valid_rows`
- `/_cols/id`
- `/_cols/score`
- `/_cols/active`

Rationale:

- `/_meta` stores logical-object manifest data
- `/_valid_rows` stores real row-visibility data
- `/_cols/<name>` stores one persisted column array per field

## Root Manifest

`/_meta` should be a small persisted `SChunk` used primarily through `vlmeta`.

Initial required manifest fields:

- `kind`
- `version`
- `schema`

Initial `CTable` manifest:

```python
{
    "kind": "ctable",
    "version": 1,
    "schema": {...},
}
```

Recommended concrete writes:

```python
tstore["/_meta"].vlmeta["kind"] = "ctable"
tstore["/_meta"].vlmeta["version"] = 1
tstore["/_meta"].vlmeta["schema"] = schema_payload
```

## Schema Persistence Format

The schema should be stored in:

- `/_meta.vlmeta["schema"]`

The schema document should be JSON-compatible, explicit, and versioned.

Recommended shape:

```python
{
    "version": 1,
    "columns": [
        {
            "name": "id",
            "py_type": "int",
            "spec": {"kind": "int64", "ge": 0},
            "default": None,
        },
        {
            "name": "score",
            "py_type": "float",
            "spec": {"kind": "float64", "ge": 0, "le": 100},
            "default": None,
        },
        {
            "name": "active",
            "py_type": "bool",
            "spec": {"kind": "bool"},
            "default": True,
        },
    ],
}
```

Notes:

- `columns` must be an ordered list, not a dict
- column order comes from the schema list
- `TreeStore` iteration order must not be used as schema authority

For the first version, do not duplicate data that can be inspected from the
stored column arrays:

- per-column `cparams`
- per-column `dparams`
- chunk/block layout
- `expected_size`
- compaction settings

## `_valid_rows` Persistence

`/_valid_rows` should be a normal persisted boolean array.

This is correct because:

- it is table data, not metadata
- it may grow large
- it participates in normal row visibility semantics

It should not be folded into `/_meta`.

## Column Persistence

Each column should be stored as its own persisted array under:

- `/_cols/<name>`

This keeps the physical layout aligned with the internal columnar design and
lets per-column storage details remain attached to the actual persisted array.

## Constructor Semantics

The intended public constructor remains:

```python
table = blosc2.CTable(
    Row,
    urlpath=None,
    mode="a",
    expected_size=1_048_576,
    compact=False,
    validate=True,
)
```

For the persistent path:

- `urlpath is None`:
  - in-memory `CTable`
- `urlpath is not None`:
  - root-level `CTable` persisted on top of a `TreeStore`

Recommended mode behavior:

- `mode="w"`:
  - create a fresh store-root `CTable`
- `mode="a"`:
  - open existing or create new
- `mode="r"`:
  - open existing read-only

## `blosc2.open()` Materialization

The root-level dispatch behavior should be:

1. `blosc2.open(urlpath)` detects a `TreeStore`
2. it opens the `TreeStore`
3. it checks for `/_meta`
4. if `/_meta.vlmeta["kind"] == "ctable"`, it materializes `CTable`
5. otherwise it returns the raw `TreeStore`

This preserves the current open layering:

- first detect the low-level container
- then optionally materialize a richer object

## Suggested Implementation Shape

### Step 1: Add Root Manifest Helpers

Add private helper(s) for root-manifest probing, e.g.:

- `_open_treestore_root_object(store)`
- `_read_treestore_root_manifest(store)`

Responsibilities:

- check whether `/_meta` exists
- open `/_meta`
- validate that it is an `SChunk`
- read `kind` / `version`
- return a manifest payload suitable for dispatch

### Step 2: Extend `blosc2.open()`

In the special-store open path:

- if opening yields a `TreeStore`
- probe the root manifest
- if recognized as `ctable`, return `CTable.open(...)` or equivalent internal
  constructor
- otherwise return the `TreeStore`

This logic should be localized so the generic `open()` path remains easy to
follow.

### Step 3: Add `CTable` Root-Manifest Read/Write Helpers

In the `CTable` persistence layer, add helpers for:

- creating `/_meta`
- writing `kind`
- writing `version`
- writing `schema`
- reading and validating the root manifest

This should be the only place that knows the `CTable` manifest schema.

### Step 4: Wire Creation

When a persistent `CTable` is created:

- create/open the backing `TreeStore`
- create `/_meta`
- write the root manifest
- create `/_valid_rows`
- create `/_cols/<name>` arrays

### Step 5: Wire Reopen

When a persistent `CTable` is reopened:

- read `/_meta.vlmeta["schema"]`
- rebuild the compiled schema
- reopen `/_valid_rows`
- reopen each persisted column from `/_cols/<name>`

### Step 6: Keep Internal Names Reserved

Validation should reject user column names that collide with internal names:

- `_meta`
- `_valid_rows`
- `_cols`

This already aligns with the existing schema compiler reserved-name logic.

## Validation Rules

For `CTable` root-manifest detection:

- if `/_meta` does not exist:
  - not a persisted `CTable`
- if `/_meta` exists but is malformed:
  - raise clear error on attempted `CTable` materialization
- if `kind != "ctable"`:
  - return raw `TreeStore`
- if `kind == "ctable"` but required fields are missing:
  - raise clear error

Recommended required fields for version 1:

- `kind`
- `version`
- `schema`

## Deferred Scope

This plan intentionally does not cover:

- multiple `CTable` objects in one `TreeStore`
- subtree object roots such as `/users/_meta`
- automatic materialization when indexing a subtree from `TreeStore`
- `Ref` support for store-subtree logical objects
- schema evolution beyond append-only behavior

These should be handled in later phases after the root-level path is stable.

## Tests

Add coverage for:

- create persistent root-level `CTable`
- reopen via `blosc2.open(urlpath)` and get `CTable`
- reopen via `CTable.open(urlpath, mode="r")`
- root manifest present and schema readable from `/_meta.vlmeta`
- store with no `/_meta` still opens as raw `TreeStore`
- store with unknown root-manifest `kind` still opens as raw `TreeStore`
- malformed `CTable` manifest raises clear error
- append rows after reopen
- read-only reopen rejects writes

## Recommended Implementation Order

1. write root-manifest probe helpers for `TreeStore`
2. extend `blosc2.open()` with root-manifest dispatch
3. add `CTable` manifest read/write helpers
4. wire persistent create/open around the manifest
5. add tests for dispatch and round-trip

## Summary

The first `TreeStore` extension should treat root `/_meta` as the logical
manifest for the whole store.

For `CTable`, this yields a simple and coherent open story:

- low-level metadata says "this is a `TreeStore`"
- root `/_meta` says "this store materializes as a `CTable`"
- `blosc2.open(urlpath)` returns the richer object directly

This keeps the first implementation small while staying compatible with a later
generalization to subtree object roots.
