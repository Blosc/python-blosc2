# TreeStore Extension Objects

## Goal

Define a general mechanism for representing richer logical objects on top of a
`TreeStore`, while keeping the underlying persisted container recognizable as a
plain `TreeStore`.

The initial driver is persisted `CTable`, but the mechanism should be generic
enough to support other logical object kinds later.

## Core Idea

`TreeStore` already has a low-level container identity:

- `storage.meta["b2tree"] = {"version": 1}`

That answers:

- "what physical container is stored here?"

However, a `TreeStore` can also act as a substrate for a richer logical object.
For that, we introduce a reserved manifest entry:

- `<object_root>/_meta`

This answers:

- "what logical object is represented by this store subtree?"

The manifest is a small persisted `SChunk` whose `vlmeta` is the source of
truth for logical-object identity and configuration.

## Object Root Model

An object root is any `TreeStore` subtree that contains:

- `<object_root>/_meta`

Examples:

- whole-store object:
  - `/_meta`
- subtree object:
  - `/users/_meta`
  - `/orders/_meta`

This gives one uniform rule:

- if a subtree has `/_meta`, it may represent a richer logical object

The whole-store case is just the special case where the object root is the
store root.

## Why Use `/_meta`

### Separation Of Roles

- container `.meta` remains about the low-level container type (`b2tree`)
- `/_meta` is about higher-level logical identity
- user-facing `tstore.vlmeta` remains available for user metadata

### Mutable Object Metadata

Unlike fixed container metalayers, `/_meta.vlmeta` can evolve over time.
That matters for store-backed logical objects that may need mutable metadata,
such as:

- schema evolution state
- object versioning
- feature flags
- migration markers

### Generic Store Extension Point

`/_meta` should not be a one-off `CTable` special case.
It should be the general manifest contract for any richer object represented on
top of a store subtree.

## Manifest Representation

`<object_root>/_meta` should be:

- a small persisted `SChunk`
- primarily used through `vlmeta`

The initial required fields in `/_meta.vlmeta` are:

- `kind`
- `version`

Example:

```python
tree_store["/_meta"].vlmeta["kind"] = "ctable"
tree_store["/_meta"].vlmeta["version"] = 1
```

Additional fields are object-kind-specific.

For example, a `CTable` manifest may add:

- `schema`

## Reserved Internal Names

Within an object root, the following path is reserved:

- `<object_root>/_meta`

Logical objects may reserve additional internal paths under the same root.

For example, `CTable` is expected to reserve:

- `<object_root>/_valid_rows`
- `<object_root>/_cols`

These reserved names are internal implementation detail and must not be treated
as user data nodes.

## `blosc2.open()` Contract

When opening a persisted path:

1. low-level store detection happens first
2. if the opened object is a `TreeStore`, object-manifest detection may happen
3. if a recognized manifest is found, materialize the richer logical object
4. otherwise, return the raw `TreeStore`

For the whole-store case, the detection rule is:

- open the path as a `TreeStore`
- look for `/_meta`
- if `/_meta.vlmeta["kind"]` is recognized, dispatch to the corresponding
  higher-level constructor/open path

This preserves the current layering:

- low-level open still discovers a `TreeStore`
- logical-object open is an extra step on top

## Root-Only First Implementation

The design should anticipate subtree object roots, but the first implementation
does not need to support them yet.

Initial scope:

- only the store root may be materialized as a richer object
- only `/_meta` at store root is consulted by `blosc2.open(urlpath)`

Deferred scope:

- subtree object roots such as `/users/_meta`
- multiple richer objects in one `TreeStore`
- automatic materialization of `tstore["/subtree"]`
- explicit references to store-subtree logical objects

This staged approach keeps the first implementation simple while preserving a
clear path toward multi-object stores later.

## Dispatch API Shape

The first implementation should support:

```python
obj = blosc2.open(urlpath)
```

Behavior:

- if `urlpath` resolves to a plain store with no recognized root manifest,
  return `TreeStore`
- if `urlpath` resolves to a `TreeStore` with recognized `/_meta`, return the
  richer object

For the deferred subtree-aware model, the API question is still open:

- `blosc2.open(urlpath, key="/users")`
- `blosc2.Ref` support for store-subtree objects
- other path-addressing schemes

These should be designed in a later phase.

## Error Handling

The generic manifest contract should distinguish:

- no `/_meta` present:
  - return raw `TreeStore`
- `/_meta` present but missing required fields:
  - error clearly
- `/_meta` present with unknown `kind`:
  - either return raw `TreeStore` or raise a dedicated error

Recommended first behavior:

- missing manifest: return raw `TreeStore`
- malformed recognized manifest: raise error
- unknown manifest kind: return raw `TreeStore`

This is conservative and avoids breaking forward compatibility unnecessarily.

## Recommended Invariants

- `/_meta` must always be a persisted `SChunk`
- `/_meta.vlmeta["kind"]` must be a string
- `/_meta.vlmeta["version"]` must be an integer
- logical object implementations own the schema of additional fields
- object materialization should not depend on `TreeStore` iteration order

## Example: `CTable`

With this contract, a root-level `CTable` would look like:

- `/_meta`
- `/_valid_rows`
- `/_cols/id`
- `/_cols/score`

And the manifest would contain:

```python
{
    "kind": "ctable",
    "version": 1,
    "schema": {...},
}
```

`blosc2.open(urlpath)` would:

1. detect `b2tree`
2. open `TreeStore`
3. inspect `/_meta`
4. see `kind == "ctable"`
5. return `CTable`

## Open Questions

- Should unknown manifest kinds return raw `TreeStore`, warn, or raise?
- Should there eventually be a helper such as `blosc2.open_store_object(...)`
  for explicit manifest-driven dispatch?
- Should `TreeStore` grow a helper for probing object roots, e.g.
  `get_object_manifest("/")` or `has_object_manifest(path)`?
- Should object-manifest detection be limited to `TreeStore`, or later be
  generalized to other store-like containers?

## Recommended Next Step

Use this contract for the first root-level `CTable` implementation:

- generic manifest mechanism defined here
- `CTable` as the first supported manifest `kind`
- root-only dispatch in `blosc2.open()`

Once that is stable, subtree object roots can be added without changing the
basic meaning of `/_meta`.
