# Plan For Changing `blosc2.open()` Default Mode To Read-Only

## Goal

Change the default mode for `blosc2.open(...)` from `"a"` to `"r"` so that
opening an existing object is non-mutating and unsurprising by default.

The change should:

- reduce accidental write access
- avoid implicit unpack / rewrite work for store-backed containers
- align with user expectations for a generic `open(...)` API
- preserve a smooth migration path for existing code that relied on writable
  opens without an explicit `mode=`

This plan is for later consideration and rollout design. It does not assume
that the change should land immediately.

## Motivation

Today, `blosc2.open(...)` defaults to `"a"` in
[src/blosc2/schunk.py](/Users/faltet/blosc/python-blosc2/src/blosc2/schunk.py).

That means:

- opening a `.b2z` store without `mode=` may create a writable working copy
- append-mode store opens may unpack zip-backed stores into a temporary working
  directory immediately
- code that only intends to inspect metadata or query data can still enter a
  mutation-capable path by accident

This is especially surprising for:

- `TreeStore`
- `DictStore`
- `CTable`
- other container-like objects opened through the generic dispatcher

By contrast, users generally expect a bare `open(path)` call to be safe for
inspection unless they explicitly request write access.

## Current Situation

### Default values today

The following default to `"a"` today:

- `blosc2.open(...)`
- `DictStore(...)`
- `TreeStore(...)`
- `CTable(...)` constructor when opening/creating through `urlpath`

At the same time:

- `CTable.open(...)` already defaults to `"r"`

This creates an inconsistency where:

- `blosc2.open("table.b2z")` is writable by default
- `blosc2.CTable.open("table.b2z")` is read-only by default

### Concrete user surprise

For a `.b2z` store, append mode currently does extra work:

1. create a working directory (usually temporary)
2. extract the archive into that working directory
3. serve reads/writes from the extracted layout
4. repack on close

This is implemented in
[src/blosc2/dict_store.py](/Users/faltet/blosc/python-blosc2/src/blosc2/dict_store.py).

That behavior is reasonable when the caller explicitly asked for `"a"`, but
surprising when it is triggered only because `mode` was omitted.

## Desired End State

The target behavior is:

```python
blosc2.open(path)
```

should behave as if the user had written:

```python
blosc2.open(path, mode="r")
```

unless the object category does not support read-only opening for technical
reasons. In such cases, the exception should be explicit and documented.

The user should need to opt into mutation with:

- `mode="a"`
- `mode="w"`

## Design Principles

The migration should follow these rules:

- do not silently change semantics without a warning phase
- make the warning text concrete and actionable
- update all docs and examples before flipping the default
- keep the opt-in writable paths unchanged
- avoid introducing ambiguity about whether a store may be mutated
- prefer explicit `mode=` in library docs even after the default changes

## Recommended Rollout

### Phase 0: prepare the codebase

Before warning users:

1. audit internal calls to `blosc2.open(...)`
2. make all internal call sites spell out `mode=`
3. update examples, docs, and tests to use explicit modes
4. document the difference between:
   - `mode="r"`: inspect/query only
   - `mode="a"`: may unpack and repack stores
   - `mode="w"`: overwrite/create

This phase reduces ambiguity and makes later warning noise much more useful.

### Phase 1: deprecation warning

Keep the runtime default as `"a"`, but emit a `FutureWarning` when:

- `blosc2.open(...)` is called without an explicit `mode=`

The warning should fire only when `mode` was omitted, not when the caller
explicitly requested `"a"`.

Recommended warning text:

```python
FutureWarning(
    "blosc2.open() currently defaults to mode='a', but this will change "
    "to mode='r' in a future release. Pass mode='a' explicitly to keep "
    "writable behavior, or mode='r' for read-only access."
)
```

Notes:

- the wording should mention both the current and future defaults
- the wording should explain how to preserve current behavior
- the wording should not be container-specific

### Phase 2: flip the default

In the next planned breaking-compatible release window:

- change the default mode in `blosc2.open(...)` from `"a"` to `"r"`

At that point:

- calls with omitted `mode` become read-only
- code that needs writable behavior must use `mode="a"` explicitly

### Phase 3: remove warning-specific scaffolding

After the default flip has been out for one full release cycle:

- remove temporary warning helpers and migration notes that are no longer
  useful
- keep release notes and changelog entries for historical context

## Implementation Notes

### Tracking whether `mode` was omitted

To emit a warning only when appropriate, `blosc2.open(...)` needs to
distinguish:

- caller omitted `mode`
- caller passed `mode="a"` explicitly

A practical implementation is:

1. change the function signature internally to use a sentinel
2. resolve the effective mode inside the function
3. warn only when the sentinel path is used

For example:

```python
_MODE_SENTINEL = object()


def open(urlpath, mode=_MODE_SENTINEL, **kwargs):
    mode_was_omitted = mode is _MODE_SENTINEL
    if mode_was_omitted:
        mode = "a"  # Phase 1
        warnings.warn(...)
```

Later, in Phase 2:

```python
if mode_was_omitted:
    mode = "r"
```

This is better than relying on `mode="a"` in the signature because that
signature cannot tell whether the user explicitly passed `"a"`.

### Scope of change

This plan is specifically about `blosc2.open(...)`.

It does **not** require changing the defaults of:

- `DictStore(...)`
- `TreeStore(...)`
- `CTable(...)`

at the same time.

However, the docs should explain that:

- constructor-style APIs may still default to `"a"`
- generic `blosc2.open(...)` becomes read-only by default

This narrower scope reduces breakage and focuses on the highest-surprise entry
point first.

## Compatibility Risks

The main breakage risk is downstream code that relies on:

```python
obj = blosc2.open(path)
obj[...] = ...
```

without ever spelling out `mode="a"`.

After the default flip, that code may:

- fail with a read-only error
- stop persisting modifications
- expose behavior differences only at runtime

This is why the warning phase is important.

### Secondary risk: tests that mutate after open

Internal and downstream tests may open objects generically and then mutate
them. These need to be found and updated during Phase 0.

### Secondary risk: docs and notebooks

Tutorials that currently omit `mode=` may accidentally teach users the old
behavior. These should be updated before the warning phase begins.

## Documentation Changes

### API docs

Update the docstring for `blosc2.open(...)` to:

- describe the migration
- clearly document the meaning of each mode
- mention that read-only is the recommended mode for inspection/querying

### Examples

Update examples to use explicit modes consistently:

- inspection/querying: `mode="r"`
- mutation of existing stores: `mode="a"`
- create/overwrite: `mode="w"`

### User-facing migration note

Add a short migration note to release notes:

- “`blosc2.open()` now defaults to read-only; pass `mode='a'` explicitly if
  you need writable behavior.”

## Testing Plan

### Phase 1 tests

Add tests that verify:

- omitted `mode` emits `FutureWarning`
- explicit `mode="a"` does not warn
- explicit `mode="r"` does not warn
- effective behavior remains writable during the warning phase

### Phase 2 tests

After the flip, add/update tests that verify:

- omitted `mode` is read-only
- writes after omitted-mode open fail clearly
- explicit `mode="a"` still allows mutation
- `.b2z` omitted-mode open does not enter append-style write setup

### Documentation tests

Where practical, examples should use explicit `mode=` so doctests remain clear
and stable across the transition.

## Optional Compatibility Escape Hatch

If downstream breakage risk is considered high, one temporary option is an
environment-variable override for one transition cycle, for example:

- `BLOSC2_OPEN_DEFAULT_MODE=a`

This should only be used if needed. It adds complexity and should not become a
permanent configuration surface unless there is a strong operational reason.

## Related Follow-Up Worth Considering

Even if the default changes to `"r"`, append mode for `.b2z` may still be more
eager than desirable.

A separate improvement could make `.b2z` append behavior lazier:

- open in `"a"` without extracting immediately
- extract only on first mutation
- keep read-only-style fast paths for pure reads

That is orthogonal to the default-mode change and can be planned separately.

## Summary

The recommended path is:

1. make internal/docs/example usage explicit
2. add a `FutureWarning` when `blosc2.open(...)` is called without `mode=`
3. flip the default from `"a"` to `"r"` in the next suitable release window
4. keep writable behavior available via explicit `mode="a"`

This delivers a safer and less surprising user experience while still giving
existing code a clear migration path.
