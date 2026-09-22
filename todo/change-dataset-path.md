# Add path= as an alias for dataset=

Status: implemented.

Add `path=` as a backward-compatible alias for `dataset=` in remote opening
APIs. The selected node can be a group, table or array; `path=` describes this
more clearly than `dataset=`.

Example:

```python
store = blosc2.RemoteStore(
    "https://example.org/weather.zarr",
    path="europe/spain",
)
```

- Keep existing `dataset=` calls working; no removal or deprecation is planned.
- Apply the alias consistently across relevant constructors, opening helpers
  and sparse-cache factories.
- Define and test behavior when both names are supplied; reject conflicting
  values with a clear error.
- Normalize to the existing internal representation so persisted source
  descriptors and artifacts remain compatible.
- Prefer `path=` in documentation and examples, explaining that it selects a
  node within the source rather than the source URL itself.

Implementation details:

- `path` is keyword-only; existing positional `dataset` arguments are unchanged.
- `None` is unspecified; `""` and `"/"` explicitly select the root. Both names
  may be supplied when equal after stripping outer slashes.
- URL-selector conflict rules remain unchanged.
- Supported by `open` (including local containers), remote array/store/table
  constructors, store/table sparse-cache factories, B2Z/HDF5 source constructors,
  and HDF5 index scanning/validation helpers.
- One boundary resolver preserves internal `dataset` names, cache identities,
  and persisted artifacts. Regression tests are in `tests/test_dataset_path.py`.

Related plan: [Nested remote stores](../plans/remote-nested-store.md).
