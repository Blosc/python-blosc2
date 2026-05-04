# CTable Parquet/Arrow Metadata Plan

## Summary

Add an optional, internal metadata section to the persistent `CTable` schema dict so Arrow/Parquet import paths can preserve enough original Arrow schema information for better future round-tripping.

The immediate motivation is supporting nested Parquet columns such as:

```text
struct<ciqual_food_code: int32, agribalyse_food_code: int32>
list<struct<lang: string, text: string>>
```

The target design is full internal `StructSpec` support, including `list<struct<...>>`, with metadata preserving the original Arrow schema so `parquet -> CTable -> parquet` can round-trip supported nested columns without schema loss.

This is intended as an implementation detail, not a public `t.metadata` API.

---

## Goals

1. Preserve original Arrow/Parquet schema information when importing into CTable.
2. Keep the metadata optional and backward-compatible.
3. Avoid changing the physical data layout for existing columns.
4. Avoid exposing a public metadata API for now.
5. Enable Parquet round-tripping for nested columns, including `struct` and `list<struct>` columns.
6. Implement full internal `StructSpec` support for Arrow/Parquet nested schemas.
7. Preserve enough Arrow schema fidelity so supported nested columns can round-trip through `parquet -> CTable -> parquet` without schema loss.
8. Ensure old tables without metadata continue loading unchanged.
9. Ensure readers that do not understand the metadata can ignore it safely.

---

## Non-goals

- Expose user-facing metadata management methods.
- Guarantee exact Arrow schema reconstruction for all possible Arrow schemas outside the supported nested V1 scope.
- Store large amounts of per-row or per-column statistics in schema metadata.

---

## Proposed schema dict extension

Current persistent CTable schema serialization stores column definitions in a JSON-like dict. Extend this dict with an optional top-level `metadata` field:

```json
{
  "columns": [
    {"name": "code", "spec": {"kind": "string", "max_length": 32}},
    {"name": "generic_name", "spec": {"kind": "list", "item": {"kind": "struct", "fields": [...]}}}
  ],
  "metadata": {
    "arrow": {
      "schema_ipc_base64": "...",
      "fields": {
        "generic_name": {
          "original_arrow_type": "list<item: struct<lang: string, text: string>>",
          "ctable_storage": "list<struct<lang: string, text: string>>",
          "conversion": "native_struct"
        }
      }
    }
  }
}
```

The exact top-level schema dict may contain additional existing keys; this plan only adds an optional sibling key named `metadata`.

### Compatibility rules

- If `metadata` is absent, treat it as `{}`.
- If `metadata` is present but contains unknown keys, preserve them when possible.
- If `metadata.arrow` is absent, Arrow/Parquet code falls back to ordinary inference.
- Existing table loading should not require or validate Arrow metadata.

---

## Internal representation

Prefer storing metadata on `CompiledSchema`:

```python
@dataclass
class CompiledSchema:
    row_cls: type | None
    columns: list[CompiledColumn]
    columns_by_name: dict[str, CompiledColumn]
    metadata: dict[str, Any] = field(default_factory=dict)
```

No public property is required. CTable internals can access:

```python
self._schema.metadata
```

or helper methods if needed:

```python
def _schema_metadata(self) -> dict[str, Any]:
    return self._schema.metadata
```

Do not expose `t.metadata` initially.

---

## Serialization changes

### `schema_to_dict()`

Emit metadata only when non-empty:

```python
def schema_to_dict(schema: CompiledSchema) -> dict:
    d = {...}
    if schema.metadata:
        d["metadata"] = schema.metadata
    return d
```

### `schema_from_dict()`

Read metadata defensively:

```python
def schema_from_dict(d: dict) -> CompiledSchema:
    return CompiledSchema(
        row_cls=None,
        columns=columns,
        columns_by_name=columns_by_name,
        metadata=dict(d.get("metadata", {})),
    )
```

### Existing code paths

Update any manual `CompiledSchema(...)` construction to pass no metadata or explicitly preserve it. Because `metadata` defaults to `{}`, most call sites should continue to work.

Important places to check:

- dataclass schema compilation
- Pydantic schema compilation
- `CTable.open()`
- `CTable.load()`
- `CTable.save()`
- `CTable.select()` / views
- `CTable.from_arrow()`
- `CTable.from_arrow_batches()`
- `CTable.from_parquet()`
- schema mutation methods

For views/selections, metadata should probably be filtered to selected columns where practical, but this can be deferred if metadata remains internal.

---

## Arrow schema encoding

Do not rely solely on `schema.to_string()` for round-trip fidelity. Store a serialized Arrow schema when possible.

Potential encoding:

```python
import base64
import pyarrow as pa

sink = pa.BufferOutputStream()
with pa.ipc.new_stream(sink, schema):
    pass
schema_ipc = sink.getvalue().to_pybytes()
schema_ipc_base64 = base64.b64encode(schema_ipc).decode("ascii")
```

Store under:

```json
{
  "metadata": {
    "arrow": {
      "schema_ipc_base64": "...",
      "schema_string": "... optional debug string ..."
    }
  }
}
```

Open question: verify the best PyArrow API for schema-only IPC serialization. If `pa.ipc` exposes a direct schema serialization API, prefer that.

---

## Column-level Arrow metadata

For fields that are transformed during import, store explicit conversion notes:

```json
{
  "metadata": {
    "arrow": {
      "fields": {
        "generic_name": {
          "original_arrow_type": "list<element: struct<lang: string, text: string>>",
          "ctable_storage": "list<struct<lang: string, text: string>>",
          "conversion": "native_struct"
        },
        "no_nutrition_data": {
          "original_arrow_type": "bool",
          "ctable_storage": "list<bool>",
          "conversion": "nullable_scalar_wrapped_as_singleton_list"
        },
        "ecoscore_data": {
          "original_arrow_type": "string",
          "ctable_storage": "list<string>",
          "conversion": "long_nullable_scalar_wrapped_as_singleton_list"
        }
      }
    }
  }
}
```

This lets exporters distinguish native CTable schemas from import-time adaptations.

---

## StructSpec and nested list support

Add a first-class internal `StructSpec` so CTable can represent Arrow structs directly:

```python
blosc2.struct(
    {
        "lang": blosc2.string(),
        "text": blosc2.string(),
    }
)
blosc2.list(blosc2.struct({"lang": blosc2.string(), "text": blosc2.string()}))
```

Then import can map:

```text
struct<...>       -> blosc2.struct(...)
list<struct<...>> -> blosc2.list(blosc2.struct(...), nullable=True)
```

For `ListArray`, list cells can continue to be stored row-wise as Python values such as `list[dict]`, but the `ListSpec.item_spec` should be a typed `StructSpec`, not an opaque object spec. Coercion validates each dict-like item against the struct fields, and Arrow export uses the saved/original Arrow struct schema where available.

Metadata records the original Arrow field and the native struct conversion:

```json
{
  "original_arrow_type": "list<element: struct<lang: string, text: string>>",
  "ctable_storage": "list<struct<lang: string, text: string>>",
  "conversion": "native_struct"
}
```

Opaque object support can remain a fallback for Arrow types outside the supported nested V1 scope, but the primary goal is native `StructSpec` support and full Parquet round-tripping for supported nested columns.

---

## Import behavior proposal

When `CTable.from_parquet()` or `CTable.from_arrow_batches()` receives an Arrow schema:

1. Preserve the original Arrow schema IPC bytes in schema metadata.
2. For columns imported without transformation, no field-level entry is required.
3. For transformed columns, add a field-level entry describing the conversion.
4. For unsupported columns that are skipped by a custom importer, optionally record them under `metadata.arrow.skipped_fields` if the importer chooses to.

Potential internal helper:

```python
def _arrow_metadata_from_schema(
    schema, *, field_conversions=None, skipped_fields=None
) -> dict:
    return {
        "arrow": {
            "schema_ipc_base64": encode_arrow_schema(schema),
            "schema_string": schema.to_string(),
            "fields": field_conversions or {},
            "skipped_fields": skipped_fields or {},
        }
    }
```

---

## Export behavior proposal

`CTable.to_parquet()` should check:

```python
arrow_meta = self._schema.metadata.get("arrow", {})
```

For ordinary columns:

- Continue using normal CTable → Arrow conversion.

For columns with known conversions:

- `struct` and `list<struct>` columns:
  - Convert Python dict/list-of-dict values to `pa.array(..., type=original_arrow_type)`.
- singleton-list wrapped nullable scalars:
  - Optionally unwrap back to nullable scalar Arrow columns.
- long string wrapped as `list<string>`:
  - Optionally unwrap back to original nullable scalar string column if metadata says so.

If metadata is missing, malformed, or incompatible with current data, fall back to safe behavior or raise a clear error depending on export mode.

Possible control flag:

```python
t.to_parquet(path, use_original_arrow_schema=True)
```

For supported nested columns, the implementation goal is to use the original Arrow schema by default when metadata is available and compatible. If metadata is unavailable or incompatible, fail clearly or fall back according to the selected export mode.

---

## Tests

### Schema metadata persistence

1. Creating a CTable without metadata produces a schema dict without `metadata` or with an empty metadata field omitted.
2. Loading old schema dicts without `metadata` works.
3. Saving and opening a CTable with metadata preserves the metadata exactly.
4. `CTable.save()` and `CTable.load()` preserve metadata.

### Arrow metadata

1. `from_parquet()` stores original Arrow schema metadata when enabled.
2. `from_arrow_batches()` stores original Arrow schema metadata when given a schema.
3. Metadata survives close/open for `.b2z` and directory-backed stores.
4. Unknown metadata keys survive round-trip.

### Struct/nested round-trip tests

1. Import a top-level Arrow `struct` column as `StructSpec`.
2. Import a `list<struct>` column as `list(StructSpec)`.
3. Stored values are Python dicts or lists of dicts with validated fields.
4. Metadata records the original Arrow type.
5. Export reconstructs the original `struct` / `list<struct>` Arrow type.
6. Parquet → CTable → Parquet preserves schema and values for supported nested fields.

---

## Migration/backward compatibility

This change should be backward-compatible because:

- `metadata` is optional.
- Existing schema dicts remain valid.
- Existing code paths can ignore unknown metadata.
- No physical storage format changes are required.
- Public APIs do not change.

Potential compatibility concern:

- If older versions of python-blosc2 strictly reject unknown top-level schema keys, tables written with metadata may not load in old versions. Check existing `schema_from_dict()` behavior. If necessary, store metadata in a nested location old readers already ignore, or accept that forward compatibility to older versions is limited.

---

## Open questions

1. What exact PyArrow API should be used for schema-only serialization?
2. Should metadata be preserved exactly or normalized/sanitized on save?
3. Should selected views filter metadata to selected columns?
4. Should `from_parquet()` always store Arrow metadata, or only when nested/transformed columns are present?
5. Should metadata be compressed if the Arrow schema is large?
6. Should there eventually be a public metadata API, or should this remain strictly internal?
7. For singleton-list wrapped scalars, should `to_parquet()` unwrap by default when original Arrow metadata is present?

---

## Suggested milestones

### Milestone 1: Generic schema metadata plumbing

- Add optional `metadata` to `CompiledSchema`.
- Update `schema_to_dict()` and `schema_from_dict()`.
- Ensure save/open/load preserve metadata.
- Add tests for backward compatibility and metadata preservation.

### Milestone 2: Arrow schema metadata helpers

- Add private helpers to encode/decode Arrow schema metadata.
- Store Arrow schema metadata in `from_arrow_batches()` / `from_parquet()`.
- Add tests using simple Arrow schemas.

### Milestone 3: StructSpec and ListArray nested support

- Add internal `StructSpec` with field specs, metadata serialization, and validation/coercion.
- Allow `ListSpec(StructSpec)` for msgpack-backed ListArray.
- Map Arrow `struct` and `list<struct>` to `StructSpec` / `list(StructSpec)` in import.
- Store field conversion metadata.

### Milestone 4: Metadata-aware Parquet export

- Teach `to_parquet()` to consult original Arrow metadata.
- Reconstruct `struct` and `list<struct>` arrays from stored dict/list-of-dict values.
- Optionally unwrap singleton-list transformed scalar columns.
- Add Parquet → CTable → Parquet round-trip tests for selected nested fields.
