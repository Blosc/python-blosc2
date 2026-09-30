# Safe-by-default deserialization for persisted Blosc2 objects

Status: implemented, 2026-09-30.

## Objective

Add an explicit deserialization policy to Python-Blosc2 and make persisted-input
boundaries safe by default:

```python
obj = blosc2.open(path)  # deserialize="safe"
obj = blosc2.open(path, deserialize="safe")
obj = blosc2.open(path, deserialize="full")  # explicit trusted input

obj = blosc2.from_cframe(frame)  # deserialize="safe"
obj = blosc2.from_cframe(frame, deserialize="full")
```

Safe mode must reject referential, executable, and unknown serialized
extensions before reconstructing them. Full mode preserves the current rich
MessagePack behavior for callers that trust the persisted data and intentionally
want nested Blosc2 objects, references, remote objects, and lazy objects.

The policy must follow the returned object into later metadata access, item
access, store traversal, CTable column reads, slices, views, and copies. Merely
checking the initial `open()` call is insufficient because most variable-length
values are decoded lazily.

## Motivation

Python-Blosc2 MessagePack extensions can represent more than passive values.
Normal decoding currently supports:

- embedded Blosc2 cframes reconstructed through `blosc2.from_cframe()`;
- `Ref` and C2Array values;
- RemoteArray carriers;
- LazyExpr values whose operands are opened during reconstruction;
- LazyUDF values whose persisted DSL source is parsed or compiled;
- NumPy object arrays whose elements may recursively contain these values.

Consequently, opening or indexing an untrusted ObjectArray, BatchArray,
ListArray, CTable, EmbedStore, or user metadata value can cause filesystem or
network access, use ambient credentials, create caches, compile persisted code,
or recursively allocate substantial object graphs.

The existing `_safe_msgpack_unpackb()` already establishes a useful precedent:
RemoteCTable's read-only BatchArray adapter uses it to allow passive values while
rejecting active extension codes. The capability should become a supported,
consistent policy across all persisted object paths.

## API decision

Use a mode rather than a boolean:

```python
deserialize: Literal["safe", "full"] = "safe"
```

The name describes the controlled operation more accurately than `safe=True`.
For example:

```python
blosc2.open("https://example/data.b2z", deserialize="safe")
```

still accesses the URL explicitly supplied by the caller, but values found
inside that object cannot silently open additional resources. A broad `safe`
boolean could be misread as prohibiting all network or filesystem activity.

Reserve, but do not initially expose, a possible third mode:

```python
deserialize = "raw"
```

Raw mode would preserve extension payloads as inert values. It should only be
added after its return types, recursive behavior, and round-trip guarantees are
designed. Do not accept a mode that is silently treated as safe.

Make `deserialize` an explicit keyword-only parameter rather than leaving it in
`**kwargs`. Normalize it once to an internal enum, for example:

```python
class DeserializeMode(Enum):
    SAFE = "safe"
    FULL = "full"
```

Internal code should pass the enum, not repeatedly compare user strings.
Unknown values and booleans must raise a clear `ValueError` or `TypeError`.

## Default and compatibility policy

Default to `deserialize="safe"` at every API that accepts existing serialized
bytes or opens existing persistence:

- `blosc2.open()` and `blosc2.load()`;
- `blosc2.from_cframe()`;
- `schunk_from_cframe()`, `ndarray_from_cframe()`,
  `objectarray_from_cframe()`, `estore_from_cframe()`, and
  `ctable_from_cframe()` where logical or metadata decoding can occur;
- constructors reopening an existing `urlpath`;
- local and remote stores reconstructing child frames;
- CTable reconstruction and backing-column opening.

New containers created directly in memory may retain full behavior because the
application supplied their values itself:

```python
objects = blosc2.ObjectArray()  # new application-owned object
objects.append(nested_array)
```

Constructors that can either create or reopen must determine the policy before
opening existing storage. Reopening existing bytes defaults safe; creation may
use full unless the caller explicitly selects safe. The behavior must be
documented and tested so it does not depend accidentally on implementation
order.

Changing the persisted-input default is intentionally behavior-breaking for
objects containing active extensions. Ordinary arrays, passive metadata, UTF-8,
dictionary data, and normal variable-length scalar/list values should continue
to work. Introduce the change at a clearly announced compatibility boundary,
preferably a major release if the current stable line follows semantic
versioning. Do not preserve unsafe behavior merely to avoid an actionable error.

## Safety contract

### Safe mode allows

- MessagePack null, booleans, integers, floats, strings, and bytes;
- nested lists, tuples, and mappings;
- the existing complex-number and set extensions;
- validated numeric, string, structured, and other non-object NumPy arrays;
- NumPy object arrays only when every element recursively satisfies safe mode;
- known internal metadata composed entirely of passive values;
- supported Arrow values after validating their schema and extension types.

### Safe mode rejects before reconstruction

- embedded Blosc2 cframes, initially including otherwise passive NDArray and
  SChunk values;
- `Ref`, C2Array, and RemoteArray values;
- Proxy and source descriptors encountered through persisted object metadata;
- LazyExpr and LazyUDF values;
- persisted DSL kernels or other executable recipes;
- unknown MessagePack extension codes;
- Arrow extension types or object-producing conversion paths that have not been
  explicitly audited.

Rejecting all embedded cframes in the first version is deliberate. Determining
whether an embedded cframe is passive requires recursively parsing and
classifying it. That can be added later with depth and resource accounting.

### Full mode

Full mode retains current reconstruction semantics. It does not waive format
validation, supported-version checks, shape overflow checks, or ordinary memory
safety checks. It only opts into active semantic reconstruction.

### What safe mode does not guarantee

`deserialize="safe"` controls authority-bearing semantic reconstruction. It
does not mean:

- the explicitly supplied path or URL is not accessed;
- decompression uses no memory or CPU;
- arbitrary input sizes are acceptable;
- a caller does not need application-level byte, nesting, timeout, or
  concurrency limits.

Document this boundary prominently. Servers such as cat2lite must still enforce
operational resource limits and source authorization.

## Error model

Add a dedicated public exception, for example:

```python
class UnsafeDeserializationError(ValueError):
    pass
```

Use it whenever safe mode encounters an active or unaudited extension. This
allows applications to distinguish a trust-policy failure from corruption or
an unsupported file-format version.

Errors should be actionable:

```text
Encountered active serialized value 'c2array' while using
deserialize='safe'. Reopen with deserialize='full' only if you trust this data
and intend to allow reference resolution and lazy-object reconstruction.
```

Where available, include a non-sensitive location such as an ObjectArray index,
BatchArray batch number, CTable column name, or metadata key. Do not include
credentials, signed URLs, complete reference payloads, arbitrary `repr()`
output, or unrestricted local paths.

Unknown extensions should name their numeric code. Nested errors should retain
the original exception as their cause while adding location context.

## Decoder implementation

Refactor `msgpack_utils.py` around one policy-aware entry point instead of
maintaining unrelated full and private-safe implementations:

```python
def msgpack_unpackb(payload, *, deserialize=DeserializeMode.SAFE): ...
```

Internal trusted call sites must pass `FULL` explicitly. For compatibility with
code that directly calls the currently public helper, decide whether its default
changes immediately or whether it remains a low-level full decoder; the
persistence APIs must not rely on an unsafe helper default. Prefer separate
clearly named internal functions if that avoids ambiguous defaults.

The safe extension hook should:

1. Decode known passive extension types directly.
2. Validate NumPy shape multiplication before allocation.
3. Recursively apply the same policy to NumPy object values and set payloads.
4. Reject cframe and structured-reference extension codes without calling
   `blosc2.from_cframe()`, `decode_b2object_payload()`, or `Ref.open()`.
5. Reject unknown extension codes rather than returning a live or ambiguous
   value.

Do not implement safe mode by performing full decoding and inspecting the
result afterward. Side effects may already have occurred.

Keep decoding policy conceptually separate from resource budgets. Optional
depth, element, and decoded-byte counters may be useful, but cat2lite and other
servers still need request-level limits spanning decompression and result
construction.

## Policy ownership and propagation

Each object capable of lazy decoding must retain an immutable effective
deserialization policy. The policy should not be persisted into the data; it is
a property of the current reader and trust decision.

Provide a single internal accessor rather than scattering attributes such as
`_safe` across classes. The effective policy must be reachable from:

- SChunk fixed metadata (`Meta`);
- SChunk variable-length metadata (`vlmeta`);
- NDArray through its SChunk;
- ObjectArray and BatchArray;
- ListArray and its backend;
- EmbedStore, DictStore, and TreeStore;
- CTable and every storage/backend column;
- RemoteStore and RemoteCTable wrappers;
- objects reconstructed from cframes.

If the extension-backed SChunk type cannot reliably carry an arbitrary Python
attribute, add a supported field/property in the wrapper or pass an owning
policy object to metadata proxies. Avoid global mode variables and weak maps
whose lifetime or identity behavior can silently lose the policy.

The policy should be immutable after data has been opened. A caller wishing to
change trust level should reopen or reconstruct explicitly. This prevents a
child handle or cached metadata value from having been decoded under a previous
policy.

## Metadata handling

Both fixed and variable-length metadata currently call the general MessagePack
decoder on access. Update:

- `Meta.__getitem__()` and `Meta.getall()`;
- `vlmeta.__getitem__()` and `vlmeta.getall()`;
- metadata-backed object classification;
- user attributes exposed by NDArray, SChunk, CTable, and stores.

Internal markers such as `b2nd`, `batcharray`, `listarray`, `vlarray`, `b2o`,
and store maps must be decoded under safe mode for classification. Their schemas
should contain only passive values. An active extension in an internal marker
is malformed or unsafe and must fail before object reconstruction.

Classification must inspect marker kind/version safely before deciding whether
to reconstruct a top-level special object. Under safe mode:

- ordinary NDArray/SChunk and passive logical containers open normally;
- a persisted `b2o` object requiring active reconstruction raises
  `UnsafeDeserializationError`;
- a legacy Proxy/source carrier requiring source opening raises the same error;
- explicit remote input supplied as the `urlpath` argument remains governed by
  existing remote-opening behavior, while nested persisted references remain
  blocked.

Reading a metadata mapping must not partially return entries before failing on
an unsafe value.

## ObjectArray, BatchArray, and ListArray

### ObjectArray

Replace unconditional `msgpack_unpackb(payload)` in item access with the
policy-aware decoder. Slice access must preserve the source policy and add index
context to failures.

### BatchArray

Replace `_deserialize_msgpack_block()` with a policy-aware implementation.
Remove the special divergence in `_RemoteBatchArray` once the base BatchArray
can carry safe mode correctly. Remote-backed containers should continue to
force or default to safe mode unless an API explicitly supports trusted remote
content.

Arrow serialization needs an equivalent policy audit. Safe mode should reject
unregistered/custom extension types and avoid automatic Python object
reconstruction. Supported primitive, nested-list, dictionary, and struct types
should continue to work.

### ListArray

Propagate policy into the ObjectArray or BatchArray backend and through remote
read wrappers, copying, slicing, and Arrow conversion. A ListArray may not
replace a safe backend with a full backend during optimization or reconstruction.

## CTable

CTable is a priority consumer because variable-length columns use ObjectArray,
BatchArray, and ListArray internally.

Propagate the table's policy through:

- table schema and storage metadata loading;
- `_ScalarVarLenArray` backends;
- ListArray columns;
- schema-less object and structured columns;
- projection, `slice()`, row iteration, and Arrow conversion;
- in-memory copies and cframe reconstruction;
- local, remote, and source-bound column opening.

Safe mode must decode normal `vlstring`, `vlbytes`, passive `struct`, object,
and list values successfully while rejecting active nested extensions before
they are materialized. A projected slice must not decode unselected columns.

`ctable_from_cframe()` reconstructs through an EmbedStore and must pass the
policy through both layers. RemoteCTable's existing safe BatchArray behavior
should become ordinary policy propagation rather than a private subclass-only
override.

## Store propagation

### EmbedStore

Store-map metadata must use safe decoding. Retrieving an embedded frame must
call `from_cframe(..., deserialize=<store policy>)`. A stored C2Array descriptor
must raise in safe mode before constructing or opening the C2Array.

### DictStore and TreeStore

The owner policy must propagate to:

- embedded values;
- external files;
- aliases and logical-object processing;
- member-window opens;
- CTable roots and private columns;
- values returned from `__getitem__`, iteration, and descendant traversal.

Discovery must not reconstruct a special leaf merely to identify its type.
Inspect its fixed marker safely first. Existing cycle avoidance for b2object
carriers must remain intact.

### RemoteStore and RemoteCTable

Remote responses and cached frames default safe. Explicit remote source access
is not itself disabled by this option, but values carried inside responses may
not activate additional references. Cache hits and misses must produce objects
with the same effective policy.

## Cframe APIs

Add `deserialize` to the generic and logical cframe entry points. Physical
`schunk_from_cframe()` and `ndarray_from_cframe()` may not decode values
immediately, but they still expose metadata that does; therefore, they must
attach the selected policy to the returned carrier.

Nested full decoding must propagate explicitly:

```python
blosc2.from_cframe(data, deserialize="full")
```

must ensure a later `obj[0]` or `obj.attrs[...]` also uses full mode. Conversely,
safe reconstruction must not call an internal helper whose default accidentally
switches to full.

Zero-copy cframe pinning and source-buffer lifetime behavior must remain
unchanged.

## Copy, slicing, and serialization

- Logical copies, slices, projections, and views inherit the source policy.
- A compressed chunk copy does not decode values and may copy active extensions
  inertly, but the destination remains safe and rejects them when accessed.
- `to_cframe()` serializes existing storage without changing its reader policy;
  the policy itself is not written into metadata.
- A newly constructed destination with explicitly supplied
  `deserialize="full"` may use full mode, but implicit optimization paths may not
  escalate policy.
- Pickling, if supported for these wrappers, must preserve or conservatively
  restore safe mode without embedding trust decisions into portable data.

## Internal call-site audit

Audit every current `msgpack_unpackb()`, `from_cframe()`, `process_opened_object()`,
and `open_b2object()` call. Important locations currently include:

- SChunk fixed and variable-length metadata;
- ObjectArray item access;
- BatchArray block access;
- EmbedStore map loading and child retrieval;
- DictStore value copying and reopening;
- CTable cframe reconstruction;
- proxy source metadata;
- HDF5 source helper cframes;
- RemoteStore and RemoteCTable response/cached-frame handling;
- generic `process_opened_object()` dispatch.

Every call must either propagate an existing policy or state why full decoding
is correct for newly generated, process-owned bytes. No persistence read should
rely on an implicit full-decoder default.

## Testing

### Decoder unit tests

- all passive MessagePack primitives and nested combinations;
- tuple, complex, set, and NumPy extension round trips;
- numeric and object NumPy arrays, including recursive active values;
- malformed shapes, multiplication overflow, inconsistent buffers, and unknown
  extension codes;
- every active structured kind rejected before its constructor/open function is
  called;
- full mode retains existing rich round trips.

Use mocks or network traps to prove that safe failures happen without opening a
path, resolving DNS, creating a cache, compiling DSL, or calling
`blosc2.from_cframe()` recursively.

### API tests

- `open()` and every `*_from_cframe()` entry point default safe;
- explicit `deserialize="full"` restores current behavior;
- invalid modes and booleans fail clearly;
- policy survives deferred metadata and item access;
- policy survives slicing, copying, projection, views, and store traversal;
- policy is not persisted into cframes;
- explicit URL opening still works while nested references are rejected.

### Object matrix

Exercise safe and full modes for:

- SChunk and NDArray metadata;
- ObjectArray;
- msgpack and Arrow BatchArray;
- both ListArray storage backends;
- CTable fixed, UTF-8, dictionary, varlen scalar, struct, object, and list
  columns;
- EmbedStore, DictStore, and TreeStore inline/external children;
- b2object, legacy Proxy, RemoteArray, RemoteStore, and RemoteCTable carriers;
- empty, nullable, multi-chunk, and malformed objects.

### Compatibility tests

- existing ordinary datasets open unchanged under the new default;
- documented rich-object examples are updated to request full mode;
- remote CTable safe behavior remains at least as restrictive as before;
- cframes written in full mode can be transported and copied in safe mode
  without activating their values;
- Python version, ABI3 wheel, and supported platform matrices remain green.

## Documentation and migration

Add a security section explaining:

- why safe is the default;
- exactly which values it allows and rejects;
- that explicit input URLs may still be accessed;
- that full mode is appropriate only for trusted data;
- that application-level resource limits remain necessary.

Update examples that intentionally persist nested arrays, C2Array references,
RemoteArray, LazyExpr, or LazyUDF:

```python
obj = blosc2.open(path, deserialize="full")
```

Release notes should call out the default change prominently and show the
actionable error. Avoid recommending a global environment variable that turns
full decoding back on; trust must remain visible at each boundary. If a staged
migration is required, warnings may announce the future change, but the final
API must default safe without silently retrying in full mode.

## Implementation stages

1. Add the internal mode enum, normalization, and public exception.
2. Refactor MessagePack decoding into policy-aware full and safe paths.
3. Attach immutable policy to SChunk carriers and metadata proxies.
4. Add and propagate `deserialize` through cframe entry points.
5. Add it explicitly to `open()`, `load()`, and existing-storage constructors.
6. Harden object classification, b2object, and legacy Proxy reconstruction.
7. Propagate through ObjectArray, BatchArray, and ListArray.
8. Propagate through EmbedStore, DictStore, and TreeStore.
9. Propagate through CTable, its storage backends, and variable-length columns.
10. Unify RemoteCTable's private safe decoder with the general policy.
11. Audit Arrow paths and reject unaudited extension types in safe mode.
12. Complete the call-site audit, adversarial tests, documentation, and release
    migration notes.

## Acceptance criteria

- Every persisted-input API defaults to `deserialize="safe"`.
- Active extensions are rejected before constructors, reference opening,
  network access, filesystem access, cache creation, or DSL compilation.
- The effective policy remains attached through all lazy access and nested
  container traversal.
- Normal passive ObjectArray, BatchArray, ListArray, and CTable data continue to
  work under the default.
- `deserialize="full"` preserves current trusted-data behavior.
- Errors explain the blocked kind and the explicit trusted-data opt-in without
  leaking sensitive payloads.
- No internal persistence-read path accidentally regains full decoding through
  a helper default.
- The policy is not serialized into datasets and does not alter their portable
  format.
