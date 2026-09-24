# RemoteCTable: batch-backed columns

Status: implemented and verified on 2026-09-18 as follow-up 2 in
`remote-ctable.md`.

## Objective and scope

Extend the existing read-only RemoteCTable to read external `.b2b` members on
demand. Reuse BatchArray serialization and row lookup, ListArray,
_ScalarVarLenArray and DictionaryColumn rather than introducing remote versions
of their table algorithms.

The intended column coverage is:

- Batch-backed variable-length strings and bytes.
- Batch-backed lists, with existing MessagePack and optional Arrow serializers.
- Struct/object scalar values supported by the existing serializer, subject to
  the remote decoding safety audit below.
- Dictionary columns: RemoteArray integer codes plus a batch-backed vocabulary.

Keep existing schema, null, deleted-row and spare-capacity semantics. Unsupported
representations must not prevent schema inspection or reading supported siblings.
Errors must identify the affected column and limitation.

No new public remote-container class, dependency, archive format or constructor
option is planned. Persisted indexes, portable references, remote writes and
variable-length block transport remain separately scoped. ListArray
`storage="vl"` uses ObjectArray rather than BatchArray and is outside this
extension; retain an explicit diagnostic for it.

## Evidence and implementation boundaries

- `ctable_storage.py`: RemoteTableStorage already supplies the storage boundary;
  its list, non-UTF-8 variable-length and dictionary openers remain unsupported.
- `batch_array.py`: BatchArray handles batch metadata, item lookup and
  MessagePack/Arrow decoding, but its reads currently use a local SChunk.
- `scalar_array.py`: _ScalarVarLenArray constructs its persisted row count from
  batch lengths; missing lengths can currently cause every batch to be decoded.
- `list_array.py`: ListArray has batch-backed reads and a separate ObjectArray
  representation. Its cached batches and local copy optimizations need auditing.
- `dictionary_column.py`: DictionaryColumn lazily loads the full vocabulary to
  construct value-to-code and code-to-value caches on first use.
- `b2z_source.py` and `proxy_source.py`: B2Z member windows and frame-parsing
  helpers provide reusable byte-range machinery. The existing NDArray block
  parser explicitly excludes variable-length blocks.
- `ctable_remote_read.py`: metadata and row transport already have bounded
  scheduling. Batch integration must reuse that mechanism where applicable.

Trace the callers of each shared helper before changing it. Preserve local
optimized paths and avoid a broad source/protocol refactor. No C/Cython changes
are expected unless the initial transport experiment demonstrates a need.

## Storage, transfer and caching decisions

Use a whole compressed batch (one BatchArray chunk) as the initial payload
transfer unit. Locate it using frame metadata and chunk offsets, fetch only that
range, and decode locally using existing Blosc2 and BatchArray machinery. Do not
route variable-length blocks through the fixed-width NDArray block parser.

A small row selection may require whole batches. A single unusually large batch
therefore requires a correspondingly large read and decode allocation. Transport
buffer limits bound scheduling; they do not promise to subdivide a batch or bound
its decoded Python representation. Document this ceiling and mark the deliberate
whole-batch simplification with a `ponytail:` comment at its implementation.

Require valid persisted batch lengths for nonempty remote batch columns in this
first implementation. Validate their count, nonnegative integer values and
prefix-sum bounds before row lookup. Reject missing or invalid lengths with a
column-specific diagnostic instead of silently decoding the column during open.
Empty columns remain valid without a nonempty lengths catalog. Schema inspection
and supported sibling reads must still work for rejected columns.

Reuse archive identity, member paths and the existing storage-options fingerprint
for cache identity. NONE, MEMORY and DISK must work. Retained compressed batch
payloads must share the owner's aggregate budget with NDArray leaves, traffic
accounting and eviction machinery; do not add an independent per-column budget.
Batch counts, lengths and frame indexes are metadata and should be accounted for
consistently with existing remote metadata. Do not persist credentials.

Group requested rows by batch where needed so NONE does not repeatedly fetch the
same batch within one operation. Audit decoded wrapper caches separately from the
compressed transport cache; do not imply that the latter bounds all Python memory.
Retain the existing bounded small-member prefetch allowance.

## Dictionary behavior

Open dictionary codes through RemoteArray and the vocabulary through the same
batch backend used by variable-length string columns. Reuse DictionaryColumn's
existing lookup and predicate behavior.

Preserve full-vocabulary loading on first use for this extension. Opening the
table or inspecting its schema must remain lazy, but decoding rows or resolving
query literals may read the complete vocabulary. This costs O(dictionary
cardinality) transfer and decoded memory, not O(table row count). The lookup maps
are decoded Python state outside the compressed transport cache budget.

Document that cost and measure it separately from code-array reads. Selective
vocabulary lookup is deferred until measurements establish a need. Persisted
indexes remain disabled, including dictionary indexes.

## Implementation sequence

1. **Prove one remote batch read.** Generate a nullable `vlstring` table with
   several batches, archive it and expose it through the existing instrumented
   fsspec `memory://` filesystem. Read the `.b2b` member header, metadata and chunk
   offsets using the existing B2Z window and frame helpers. Fetch one distant
   compressed batch and decode it with existing machinery. Compare against the
   local archive and record the ranges transferred. Use a member large enough
   that bounded metadata prefetch cannot mask a whole-member download.

2. **Connect a minimal internal batch reader.** Choose the smallest backend
   boundary demonstrated by the experiment: reuse BatchArray's serialization,
   batch lengths and item mapping, adapting only its local-SChunk assumptions
   needed for reads. Support metadata, batch counts and lengths, compressed-batch
   access, decoding and size reporting. Reuse owner leases, generation checks,
   archive transport and frame helpers. Do not emulate unrelated writable SChunk
   APIs or add a public RemoteBatchArray as part of this task.

3. **Deliver the first end-to-end column.** Add remote variable-length opening
   through _ScalarVarLenArray and verify one nullable `vlstring` column with NONE
   caching. A distant scalar read must match local behavior without downloading
   the whole column or archive. Include batch boundaries and null/empty-string
   distinction. This is the first implementation milestone and the checkpoint
   for confirming the backend design before expanding type coverage.

4. **Complete shared caching and transport integration.** Connect compressed
   batch retention to the existing owner cache coordinator for MEMORY and DISK,
   including eviction and reopen. Extend `.b2nd`-specific member selection in
   remote table metadata opening to include required `.b2b` members and dictionary
   companions. Reuse the existing bounded transport scheduler; keep parsing,
   decoding and cache mutation on the owning thread. Check repeated reads and
   mixed NDArray/batch pressure against the aggregate cache limit.

5. **Enable the remaining wrappers.** Add variable-length bytes, batch-backed
   lists, and supported struct/object scalars using their existing wrappers.
   Preserve optional Arrow dependency behavior. Then add dictionary opening with
   RemoteArray codes and the remote vocabulary store. Reuse existing storage
   suffixes, schema reconstruction and role-metadata validation. Validate
   companion existence and dictionary code layout; release every acquired handle
   on partial-open failure.

6. **Audit shared reads, mutation and lifetime.** Trace scalar, slice, strided,
   reverse and fancy reads, iteration, filtered views, supported predicates,
   null handling, size reporting, repr and local copy/save. Repair shared helpers
   where they assume a native SChunk, local path or extension input. Reject
   mutations before pending lists or dictionary caches change, including through
   raw wrappers and metadata handles. Check lifetime even on empty reads and
   cached results. Root-table close and RemoteStore refresh must invalidate
   borrowed columns/views; parent-store close must leave an acquired table usable.
   Detached materializations must be ordinary writable local tables.

7. **Document and demonstrate the extension.** Update API support descriptions
   and the remote handling example with a small representative set of batch-backed
   columns, including a dictionary. Report ordinary batch cold/warm reads and
   dictionary first-use costs separately. Document whole-batch granularity,
   missing-length diagnostics, decoded-memory costs and remaining unsupported
   representations. Update follow-up 2 in the original plan only after verification.

## Validation and safe decoding

Validate remote frame/member boundaries, offsets, metadata versions, serializer
names, batch lengths and required companions before using them. Preserve existing
rejections for encrypted, ZIP-compressed and unsupported embedded payloads.
Do not silently localize an entire archive to satisfy an unsupported read.

Before enabling object values, audit `msgpack_utils.py` and its extension decoders.
They can reconstruct Blosc2 objects and serialized expressions, so ordinary
MessagePack decoding cannot simply be assumed to be passive for every value.
Remote reads must not execute arbitrary serialized callables, import code named
by remote metadata or unexpectedly resolve external references. Reuse safe data
decoding where possible; explicitly reject unsafe extension forms with a useful
diagnostic. Keep local decoding behavior unchanged. Record any resulting object
value limitations in the supported-type documentation and regression tests.

## Verification and acceptance

Use the `blosc2` conda environment for all Python, test and build commands. Extend
existing helpers in `tests/ctable/test_remote_ctable.py`, `tests/test_batch_array.py`,
`tests/test_b2z_source.py` and `tests/test_remote_store.py` as appropriate rather
than adding another transport test harness.

- Compare against a local read-only CTable from the exact same archive. Cover
  empty/nonempty columns, unequal and partial final batches, batch boundaries,
  null versus empty values, multilingual strings, bytes, supported lists and
  objects, dictionary nulls, deleted rows and spare capacity.
- Exercise scalar, contiguous, strided, reverse and fancy reads, iteration,
  filtered views, mixed fixed-width/UTF-8/batch predicates, existing supported
  computations and local materialization. Preserve local errors for unsupported
  operations. Include nested column paths and RemoteStore-nested tables.
- Verify lazy opening and metadata-only size reporting without payload scans,
  allowing bounded small-member prefetch. Missing batch-length metadata must fail
  explicitly for nonempty columns without affecting supported siblings.
- Cover malformed metadata, invalid lengths/offsets, missing companions,
  unsupported serializers/representations, unsafe object extension payloads and
  partial-open cleanup. Exercise Arrow when installed and its existing missing
  dependency diagnostic otherwise.
- Exercise NONE, MEMORY and DISK, mixed-leaf aggregate eviction, disk reopen and
  source identity isolation. A small selection must transfer only the required
  batches plus bounded metadata. A repeated cached selection that fits the budget
  must need no new payload requests. NONE must avoid duplicate batch fetches
  within a grouped read without retaining a persistent payload cache.
- Check read-only enforcement before pending-state changes, cached reads after
  close/refresh, parent-store close, borrowed views and detached local exports.
- Reuse the deterministic HTTP range server to verify actual range transport.
  Cloud access is optional manual validation, not an automated-test requirement.
- Record a larger cold/warm experiment with archive size, batch geometry,
  requested rows, requests, transferred bytes and retained compressed payload.
  Report dictionary vocabulary loading separately and identify decoded-memory
  costs. No universal latency target is required.
- Run focused batch/list/scalar/dictionary, CTable, remote table/store and B2Z
  tests, then the default suite. Run Ruff on changed Python files. Warnings remain
  errors.

Completion means supported batch-backed reads and queries match local behavior,
remain read-only, obey existing lifetime guarantees, and fetch payloads on demand
at documented batch granularity. Whole-column query scans and first-use dictionary
vocabulary loading must be explicit costs, not presented as selective row I/O.

## Implementation results

The seven implementation steps were completed as separate commits:

1. `c5e17608` — prove remote batch range reads.
2. `c941ed94` — add the internal remote batch reader.
3. `3bbb744a` — read remote variable-length strings.
4. `3c4f1821` — integrate remote batch caching.
5. `3f58deed` — enable the remaining batch-backed columns.
6. `b3dfaaf9` — harden shared reads, mutation rejection and lifetime handling.
7. `4619082c` — document and demonstrate the extension.

The resulting reader supports variable-length strings and bytes, MessagePack and
Arrow batch-backed lists, passive struct/object values, and dictionary columns.
Compressed batches participate in the owner's shared NONE, MEMORY or DISK cache
policy. Remote MessagePack decoding rejects embedded Blosc2 containers and
serialized references. Nonempty columns require valid persisted batch lengths,
and `storage="vl"` lists remain explicitly unsupported.

The default suite passed with 10,315 tests and 36 skips. The focused remote table
suite passed with 90 tests, the related batch/list/dictionary/remote suites passed
with 413 tests and 5 skips, Ruff passed, and the normal Sphinx documentation build
completed. The strict `sphinx -W` build remains affected by pre-existing
repository-wide orphan-page, theme-option and ambiguous-reference warnings.

An instrumented `memory://` experiment used a 55,026,931-byte archive containing
100,000 rows, 1,024 rows per variable-length batch and a 20,000-value dictionary.
A distant five-row batch read used 5 requests and transferred 575,310 bytes,
retaining 558,335 compressed bytes; the warm repeat used 0 requests and 0 bytes.
The dictionary code slice used 2 requests and 4,612 bytes. First decode loaded the
vocabulary with 10 requests and 95,494 bytes. Its decoded Python strings and maps
occupied approximately 4,028,168 bytes by `sys.getsizeof`, outside the compressed
transport cache budget. Metadata opening used 2 requests and 9,297 bytes.
