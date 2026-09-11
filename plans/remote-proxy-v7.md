# Remote proxy v7: Caterva2 sparse-cache lifecycle and soft quota

## Status and objective

Implementation checkpoint: Caterva2 now uses the sparse backend as its runtime
default. The implementation includes authorized source attachment, shared quota admission, private
generations, coldify, retirement/cleanup, offline pruning, conservative recovery,
and warm export artifacts. Endpoint benchmarks
and their raw samples live in Caterva2's `examples/benchmarks/remote_proxy_v7.md`
and `remote_proxy_v7_results/`.

This is the first implementation/measurement iteration, not completion of all
acceptance criteria below. It uses full-generation measurement/fsync after writes,
discards interrupted generations rather than resuming their transitions, and
reserves the full staging budget for one warm export. Incremental mutation
reports, bounded inventories, configurable maintenance tuning, large logical
chunk-count/RSS benchmarks, and power-loss validation remain follow-up work.
The rebuilt C-Blosc2 now removes payload-free sparse chunk files on successful
eviction; zero-length leftovers mentioned below apply to legacy/interrupted state.

The public carrier remains contiguous and portable, while mutable DISK state lives
in private sparse generations. The implementation is applied to Caterva2's server,
quota coordinator, recovery paths, API tests, and benchmark harness.

The server-boundary benchmark uses real ASGI routes and the rebuilt C-Blosc2.
For an 8 MiB source with a 1 MiB cache, v7 was 2.42x faster on warm hits and
1.98x faster on partial growth, but 0.57x on eviction-heavy churn. For a 64 MiB
source with a 32 MiB cache, v7 was 2.65x faster on cold fill, 8.82x on warm hits,
1.90x on churn, and 6.73x on partial growth. Raw samples and methodology live
in Caterva2's `examples/benchmarks/remote_proxy_v7.md`.

## Decisions fixed by v7

- Runtime layout is `<statedir>/.remote-cache/<object-id>/<generation-id>/`.
  Both identifiers are random UUID hex strings assigned by the server. User
  paths and source URLs never appear in private filenames.
- The public path remains a contiguous RemoteProxy carrier. It is the portable
  descriptor and never becomes a directory.
- An uploaded warm carrier is migrated once into a new sparse generation. Once
  that generation is active, it is the only authoritative cache. Caterva2 then
  cold-replaces the public carrier. Both copies remain charged until replacement
  completes.
- `NONE` and `MEMORY` continue to execute without server retention. Only requested
  `DISK` policy creates a runtime generation.
- Cache lifecycle exists even when customer quota is disabled. Quota controls
  admission and pruning; it must not be the owner registry.
- The customer quota is soft for runtime-cache fills. Ordinary dataset writes
  retain their existing staged admission and `quota_work_bytes` behavior.
- Charge regular dataset files by their existing `st_size` rule. Charge every
  entry in an active, retired, or trash-resident sparse generation by allocated
  size (`st_blocks * 512`), falling back to `st_size` where allocated size is
  unavailable. Include `chunks.b2frame`, `.b2lock`, numbered chunk files,
  directories, and zero-length entries' directory allocation. Stable lifecycle
  locks under `.storage` remain operational and excluded.
- Cache pruning uses whole-chunk LRU batches. It never chooses ordinary datasets.
- Copy creates a new cache identity and exports a cold carrier, including when
  the source is a legacy warm carrier. Move is implemented as
  copy-plus-delete and also starts cold at the destination. This deliberately
  gives up cached data to keep current non-atomic directory move semantics and
  crash recovery simple.
- Removing or replacing a RemoteProxy retires its active generation. The API may
  finish after the generation is atomically moved to private trash, but its bytes
  remain charged until background or startup deletion completes.
- Rollback ignores private caches and serves public cold carriers safely. It does
  not translate sparse generations back into a mutable public carrier.

## Scope and non-goals

The first deployment remains one customer state directory shared by local worker
processes on one host. SQLite and file locks on a network filesystem, shared state
across hosts, hard physical-volume guarantees, and conversion of ordinary writes
to in-place mutation are out of scope.

Do not weaken the existing HTTPS allowlist, DNS pinning, redirect refusal,
credential-free descriptor validation, geometry limits, or embedded-reference
guard. Every request resolves and authorizes the source before consulting a
private cache, including a cache hit.

Peer cache storage retains its independent quota and pruning implementation.
Media, SQLite/WAL files, staged candidates, authentication state, and lifecycle
locks keep their existing operational accounting classifications.

## Python-Blosc2 contract required by Caterva2

The Caterva2 integration now uses the supported Python-Blosc2 sparse-cache
contract below rather than reaching through `_proxy` internals:

1. Let `with_sparse_cache()` accept Caterva2's already-authorized
   `FsspecNDSource` together with its credential-free source descriptor. It must
   retain that source object and its pinned filesystem; it must not reopen the URL
   through default fsspec transport. Validate the descriptor used for persistence
   against the supplied source.
2. `cache_contains(item=(), *, nchunk=None) -> bool`, evaluated under the sparse
   frame lock, so a pure hit avoids a SQLite reservation and filesystem scan.
   Also expose a supported cache-operation context that keeps the frame lock held
   across checking and reading, or an atomic `read_cached()` returning a hit/result
   pair. A separate check followed by an ordinary mutating read is insufficient.
3. `trim_cache(target_bytes, *, max_chunks) -> CacheMutation`, which evicts no
   more than `max_chunks` least-recently-used chunks, updates fetched/block/index
   state through Proxy, and returns the affected chunk numbers plus payload sizes
   before and after. Persist enough LRU ordering in the sparse metadata for a new
   process to make the same coarse ordering decision; do not create one SQLite row
   per chunk.
4. A mutation result for reads/fetches containing whether storage changed,
   affected chunk numbers, payload bytes before/after, and the index/metadata
   files that changed. Caterva2 uses this to stat only affected files.
5. A stable warm export to a destination path while holding one consistent cache
   snapshot. Confirm that sparse-to-contiguous `save()` has bounded peak memory;
   otherwise add a bounded serializer before enabling warm exports.
6. Offline sparse inspection and dirty recovery without constructing a remote
   source. Startup must not call the URL-based constructor or perform outbound
   requests. If a damaged frame cannot be recovered offline, retire it and let a
   later authorized request create a fresh generation.

Specify mutation results as a supported value type: affected chunk IDs (including
evictions), old/new payload bytes, changed metadata paths, and created/deleted
entries. Paths must be relative to the generation and validated by the server.
The result must include writes performed during attachment/recovery, not just
explicit fetches. Define `target_bytes` and `max_cache_bytes` as compressed cached
payload limits; customer charge additionally includes allocated metadata and
directory space. An exception may follow a partial mutation: retain durable
accounting intent and reconcile even when no mutation result was returned.

The existing persistent `proxy-dirty` marker remains the payload-integrity fence.
All processes attach with `with_sparse_cache()`, which enables compatible frame
locking, refreshes shared fetched state, and recovers a marker only after obtaining
the frame lock. Caterva2 adds lifecycle and accounting intent around it.

The current server scans one generation after a mutation when a precise mutation
report is unavailable, but never scans on a hit. This is the safe initial fallback;
replace it with mutation-result accounting after benchmarking millions of logical
chunks and long-running quota convergence.

## Private paths and API isolation

Add these settings-derived paths without adding them to `storage_quota.ROOTS` or
any provider/root registry:

```text
<statedir>/.remote-cache/          active and retired generation directories
<statedir>/.remote-cache/.trash/   atomically detached generations awaiting deletion
<statedir>/.storage/exports/       controlled warm-export artifacts
```

Create them mode `0700` where supported. Reject `.remote-cache` and `.storage`
explicitly in path resolution and writable-path helpers even though public APIs
currently accept only `@public`, `@shared`, and `@personal`. Static mounts, root
listing, dataset walking, providers, the web viewer, archive expansion, and HDF5
unfolding must never traverse these paths.

Only the remote-cache registry constructs private paths. Validate stored relative
paths as exactly `.remote-cache/<32 hex>/<32 hex>` or
`.remote-cache/.trash/<32 hex>` before opening, renaming, or deleting them. Reject
symlinks at every component and never follow links during measurement or cleanup.

## SQLite schema version 2

Initialize `storage.sqlite` at schema version 2 inside the existing startup
initialization guard. Preserve `objects`, `operations`, and `account` for ordinary
file publication. The existing `objects.cache` column remains for compatibility;
sparse runtime bytes never use it.

Add:

```sql
CREATE TABLE remote_objects (
    object_id TEXT PRIMARY KEY,
    path TEXT UNIQUE,
    carrier_generation TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    source_stamp TEXT,
    active_generation TEXT,
    parent_charge_bytes INTEGER NOT NULL DEFAULT 0 CHECK(parent_charge_bytes >= 0),
    updated REAL NOT NULL
);

CREATE TABLE remote_generations (
    generation_id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL,
    relpath TEXT UNIQUE NOT NULL,
    state TEXT NOT NULL CHECK(state IN
        ('building', 'active', 'retired', 'trash')),
    spec_hash TEXT NOT NULL,
    source_stamp TEXT NOT NULL,
    max_cache_bytes INTEGER,
    payload_bytes INTEGER NOT NULL CHECK(payload_bytes >= 0),
    charge_bytes INTEGER NOT NULL CHECK(charge_bytes >= 0),
    inode_count INTEGER NOT NULL CHECK(inode_count >= 0),
    touched REAL NOT NULL,
    created REAL NOT NULL
);

CREATE UNIQUE INDEX one_active_remote_generation
ON remote_generations(object_id) WHERE state='active';

CREATE TABLE remote_operations (
    id TEXT PRIMARY KEY,
    generation_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK(kind IN
        ('build', 'fill', 'prune', 'retire', 'delete', 'coldify')),
    estimate INTEGER NOT NULL CHECK(estimate >= 0),
    previous_charge INTEGER NOT NULL CHECK(previous_charge >= 0),
    details BLOB NOT NULL,
    started REAL NOT NULL
);

CREATE TABLE remote_work (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('export', 'rebuild')),
    reserved INTEGER NOT NULL CHECK(reserved >= 0),
    relpath TEXT NOT NULL,
    started REAL NOT NULL
);

CREATE TABLE remote_orphans (
    id TEXT PRIMARY KEY,
    relpath TEXT UNIQUE NOT NULL,
    charge_bytes INTEGER NOT NULL CHECK(charge_bytes >= 0),
    inode_count INTEGER NOT NULL CHECK(inode_count >= 0),
    updated REAL NOT NULL
);
```

Add `cache_fill_suspended INTEGER NOT NULL DEFAULT 0` to `account`. Use `NULL`
quota internally to mean unlimited if the coordinator is instantiated only for
lifecycle. Alternatively retain the public `settings.quota == 0` convention and
branch admission before reading the account limit; do not encode unlimited as an
arbitrarily large integer.

Maintain the invariant that `remote_objects.active_generation` is either NULL or
names that object's sole `state='active'` row. Change both fields in the same
transaction whenever a generation activates or retires.

`remote_objects.path` is the current binding, not permanent ownership history.
On deletion/replacement set the old object's path to NULL and retire its active
generation in the same transaction. Its UUID continues to own retired/trash rows
while a new object can claim the same public path. Keep the former path in the
operation details for recovery. Every lookup must revalidate its binding after
acquiring the object lock; acquiring a lock for an obsolete lookup is not enough.

`remote_operations.details` is a versioned, credential-free msgpack record. For
build/coldify it records the expected public signature, expected cold artifact
digest, spec/stamp, and associated ordinary-publication operation ID; for
retire/delete it records old and intended trash paths. Insert intent before the
first filesystem mutation. Recovery must distinguish its own completed coldify
from an unrelated replacement with the same descriptor.

`spec_hash` is SHA-256 over canonical msgpack or sorted JSON containing the
credential-free RemoteProxy payload, shape, dtype string, chunks, blocks, and
runtime cache-format version. `source_stamp` is the stable validator observed from
the already-authorized source. `carrier_generation` is the existing
`storage_quota.signature()` JSON for the public carrier.

Do not enable SQLite foreign-key cascades for filesystem ownership. Filesystem
removal must be recoverable and charged until it succeeds; explicit state
transitions are clearer than a database cascade that forgets live bytes.

Extend `StorageQuota.usage()` diagnostics with:

```text
dataset_used       sum(objects.size)
remote_cache_used  generation charge + object-parent charge + orphan charge
used               dataset_used + remote_cache_used
reserved           existing publication growth + remote operation estimates
working            existing candidates + remote_work reservations
cache_fill_suspended
```

Keep existing keys for API compatibility.

Update every ordinary `StorageQuota.publish()` admission query to include
`remote_cache_used` and outstanding remote estimates. Likewise, remote fill
admission includes ordinary publication reservations. This is one shared account:
an upload and a cache fill cannot independently spend the same headroom. When no
quota is configured, the coordinator still provides compare-and-swap publication
for internal coldify and lifecycle operations but skips customer-capacity denial.

## Locks and ordering

Use stable lifecycle locks under `.storage`, keyed by object or generation UUID,
in addition to Python-Blosc2's sparse-frame `.b2lock`:

- Dataset path locks protect public carrier compare-and-swap.
- Object locks protect path binding, active-generation replacement, and coldify.
- Generation locks protect registry state, retirement, trash rename, measurement,
  and deletion.
- Sparse frame locks protect chunks and their index/bitmaps.

Rules:

1. Never acquire an OS lock from inside a SQLite transaction.
2. For multiple dataset paths, acquire path locks in lexical relative-path order.
3. Acquire dataset path, object, generation, then sparse frame lock in that order.
4. Begin any short SQLite transaction only after required OS locks are held; no
   code elsewhere may hold SQLite and then wait for those locks.
5. Perform DNS, network, serialization, directory walking, fsync, rename, and
   deletion outside SQLite transactions.
6. Never use elapsed time to steal ownership. Recovery attempts the stable OS lock
   non-blocking; failure means a live worker may still own the operation.

The initialization guard precedes path/object/generation locks. Acquire global
migration/prune guards before their subordinate locks and only non-blocking.
Refactor `publish()` into an outer lock-acquiring wrapper and an internal
`publish_locked()` with an explicit caller-held path-lock contract. Coldify and
CRUD hooks call the latter; reacquiring the same file lock through a second file
descriptor can deadlock. Do not trigger recursive global pruning from inside a
generation guard. Queue it after releasing request locks.

For v7, hold an exclusive generation guard for every sparse operation, including
attachment, cached reads, fills, export, and recovery. Close cache handles before
releasing it; no process may retain an open sparse handle across retirement or
trash deletion. This intentionally serializes operations on one generation.
Different generations remain concurrent. Optimize reader leases only after
benchmarking, with a separate documented deletion protocol.

The process-local `dataset_lock()` remains useful for avoiding event-loop thread
contention, but correctness must depend only on cross-process locks and SQLite.

## Resolution and generation attachment

Refactor `caterva2/services/remote_proxy.py` so `resolve()` receives the public
carrier path and storage coordinator. Preserve the current policy validation and
source creation before registry lookup.

For every request:

1. Inspect the public carrier without resolving embedded references.
2. Validate the descriptor and authorize the HTTPS destination.
3. Construct the pinned remote source, obtain its stable stamp, and validate
   geometry and configured limits.
4. Compute `spec_hash` and snapshot the public carrier generation.
5. Acquire the path lock, recheck the signature against the authorized snapshot,
   then find/create the binding under its object lock. If the snapshot changed,
   release locks and restart resolution with a bounded retry count; never attach
   storage authorized for the previous carrier to its replacement.
6. Reuse the active generation only when object ID, spec hash, source stamp,
   requested DISK policy, and runtime format all match.
7. If the source stamp, descriptor, geometry, policy, or externally observed
   carrier generation changed, build a new generation and retire the old one.
8. Attach with `blosc2.RemoteProxy.with_sparse_cache(authorized_source,
   runtime_cache_path, source_descriptor=payload["source"],
   carrier=warm_carrier, max_cache_bytes=requested_limit)`.

An old `ServerRemoteProxy` retains its generation ID. Before publishing any cache
mutation it revalidates that the row is still active while holding the generation
lock. If it has been retired, it returns source data without retention. Existing
readers may finish; an old writer cannot mutate or reactivate a newer generation.

A source with no stable stamp always executes without retention. Do not create a
runtime directory for it.

## First attachment and warm-carrier migration

Warm migration is lazy: it happens on the first authorized operation after upload,
not during upload, so uploading a descriptor never initiates an outbound request.
Only one process obtains a non-blocking global migration lock before duplicating a
warm carrier; concurrent requests serve its valid warm chunks read-only and fetch
misses without retention until migration finishes.

Under path, object, and new-generation locks:

1. Insert a `building` generation and `build` operation with a coarse estimate in
   a short transaction.
2. Close the transaction and call `with_sparse_cache(authorized_source, ...,
   source_descriptor=payload["source"], carrier=carrier)`. The helper copies only
   fetched state validated against the authorized source stamp.
3. Fsync the sparse generation and its parent, measure its full charge, and mark it
   active. Retire any previous active generation in the same transaction and
   release the estimate.
4. Build a cold contiguous carrier from the immutable public snapshot, preserving
   the RemoteProxy payload, fixed metadata, and every non-reserved user vlmeta.
   Remove fetched, cache-size, proxy-index, source-stamp, and dirty bookkeeping.
5. Cold-replace the public carrier through `StorageQuota.publish()` using its
   original signature. This is a shrinking ordinary write and retains existing crash
   recovery.
6. Update `carrier_generation` to the resulting signature. If compare-and-swap
   loses to a user replacement, keep the new sparse generation retired and never
   bind it to the replacement.

The active sparse cache is authoritative from step 3. A crash before coldify may
leave both warm copies, which is safe and fully charged. Startup recovery retries
coldify only when path, descriptor hash, source stamp, and recorded public
generation still match. It never reimports the warm carrier into an existing
generation.

Warm migration is necessary lifecycle work rather than a discretionary cache fill,
so customer quota may be exceeded temporarily by the duplicate. Record its full
estimate so other workers do not interpret that headroom as free. Before copying,
check configured operational free-space headroom; on denial, ENOSPC, or another
failure, retire and clean the partial generation, leave the public warm carrier
unchanged, and continue serving it read-only. A later request may retry migration.

Use `publish_locked()` for step 5 because the path guard is already held. Record
a `coldify` intent before publishing, with its ordinary-publication operation ID
durable before candidate rename. Reconciliation recovers this intent before
interpreting a changed public signature as an external replacement. A failed
coldify after successful activation leaves the active cache authoritative and
retries only coldify; it must not restart seed migration. The partial-generation
cleanup fallback above applies to failures before activation.

Pending coldify is persistent work even after a successful build has released its
estimate. Atomically replace the build operation with coldify intent on activation;
the unique operation-per-generation constraint must never erase unfinished work.
Subsequent requests serve that generation read-only until coldify completes, so a
fill cannot overwrite its recovery record. Back off retries on staging failures.

## Soft-quota fill protocol

Route `ServerRemoteProxy.quota_read()` through the sparse generation for every
retained DISK cache. A denied or failed retention attempt falls back to a
no-retention read and still returns the logical result.

For a logical slice or compressed chunk request:

1. Authorize and resolve the source as described above. No cache hit skips this.
2. Acquire the generation guard, revalidate its active binding, and keep it until
   handles are closed and accounting is finalized. Check and read a hit atomically
   under the frame lock; update coarse recency at most once per ten seconds and
   avoid SQLite admission. Recency persistence is best-effort and cannot turn a
   successful cached read into an error.
3. For a miss, compute a conservative estimate from missing compressed blocks or
   chunks and allocated metadata growth. Bound only the payload component by
   remaining per-proxy `max_cache_bytes` where finite. Estimates
   coordinate workers; they are not physical-growth guarantees.
4. In a short `BEGIN IMMEDIATE` transaction, sum dataset charge, generation
   charge, existing reservations, and remote estimates. If fills are suspended or
   projected usage exceeds quota, refuse retention. Otherwise insert a `fill`
   operation. With no configured quota, insert intent with estimate zero for
   recovery/accounting only.
5. Close the transaction, then call the supported combined fetch/read-and-mutate
   operation under the generation and frame guards. Durable SQLite intent must
   precede all writes; Python-Blosc2 brackets payload writes with its dirty marker.
   V7 does not assume a separate fetch/publish API. Holding these guards across
   network I/O is acceptable initially; benchmark its same-cache contention.
6. Enforce requested `max_cache_bytes` within that operation and obtain the logical
   result plus mutation report, including any evictions. A request larger than the
   per-proxy limit must still return its full result without retaining all of it.
7. Stat only affected payload and metadata entries, fsync as required, and update
   `payload_bytes`, `charge_bytes`, `inode_count`, and `touched`; delete the
   operation in one short transaction.
8. If actual aggregate usage exceeds quota, set `cache_fill_suspended=1` and queue
   bounded pruning. Return the already assembled logical result.

Any SQLite admission failure, busy timeout, stale generation, cache-write error,
ENOSPC, or retention denial falls back to a no-retention read. If data was already
assembled, return it directly; otherwise refetch without retention. Upstream
authorization, validation, or fetch errors still propagate normally.

Use a shared low watermark of 90% of configured quota. Once suspended, admit no
new discretionary cache fill until pruning or reconciliation observes usage at or
below that watermark. If ordinary dataset bytes alone exceed the watermark, cache
fills remain suspended. Do not claim a fixed maximum overshoot.

Admission estimates cover positive allocated growth, including index and directory
changes; a payload limit of zero does not imply zero filesystem overhead. Do not
subtract proposed evictions until they have actually freed charge. A finite
payload cap only bounds the payload component of the estimate.

## Accounting and reconciliation

Maintain charge from Python-Blosc2 mutation results where available. The current
release still performs a full-generation measurement after mutations as a safe
fallback; a cache hit does not stat every chunk. Coarse recency updates occur at
most every ten seconds.

Run full reconciliation:

- at coordinator startup under the initialization guard;
- after recovery of a dead operation;
- after an unexpected mutation/accounting error;
- periodically, default every five minutes, in bounded generation batches;
- on an administrator diagnostic request.

Reconciliation walks `.remote-cache` without following symlinks, measures files
and directories, repairs charge/inode counts, discovers registered directories
missing on disk, and moves unregistered validly-named directories to trash before
deletion. It never adopts an orphan as active from pathname alone.

It also checks every `remote_objects.path` against the current public file without
making outbound requests. A missing file, changed carrier signature, non-RemoteProxy
replacement, or changed descriptor/geometry retires the bound generation. A
matching descriptor whose remote source changed is detected later during an
authorized request, when obtaining the source stamp is permitted.

Skip NULL path bindings. Reconcile under the corresponding generation guard and
recover pending lifecycle/publication intents before comparing signatures. Do
not classify a live `building` directory as an orphan. Record discovered orphan
charge and durable cleanup intent before trash movement; missing files release
charge only after guarded verification. Charge object-parent directories once
per object in a separate accounting total included in `remote_cache_used`, rather
than once per generation. Shared cache/trash roots remain operational overhead.

Missing active storage retires the generation and makes subsequent reads cold.
Malformed paths, symlinks, descriptor mismatches, or unreadable sparse metadata
are quarantined in trash and remain charged until removed. A reduced quota sets
fill suspension and schedules pruning; startup does not delete ordinary datasets.

## Batched pruning

Trigger pruning after an overshooting fill, quota reduction, periodic
reconciliation above quota, or an admission refusal. Only one process obtains the
non-blocking global prune lock; others return their logical results.

Select active DISK generations ordered by coarse `touched`, excluding the
generation serving the triggering result. For each candidate:

1. Acquire its generation lock non-blocking and revalidate active state.
2. Compute bytes still needed to reach the aggregate low watermark and call
   `trim_cache(max(0, candidate_payload_bytes - needed), max_chunks=64)`.
3. Finalize affected-file charge and recency in a short transaction.
4. Stop after 64 chunks total, four generations, 100 ms of mutation work, or
   aggregate usage at/below the low watermark, whichever comes first.

Schedule another batch if still over quota. Empty generations remain valid cold
caches; zero-length chunk files and directory/index overhead remain charged.
Whole-generation retirement is a fallback only for corrupt, stale, or deleted
objects, not normal quota pressure.

The triggering generation is excluded only while its request guard is held. A
later background batch must include it; otherwise a customer with one cache can
remain suspended forever. Recompute actual allocated bytes after each batch:
payload bytes evicted are not necessarily physical bytes reclaimed. If no payload
can be reclaimed, stop rescheduling immediate batches, retain suspension, and
report irreducible metadata/dataset charge. Retry on periodic reconciliation or
a storage change. The 100 ms limit is checked between chunks and cannot bound a
single blocking filesystem operation.

Record a `prune` intent before calling `trim_cache()` and leave it for recovery
if eviction or accounting fails. Exclude generations with pending coldify/build
work; the unique operation row is never replaced by a competing operation.

## Export and download behavior

`include_cache=false` reads the public cold carrier snapshot and returns it without
opening or mutating the sparse generation. Before first migration, when the public
carrier may still be warm, build a cold snapshot locally by stripping its cache
state without resolving the source. Preserve requested policy and limit, user
metadata, and existing credential-free descriptor semantics.

Default warm export uses a controlled artifact:

1. Authorize the source and bind the active generation.
2. Reserve estimated artifact bytes in `remote_work` against
   `quota_work_bytes`. A cold export remains available if this reservation fails.
3. Create `.storage/exports/<operation-id>.b2nd` with `O_EXCL` and mode `0600`.
4. Under the generation/frame snapshot guard, serialize a contiguous carrier that
   merges valid runtime state and preserves public user metadata.
5. Fsync and close the artifact, record its exact length and a strong ETag derived
   from its bytes or immutable operation identity plus digest, then release the
   generation guard.
6. Serve all full and range responses from that one artifact. Never range-read a
   changing sparse directory.
7. Remove the artifact and `remote_work` row after response completion or
   cancellation. Startup removes artifacts whose OS owner lock is obtainable.

Acquire a stable per-export owner lock before inserting its reservation and hold
it through streaming and cleanup. Preserve actual artifact charge/reservation if
unlink fails. Cold snapshots that need serialization use the same work budget and
cleanup protocol. Reserve work jointly with ordinary publication candidates;
adjust an underestimated reservation before further growth or abort the export.
A staging failure returns a documented capacity error for a requested warm export;
do not silently substitute a cold result. Each HTTP request creates its own
snapshot: honor `If-Range` against that snapshot's ETag and return a full response
when it differs. Cross-request artifact reuse is outside v7 scope.

Do not hold a SQLite transaction while serializing or streaming. Export artifacts
are operational working storage and not customer `used` bytes, but they consume
the separately configured work budget. Document that soft customer quota cannot
prevent ENOSPC and that warm export may fail when staging headroom is unavailable.

## Dataset lifecycle

Route every coordinated public/shared/personal mutation through remote-cache
lifecycle hooks in `server.py` and `storage_quota.py`:

- **Upload or replacement:** after successful public publish, compare the prior
  registry binding. Retire old generations even when the new carrier has the same
  URL or descriptor. The replacement receives a new object ID on first resolve.
- **Delete:** acquire the dataset/object/generation guards, remove the public file
  through ordinary publication, mark the generation retired, atomically rename its
  directory to `.remote-cache/.trash/<generation-id>`, then delete outside the
  locks. Charge remains until deletion is measured complete.
- **Directory delete:** retain current file-by-file semantics; each RemoteProxy
  child executes the same retirement path.
- **Copy:** export a cold portable carrier locally, including from a legacy warm
  source, preserving descriptor, policy, limit, and user metadata. Do not resolve
  its source. Never share an object ID or sparse directory.
- **Move:** use current copy-plus-delete behavior. Destination starts with a new
  identity; source cache retires after the source generation is successfully
  removed.
- **Append/update/resize:** these already reject B2 object carriers where
  applicable. Any generic replacement route must still invoke retirement.
- **Customer removal/state cleanup:** retire every `remote_objects` row, move all
  generations to trash under their guards, and keep their charge until deletion.

Centralize these hooks in the storage coordinator rather than adding endpoint-only
cleanup. CLI, web, API, HDF5 workflows, and future writers must receive identical
behavior.

Remove `remote_objects` only after its public path is gone or no longer a matching
RemoteProxy and all of its generation rows have been deleted. A row with retired
or trash storage remains the ownership anchor during cleanup.

## Recovery state machine

At startup and on demand, inspect each `remote_operations` row. Attempt the
generation lock non-blocking; skip it if busy.

- `build`: if the directory is absent, delete the generation and release estimate.
  If present, use offline inspection/recovery to validate recorded spec/stamp and
  measure it. Activate only a durably completed build whose public binding still
  matches; retire an incomplete or ambiguous build. Source freshness is checked
  again on the next authorized request.
- `fill` or `prune`: recover offline under the frame lock. Python-Blosc2 clears untrusted
  fetched state left by a dirty owner. Measure the generation, finalize charge,
  clear the reservation, and leave it active only if its registry binding matches.
- `retire`: complete the active-to-retired transition and trash rename.
- `delete`: finish trash deletion, then remove the charged generation row.
- `coldify`: recover the public-path publication first, then compare the resulting
  carrier. Update its generation only if signature/digest and publication intent
  identify the expected cold artifact; otherwise
  retire the cache rather than binding it to unknown bytes.

Recovery is idempotent. At every state, the public cold/warm carrier remains an
independent source descriptor, so discarding a private generation cannot lose user
data. Never trust a fetched bit merely because SQLite says a fill completed; the
sparse frame and its dirty marker are authoritative for payload integrity.

## Configuration, diagnostics, and operations

The runtime has no public cache-backend selector. Sparse v7 is always used for
retained DISK caches; `NONE` and `MEMORY` retain nothing on the server. Keep only
the operational tuning settings that are implemented:

```toml
[server.remote_proxy]
cache_maintenance_seconds = 60
cache_low_watermark = 0.90
cache_reconcile_seconds = 300
cache_prune_chunks = 64
cache_prune_generations = 4
cache_min_free_bytes = "1G"
```

Sparse mode requires schema v2 and the pinned Python-Blosc2 API. The public carrier
remains the rollback-safe cold descriptor: discarding a private generation never
requires translating it back into a mutable public carrier.

The free-space check requires `free - estimated_operation_bytes` to remain above
`cache_min_free_bytes` before migration, rebuild, or export starts. It is a
best-effort ENOSPC guard rather than a reservation against unrelated processes.

Expose authenticated diagnostics containing aggregate dataset/cache charge,
outstanding estimates, work reservations, suspended state, active/retired/trash
generation counts, inode count, oldest recency, overshoot bytes and age, recovery
count, prune work, and last reconciliation. Do not expose source URLs, opaque
filesystem paths, or cache contents in the web viewer.

Log generation IDs and hashed object IDs, not credential-bearing URLs. Emit
metrics for fill hit/miss/refusal, actual-versus-estimated growth, dirty recovery,
pruned chunks/bytes, trash backlog, export size/time/RSS, quota overshoot magnitude,
and time to return below the low watermark.

Start maintenance tasks in the server lifespan and cancel/join them on shutdown.
Each worker may wake a task, but non-blocking global guards elect one executor for
each batch. Run blocking cache/filesystem work in the existing worker-thread path,
never on the event loop. Keep a persistent cursor for bounded reconciliation;
resume failed trash deletion with capped exponential backoff. Cancellation of an
HTTP request must not release generation/owner locks while its worker thread still
mutates storage. Let it finish accounting or leave recoverable intent before
closing handles. Cleanup failures remain visible and charged.

Define the durability boundary explicitly: after intent commit, write and fsync
payload/index state in the order required by Python-Blosc2, durably clear its dirty
marker, and only then finalize SQLite accounting. Fsync both parents of trash
renames. Validate this ordering against the actual helper implementation before
claiming power-loss safety; SIGKILL tests alone establish process-death behavior.

## Implementation sequence

1. Land and pin the supported Python-Blosc2 accounting, pruning, and export APIs.
2. Add schema-v2 lifecycle initialization and combined dataset/cache accounting.
3. Implement private path validation, locks, operation recovery, reconciliation,
   generation creation, warm migration, coldify, retirement, and trash cleanup.
4. Route sparse hits/fills, logical slices, and `/api/chunk` through soft
   admission with no-retention fallback.
5. Add bounded pruning, hysteresis, maintenance scheduling, diagnostics, and
   warm export artifacts with ranges, ETags, cancellation, and work reservations.
6. Run fault-injection, multiprocess, endpoint, and v5 comparison benchmarks.
   This sequence is complete for the current release candidate; the remaining
   work is listed under follow-up items below.

## Follow-up items after the current implementation

These items are useful improvements, but do not block the current sparse-default
release candidate:

- Replace full-generation post-mutation measurement with incremental accounting
  from Python-Blosc2 mutation reports. The current scan is correct and bounded by
  the generation size, but it adds latency to fills and pruning.
- Add bounded reconciliation cursors and benchmark millions of logical chunks,
  sparse metadata, inode counts, peak RSS, and long-running quota convergence.
- Measure multiprocess throughput and same-generation contention separately from
  the existing correctness tests.
- Add power-loss testing for filesystem and SQLite durability. Process-death
  recovery is covered; the implementation does not claim crash atomicity after a
  power failure.
- Tune staging estimates and warm-export reservations instead of reserving the
  full work budget for one export.
- Retain the internal contiguous compatibility path for deterministic comparison
  and explicit rollback tests; it is not a public Caterva2 deployment setting.

## Caterva2 module map

- `caterva2/services/remote_proxy.py`: authorized-source attachment, generation
  binding, sparse read/chunk routing, warm migration, coldify, pruning adapter,
  export snapshot, and no-retention fallback.
- `caterva2/services/storage_quota.py`: schema migration, always-available storage
  coordinator, combined admission, private measurement, operation recovery,
  reconciliation, retirement, trash deletion, and diagnostics.
- `caterva2/services/server.py`: initialize the coordinator, pass public paths into
  resolution, replace `quota_read()` routing, and call centralized lifecycle hooks
  from write/remove/copy/move paths and download endpoints.
- `caterva2/services/settings.py`, `caterva2-server.sample.toml`, and
  `doc/utilities/cat2-server.md`: runtime tuning, hysteresis, reconciliation, work-budget,
  accounting, cleanup, and operational-headroom configuration/documentation.
- `caterva2/tests/test_remote_proxy.py`: source policy, sparse behavior, generation
  binding, migration, export, and fallback tests.
- `caterva2/tests/test_storage_quota.py` and
  `caterva2/tests/test_storage_quota_api.py`: schema, multiprocess admission,
  lifecycle, recovery, pruning, endpoint, and quota-accounting tests.
- `examples/benchmarks/remote_proxy_v7.py` and
  `examples/benchmarks/remote_proxy_v7.md`: server-boundary v5/v7 comparison using
  the same deterministic local range source and machine-readable raw results.

## Required tests

Use deterministic local range-capable sources before external HTTPS tests.

Add targeted regressions for the review's implementation boundaries:

- Replace/delete/recreate the same public path while old trash cannot be removed;
  a new UUID can bind the path and both generations remain charged.
- Pause between hit detection and data extraction while another process attempts
  eviction/deletion; no unreserved fill or access to removed handles occurs.
- Crash after cold publication but before binding finalization; recovery recognizes
  the exact artifact, while a same-descriptor user replacement gets a new identity.
- Run startup recovery with all outbound transport constructors forbidden.
- Overshoot with one cache, and with only irreducible metadata remaining; pruning
  either reaches the watermark or reports suspension without a busy loop.
- Cancel an in-flight read/export with a live worker thread; locks and reservations
  remain owned until mutation/stream cleanup ends.
- Verify schema initialization and quota-disabled lifecycle.

- Two and eight processes filling the same generation, different generations,
  and overlapping partial blocks; readers overlap eviction without corrupt or
  zero-filled results.
- Competing fills and ordinary uploads cannot all treat the same quota headroom as
  free. Actual overshoot is charged and converges through bounded pruning.
- Worker death before/after every dirty marker, chunk truncate/write, index write,
  fetched bitmap, accounting finalization, active-generation switch, cold publish,
  trash rename, and deletion boundary.
- Recovery never steals a live worker's reservation or generation and is
  idempotent across repeated restarts.
- Source replacement with identical geometry creates a new generation. Geometry,
  descriptor, policy, and limit changes cannot attach stale storage.
- Authorization executes on cache hits; allowlist removal immediately prevents
  use of already-warm data.
- Warm migration preserves valid uploaded chunks and user metadata, coldifies the
  public carrier, counts both copies during the transition, and never resurrects
  an evicted seed chunk.
- `include_cache=false` is cold and non-mutating. Warm full/range downloads use one
  stable artifact and ETag; cancellation, timeout, restart, and ENOSPC remove or
  recover reservations and artifacts.
- Delete, directory delete, replacement, move, copy, and customer removal produce
  the lifecycle specified above. Private bytes remain charged until trash removal.
- Startup inventory includes registered and orphan private generations while API,
  root listings, search, providers, and the web viewer never reveal them.
- Bounded and unlimited per-proxy limits remain distinct from customer quota.
  MEMORY and NONE retain nothing.
- SQLite busy/error, malformed registry rows, missing directories, symlinks,
  permission failures, and quota disabled/enabled transitions preserve logical
  reads and produce reconciled accounting.

## Benchmarks and acceptance

Repeat the v6 cold fill, warm hit, LRU churn, and round-robin partial-block tests
through Caterva2 endpoints. Add concurrent same/different proxy fills, millions of
logical chunks with few resident chunks, pruning, startup reconciliation, cold and
warm export, range download, peak RSS, metadata operations, inode count, actual
allocated blocks, estimate error, and quota overshoot duration.

Accept v7 when:

- ordinary fills, partial growth, and eviction never rewrite unrelated cached
  payload;
- operation memory does not scale with the remaining carrier tail;
- no stale or dirty fetched bit can serve incomplete data after process death;
- every retained private byte is eventually reflected in account usage, and
  overshoot converges without failing logical reads;
- delete and replacement cannot leave an unregistered, uncharged active cache;
- public carriers remain portable and private directories remain unreachable from
  API and web namespaces;
- warm/cold export semantics, authorization, validators, geometry checks, MEMORY
  behavior, and client defaults remain compatible;
- discarding sparse state can serve the public carrier without migration.

No hard RAM, fixed overshoot, or physical-volume guarantee is implied. Operational
monitoring and free-space headroom remain required even with correct soft-quota
accounting.
