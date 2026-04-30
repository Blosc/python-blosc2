# OPSI indexing implementation plan

## Goal

Implement a new index construction method, tentatively named **OPSI**. OPSI builds a completely sorted index (CSI) by repeatedly combining:

1. **Stage 1**: sort every chunk completely, carrying positions together with values.
2. **Stage 2**: compute a global block relocation order from per-block boundary keys, alternating between block minimum and block maximum values.

Unlike the in-memory `full` builder, OPSI should avoid sorting the full value array in memory. The current persistent/on-disk `full` builder already uses an out-of-core global-sort path; OPSI is intended as an alternative full-index builder with global memory proportional to the number of blocks, not the number of indexed elements.

The final output should be usable as a full sorted index: sorted values plus original/global positions.

## Existing context

Current index kinds are:

- `summary`
- `bucket`
- `partial`
- `full`

Only `full` currently guarantees a completely sorted global index.

Relevant files/functions:

- `src/blosc2/indexing.py`
  - `_build_full_descriptor()`
  - `_build_full_descriptor_ooc()`
  - `_build_partial_descriptor()`
  - `_build_partial_descriptor_ooc()`
  - `_sort_chunk_intra_chunk()`
  - `_build_partial_chunk_payloads_intra_chunk()`
  - `_prepare_chunk_index_payload_sidecars()`
  - `_finalize_chunk_index_payload_storage()`
  - `_read_ndarray_linear_span()`
  - `_write_ndarray_linear_span()`
  - `_rebuild_full_navigation_sidecars_from_handle()` / `_rebuild_full_navigation_sidecars_from_path()`
- `src/blosc2/indexing_ext.pyx`
  - existing stable merge-sort helpers for values + positions
  - new keysort-like helpers should live here
- `src/blosc2/ctable.py`
  - index kind validation
  - persistent index construction dispatch
  - catalog descriptor handling

## Product/API decision: OPSI is a full-index build method

OPSI should be implemented as an alternative **builder** for a `full` index, not as a new persistent index kind.

The semantic meaning of `kind="full"` remains:

```text
the final index is completely globally sorted and can provide sorted values plus original/global positions
```

The build method controls how that full/CSI index is produced:

```python
arr.create_index(kind="full")  # current behavior, defaults to method="global-sort"
arr.create_index(
    kind="full", method="global-sort"
)  # explicit current full-index builder
arr.create_index(kind="full", method="opsi")  # new OPSI builder
```

So OPSI does **not** replace the current full-index construction path. The current method remains available and is named `global-sort` when an explicit method name is needed.

OPSI internally builds the same sidecars as a full index and stores the final descriptor as a normal `full` descriptor:

```python
{
    "kind": "full",
    "full": {
        "values_path": ...,
        "positions_path": ...,
        "runs": [],
        "next_run_id": 0,
        "l1_path": ...,
        "l2_path": ...,
        "sidecar_chunk_len": ...,
        "sidecar_block_len": ...,
        "build_method": "opsi",
        "opsi_cycles": ...,
    },
}
```

For indexes built by the current full builder, descriptor metadata can use:

```text
"build_method": "global-sort"
```

or omit `build_method` for backward compatibility and treat missing metadata as `global-sort`.

This avoids touching most query-planning and ordered-access code, because the final OPSI result is logically a full sorted index.

A future `IndexKind.OPSI` is not recommended unless there is a strong user-facing reason. If it is ever added, it should be normalized to `kind="full", method="opsi"` rather than stored as a separate persistent catalog kind.

## Core algorithm

Let:

```text
N = number of indexed elements
C = chunk_len = int(array.chunks[0])
B = block_len = int(array.blocks[0])
nchunks = ceil(N / C)
nblocks = ceil(N / B)
```

OPSI maintains one completed Stage-1 sidecar set at a time:

```text
values_sidecar
positions_sidecar
block_mins_sidecar
block_maxs_sidecar
```

Stage 2 is **not materialized**. It computes a block permutation in memory and feeds the next Stage 1.

Pseudo-code:

```python
cycle = 0
source = original_array_or_previous_stage1_sidecars
block_order = None  # identity for first Stage 1

while True:
    current = build_stage1_sidecars(source, block_order)

    if is_csi(current.block_mins, current.block_maxs):
        finalize_as_full_index(current)
        remove_previous_stage1_sidecars()
        break

    if cycle >= max_cycles:
        cleanup_current_sidecars_if_needed()
        raise RuntimeError("OPSI did not converge within max_cycles")

    if cycle % 2 == 0:
        keys = load_block_mins(current)
    else:
        keys = load_block_maxs(current)

    block_order = sort_block_ids_by_keys(keys)

    remove_previous_stage1_sidecars()
    source = current
    cycle += 1
```

The alternation is:

```text
cycle 0 relocation: block mins
cycle 1 relocation: block maxs
cycle 2 relocation: block mins
cycle 3 relocation: block maxs
...
```

## Stage 1 details

Stage 1 receives a logical stream of chunks. For the first cycle this stream is the original target values in original order. For later cycles this stream is built by gathering blocks from the previous Stage-1 sidecars according to the most recent Stage-2 `block_order`.

For each destination chunk:

1. Materialize `chunk_values` in memory.
2. Materialize `chunk_positions` in memory.
3. Sort `chunk_values` and carry `chunk_positions` in lockstep.
4. Write sorted values and positions to new sidecars.
5. Compute per-block min/max from the sorted chunk and write/update min/max sidecars.

### First Stage 1

Read values from the indexed target. Positions are global row positions and can be synthesized:

```python
chunk_positions = np.arange(chunk_start, chunk_stop, dtype=np.int64)
```

or, if the sorting helper returns an order/position vector, converted to global positions:

```python
sorted_positions = chunk_start + local_order
```

### Later Stage 1 cycles

Build each destination chunk by gathering relocated source blocks from the previous Stage-1 sidecars.

For destination block slot `dst_block_id`:

```python
src_block_id = block_order[dst_block_id]
src_start = src_block_id * B
src_stop = min(src_start + B, N)
```

Read the source values and source positions block, then copy both into the destination chunk buffer at the destination block offset.

Positions must always be copied and sorted together with their corresponding values.

### Per-block min/max sidecars

After sorting a chunk, each block inside that chunk is sorted. For every global block:

```python
block_min = sorted_block[0]
block_max = sorted_block[-1]
```

Store these in sidecars:

```text
opsi_stageX.block_mins
opsi_stageX.block_maxs
```

These sidecars are regenerated after every Stage 1. They must not be reused across cycles, because Stage 1 re-sorts mixed chunks and therefore changes block boundaries.

## CSI check

After every Stage 1, check whether the current Stage-1 sidecars represent a completely sorted global index.

The condition is:

```python
block_maxs[i] <= block_mins[i + 1]
```

for all adjacent blocks.

For integer, bool, datetime, and timedelta dtypes, normal `<=` semantics are sufficient.

For floating dtypes, use ordered NaN-aware comparison:

```text
finite values < NaN
NaN == NaN for ordering purposes
```

So:

```text
finite <= finite  -> normal comparison
finite <= NaN     -> True
NaN <= NaN        -> True
NaN <= finite     -> False
```

Do **not** use raw NumPy `<=` for float CSI checks, because `np.nan <= np.nan` is false.

The CSI check should be streamable. It does not need both complete min/max arrays in memory. It can read min/max sidecars by spans and compare boundaries, carrying the previous block max across span boundaries.

## Stage 2 details

Stage 2 chooses one scalar key per block:

```python
keys = block_mins  # min cycle
# or
keys = block_maxs  # max cycle
```

Then it sorts block ids by these keys.

The result is:

```python
block_order[dst_block_id] = src_block_id
```

This block order is then consumed by the next Stage 1. No relocated values/positions sidecars are written in Stage 2.

### Block id dtype

Use `int64` for block ids/order.

Rationale:

- `nblocks <= 2**32 - 1` cannot be guaranteed.
- Source/destination offsets are computed as `block_id * block_len` and must not overflow.
- NumPy and Cython indexing paths naturally use `int64` / `intp` on 64-bit systems.
- Simpler and safer than conditional `uint32` block ids.

### Stage 2 sorting implementation

Preferred implementation: a keysort-like Cython helper:

```python
indexing_ext.keysort_keys_indices(keys, block_ids)
```

This sorts `keys` in place and carries `block_ids` with it. After sorting, `block_ids` is the block relocation order.

Comparator:

```text
primary key: block key value
secondary key: current block id
```

For floats, use the same NaN convention:

```text
finite values < NaN
NaNs compare equal by value, then tie-break by block id
```

Fallback implementation:

```python
block_order = np.argsort(keys, kind="stable").astype(np.int64, copy=False)
```

The fallback is correct but can allocate more memory and may be slower.

## Keysort / paired-sort implementation

OPSI should add keysort-style Cython primitives in `src/blosc2/indexing_ext.pyx`.

### Stage 1 paired value/position sort

Add a helper similar to PyTables' `keysort`, but adapted for OPSI:

```cython
def keysort_values_positions(np.ndarray values, np.ndarray positions):
    """Sort values in-place and carry positions with the same swaps."""
```

Properties:

- in-place
- numeric dtypes supported by the existing indexing extension
- positions are `int64` initially
- comparator sorts by `(value, position)`
- float comparator sorts finite values before NaNs
- NaN values are considered equal for primary-key ordering and tie-broken by position

Comparator for non-floating dtypes:

```text
(a_value, a_pos) < (b_value, b_pos)
```

Comparator for floats:

```text
a_nan = isnan(a_value)
b_nan = isnan(b_value)

if a_nan:
    if b_nan:
        return a_pos < b_pos
    return False        # NaN after finite
if b_nan:
    return True         # finite before NaN
if a_value < b_value:
    return True
if a_value > b_value:
    return False
return a_pos < b_pos
```

This makes algorithmic stability unnecessary. Even with quicksort, equal values get deterministic ordering via positions.

### Stage 2 key/block-id sort

Add a second helper:

```cython
def keysort_keys_indices(np.ndarray keys, np.ndarray indices):
    """Sort scalar keys in-place and carry int64 indices with them."""
```

This can share most implementation with `keysort_values_positions`, with `indices` as the tie-breaker.

### Why quicksort/keysort

Compared with:

```python
order = np.argsort(values, kind="stable")
sorted_values = values[order]
sorted_positions = positions[order]
```

an in-place keysort avoids:

- explicit `order` allocation
- two gather operations
- merge-sort temporary buffers
- extra memory bandwidth

This is important for OPSI because chunk sorting happens every cycle and positions must travel with values.

### Relationship to existing merge-sort helpers

`indexing_ext.pyx` already contains stable merge-sort helpers. They are useful references and can remain as fallback paths, but OPSI should use the new keysort path for memory efficiency.

The existing float NaN ordering logic should be mirrored for consistency.

## Reading and writing sidecars

### Reading blocks

For Stage 1 after the first cycle, read source blocks from previous sidecars.

`get_1d_span_numpy()` is the preferred low-level primitive for 1-D sidecar reads. Existing wrapper `_read_ndarray_linear_span()` is also useful because it handles spans crossing sidecar chunk boundaries and includes fallback behavior.

If OPSI sidecar geometry enforces:

```python
sidecar_chunk_len % block_len == 0
```

then every block lies inside one sidecar chunk, and direct `get_1d_span_numpy()` is ideal.

### Writing chunks

Build one destination chunk in memory and write it once to each sidecar:

```python
_write_ndarray_linear_span(values_handle, chunk_start, sorted_values)
_write_ndarray_linear_span(positions_handle, chunk_start, sorted_positions)
```

This minimizes small writes.

### Coalescing reads

Initial implementation can read one source block at a time.

Later optimization: detect runs where consecutive destination block slots map to consecutive source blocks, e.g.:

```text
dst blocks: 0, 1, 2
src blocks: 8, 9, 10
```

and read one larger span instead of three block reads.

## Sidecar lifecycle and failure behavior

During each Stage 1, create a fresh sidecar set:

```text
values
positions
block_mins
block_maxs
```

Only after the new Stage-1 sidecars are fully written and validated should the previous Stage-1 sidecars be removed.

If the process fails mid-build, the expected recovery is to restart the build, not to resume from a partially completed Stage 2. Therefore Stage 2 does not need persistent sidecars.

For persistent arrays, temporary sidecar names should include a token, cycle, and stage identifier, for example:

```text
<token>.opsi.cycle000.values
<token>.opsi.cycle000.positions
<token>.opsi.cycle000.mins
<token>.opsi.cycle000.maxs
```

Finalized full-compatible sidecars should use the normal full sidecar names:

```text
full.values
full.positions
full_nav.l1
full_nav.l2
```

or should be copied/renamed into that canonical layout.

Temporary sidecars must be deleted on successful completion and best-effort deleted on exceptions.

## Navigation sidecars

Once CSI is reached, build full navigation sidecars for the final sorted values:

```text
full_nav.l1
full_nav.l2
```

Use existing helpers where possible:

- `_rebuild_full_navigation_sidecars_from_handle()` if final values sidecar is already open
- `_rebuild_full_navigation_sidecars_from_path()` if final values path is known
- `_rebuild_full_navigation_sidecars()` if final values are in memory, which is less likely for OPSI

The final `full` descriptor must include:

```text
"sidecar_chunk_len": int(values_sidecar.chunks[0])
"sidecar_block_len": int(values_sidecar.blocks[0])
"l1_path": ...
"l2_path": ...
"l1_dtype": ...
"l2_dtype": ...
```

as expected by existing full-index query paths.

## Memory requirements

OPSI's memory pressure is primarily:

1. one chunk of values + positions
2. one full block-key array + one full block-order array during Stage 2

Let:

```text
N = number of elements
B = block length
C = chunk length
S = value dtype itemsize
P = position itemsize, initially 8
K = block id itemsize, 8
nblocks = ceil(N / B)
```

Approximate peak memory with keysort:

```text
C * (S + P) + nblocks * (S + K)
```

plus small temporary block buffers and sidecar API overhead.

If Stage 1 used merge sort or argsort, extra chunk-sized temporary arrays would be required. The keysort approach is therefore strongly preferred.

### Example: float64/int64 values, int64 positions, block_len=4096

```text
S = 8
P = 8
K = 8
Stage 2 global memory = nblocks * 16 bytes
```

Approximate Stage-2 key/order memory:

| N items | nblocks | key + order memory |
|---:|---:|---:|
| 100 million | ~24,414 | ~0.39 MB |
| 1 billion | ~244,141 | ~3.9 MB |
| 10 billion | ~2.44 million | ~39 MB |
| 100 billion | ~24.4 million | ~390 MB |

For smaller block sizes memory grows proportionally. For example, `block_len=256` uses 16x more block-key/order memory than `block_len=4096`.

## Convergence and safeguards

Add a `max_cycles` safeguard.

Initial options:

- internal default, e.g. `max_cycles=32`
- user-visible keyword for experimental tuning
- environment variable for testing

If CSI is not reached within `max_cycles`, the initial implementation falls back to the existing `global-sort` full-index builder so that `method="opsi"` still produces a valid full index. The descriptor records this with:

```text
"opsi_fallback": "global-sort"
```

A stricter future mode may instead raise:

```python
RuntimeError("OPSI did not converge within max_cycles")
```

Track metadata:

```text
"build_method": "opsi"
"opsi_cycles": cycles_completed
"opsi_max_cycles": max_cycles
```

## API integration

Use `method` as the explicit full-index build-method keyword:

```python
arr.create_index(kind="full")  # equivalent to method="global-sort"
arr.create_index(kind="full", method="global-sort")  # current full builder
arr.create_index(kind="full", method="opsi")  # OPSI builder
```

The `method` keyword should only apply to `kind="full"` initially. Passing `method="opsi"` with `summary`, `bucket`, or `partial` should raise a clear `ValueError`.

Descriptor metadata:

```text
"build_method": "global-sort"  # current/default full builder
"build_method": "opsi"         # OPSI builder
```

For backward compatibility, a missing `build_method` in an existing full descriptor should be interpreted as `global-sort`.

Do not add a persistent `kind="opsi"` catalog entry in the initial implementation. If a user-facing `IndexKind.OPSI` is later desired, normalize it to `kind="full", method="opsi"` during argument handling.

## Implementation steps

### 1. Add Cython keysort helpers

In `src/blosc2/indexing_ext.pyx`:

- add in-place keysort for supported value dtypes + `int64` positions
- comparator sorts by `(value, position)`
- float comparator handles NaNs as last
- add in-place keysort for block keys + `int64` block ids
- add tests against NumPy reference ordering

Reference behavior:

```python
expected_order = np.lexsort((positions, values))
```

For floats with NaNs, use a reference that maps NaNs to a final ordering bucket and then tie-breaks by positions.

### 2. Add OPSI Stage-1 builder helpers

In `src/blosc2/indexing.py`:

- helper to create temporary Stage-1 sidecar set
- helper to build first Stage 1 from original target values
- helper to build later Stage 1 by gathering blocks from previous sidecars according to `block_order`
- write values/positions chunk-by-chunk
- generate block min/max sidecars every Stage 1

Possible internal structure:

```python
@dataclass
class OpsiStageSidecars:
    values_path: str | None
    positions_path: str | None
    mins_path: str | None
    maxs_path: str | None
    values_handle: blosc2.NDArray | None
    positions_handle: blosc2.NDArray | None
    mins_handle: blosc2.NDArray | None
    maxs_handle: blosc2.NDArray | None
    chunk_len: int
    block_len: int
    nblocks: int
```

### 3. Add CSI check helper

Implement:

```python
def _opsi_is_csi(mins_handle, maxs_handle, dtype) -> bool: ...
```

- stream sidecars by spans
- use raw `<=` for ordered non-float types
- use NaN-aware ordered comparison for float types

Consider adding a Cython helper for vectorized float CSI checks if Python/NumPy span checks are too slow.

### 4. Add Stage-2 block-order helper

Implement:

```python
def _opsi_compute_block_order(stage, use_max: bool) -> np.ndarray:
    keys = _read_sidecar_span(
        stage.maxs_handle if use_max else stage.mins_handle, 0, nblocks
    )
    block_ids = np.arange(nblocks, dtype=np.int64)
    indexing_ext.keysort_keys_indices(keys, block_ids)
    return block_ids
```

Fallback to NumPy if unsupported dtype:

```text
order = np.argsort(keys, kind="stable")
return order.astype(np.int64, copy=False)
```

For deterministic equal-key behavior, prefer keysort comparator `(key, block_id)`.

### 5. Add OPSI full descriptor builder

Implement something like:

```python
def _build_full_descriptor_opsi(
    array,
    target,
    token,
    kind,
    dtype,
    persistent,
    cparams=None,
    max_cycles=32,
) -> dict: ...
```

It should:

1. run OPSI cycles
2. finalize/copy final Stage-1 values/positions as full sidecars
3. build full navigation sidecars
4. return a normal `full` descriptor with OPSI metadata
5. clean temporary sidecars

### 6. Wire API/build dispatch

Add/propagate the `method` keyword for full indexes:

```python
create_index(kind="full", method="global-sort")  # current full builder
create_index(kind="full", method="opsi")  # OPSI builder
```

Default behavior remains unchanged:

```python
create_index(kind="full")  # method="global-sort"
```

Update:

- full-index method normalization/validation
- CTable `_build_index_persistent()` dispatch
- top-level `create_index()` dispatch
- descriptor creation kwargs for rebuild/compact so `method="opsi"` survives rebuilds
- descriptor metadata so current full indexes are identified as `global-sort` when metadata is present

Do not add `opsi` to persistent index-kind validation. OPSI is a method for creating `kind="full"`.

### 7. Tests

Add unit tests covering:

- small arrays with random integers
- duplicate-heavy arrays
- already sorted arrays
- reverse sorted arrays
- arrays with many equal block mins/maxs
- float arrays with NaNs
- final values sidecar is globally sorted according to expected ordering
- positions sidecar maps sorted values back to original values
- CSI condition passes
- query results match existing full index for predicates
- ordered indices match expected sorted positions
- persistent table index create/drop/rebuild if integrated into CTable
- failure on too-small `max_cycles` cleans temporary sidecars best-effort

Reference check:

```python
expected_order = np.argsort(values, kind="stable")
expected_values = values[expected_order]
expected_positions = expected_order.astype(np.int64)
```

For OPSI with `(value, position)` comparator, this should match stable NumPy ordering for equal values when original positions are increasing.

For floats with NaNs, ensure NumPy reference places NaNs last and duplicate NaNs retain position order.

### 8. Benchmarks

Compare:

- existing `full` builder
- OPSI builder with keysort
- OPSI fallback using argsort

Datasets:

- uniform random int64/float64
- already sorted
- reverse sorted
- low-cardinality duplicates
- data with NaNs
- large out-of-core persistent arrays

Measure:

- build time
- peak memory
- compressed sidecar sizes
- cycles to convergence
- query performance after build

## Future work: OOC Stage 2 block-key sorting

The initial OPSI design may compute Stage 2 block order in memory:

```python
keys = read_all(block_mins_or_maxs)
block_ids = np.arange(nblocks, dtype=np.int64)
keysort_keys_indices(keys, block_ids)
```

This requires memory proportional to:

```text
nblocks * (value_itemsize + 8)
```

For extremely large persistent arrays, even this block-level memory can become too large. A future OOC Stage 2 can avoid loading all block keys/order entries at once by using an external merge-sort subroutine over block boundary keys.

This would mimic the current full OOC builder's sorted-run structure, but over **block keys**, not over all indexed values.

### OOC Stage 2 outline

For each Stage 2 cycle:

1. Choose the key sidecar:

   ```text
   key_sidecar = block_mins if use_min_cycle else block_maxs
   ```

2. Read the key sidecar in runs:

   ```text
   for run_start in range(0, nblocks, run_items):
       keys = read_span(key_sidecar, run_start, run_stop)
       block_ids = run_start + np.arange(run_stop - run_start, dtype=np.int64)
       keysort_keys_indices(keys, block_ids)
       materialize_sorted_block_key_run(keys, block_ids)
   ```

3. Merge sorted block-key runs pairwise, reusing the same pair comparator:

   ```text
   primary key: block min/max value
   secondary key: block id
   NaNs sort last for float keys
   ```

4. The final merged run's `positions` sidecar is the block-order stream:

   ```text
   block_order[dst_block_id] = src_block_id
   ```

5. The next Stage 1 reads the block-order stream by destination chunk/block span instead of requiring the full `block_order` array in memory.

### Relationship to merge sort

This does not make OPSI a global merge-sort index builder. The current full OOC builder globally sorts all `N` value-position pairs. OOC Stage 2 for OPSI would sort only:

```python
nblocks = ceil(N / block_len)
```

boundary-key/block-id pairs. OPSI remains an iterative local-sort plus block-relocation algorithm; external merge sort is only an implementation detail for producing the block relocation order when `nblocks` is too large for memory.

### Abstraction needed

To support both in-memory and OOC Stage 2, introduce a small block-order abstraction:

```python
class OpsiBlockOrder:
    def read_span(self, start_block: int, stop_block: int) -> np.ndarray: ...
```

Implementations:

- memory-backed: wraps a NumPy `int64` array
- sidecar-backed: wraps the final OOC block-order positions sidecar

Then Stage 1 can gather relocated blocks without caring whether Stage 2 was memory or OOC.

### Tradeoff

OOC Stage 2 lowers peak memory but adds temporary disk IO every OPSI cycle. It should therefore be selected only when necessary, for example under `build="ooc"` or when `build="auto"` detects that `nblocks * (value_itemsize + 8)` exceeds a configured memory threshold.

## Future work: reuse keysort in existing index builders

Even if OPSI is not competitive as a general full-index builder, the in-place paired keysort primitives developed for OPSI can be useful elsewhere in the indexing engine.

The useful primitives are:

```python
indexing_ext.keysort_values_positions(values, positions)
indexing_ext.keysort_keys_indices(keys, indices)
```

They sort in place by:

```text
primary key: value/key
secondary key: position/index
```

with float NaNs ordered last. This gives deterministic stable-sort-like semantics when positions/indices are unique and initialized in original order.

### Candidate integration points

#### Full in-memory builder

Current full in-memory construction uses a global argsort-like pattern:

```python
order = np.argsort(values, kind="stable")
sorted_values = values[order]
positions = order.astype(np.int64)
```

A keysort-based alternative is:

```python
sorted_values = values.copy()
positions = np.arange(values.size, dtype=np.int64)
indexing_ext.keysort_values_positions(sorted_values, positions)
```

This avoids the explicit `order` array and avoids a separate value gather. It should be benchmarked against NumPy's highly optimized argsort, especially for large arrays where memory bandwidth and temporary allocation pressure matter.

#### Full OOC run generation

Current full OOC construction creates sorted runs from spans of the base array. This is a strong candidate for keysort:

```python
run_values = _slice_values_for_target(...).copy()
run_positions = np.arange(run_start, run_stop, dtype=np.int64)
indexing_ext.keysort_values_positions(run_values, run_positions)
materialize_sorted_run(run_values, run_positions)
```

This may reduce per-run temporary memory compared with merge-sort-style helpers and can lower memory bandwidth during run creation. The global OOC merge phase remains unchanged.

#### Partial chunk-local sorting

Partial indexes sort chunk-local payloads and carry local positions. This is also a natural fit:

```python
chunk_values = values.copy()
chunk_positions = np.arange(chunk_size, dtype=position_dtype)
keysort_values_positions(chunk_values, chunk_positions)
```

However, current partial indexes use compact position dtypes such as `uint8`, `uint16`, or `uint32` when possible. The initial keysort helper expects `int64` positions, so this integration should either:

1. extend keysort to support compact position dtypes while using widened comparisons for tie-breaking, or
2. restrict keysort use to paths where `int64` positions are already acceptable.

Keeping compact position sidecars is important for partial-index size.

#### Bucket and metadata sorting

`keysort_keys_indices()` may also be useful for sorting smaller metadata arrays, boundary-key arrays, or future bucket/navigation construction steps where scalar keys need to carry ids or offsets.

### Benchmarking needed

Before replacing existing sort paths, benchmark:

- current full in-memory builder vs keysort full in-memory builder
- current full OOC run generation vs keysort run generation
- current partial chunk sorting vs keysort chunk sorting, after compact position dtype support

Datasets should include:

- random numeric data
- duplicate-heavy data
- already sorted data
- reverse sorted data
- float data with NaNs

Expected outcome:

- full OOC run generation: likely improvement
- partial chunk-local sorting: likely improvement after compact position dtype support
- full in-memory builder: uncertain; depends on NumPy argsort speed vs reduced allocation/memory traffic

## Open questions

1. Default `max_cycles` value.
2. Whether non-persistent/in-memory arrays should support OPSI initially or only persistent/OOC paths.
3. Whether final sidecars should be renamed/copied into canonical `full.*` paths or whether descriptor can point directly to final OPSI temporary paths.
4. Whether to add Cython coalesced block-gather helpers after the Python implementation is correct.
5. Memory threshold policy for choosing in-memory vs OOC Stage 2 under `build="auto"`.

## Summary

OPSI should be implemented as an iterative chunk-local sort plus global block-relocation method. Stage 1 produces sorted chunks and regenerates block min/max sidecars. Stage 2 sorts one scalar key per block, alternating min and max keys, and produces only an in-memory block permutation. The next Stage 1 gathers blocks according to that permutation and immediately re-sorts chunks.

Correctness depends on:

- positions always traveling with values
- block min/max sidecars being regenerated after every Stage 1
- CSI being checked as `max(block_i) <= min(block_i + 1)`
- NaNs being ordered consistently as last

Efficiency depends on:

- not materializing Stage 2
- using `get_1d_span_numpy()` / `_read_ndarray_linear_span()` for block reads
- writing full chunks to sidecars
- using in-place keysort-style paired sorting for values/positions and keys/block ids

The product integration is to expose OPSI as `method="opsi"` for `kind="full"`, while the current full builder remains available as `method="global-sort"` and remains the default. OPSI emits normal full-compatible sidecars with metadata recording `build_method="opsi"`.
