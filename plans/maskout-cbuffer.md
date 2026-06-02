# Plan: `b2nd_get_maskout_cbuffer()` — block-maskout sparse gather

## Status

Proposal / analysis. **Strongly validation-first, and the maskout API is
deprioritized** in favor of reusing the existing element-coords sparse gather
(`b2nd_get_sparse_cbuffer` / `blosc2_schunk_get_sparse_buffer`).

**Key prior-art finding (2024):** c-blosc2 *already contains* maskout-based block
reading in `blosc/schunk.c` (the `schunk_get_slice` path, ~lines 1446–1475), and
it is **disabled** with the comment:

> "After extensive timing I have not been able to see lots of situations where a
> maskout read is better than a getitem one. Disabling for now."

So a maskout *read* has already been measured against `getitem` and found not
consistently better. Combined with our profiling (the per-block decompression of
scattered candidate blocks is the floor; we already skip non-candidate blocks via
per-block getitem + prefilter early-return), this is strong evidence that a new
`b2nd_get_maskout_cbuffer()` would **not** beat the existing getitem-based reads.

**Therefore the recommended direction is the coords + existing-sparse-gather path
(below); the maskout API is kept only as a documented fallback to revisit if the
sparse-gather prototype shows the per-coord/coords-materialization overhead is the
bottleneck.**

## Motivation

CTable SUMMARY-index `where()` evaluates the predicate through the miniexpr
prefilter with a candidate-block bitmap, skipping non-candidate blocks. Profiling
`compute 1` (`payment.tips > 100`, 24.3M rows) shows:

- The matching rows are **6.5% of blocks** (97 / 1485) but **scatter across 22 of
  23 chunks** (block selectivity ≠ chunk clustering).
- The eval (~25 ms) is **dominated by operand ZSTD decompression**, not result
  compression (the mask is mostly-False → compresses to ~nothing). Native sample:
  `ZSTD_decompressBlock_internal` ≫ `me_eval*` ≫ `ZSTD_compress*`.
- We **already** skip non-candidate blocks (per-block `blosc2_getitem_ctx` +
  early-return in `aux_miniexpr`). So `blosc2_set_maskout`'s "skip blocks" is
  *redundant* with what we do.
- The remaining inefficiency: `aux_miniexpr` creates a fresh decode context
  **per block** (`blosc2_create_dctx` inside the per-block loop, `blosc2_ext.pyx`
  ~line 2581) and the prefilter is invoked **per output block** (with input-cache
  locking + per-block buffers + a full result-mask write).

So the candidate-block *data* we read is already minimal; what's expensive is
(a) per-block decode-context setup and (b) the prefilter/result-mask machinery
wrapped around it.

## Idea

A block-level companion to `b2nd_get_sparse_cbuffer()` (which gathers by element
coords). Given a **block maskout**, decompress the *kept* blocks of an array into
a packed buffer in one call — reusing a single decode context and skipping the
discarded blocks — enabling a clean "gather candidate blocks → evaluate compactly
→ collect positions" path that bypasses the prefilter and the result-mask entirely.

```c
// schunk level
int blosc2_schunk_get_maskout_buffer(blosc2_schunk *schunk,
                                     const bool *block_maskout, int64_t nblocks,
                                     void *buffer, int64_t buffersize);
// b2nd level (1-D first; N-D later)
int b2nd_get_maskout_cbuffer(const b2nd_array_t *array,
                             const bool *block_maskout, int64_t nblocks,
                             void *buffer, int64_t buffersize);
```

Semantics (proposed): `block_maskout[i] == true` ⇒ **skip** block `i` (mirrors
`blosc2_set_maskout`). Kept blocks are written packed, in ascending global block
order, `blocknitems` items each (full blocks, including chunk-tail padding items
for the last block of each chunk — the caller already knows valid_nitems per
block). The caller maps a packed item index `p` back to a global element index
via a prefix-sum over `~block_maskout`:
`global = kept_block_global[p // blocknitems] * blocknitems + (p % blocknitems)`.

## Where it would plug in (python-blosc2)

A new evaluation path for SUMMARY (and later BUCKET) in `_try_index_where`,
*replacing* the prefilter-bitmap compute for sufficiently dense maskouts:

```python
maskout = ~candidate_block_bitmap  # skip non-candidates
packed = {op: ndarray.get_maskout_buffer(maskout) for op in operands}  # 1 call/op
m = numexpr.evaluate(predicate, packed)  # compact eval, no result-mask
# map compact True positions -> global via kept-block prefix sum
positions = _map_packed_to_global(np.flatnonzero(m), maskout, blocknitems, nrows)
```

This avoids: the full result-mask materialization, the per-output-block prefilter
invocation + input-cache locking, and the per-block `create_dctx`.

## Honest expected win (and the floor)

From the native sample, the recoverable portion is roughly:

- **Amortizable**: per-block `create/freeDCtx` (~10% of decompression samples) and
  the prefilter orchestration (locks, per-block buffers, result write).
- **Floor (not recoverable here)**: `ZSTD_decompressBlock_internal` + per-block
  FSE/HUF table decode. Each blosc2 block is an independent ZSTD frame, so the
  97 scattered candidate blocks must each be decoded — this primitive does not
  merge them.

Rough estimate: ~10–30% off the `compute 1` eval, **not** a step change. The big
lever for this workload remains **data clustering** (sorting by the filtered
column) so candidate blocks are few/contiguous — orthogonal to this API.

The primitive is most valuable when the maskout is **dense** (most blocks
discarded) *and* the surrounding machinery (prefilter/result-mask) is a
meaningful fraction of the total — e.g. count/compose queries, or wide multi-op
predicates where avoiding the result-mask helps.

## Preferred direction: coords + existing `b2nd_get_sparse_cbuffer` (no new API)

For **high-selectivity** queries (few candidate blocks survive the summary prune),
skip the miniexpr prefilter + result-mask entirely:

1. From the candidate-block bitmap, build flat element coords for the candidate
   blocks (or directly the candidate rows if a coarser filter is available).
2. `b2nd_get_sparse_cbuffer(operand, coords)` for each operand — this already
   groups coords by chunk→block, decompresses each needed block **once**, and is
   multithreaded (`blosc/schunk.c::blosc2_schunk_get_sparse_buffer`).
3. Evaluate the predicate on the packed buffers (one numexpr call) → positions.
4. Map packed positions back to global element indices.

Gate it with a **threshold on the candidate-block count** (the block mask
popcount): use the coords path only when few enough blocks survive that the
coords list and gather are cheap; otherwise stay on the current prefilter/scan.
Note the threshold is on *candidate blocks*, not match count — coords scale with
candidate blocks (all elements of each candidate block), not with the eventual
number of matches.

Caveat: this still decompresses the same candidate blocks (the floor); the win is
**avoiding the result-mask materialization and the per-output-block prefilter
machinery (locks, buffers, 23-chunk iteration)** — most visible at extreme
selectivity (e.g. 1–3 candidate blocks), neutral-to-worse at moderate selectivity
where the coords list grows large.

## Phasing (validation-first)

- **Phase 0a — coords path, no C-blosc2 change (do this first).** Prototype the
  coords + `b2nd_get_sparse_cbuffer` path above and measure vs the current
  prefilter on `tips>100` / `tips>500` / `tips>800`, sweeping the candidate-block
  threshold. This needs **zero** c-blosc2 changes and reuses an optimized gather.
- **Phase 0b — dctx-reuse probe (optional).** Hoist `create_dctx` out of the
  per-block loop in `aux_miniexpr` to quantify how much of the eval is pure
  context-alloc. Informs whether *any* read-side restructure is worth it.
- Decide from 0a/0b whether the maskout C API (Phases 1–3 below) is worth it.
  Given the disabled-maskout prior art, the bar is high.
- **Phase 1 — C-blosc2.** Add `blosc2_schunk_get_maskout_buffer` (one reused
  `dctx`, `blosc2_set_maskout` + `blosc2_decompress_ctx` per chunk, pack kept
  blocks) and the 1-D `b2nd_get_maskout_cbuffer` wrapper. Tests + bench in
  c-blosc2.
- **Phase 2 — python-blosc2 binding.** Expose as `NDArray.get_maskout_buffer`
  (or similar) in `blosc2_ext.pyx`.
- **Phase 3 — integrate** into the SUMMARY `where()` path behind the existing
  cost gate; extend to BUCKET (needs a block bitmap from its chunk-local bucket
  pruning).

## Design considerations / open questions

- **Padding**: last block of a chunk is partial (chunk padded to block multiple).
  Pack full `blocknitems` and let the caller drop padding (it knows per-chunk
  valid counts), or pack valid items only (variable, harder to index). Lean to
  full-blocks for simple indexing.
- **Multi-dim**: scope 1-D first (CTable columns are 1-D). N-D needs block-coord
  enumeration; defer.
- **dtype / typesize**: buffer sized `kept_blocks * blocknitems * typesize`.
- **Maskout polarity**: match `blosc2_set_maskout` (`true` = skip) for
  consistency; document clearly (it's the opposite of a "keep" mask).
- **Return value**: number of items (or bytes) written; the caller derives the
  kept-block mapping from the maskout (prefix sum) — or the API optionally fills
  an `int64_t *kept_block_index` out-array to save the caller a cumsum.
- **Relationship to `get_sparse_cbuffer`**: coords (element) vs maskout (block).
  Maskout is cheaper when selection is block-aligned and dense; coords when truly
  scattered at element granularity. They are complementary, not redundant.

## Verdict

Worth doing as a **general, reusable primitive** (block-level peer of the
element-level sparse gather), but **gate it on Phase 0** — the motivating
workload's win is modest because the per-block decode of scattered candidate
blocks is the irreducible floor. If Phase 0 shows the prefilter/context overhead
is a big enough slice, proceed; otherwise the effort is better spent on
clustering (sort-on-filter) and the cheap `dctx`-reuse tweak.
