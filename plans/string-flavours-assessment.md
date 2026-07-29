# String column flavours: what works today, and what to do about utf8

Measured on branch `dsl-string-support`, blosc2 `4.9.2.dev0`, NumPy 2.4.6. Every cell below was
probed on a live `CTable`, not read off the docs — the doc table in `doc/reference/ctable.rst`
(§ChoosingStringType) is accurate but predates the compute surface and omits it.

**All timings are post-`92c39aa6`**, i.e. they include the two performance bugs this assessment
turned up and which have since been fixed: the dictionary per-row decode (`a9446841`) and the bucket
index pessimization (`92c39aa6`). Where a number changed materially the old value is kept alongside,
because two of the conclusions below originally rested on it.

## Capability matrix

| | `string()` `<Un` | `bytes()` `<Sn` | `utf8()` | `dictionary()` | `vlstring()` / `vlbytes()` |
|---|---|---|---|---|---|
| **Storage** | fixed-width UTF-32 NDArray | fixed-width bytes NDArray | int64 offsets + UTF-8 blob | int32 codes + uniques | msgpack batches |
| Per-row cost (raw) | 4 × `max_length` B | `max_length` B | exact bytes + 8 B offset | 4 B | value + framing |
| Length limit | `max_length` | `max_length` | none | none | none |
| `col[:]` returns | `<Un` ndarray | `|Sn` ndarray | `StringDType` ndarray | Python `list` | Python `list` |
| Nulls | sentinel | sentinel | sentinel | native | native `None` |
| **Query** | | | | | |
| `t[t.c == v]` (operator) | ✓ | ✓ | ✓ | ✓ | ✗ NotImpl |
| `where("c == 'v'")` | ✓ | ✓ | ✓ | ✓ | ✗ NotImpl |
| `where("startswith(c,…)")` | ✓ | ✓ | ✓ | ✗ *Unknown symbol* | ✗ NotImpl |
| `where("upper(c) == …")` | ✓ | ✓ | ✓ | ✗ *Unknown symbol* | ✗ NotImpl |
| `sum(where=…)` | ✓ | ✓ | ✓ | ✗ | ✗ |
| `sort_by` | ✓ | ✓ | ✓ | ✓ | ✗ TypeError |
| `group_by` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `create_index` | ✓ all 5 kinds | ✓ all 5 kinds | ✓ rank, `FULL` only ⁷ ⁹ | ✓ rank, `FULL` only ⁷ ⁹ | ✗ |
| **Compute (string-returning)** | | | | | |
| `add_computed_column("'x='+c")` | ✓ | ✓ | ✗ NotImpl, routes ¹⁰ | ✗ | ✗ |
| `assign(new=…)` | ✓ | ✓ | ✗ NotImpl, routes ¹⁰ | ✗ | ✗ |
| `t.apply(dsl_kernel)` / `lazyudf` | ✓ | ✓ | ✗ NotImpl, routes ¹ ¹⁰ | ✗ | ✗ RuntimeError |
| nested (dotted) leaf in expr | ✓ | ✓ | ✗ NotImpl | ✗ | ✗ |
| **Bare container (no CTable)** | | | | | |
| `lazyexpr(expr, {a: col})` | ✓ NDArray | ✓ | ✓ span driver, returns `UTF8Array` ² | ⚠ padded ⁶ | ⚠ numpy |
| `col == "scalar"` | ✓ LazyExpr | ✓ LazyExpr | ✓ bool mask ³ | ✓ bool mask ³ | ✓ bool mask ³ |
| **Interop** | | | | | |
| `to_arrow` | `string` | `large_binary` | `large_string` | `dictionary<…>` | `string` / `large_binary` |
| save + reopen | ✓ | ✓ | ✓ | ✓ | ✓ |
| NumPy requirement | any | any | ≥ 2.0 | any | any |

¹ Was `ValueError: malformed node or string … StringDType()` — `lazyudf` allocates an NDArray
output with `dtype=StringDType()`, which `NDArray.dtype`'s `ast.literal_eval` round-trip cannot
parse (`blosc2_ext.pyx:3818`). The underlying limit stands (that is G2, dropped); since `8e3868ba`
the operand is refused up front instead, with a message that names the conversion. See ¹⁰.
² Fixed in `0b486b07` — was correct values down the wrong path: a `SimpleProxy` widened the column
to a fixed `<Un` and evaluation fell into `slices_eval`, never reaching miniexpr, ignoring
`_UTF8_EXPR_BUDGET` and losing the utf8 container. The span driver now lives at module level and
`lazyexpr()` routes utf8 operands to it. Whole-array evaluation only.
³ Fixed in `3692673f` (utf8) and `486be882` (dictionary, vlstring) — each was a plain `False`,
object identity, silently wrong, because none of the three defined comparison operators. `UTF8Array`
answers a scalar `str` with its existing raw-byte scanners (all six operators, no row decoded);
`DictionaryColumn` compares codes, so also no decode; `_ScalarVarLenArray` decodes and delegates to
NumPy. The bug survived because `CTable` answers `==` through its own predicate path and never asks
the container.

### Measured cost — 200 k rows of free text, max 37 chars

| | raw | compressed | cratio | ingest | full read | `where(startswith)` |
|---|---|---|---|---|---|---|
| `string(max_length=37)` | 148.0 MB | 1.38 MB | 107× | 47 ms | 8.6 ms | **73 ms** |
| `utf8()` | **8.0 MB** | **1.09 MB** | 7.4× | 64 ms | 9.2 ms | 120 ms |
| `dictionary()` | 9.5 MB | 1.14 MB | 8.4× | 263 ms | 30.3 ms ⁴ | n/a |
| `vlstring()` | 5.5 MB | 0.92 MB | 6.0× | 95 ms | 47.9 ms | n/a |

⁴ Was 44 s before `a9446841`. `DictionaryColumn.decode()` indexed the dict store once per row and
each such index decompresses a whole msgpack batch, so a read cost O(N) decompressions. Fixed by
caching the code→value map that `_ensure_cache()` was already building the forward half of.

## Indexing, in detail

`create_index` is the one place where the flavours are not merely *slower* or *less convenient* than
each other — they are in different performance classes. All five index kinds
(`SUMMARY`/`BUCKET`/`PARTIAL`/`FULL`/`OPSI`) work on `string()` and `bytes()`. `utf8()` gained an
index in `b1bbc54e`, and `dictionary()` has always had one, but both are **`FULL`-only** — see §⁹.
What follows was written when utf8 had no index at all; the ⁷ note records what changed.

### What a FULL index buys on `<U` — 2 M rows, cardinality 20 k, `<U15`

| query | no index | `kind=BUCKET` | `kind=FULL` |
|---|---|---|---|
| `c == 'taxi-10000'` | 15 ms | 23 ms | **2 ms** |
| range, 0.1 % selectivity | 27 ms | 34 ms ⁵ | **4 ms** |
| `startswith(c, 'taxi-000')` | 18 ms | 19 ms | 19 ms |
| build cost | — | 2531 ms | 2273 ms |

⁵ Was 289 ms before `92c39aa6` — see "Two warts" below. BUCKET now declines the plan on this
workload and falls back to the scan, so its column is the scan cost plus ~7 ms of planning.

And on the ordering side — 1 M rows, **persistent** table (the `sorted_slice` window path requires a
persistent FULL index; on an in-memory table it silently falls back, `ctable.py:11468`):

| | no index | `kind=FULL` (build 1148 ms) |
|---|---|---|
| `sorted_slice("c", :100)` (top-*k*) | 371 ms | **8.2 ms** |
| `sort_by("c", view=True)[:100]` | 374 ms | **7.4 ms** |
| `group_by("c").sum("v")` | 217 ms | 208 ms |

Three things fall out of this, and they are not the same thing:

1. **Point and range lookups: 7×.** Exact filtering is what FULL is for.
2. **Top-*k* windows: 45×.** This is the biggest single number in this whole assessment. It is also
   the one that does not degrade gracefully — without the index you pay a full sort of the column to
   read 100 rows.
3. **`startswith` and `group_by` get nothing.** Prefix scans and grouping don't consult the index at
   all, on any flavour. So "utf8 has no index" costs *nothing* for those two workloads.

### The same table for utf8

| query | utf8, no index (only option) | `<U` + FULL | ratio |
|---|---|---|---|
| `c == …` | 31 ms | 2 ms | **15×** |
| range 0.1 % | 160 ms | 4 ms | **40×** |
| `startswith` | 513 ms | 19 ms | **27×** |
| `sorted_slice` top-100 | 427 ms | 8.2 ms | **52×** |
| `group_by.sum` | 217 ms | 208 ms | 1.0× |

The `startswith` row deserves flagging separately: it is 27× slower on utf8 **and the index is not
the reason** — `<U` gets 19 ms without any index too. That gap is the span driver's decode +
`astype` per span, i.e. the cost the parity plan explicitly accepts. So on this workload utf8 is
losing on two independent axes at once, and only one of them is about indexing.

(Absolute numbers are workload-dependent — on a wider, higher-cardinality column measured earlier,
`<U` `startswith` was 77 ms against utf8's 120 ms. The direction is consistent; the magnitude is not.)

### Two warts found along the way — both now fixed

- **`kind=BUCKET` made queries slower than no index at all**, on *every* indexable dtype, not just
  strings. Scattered matches were read one bucket run at a time, re-decompressing each block many
  times (1379 spans of ~320 elements over a column with 128 blocks), and the planner gated on
  selectivity in *buckets* while the cost is paid in *blocks* — a mask selecting 21 % of buckets
  touched 96 % of blocks. Fixed in `92c39aa6` by coalescing spans within a block and declining plans
  above a block-fraction threshold. Scattered 0.1 % range, before → after:

  | | no index | bucket, before | bucket, now |
  |---|---|---|---|
  | `int32` | 4.9 ms | 45.6 ms | 5.5 ms |
  | `int64` | 5.8 ms | 56.9 ms | 6.5 ms |
  | `float64` | 6.3 ms | 77.9 ms | 6.6 ms |
  | `<U15` | 49.9 ms | 228.2 ms | 58.0 ms |
  | `<U32` | 96.6 ms | 274.6 ms | 97.6 ms |

  The *relative* penalty was worst on the numeric dtypes — their baseline scan is fast enough that
  redundant decompression dominates outright. Clustered data, where the index prunes real work,
  still uses it and still wins (`int64` 3.7→2.4 ms, `<U32` 77.4→2.7 ms).

- ~~**`dictionary`'s FULL index gives no equality speedup**~~ (25 ms indexed vs 25 ms unindexed at
  2 M rows) — **fixed**, `1c22fdf5`; the operator form `t[t.c == v]` is now 329.6 → 8.4 ms. The
  reading recorded here first, *"by design — the rank index is built for ordering, not filtering"*,
  was **wrong**: instrumenting the planner showed the index was simply never consulted, because a
  dictionary `==` is rewritten into a code comparison and evaluated directly. A wiring gap, not a
  property of rank indexes — ranks are order-preserving, so a literal maps to a rank by binary
  search and equality is a contiguous run. The correction is worked through under "Could utf8 have
  an index?" below, which is where the fix came from. The docs no longer oversell it either
  (`e707c91c` added the `[#rankindex]` footnote).

A methodology note worth keeping: the first pass at the dtype sweep concluded "strings only",
because the numeric query literals were built with `repr()` and NumPy 2 renders `repr(np.int64(5))`
as `np.int64(5)`, which the planner cannot parse — so every numeric plan was declined for an
unrelated reason and the pessimization was invisible. Any future sweep should assert that the plan
was actually *taken* before reading anything into its timings.

### The dictionary rank index, measured — 1 M rows, persistent

| | no index | `kind=FULL` (rank, build **365 ms**) |
|---|---|---|
| `sorted_slice` top-100 | 759 ms | 51.3 ms |
| `sort_by(view)[:100]` | 764 ms | 49.8 ms |
| `group_by.sum` | 81 ms | 75 ms |

**These numbers were 235 *seconds* in the no-index column before `a9446841`** — the same
per-row decode described in footnote 4, reached this time through `_build_lex_keys`. The remaining
15× is an ordinary index payoff, not a pathology.

**The rank index is the cheapest index of the three and it works.** 365 ms to build against 1148 ms
for `<U`'s FULL index, because it sorts int32 ranks rather than `<U15` payloads. It is ~6× slower
than `<U`+FULL at query time (51.3 vs 8.2 ms, the difference being decoding the resulting rows).

Also note `group_by` is *fastest* on dictionary (81 ms vs 217 ms for `<U`, 217 ms for utf8) — the
codes are already integers, so there is nothing to factorize. This was true even before the decode
fix, because `group_by` was the only caller that used the batched `decode_batch()` path.

One optimization deliberately left on the table: `_build_lex_keys` (`ctable.py:11292`) still decodes
a dictionary column to an object array and lexsorts that. Sorting the int32 *ranks* instead — the
trick `create_index` already uses — measured ~4× faster again (107 ms → 27 ms per 200 k rows). Not
done, because it is a tuning item rather than a bug once the decode is O(D) instead of O(N).

### Could utf8 have an index? Yes, and the pattern is already in-tree

The dictionary rank index (`ctable_indexing.py:766-783`, `_DictRankWrapper`) sidesteps string
comparison entirely: derive an **alphabetical rank per row as int32**, index *that*, and the whole
numeric FULL-index machinery — sort, window, null block, OOC merge — works unchanged. Sorting by
rank is sorting by string.

utf8 has the missing ingredient already: `UTF8Factorizer` (`_utf8_array.py:714`) produces global
codes plus a decoded vocabulary without decoding a single row, and `group_by` already runs it at
scale — 227 ms for 1 M rows above. So a utf8 rank index is roughly: factorize (~230 ms/M rows),
`argsort` the vocabulary, map codes→ranks, hand an int32 array to the existing builder. Build cost
would land near `<U`'s 1148 ms.

What it would and would not deliver, now that the dictionary version has been measured rather than
assumed:

- ✓ `sorted_slice` / `sort_by` top-*k*. Expect utf8 to land near dictionary's 51.3 ms — call it
  **~8× faster than utf8's current 427 ms**, still ~6× behind `<U`+FULL's 8.2 ms.
- ✓ **equality and range filtering too** — this corrects an earlier reading of this document.
  Dictionary's rank index shows no equality speedup (27.2 ms indexed vs 24.9 ms unindexed at 2 M
  rows), but instrumenting the planner shows **the index is never consulted**: a dictionary `==` is
  rewritten into a code comparison and evaluated directly, bypassing `plan_query` entirely. That is
  a wiring gap, not a limit of rank indexes. Ranks are order-preserving, so `col == 'lit'` becomes
  `rank == r` with `r` a binary search of the literal in the sorted vocabulary — **0.59 ms** once
  per query for a 20 k vocabulary, against the **34.8 ms** raw-byte scan (`equal_mask_span`) that
  is where utf8's 43 ms equality actually goes. The same translation serves `<`, `>` and ranges,
  because rank order *is* lexicographic order.
- ✗ `startswith` — nothing indexes prefixes on any flavour.
- ⚠ staleness: ranks shift when new values arrive. Dictionary handles this with a stored
  `dict_hash` (`_dict_rank_hash`) and falls back to lexsort when stale; utf8 would need the same,
  and a utf8 column is much more likely to gain new values than a category column is — which means
  the fallback would fire often, and the fallback is a full lexsort.

**Ordering shipped in `b1bbc54e`** (see ⁷); equality and ranges did not.

Realistic remaining target: `<U`+FULL territory, **~2-4 ms against today's 43 ms** for equality and
ranges — not the "ordering only" partial fix an earlier draft of this section claimed.

Rough cost: more than the 2-3 days first estimated, because the literal→rank translation in the
planner **does not exist for any flavour** — dictionary has a working rank index and still cannot
use it for `==`. That is new work rather than reuse, though dictionary inherits the same speedup
once it exists. Index build is dominated by factorizing: 279 ms per 1 M rows, against 1148 ms for
`<U`'s FULL build, so the build cost is not the problem. Staleness is: any new value shifts ranks,
and a utf8 column gains new values far more readily than a category column.

## Pros and cons

**`string()` / `bytes()` (fixed-width)** — everything works, always. Fastest filter of the lot, and
after compression only 27 % larger than utf8 on this data. Cost is the raw footprint: 4 bytes per
character per row, materialized in full on every read — 148 MB here, and the bench script's own
comment notes `<U130` at 10⁷ rows needs a ~5 GB ingest buffer. Needs `max_length` up front.

**`utf8()`** — 18× smaller raw footprint, no length ceiling, Arrow-native (`large_string`, the same
layout pandas 3 string columns use). The whole *query* surface is already at parity. Missing:
string-returning expressions, DSL kernels, `create_index`, dotted leaves. Reads need NumPy ≥ 2.0.

**`dictionary()`** — unbeatable for low-cardinality categories, the fastest `group_by` of any
flavour, and the first variable-length flavour to get a working index (utf8 has one since
`b1bbc54e`). Its rank index accelerates ordering *and*, since `1c22fdf5`, equality — `FULL` is the
only kind that reaches it (§⁹). But it is not a general string type: string *functions* don't reach
it at all (`Unknown symbol`).

**`vlstring()` / `vlbytes()`** — storage only. Native `None` nulls with no sentinel, works on
NumPy 1.x. Nothing else works; every query and compute surface raises.

## On the proposal (`<U` + bytes only, plus `.to_utf8()` / `blosc2.from_utf8()`)

The diagnosis is right — the asymmetry is real and it is exactly where a user would trip. But the
proposal is stated one notch too strong, in two directions:

**1. utf8 is functionally at parity on the query surface, but not at parity on speed.** Filters,
`where()` including string functions, `sum(where=)`, `sort_by`, `group_by` and Arrow all work today.
Demoting utf8 to a storage-only type that you must convert before touching would be a *regression*
against what already ships, and would hand back the 18× raw-footprint win for column scans. But the
indexing section above shows the gap is not cosmetic: against an indexed `<U` column, utf8 is 15×
slower on equality, 40× on ranges and 52× on top-*k* windows. `create_index` is the only *functional*
query-side hole, and it is the one that matters most.

**2. The conversion pair the proposal wants mostly exists already.** Probed end to end:

```python
arr = t["c"][:].astype("<U37")  # from_utf8, works today
res = blosc2.lazyexpr("'x=' + a", {"a": blosc2.asarray(arr)}).compute()[:]
out = blosc2.utf8_array(list(res))  # to_utf8, works today
```

The one genuinely missing piece is writing that result back as a table column: `add_column()` has no
`values=` parameter, so today it takes `add_column("out", field(utf8(), default=""))` followed by the
private `_cols["out"].set_all(...)`. **Fixed in `f8af0714`** (see ⁸); the `blosc2.asarray()` in the
snippet above also turns out to be unnecessary — `lazyexpr` takes a plain numpy operand, and wrapping
one that is already in memory only adds a compression round trip (36.2 → 30.1 ms per 1 M rows).

So the choice is not "parity vs. conversion" — it is **which rule do we publish**:

| | full compute parity (`utf8-string-support.md`) | storage+query utf8, compute on `<U` |
|---|---|---|
| effort | ~1 week (G2+G3+G5); G4 is now done | ~1–2 days |
| rule the user learns | "everything works everywhere" | "utf8 stores and filters; convert to compute" |
| perf honesty | hides a 3–5× penalty behind an identical API | the `.astype()` is visible, so the cost is |
| persistence risk | G2 must serialize `StringDType` — get it wrong and the **table won't reopen** | none; no new dtype is persisted |
| `create_index` on utf8 | still missing ⁺ | still missing ⁺ |
| top-*k* windows on utf8 | still 52× slower ⁺ | still 52× slower ⁺ |

The second column is the better trade for the *compute* surface, and it is the honest one: the span
driver's cost *is* a decode + `astype` per span, so making the user write `.astype()` describes what
actually happens.

But note the last two rows: **neither option touches the biggest measured gap.** Both plans leave
utf8 without an index, and the parity plan spends a week doing so. Given that the index rank trick
is already proven in-tree for dictionary and costs about as much as G2 alone, the priority order
inverts:

⁺ Both rows were true when this table was written and are **no longer**: the utf8 rank index shipped
in `b1bbc54e` (§⁷), taking top-*k* to 43.2 ms and `sort_by` to parity with `<U`+FULL. The comparison
is kept as written because it is what the decision was made on — the argument for the second column
never rested on these two rows, which is exactly the point the next paragraph makes.

1. ~~**Fix what is wrong, not merely absent**~~ — **done**. `UTF8Array` comparisons (`3692673f`)
   and the bare-array `lazyexpr` fallback (G4, `0b486b07`). Both were silent-wrong results, and
   neither depended on which rule is chosen below. **No known silently-wrong utf8 path remains.**
2. ~~**Publish the conversion pair.**~~ — **done**, `f8af0714` + `5b31abe4`. See ⁸.
3. ~~**Then the utf8 rank index**~~ — **done**, `b1bbc54e` (ordering) + `f132d6df` (scalar
   predicates). See ⁷ and ⁹. The caveat written here — *"go in knowing it does not close the
   15×/40× equality and range gap"* — proved **too pessimistic**: persisting the sorted vocabulary
   made a literal→rank lookup one `searchsorted`, so `==` went 29.00 → 5.49 ms and `<` 34.57 →
   5.45 ms. It was inherited from the "ordering only" misreading corrected two sections above.
4. ~~**Make the error messages route.**~~ — **done**, `8e3868ba`. See ¹⁰.
5. **Drop G2/G3/G5**, or park them behind a concrete user request. G2 in particular buys a
   `StringDType`-in-schema serialization hazard for a surface the conversion pair already covers.

Found while measuring, unrelated to the utf8 decision:

- ~~`sort_by` on an unindexed `dictionary` column takes 235 s at 1 M rows~~ — **fixed**, `a9446841`.
  Per-row decode; now 759 ms.
- ~~`kind=BUCKET` is a pessimization~~ — **fixed**, `92c39aa6`. Affected every indexable dtype.
- ~~The doc table's `create_index` entries are stale~~ — **done**, `e707c91c`. The reference table
  and the `utf8()` docstring both claimed utf8 could not be indexed, which my own change had made
  false; 4.9.2 release notes added for this whole line of work.
- ~~`min()`/`max()` from the block summaries are silently wrong~~ — **fixed**, see §⁹. Two separate
  bugs, both affecting *every* indexable dtype, neither utf8-specific.
- ~~`create_index` accepts any of the five kinds on a rank-indexed column and silently builds one
  that is never consulted~~ — **fixed**, see §⁹.
- Still open: `_build_lex_keys` could sort dictionary ranks instead of decoded strings (~4×), and
  the `where("c == 'x'")` string form still bypasses the index for both flavours.

⁶ `blosc2.lazyexpr` over a bare `DictionaryColumn` returns the **capacity-padded** slot array —
1 048 576 rows for a 3-row table. Not a container bug: `DictionaryColumn.__len__` is documented as
the physical slot capacity and `CTable` indexes it with live positions. It only misleads because
these are internal classes reachable through `t._cols`, never a public route. Left alone
deliberately: changing the length semantics would touch every table-layer caller.

---

## ⁷ utf8 `create_index` — what shipped

`b1bbc54e`. `_utf8_rank_arrays()` factorizes the column and materializes an `int32` alphabetical
rank per row, which drives the existing numeric index machinery unchanged. Measured at 1 M rows,
cardinality 20 k, persistent:

| | no index | utf8 + FULL | `<U` + FULL |
|---|---|---|---|
| `sorted_slice` top-100 | 458.5 ms | **43.2 ms** | 9.9 ms |
| `sort_by(view)[:100]` | 424.2 ms | **7.2 ms** | 7.9 ms |
| index build | — | **277 ms** | 867 ms |

`sort_by` reaches parity with an indexed `<U` column and the index is the cheapest of the three to
build, because it sorts `int32` ranks rather than `<U` payloads. `sorted_slice` keeps a ~4× gap:
that is the cost of gathering the result rows out of the offsets/blob, not of the index.

**Scalar predicates followed in `f132d6df`.** The sorted vocabulary is now persisted beside the
index's other sidecars, so a literal maps to a rank by one `searchsorted` and the matching rows are
a contiguous run of the sorted-positions sidecar. Mask construction, same workload:

| | scan | rank index |
|---|---|---|
| `==` | 29.00 ms | **5.49 ms** |
| `!=` | 28.55 ms | 10.24 ms |
| `<` | 34.57 ms | **5.45 ms** |
| `>=` | 34.16 ms | 8.07 ms |

It hangs off `Column._utf8_scalar_mask`, which every scalar predicate already funnels through, and
returns `None` to fall back to the scan whenever the index cannot answer. Note the end-to-end
`where()` gain is smaller than these numbers — materializing the result rows out of the
offsets/blob dominates once the mask is cheap.

**Dictionary followed in `1c22fdf5`**, minus the persistence — a dictionary already holds its
vocabulary in memory. The operator form `t[t.c == v]` goes **329.6 ms → 8.4 ms**.

Two things surfaced while wiring it, both worth more than the feature:

- **The staleness check cost more than the scan it saved.** `_dict_rank_index_stale` SHA1s the whole
  dictionary — 24 ms for 20 k entries — on *every* query, including the ordering path that already
  used the index. It now settles from the value epoch first, which also drops dictionary
  `sorted_slice` from 51.3 ms to 37.7 ms. Wiring the index made queries *slower* until this was found.
- **`col != value` raised `IndexError`** on any table with capacity padding: `__ne__` negated the
  result of `_dictionary_eq`, which had already been intersected with the live-row mask, so
  `~(pred & valid)` turned every dead slot True. Pre-existing and unrelated to indexing; found by
  fuzzing indexed against unindexed results.

**The `where("c == 'x'")` string form is deliberately left alone** for both flavours. Rewriting to a
code comparison keeps it a single fused numeric expression; substituting a precomputed mask measured
*slower* (22.9 ms → 28.7 ms) even though the mask costs 4.8 ms. `plan_query` is still never consulted
for a utf8 or dictionary predicate — both routes bypass it rather than fix it, and accelerating the
string form needs the planner to consume index *positions* rather than a mask.

**Also found here:** NumPy 2.4 does not match a lone `"\x00"` against a `StringDType` array
(`np.array(["\x00"], dtype=StringDType()) == "\x00"` is `False`), while `"\x00x"` and `"a\x00b"`
compare correctly. Every null mask in the utf8 paths is such a comparison, so that sentinel would
silently stop marking anything as null. `blosc2.utf8(null_value="\x00")` now rejects it. The
default sentinel is `'__BLOSC2_NULL__'`, so no shipped configuration was affected.

---

## ⁸ The conversion pair — what shipped

`f8af0714` and `5b31abe4`. The rule is now published rather than implied: **utf8 stores and filters;
fixed-width computes.**

```python
fixed = blosc2.from_utf8(t["name"])  # -> <Un ndarray, exact width
res = blosc2.lazyexpr("'x=' + a", {"a": fixed}).compute()[:]
t.add_column("prefixed", blosc2.utf8(), values=blosc2.to_utf8(res))
t["name"].assign(res)  # or overwrite in place
```

Four pieces, all of which were missing:

- **`add_column(..., values=)`** — one entry per *live* row, coerced to the column's dtype. A
  declared default still governs rows appended later, so the two combine.
- **`blosc2.from_utf8()` / `blosc2.to_utf8()`**, plus `UTF8Array.astype()` as the method form.
- **`Column.assign()` on utf8 and the other varlen scalar columns**, which raised
  `TypeError: UTF8Array assignment index must be int` — there was no public way to overwrite a
  variable-length column at all.
- **The rule itself**, in `doc/reference/ctable.rst` (§Utf8Compute), with a compute row added to the
  flavour comparison table.

Two design points worth keeping:

- **Width inference is by codepoint, not byte.** A UTF-8 codepoint starts at every non-continuation
  byte, so the exact `<U` width is a masked count over the raw blob — no row decoded. Byte lengths
  would only *bound* it, over-allocating 3–4× on CJK. Spans whose byte lengths cannot beat the
  running best are skipped without touching data, and an all-ASCII span settles from the offsets
  alone, so the inference costs ~2 ms per 500 k rows on top of the copy (16.8 ms against 14.8 ms for
  a hand-sized `astype("<U13")` — which was also 1.3× wider than needed).
- **The varlen columns are rewritten whole, not row by row.** `_ScalarVarLenArray.__setitem__`
  rewrites an entire msgpack batch per row, the same O(N × batch) shape as the dictionary decode bug
  in ⁴. `set_all()` (which `UTF8Array` already had, and `_ScalarVarLenArray` now grows) writes each
  batch once. utf8 `assign` measures 122 ms per 200 k rows.

Found while doing it, both pre-existing:

- **`add_column()` on a varlen column was broken on any table with deleted rows.** Those columns are
  indexed by *physical* position but were filled with only `n_live` entries, so the first read after
  a `delete()` raised `IndexError`. Hit the plain `default=` path too, not just the new one.
- **`add_column()` on a `dictionary()` column** raised `AttributeError` from inside the fixed-width
  path (`DictionarySpec` has no dtype). Now a `TypeError` naming the limitation; actually supporting
  it would mean wiring `create_dictionary_column`, which nothing has asked for.

Still open after this, and now the top of the list: **item 4, the error messages.** A DSL kernel
returning strings over a utf8 column fails with NumPy's `DTypePromotionError`, which names neither
the column nor the workaround. `add_computed_column` and `assign` at least raise
`NotImplementedError` naming the column — but still not the two-line conversion that fixes it.

---

## ⁹ Index kinds and summary reductions — two silent-wrong bugs and a refused kind

Started as a narrow question — *does `create_index` support all five kinds on utf8 the way it does on
`<U`?* — and the answer turned out to be interesting only because of what verifying it turned up.

### The kinds question: all five build, only FULL is ever consulted

No kind restriction existed. `_utf8_rank_arrays` materializes the int32 ranks *before* the kind
dispatch, so any kind builds over them, returns correct results, and costs build time and disk.
Measured, 1 M rows, cardinality 20 k, persistent:

| kind | utf8 `==` | utf8 `<` | `<U11` `==` | `<U11` `<` |
|---|---|---|---|---|
| no index | 24.15 ms | 29.01 ms | 4.88 ms | 12.84 ms |
| `SUMMARY` | ≈ no index | ≈ no index | 3.25 ms | — |
| `BUCKET` | 24.73 ms | 29.79 ms | 15.99 ms | 27.97 ms |
| `PARTIAL` | ≈ no index | ≈ no index | 11.95 ms | — |
| `FULL` | **6.50 ms** | **6.98 ms** | 3.51 ms | 6.21 ms |
| `OPSI` | 25.22 ms | ≈ no index | **2.74 ms** | 5.78 ms |

Two gates explain it, and the second is the real one. `_utf8_index_mask` requires `kind == "full"`
plus persistent sidecars. And **`plan_query` is never invoked for a rank-indexed predicate at all** —
instrumented, it is called 4× per query on `<U11` whenever an index exists and **0×** on utf8 in
every configuration. BUCKET/PARTIAL/SUMMARY/OPSI are precisely the kinds that only pay off *through*
the planner, so on these flavours they have no consumer. Same for ordering: `_sorted_slice_positions`
and `_sorted_positions_from_full_index` both require FULL.

This is a property of the **rank-index design**, not of utf8 — `_dictionary_index_mask` gates
identically. And the default `kind=IndexKind.BUCKET` meant `create_index("c")` on a dictionary column
had *always* built an index nothing could use.

**Fixed.** `kind` now defaults to `None` and resolves to `FULL` for `UTF8Spec`/`DictionarySpec`
(`BUCKET` unchanged everywhere else); an *explicit* non-FULL kind on those flavours raises
`ValueError` naming the reason. Erroring rather than warning because it is not a trade-off — there is
no workload where those kinds help — and because relaxing an error later is non-breaking while
tightening a warning is not.

### Bug A — capacity padding leaks into the block summaries

`_index_summary_minmax` reduces the per-block `(min, max, flags)` sidecars, a ~780× shortcut on a
string column. But the summaries cover the column's **physical** extent, which is padded to slot
capacity with zeros/empty strings — values that beat any real datum on `min`:

| rows | capacity | `<U11` `.min()` | `int64` `.min()` |
|---|---|---|---|
| 1 000 000 | 1 048 576 | `''` — **wrong** (`'taxi-00000'`) | `0` — **wrong** (`5`) |
| 1 048 576 | 1 048 576 | ✓ | ✓ |
| 1 000 | 1 048 576 | `''` — **wrong** | `0` — **wrong** |

Correct only when `n_rows` lands exactly on capacity. `max` was unaffected (padding is below the
data). Both wrong rows are **non-nullable** columns, so the existing sentinel gate — which was
written for exactly this class of leak — never fired. The docstring asserted the opposite: *"capacity
padding never enters the summaries"*.

### Bug B — `delete()` never marked the index stale

Independent of padding, and it had to be fixed first because the boundary fix assumes live rows are
the prefix `[0, n_rows)`. On a **zero-padding** table, after deleting the rows holding the extremes:
`min()` still returned `5` (true `6`), `max()` still returned `1048580` (true `1048579`). Deleted
rows keep sitting in their block and keep contributing to its extrema.

The signal already existed and was unused: `delete()` bumps a `visibility_epoch` that nothing
recorded or compared. Index builds now store `built_visibility_epoch`, and the shortcut declines when
it has moved. Deliberately *not* by marking the index stale — that would forfeit query acceleration
after any delete.

### The fix, and what it costs

Blocks wholly below `n_rows` are read from the sidecar as before; the single block straddling the
boundary is rescanned (one block decompression); blocks past it are dropped. `_summary_minmax_source`
now returns `segment_len` so the caller can do that arithmetic.

| | indexed | scan | |
|---|---|---|---|
| `<U11`, 1 000 000 rows (576-row tail) | 1.44 ms | 116.52 ms | **81×** |
| `<U11`, 1 048 576 rows (no tail) | 0.18 ms | 119.78 ms | 670× |
| `int64`, 1 000 000 rows | 1.31 ms | 3.92 ms | 3× |

So the shortcut survives at 81× where it matters. The `int64` row is the honest caveat: its scan is
fast enough that a ~1.2 ms boundary read eats most of the win. Reading the tail from the raw NDArray
instead of through `Column.__getitem__` would recover it; not done, since the string case is where
the shortcut earns its keep.

**Note for any future sweep**, in the same spirit as the `repr(np.int64(5))` methodology note above:
an early run showed `OPSI` on utf8 at 38 ms against a 24 ms baseline and I nearly wrote it up as a
pessimization. Re-run paired A/B it is 25.2 vs 23.4 ms — neutral. Single-shot timings across
separately-built tables are not comparable; build the A and B tables in the same process and
interleave.

Also worth recording: `_summary_minmax_source` excludes utf8 via the `is_varlen_scalar` catch-all, so
a `SUMMARY` index on a utf8 column writes block extrema nothing will ever read. That is now moot —
the kind is refused outright — but if utf8 `min()`/`max()` is ever wanted, the extrema are
well-defined and the exclusion is the only thing in the way.

---

## ¹⁰ The compute-side error messages — what shipped

`8e3868ba`. Item 4 turned out to be two jobs, not one: most paths raised something clear but
unhelpful, two raised something *un*clear, and one did not raise at all.

Every refusal now names the column and prints the recipe, echoing the user's own expression:

```
Column 'name' is a variable-length utf8 column; string expressions that reference one
are not supported here.
utf8 stores and filters; fixed-width computes. Convert, compute, write back:
    fixed = blosc2.from_utf8(t['name'])
    res = blosc2.lazyexpr('upper(name)', {'name': fixed}).compute()[:]
    t.add_column('out', blosc2.utf8(), values=blosc2.to_utf8(res))   # or t['name'].assign(res)
See 'Computing strings on a utf8 column' in the CTable reference docs.
```

The full inventory, probed rather than assumed:

| path | before | now |
|---|---|---|
| `add_computed_column("upper(c)")` | NotImpl, no route | routes |
| `assign(new="upper(c)")` | NotImpl, no route | routes |
| `add_generated_column(values="upper(c)")` | NotImpl, no route | routes |
| `add_computed_column(kernel, inputs=["c"])` | **accepted, then broke the table** | refused at registration |
| `t.apply(kernel)` | NumPy `DTypePromotionError` | routes, names the column |
| `lazyudf(kernel, (t["c"],))` | `ValueError: malformed node … StringDType()` | routes, names the column |
| `lazyudf(kernel, (utf8_array,))` | same | routes (no `.assign` line — no table) |
| `lazyexpr(expr, {"a": utf8_array})` | works (span driver, ²) | unchanged |

Three things worth keeping:

- **The `inputs=` route was a table-breaker, not just a bad message.** `add_computed_column(name,
  kernel, inputs=["utf8_col"])` registered fine; afterwards every read of that column *and*
  `str(table)` raised `ValueError: malformed node or string`, so the table could not even be
  displayed. The guard is now on the kernel's dependencies and fires at registration, while the
  table is still untouched.
- **It is the utf8 *operand* that cannot work, not the string output.** A kernel returning a bool
  (`name > "b"`) fails identically — the operand is widened to a `SimpleProxy` and the output
  container is allocated from a `StringDType` the NDArray dtype round-trip cannot parse. So the
  guard is on inputs, and the docs say so; an earlier draft of this document implied the output
  type was the problem.
- **`lazyudf()` needed the guard twice.** The `DTypePromotionError` fires in the `lazyudf()`
  function's dtype inference, before `LazyUDF.__init__` runs, so guarding the constructor alone
  left `t.apply()` untouched. Both now check; `apply` also guards at the CTable level, because
  `lazyudf` only ever sees the container and cannot say *which* column.

Each printed recipe was run verbatim before the message shipped.

Also fixed here: the `.. _Utf8Compute:` anchor added in `5b31abe4` collided with the
`[#utf8compute]` footnote label — docutils normalizes both to the same target name, which cost the
footnote its reference. Renamed to `ComputingUtf8Strings`.

With this, items 1–4 of the priority list are done and only item 5 (drop G2/G3/G5, a decision
rather than work) remains.

---

## ¹¹ NumPy `StringDType` convention — what was adopted, and what could not be

`0ed38238`. The question was whether blosc2 should follow NumPy, which builds variable-length text
through a *dtype* (`np.array(v, dtype=StringDType())`) rather than through a separate constructor
(`blosc2.utf8_array(v)`). Answer: adopt the **dispatch**, not the dtype.

**Why the dtype itself cannot be adopted.** `StringDType` is not a storage format:

| | |
|---|---|
| `memoryview(arr)` | `ValueError: cannot include dtype 'StringDType' in a buffer` |
| `itemsize` | 16, whatever the content |
| `np.array(["x"*100], dtype=StringDType()).nbytes` | **16** — the payload is elsewhere |
| `.tobytes()` | a handle, not the text (≤15-byte strings are inlined; longer ones are pointers) |

blosc2's NDArray compresses *buffers*, so `NDArray(dtype=StringDType())` would persist pointers —
garbage on reopen, in another process, or on another machine. Arrow reached the same conclusion, and
`UTF8Array`'s layout **is** Arrow's `large_string`, which is what makes `to_arrow` zero-copy. (The
`ast.literal_eval` failure in `NDArray.dtype` is a symptom, ~5 lines to fix, and fixing it buys
nothing.)

**Why the schema layer was left alone.** `blosc2.field()` accepts a spec and never a raw dtype, for
*every* column type — `field(np.dtype("int32"))` is a `TypeError` too. Specs carry nullability, the
null sentinel, `ge`/`le`, storage config, `batch_rows`. Making utf8 the one dtype-addressable type
would have *broken* schema uniformity, not restored it. (Also: the runtime floor is `numpy>=1.26`,
where `StringDType` does not exist; `UTF8Spec.dtype = None` is deliberate.)

**What shipped.** Constructors dispatch on the target dtype, matching NumPy's fill values exactly:

```python
blosc2.asarray(np.array(["a", "bb"], dtype=StringDType()))  # -> UTF8Array
blosc2.zeros(3, dtype=StringDType())  # -> UTF8Array, ['', '', '']
blosc2.ones(3, dtype=StringDType())  # -> UTF8Array, ['1', '1', '1']
blosc2.asarray(utf8_source, dtype="<U8")  # -> NDArray, fixed width
```

Two container gaps closed along the way, both worth more than the dispatch:

- **`UTF8Array` failed the `blosc2.Array` protocol**, and `.shape` was the *only* member it lacked —
  for a container `CTable` uses throughout. It now has `.shape`/`.ndim`/`.size`.
- **`np.asarray(utf8_arr)` silently widened** to a fixed-width `<Un`, because there was no
  `__array__` and NumPy fell back to iterating rows: 1600 bytes for two 200-character values whose
  payload is 203, and a *different dtype* than `arr[:]` reported for the same object. Same family as
  the ² and ³ bugs.

**One interaction worth recording.** `SimpleProxy.__init__` tested `hasattr(src, "shape")` to decide
whether to fall back to `np.asarray` — so adding `.shape` stopped it widening utf8 operands, and a
`startswith` test failed. That widening is genuinely wanted (the compute engine indexes chunk-wise
into fixed-width elements; the span driver is the path that avoids it), so it is now explicit rather
than incidental, and goes through `astype()`, which sizes the result without decoding a row. A
reminder that `hasattr` probes for capability make silent contracts out of missing attributes.

Deliberately not done: making `UTF8Spec.dtype` return `StringDType()`. It would have to stay lazy
for NumPy 1.26, `dtype is None` is load-bearing at three sites, and `Column.dtype` already reports
`StringDType()` — so the win is cosmetic.
