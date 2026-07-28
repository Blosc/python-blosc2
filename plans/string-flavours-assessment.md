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
| `create_index` | ✓ all 5 kinds | ✓ all 5 kinds | **✗ NotImpl (all kinds)** | ✓ rank, ordering only | ✗ |
| **Compute (string-returning)** | | | | | |
| `add_computed_column("'x='+c")` | ✓ | ✓ | **✗ NotImpl** | ✗ | ✗ |
| `assign(new=…)` | ✓ | ✓ | **✗ NotImpl** | ✗ | ✗ |
| `t.apply(dsl_kernel)` / `lazyudf` | ✓ | ✓ | **✗ ValueError** ¹ | ✗ | ✗ RuntimeError |
| nested (dotted) leaf in expr | ✓ | ✓ | ✗ NotImpl | ✗ | ✗ |
| **Bare container (no CTable)** | | | | | |
| `lazyexpr(expr, {a: col})` | ✓ NDArray | ✓ | ⚠ returns `<Un`, numpy fallback ² | ⚠ | ⚠ |
| `col == "scalar"` | ✓ LazyExpr | ✓ LazyExpr | ✓ bool mask ³ | ⚠ `False` | ⚠ `False` |
| **Interop** | | | | | |
| `to_arrow` | `string` | `large_binary` | `large_string` | `dictionary<…>` | `string` / `large_binary` |
| save + reopen | ✓ | ✓ | ✓ | ✓ | ✓ |
| NumPy requirement | any | any | ≥ 2.0 | any | any |

¹ `ValueError: malformed node or string … StringDType()` — `lazyudf` tries to allocate an NDArray
output with `dtype=StringDType()`, which `NDArray.dtype`'s `ast.literal_eval` round-trip cannot
parse (`blosc2_ext.pyx:3818`).
² Correct values down the wrong path: it never reaches miniexpr, ignores `_UTF8_EXPR_BUDGET`, and
loses the utf8 container.
³ Fixed in `3692673f` — was a plain `False` (object identity, silently wrong) because `Utf8Array`
defined no comparison operators. All six now return a boolean mask, answering a scalar `str` with
the existing raw-byte scanners so no row is decoded. `dictionary` and `vlstring` are still
identity-compared.

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
(`SUMMARY`/`BUCKET`/`PARTIAL`/`FULL`/`OPSI`) build on `string()` and `bytes()`; **all five raise
`NotImplementedError` on `utf8()`**, from a single guard at `ctable_indexing.py:753`.

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

- **`dictionary`'s FULL index gives no equality speedup** (25 ms indexed vs 25 ms unindexed at 2 M
  rows). That is by design — the rank index is built for *ordering*, not filtering — but the docs
  present `create_index` on dictionary as a plain ✓, which oversells it. Still open.

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

utf8 has the missing ingredient already: `Utf8Factorizer` (`_utf8_array.py:714`) produces global
codes plus a decoded vocabulary without decoding a single row, and `group_by` already runs it at
scale — 227 ms for 1 M rows above. So a utf8 rank index is roughly: factorize (~230 ms/M rows),
`argsort` the vocabulary, map codes→ranks, hand an int32 array to the existing builder. Build cost
would land near `<U`'s 1148 ms.

What it would and would not deliver, now that the dictionary version has been measured rather than
assumed:

- ✓ `sorted_slice` / `sort_by` top-*k*. Expect utf8 to land near dictionary's 51.3 ms — call it
  **~8× faster than utf8's current 427 ms**, still ~6× behind `<U`+FULL's 8.2 ms.
- ✗ equality and range filtering — dictionary's rank index does not accelerate those (25 ms indexed
  vs 25 ms unindexed at 2 M rows), so utf8's wouldn't either without more work. **utf8 would stay
  15×/40× behind an indexed `<U` on point and range lookups.**
- ✗ `startswith` — nothing indexes prefixes on any flavour.
- ⚠ staleness: ranks shift when new values arrive. Dictionary handles this with a stored
  `dict_hash` (`_dict_rank_hash`) and falls back to lexsort when stale; utf8 would need the same,
  and a utf8 column is much more likely to gain new values than a category column is — which means
  the fallback would fire often, and the fallback is a full lexsort.

Rough cost: **2–3 days**, most of it staleness and the persistent-sidecar round trip, not the rank
computation. Comparable to G2 in the parity plan, and it buys something G2 does not — but it is a
partial fix, not parity with an indexed `<U` column.

## Pros and cons

**`string()` / `bytes()` (fixed-width)** — everything works, always. Fastest filter of the lot, and
after compression only 27 % larger than utf8 on this data. Cost is the raw footprint: 4 bytes per
character per row, materialized in full on every read — 148 MB here, and the bench script's own
comment notes `<U130` at 10⁷ rows needs a ~5 GB ingest buffer. Needs `max_length` up front.

**`utf8()`** — 18× smaller raw footprint, no length ceiling, Arrow-native (`large_string`, the same
layout pandas 3 string columns use). The whole *query* surface is already at parity. Missing:
string-returning expressions, DSL kernels, `create_index`, dotted leaves. Reads need NumPy ≥ 2.0.

**`dictionary()`** — unbeatable for low-cardinality categories, the fastest `group_by` of any
flavour, and the only variable-length flavour with a working index. But it is not a general string
type: string *functions* don't reach it at all (`Unknown symbol`), and its index accelerates
ordering only, not filtering.

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
private `_cols["out"].set_all(...)`.

So the choice is not "parity vs. conversion" — it is **which rule do we publish**:

| | full compute parity (`utf8-string-support.md`) | storage+query utf8, compute on `<U` |
|---|---|---|
| effort | ~1 week (G2+G3+G5), ~2 with G4 | ~1–2 days |
| rule the user learns | "everything works everywhere" | "utf8 stores and filters; convert to compute" |
| perf honesty | hides a 3–5× penalty behind an identical API | the `.astype()` is visible, so the cost is |
| persistence risk | G2 must serialize `StringDType` — get it wrong and the **table won't reopen** | none; no new dtype is persisted |
| `create_index` on utf8 | still missing | still missing |
| top-*k* windows on utf8 | still 52× slower | still 52× slower |

The second column is the better trade for the *compute* surface, and it is the honest one: the span
driver's cost *is* a decode + `astype` per span, so making the user write `.astype()` describes what
actually happens.

But note the last two rows: **neither option touches the biggest measured gap.** Both plans leave
utf8 without an index, and the parity plan spends a week doing so. Given that the index rank trick
is already proven in-tree for dictionary and costs about as much as G2 alone, the priority order
inverts:

1. ~~**Fix what is wrong, not merely absent** — `Utf8Array.__eq__` returning `False`~~ — **done**,
   `3692673f`; `==`, `!=`, `<`, `<=`, `>`, `>=` all return boolean masks now. The bare-array
   `lazyexpr` numpy fallback (G4) is the remaining silent-wrong result and still worth fixing
   whichever rule is chosen.
2. **Publish the conversion pair.** `Utf8Array.astype("<U")` / `blosc2.from_utf8()` and `.to_utf8()`,
   plus `add_column(..., values=)` so the result can land back in the table without a private call.
   ~1–2 days, and it closes the whole compute asymmetry by declaring a rule instead of chasing it.
3. **Then the utf8 rank index** (2–3 d) — ~8× on top-*k*, using `Utf8Factorizer` exactly as
   dictionary uses its `dictionary` list. Worth doing, but go in knowing it does *not* close the
   15×/40× equality and range gap; if those workloads are the target, `<U` + FULL stays the answer
   and the docs should say so plainly.
4. **Make the error messages route.** `add_computed_column` on utf8 currently says only "not
   supported"; it should name the two-line workaround. That, more than parity, is what removes the
   confusion you are objecting to.
5. **Drop G2/G3/G5**, or park them behind a concrete user request. G2 in particular buys a
   `StringDType`-in-schema serialization hazard for a surface the conversion pair already covers.

Found while measuring, unrelated to the utf8 decision:

- ~~`sort_by` on an unindexed `dictionary` column takes 235 s at 1 M rows~~ — **fixed**, `a9446841`.
  Per-row decode; now 759 ms.
- ~~`kind=BUCKET` is a pessimization~~ — **fixed**, `92c39aa6`. Affected every indexable dtype.
- Still open: the doc table's plain ✓ for `create_index` on `dictionary` should read "ordering
  only", and `_build_lex_keys` could sort dictionary ranks instead of decoded strings (~4×).
