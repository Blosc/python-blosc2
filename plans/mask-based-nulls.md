# Mask-based nullable columns for CTable

> **Status: IN PROGRESS — Phases 0–4 landed 2026-08-08.** Mask columns are now fully usable
> in memory and on disk; next up is Phase 5 (expressions + reductions), then Phase 6
> (Arrow/Parquet). Two premises were disproven during implementation and are corrected in place,
> each in a blockquote beside the text it corrects: the index path cannot be fixed by a null-aware
> expression (§Expression layer), and the bool dtype-flip cannot move out of `__init__`
> (§Schema layer). Drafted 2026-08-08.
> Revised 2026-08-08 after review: lazy sidecar materialization (decision 9), `NullPolicy`
> inference instead of raising, staged default flip (Phase 9), null-predicate rewrite pulled
> forward to Phase 1, sidecar suffix renamed `.notnull`.
>
> **This is the reference document for how CTable stores null values from now on.** It
> supersedes the sentinel-only decisions recorded in `plans/ctable-nulls.md` (whose non-goals
> list *"add separate validity bitmap storage for scalar CTable columns"*),
> `plans/enhancing-ctable.md` §Gap C, `plans/enhancing-ctable-phase2.md` §P5, and
> `plans/enhancing-ctable-phase3.md` §"Scope decision". Those remain accurate records of what
> was decided and shipped at the time and are not being amended; read them as history, and
> read this document for intent.

## Context

CTable represents nulls three different ways today: an **in-band sentinel** (`null_value`) for
numeric/bool/timestamp/string/bytes/utf8/ndarray, a **reserved code** `-1` for dictionary
columns, and **native `None`** for vlstring/vlbytes/list/struct/object. The sentinel model is
the default and it is lossy by construction — a sentinel steals a value from the dtype's range:

- `bool(nullable=True)` silently becomes physical `uint8` with `0/1/255`, leaking raw `255`
  through `col[:]` and row tuples, and requiring a whole layer of filter rewrites
  (`flag == True → raw == 1`) so nulls don't leak into predicates.
- `int8`/`uint8` cannot use their full range alongside nulls.
- Free-text `utf8`/`string` has no safe sentinel — any value is legal. `blosc2.utf8(null_value="\x00")`
  is *rejected outright* because NumPy 2.4 won't match a lone `"\x00"` against `StringDType`,
  which would silently stop marking anything as null.
- Parquet import already carries a workaround artifact in schema metadata:
  `"conversion": "nullable_scalar_wrapped_as_singleton_list"` — a nullable **bool** wrapped as a
  one-element list because the sentinel couldn't represent it.
- `_compiled_columns_from_arrow` (`ctable.py:7457`) outright **raises** when a nullable Arrow
  column has no available sentinel, so those types can't be imported at all.

The goal is lossless Arrow/Parquet round-trip for every scalar type, achieved by moving nullity
into a **sidecar validity array** per column — Arrow's own model — while keeping the sentinel
path fully supported for API compatibility and giving existing tables an explicit migration.

This design is already recorded and parked in `plans/enhancing-ctable-phase2.md` §P5
("Mask-based nullable columns: PARKED — design recorded, do not build yet"). Its first unpark
criterion, *"a user asks for nullable bool without the 255 reservation"*, is what this work
answers.

## Decisions (settled during planning — do not re-litigate)

1. **Mask becomes the default** for `nullable=True` on newly created tables — in two steps:
   the capability ships opt-in first (Phase 6), and the default flips no earlier than one
   release later (Phase 9), so version-3-capable readers are in circulation before
   default-created tables require them. Sentinel remains fully supported and readable forever,
   selectable per column (`null_value=...`, `null_storage="sentinel"`) or globally via
   `NullPolicy`. Existing on-disk tables keep working unchanged.
2. **V1 scope** = fixed-width scalars + utf8: numeric (incl. **complex**, which gains nullability
   for the first time), bool, timestamp, `string` (`U*`), `bytes` (`S*`), `utf8`, `ndarray`.
   Dictionary keeps `null_code=-1`; vlstring/vlbytes/list/struct/object keep native `None`.
   `is_null()` remains the uniform user-facing API across all kinds.
3. **Layout** = plain `np.bool_` NDArray, 1 byte/row, `True = valid` (Arrow polarity), mirroring
   `/_valid_rows`. Not bit-packed; `np.packbits` only at the Arrow boundary.
4. **Read semantics unchanged**: `col[:]` returns raw values with a deterministic fill in null
   slots; `is_null()`/`notnull()` read the mask. Adds opt-in `col.to_numpy(masked=True)`.
5. **Fill is loud**: `float → NaN`, `timestamp → int64.min` (decodes to `NaT` for free via
   `_maybe_decode_timestamp_values`), everything else dtype zero / `""` / `b""` / `0j`.
   The fill is **not** part of the format contract and is **not** recorded in the schema —
   recording it would recreate sentinel collisions at the metadata layer.
6. **NaN is a value, not a null**, in a mask-backed float column. Only `mask=False` is null.
   This matches Arrow and is the point of a side channel. Sentinel float columns keep
   NaN-as-null forever. Consequence: `dropna`/`groupby`/`min`/`max` differ between the two
   storages for float — must be documented and covered by the equivalence tests. Frame the
   docs positively — "mask columns follow Arrow semantics for NaN" — not as a changelog caveat.
7. **Nothing auto-migrates.** Not on `open()`, not on `copy()`, and — importantly —
   not on `save()`/`to_cframe()`, which must *preserve* each column's `null_storage`.
8. **Kleene three-valued logic stays out of scope.** Masks make it possible (they supply the
   validity channel `plans/enhancing-ctable.md` §Gap C named as the blocker), but
   `_null_aware_compare` deliberately collapses null → `False` (SQL `WHERE` semantics).
   Named deferred follow-up.
9. **An absent sidecar means all-valid.** The `.notnull` array is materialized lazily, on the
   first write that actually contains a null; a mask-storage column with no `.notnull` key on
   disk is a valid — and expected common — state meaning "no nulls so far". Null-free nullable
   columns therefore cost zero bytes on disk and zero read-path work: `null_pred()` returns
   `None`, which the expression layer already treats as never-null. It also makes
   `convert_nulls(to="mask")` on a null-free sentinel column a pure schema update.

## Architecture

### Invariants to encode in the code, not just here

- **Chunk pinning**: the mask array's `chunks[0]`/`blocks[0]` must equal the value column's, not
  whatever `compute_chunks_blocks` picks for a bool dtype. Otherwise `_nonnull_chunks`, the
  chunk-aligned writers, and index-segment alignment all re-align on every read.
- **Values under `mask=False` are unobservable** through the `Column` API.
- **Crash safety is inherited, not added**: `extend()` flips `self._valid_rows[start:end] = True`
  at `ctable.py:12923`, *after* every column write. Mask writes inserted before that line are
  invisible until the row goes live. Put a comment there so nobody reorders it.

### New module: `src/blosc2/ctable_nulls.py`

`ctable.py` is already 13.5k lines and already delegates to `ctable_storage.py` /
`ctable_indexing.py`; a fourth sibling fits.

```text
NULL_NONE, NULL_MASK, NULL_SENTINEL, NULL_CODE, NULL_NATIVE = "none", "mask", "sentinel", "code", "native"

class NullChannel:
    """Uniform read/write accessor for one column's validity channel.

    Subsumes all four representations CTable uses -- sidecar validity array,
    in-band sentinel, dictionary null code, native ``None`` -- so callers ask
    *what is null* without knowing which one a column uses.
    """
    __slots__ = ("_table", "_name", "_spec", "kind", "fill_value")

    def valid_array(self)          # physical NDArray; mask kind only, else None
    def null_pred(self)            # physical LazyExpr, True where null; None if never null
    def valid_pred(self)           # physical, True where valid
    def null_mask(self, key)       # logical numpy bool; key = slice | positions | None
    def valid_slice(self, a, b)    # physical numpy bool for [a, b) -- Arrow export
    def null_count(self)
    def coerce_batch(self, values, n)  # -> (values_with_fill, valid_np | None)
    def coerce_scalar(self, value)     # -> (storage_value, is_valid)
    def set_valid(self, key, valid); def resize(self, n); def gather(self, positions)
```

Owned by `CTable` as `self._null_channels: dict[str, NullChannel]`, invalidated whenever
`_schema`/`col_names` change. The mask NDArrays live in `CTable._null_masks`, opened **lazily**
via `storage.open_null_mask(name)` — mirror `_LazyColumnDict` (`ctable.py:3658-3746`) rather
than opening a sidecar per column on `open()`. Creation is lazy too (decision 9): `NullChannel`
materializes the sidecar the first time `coerce_batch`/`coerce_scalar` actually reports a null;
until then `valid_array()`/`null_pred()` return `None` and the column reads as never-null.

`Column` gets `self._nulls` and a public `Column.null_storage` property.

This module is what keeps the work from becoming a 50-site grep, so **it lands first,
sentinel-only, with zero behavior change** (Phase 0).

> **As built (2026-08-08).** Three departures from the sketch above, all in the same direction —
> away from snapshotting state that can go stale:
>
> - **Bound to a `Column`, not to `(table, name)`.** Logical reads (`is_null()`, `null_count()`)
>   have to honour the column's view — sorted order, row filter — and `Column` already carries
>   that. `Column._nulls` builds one on first use and keeps it; a table-level accessor can be
>   added in Phase 3 for the physical write paths, which have no `Column`.
> - **`kind` and `fill_value` are properties, not `__slots__` entries.** `_resolve_nullable_specs`
>   mutates `spec.null_value` *in place* (`ctable.py:4536`), so a channel that snapshotted at
>   construction would report the wrong kind afterwards. Everything reads through to the live
>   schema. Pinned by `test_channel_reads_through_to_live_schema`.
> - **`null_mask()` takes no `key` yet.** Dictionary columns answer through `_dictionary_eq`,
>   which is whole-column, so a `key` argument would have been silently ignored for them. It
>   arrives in Phase 4, when masks make it meaningful.
>
> The `NULL_*` constants and `fill_value_for` ended up **defined in `schema.py`** and re-exported
> here, because this module imports the spec classes from `schema.py` — that is the lower layer.
> Import them from either place.
>
> Also landed here, not in the sketch: `sentinel_mask`, `is_nan_sentinel`, `is_null_value`,
> `kind_of_spec`, `sentinel_guard_expr` and `rewrite_null_predicates` (Phase 1). Unifying the
> "is this sentinel a NaN" test incidentally fixed several sites that spelled it
> `isinstance(nv, float)` and so missed a `float32` NaN sentinel.

### Schema layer — `src/blosc2/schema.py`, `schema_compiler.py`

Every spec re-declares `nullable`/`null_value` today (`_NumericSpec:94`, `timestamp:251`,
`bool:285`, `string:335`, `bytes:384`, `UTF8Spec:622`, `NDArraySpec:784`). Add a
`_NullableSpecMixin` above `_NumericSpec` with `_init_nulls(...)`, `uses_mask`/`uses_sentinel`
properties, and `_null_metadata()`; each spec's `__init__` gains one `null_storage=None` kwarg
and one call, and each `to_metadata_dict` swaps its two nullability blocks for
`**self._null_metadata()`. `_null_metadata` **never emits `"sentinel"`**, so sentinel tables
serialize byte-identically to today.

Do *not* dodge the signature edits by post-setting `null_storage` in `spec_from_metadata_dict` —
`spec_cls(**data)` (`schema_compiler.py:447`) is the clean-fail mechanism for old readers.

Two spec fixes fall out:
- **`bool`** (`schema.py:275-300`): drop the `null_value != 255` rejection under mask storage, and
  move the `bool_ → uint8` dtype flip **out of `__init__`** into `_resolve_nullable_specs`, which
  already flips there (`ctable.py:4543`). Same for `NDArraySpec` bool (`ctable.py:4510-4516`, `4545-4551`).
- **`complex64`/`complex128`** (`schema.py:208-238`): gain nullability via the mixin, fill `0j`.

> **Correction (implemented 2026-08-08).** The dtype-flip relocation above is wrong as stated and
> **was not done**. `_resolve_nullable_specs` runs only on the *creation* paths; **opening a stored
> table never calls it** — `open()` rebuilds each spec through `spec_cls(**data)`
> (`schema_compiler.py:447`) and that is the whole of it. Moving the flip out of `__init__` would
> therefore bring every persisted nullable-bool column back as `np.bool_` while its bytes are
> `uint8`, silently misreading `255` as `True`. Verified by instrumenting the resolver across a
> save/open cycle.
>
> What shipped instead splits the responsibility by what each site can know. `__init__` resolves as
> far as metadata alone allows — `nullable and not uses_mask → uint8` — which is exactly the
> information a reopened table carries, so persisted columns come back correct with no resolver
> involved. `_resolve_nullable_specs` then corrects it **in both directions** once the policy has
> spoken, via `_unflip_mask_bool_dtype`: a bare `nullable=True` that resolves to mask gets its
> `uint8` undone. Same split for `NDArraySpec` bool. Regression-pinned by
> `test_stored_uint8_bool_reopens_as_uint8_without_the_resolver`.
>
> Also implemented, beyond what this section specified: **complex is mask-only**. There is no
> complex value safe to reserve, so `complex64(null_value=...)` raises and `nullable=True` resolves
> to mask regardless of the policy default. The spec classes carry a `supports_sentinel` class flag
> for this, which also replaced a `hasattr(spec_cls(), "null_value")` duck-type test in the Arrow
> importer (`ctable.py:7256`) that the mixin would otherwise have made answer True for complex.

**Version gating** — `schema_to_dict` (`schema_compiler.py:487`) computes the version as an
explicit feature max:

```python
uses_mask = any(getattr(c.spec, "null_storage", None) == "mask" for c in schema.columns)
schema_version = (
    3 if uses_mask else (2 if schema.metadata.get("nested") is not None else 1)
)
```

and `schema_from_dict` accepts `(1, 2, 3)`. This deviates from P5's "no global version bump" but
preserves its actual goal — the bump is *conditional on a mask column existing*, so only
mask-using tables are unreadable by old readers. What it buys is the failure message: a readable
`ValueError: Unsupported schema version 3` instead of a `TypeError` from deep inside
`spec_cls(**data)`. Include a hint naming `convert_nulls(to="sentinel")`.

**`NullPolicy`** (`ctable.py:117-197`) gains `null_storage: Literal["mask", "sentinel"]`
(default `"sentinel"` until Phase 9 flips it).
`CTable._resolve_nullable_specs` (`ctable.py:4496-4553`) stays the single decision point, resolving
in order: explicit `spec.null_storage` → explicit `spec.null_value` (⇒ sentinel) →
`policy.column_null_values[name]` (⇒ sentinel) → a set type-wide sentinel field matching the
column's kind (⇒ sentinel) → `policy.null_storage`. Under `"mask"` it skips sentinel selection
entirely — **no `max_length` widening, no bool dtype flip**.

Note: once the default flips, `NullPolicy`'s type-wide sentinel fields (`string_value`,
`float_value`, `signed_int_strategy`, …) would become silently inert for plain `nullable=True`.
Do **not** have `__post_init__` raise on that combination — that would break existing working
code (`NullPolicy(float_value=...)`) on the very release that flips the default. Instead,
setting any type-wide sentinel field **implies `null_storage="sentinel"`** for the types it
covers (the resolution-order entry above); `__post_init__` raises only when
`null_storage="mask"` is passed *explicitly* alongside them, which is a genuine contradiction
the user wrote. `column_null_values` stays meaningful — it forces sentinel per column.

Add one function `schema.fill_value_for(spec)` implementing decision 5.

### Storage layer — `src/blosc2/ctable_storage.py`

New key suffix beside `_UTF8_DATA_SUFFIX` (`:410`), collision-free by the same documented
argument (no column name can map to a key containing a literal `.`):

```python
_NOTNULL_SUFFIX = ".notnull"  # -> /_cols/<pct-encoded name>.notnull, extension .b2nd
```

Named `.notnull` — deliberately **not** `.valid` — because `/_valid_rows` already exists and
means something different (row liveness vs. per-column null validity); two bool arrays with
near-identical names would invite conflation. The name also states its polarity: `True` = not
null, matching Arrow.

Five new `TableStorage` methods (`create_null_mask` / `install_null_mask` / `open_null_mask` /
`has_null_mask` / `delete_null_mask`), implemented across all four backends exactly as
`create_valid_rows`/`open_valid_rows` (`:132-141`) already are. Per decision 9,
`has_null_mask(name) is False` is the common case and means all rows valid; `create_null_mask`
is called by `NullChannel` on the first null actually written, never at column creation:

| backend | notes |
|---|---|
| `InMemoryTableStorage:215` | `blosc2.zeros(shape, np.bool_, ...)`; `open_*` raises as `open_column` does |
| `FileTableStorage:560` | `store[key + _NOTNULL_SUFFIX] = arr` |
| `TreeStoreTableStorage:1081` | copy the utf8-companion block at `:1304-1319` (`_dest_path`, `map_tree`, `_modified`) |
| `EmbedStoreTableStorage:413` | read-only: `open_null_mask` only; creates go in the `_not_supported` list at `:527-537` |

The mask is a TreeStore *key* like the utf8 offsets array, so `.b2z`/`mmap_mode` are handled by
`_open_store()[key]` — no `store.offsets` special-casing (`open_varlen_scalar_column:737-741`
is the precedent).

`delete_column` / `rename_column` already carry a bespoke `+_UTF8_DATA_SUFFIX` branch in both
`FileTableStorage` (`:901-933`) and `TreeStoreTableStorage` (`:1496-1540`). **Generalize to a loop
over `_COMPANION_KEY_SUFFIXES = (_UTF8_DATA_SUFFIX, _NOTNULL_SUFFIX)`** rather than adding a third copy.

Capacity/persistence sites in `ctable.py`: `_grow` (`:4902`), `trim_capacity` (`:4875`),
`compact` (`:11391`), `_save_to_storage` (`:5827` — **both** the `install_column` reblock fast path
at `:5952` and the `create_column` path at `:5964`), `to_cframe` (`:5754`, per-column loop `:5809-5823`),
and **`CTable.load` (`:6099`), a second parallel open path that is easy to miss**. All of these
operate only on masks that exist — an absent sidecar needs no growing, trimming, or copying,
which is most of the lazy-materialization payoff.

> **As built (2026-08-08).** All five storage methods, four backends, both companion-suffix loops,
> and every capacity/persistence site above. Six notes, the first two of which are corrections:
>
> - **The site list was incomplete.** `copy()`'s in-memory path (`ctable.py:12563`) builds its
>   result through `_empty_copy` + a per-column gather and never touches `_save_to_storage`, so a
>   copied table would have silently dropped its sidecars. It gathers them by the same `live_pos`
>   now. `to_b2z`/`to_b2d` need nothing: their physical-pack fast paths zip the TreeStore leaves
>   as-is, and `.notnull` is one of them.
> - **`resize()` zero-fills, i.e. *invalid*.** `_grow` must write `True` over the new tail
>   explicitly, or every appended row past the old capacity reads as null. Same for the freed tail
>   in `compact`. This is the one place where "absent means all-valid" and "present means read the
>   bytes" have to be reconciled by hand. Pinned by `test_grow_extends_the_sidecar_as_all_valid`.
> - **Iteration must consult storage, not just the cache.** `_existing_null_masks()` asks the
>   storage backend per mask-storage column rather than iterating what happens to be open;
>   `_grow`/`trim_capacity` on a *freshly reopened* table have opened nothing yet, and skipping an
>   unopened sidecar would leave it out of step with its column. Pinned by
>   `test_capacity_paths_find_an_unopened_sidecar`.
> - **Chunk pinning has a documented fallback.** `_null_mask_grid` pins to the value column's
>   `chunks[0]`/`blocks[0]`, dictionary columns to their `codes`, and anything whose payload is not
>   a plain row-indexed NDArray — utf8, whose offsets array carries `n + 1` entries — to the
>   table-wide `_valid_rows` grid, which is the same shared grid the fixed-width columns use.
>   `_save_to_storage` records the grid each column *actually landed on* (`dest_grids`) rather than
>   re-deriving it, because `chunks_override` and the reblock fast path can both change it.
> - **`_null_masks` is a property that resolves through `base`**, so the six view-construction
>   sites that build a `CTable` via `__new__` needed no edits: a view shares its base's sidecars
>   exactly as it already shares its `_cols` NDArrays.
> - **Rename drops the cached handle on disk, carries it in memory.** `storage.rename_column`
>   re-keys the sidecar, so a cached handle points at a key that no longer exists; in-memory
>   storage re-keys nothing, so there the handle *is* the sidecar and must be carried.

### Read / write paths

Today fixed-width scalar columns cannot accept `None` at all — `_coerce_row_to_storage`'s
else-branch (`ctable.py:4823`) is `np.array(val, dtype=col.dtype).item()`, so users write the
sentinel literally. Under masks **`None` becomes the canonical way to write a null**, which is
new capability, not re-plumbing. `NullChannel.coerce_batch` null-detection rules:

- Python object sequence: `v is None` → null; `pandas.NA`/`pyarrow.NA` → null (duck-typed, **not** `float('nan')`).
- NumPy float array input: NaN is a **value** (decision 6).
- `np.ma.MaskedArray` input: `~arr.mask` is the validity verbatim.
- Arrow/pandas nullable input: the source's own validity is authoritative.
- Returns `valid_np is None` when nothing was null, so callers skip the mask write entirely.

Sites: `extend` (`:12718-12927` — coercion loop `:12845-12886`, write loop `:12898-12920`, and the
timestamp `None`-substitution at `:12855-12873`); `append` (`:12628`) plus
`_coerce_row_to_storage` returning `(dict, null_names)`; `Column.assign` (`:2503`);
`_write_arrow_batch` (`:7759`) gaining a parallel `_ChunkAlignedWriter` per masked column.

**`Column.__getitem__` / `_values_from_key` (`:1219-1316`) do not change** — that's what makes the
read side cheap.

**`Column.__setitem__` (`:1394-1527`) is the highest-risk function in the plan**: four key
branches × three fast paths plus two chunked loops (`:1476-1490`, `:1511-1518`). For V1, route
mask columns through the single unified `else` path (`:1491-1520`) — compute `phys_indices`
explicitly, coerce once, write values and mask together — and skip the fast paths. Measure, then
re-add them in a follow-up. Threading mask writes through all six paths at once is where this
project would break.

> **As built (2026-08-08).** `__setitem__` was not the hard part; the reroute was one added
> condition on the NDArray fast path plus one `_assign_validity` call per branch. Four things that
> were not in this section turned out to matter more:
>
> - **`NullChannel` must not be cached on its `Column`.** The pair was a two-object reference cycle
>   (`Column._null_channel → NullChannel._col → Column`), and since a `Column` also holds its
>   `CTable`, the moment `extend` started building a channel per column *every* write pinned a whole
>   table until the next gc pass. Caught by `test_persistent_releases_without_gc`, which had been
>   passing since long before this work. A weakref back to the column is the wrong fix — the channel
>   is often the only owner (`table._null_channel(name)` builds a throwaway `Column`) — so the cache
>   is gone instead and `_nulls` builds a fresh one-slot object per access. Pinned by
>   `test_a_table_is_freed_without_a_gc_pass`.
> - **Validation runs before coercion**, so `schema_vectorized` and its ndarray branch had to learn
>   that a bare `None` in the batch is a null for a mask column: a null cell has no value to
>   constrain. Without this, `extend` raised out of `_validate_string_lengths` before reaching any
>   of the new code.
> - **An overwrite replaces validity, it does not merge it.** `assign`/`__setitem__` must write
>   `True` back over rows that were null before, which means `coerce_batch` returning `valid=None`
>   ("nothing null in this batch") cannot simply be skipped there the way it can on append.
>   `Column._assign_validity` is that distinction, and the one exception it keeps is a column with
>   no sidecar at all, which is already all-valid.
> - **Every materialization path had to be found, not just the write paths.** `sort_by` (all three
>   forms), `take`, `slice`, and `_sorted_small_copy_from_live_positions` each rebuild a table from
>   gathered rows and silently dropped the sidecar. Factored into `_permute_null_masks` (in-place,
>   shared with `compact`) and `_gather_null_masks_into` (copy). The latter skips columns whose
>   gathered selection happens to be all-valid, so a copy that contains no null stays sidecar-free.
>
> Also landed here rather than in Phase 5: `null_pred`/`valid_pred` return the sidecar for mask
> columns. It is six lines inside `NullChannel`, and `_lazy_nonnull_mask` already consumes them, so
> leaving it out would have shipped aggregates that silently counted fill values. Fixed-shape
> ndarray columns still return `None` (an N-D values array does not broadcast against a
> one-flag-per-row sidecar) — that part stays Phase 5.

### Null API — `ctable.py:2587-2721`

`is_null()` (`:2627`) becomes `~channel.null_mask(...)`, i.e. O(1 byte/row) instead of
O(itemsize/row). `null_count()` (`:2645`) on a hole-free base table is
`n - blosc2.count_nonzero(mask[:n])` — bool NDArrays compress to near nothing, so effectively
O(chunks). `fillna` (`:2659`) becomes correct even when the fill collides with real data
(impossible under sentinels). `_nonnull_chunks` (`:2673`) zips value and mask chunks — this is
what the chunk-pinning invariant is for. New `Column.to_numpy(masked=False)` returns
`np.ma.MaskedArray` when `masked=True` and **must work for sentinel columns too** (derive the
mask from the sentinel) so the API is uniform. `_is_nullable_column` (`:13265`) becomes
`channel.kind != NULL_NONE`; `info_items` (`:1738`) gains `null_storage`.

Struct/object columns keep the per-row Python loop at `:2638` — not made worse, worth a docs note.

### Expression layer

This gets *simpler*, because Phase 0 already centralized it. `Column._raw_null_pred()`
(`:1863-1881`) becomes `return self._nulls.null_pred()`, which for masks is `~mask_ndarray`
(a `LazyExpr` that composes with `&`). **`_combined_null_pred` (`:1883`), `_null_aware_arith`
(`:1899`), `_null_aware_compare` (`:1913`) and `NullableExpr` (`:859-1060`) are unchanged** —
they already consume an opaque boolean predicate. Keep null-pred (not validity) polarity so the
diff stays at zero; the single `~` fuses into the lazy expression.

Two gains fall out: `_raw_null_pred` currently returns `None` for `is_ndarray` columns because a
per-item sentinel mask doesn't align 1:1 with rows — a mask *is* row-level, so **ndarray columns
gain null propagation in expressions for free**. And `_lazy_nonnull_mask` (`:2775-2802`) reads a
stored array instead of synthesizing a comparison, keeping the miniexpr reduction fast path with
one fewer computed operand.

`_is_nullable_bool` (`:1824-1831`) becomes `kind == "bool" and channel.kind == NULL_SENTINEL`;
the `raw_col == 1` rewrites (`:2039`, `:2049`, `:2767`, `:13392`) go dead for mask bools and stay
alive forever for sentinel ones.

### Arrow / Parquet

**Export** — `iter_arrow_batches` (`:6981-7122`): hoist `valid = channel.valid_slice(start, stop)`
to the top of the per-column body, then per branch pass `mask=~valid` to `pa.array` (pyarrow packs
the numpy bool itself). The bool branch (`:7103-7107`) loses its `arr == 1`; the `U`/`S` branch
(`:7093-7102`) loses its `[None if null_mask[i] else v ...]` Python list comprehension; the ndarray
branch (`:7070-7087`) stops requiring *every element* to equal the sentinel (a lossy, surprising
rule) and uses row-level validity. Dictionary (`:7043`) and varlen/list/struct (`:7038`) are unchanged.

`UTF8Array.arrow_slice` (`_utf8_array.py:961-984`) is the **one** place `np.packbits` is genuinely
needed, because it builds Arrow buffers directly:

```python
validity = pa.py_buffer(np.packbits(valid, bitorder="little"))  # bitorder is MANDATORY
```

Arrow validity bitmaps are LSB-first. This is the easiest bug in the plan to introduce and the
hardest to catch — a round-trip test passes with **either** bit order if import unpacks the same
way. Pin it with a test asserting the literal packed bytes for a known pattern.

**Import** — in `_compiled_columns_from_arrow` (`:7369-7482`), when the resolved storage is
`"mask"` the entire sentinel-selection block is skipped and **the `"no null_value sentinel is
available"` error at `:7457` never fires**. That single deletion is what makes nullable bool,
full-range `int8`/`uint8`, and free-text utf8 importable at all. Add a `null_storage=` parameter
to `from_arrow`/`from_parquet` beside the nullable knob at `:8349`, defaulting to the policy.

New `_arrow_column_to_numpy_masked(arrow_col, col) -> (values, valid)` beside
`_arrow_column_to_numpy` (`:7794-7828`): `arrow_col.fill_null(fill_value_for(spec)).to_numpy(zero_copy_only=False)`
(no Python loop, and the fill is exactly ours), with `valid = arrow_col.is_valid().to_numpy(...)`
for correctness first — optimize to `np.unpackbits(..., bitorder="little", count=n)` off the raw
validity buffer later, honoring `arrow_col.offset` and chunking.

When extending an **existing** table from Arrow, the stored schema's `null_storage` wins; only
inferred schemas consult the policy.

**Round-trip contract** (stated and tested): `to_arrow(from_arrow(x)).equals(x)` exactly, for
nullable `bool`; `int8`/`uint8` using **all 256 values** plus nulls; `float64` containing `nan`,
`±inf`, `0.0`, `-0.0` **as values** plus separate nulls; `utf8` containing `""`, `"\x00"`,
`"__BLOSC2_NULL__"`, 4-byte UTF-8 plus nulls; `timestamp` with `int64.min` as a value plus
separate nulls; `string(max_length=4)`/`bytes(max_length=4)` fully occupying the width. **None of
these round-trip under sentinels.** Same list for Parquet.

### Reductions and summary indexes

Mechanical: `_ndarray_values_for_reduction` (`:2843-2867`) and `argmin`/`argmax` (`:3218-3271`)
swap `_null_mask_for` for `channel.null_mask`.

**`_summary_minmax_source` (`:3004-3068`) is NOT fixed by masks** — verified. The bail at `:3046`
(`if nullable and not is_nan_float`) exists because a non-NaN sentinel pollutes per-block extrema,
and the **fill value pollutes them identically**: the summary builder reads the physical array and
never consults a side channel. Two honest routes:

1. *Free, partial*: with the NaN float fill (decision 5), mask-backed float columns qualify under
   the existing `is_nan_float` escape hatch at `:3045` with a one-condition change. Floats only —
   `int64.min` **is** the block minimum, so timestamps get nothing free.
2. *Real fix (Phase 10)*: make the summary builder in `ctable_indexing.py` mask-aware — extrema over
   `values[valid]`, a per-segment `all_null` flag, `"null_aware": true` in the descriptor. This
   retroactively enables the fast path for **sentinel** columns too.

Until (2) lands, mask columns take the same bail as sentinel columns. **Do not ship a fast path
that is silently wrong.**

### Sort and indexes

`_build_lex_keys` (`:11653-11730`): the null-indicator key becomes `(~valid[live_pos]).astype(np.intp)` —
cheaper than a sentinel compare, and no string comparison for `U`/`S`. Semantics unchanged.

`_sorted_positions_from_full_index` (`:11519-11651`): line `:11633` currently reads the **entire raw
column** just to compute `null_phys`; with a mask that becomes `~np.asarray(mask[:])`, an 8×–64×
I/O reduction on exactly the path whose comment at `:11631` apologizes for its temporaries.
**Highest value-per-line change in the plan.**

`_utf8_rank_arrays` (`ctable_indexing.py:99-143`) — verified landmine. Line `:120` is
`is_null = uniques == null_value`; under masks there's no sentinel in the vocabulary and the `""`
fill factorizes as an ordinary entry with **rank 0, so nulls would sort first and collide with
genuine empty strings**. New signature `_utf8_rank_arrays(col, n_phys, null_value=None, *, valid=None)`
with `ranks[~valid] = null_rank` after the `code_to_rank[codes]` gather, and `"null_aware": True`
in the returned meta. Comment the hazard in place. `_DictRankWrapper` (`:167-209`) is unchanged.

Index descriptors gain `{"null_aware": true, "null_order": "last"}` (anticipated by
`plans/ctable-nulls.md:614-623`, present in neither `plans/ctable-indexes-opsi.md` nor the code).
Read with `.get("null_aware", False)`; bump the build token so stale indexes rebuild.

**The indexed-OR bail (`ctable_indexing.py:1459-1461`) is not fixed by masks either** — the problem
is that global post-filtering (`_exclude_null_positions:1498-1509`) drops rows that legitimately
match via the *other* branch. The right fix is per-leaf and **independent of storage**: add
`_rewrite_null_predicates(expr, operands)` alongside `_rewrite_dictionary_predicates` (`:12953`)
and `_rewrite_utf8_predicates` (`:13158`), rewriting each nullable comparison leaf `a > 10` into
`(a > 10) & _valid_a` (`~mask` for mask columns, `a != nv` for sentinel ones). Because this helps
sentinel columns that exist today and is independent of storage, **it is pulled forward: it is
Phase 1**, landing right after the `NullChannel` refactor and before any mask work.

> **Correction (implemented 2026-08-08).** This paragraph originally claimed that "AND and OR both
> become correct with no post-filter", so `_exclude_null_positions` and the
> `nullable_indexed`/`nullable_needs_exclude` bookkeeping at `:1443-1457` would disappear.
> **That is wrong, and Phase 1 shipped with the post-filter retained.** An ordered index does not
> *evaluate* the predicate: it answers `a > 90` by taking a **range of the sorted column**, and the
> sentinel lies inside that range whether or not it would satisfy the comparison. Measured on 1M
> rows with a NaN sentinel: the scan matches 83 146 rows, while the FULL index returns 160 070
> positions, 76 924 of them NaN. Making the *expression* null-aware therefore cannot make the
> *index result* correct — the post-filter is load-bearing, and the indexed-OR bail stays too.
> Genuinely indexed OR over a nullable column needs the index itself to know about nulls, which is
> Phase 10 (`null_aware`/`null_order` descriptors), not something the expression layer can deliver.
>
> What Phase 1 did land is larger than this paragraph anticipated: **string predicates were not
> null-aware at all**. Only the operator form (`Column._null_aware_compare`) forced nulls to False;
> `t.where("a > 10")` compared the raw sentinel, so any sentinel that satisfies the predicate
> (`null_value=999` against `> 10`) returned its nulls as matches — on the scan path, indexed or
> not. That is the bug the rewrite fixes, and it makes the scan fallback correct, which is what
> finally makes the *bailing* OR path return the right answer.
>
> Two refinements the original text did not have:
>
> - **Guards are emitted inline, not as an injected operand.** `(a > 90) & (a != 999)` keeps the
>   guard a predicate on the same column; an opaque extra boolean operand pushed the planner off
>   the index onto a full scan (measured ~13 % slower on the OR probe).
> - **A guard is emitted only when the sentinel could actually satisfy the leaf.** `-1` cannot
>   match `> 10`, and NaN cannot match anything but `!=`, so those leaves are left alone. Without
>   this, every nullable-column query loses its index. Never applied inside a negation: a sentinel
>   that *fails* `a > 10` *passes* `not (a > 10)`.
>
> **Negation caveat, resolved.** The original text proposed either bailing on `~`/`not` or pushing
> validity to the outermost conjunction. Bailing was rejected: the index path is SQL-correct for
> negation today while the scan path is not, so bailing would have regressed the index. Validity is
> pushed to the negation point — `(~(a > 10)) & valid_a` — which is exact for every tested form and
> conservative in one three-valued corner: `not (a > 10 and b == 999)` with a false second term
> makes SQL's `NULL AND FALSE` collapse to `FALSE`, so the row should survive, while the guard drops
> it. Rows are only ever dropped, never wrongly returned.
>
> **Addendum (2026-08-08): the *operator* form has its own negation leak — a pre-existing bug, not
> a Phase 1 artifact.** `_null_aware_compare` collapses null → False at the comparison leaf and
> returns a plain `LazyExpr`, so `~(t.a > 10)` inverts the collapsed False and **wrongly returns
> null rows** — the failure direction the guarantee above rules out for the string form. Measured:
> `[0, 2]` where SQL and the string form both give `[0]`. It also means the operator form's SQL-
> exact answer in the three-valued corner above is accidental (plain booleans at `~`, not Kleene
> logic). Both behaviors are pinned in `tests/ctable/test_null_predicate_rewrite.py`:
> `test_negation_over_and_corner_is_conservative` (the intentional string/operator divergence) and
> a `strict=True` xfail, `test_operator_form_negation_drops_nulls` (the leak). A real fix needs
> comparison results to carry their null predicate through `~` — a boolean analogue of
> `NullableExpr` with `__invert__` — and folds naturally into decision 8's deferred Kleene
> follow-up rather than Phase 1.

### Groupby

`groupby.py:_null_mask` (`:2116-2137`) is already the single central helper, but it receives an
already-gathered `values` chunk, so validity has to be gathered at each of ~8 independent read
sites: `:534`, `:583-589`, `:624-637`, `:716-718`, `:1005-1017`, `:1394`, `:1430`, `:1757`, plus
`_null_value_for` (`:2589`) and `_null_output_value` (`:2239`), which for a mask-backed output
column become "write fill + mask=False". Signature grows `*, valid=None`. **Messiest integration
after `__setitem__`** — budget accordingly.

One semantic improvement follows from decision 6: NaN in a float *value* column is no longer
missing. Keep the `is_key` NaN coercion at `:2131-2134` for keys (dropna semantics).

### Nullable-bool cleanup

Under masks, `bool(nullable=True)` yields physical `np.bool_` — no `uint8`, no reserved `255`.
`t.flag[:]` returns booleans; `t.where(t.flag)` works directly; export is
`pa.array(arr, mask=~valid, type=pa.bool_())`.

**No deprecation cycle is needed, and that's the point of the design**: the change is scoped by
construction to tables created *after* the default flips. Existing tables carry `null_value: 255`
in their schema and stay uint8 forever; `blosc2.bool(nullable=True, null_value=255)` and
`null_storage="sentinel"` keep producing uint8 explicitly; `_is_nullable_bool` and its rewrite
sites stay permanently. What is needed: the two dtype-flip relocations, a
`doc/reference/ctable.rst` note, a release-note entry, and a test that opens a checked-in fixture
table with a uint8 nullable-bool column and asserts nothing changed.

### Migration

```python
def convert_nulls(
    self, columns=None, *, to="mask", null_value=None, inplace=False
) -> CTable:
    """Convert nullable columns between sentinel and validity-mask storage.

    Never called implicitly: opening, copying, and saving a table all preserve
    each column's existing null storage.
    """
```

Per kind for `to="mask"`: a null-free sentinel column (no slot holds the sentinel) converts as
a **pure schema update** — no sidecar is written, per decision 9. Otherwise
numeric/timestamp/string/bytes convert chunkwise in one pass
(`valid = ~sentinel_mask(chunk)`; overwrite sentinel slots with the fill), dtype unchanged,
in-place possible — but `string`/`bytes` keep their widened `max_length` (shrinking is a dtype
change; document `copy()` as the way to reclaim it, don't shrink silently). `bool` needs
uint8 → `np.bool_`, i.e. a new array. `ndarray` derives the row mask under the old all-elements
rule first. `utf8` rewrites null rows to zero-length spans (effectively a column rebuild).
Dictionary and varlen/list/struct/object are **no-ops** — raise if named explicitly, skip silently
if implicit.

`to="sentinel"` is the inverse and must **reject** what it cannot represent: full-range
`int8`/`uint8`, and utf8/string whose data already contains the proposed sentinel. Check before
writing; raise naming the offending value.

`inplace=True` ordering **is** the crash-safety argument, so put it in the docstring: (1) write
the complete `.notnull` sidecar, (2) rewrite value slots to fill, (3) update `/_meta` schema last.
A crash after (1) or (2) leaves the schema saying `sentinel`, the orphan `.notnull` key unread, and
the table intact. `inplace=False` (default, recommended) builds a new table via `copy()`.

Detection is `Column.null_storage` plus an `info()` column — no separate report function.

## Verification

**Differential oracle — the single highest-value test.** New
`tests/ctable/test_null_storage_equivalence.py`: build the same logical data twice
(sentinel-backed and mask-backed), assert every public API agrees — `is_null`, `notnull`,
`null_count`, `fillna`, `dropna`, `sort_by` (both directions, single and multi-key), `group_by`
(as key and as value, `dropna=True/False`), `where` (indexed and unindexed, AND and OR), all
reductions, `argmin`/`argmax`, `unique`, `value_counts`, `to_arrow`, `to_pandas` — parametrized
over every V1 kind, with float NaN cases marked as intentionally divergent per decision 6. This
converts "mask is a second implementation" from a permanent liability into a checked invariant.

Additions to existing suites:
- `tests/ctable/test_nullable.py` (756 → ~1150): parametrize over `null_storage`; mask-only cases
  for nullable bool `dtype == np.bool_`, full-range `int8`, utf8 containing `"__BLOSC2_NULL__"`/`""`/`"\x00"`,
  `string(max_length=4, nullable=True)` staying `U4` (regression against the widening at `:4538`),
  fill determinism, `to_numpy(masked=True)` for both storages, `null_count` with deletions,
  `is_null()` on sorted and `where()` views, `fillna` with a value equal to real data, and `None`
  accepted by `append`/`extend`/`__setitem__`/`assign` for every V1 kind. Plus the decision-9
  invariant: a mask column written with no nulls has **no** `.notnull` key on disk and answers
  `is_null()` all-`False`; the key appears exactly when the first null is written. And the
  `NullPolicy` resolution rules: type-wide sentinel fields imply sentinel storage for the types
  they cover; explicit `null_storage="mask"` combined with them raises.
- `test_null_expressions.py` (284 → ~440): mask/sentinel differential for arithmetic and comparison
  propagation; ndarray-column propagation (new capability); NaN-is-a-value assertions.
- `test_arrow_interop.py` (630 → ~840) and `test_parquet_interop.py` (1323 → ~1500): the round-trip
  contract list exhaustively, plus the literal-bytes bit-order test.

New files:
- `tests/ctable/test_null_persistence.py`: mask survives `.b2d`, `.b2z`, `to_cframe`/`ctable_from_cframe`,
  inline TreeStore save/load, `mmap_mode="r"`, `CTable.load()`, `delete_column`, `rename_column`,
  `compact()`, `trim_capacity()`, and repeated `_grow()` cycles.
- `tests/ctable/test_null_migration.py`: `convert_nulls` per kind both directions, in-place and copy;
  the null-free schema-only fast path; crash-ordering (an orphan `.notnull` key with an unmodified
  schema still reads as sentinel); and the two guarantees — old tables open unchanged, `save()`
  preserves storage under a mask-default policy.
- A version-gate test: hand-build a `version: 3` schema dict and assert a simulated old accept-list
  `(1, 2)` raises a clear `ValueError` naming the version.

End-to-end smoke, run manually: import a nullable-bool + full-range-`int8` + free-text-utf8 Parquet
file, round-trip it, and assert `pq.read_table(out).equals(pq.read_table(in))` — the case that is
impossible today. Also re-run the OFF importer round-trip (`plans/ctable-nulls.md` §Tests) and
confirm the `nullable_scalar_wrapped_as_singleton_list` workaround can be deleted.

## Phasing

Each phase is independently landable. **The default does not flip until Phase 9**, which is a
deliberate one-release-minimum lag behind Phase 6 so version-3-capable readers circulate before
default-created tables require them.

| # | Phase | Size | Risk |
|---|---|---|---|
| 0 | ✅ **`NullChannel` refactor, sentinel-only.** New `ctable_nulls.py`; route the ~40 `getattr(spec, "null_value")` sites through it across `ctable.py`, `groupby.py`, `ctable_indexing.py`, `schema_validation.py`, `schema_vectorized.py`. Test suite must pass **unmodified**. | M | Low |
| 1 | ✅ **Per-leaf null-predicate rewrite** *(storage-independent — pulled forward because it fixes sentinel tables that exist today)*. `_rewrite_null_predicates`, guards emitted inline and only where the sentinel could satisfy the leaf; validity pushed to the negation point. **`_exclude_null_positions` and the indexed-OR bail are retained** — see the correction above; an ordered index never evaluates the predicate, so a null-aware expression cannot fix it. Real payoff: string predicates become null-aware at all. | M | Med |
| 2 | ✅ **Schema plumbing.** `_NullableSpecMixin`, `null_storage` kwarg on ~9 specs, conditional version 3, `NullPolicy.null_storage` (**still defaulting to `"sentinel"`**) with sentinel-field inference, `_resolved_null_storage` as the single decision point, `fill_value_for`, complex nullable (mask-only). Dtype-flip relocation **not** done — see the correction above. | S | Low |
| 3 | ✅ **Storage sidecar.** 5 methods × 4 backends, lazy creation (absent key = all valid); `_grow`/`trim_capacity`/`compact`/`_save_to_storage`/`to_cframe`/`load`, **plus `copy()`'s in-memory path**, which the section above had missed; companion-suffix loop in delete/rename. `tests/ctable/test_null_persistence.py` (38 tests) drives it with a hand-built mask. | M | Low |
| 4 | ✅ **Read/write + null API.** `extend`/`append`/`_coerce_row_to_storage`/`__setitem__`/`assign`; `is_null`/`notnull`/`null_count`/`fillna`/`_nonnull_chunks`/`to_numpy(masked=)`/`dropna`; **plus every gather-and-rebuild path** (`sort_by` ×3, `take`, `slice`), which this section had not listed. Mask columns fully usable. `tests/ctable/test_null_mask_api.py` (90 tests). | **L** | **High** (turned out to be the reference cycle, not `__setitem__`) |
| 5 | **Expressions + reductions.** `_raw_null_pred`, `_lazy_nonnull_mask`, `_ndarray_values_for_reduction`, argmin/argmax, `_is_nullable_bool`. Includes the ndarray-propagation gain. *(The base `null_pred`/`valid_pred` mask support landed in Phase 4 — see the note above.)* | M | Med |
| 6 | **Arrow/Parquet.** Import + export for all V1 kinds, `packbits`/`unpackbits` LSB-first, `arrow_slice(validity=)`, delete the "no sentinel available" import error. Ships **opt-in** (`null_storage="mask"`); the default stays `"sentinel"`. | M | Med |
| 7 | **Sort + groupby.** `_build_lex_keys`, `_sorted_positions_from_full_index` (big I/O win), `_utf8_rank_arrays(valid=)`, groupby `_null_mask` threading. | M | Med-High |
| 8 | **Migration + docs.** `convert_nulls`, `Column.null_storage`, `info()`, `doc/reference/ctable.rst` null-policy rewrite, release notes. | S–M | Low |
| 9 | **Default flips to `"mask"`.** A one-line `NullPolicy` change plus release notes — lossless round-trip is why the default exists. Lands **no earlier than one release after Phase 6** so older readers in the wild already understand schema version 3. | S | Low |
| 10 | **Index null-awareness remainder** *(independent)*. Mask-aware summary builder; `null_aware`/`null_order` descriptors; re-enable `_summary_minmax_source` for mask and sentinel columns alike. | **L** | High |

The riskiest, most-coupled work is isolated into Phases 4, 7 and 10, each of which can slip
without blocking the others. Phase 9 is a policy change, not code — its only prerequisite is
that Phases 2–8 have soaked for a release.
