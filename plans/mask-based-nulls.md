# Mask-based nullable columns for CTable

> **Status: complete — Phases 0–10 landed 2026-08-08.** Lossless Arrow/Parquet
> round-trip works for every V1 kind, sort/groupby/query honour a sidecar, `convert_nulls` migrates
> columns in either direction, **mask storage is now the default** — a bare `nullable=True`
> resolves to it — and column indexes summarise only the rows that carry a value, so `min`/`max`
> and indexed `OR` no longer fall back on a nullable column. Six premises were disproven during
> implementation and are corrected in place, each in a blockquote beside the text it corrects: the
> index path cannot be fixed by a null-aware expression (§Expression layer), the bool dtype-flip
> cannot move out of `__init__` (§Schema layer), ndarray columns do not get lazy null propagation
> for free (§Expression layer), the "free" summary min/max fast path is unsound (§Reductions),
> `np.packbits` is not needed and avoiding it is safer (§Arrow/Parquet), and `.equals()` cannot
> express the round-trip contract (§Arrow/Parquet). Phase 7 added a seventh — **Phase 1 left the
> mask half of the query path undone**, and nobody noticed until sort work went looking
> (§Expression layer, "Addendum 2") — and Phase 8 an eighth: the in-place migration ordering
> recorded below is **wrong at its middle step**, and moving that step last makes every
> intermediate state correct rather than merely recoverable (§Migration). Phase 10 added a ninth,
> which is a partial retraction of the fourth: the summary fast path is unsound only for a *genuine*
> NaN, not for a whole mask float column, and `null_order` — planned beside `null_aware` — records
> a promise no index kind keeps (§Reductions, §Sort and indexes). It also narrows the first: an
> ordered index cannot be fixed by a null-aware expression, but the *segment* index was never
> ordered and needed no fixing, which is what lets indexed `OR` work
> (§Expression layer, "Addendum 3"). Drafted 2026-08-08.
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
   default-created tables require them. *(The two-step staging was not kept — both shipped in
   4.11.0; see the deviation note under §Phasing.)* Sentinel remains fully supported and readable forever,
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
   Named deferred follow-up. *(Discharged 2026-08-10 — see `plans/kleene-logic.md`. The collapse
   stays, as the **truth channel** of a three-valued result; what changed is that the null
   channel now survives alongside it, which is what `~` needs. `WHERE` semantics are unchanged.
   That work also reverses this document's ruling that a conservative string-form negation is
   good enough, and corrects two things recorded below: the negation caveat under §Expression
   layer, and the divergence pinned by `test_negation_over_and_corner_is_conservative`.)*
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

> **Correction (measured 2026-08-08).** The second gain is real and landed. **The first is not,
> and was not done.** Row-level is necessary but not sufficient: the values array is
> `(n, *item_shape)` and the sidecar is `(n,)`, and reshaping it to `(n, 1, …)` — the obvious fix —
> fails twice. `blosc2.where(pred_(n,1), nan, values_(n,3))` returns shape `(n, 1)`, silently
> **dropping the item dimension from the values** rather than broadcasting; and `NullableExpr`'s
> reduction mask combines the predicate with the `(n,)` `_valid_rows`, where `(n,) & (n,1)` explodes
> to `(n, n)`. Row-level null propagation for ndarray columns needs broadcasting support in the lazy
> layer, not a reshape at this call site.
>
> What ndarray columns *do* gain in this phase is null-aware **reductions**, which are NumPy-based
> and row-level throughout: `min`/`max`/`sum`/`mean`/`argmin`/`argmax` now skip null rows instead of
> reducing over the fill item. Pinned by `test_ndarray_columns_still_get_no_lazy_null_predicate`,
> which asserts both halves — no predicate, working reduction — so the gap stays deliberate.

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

> **As built (2026-08-08): `np.packbits` is not needed at all, and avoiding it is strictly safer.**
> `arrow_slice` already had the answer for its sentinel path — `pa.array(~mask).buffers()[1]`.
> Arrow packs booleans and validity bitmaps identically (LSB-first), so handing pyarrow the
> booleans and taking the resulting array's *data* buffer borrows its packing and makes the bit
> order unrepresentable-as-wrong rather than merely tested. The mask path does the same with
> `pa.array(valid).buffers()[1]`. The literal-bytes test was still worth writing and is the useful
> half of the advice above: `test_utf8_validity_bitmap_is_lsb_first` pins
> `[valid, null, valid, valid, valid, null, valid, valid]` to `0xDD`, not the MSB-first `0xBB`.

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

> **Correction (2026-08-08): `.equals()` cannot express this contract.** `pyarrow.Array.equals`
> compares floats with IEEE semantics, so two *identical* arrays containing NaN compare unequal —
> the float case of the list above fails by construction, whatever the implementation does. The
> tests use `assert_same_logical` instead, which compares what is actually observable: the validity
> bitmap, and the values under valid rows, with NaN equal to NaN and signed zeros kept distinct.
> Values under `valid=False` are deliberately **not** compared — the fill is explicitly not part of
> the format contract (decision 5), so asserting on it would pin something the design says may
> change.
>
> **The "none of these round-trip under sentinels" claim is understated, and now measured.** The
> sentinel path does not fail — it *silently returns different data*. `pa.array([-128, None, 127],
> int8)` imports and re-exports as `[None, None, 127]`, because `-128` is the sentinel `int8`
> picks; `["", "__BLOSC2_NULL__", None]` comes back as `["", None, None]`, because that literal is
> the utf8 sentinel. Both are pinned side by side against the mask result in
> `test_sentinel_storage_is_lossy_where_mask_storage_is_not`, which is the single most direct
> statement of why this project exists.

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

   > **Correction (measured 2026-08-08). Route 1 is unsound and was not taken.** It is defeated by
   > decision 6, three sections up: the summary builder drops NaNs, and under mask storage a NaN is
   > a *value*, so it drops real data too. On `[1.0, nan, 5.0, null, 3.0]` the scan gives `nan`
   > (NumPy semantics, NaN participates) while the summaries would answer `1.0`/`5.0` — the same
   > query returning different answers depending on whether an index happens to exist. Contrast the
   > sentinel-NaN column the hatch was written for, where NaN *is* the null, so dropping it is
   > exactly right and the two paths agree. Mask columns keep the bail; the reasoning is now a
   > comment at the bail site so nobody re-derives the one-liner. Both halves pinned by
   > `test_summary_minmax_shortcut_stays_disabled_for_mask_columns` and
   > `test_sentinel_nan_float_keeps_its_summary_shortcut`.
2. *Real fix (Phase 10)*: make the summary builder in `ctable_indexing.py` mask-aware — extrema over
   `values[valid]`, a per-segment `all_null` flag, `"null_aware": true` in the descriptor. This
   retroactively enables the fast path for **sentinel** columns too.

Until (2) lands, mask columns take the same bail as sentinel columns. **Do not ship a fast path
that is silently wrong.**

> **As built (2026-08-08, Phase 10).** Route 2, and it lands in `indexing.py` rather than
> `ctable_indexing.py` — the summary builder lives there; what `ctable_indexing.py` contributes is
> the validity channel to build with. Five notes:
>
> - **The builder takes a callable, not an array.** `validity(values, start, stop) -> valid | None`
>   is threaded through `_build_levels_descriptor{,_ooc}` and called per chunk. That shape is what
>   lets a *sentinel* column answer for free from the values the builder already decompressed
>   (`~sentinel_mask(values, nv)`) while a mask column reads one byte per row off its sidecar —
>   the same split every other phase made, arriving here unchanged.
> - **`FLAG_ALL_NULL` is set together with `FLAG_ALL_NAN`, deliberately.** The established meaning
>   of `FLAG_ALL_NAN` at every consumer is "the extrema are placeholders, skip this segment", which
>   is exactly what an all-null segment needs; riding along with it means the pruning path
>   (`_candidate_units_from_summary`) and the min/max path both got it right with no edit.
> - **The prediction that this is not a mask feature is confirmed, and pinned by a test that was
>   already there.** `test_minmax_matches_reference`'s `k` column — an `INT64_MIN`-sentinel `int64`
>   — was the suite's canonical *fallback* case, on the grounds that the sentinel **is** the block
>   minimum. It takes the shortcut now, and the parametrization flipped from `False` to `True`.
> - **`FLAG_HAS_NAN` computed over the valid rows is what makes decision 6 expressible.** Phase 5
>   disabled the shortcut for the whole mask-float column because a NaN fill and a NaN value were
>   indistinguishable to the summary. They are distinguishable to a *null-aware* summary: the flag
>   marks only a NaN among valid rows, so a mask float column qualifies as a source and declines
>   only when it actually holds a NaN — where the scan poisons to NaN and the summaries must not
>   contradict it. `test_summary_minmax_shortcut_stays_disabled_for_mask_columns` was rewritten in
>   place as `test_a_genuine_nan_still_keeps_a_mask_float_column_off_the_shortcut`, with its
>   converse beside it.
> - **Measured: 236.89x** for `min()` on a 20M-row nullable `int64` (39.32 ms → 0.17 ms), which is
>   the same order as the ~240x the non-nullable path already claimed. The cost is on the build
>   side: the per-block summaries folded incrementally during writes carry no validity, so a
>   nullable column holding a null cannot use them and pays one decompression pass at `close()`
>   (1.8 ms → 33.2 ms for that column). A nullable column with **no** nulls keeps the fast path
>   untouched — it needs no validity provider, and is marked `null_aware` anyway. Threading
>   validity into `_ColumnSummaryAccumulator` is the named follow-up; it was left out because it
>   reaches into `extend`'s write loop and one of its two feed sites (the Arrow writer's
>   `on_write=` callback) has no validity to give.

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

> **As built (2026-08-08, Phase 10): `null_aware` yes, `null_order` no, and no token bump.**
> `null_order` would have recorded something untrue. No index kind *reorders* nulls: a FULL index
> sorts them wherever their sentinel or fill lands, which is why `_sorted_slice_positions` bails
> (Phase 7) rather than locating a null run. The nulls-last contract lives in `_build_lex_keys`,
> not in a descriptor. Recording `"null_order": "last"` beside it would have read as a promise the
> index does not keep — see the follow-up on a null-run FULL index below for what would earn it.
>
> The token bump is not needed either, and skipping it is the safer choice: `.get("null_aware",
> False)` already makes an older index take the old bail, so it is *correct* rather than stale, and
> a bump would rebuild every stored index in the wild to buy a shortcut. This is the same
> distinction Phase 7 drew for `_utf8_rank_arrays`, whose pre-mask indexes were genuinely *wrong*
> and did have to be invalidated. `rebuild_index()` promotes an old index on request.

> **As built (2026-08-08).** All four items, plus one more site and two pre-existing sort bugs that
> only mask storage can reach.
>
> - **`_build_lex_keys`: the indicator key is not a refinement here, it is the whole of nulls-last.**
>   For a sentinel the value key already groups the nulls together and the indicator only decides
>   *where* that group goes; for a mask column the fill sorts wherever an ordinary `0` or `""` would,
>   so without the key nulls came out **first** ascending and last descending — the contract exactly
>   inverted, and silently. `valid[live_pos]` is one byte per row and needs no string comparison for
>   `U`/`S`.
> - **`_sorted_positions_from_full_index`: real, and smaller than advertised.** The three
>   partition branches (dict rank / mask / sentinel) collapsed into one shared body. Measured on 2M
>   rows: **3.0x** faster for a `U16` string column (17.1 ms → 5.7 ms) but only **1.2x** for `int64`
>   (8.5 → 6.9 ms). The 8x–64x is real as *bytes read*; both arrays compress well, so wall time
>   tracks it only where the itemsize gap is wide. Still the best line-for-line change in the phase,
>   just not by the margin the paragraph above claims.
> - **`_sorted_slice_positions` had to be found, and it bails.** Not in this section's list. The
>   window read locates the null block by bisecting the *sorted values* sidecar for the null's stored
>   value; under mask storage that value is the fill, which genuine rows share, and the validity
>   sidecar is indexed by *physical* position where the window is indexed by sorted position — so the
>   window cannot tell them apart at all. It declines for a mask column that has a sidecar and falls
>   back to the full sorted view, which is now mask-aware. A null-free mask column keeps the window
>   path: there is no null block to locate.
> - **`_utf8_rank_arrays` is the landmine this section promised, and the staleness rule matters more
>   than the fix.** The fix is three lines (`ranks[~valid] = null_rank`, after the gather — the fill
>   is a legitimate vocabulary entry other rows may share, so it has to be per row). The subtlety is
>   that an index built *before* those three lines is wrong and **neither O(1) staleness signal can
>   see it**: the column's row count and byte size are exactly what they were. `null_aware` absent
>   from the meta is what marks it, and only for a column that has a sidecar — a sentinel column's
>   nulls have always carried `null_rank`, so invalidating those would rebuild every stored utf8
>   index in the wild for nothing.
> - **Two pre-existing sort bugs, neither about nulls, both newly reachable.** The descending value
>   key is built by negating the values, which breaks on the two dtypes a *nullable* column could not
>   previously be: `bool` has no unary minus (`TypeError`, and this fires for a plain non-nullable
>   bool column today), and a narrow signed dtype wraps on its own minimum — `-(-128)` is `-128` in
>   int8 — so that row sorts as if it were the largest. A sentinel had to reserve int8's `-128` and a
>   nullable bool was physically `uint8`, which is why the combination never came up. Fixed by
>   negating in int64 for `b`/`u`/`i` alike.

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
>
> **Resolved 2026-08-10 (`plans/kleene-logic.md`).** The predicted fix is what shipped, with one
> departure: the boolean analogue **subclasses** `LazyExpr` instead of wrapping it, because
> `isinstance` checks at three consumers (and in user-shaped test code) took the wrong branch for a
> wrapper. The xfail is now a plain test. The "intentional divergence" pinned beside it did **not**
> survive: with the operator form exact, a conservative string form would have meant the two
> spellings of one predicate returning different rows, which is worse than either error alone —
> so the string rewrite carries two channels under a negation and both forms are now SQL-exact.
>
> **Addendum 2 (2026-08-08): Phase 1 only did the sentinel half, and Phase 6 shipped the gap.**
> `_rewrite_null_predicates` tested `kind_of_spec(spec) != NULL_SENTINEL` and skipped everything
> else, so a mask column got no guard at all — and the same one-kind test in
> `ctable_indexing.py:1453` left it out of `nullable_indexed`, so no post-filter either. Both were
> measured on a 2000-row table with 1 % nulls, and both leaked:
>
> - **Scan.** `t.where("a < 500")` returned **every null**, because the int fill is `0`. Same shape
>   of bug Phase 1 fixed for sentinels, read the other way round: there the stored value was a
>   sentinel that happened to satisfy the leaf, here it is a fill that does.
> - **Index.** `t.where("f > 0.5")` over a float column returned **every null** even after the scan
>   was fixed, because an ordered index answers by taking a range of the sorted column and the NaN
>   fill sorts to the end of it. This is precisely the correction above — a null-aware expression
>   cannot fix an index that never evaluates the expression — arriving a second time for a second
>   storage.
>
> Fixed by giving mask storage the same two mechanisms:
>
> - the guard is `valid_pred()` injected as a `__nv{i}` operand, since there is no in-band literal
>   to compare against, and **the fill stands in for the sentinel in `_sentinel_can_match`** — which
>   is exactly right, being what a null row actually holds. `0` cannot satisfy `a > 10` and NaN
>   cannot satisfy any ordered comparison, so those leaves stay unguarded and stay on their index;
> - a mask column with a sidecar joins `nullable_indexed`, so `_exclude_null_positions` filters it —
>   reading **one byte per candidate off the sidecar**, without touching the values at all, which is
>   the cheaper half of what masks buy. It stays out of `nullable_needs_exclude` for the same reason
>   a NaN sentinel does: that fall-back exists for the mask-direct path, which evaluates the
>   (guarded) predicate through miniexpr.
>
> Two more sites the one-kind test had hidden, both utf8, both fixed in place rather than through
> the expression layer: `_utf8_scalar_mask`/`_utf8_compare_column` compared away the sentinel string
> and so let the `""` fill through, and `utf8_span_eval` derived its per-span nulls by looking for
> the sentinel in the values — it now takes a `valids=` dict alongside `sentinels=`. A **string**
> result keeps the fill rather than a sentinel: there is nothing to write back, and a bare
> `UTF8Array` has nowhere to carry a validity channel.
>
> One drive-by fix falls out of the `partial_exact_positions` refinement block, which narrowed
> `pos` column by column while trimming the prefetched primary values only for the primary column's
> own filter: with two nullable indexed columns and nulls in the non-primary one, `prefetched` came
> out **misaligned with `candidates`**. Rewritten as one combined keep-mask applied once, which is
> both correct and shorter.
>
> Verified against a NumPy SQL oracle over 32 combinations — {mask, sentinel} × {indexed,
> unindexed} × 8 expression shapes including `|` and `~` — all agreeing exactly.
>
> **Addendum 3 (2026-08-08, Phase 10): the indexed-OR bail is lifted for the path that evaluates,
> and kept for the paths that do not.** The correction above is right that a null-aware expression
> cannot fix an index which never evaluates it — but it over-generalized from there to "OR over a
> nullable indexed column must fall back to the scan". Only *some* index paths answer without
> evaluating. The segment/candidate-unit path prunes blocks by their summaries and then runs the
> predicate through miniexpr over the survivors, so its result is exact and
> `_exclude_null_positions` there was not merely wrong for OR — it was unnecessary. That path now
> serves OR with the filter skipped. The exact-position paths (FULL/PARTIAL/BUCKET), which answer
> by slicing the sorted column, still bail.
>
> Two details:
>
> - **Those three bails are unreachable today, and are kept as a guard.** `_plan_exact_conjunction`
>   declines any expression containing an OR, so an exact plan cannot coexist with one. They exist
>   so a planner that learns to build one cannot silently reintroduce the bug.
> - **The OR test now parses.** `"|" in expression` also fires on a string literal that contains
>   one (`name == 'a|b'`), which took the index away from a query with no OR in it at all;
>   `_expression_has_or` walks the AST for `ast.Or`/`ast.BitOr` instead.
>
> Measured on a 20M-row two-column probe, `(a > 4000) | (b > 6000)` with SUMMARY indexes on both:
> **1.61x** (12.35 ms → 7.65 ms). Modest, and honestly so — the miniexpr scan it was falling back
> to is already fast; the win is proportional to how much the summaries prune.

### Groupby

`groupby.py:_null_mask` (`:2116-2137`) is already the single central helper, but it receives an
already-gathered `values` chunk, so validity has to be gathered at each of ~8 independent read
sites: `:534`, `:583-589`, `:624-637`, `:716-718`, `:1005-1017`, `:1394`, `:1430`, `:1757`, plus
`_null_value_for` (`:2589`) and `_null_output_value` (`:2239`), which for a mask-backed output
column become "write fill + mask=False". Signature grows `*, valid=None`. **Messiest integration
after `__setitem__`** — budget accordingly.

One semantic improvement follows from decision 6: NaN in a float *value* column is no longer
missing. Keep the `is_key` NaN coercion at `:2131-2134` for keys (dropna semantics).

> **As built (2026-08-08).** It was not the messiest integration — that was the *key* side, which
> this section does not mention at all.
>
> - **A mask key column needs recoding, not a threaded flag.** Threading `valid=` fixes value
>   columns, and that part is as described (`_null_mask` grows the kwarg; the generic path and the
>   dense single-key path gather validity by the same `live_mask` as the values). But a *key* column
>   has a second problem no validity flag solves: with `dropna=False` the null rows have to form a
>   group of their own, and their fill is a value a genuine row may hold, so they would merge into
>   the `0` group of an int key or the `""` group of a string one. The fix is to give them a reserved
>   code — exactly what a dictionary column gets for free from `null_code`. `_Utf8KeyChunk` was
>   already the right shape for that, so it became `_CodedKeyChunk` with a `null_code` field, and
>   `_coded_chunk_with_nulls` recodes any mask key chunk into one. `uniques[null_code] is None`, so
>   the null group comes out keyed **`None`** and a mask-storage output column writes it back as a
>   real null — where a sentinel column can only offer its sentinel (`group_by(dropna=False)` over
>   one returns a group keyed `-1`).
> - **`_null_output_value` returns `None` for a mask output spec**, which is the same point one layer
>   up: a group with no non-null input no longer has to come back as `0` and hope nobody reads it as
>   data. Note `sum`'s output spec is a fresh non-nullable `float64`/`int64`, so *that* aggregate
>   still spells missing as `NaN` — unchanged, and the same for both storages.
> - **The fast paths bail, but not all of them, and the difference is 4.5x.** Every path reads
>   nullity out of the values; a Cython kernel is handed a `skip_nan` flag, not a validity array, so
>   the four Cython paths defer to the generic path whenever a mask column in play holds a null. The
>   dense single-int-key path is plain NumPy and already routes value columns through `_null_mask`,
>   so it only needed the sidecar — and keeping it matters: 2M-row `sum` grouped by an int key went
>   91.8 ms → 20.4 ms once it stayed, against 10.8 ms for the sentinel equivalent. A mask *key*
>   column still has to leave it, since a `_CodedKeyChunk` is not the array of dense non-negative
>   ints that path indexes with. Threading validity into the kernels is the named follow-up.
> - **One pre-existing bug, storage-independent.** `min`/`max` seed a per-group accumulator with the
>   dtype's opposite identity, and `_max_identity`/`_min_identity` had no `bool` case — so
>   `np.full(n, None, dtype=bool)` gave `False`, a min accumulator could never rise above it, and
>   every all-`True` group reduced to `False`. Reachable today with a plain non-nullable bool column
>   on any generic-path aggregation (a string key is enough); a nullable one was `uint8`, whose
>   identities are fine, which is why mask storage is what surfaced it.

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

> **Correction and as-built (2026-08-08).** The ordering above is **wrong at step 2, and the fix is
> to move it rather than to accept the window.** After (2) the table is *not* intact: the null slots
> now hold the fill, and a schema still saying `sentinel` reads a fill `0` as the value `0`. What
> shipped is **(1) sidecar, (2) schema, (3) fill** — and every intermediate state is then correct,
> not merely recoverable. A crash before (2) leaves an orphan `.notnull` key that a sentinel column
> never opens; a crash before (3) leaves a correct mask column whose null slots happen to still hold
> the old sentinel, which is unobservable through the `Column` API and which decision 5 explicitly
> excludes from the format contract. `to="sentinel"` runs the same argument backwards: sentinel into
> the null slots first (harmless while the sidecar is still authoritative), then the schema, then
> drop the sidecar. Both orderings are asserted in `test_null_migration.py`.
>
> Four further departures:
>
> - **A dtype change is refused for a persistent in-place conversion**, and this is the one real
>   capability limit. `bool` (`uint8` ↔ `np.bool_`) and a `string`/`bytes` column too narrow for its
>   sentinel need the stored array *replaced*, and there is no ordering of that write and the schema
>   update a crash cannot land between. `inplace=False` has no such window — it builds a new
>   table — so those columns raise, naming themselves and pointing at it. In-memory `inplace=True` is
>   allowed: there is nothing to crash into. The check runs in `_convert_null_targets`, before any
>   write, alongside the sentinel-availability checks, so a refusal never leaves litter.
> - **`copy()` shares its schema object with the source**, which conversion — the one operation that
>   mutates a spec in place — cannot live with: a converted copy relabelled its *source's* columns
>   too, leaving that table reporting a storage its data does not use. `_detach_schema()` deep-copies
>   the specs first, and `_convert_nulls_inplace` always calls it, so `copy()` followed by an
>   in-place conversion is safe as well.
> - **The all-elements ndarray rule needs no special handling.** `sentinel_mask(item_ndim=)` already
>   implements it, and the sentinel direction writes the scalar sentinel across every element of a
>   null row, which is the same rule read backwards.
> - **`Column.null_storage` and the `info()` rows already existed**, from Phases 0 and 4. What Phase
>   8 added is the *table-level* tag (`int64 nullable[mask]` in `info`'s per-column summary), which is
>   what you actually read to decide whether a table needs converting.
>
> Also fixed here, storage-independent and pre-existing: `copy()` recorded its write watermark as
> `n - 1` where `_resolve_last_pos()` and every other writer mean an exclusive bound, so
> `add_column()` on a copied table backfilled one row short — and *raised* for a variable-length
> column. And `_unflip_mask_bool_dtype` keyed off the dtype rather than off whether the flip had
> happened, so a nullable **`uint8` ndarray** column under mask storage came back as `bool_`, every
> byte truncated to a flag; the specs now record `bool_widened_to_uint8`.

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
- `tests/ctable/test_null_aware_indexes.py` (Phase 10): the summaries at unit level (extrema over
  the valid rows, `FLAG_ALL_NULL`, `FLAG_HAS_NAN` marking a value and not a fill); the `null_aware`
  claim, including the null-free mask column that earns it without a sidecar and the older index
  that must not be trusted with it; `min`/`max` against the scan for every V1 kind and both
  storages, plus full-range `int8`, the straddling tail block, an all-null column, and a null
  written *after* the build; and indexed `OR`, asserted to use the index rather than only to be
  right.

End-to-end smoke, run manually: import a nullable-bool + full-range-`int8` + free-text-utf8 Parquet
file, round-trip it, and assert `pq.read_table(out).equals(pq.read_table(in))` — the case that is
impossible today. Also re-run the OFF importer round-trip (`plans/ctable-nulls.md` §Tests) and
confirm the `nullable_scalar_wrapped_as_singleton_list` workaround can be deleted.

> **Correction (2026-08-08).** The smoke test is now a real test rather than a manual one —
> `test_parquet_round_trip_is_lossless`, parametrized over the whole contract list.
>
> The `nullable_scalar_wrapped_as_singleton_list` workaround **cannot be deleted, and does not need
> to be.** `src/blosc2/cli/parquet_to_blosc2.py` stopped *producing* it before this work began — the
> conversion table at `:410-496` emits `nullable_scalar_sentinel` for that case now. The two
> surviving references (`:1405-1406`) are in the *export* path, which reads the tag out of archive
> metadata to unwrap what an older version wrote. That is a permanent backward-compatibility reader,
> in the same category as `_is_nullable_bool` and its rewrite sites: it stays forever.

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
| 5 | ✅ **Expressions + reductions.** `_ndarray_values_for_reduction`, argmin/argmax (both were reducing over the *fill*), `_reduction_null_mask` as the one storage-agnostic entry point. `_raw_null_pred`/`_lazy_nonnull_mask`/`_is_nullable_bool` needed nothing — Phases 0–4 had already made them storage-agnostic. **The ndarray-propagation gain is not real and was not done**, and the free summary fast path is unsound; both corrections are above. `tests/ctable/test_null_mask_expressions.py` (39 tests). | S (was M) | Low (was Med) |
| 6 | ✅ **Arrow/Parquet.** Import + export for all V1 kinds; `arrow_slice(valid=)`; `null_storage=` on `from_arrow`/`from_parquet`; the "no sentinel available" import error now names the way out instead of being deleted (it still fires for sentinel storage, which still cannot represent those types). No `packbits` — pyarrow's own packing is borrowed instead. Ships **opt-in**; the default stays `"sentinel"`. `tests/ctable/test_null_mask_arrow.py` (42 tests). | M | Med |
| 7 | ✅ **Sort + groupby.** `_build_lex_keys` (the indicator key is nulls-last *entirely*, not a refinement), `_sorted_positions_from_full_index` (3.0x for `U16`, 1.2x for `int64` — the I/O win is in bytes, not proportionally in time), `_utf8_rank_arrays(valid=)` plus a `null_aware` staleness rule no O(1) signal could replace, `_sorted_slice_positions` bails, groupby `_null_mask(valid=)` **plus `_CodedKeyChunk`**, which this section had not anticipated: a mask *key* column needs a reserved null code, not a threaded flag. **Plus the mask half of Phase 1**, which had never been done — `where()` leaked nulls on both the scan and the index (see §Expression layer, Addendum 2). Three pre-existing storage-independent bugs fixed on the way: descending sort of `bool` (raised) and of full-range signed ints (wrong order), and groupby `min`/`max` over `bool` (always `False`). `tests/ctable/test_null_mask_sort_groupby.py` (75 tests). | **L** (was M) | Med-High |
| 8 | ✅ **Migration + docs.** `convert_nulls` both directions for every V1 kind, refusing what a sentinel cannot represent; the crash ordering **corrected** (fill after the schema flip, not before) and asserted; a persistent in-place dtype change refused with a reason; `_detach_schema` so a converted copy stops relabelling its source; the table-level `info` null tag (`Column.null_storage` and the per-column `info` rows already existed). `doc/reference/ctable.rst` gains a "Where nulls are stored" section and a rewritten null-policy resolution order; release notes. Three storage-independent bugs fixed on the way: `copy()`'s off-by-one write watermark (which *raised* in `add_column`), and a nullable `uint8` ndarray column coming back as `bool_`. `tests/ctable/test_null_migration.py` (50 tests). | M (was S–M) | Low |
| 9 | ✅ **Default flips to `"mask"`.** Not a one-line change: `null_storage` had to become tri-state (`None` = unspecified) so that a type-wide sentinel field can still imply sentinel storage without contradicting the new default, and ~65 tests that wrote a sentinel literally to mean "null" had to say `null_storage="sentinel"` and mean it. Three real gaps the flip exposed, all fixed: **CSV import/export was sentinel-only** (`from_csv` raised on an empty field, `to_csv` wrote the fill as data), **`~` on a mask bool column selected its nulls**, and **the Arrow importer ignored the type-wide sentinel inference**, so `NullPolicy(signed_int_strategy="max")` meant one thing for a declared schema and another for an inferred one. Landed in the **same** session as Phase 6, not a release later — see the note below. | M (was S) | Med (was Low) |
| 10 | ✅ **Index null-awareness remainder** *(independent)*. Summary builder takes a per-column validity provider (one callable, both storages); `FLAG_ALL_NULL` riding on `FLAG_ALL_NAN`; `null_aware` in the descriptor, **`null_order` deliberately not recorded** and **no token bump** — see the corrections above. `_summary_minmax_source` re-enabled for mask *and* sentinel columns (**236.89x** on a 20M-row `int64` `min()`), with `FLAG_HAS_NAN` over the valid rows narrowing Phase 5's whole-column bail down to the one genuine-NaN case it was really about. Plus the **indexed-OR lift**, which this row did not anticipate: the segment path evaluates the predicate, so it never needed the global post-filter that forced the bail (1.61x on a 20M-row probe). One cost, recorded: an index over a nullable column holding nulls can no longer use the incremental write-time summaries and pays a decompression pass at `close()`. `tests/ctable/test_null_aware_indexes.py` (22 tests). | M (was L) | Med (was High) |

The riskiest, most-coupled work is isolated into Phases 4, 7 and 10, each of which can slip
without blocking the others. Phase 9 is a policy change, not code — its only prerequisite is
that Phases 2–8 have soaked for a release.

Phase 10 came in smaller than budgeted for the same reason Phase 5 did: Phases 0–7 had already
made every consumer read nullity through one channel, so the remaining work was to *supply* that
channel to one more builder rather than to teach a new subsystem about nulls.

> **Deviation from decision 1 (2026-08-08), recorded deliberately.** Phase 9 landed in the same
> session as Phase 6, not a release later. The staging existed so that version-3-capable readers
> would be in circulation before default-created tables required them; shipping both at once means
> the first release carrying the flip is also the first release able to read what it writes. The
> mitigation is unchanged and was always the real safety net: the version bump is **conditional on a
> mask column existing**, so an older reader meets a clear `ValueError: Unsupported schema version 3`
> rather than misreading anything, and `null_storage="sentinel"` remains one keyword away for data
> that has to stay readable by them. Flagged to the maintainer at the time as a release-scheduling
> call rather than a technical one.

## Named follow-ups (not blocking any phase)

- **Validity through the Cython groupby kernels.** A mask column holding a null costs ~1.9x on a
  2M-row grouped `sum` against the sentinel equivalent, because the four Cython paths bail; the
  kernels take a `skip_nan` flag where they would need a `values_valid` array. Two of them
  (`groupby_hash_i64x2_f64`, `groupby_dense_int_count_checked`) already accept one, so this is
  partly a matter of using what is there.
- **A mask *key* column back on the dense single-key path.** It leaves because
  `_CodedKeyChunk.codes` are chunk-local, not the dense global ints that path indexes with.
- **`__setitem__`'s fast paths** for mask columns (deferred in Phase 4, still unmeasured).
- ~~**Kleene three-valued logic**, decision 8 — which now also owns the operator-form negation leak
  pinned as a `strict=True` xfail in `tests/ctable/test_null_predicate_rewrite.py`.~~
  **Done 2026-08-10**, `plans/kleene-logic.md`: the xfail is a passing test, the string form's
  conservative negation became exact, and a dictionary `!=` stopped returning its nulls. Cost of
  the exact answer, measured: 1.15x on a negated two-column conjunction over 20M rows; every other
  predicate shape unchanged.
- **Validity through `_ColumnSummaryAccumulator`** (Phase 10). The per-block summaries folded
  during writes carry no validity, so a nullable column holding a null cannot use them and pays a
  decompression pass at `close()` — 1.8 ms → 33.2 ms for a 20M-row `int64`. `extend`'s feed site
  already has the validity in hand (`batch_valid`); the Arrow writer's `on_write=` callback does
  not, so the accumulator would need a "nullable columns must be fed validity or invalidate" rule
  and that path would keep today's cost.
- **A null-run FULL index**, which is what would earn a truthful `"null_order": "last"`. Sorting
  nulls last *within ties* makes them a contiguous range of the sorted array, so a range query
  could subtract it in sorted space — exact, per leaf, no I/O. That would let
  `_exclude_null_positions` go away entirely, put `OR` on the exact-position paths, and give
  `_sorted_slice_positions` (which bails since Phase 7) its window read back. It needs the sort key
  and the external-merge builder to carry validity, in every index kind.
