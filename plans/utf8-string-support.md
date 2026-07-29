# Compute parity for utf8 strings

> ## Outcome: **G2, G3 and G4 withdrawn; G5 shipped.** Not superseded — decided against.
>
> The plan was written before the conversion pair existed. What shipped instead is the opposite
> rule: **utf8 stores and filters; fixed-width computes** (`f8af0714`, `5b31abe4`), with every
> compute-side refusal printing the two-line recipe that fixes it (`8e3868ba`). See
> `plans/string-flavours-assessment.md`, which measured the flavours end to end and is the
> document of record.
>
> | | verdict | why |
> |---|---|---|
> | G1 | moot | subsumed by G2, which is withdrawn |
> | **G2** computed columns | **withdrawn** | see below |
> | **G3** DSL kernels / `apply()` | **withdrawn** | same output-container problem, same recipe covers it |
> | **G4** bare `UTF8Array` | **withdrawn**, minus the two real bugs | `__eq__` fixed in `3692673f`; the wrong-path `lazyexpr` fixed in `0b486b07`. The remaining "lift the driver" work has no asked-for use case |
> | **G5** nested leaves | **shipped**, as a *query* fix | not a compute gap at all — see below |
>
> **Why G2 and G3 are withdrawn.** They would buy an API that looks identical to `<U` and runs
> 3–5× slower (the span driver's decode + `astype` per span is unavoidable), paid for with a
> serialization hazard whose failure mode is an **unopenable table**: `_schema_dict_with_computed`
> saves `str(dtype)` and `np.dtype("StringDType()")` raises on load. The published rule is both
> cheaper and more honest — the `.astype()` the user writes *is* what the driver would have done
> silently. Reopen only on a concrete user request for utf8-typed computed columns; the ~1 week
> estimate below still stands, and the `"utf8"` dtype sentinel in §G2 is still the way to do it.
>
> **Why G5 was not withdrawn with them.** It is filed here under compute, but a nested utf8 leaf
> could not be *filtered* either — `t.where("trip.name == 'x'")` raised, while the same query on a
> `<Un`, `bytes()` or `dictionary()` leaf worked. That made "utf8 stores and filters" false for
> nested columns, i.e. a hole in the rule the other four gaps were withdrawn in favour of. Fixed
> by aliasing dotted utf8 names in `_lazyexpr_over_cols`; the diagnosis in §G5 below was wrong
> about the mechanism (see the note there).

Give `utf8()` columns (and `Utf8Array`) the same computing surface `<U` columns have, accepting
that they will be slower. Follow-on to `plans/string-support.md`, whose Phases 1–5 built the
machinery this plan only needs to *call*.

## Verdict: feasible, and **no miniexpr / C work at all**

Phase 3's span driver already does the hard part: `CTable._utf8_span_eval()` (`ctable.py:12836`)
walks a utf8 column in row spans, materializes each span to a fixed-width `<Un` array, hands it to
miniexpr, and — since blosc2 `9cb490ac` — reassembles a **`Utf8Array`** when the result is a string.
Width bucketing, the `_UTF8_EXPR_BUDGET` byte ceiling, the null-sentinel policy (§3c) and the
contagion rule are all in place and tested.

What is missing is *callers*. Every gap below is Python plumbing in `ctable.py`.

## Measured starting point

Probed on a live table with a `<U` column and a `utf8` column holding the same values:

| surface | `<U` | `utf8` |
|---|---|---|
| `where("startswith(c,'h')")` | ok | **ok** |
| `sum(where=...)` | ok | **ok** |
| `where("upper(c) == 'HELLO'")` | ok | **ok** |
| `add_computed_column("'x=' + c")` | ok | `NotImplementedError` |
| `assign(new="'x=' + c")` | ok | `NotImplementedError` |
| `lazyudf(dsl_kernel, (col,))` / `t.apply()` | ok | `ValueError: malformed node ... StringDType()` |
| `blosc2.lazyexpr` over the bare array | ok | "works", wrong path — see G4 |
| `group_by` on the column | ok | ok (factorizer, `groupby.py:1565`) |

So predicates and boolean expressions already have parity. **String-returning expressions and DSL
kernels do not**, and that is the whole of the work.

---

## G1 — string-returning utf8 expressions have no public caller

`_utf8_span_eval()` returns a `Utf8Array` for a string result, but its only two callers are
`CTable.where()` (`ctable.py:13026`) and `Column.sum(where=)` (`ctable.py:2601`), both of which want
a **boolean**. Nothing user-facing can ask for `"x=" + name` over a utf8 column; the tests reach it
by calling the private method (`tests/ctable/test_utf8.py:1386`).

Not a separate job — G2 is its consumer. Listed so the state is not mistaken for "unbuilt".

**Effort: 0** (subsumed).

---

## G2 — computed columns over utf8  ← the bulk of the work

`_normalize_expression_transformer` (`ctable.py:9777`) calls `_guard_scalar_expression(expr)` at
**`ctable.py:9787`** without `allow_utf8=True`, so `add_computed_column()` (`10409`) and
`assign()` (`10631`) both raise.

Behind that guard:

- A `kind: "utf8_expression"` descriptor threaded through `_normalize_transformer` (`9865`) →
  `add_computed_column` → `_build_computed_lazy` (`9937`) → `_schema_dict_with_computed` (`9216`) →
  `_load_computed_cols_from_schema` (`9445`).
- **The "needs a `LazyExpr`, not a materialized value" objection is already answered in the file.**
  `_build_computed_lazy` *eagerly materializes* its `kind == "dsl"` branch — `lazyudf(...).compute()`
  on every access — because the miniexpr DSL path has no partial-slice getitem. A utf8 entry follows
  that precedent: run the span driver, return the `Utf8Array`. Consumers slice it (`lazy[int]`,
  `lazy[a:b]`), which `Utf8Array` supports.
- `materialize_computed_column` (`9703`) — target spec must be `utf8()`, not a `<Un` string spec.
- `_readable_computed_expr` (`9155`), and `where()` over such a column.

### Persistence is the trap, and it is confirmed

`_schema_dict_with_computed` writes `str(cc["dtype"])` (`9233`, `9245`) and
`_load_computed_cols_from_schema` reads it back with `np.dtype(...)`. For a utf8 result the dtype is
`np.dtypes.StringDType()` and `np.dtype("StringDType()")` **raises** — the column would save fine
and then make the table **unopenable**.

This is not hypothetical: the same round-trip is what blows up G3 today. `NDArray.dtype` does
`np.dtype(ast.literal_eval(str_dtype))` (`blosc2_ext.pyx:3818`), which is the exact
`ValueError: malformed node or string ... StringDType()` the probe hit.

Fix: serialize a `"utf8"` sentinel and map it back on load. Then audit the sites that consume
`cc["dtype"]` / index a computed lazy: `ctable.py:5229`, `9198`, `9202`, `9679`, `11287`.

**Effort: 3–4 days**, most of it the save/reopen round-trip tests rather than the descriptor.

---

## G3 — DSL kernels, `lazyudf` and `t.apply()` over utf8

Fails today because `lazyudf` tries to allocate an **NDArray output with `dtype=StringDType()`**
(`ndarray.py:5796` → `blosc2_ext.pyx:3818`). The container for a variable-width string result must
be a `Utf8Array`, which is what the span driver already builds.

`_utf8_span_eval` hardcodes `blosc2.lazyexpr(expr, span)` at **`ctable.py:12880`**. Generalize it to
take either an expression string or a `DSLKernel`, calling `lazyudf(kernel, span_operands).compute()`
in the latter case. The reverted `compute_varlen()` (blosc2 `f6b06438`, reverted `4e364900`) already
proved that shape runs; only the varlen *output* half of it was the part that did not pay.

Then route: `CTable.apply()` (`10060`) and the `kind == "dsl"` branch of `add_computed_column`.

**Effort: 1–2 days.**

---

## G4 — standalone `Utf8Array`, outside any CTable

`blosc2.lazyexpr("'x=' + a", {"a": utf8_arr}).compute()` returns *correct values* today — and takes
the wrong path entirely:

- it materializes the whole column through NumPy, ignoring `_UTF8_EXPR_BUDGET`;
- it returns a fixed-width `<U22`, not a `Utf8Array` — the contagion rule does not apply;
- it never reaches miniexpr: `compute(strict_miniexpr=True)` is rejected as an unknown kwarg, i.e.
  this is the `slices_eval` numpy fallback. **This is the same trap §Phase 1 and §Phase 3 both
  document** — correct-looking results from a path that was never the intended one.

Separately, **`utf8_arr == "hello"` returns plain Python `False`** — `Utf8Array` defines no
comparison operators, so this is object identity, silently wrong. Independent of everything else
here and worth fixing on its own.

Fix: lift `_utf8_span_eval` to a module-level function over `{name: Utf8Array}`. It is nearly there
already — it touches `self._cols`, `self._valid_rows` and the per-column sentinel, and nothing else
table-shaped.

**Effort: 2–3 days** for the lift; **~1 hour** for `__eq__` (make it correct, or make it raise).

**Question this one before building it.** `Utf8Array`'s documented route is a CTable column or
`blosc2.utf8_array(...)`; if nobody asks for bare-array expressions, G2 + G3 already deliver parity
where users actually are. The `__eq__` bug should be fixed regardless.

---

## G5 — nested (dotted) utf8 leaves — **shipped**

`_lazyexpr_over_cols` raises for them (`ctable.py:12825-12830`): `_rewrite_nested_expression`
aliases a dotted name away before the driver could find it, so `_utf8_names_in` rejects dotted names
explicitly. Make the driver alias-aware (carry the alias→original map through the rewrite).

**Effort: ~0.5–1 day.**

**Done.** The diagnosis above was wrong, and carrying the alias map through changed nothing:
`_rewrite_nested_expression` only rewrites names it finds in `operands`, and utf8 columns are
*excluded* from the operand namespace (`_where_expression_operands`), because a variable-length
column cannot be an expression operand. So a dotted utf8 leaf never reached that rewrite at all and
arrived at the span driver still spelled with dots, which `blosc2.lazyexpr` rejects as an
identifier. The fix is to alias them in `_lazyexpr_over_cols` itself, sharing one `_alias_dotted`
helper with the nested rewrite; `_rewrite_utf8_predicates` and `_utf8_span_eval` take the
`alias -> column` map so they can still reach storage and null sentinels.

This is a **query** fix, not a compute one — hence shipping while G2/G3 are withdrawn. Scalar
comparisons, `startswith`/`upper`, mixed numeric predicates and `sum(where=)` all work on a dotted
utf8 leaf now, and the answers match the same data in a flat column.

---

## Not gaps

- **miniexpr.** Nothing. utf8 reaches it as fixed-width `<Un` spans, which Phase 1 handles.
- **Native `ME_UTF8`.** Phase 4's gate measured it and said no; it is a *performance* item and this
  plan explicitly accepts slower. Do not revisit it here.
- **Varlen output (`me_eval_varlen`).** Phase 5 measured it: bigger and slower than the compressed
  fixed-width result in blosc2. Irrelevant to parity.
- **Path-sensitive `null_out` (§3c).** utf8 nulls are a *sentinel string*, so there is no separate
  nullity to propagate — the two-line span-level policy already in `_utf8_span_eval` covers it.
- **`group_by`, predicates, `where`, `sum(where=)`.** Already at parity.

---

## Sequencing and total

**G2 → G3 → G5 → (G4, if wanted).** G2 first because it is G1's consumer and because it forces the
dtype-serialization decision every later piece inherits.

| | effort |
|---|---|
| G2 computed columns | 3–4 d |
| G3 DSL kernels / `apply()` | 1–2 d |
| G5 nested leaves | 0.5–1 d |
| G4 standalone `Utf8Array` | 2–3 d (optional) + 1 h for `__eq__` |
| **total** | **~1 week for CTable parity, ~1.5–2 weeks with G4** |

## What to expect on performance

The span driver pays a decode plus an `astype` per span. From `plans/string-support.md` §Phase 4, at
1 M rows of `<U32`-class values: decode 27 ms, `astype(<U32)` 116 ms, feeding miniexpr 46 ms, against
20 ms of actual evaluation. So budget **~3–5× the `<U` cost** for string-returning work. Scalar
comparisons are unaffected — `_rewrite_utf8_predicates` answers them with a raw-byte scan, already
at operator-form speed.

## Verification

Tests must pin the **route**, not just the values — the `strict_miniexpr` trap from Phase 1 and the
`slices_eval` trap from Phase 3 both produced correct answers down the wrong path, and G4 is a live
instance of it today.

```bash
cd /Users/faltet/blosc/python-blosc2
pytest tests/ctable/test_utf8.py -q
pytest tests/ctable/test_ctable_computed_cols.py -q
```

Per gap, the assertions that actually matter:

- **G2** — a save / reopen round trip. That is where the `np.dtype("StringDType()")` failure shows
  up; a same-session test passes without it.
- **G2/G3** — the computed column's values are byte-identical to the same expression over the
  equivalent `<U` column, and its container is a `Utf8Array` (contagion), not a `<Un` NDArray.
- **G3** — `strict_miniexpr=True` inside the span, so a numpy fallback fails the test.
- **G4** — `compute(strict_miniexpr=True)` on a bare `Utf8Array` must be accepted *and* pass; a span
  larger than `_UTF8_EXPR_BUDGET` must still split.
