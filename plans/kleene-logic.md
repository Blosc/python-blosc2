# Three-valued (Kleene) logic for CTable predicates

> **Status: complete — landed 2026-08-10.** Both query forms reason in three values,
> the `strict=True` xfail this work inherited is now a passing test, and the whole suite
> (9217 tests) is green. Three premises from the design below were disproven while building it
> and are corrected in place, each beside the text it corrects: the result type must **subclass**
> `LazyExpr` rather than wrap it (§The type), the dictionary `!=` bug was not in scope and turned
> out to be the same bug (§Producers), and the string form's "conservative is good enough" ruling
> from `plans/mask-based-nulls.md` does not survive contact with an exact operator form
> (§The string form). Drafted and implemented 2026-08-10.
>
> This discharges **decision 8** of `plans/mask-based-nulls.md` ("Kleene three-valued logic stays
> out of scope … named deferred follow-up") and the follow-up entry that inherited the
> operator-form negation leak. Read that document for how nulls are *stored*; read this one for
> what a predicate *means* when it meets one.

## Context

Every phase of the mask-nulls work collapsed a null comparison to `False` at the leaf —
`_null_aware_compare` returns `raw & ~null_pred`, `rewrite_null_predicates` conjoins
`& (a != 999)` onto each leaf. That is exactly right for `WHERE`, which keeps what a predicate
is *true* for, and it is why every consumer could stay two-valued for ten phases.

It is wrong for anything that reads the result as a boolean *value* rather than as a filter:

- **`~` inverts the collapse.** `~(t.a > 10)` returned every null row — the precise opposite of
  the intent. Pinned since Phase 1 as a `strict=True` xfail
  (`test_operator_form_negation_drops_nulls`).
- **The string form's fix for the same problem is not exact.** Pushing validity to the negation
  point — `(~(a > 10)) & valid_a` — drops rows SQL keeps: `not (a > 10 and b == 999)` with a false
  second term is `not FALSE` = true, because `NULL AND FALSE` collapses to `FALSE`. Pinned as
  `test_negation_over_and_corner_is_conservative`, deliberately, with the note that the two forms
  disagreed here.
- **A dictionary `!=` returned its nulls.** The reserved code differs from every value's code, so
  `codes != target` was true for a null row. Same bug, different subsystem — found while auditing
  the producers, not anticipated.

The missing piece is not more guards. It is the third value itself: `unknown & false` is **false**,
not unknown, and no conjunction of validity predicates can say that. That is Kleene's logic, and it
needs the null channel to survive as far as the operator that consumes it.

## Decisions

1. **The truth channel stays collapsed.** A three-valued predicate carries `(true, unknown)` where
   `true` is already "definitely true" (disjoint from `unknown`), not a raw comparison plus a flag.
   Every existing consumer reads the truth channel and gets exactly today's answer; only the four
   boolean operators consult the second one. This is what keeps the diff at the edges of the system
   rather than through it.
2. **`WHERE` semantics do not change.** SQL keeps what is true, so an unknown row is dropped like a
   false one. The rows returned for a plain comparison are byte-for-byte what they were.
3. **Both query forms, one semantics.** The string form is rewritten to the same rules rather than
   left conservative — see §The string form for why "it only ever drops rows" stopped being an
   acceptable divergence once the operator form was exact.
4. **`isin()` stays two-valued.** It returns a materialized array, is documented as such, and has
   its own spelling for nulls (`None` among the values). Making it three-valued would change a
   public return type to fix a question it already answers.
5. **Nothing is paid by a predicate that meets no null.** A comparison over a non-nullable column
   returns a plain `LazyExpr`, as before; the machinery starts only where `_combined_null_pred`
   already returned something.

## The type

`NullableBoolExpr` (`ctable.py`, beside `NullableExpr`) holds `_true`, `_null` and its table.

> **Correction (2026-08-10): it must *subclass* `LazyExpr`, not wrap it.** The first
> implementation was a wrapper with `__getattr__` forwarding, in the shape `NullableExpr` already
> uses. It fails on `isinstance`: `CTable.where` type-checks its argument, `_try_index_where` wants
> `.expression`/`.operands`, and a test helper in `test_dictionary_column.py` — plain user-shaped
> code — does `mask.compute() if isinstance(mask, blosc2.LazyExpr) else mask` and silently took
> the wrong branch. Subclassing fixes all of them at once, and buys two things the wrapper could
> not:
>
> - **Reflected operators come for free.** Python offers `__rand__`/`__ror__` to the right operand
>   first when its type is a subclass of the left's, so `plain_expr & three_valued` reaches the
>   Kleene operator instead of collapsing. The wrapper needed `blosc2.Operand` to defer explicitly.
> - **No forwarding surface to keep in sync.** An instance *adopts* the truth channel's state
>   (`self.__dict__.update(true_expr.__dict__)`), so it does not merely behave like the collapsed
>   predicate — it is it.
>
> `blosc2.Operand.__and__`/`__or__`/`__xor__` still defer explicitly (`_defers_boolean_op`, keyed
> on `_kleene_channels`), because an `NDArray` on the left is not a superclass of anything here.
> That also picks up a `Column` on the right, which was already a latent asymmetry.

The rules, with `None` meaning "this operand is two-valued":

| op | true | unknown |
|---|---|---|
| `a & b` | `ta & tb` | `(na & (tb\|nb)) \| (nb & (ta\|na))` |
| `a \| b` | `ta \| tb` | `(na & ~tb) \| (nb & ~ta)` |
| `a ^ b` | `(ta ^ tb) & ~null` | `na \| nb` |
| `~a` | `~ta & ~na` | `na` |

Public surface on the result: `is_null()`, `notnull()`, `null_count()`, `fillna(bool)`.
`fillna(False)` is what `where()` does implicitly; `fillna(True)` is "keep what I cannot rule out".

## Producers

`Column._null_aware_compare` is the main one and its change is one line — the collapsed predicate
it already built becomes the truth channel. Four more had to be found:

- **`Column._kleene_channels`** — a nullable *bool column* is itself a three-valued predicate, with
  its third value in band for sentinel storage (`255`) and in the sidecar for mask storage. The
  boolean operators on `Column` route through it, so `t.flag & t.other` and `~t.flag` are Kleene.
  Non-boolean columns are tested for **before** the null channel is asked, so `t.flags & 0x04`
  builds no predicate it will discard.
- **`NullableExpr` comparisons** — `(t.a + 1) > 5` carries the arithmetic's NaN-null channel into
  the comparison result. `__ne__` already had the collapse; the other five gained it.
- **utf8** — `_utf8_compare_scalar`/`_utf8_compare_column`. These are eager physical-length arrays,
  and neither a `StringDType` comparison nor a raw-byte span scan is something the lazy layer
  evaluates, so the null channel costs a scan of its own. It is therefore **deferred**: the channel
  may be a zero-argument callable, resolved on first use, so an ordinary `==` never pays for it and
  only `~`/`is_null()` does.
- **dictionary** — `_dictionary_eq`.

> **Correction (2026-08-10): the dictionary case is a bug fix, not plumbing.** `!=` compared codes,
> and a null row's reserved code differs from the target's like any other, so `t.where(t.d != "x")`
> **returned every null**. The three-valued result fixes it (`exclude_nulls=` on the wrap, True for
> every negated form and False for a positive one, where no null row can carry the code being
> looked for anyway). `d == None` stays two-valued: it is the is-null test, not a comparison
> against a value, and making it unknown-everywhere would have broken `Column.is_null`, which is
> built on it.

## The string form

`_NullPredicateRewriter` grows `_channels(node)`, returning `(true_ast, null_ast)` for a subtree,
and uses it **only under a negation** — so a non-negated expression is rewritten to exactly the
text it was before, and the index planner sees the shapes it already knows.

> **Correction (2026-08-10) to `plans/mask-based-nulls.md` §Expression layer.** That document
> settled for the conservative negation guard on the grounds that "rows are only ever dropped,
> never wrongly returned". That was a defensible ruling while the operator form was *also* wrong,
> in the other direction. With the operator form exact, leaving the string form conservative would
> mean `t.where("...")` and `t.where(t...)` returning different rows for the same predicate — a
> worse property than either error alone. Both are exact now, and the test that pinned the
> divergence (`test_negation_over_and_corner_is_conservative`) is rewritten in place as
> `test_negation_over_and_corner_agrees_with_sql`, asserting the two forms agree.

Two details:

- **A negated *leaf* keeps its short form.** `~(leaf & guard) & guard` is `~leaf & guard`, so the
  common shape is special-cased and emits the same text it always did. Only a negated *combination*
  grows the second channel.
- **`~~x` is folded.** A leaf's unknown channel is `~guard`, and negating it back is what every
  rewritten negation does, so without the fold every output carried a visible double negative.

`_expression_has_or` — renamed `_expression_defeats_global_null_filter` — now answers True for a
negation too. A *global* null post-filter is wrong for the same reason in both cases: the
predicate's answer for a null row is not simply "no match". The exact-position index paths
therefore fall back to the scan for a negation, and the segment path (which evaluates the
predicate, so it is already exact per leaf) skips the filter. Measured: negations do not reach an
exact plan today at all, so this is the same belt-and-braces the OR bail already was.

## Verification

`tests/ctable/test_kleene_logic.py` (76 tests):

- **The truth tables**, all nine cells of `&`, `|`, `^` and all three of `~`, over both storages
  and both ways of producing a three-valued operand (a comparison, and a nullable bool column).
  A test reads a row's value the way a user would: true if `where()` keeps it, unknown if
  `is_null()` says so.
- **Algebraic identities**: De Morgan holds, double negation is the identity — and the law of the
  excluded middle **does not**, pinned so no future simplification of the operators can quietly
  restore two-valued logic.
- **A NumPy Kleene oracle** over 8 expression shapes × {mask, sentinel} × {scan, indexed}, checking
  the string form, the operator form and the unknown channel against a transcription of the truth
  tables. 32 combinations agreeing, in the same shape as the Phase 7 oracle.
- **Every V1 kind** negates the same way (int, float, string, bytes, utf8, both storages), plus the
  dictionary `!=` fix and utf8 negation.
- **Operand order**: `plain & three_valued` stays three-valued, including with a materialized
  `NDArray` on the left.
- **Consumers**: `t[pred]`, `sum(where=pred)`, `assign(flag=pred)`, and a sorted view (whose
  `is_null()` must follow the view's order, not the physical one).

Changed elsewhere: `test_null_expressions.py`'s `test_inverted_comparison_selects_null_rows` became
`test_inverted_comparison_drops_null_rows` (it documented the old behaviour, in its name), and the
strict xfail became a plain test.

**Measured** on a 20M-row nullable float column: `where(a > 50)`, `where(~(a > 50))` and both
string equivalents are unchanged (170 ms, scan-bound); `where(~((a > 50) & (b < 20)))` goes
106 ms → 122 ms, **1.15x**, which is the price of the extra channel and buys the answer being
right.

## Named follow-ups

- **A mask-storage predicate cannot become a computed column.** `t.assign(flag=t.a > 10)` raises
  for a mask column because the predicate reads the `.notnull` sidecar, which is not a stored
  column of the table. Pre-existing (the collapsed predicate carried the same operand) and
  unrelated to three-valued logic, but it is what stops `assign` from being tested on both
  storages.
- **A three-valued column of results.** `fillna`/`where` are the only ways out of the third value
  today; storing one would need a computed column with its own validity sidecar.
- **`isin` and the other eager predicates** (`startswith`, `contains`) stay two-valued. If they
  ever return a lazy predicate, they should return this one.
