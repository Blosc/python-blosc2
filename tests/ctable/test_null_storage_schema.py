#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Schema plumbing for validity-mask null storage (Phase 2).

Specs can now say *where* their nulls live -- in band as a sentinel, or in a
sidecar validity array -- and the schema records it.  Nothing reads or writes
a sidecar yet; this is the declaration layer.

The default is still ``"sentinel"``, so every existing table and every schema
written by this release is byte-identical to before.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import blosc2
from blosc2 import CTable
from blosc2.schema import fill_value_for
from blosc2.schema_compiler import schema_from_dict, schema_to_dict

# ---------------------------------------------------------------------------
# Spec-level declaration
# ---------------------------------------------------------------------------

MASKABLE_SPECS = [
    ("int8", blosc2.int8, {}),
    ("int64", blosc2.int64, {}),
    ("uint8", blosc2.uint8, {}),
    ("float64", blosc2.float64, {}),
    ("bool", blosc2.bool, {}),
    ("timestamp", blosc2.timestamp, {}),
    ("string", blosc2.string, {"max_length": 4}),
    ("bytes", blosc2.bytes, {"max_length": 4}),
    ("utf8", blosc2.utf8, {}),
]


@pytest.mark.parametrize(("label", "factory", "kwargs"), MASKABLE_SPECS)
def test_spec_accepts_null_storage(label, factory, kwargs):
    spec = factory(null_storage="mask", **kwargs)
    assert spec.null_storage == "mask"
    assert spec.uses_mask
    assert spec.nullable
    assert spec.null_value is None


@pytest.mark.parametrize(("label", "factory", "kwargs"), MASKABLE_SPECS)
def test_spec_defaults_to_unresolved_storage(label, factory, kwargs):
    """``nullable=True`` alone defers the choice to the policy."""
    assert factory(nullable=True, **kwargs).null_storage is None
    assert factory(**kwargs).null_storage is None


def test_mask_and_sentinel_together_is_rejected():
    with pytest.raises(ValueError, match="cannot be combined with an explicit null_value"):
        blosc2.int64(null_storage="mask", null_value=-1)


def test_unknown_null_storage_is_rejected():
    with pytest.raises(ValueError, match="null_storage must be one of"):
        blosc2.int64(null_storage="bitmap")


def test_ndarray_spec_accepts_null_storage():
    spec = blosc2.ndarray((2,), dtype=blosc2.int64(), null_storage="mask")
    assert spec.uses_mask
    assert spec.null_value is None


# ---------------------------------------------------------------------------
# Nullable bool loses the 255 reservation under mask storage
# ---------------------------------------------------------------------------


def test_sentinel_bool_is_still_uint8():
    spec = blosc2.bool(nullable=True, null_value=255)
    assert spec.dtype == np.dtype(np.uint8)


def test_mask_bool_stays_bool():
    """The point of the design: no reserved 255, no uint8 leak."""
    assert blosc2.bool(null_storage="mask").dtype == np.dtype(np.bool_)


def test_mask_bool_accepts_any_null_value_rejection():
    with pytest.raises(ValueError, match="Nullable bool null_value must be 255"):
        blosc2.bool(nullable=True, null_value=7)


def test_stored_uint8_bool_reopens_as_uint8_without_the_resolver():
    """Opening a table rebuilds specs through ``spec_cls(**data)`` and never
    runs ``_resolve_nullable_specs``, so the uint8 flip has to survive from
    metadata alone.
    """
    spec = blosc2.bool(nullable=True, null_value=255)
    rebuilt = schema_from_dict({"version": 1, "columns": [{"name": "flag", **spec.to_metadata_dict()}]})
    assert rebuilt.columns[0].spec.dtype == np.dtype(np.uint8)
    assert rebuilt.columns[0].dtype == np.dtype(np.uint8)


# ---------------------------------------------------------------------------
# Complex gains nullability, mask-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [blosc2.complex64, blosc2.complex128])
def test_complex_is_nullable_via_mask(factory):
    spec = factory(nullable=True)
    assert spec.uses_mask, "complex has no representable sentinel, so nullable implies mask"


@pytest.mark.parametrize("factory", [blosc2.complex64, blosc2.complex128])
def test_complex_rejects_a_sentinel(factory):
    with pytest.raises(ValueError, match="cannot use an in-band null sentinel"):
        factory(null_value=0j)


@pytest.mark.parametrize("factory", [blosc2.complex64, blosc2.complex128])
def test_complex_non_nullable_is_unchanged(factory):
    spec = factory()
    assert not spec.nullable
    assert spec.to_metadata_dict() == {"kind": spec._kind}


# ---------------------------------------------------------------------------
# Serialization and version gating
# ---------------------------------------------------------------------------


def _schema(**cols):
    fields = [(name, ann, blosc2.field(spec)) for name, (ann, spec) in cols.items()]
    return CTable(dataclasses.make_dataclass("Row", fields))._schema


def test_sentinel_schema_is_unchanged_on_disk():
    """A sentinel table must serialize exactly as it did before mask storage.

    ``"sentinel"`` is never emitted, so old readers keep working.
    """
    d = schema_to_dict(_schema(v=(int, blosc2.int64(null_value=-1))))
    assert d["version"] == 1
    assert d["columns"][0] == {"name": "v", "kind": "int64", "nullable": True, "null_value": -1}
    assert "null_storage" not in d["columns"][0]


def test_mask_column_records_storage_and_bumps_version():
    d = schema_to_dict(_schema(v=(int, blosc2.int64(null_storage="mask"))))
    assert d["version"] == 3
    assert d["columns"][0]["null_storage"] == "mask"
    assert "null_value" not in d["columns"][0]


def test_version_bump_is_conditional_on_a_mask_column():
    """Only mask-using tables become unreadable by older readers."""
    mixed = _schema(
        s=(int, blosc2.int64(null_value=-1)),
        plain=(int, blosc2.int64()),
    )
    assert schema_to_dict(mixed)["version"] == 1


def test_schema_round_trip_preserves_storage():
    d = schema_to_dict(
        _schema(
            m=(int, blosc2.int64(null_storage="mask")),
            s=(int, blosc2.int64(null_value=-1)),
        )
    )
    back = schema_from_dict(d)
    assert back.columns_by_name["m"].spec.uses_mask
    assert back.columns_by_name["s"].spec.null_value == -1
    assert not back.columns_by_name["s"].spec.uses_mask
    assert schema_to_dict(back) == d


def test_schema_from_dict_accepts_version_3():
    schema_from_dict({"version": 3, "columns": [{"name": "v", "kind": "int64"}]})


def test_unsupported_version_names_the_version_and_the_way_out():
    with pytest.raises(ValueError, match="Unsupported schema version 9") as exc:
        schema_from_dict({"version": 9, "columns": []})
    assert "convert_nulls" in str(exc.value), "the message should say how to get a readable copy"


# ---------------------------------------------------------------------------
# NullPolicy
# ---------------------------------------------------------------------------


def test_policy_defaults_to_a_mask():
    """A newly created nullable column keeps its nulls out of band.

    ``null_storage`` stays ``None`` -- *unspecified* -- rather than being
    resolved at construction, and that distinction is load-bearing: it is what
    lets a type-wide sentinel field imply sentinel storage without
    contradicting anything the caller wrote.
    """
    assert blosc2.NullPolicy().null_storage is None
    assert blosc2.NullPolicy().resolve_null_storage() == "mask"
    assert blosc2.NullPolicy(null_storage="sentinel").resolve_null_storage() == "sentinel"


def test_policy_rejects_an_unknown_storage():
    with pytest.raises(ValueError, match="null_storage must be 'mask', 'sentinel' or None"):
        blosc2.NullPolicy(null_storage="bitmap")


def test_policy_mask_with_a_type_wide_sentinel_field_is_a_contradiction():
    with pytest.raises(ValueError, match="contradicts the type-wide sentinel field"):
        blosc2.NullPolicy(null_storage="mask", string_value="<NA>")


def test_policy_mask_still_allows_per_column_sentinels():
    """``column_null_values`` forces sentinel storage per column, which is
    a refinement of the default rather than a contradiction of it."""
    policy = blosc2.NullPolicy(null_storage="mask", column_null_values={"v": -1})
    assert policy.null_storage == "mask"


def test_untouched_nan_float_value_does_not_read_as_set():
    """float_value defaults to NaN, which never equals itself."""
    assert not blosc2.NullPolicy()._sentinel_field_is_set("float_value")
    assert blosc2.NullPolicy(float_value=-1.0)._sentinel_field_is_set("float_value")


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


def _resolved(spec, annotation=int, name="v", **policy_kw):
    Row = dataclasses.make_dataclass("Row", [(name, annotation, blosc2.field(spec))])
    if policy_kw:
        with blosc2.null_policy(blosc2.NullPolicy(**policy_kw)):
            return CTable(Row)._schema.columns_by_name[name]
    return CTable(Row)._schema.columns_by_name[name]


def test_explicit_storage_beats_the_policy():
    col = _resolved(blosc2.int64(null_storage="mask"), null_storage="sentinel")
    assert col.spec.uses_mask


def test_explicit_null_value_implies_sentinel_under_a_mask_policy():
    col = _resolved(blosc2.int64(null_value=-1), null_storage="mask")
    assert col.spec.null_value == -1
    assert not col.spec.uses_mask


def test_column_null_values_implies_sentinel_under_a_mask_policy():
    col = _resolved(blosc2.int64(nullable=True), null_storage="mask", column_null_values={"v": -1})
    assert col.spec.null_value == -1
    assert not col.spec.uses_mask


@pytest.mark.parametrize(
    ("field", "value", "spec", "annotation"),
    [
        ("string_value", "<NA>", blosc2.string(max_length=4, nullable=True), str),
        ("bytes_value", b"<NA>", blosc2.bytes(max_length=4, nullable=True), bytes),
        ("float_value", -1.0, blosc2.float64(nullable=True), float),
        ("timestamp_value", -1, blosc2.timestamp(nullable=True), object),
        ("signed_int_strategy", "max", blosc2.int64(nullable=True), int),
        ("unsigned_int_strategy", "min", blosc2.uint64(nullable=True), int),
    ],
)
def test_type_wide_sentinel_field_implies_sentinel_for_its_kinds(field, value, spec, annotation):
    """Existing ``NullPolicy(float_value=...)`` code must keep working once the
    default flips to mask, so setting a sentinel field opts those types in.
    """
    col = _resolved(spec, annotation, **{field: value})
    assert not col.spec.uses_mask
    assert col.spec.null_value is not None


def test_bool_value_cannot_imply_sentinel_storage():
    """``bool_value``'s default is the only value it may hold, so "was it set?"
    has no answer for it: ``255`` is the sole legal sentinel for a nullable
    bool.  ``NullPolicy(bool_value=255)`` therefore states the default and
    changes nothing -- a bool column that wants sentinel storage has to say so,
    with ``null_storage`` or ``column_null_values``.
    """
    col = _resolved(blosc2.bool(nullable=True), bool, bool_value=255)
    assert col.spec.uses_mask
    col = _resolved(blosc2.bool(nullable=True), bool, null_storage="sentinel")
    assert not col.spec.uses_mask
    assert col.spec.null_value == 255


def test_policy_mask_applies_to_plain_nullable():
    col = _resolved(blosc2.int64(nullable=True), null_storage="mask")
    assert col.spec.uses_mask
    assert col.spec.null_value is None


def test_mask_skips_string_max_length_widening():
    """The sentinel path widens ``U4`` to fit ``__BLOSC2_NULL__``; mask does not."""
    col = _resolved(blosc2.string(max_length=4, nullable=True), str, null_storage="mask")
    assert col.dtype == np.dtype("U4")

    sentinel_col = _resolved(blosc2.string(max_length=4, nullable=True), str, null_storage="sentinel")
    assert sentinel_col.dtype == np.dtype("U15")


def test_mask_skips_the_bool_uint8_flip():
    col = _resolved(blosc2.bool(nullable=True), bool, null_storage="mask")
    assert col.dtype == np.dtype(np.bool_)
    assert col.spec.dtype == np.dtype(np.bool_)


def test_mask_skips_the_ndarray_bool_uint8_flip():
    col = _resolved(blosc2.ndarray((2,), dtype=blosc2.bool(), nullable=True), object, null_storage="mask")
    assert col.spec.dtype == np.dtype(np.bool_)
    assert col.spec.itemsize == 1


def test_sentinel_storage_is_unchanged_when_asked_for():
    """The uint8 widening and the reserved 255 stay exactly as they were.

    Every stored table keeps them, and so does any column that asks -- which is
    what makes the default flip a change to *new* columns only.
    """
    col = _resolved(blosc2.bool(nullable=True), bool, null_storage="sentinel")
    assert col.dtype == np.dtype(np.uint8)
    assert col.spec.null_value == 255


# ---------------------------------------------------------------------------
# fill_value_for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (blosc2.int32(), 0),
        (blosc2.uint8(), 0),
        (blosc2.float32(), None),  # NaN, checked separately
        (blosc2.complex128(), 0j),
        (blosc2.bool(), False),
        (blosc2.string(max_length=4), ""),
        (blosc2.bytes(max_length=4), b""),
        (blosc2.utf8(), ""),
    ],
)
def test_fill_value_for(spec, expected):
    got = fill_value_for(spec)
    if expected is None:
        assert np.isnan(got)
    else:
        assert got == expected


def test_fill_value_for_timestamp_decodes_to_nat():
    """int64.min is NaT's bit pattern, so a null timestamp reads back as NaT."""
    fill = fill_value_for(blosc2.timestamp())
    assert fill == np.iinfo(np.int64).min
    assert np.isnat(np.array([fill], dtype="datetime64[us]")[0])


def test_fill_value_is_not_recorded_in_the_schema():
    """Recording it would recreate sentinel collisions at the metadata layer."""
    d = schema_to_dict(_schema(v=(float, blosc2.float64(null_storage="mask"))))
    assert "fill_value" not in d["columns"][0]
    assert "null_value" not in d["columns"][0]


# ---------------------------------------------------------------------------
# The declaration survives every persistence route
# ---------------------------------------------------------------------------

_MASK_ROW_FIELDS = [
    ("m", int, blosc2.field(blosc2.int64(null_storage="mask"))),
    ("f", bool, blosc2.field(blosc2.bool(null_storage="mask"))),
    ("s", str, blosc2.field(blosc2.string(max_length=4, null_storage="mask"))),
    ("c", complex, blosc2.field(blosc2.complex128(nullable=True))),
    ("plain", int, blosc2.field(blosc2.int64())),
]
_MASK_PAYLOAD = {
    "m": [1, 2],
    "f": [True, False],
    "s": ["ab", "cd"],
    "c": [1 + 2j, 3 + 4j],
    "plain": [7, 8],
}
_EXPECTED_STORAGE = {"m": "mask", "f": "mask", "s": "mask", "c": "mask", "plain": None}


def _mask_table():
    t = CTable(dataclasses.make_dataclass("MaskRow", _MASK_ROW_FIELDS))
    t.extend(_MASK_PAYLOAD)
    return t


def _storage_of(table):
    return {c.name: c.spec.null_storage for c in table._schema.columns}


@pytest.mark.parametrize("ext", [".b2d", ".b2z"])
def test_storage_survives_save_and_open(tmp_path, ext):
    path = tmp_path / f"masked{ext}"
    _mask_table().save(str(path))
    reopened = blosc2.open(str(path))
    assert _storage_of(reopened) == _EXPECTED_STORAGE
    assert reopened["s"][:].tolist() == ["ab", "cd"]
    assert reopened["c"][:].tolist() == [1 + 2j, 3 + 4j]
    # A mask column keeps its natural dtype across the round trip.
    assert reopened._schema.columns_by_name["f"].dtype == np.dtype(np.bool_)
    assert reopened._schema.columns_by_name["s"].dtype == np.dtype("U4")


def test_storage_survives_to_cframe():
    restored = blosc2.ctable_from_cframe(_mask_table().to_cframe())
    assert _storage_of(restored) == _EXPECTED_STORAGE


def test_storage_survives_copy():
    """Nothing auto-migrates: copy() preserves each column's storage."""
    assert _storage_of(_mask_table().copy()) == _EXPECTED_STORAGE


# ---------------------------------------------------------------------------
# The default flip (4.10.2)
# ---------------------------------------------------------------------------


def test_a_bare_nullable_column_gets_a_mask():
    """What the flip changes, stated once for each kind that has a choice."""
    for spec, annotation in [
        (blosc2.int8(nullable=True), int),
        (blosc2.float64(nullable=True), float),
        (blosc2.bool(nullable=True), bool),
        (blosc2.string(max_length=4, nullable=True), str),
        (blosc2.bytes(max_length=4, nullable=True), bytes),
        (blosc2.utf8(nullable=True), str),
        (blosc2.timestamp(nullable=True), object),
    ]:
        col = _resolved(spec, annotation)
        assert col.spec.uses_mask, spec
        assert col.spec.null_value is None, spec


def test_a_stored_table_keeps_the_storage_it_was_written_with(tmp_path):
    """The flip governs *creation* only; opening never re-resolves anything.

    Which is the whole reason it is safe: every table already on disk carries
    its own answer, sentinel included, and reading one does not consult a policy.
    """
    Row = dataclasses.make_dataclass(
        "OldRow",
        [("flag", bool, blosc2.field(blosc2.bool(nullable=True, null_storage="sentinel")))],
    )
    t = blosc2.CTable(Row, urlpath=str(tmp_path / "t.b2t"), mode="w", expected_size=4)
    t.extend([(1,), (255,), (0,)])
    del t

    reopened = blosc2.open(str(tmp_path / "t.b2t"))
    assert reopened["flag"].null_storage == "sentinel"
    assert reopened["flag"].dtype == np.dtype(np.uint8)
    assert reopened["flag"].null_value == 255
    assert reopened["flag"].is_null().tolist() == [False, True, False]


def test_a_mask_default_table_records_schema_version_3(tmp_path):
    """Only a table that *uses* a mask needs a reader that understands one."""
    from blosc2.schema_compiler import schema_to_dict

    Row = dataclasses.make_dataclass("V3Row", [("v", int, blosc2.field(blosc2.int64(nullable=True)))])
    t = blosc2.CTable(Row, expected_size=4)
    assert schema_to_dict(t._schema)["version"] == 3

    Plain = dataclasses.make_dataclass("V1Row", [("v", int, blosc2.field(blosc2.int64()))])
    assert schema_to_dict(blosc2.CTable(Plain, expected_size=4)._schema)["version"] == 1


def test_the_flip_is_one_kwarg_from_the_old_behaviour():
    col = _resolved(blosc2.int64(nullable=True), int, null_storage="sentinel")
    assert not col.spec.uses_mask
    assert col.spec.null_value == np.iinfo(np.int64).min
