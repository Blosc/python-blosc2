#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Row-level validation via an internally-generated Pydantic model.

All Pydantic-specific logic is isolated here.  CTable and the rest of the
schema layer never import from Pydantic directly.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

from pydantic import BaseModel, Field, ValidationError, create_model

from blosc2.schema_compiler import CompiledSchema  # noqa: TC001


def build_validator_model(schema: CompiledSchema) -> type[BaseModel]:
    """Return (and cache) a Pydantic model class for *schema*.

    Built once per schema; subsequent calls return the cached class.
    The model enforces all constraints declared in each column's
    :class:`~blosc2.schema.SchemaSpec` (``ge``, ``le``, ``gt``, ``lt``,
    ``max_length``, ``min_length``, ``pattern``).
    """
    if schema.validator_model is not None:
        return schema.validator_model

    field_definitions: dict[str, Any] = {}
    for col in schema.columns:
        pydantic_kwargs = col.spec.to_pydantic_kwargs()
        if col.default is MISSING:
            field_definitions[col.name] = (col.py_type, Field(**pydantic_kwargs))
        else:
            field_definitions[col.name] = (col.py_type, Field(default=col.default, **pydantic_kwargs))

    cls_name = schema.row_cls.__name__ if schema.row_cls is not None else "Unknown"
    model_cls = create_model(f"_Validator_{cls_name}", **field_definitions)
    schema.validator_model = model_cls
    return model_cls


def _is_null_value(val, null_value) -> bool:
    """Return True if *val* equals the null sentinel, handling NaN correctly."""
    import math

    if null_value is None:
        return False
    try:
        if isinstance(null_value, float) and math.isnan(null_value):
            return isinstance(val, float) and math.isnan(val)
    except TypeError:
        pass
    return val == null_value


def _safe_placeholder(col) -> Any:
    """Return a value that passes Pydantic validation for *col* (used for null bypass)."""
    spec = col.spec
    ge = getattr(spec, "ge", None)
    gt = getattr(spec, "gt", None)
    if ge is not None:
        return ge
    if gt is not None:
        return gt + 1
    # For string/bytes, use an empty string
    if col.dtype.kind in ("U", "S"):
        return ""
    # For bool
    if col.dtype.kind == "b":
        return False
    return 0


def _mask_nulls(schema: CompiledSchema, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace null sentinel values with safe placeholders.

    Returns (masked_row, null_positions) where null_positions maps
    column name → original null value for columns that were masked.
    """
    masked = dict(row)
    nulled: dict[str, Any] = {}
    for col in schema.columns:
        nv = getattr(col.spec, "null_value", None)
        if nv is None:
            continue
        val = row.get(col.name)
        if val is not None and _is_null_value(val, nv):
            nulled[col.name] = val
            masked[col.name] = _safe_placeholder(col)
    return masked, nulled


def validate_row(schema: CompiledSchema, row: dict[str, Any]) -> dict[str, Any]:
    """Validate a single row dict and return the coerced values.

    Parameters
    ----------
    schema:
        Compiled schema for the table.
    row:
        ``{column_name: value}`` mapping for one row.

    Returns
    -------
    dict
        Validated (and Pydantic-coerced) values ready for storage.

    Raises
    ------
    ValueError
        If any constraint is violated.  The message includes the column
        name and the violated constraint.
    """
    model_cls = build_validator_model(schema)
    masked_row, nulled = _mask_nulls(schema, row)
    try:
        instance = model_cls(**masked_row)
    except ValidationError as exc:
        # Re-raise as a plain ValueError so callers don't need to import Pydantic.
        raise ValueError(str(exc)) from exc
    result = instance.model_dump()
    result.update(nulled)
    return result


def validate_rows_rowwise(schema: CompiledSchema, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a list of row dicts.  Returns a list of validated dicts.

    Parameters
    ----------
    schema:
        Compiled schema for the table.
    rows:
        List of ``{column_name: value}`` mappings.

    Raises
    ------
    ValueError
        On the first row that violates a constraint, with the row index
        and the Pydantic error details.
    """
    model_cls = build_validator_model(schema)
    result = []
    for i, row in enumerate(rows):
        masked_row, nulled = _mask_nulls(schema, row)
        try:
            instance = model_cls(**masked_row)
        except ValidationError as exc:
            raise ValueError(f"Row {i}: {exc}") from exc
        validated = instance.model_dump()
        validated.update(nulled)
        result.append(validated)
    return result
