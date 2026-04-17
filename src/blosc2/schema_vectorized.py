#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Vectorized (NumPy-based) constraint validation for bulk inserts.

Used by ``CTable.extend()`` to check entire column arrays at once,
avoiding the per-row Python overhead of Pydantic validation for large
batches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blosc2.schema_compiler import CompiledColumn, CompiledSchema  # noqa: TC001


def _validate_string_lengths(col: CompiledColumn, arr: Any) -> None:
    """Check min_length / max_length constraints on a string/bytes column."""
    if arr.dtype.kind in ("U", "S"):
        lengths = np.char.str_len(arr)
    else:
        lengths = np.vectorize(len)(arr.astype(object))

    spec = col.spec
    if getattr(spec, "max_length", None) is not None:
        bad = lengths > spec.max_length
        if np.any(bad):
            first = arr.astype(object)[bad][0]
            raise ValueError(f"Column '{col.name}': value {first!r} exceeds max_length={spec.max_length}")
    if getattr(spec, "min_length", None) is not None:
        bad = lengths < spec.min_length
        if np.any(bad):
            first = arr.astype(object)[bad][0]
            raise ValueError(
                f"Column '{col.name}': value {first!r} is shorter than min_length={spec.min_length}"
            )


def _null_mask_for_spec(arr: np.ndarray, spec) -> np.ndarray | None:
    """Return a boolean mask True where values are the null sentinel, or None if no null_value."""
    null_value = getattr(spec, "null_value", None)
    if null_value is None:
        return None
    try:
        import math

        if isinstance(null_value, float) and math.isnan(null_value):
            return np.isnan(arr)
    except TypeError:
        pass
    return arr == null_value


def validate_column_values(col: CompiledColumn, values: Any) -> None:
    """Check all constraint attributes of *col*'s spec against *values*.

    Parameters
    ----------
    col:
        Compiled column descriptor (carries the spec with constraints).
    values:
        Array-like of values for this column.

    Raises
    ------
    ValueError
        If any value violates a constraint declared on the column's spec.
    """
    spec = col.spec
    arr = np.asarray(values)

    # Compute null mask so sentinels bypass constraint checks
    null_mask = _null_mask_for_spec(arr, spec)
    if null_mask is not None:
        check = arr[~null_mask]
    else:
        check = arr

    # Numeric bounds
    if getattr(spec, "ge", None) is not None:
        bad = check < spec.ge
        if np.any(bad):
            first = check[bad][0]
            raise ValueError(f"Column '{col.name}': value {first!r} violates constraint ge={spec.ge}")
    if getattr(spec, "gt", None) is not None:
        bad = check <= spec.gt
        if np.any(bad):
            first = check[bad][0]
            raise ValueError(f"Column '{col.name}': value {first!r} violates constraint gt={spec.gt}")
    if getattr(spec, "le", None) is not None:
        bad = check > spec.le
        if np.any(bad):
            first = check[bad][0]
            raise ValueError(f"Column '{col.name}': value {first!r} violates constraint le={spec.le}")
    if getattr(spec, "lt", None) is not None:
        bad = check >= spec.lt
        if np.any(bad):
            first = check[bad][0]
            raise ValueError(f"Column '{col.name}': value {first!r} violates constraint lt={spec.lt}")

    # String / bytes length bounds
    # np.char.str_len is a true C-level vectorized operation for 'U' and 'S'
    # dtypes.  Fall back to np.vectorize(len) only for unexpected object arrays.
    if getattr(spec, "max_length", None) is not None or getattr(spec, "min_length", None) is not None:
        _validate_string_lengths(col, check)


def validate_column_batch(schema: CompiledSchema, columns: dict[str, Any]) -> None:
    """Validate a dict of column arrays against all constraints in *schema*.

    Parameters
    ----------
    schema:
        Compiled schema for the table.
    columns:
        ``{column_name: array_like}`` mapping of the batch being inserted.

    Raises
    ------
    ValueError
        On the first constraint violation found, naming the column and
        the violated constraint.
    """
    for col in schema.columns:
        if col.name in columns:
            validate_column_values(col, columns[col.name])
