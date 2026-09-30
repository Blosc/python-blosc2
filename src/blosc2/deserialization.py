#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Deserialization policy shared by persisted Blosc2 objects."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class DeserializeMode(StrEnum):
    """Policy for reconstructing values stored in Blosc2 metadata or containers."""

    SAFE = "safe"
    FULL = "full"


def normalize_deserialize(value: str | DeserializeMode) -> DeserializeMode:
    """Return a validated :class:`DeserializeMode`."""

    if isinstance(value, bool):
        raise TypeError("deserialize must be 'safe' or 'full', not a boolean")
    if isinstance(value, DeserializeMode):
        return value
    try:
        return DeserializeMode(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("deserialize must be 'safe' or 'full'") from exc


def get_deserialize(obj: Any, default: str | DeserializeMode = DeserializeMode.FULL) -> DeserializeMode:
    """Return the policy attached to *obj* or its physical SChunk carrier."""

    carrier = getattr(obj, "schunk", obj)
    return normalize_deserialize(getattr(carrier, "_deserialize_mode", default))


def set_deserialize(obj: Any, value: str | DeserializeMode) -> Any:
    """Attach an immutable effective policy to an object and its carrier."""

    mode = normalize_deserialize(value)
    carrier = getattr(obj, "schunk", obj)
    current = getattr(carrier, "_deserialize_mode", None)
    if current is not None and normalize_deserialize(current) is not mode:
        raise ValueError(
            "The deserialization policy of an opened object cannot be changed; reopen it instead"
        )
    carrier._deserialize_mode = mode
    if carrier is not obj:
        current = getattr(obj, "_deserialize_mode", None)
        if current is not None and normalize_deserialize(current) is not mode:
            raise ValueError(
                "The deserialization policy of an opened object cannot be changed; reopen it instead"
            )
        obj._deserialize_mode = mode
    return obj
