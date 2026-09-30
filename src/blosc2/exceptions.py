#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################


class MissingOperands(ValueError):
    def __init__(self, expr, missing_ops):
        self.expr = expr
        self.missing_ops = missing_ops

        message = f'Lazy expression "{expr}" with missing operands: {missing_ops}'
        super().__init__(message)


class UnsafeDeserializationError(ValueError):
    """Raised when safe deserialization encounters an active serialized value."""

    def __init__(self, kind: str, *, location: str | None = None):
        where = f" in {location}" if location else ""
        super().__init__(
            f"Encountered active serialized value {kind!r}{where} while using "
            "deserialize='safe'. Reopen with deserialize='full' only if you trust "
            "this data and intend to allow reference resolution and lazy-object reconstruction."
        )
        self.kind = kind
        self.location = location
