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


class UnsupportedLegacyRecipeError(ValueError):
    """A retired source-backed computation recipe cannot be loaded."""

    def __init__(self, kind="legacy DSL/Python UDF recipe"):
        super().__init__(
            f"Unsupported {kind}: legacy computation recipes have been retired. "
            "deserialize='full' does not enable them. Use a previous release that "
            "supports this format to materialize the results as ordinary arrays; "
            "reauthor with @blosc2.menudet and compare results before upgrading. "
            "See https://blosc.org/python-blosc2/guides/migrating_to_menudet.html"
        )


class UnsafeDeserializationError(ValueError):
    """Raised when safe deserialization encounters an active serialized value."""

    def __init__(self, kind: str, *, location: str | None = None, hint: str | None = None):
        where = f" in {location}" if location else ""
        advice = hint or (
            "Reopen with deserialize='full' only if you trust "
            "this data and intend to allow reference resolution and lazy-object reconstruction."
        )
        super().__init__(
            f"Encountered active serialized value {kind!r}{where} while using deserialize='safe'. {advice}"
        )
        self.kind = kind
        self.location = location
