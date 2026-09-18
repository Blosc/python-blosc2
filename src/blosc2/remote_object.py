#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Common public API for remote Blosc2 objects."""


class RemoteObject:
    """Base class for remote arrays, stores, and tables.

    Construct a concrete remote type or use :func:`blosc2.open`; this class is
    intended for type checks and the shared remote-object contract.
    """

    def _check_open(self) -> None:
        """Raise when this handle cannot be used."""

    def close(self) -> None:
        """Release resources owned by this handle."""
        raise NotImplementedError

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
