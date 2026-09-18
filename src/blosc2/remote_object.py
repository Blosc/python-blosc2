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

    @property
    def source(self):
        """Credential-free descriptor for the selected remote object."""
        raise NotImplementedError

    @property
    def attrs(self):
        """Read-only user metadata for the selected remote object."""
        raise NotImplementedError

    @property
    def traffic(self):
        """Source-read accounting for this object or its shared owner."""
        raise NotImplementedError

    @property
    def cache_policy(self):
        """Configured payload-retention policy."""
        raise NotImplementedError

    @property
    def max_cache_bytes(self):
        """Configured compressed-payload allowance."""
        raise NotImplementedError

    @property
    def cache_bytes(self):
        """Compressed payload currently retained at this object's scope."""
        raise NotImplementedError

    @property
    def mutable(self) -> bool:
        """Default mutability for future reference exports."""
        raise NotImplementedError

    @mutable.setter
    def mutable(self, value: bool) -> None:
        raise NotImplementedError

    @property
    def is_cache_mutable(self) -> bool:
        """Whether the currently attached local cache is writable."""
        raise NotImplementedError

    def save(
        self,
        destination,
        *,
        include_cache: bool = True,
        mutable: bool | None = None,
        overwrite: bool = False,
    ) -> str:
        """Write a portable remote reference and return its output path."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources owned by this handle."""
        raise NotImplementedError

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
