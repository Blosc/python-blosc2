"""Changing the compression GIL policy must not affect the following test."""

import pytest

import blosc2


@pytest.mark.parametrize("gil", [True, False])
def test_releasegil_starts_disabled(gil):
    # Deliberately leave the setting changed. The autouse fixture must restore
    # it before the next parameter (or a Python filler/prefilter) executes.
    assert blosc2.set_releasegil(gil) is False
