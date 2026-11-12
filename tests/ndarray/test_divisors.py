#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import pytest

from blosc2 import blosc2_ext, core


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1, 1, 1),
        (12, 5, 4),
        (49, 28, 7),  # Repeated divisor at the square root.
        (101, 50, 1),  # No complementary divisor fits the bound.
        (101, 10, 1),  # The bound terminates enumeration before sqrt(a).
        (24_000_000, 125_000, 125_000),  # Exact divisor.
        (24_000_000, 125_001, 125_000),  # Nearby divisor.
        (24_000_000, 16_750_000, 12_000_000),  # Boolean chunk search.
        (24_000_000, 2_096_000, 2_000_000),  # Float64 chunk search.
        (24_000_000, 131_072, 125_000),
        (24_000_000, 16_384, 16_000),
        (1_000_000_007, 16_750_000, 1),  # Large prime dimension.
        (2**63 - 1, 127, 127),
        (2**61 - 1, 127, 1),
        (2**62, 127, 64),
        (2**63 - 1, 2**63 - 1, 2**63 - 1),
        (12, 2**63 - 1, 12),  # Bound above the dimension.
        (12, 0, 1),  # Preserve the legacy fallback for nonpositive bounds.
        (12, -(2**63), 1),
    ],
)
def test_divisor_at_most(a, b, expected):
    assert blosc2_ext.divisor_at_most(a, b) == expected
    assert blosc2_ext.nearest_divisor(a, b, strict=True) == expected
    assert core.nearest_divisor(a, b, strict=True) == expected


def test_divisor_at_most_small_reference():
    for a in range(1, 201):
        divisors = [i for i in range(1, a + 1) if a % i == 0]
        for b in range(1, a + 10):
            expected = max(i for i in divisors if i <= b)
            assert blosc2_ext.divisor_at_most(a, b) == expected


@pytest.mark.parametrize("a", [0, -1])
def test_divisor_at_most_invalid_dimension(a):
    with pytest.raises(ValueError, match="a must be positive"):
        blosc2_ext.divisor_at_most(a, 10)


def test_nearest_divisor_preserves_non_strict_behavior():
    assert blosc2_ext.nearest_divisor(12, 11) == 12
    assert core.nearest_divisor(12, 11) == 12
    # The existing implementations break ties differently; keep both behaviors.
    assert blosc2_ext.nearest_divisor(12, 5) == 6
    assert core.nearest_divisor(12, 5) == 4
    assert blosc2_ext.nearest_divisor(240_000, 125_001) == 120_000
    assert core.nearest_divisor(240_000, 125_001) == 120_000


def test_compute_partition_uses_divisor_at_most(monkeypatch):
    calls = []
    original = blosc2_ext.divisor_at_most

    def traced(a, b):
        calls.append((a, b))
        return original(a, b)

    monkeypatch.setattr(blosc2_ext, "divisor_at_most", traced)
    assert core.compute_partition(16_750_000, (24_000_000,), (125_000,)) == [12_000_000]
    assert calls == [(24_000_000, 16_750_000)]
