"""Keep the default-suite sampling deterministic and preserve every axis value."""

from conftest import orthogonal_cases


def test_orthogonal_cases_cover_independent_and_mixed_settings():
    dimensions = (["baseline", "other"], [1, 2, 3], [False, True])
    cases = orthogonal_cases(*dimensions)
    assert cases == orthogonal_cases(*dimensions)
    assert len(cases) == len(set(cases)) < 2 * 3 * 2
    for axis, values in enumerate(dimensions):
        assert {case[axis] for case in cases} == set(values)
        for value in values:
            baseline = [items[0] for items in dimensions]
            baseline[axis] = value
            assert tuple(baseline) in cases
    assert ("other", 2, True) in cases


def test_orthogonal_cases_preserve_tuple_axes_and_singletons():
    assert orthogonal_cases([None, (), (0, 2)], [True]) == [(None, True), ((), True), ((0, 2), True)]
    assert orthogonal_cases(["only"]) == [("only",)]
