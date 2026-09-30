"""Regression tests for optional-network test execution."""

from types import SimpleNamespace

import httpx
import pytest

from conftest import pytest_runtest_call


@pytest.mark.parametrize("network", [False, True])
def test_network_hook_leaves_execution_to_pytest(network):
    def unexpected_run():
        raise AssertionError("the hook must not execute the test a second time")

    item = SimpleNamespace(
        get_closest_marker=lambda name: pytest.mark.network if network else None,
        runtest=unexpected_run,
    )
    hook = pytest_runtest_call(item)
    assert next(hook) is None
    with pytest.raises(StopIteration) as finished:
        hook.send("test result")
    assert finished.value.value == "test result"


@pytest.mark.parametrize("network", [False, True])
def test_network_hook_only_skips_marked_transport_errors(network):
    item = SimpleNamespace(get_closest_marker=lambda name: pytest.mark.network if network else None)
    hook = pytest_runtest_call(item)
    next(hook)
    expected = pytest.skip.Exception if network else httpx.ReadTimeout
    with pytest.raises(expected):
        hook.throw(httpx.ReadTimeout("endpoint unavailable"))


def test_network_hook_does_not_hide_assertion_failures():
    item = SimpleNamespace(get_closest_marker=lambda name: pytest.mark.network)
    hook = pytest_runtest_call(item)
    next(hook)
    with pytest.raises(AssertionError, match="regression"):
        hook.throw(AssertionError("regression"))
