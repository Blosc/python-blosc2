"""Expression fusion must not disable comparisons in another thread."""

import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import numpy as np
import pytest

import blosc2


@pytest.mark.skipif(blosc2.IS_WASM, reason="Pyodide does not support Python threads")
def test_fusion_keeps_elementwise_equality_in_other_threads(monkeypatch):
    module = importlib.import_module("blosc2.lazyexpr")
    arr = blosc2.asarray(np.arange(4))
    expr = arr + 1
    entered, release = Event(), Event()
    original = module.check_dtype
    main_thread = get_ident()

    def paused_check_dtype(*args, **kwargs):
        if get_ident() != main_thread:
            entered.set()
            assert release.wait(10), "Comparison thread did not release fusion"
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "check_dtype", paused_check_dtype)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: expr + arr)
        try:
            assert entered.wait(10), "Fusion did not reach dtype checking"
            assert not blosc2._disable_overloaded_equal
            comparison = arr == 2
            assert isinstance(comparison, blosc2.LazyExpr)
        finally:
            release.set()
        fused = future.result(timeout=10)
    assert not blosc2._disable_overloaded_equal
    np.testing.assert_array_equal(comparison[:], np.arange(4) == 2)
    np.testing.assert_array_equal(fused[:], np.arange(4) * 2 + 1)
