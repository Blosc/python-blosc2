"""The stable core owns formula discovery and portable JIT probes."""

import numpy as np
import pytest

import blosc2


def test_formula_width_discovery():
    assert blosc2.blosc2_ext.me_output_dtype("x + y", {"x": "U3", "y": "U2"}) == np.dtype("U5")
    assert blosc2.blosc2_ext.me_output_dtype("sum(x)", {"x": "float64"}) is None
    assert blosc2.blosc2_ext.me_output_dtype("x", {"x": "S7"}) == np.dtype("S7")


def test_jit_probe_uses_portable_preparation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("portable JIT probe used the legacy ND compiler")

    monkeypatch.setattr(blosc2.NDArray, "_dsl_jit_status", forbidden)

    def scale(x, n):
        return x * n

    result = blosc2.validate_dsl_jit(scale, {"x": np.float64, "n": 3}, np.float64)
    assert result["valid"]
    assert result["compiled"]
    assert result["status"] == "ME_COMPILE_SUCCESS"


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("strings", [False, True])
def test_formula_grid_prefilter_padded_rows(jit, strings):
    shape, chunks, blocks = (3, 5), (2, 4), (2, 3)
    if strings:
        x = np.full(shape, "abc", dtype="U3")
        y = np.full(shape, "de", dtype="U2")
        expected = np.full(shape, "abcde", dtype="U5")
    else:
        x = np.arange(15, dtype="float64").reshape(shape)
        y = x / 2
        expected = x + y
    operands = {
        "x": blosc2.asarray(x, chunks=chunks, blocks=blocks),
        "y": blosc2.asarray(y, chunks=chunks, blocks=blocks),
    }
    out = blosc2.empty(shape, dtype=expected.dtype, chunks=chunks, blocks=blocks)
    out._set_pref_expr("x + y", operands, blosc2.FPAccuracy.DEFAULT, jit=jit)
    try:
        data = np.empty(out.schunk.chunksize, dtype="uint8")
        for chunk in range(out.schunk.nchunks):
            out.schunk.update_data(chunk, data, copy=False)
    finally:
        out.schunk.remove_prefilter("miniexpr")
    np.testing.assert_array_equal(out[:], expected)


def test_rejected_formula_does_not_install_prefilter():
    x = blosc2.asarray(np.arange(8, dtype="float64"))
    out = blosc2.empty(x.shape, dtype=x.dtype, chunks=x.chunks, blocks=x.blocks)
    with pytest.raises(NotImplementedError, match="Cannot compile"):
        out._set_pref_expr("unknown(x)", {"x": x}, blosc2.FPAccuracy.DEFAULT, jit=True)
    # A successful installation on the same destination proves rejection did
    # not leave callback state installed. Removal releases both handle kinds.
    out._set_pref_expr("x + 1", {"x": x}, blosc2.FPAccuracy.DEFAULT, jit=False)
    try:
        with pytest.raises(ValueError, match="already has a prefilter"):
            out._set_pref_expr("x + 2", {"x": x}, blosc2.FPAccuracy.DEFAULT)
        data = np.empty(out.schunk.chunksize, dtype="uint8")
        for chunk in range(out.schunk.nchunks):
            out.schunk.update_data(chunk, data, copy=False)
    finally:
        out.schunk.remove_prefilter("miniexpr")
    np.testing.assert_array_equal(out[:], np.arange(8) + 1)
