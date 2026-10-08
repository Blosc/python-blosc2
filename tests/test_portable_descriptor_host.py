"""Host routing tests run against the installed pin without a native 1.0 engine."""

import numpy as np
import pytest

from blosc2 import PortableKernel, blosc2_ext


def test_native_descriptor_authority_and_block_routing(monkeypatch):
    calls = []
    marker = object()

    class Handle:
        def __init__(self, record, jit):
            assert record == b"opaque record validated only by native code"

        def info(self):
            return {
                "source": "native source",
                "entry_point": "k",
                "inputs": {},
                "output_dtype": None,
                "jit": False,
            }

        def descriptor_info(self):
            return {
                "schema_version": "1.0",
                "inputs": {"x": np.dtype("int64")},
                "output_dtype": np.dtype("int64"),
                "cardinality": "block_scalar",
                "ndim": 2,
            }

        def evaluate_block(self, arrays, shape, **context):
            calls.append((arrays, shape, context))
            return marker

    monkeypatch.setattr(blosc2_ext, "PortableArtifactHandle", Handle)
    monkeypatch.setattr(blosc2_ext, "portable_descriptor_available", lambda: True, raising=False)
    kernel = PortableKernel.from_json(b"opaque record validated only by native code")
    assert kernel.result_cardinality == "block_scalar"
    assert kernel.context_ndim == 2
    with pytest.raises(ValueError, match="explicit logical_shape"):
        kernel.evaluate({"x": np.arange(4, dtype="int64").reshape(2, 2)})
    values = np.arange(8, dtype=">i8").reshape(2, 4)[:, ::2]
    mask = np.array([[True, False], [True, True]])
    assert (
        kernel.evaluate_block({"x": values}, logical_shape=(4, 5), block_origin=(2, 3), valid_mask=mask)
        is marker
    )
    arrays, shape, context = calls[0]
    assert shape == (2, 2)
    assert arrays["x"].flags.c_contiguous
    assert arrays["x"].dtype.isnative
    assert context["logical_shape"] == (4, 5)
    assert context["block_origin"] == (2, 3)
    assert context["valid_mask"] is mask
