"""Private typed boundary tests: explicit selection, never adapter fallback."""

import gc

import numpy as np
import pytest

from blosc2 import blosc2_ext

pytestmark = pytest.mark.skipif(
    not getattr(blosc2_ext, "_b2nd_exec_available", lambda: False)(),
    reason="requires opt-in C-Blosc2 typed executor",
)


def descriptor(dtype="int64"):
    return [
        {"op": 0, "dtype": dtype, "slot": 0},
        {"op": 0, "dtype": dtype, "slot": 1},
        {"op": 1, "dtype": "int64", "scalar": np.int64(3).tobytes(), "weak": True},
        {"op": 4, "args": [1, 2]},
        {"op": 2, "args": [0, 3]},
        {"op": 6, "args": [4]},
    ]


def test_typed_prepare_copies_descriptor_and_schedule_owns_plan():
    nodes = descriptor()
    plan = blosc2_ext._TypedExecPlan(nodes, 2, 5)
    nodes.clear()
    assert plan.info()["inferred_dtype"] == np.dtype("int64")
    schedule = plan.specialize([("int64", (17,)), ("int64", (17,))], 8, 64)
    del plan
    gc.collect()
    assert schedule.info() == {
        "items": 17,
        "tile_items": 8,
        "mapped_bytes": 64,
        "reducer_bytes": 8,
        "output_items": 1,
    }


def test_typed_errors_and_metadata_only_specialization():
    nodes = descriptor()
    nodes[3]["args"] = [3, 2]
    with pytest.raises(ValueError, match="reference"):
        blosc2_ext._TypedExecPlan(nodes, 2, 5)
    plan = blosc2_ext._TypedExecPlan(descriptor(), 2, 5)
    with pytest.raises(ValueError, match="budget"):
        plan.specialize([("int64", (17,)), ("int64", (17,))], 8, 63)
    with pytest.raises(NotImplementedError, match="same-shape"):
        plan.specialize([("int64", (17,)), ("int64", (16,))])
    with pytest.raises(ValueError, match="rank"):
        plan.specialize([("int64", (1,) * 17), ("int64", (1,) * 17)])
