"""Exchange b2o v2 payloads with the dependency-free native C fixture peer."""

import copy
import os
import subprocess

import msgpack
import numpy as np
import pytest

import blosc2
from blosc2.b2objects import decode_b2object_payload, encode_b2object_payload
from blosc2.msgpack_utils import msgpack_packb, msgpack_unpackb

PEER = os.environ.get("BLOSC2_MENUDET_INTEROP")
pytestmark = pytest.mark.skipif(not PEER, reason="Set BLOSC2_MENUDET_INTEROP to recipe_v2_peer")


def ramp():
    recipe = {
        "version": 1,
        "language_version": "0.2",
        "source": "def ramp(start: float64) -> float64:\n    return start + _flat_idx\n",
        "entry_point": "ramp",
        "inputs": [],
        "constants": [
            {
                "name": "start",
                "dtype": "float64",
                "encoding": "ieee754-hex",
                "value": "3ff0000000000001",
                "category": "typed_scalar",
            }
        ],
        "output": None,
        "cardinality": "elementwise",
        "ndim": 1,
        "casting": "unsafe",
    }
    return blosc2.MenudetKernel.from_recipe(recipe, {}, jit=False).lazy({}, shape=(4,), partitions=(2,))


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_c_write_python_read(tmp_path, mode):
    path = tmp_path / "c.msgpack"
    subprocess.run([PEER, "write", str(path), str(mode)], check=True)
    raw = path.read_bytes()
    if mode == 2:
        restored = msgpack_unpackb(raw)
    else:
        restored = decode_b2object_payload(msgpack.unpackb(raw, raw=False))
    expected = np.arange(1, 5, dtype="float64") * (2 if mode else 1)
    np.testing.assert_array_equal(restored[:], expected)
    # Put the actual C-authored recipe into a standard Blosc2 carrier, then
    # exercise the fixed marker, file reopening and CFrame paths as well.
    if mode != 2:
        from blosc2.b2objects import make_b2object_carrier, write_b2object_payload

        carrier = make_b2object_carrier(
            "menudet",
            (4,),
            np.float64,
            chunks=(4,),
            blocks=(2,),
            urlpath=str(tmp_path / "c.b2nd"),
            mode="w",
        )
        write_b2object_payload(carrier, msgpack.unpackb(raw, raw=False))
        np.testing.assert_array_equal(blosc2.open(tmp_path / "c.b2nd")[:], expected)
        np.testing.assert_array_equal(blosc2.from_cframe(carrier.to_cframe())[:], expected)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_python_write_c_read(tmp_path, mode):
    lazy = ramp()
    if mode:
        # The unannotated input deliberately stays generic in the recipe; C
        # specializes it using the nested operand's actual output dtype.
        recipe = {
            "version": 1,
            "language_version": "0.2",
            "source": "def twice(x) -> float64:\n    return 2 * x\n",
            "entry_point": "twice",
            "inputs": ["x"],
            "constants": [],
            "output": None,
            "cardinality": "elementwise",
            "ndim": 0,
            "casting": "unsafe",
        }
        kernel = blosc2.MenudetKernel.from_recipe(recipe, {"x": "float64"}, jit=False)
        lazy = kernel.lazy({"x": lazy}, partitions=(2,))
    path = tmp_path / "python.msgpack"
    path.write_bytes(
        msgpack_packb(lazy) if mode == 2 else msgpack.packb(encode_b2object_payload(lazy), use_bin_type=True)
    )
    result = subprocess.run([PEER, "read", str(path)], check=True, capture_output=True, text=True)
    np.testing.assert_array_equal(np.fromstring(result.stdout, sep="\n"), lazy[:])


@pytest.mark.parametrize("change", ["version", "shape", "capture"])
def test_c_peer_rejects_unsupported_subset(tmp_path, change):
    payload = copy.deepcopy(encode_b2object_payload(ramp()))
    if change == "version":
        payload["version"] = 99
    elif change == "shape":
        payload["shape"] = [5]
    else:
        payload["recipe"]["constants"][0]["dtype"] = "float32"
    path = tmp_path / "invalid.msgpack"
    path.write_bytes(msgpack.packb(payload, use_bin_type=True))
    result = subprocess.run([PEER, "read", str(path)], capture_output=True, text=True)
    assert result.returncode != 0
