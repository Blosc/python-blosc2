import numpy as np
import pytest
from test_portable_descriptor import manifest

import blosc2
from blosc2 import blosc2_ext

pytestmark = pytest.mark.skipif(
    not getattr(blosc2_ext, "portable_descriptor_available", lambda: False)(),
    reason="Installed runtime has no draft descriptor ABI",
)


def kernel(source, *, scalar=False, nd=False, string=False):
    dtype = "unicode32" if string else "int64"
    inputs = [{"name": "x", "dtype": dtype}]
    output = {"dtype": dtype, "contract": "block_scalar" if scalar else "elementwise"}
    requires = ["numeric"]
    if string:
        inputs[0]["itemsize"] = output["itemsize"] = 16
        requires.append("fixed-strings")
    if scalar:
        requires.append("block-reductions")
    if nd:
        requires.append("nd-context")
    return blosc2.PortableKernel.from_json(
        manifest(source, inputs, output, requires=requires, ndim=2 if nd else 0), jit=False
    )


@pytest.mark.parametrize("mutation", ["kernel", "handle", "metadata", "artifact", "native_rebinding"])
def test_safe_graph_checks_portable_kernel_closure_before_hooks(monkeypatch, mutation):
    native = kernel("def k(x):\n    return x + 1\n")
    lazy = native.lazy({"x": blosc2.asarray(np.arange(4, dtype="int64"))})
    expr = blosc2.lazyexpr("x + 1", {"x": lazy}, evaluation="safe")
    np.testing.assert_array_equal(expr[:], np.arange(4) + 2)

    class Hostile:
        @property
        def output_dtype(self):
            pytest.fail("Read hostile portable output dtype")

        def __eq__(self, other):
            pytest.fail("Compared hostile portable metadata")

    if mutation == "kernel":
        monkeypatch.setattr(lazy, "kernel", Hostile())
    elif mutation == "handle":
        monkeypatch.setattr(native, "_handle", Hostile())
    elif mutation == "metadata":
        monkeypatch.setitem(native._info, "output_dtype", Hostile())
    elif mutation == "artifact":
        monkeypatch.setattr(native, "_artifact", b"{}")
    else:
        other = kernel("def k(x):\n    return x + 2\n")
        monkeypatch.setattr(native, "_handle", other._handle)
    with pytest.raises((blosc2.UnsafeDeserializationError, ValueError)):
        expr.compute()


@pytest.mark.parametrize("mutation", ["mapping", "domain", "partitions", "shape", "grid", "input_dtype"])
def test_safe_graph_checks_portable_domain_and_binding_metadata(monkeypatch, mutation):
    lazy = kernel("def k(x):\n    return x + 1\n").lazy(
        {"x": blosc2.asarray(np.arange(4, dtype="int64"))}, partitions=(2,)
    )
    expr = blosc2.lazyexpr("x + 1", {"x": lazy}, evaluation="safe")

    class HostileMapping:
        def values(self):
            pytest.fail("Read hostile portable input mapping")

    if mutation == "mapping":
        monkeypatch.setattr(lazy, "inputs", HostileMapping())
    elif mutation == "domain":
        monkeypatch.setattr(lazy, "_domain", (object(),))
    elif mutation == "partitions":
        monkeypatch.setattr(lazy, "_partitions", (0,))
    elif mutation == "shape":
        monkeypatch.setattr(lazy, "_shape", (5,))
    elif mutation == "grid":
        monkeypatch.setattr(lazy, "_grid", (5,))
    else:
        monkeypatch.setitem(lazy.inputs, "x", blosc2.asarray(np.arange(4, dtype="float64")))
    with pytest.raises((blosc2.UnsafeDeserializationError, ValueError)):
        expr.compute()


@pytest.mark.parametrize("scalar", [False, True])
def test_logical_groups_partial_and_rechunk(tmp_path, scalar):
    values = np.arange(35, dtype="int64").reshape(5, 7)
    stored = blosc2.asarray(values, chunks=(3, 4), blocks=(1, 2), urlpath=tmp_path / "input.b2nd")
    k = kernel(
        "def k(x):\n    return sum(x)\n" if scalar else "def k(x):\n    return x + _flat_idx\n",
        scalar=scalar,
        nd=not scalar,
    )
    lazy = k.lazy({"x": stored}, partitions=(2, 3))
    expected = (
        np.array([[values[i : i + 2, j : j + 3].sum() for j in range(0, 7, 3)] for i in range(0, 5, 2)])
        if scalar
        else values * 2
    )
    assert lazy.partitions == (2, 3)
    np.testing.assert_array_equal(lazy[:], expected)
    np.testing.assert_array_equal(lazy[1:, ::2], expected[1:, ::2])
    np.testing.assert_array_equal(
        lazy.compute(item=(slice(1, None), slice(None, None, 2)))[:], expected[1:, ::2]
    )
    np.testing.assert_array_equal(lazy.rechunk(chunks=(1, 2), blocks=(1, 1))[:], expected)
    path = tmp_path / "lazy.b2nd"
    lazy.save(path)
    reopened = blosc2.open(path)
    assert reopened.partitions == (2, 3)
    np.testing.assert_array_equal(reopened[:], expected)
    mixed = lazy + 3
    mixed.save(tmp_path / "mixed.b2nd")
    np.testing.assert_array_equal(blosc2.open(tmp_path / "mixed.b2nd")[:], expected + 3)


def test_string_roundtrip_and_native_only_load(tmp_path, monkeypatch):
    values = np.array(["ß a", "ab", "z"], dtype="U4")
    stored = blosc2.asarray(values, urlpath=tmp_path / "text.b2nd")
    lazy = kernel("def k(x):\n    return upper(x)\n", string=True).lazy({"x": stored}, partitions=(2,))
    lazy.save(tmp_path / "textlazy.b2nd")
    import importlib

    dsl_module = importlib.import_module("blosc2.dsl_kernel")
    monkeypatch.setattr(
        dsl_module, "kernel_from_source", lambda *a, **k: pytest.fail("Python reconstruction")
    )
    import blosc2.b2objects as b2objects

    monkeypatch.setattr(
        b2objects, "kernel_from_source", lambda *a, **k: pytest.fail("Python reconstruction")
    )
    np.testing.assert_array_equal(blosc2.open(tmp_path / "textlazy.b2nd")[:], ["SS A", "AB", "Z"])


def test_preflight_preserves_destination(tmp_path):
    lazy = kernel("def k(x):\n    return x + 1\n").lazy({"x": np.arange(3, dtype="int64")})
    path = tmp_path / "existing"
    path.write_bytes(b"sentinel")
    with pytest.raises((TypeError, ValueError)):
        lazy.save(path)
    assert path.read_bytes() == b"sentinel"


@pytest.mark.parametrize("route", ["frame", "msgpack", "embed", "dict", "tree"])
def test_container_routes(tmp_path, route):
    from blosc2.msgpack_utils import msgpack_packb, msgpack_unpackb

    source = blosc2.asarray(np.arange(9, dtype="int64"), urlpath=tmp_path / "source.b2nd")
    lazy = kernel("def k(x):\n    return sum(x)\n", scalar=True).lazy({"x": source}, partitions=(4,))
    expected = np.array([6, 22, 8])
    if route == "frame":
        reopened = blosc2.from_cframe(lazy.to_cframe())
    elif route == "msgpack":
        reopened = msgpack_unpackb(msgpack_packb(lazy), deserialize="safe")
    else:
        cls = {"embed": blosc2.EmbedStore, "dict": blosc2.DictStore, "tree": blosc2.TreeStore}[route]
        with cls(str(tmp_path / "store.b2z"), mode="w") as store:
            store["/lazy"] = lazy
            reopened = store["/lazy"]
            np.testing.assert_array_equal(reopened[:], expected)
        with cls(str(tmp_path / "store.b2z"), mode="r") as store:
            reopened = store["/lazy"]
            np.testing.assert_array_equal(reopened[:], expected)
        return
    np.testing.assert_array_equal(reopened[:], expected)


def test_partial_reads_only_original_groups():
    class Operand:
        dtype = np.dtype("int64")
        shape = (12,)

        def __init__(self):
            self.reads = []

        def __getitem__(self, item):
            self.reads.append(item)
            return np.arange(12, dtype="int64")[item]

    operand = Operand()
    lazy = kernel("def k(x):\n    return sum(x)\n", scalar=True).lazy({"x": operand}, partitions=(4,))
    np.testing.assert_array_equal(lazy[1:2], [22])
    assert operand.reads == [(slice(4, 8),)]
    assert lazy[1] == 22
    np.testing.assert_array_equal(lazy[::-1], [38, 22, 6])
    author = blosc2.DSLKernel.from_source("def k(x):\n    return x + sum(x)\n")
    elementwise = blosc2.PortableKernel.from_json(author.export({"x": "int64"}, "int64", version="1.0"))
    grouped = elementwise.lazy({"x": operand}, partitions=(4,))
    operand.reads.clear()
    np.testing.assert_array_equal(grouped[7:3:-2], [29, 27])
    assert operand.reads == [(slice(4, 8),)]
    np.testing.assert_array_equal(grouped[4:4], np.array([], dtype="int64"))
    assert operand.reads == [(slice(4, 8),)]


def test_broadcast_coordinates_keep_output_domain():
    values = np.arange(7, dtype="int64").reshape(1, 7)
    k = kernel("def k(x):\n    return x + _flat_idx\n", nd=True)
    lazy = k.lazy({"x": values}, shape=(5, 7), partitions=(2, 3))
    expected = values + np.arange(35).reshape(5, 7)
    np.testing.assert_array_equal(lazy[3:, 4:], expected[3:, 4:])
