"""String-prefilter binding/group contracts; native string semantics live upstream."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2


@pytest.fixture(autouse=True)
def string_prefilter_runtime(native_graph_runtime):
    probe = blosc2.asarray(np.array(["a"], dtype="U1"), chunks=(8,), blocks=(4,))
    if not blosc2.blosc2_ext.menudet_prefilter_inputs_eligible({"text": probe}, output_dtype="U16"):
        pytest.skip("Installed extension lacks fixed-width string prefilter bindings")


@blosc2.menudet(jit=False)
def grouped_label(text) -> np.dtype("U9"):
    # Deliberately group-dependent: splitting or merging groups changes values.
    if block_sum(int64(text == "a")) > 1:  # noqa: F821
        return "many|" + text
    return "few|" + text


@blosc2.menudet(jit=False, ndim=1)
def contextual_label(text) -> np.dtype("U4"):
    if _flat_idx % 2 == 0:  # noqa: F821
        return text.upper()
    return text.lower()


def grouped_input():
    values = np.array(["a", "a", "b", "b", "a", "b", "b", "b", "a", "a", "b"], dtype="U4")
    return values, blosc2.asarray(values, chunks=(8,), blocks=(4,))


def grouped_reference(values, partition):
    expected = []
    for start in range(0, len(values), partition):
        group = values[start : start + partition]
        prefix = "many|" if np.count_nonzero(group == "a") > 1 else "few|"
        expected.extend(prefix + value for value in group)
    return np.array(expected, dtype="U9")


@pytest.mark.parametrize("clevel", [0, 1])
def test_exact_original_groups_and_tail(clevel):
    values, source = grouped_input()
    kernel = grouped_label.specialize(source)
    assert not kernel.native_partitionable
    lazy = kernel.lazy({"text": source}, partitions=source.blocks)
    result = lazy.compute(cparams=blosc2.CParams(clevel=clevel, nthreads=2))
    np.testing.assert_array_equal(result[:], grouped_reference(values, 4))
    report = lazy._source_execution_report
    assert report["route"] == "menudet-prefilter"
    assert report["threads"] == 2
    assert report["evaluated_blocks"] == 3
    assert result.chunks == source.chunks
    assert result.blocks == source.blocks
    destination = blosc2.empty(source.shape, dtype="U9", chunks=source.chunks, blocks=source.blocks)
    assert lazy.compute(out=destination) is destination
    np.testing.assert_array_equal(destination[:], result[:])


def test_mismatched_partitions_and_output_grid_retain_original_groups():
    values, source = grouped_input()
    kernel = grouped_label.specialize(source)
    lazy = kernel.lazy({"text": source}, partitions=(8,))
    result = lazy.compute(chunks=source.chunks, blocks=source.blocks)
    np.testing.assert_array_equal(result[:], grouped_reference(values, 8))
    assert not hasattr(lazy, "_source_execution_report")
    lazy = kernel.lazy({"text": source}, partitions=source.blocks)
    result = lazy.compute(chunks=(8,), blocks=(2,))
    np.testing.assert_array_equal(result[:], grouped_reference(values, 4))
    assert not hasattr(lazy, "_source_execution_report")


def test_contextual_indices_and_rejection_are_authoritative(monkeypatch):
    values = np.resize(np.array(["ab", "Cd"], dtype="U4"), 11)
    source = blosc2.asarray(values, chunks=(8,), blocks=(4,))
    lazy = contextual_label.specialize(source).lazy({"text": source}, partitions=source.blocks)
    result = lazy.compute()
    expected = np.array(
        [value.upper() if index % 2 == 0 else value.lower() for index, value in enumerate(values.tolist())],
        dtype=values.dtype,
    )
    np.testing.assert_array_equal(result[:], expected)
    assert lazy._source_execution_report["route"] == "menudet-prefilter"
    _, source = grouped_input()
    kernel = grouped_label.specialize(source)
    monkeypatch.setattr(kernel, "graph_partitionable", False, raising=False)
    lazy = kernel.lazy({"text": source}, partitions=source.blocks)
    result = lazy.compute()
    np.testing.assert_array_equal(result[:], grouped_reference(source[:], 4))
    assert not hasattr(lazy, "_source_execution_report")


def test_native_failure_is_not_replayed(monkeypatch):
    _, source = grouped_input()
    kernel = grouped_label.specialize(source)
    handle = kernel._handle
    calls = []

    class FailingHandle:
        def evaluate_prefilter(self, *args, **kwargs):
            calls.append(True)
            handle.evaluate_prefilter(*args, **kwargs)
            raise RuntimeError("failure after native execution")

    lazy = kernel.lazy({"text": source}, partitions=source.blocks)
    monkeypatch.setattr(kernel, "_handle", FailingHandle())
    with pytest.raises(RuntimeError, match="after native execution"):
        lazy.compute()
    assert len(calls) == 1


def test_older_extension_gate_and_concurrent_calls(monkeypatch):
    values, source = grouped_input()
    kernel = grouped_label.specialize(source)

    def compute(_):
        lazy = kernel.lazy({"text": source}, partitions=source.blocks)
        return lazy.compute(cparams=blosc2.CParams(nthreads=2))[:]

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(compute, range(3)))
    for result in results:
        np.testing.assert_array_equal(result, grouped_reference(values, 4))
    monkeypatch.setattr(blosc2.blosc2_ext, "menudet_prefilter_inputs_eligible", lambda *a, **k: False)
    lazy = kernel.lazy({"text": source}, partitions=source.blocks)
    np.testing.assert_array_equal(lazy.compute()[:], grouped_reference(values, 4))
    assert not hasattr(lazy, "_source_execution_report")


def test_wide_unicode_and_byte_string_buffers():
    @blosc2.menudet(jit=False)
    def wide(text) -> np.dtype("U100"):
        return text

    @blosc2.menudet(jit=False)
    def byte_label(text) -> np.dtype("S8"):
        return text.lower()

    @blosc2.menudet(jit=False)
    def first(text) -> np.dtype("U1"):
        return substr(text, 0, 1)  # noqa: F821

    wide_values = np.array(["x" * 100, "Ä" * 99, ""], dtype="U100")
    for author, values in (
        (wide, wide_values),
        (first, wide_values),
        (byte_label, np.array([b"ABC", b"dE", b""], dtype="S8")),
    ):
        source = blosc2.asarray(values, chunks=(8,), blocks=(4,))
        lazy = author.specialize(source).lazy({"text": source}, partitions=source.blocks)
        result = lazy.compute()
        expected = (
            values
            if author is wide
            else np.array([value[:1] for value in values.tolist()], dtype=values.dtype)
            if author is first
            else np.array([value.lower() for value in values.tolist()], dtype=values.dtype)
        )
        np.testing.assert_array_equal(result[:], expected)
        if author is wide:
            # C-Blosc2 represents records wider than 255 bytes with typesize=1;
            # its callback output stride cannot represent these string slots.
            assert not hasattr(lazy, "_source_execution_report")
        else:
            # Wide *inputs* are byte-addressed safely; only result items need a
            # complete callback stride no larger than the C-Blosc2 typesize cap.
            assert lazy._source_execution_report["route"] == "menudet-prefilter"


def test_input_free_string_generator_retains_general_path():
    @blosc2.menudet(jit=False, ndim=1)
    def constant() -> np.dtype("U1"):
        return "x"

    result = constant.specialize().lazy({}, shape=(11,), partitions=(4,)).compute()
    np.testing.assert_array_equal(result[:], np.array(["x"] * 11, dtype="U1"))


def test_native_string_output_cannot_alias_source():
    @blosc2.menudet(jit=False)
    def identity(text) -> np.dtype("U4"):
        return text

    values, source = grouped_input()
    kernel = identity.specialize(source)
    with pytest.raises(ValueError, match="alias"):
        kernel._handle.evaluate_prefilter({"text": source}, out=source)
    np.testing.assert_array_equal(source[:], values)


def test_invalid_unicode_failure_cleans_native_context():
    @blosc2.menudet(jit=False)
    def lower(text) -> np.dtype("U1"):
        return text.lower()

    values = np.array([ord("A"), 0xD800, ord("B")], dtype=np.uint32).view("U1")
    source = blosc2.asarray(values, chunks=(8,), blocks=(4,))
    kernel = lower.specialize(source)
    lazy = kernel.lazy({"text": source}, partitions=source.blocks)
    with pytest.raises(blosc2.MenudetError):
        lazy.compute()
    valid = blosc2.asarray(np.array(["A", "B", ""], dtype="U1"), chunks=(8,), blocks=(4,))
    result = kernel.lazy({"text": valid}, partitions=valid.blocks).compute()
    np.testing.assert_array_equal(result[:], ["a", "b", ""])
