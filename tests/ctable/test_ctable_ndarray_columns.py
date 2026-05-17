from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int32())
    embedding: object = blosc2.field(blosc2.ndarray((3,), dtype=blosc2.float32()))
    image: object = blosc2.field(blosc2.ndarray((2, 2, 3), dtype=blosc2.float32()))


DATA = [
    (1, np.array([1, 2, 3], dtype=np.float32), np.ones((2, 2, 3), dtype=np.float32)),
    (2, np.array([4, 5, 6], dtype=np.float32), np.full((2, 2, 3), 2, dtype=np.float32)),
]


def table():
    return blosc2.CTable(Row, new_data=DATA)


def test_ndarray_column_metadata_and_tuple_indexing():
    t = table()

    assert t.embedding.shape == (2, 3)
    assert t.embedding.item_shape == (3,)
    assert t.embedding.ndim == 2
    assert t.embedding.item_ndim == 1
    assert t.embedding.size == 6
    assert t.embedding.item_size == 3

    np.testing.assert_array_equal(t.embedding[:, 0], np.array([1, 4], dtype=np.float32))
    np.testing.assert_array_equal(t.embedding[1, :2], np.array([4, 5], dtype=np.float32))
    np.testing.assert_array_equal(t.image[:, :, :, 0], np.stack([np.ones((2, 2)), np.full((2, 2), 2)]))


def test_ndarray_column_comparison_and_scalar_operation_guards():
    t = table()

    with pytest.raises(TypeError, match="Cannot compare ndarray column 'embedding' directly"):
        _ = t.embedding > 0
    with pytest.raises(TypeError, match="String expressions only support scalar columns"):
        t.where("embedding > 0")
    with pytest.raises(TypeError, match="Cannot sort by ndarray column 'embedding'"):
        t.sort_by("embedding")
    with pytest.raises(TypeError, match="Cannot group by ndarray column 'embedding'"):
        t.group_by("embedding")
    with pytest.raises(ValueError, match="Cannot create an index on ndarray column 'embedding'"):
        t.create_index("embedding")


def test_ndarray_column_axis_reductions_and_where_projection():
    t = table()

    assert t.embedding.sum() == np.float32(21)
    np.testing.assert_array_equal(t.embedding.sum(axis=0), np.array([5, 7, 9], dtype=np.float32))
    np.testing.assert_array_equal(t.embedding.sum(axis=1), np.array([6, 15], dtype=np.float32))
    np.testing.assert_allclose(t.embedding.norm(axis=1), np.linalg.norm(t.embedding[:], axis=1))

    filtered = t.where(t.embedding[:, 0] > 1)
    np.testing.assert_array_equal(filtered.id[:], np.array([2], dtype=np.int32))


def test_generated_column_row_transformer_append_refresh_and_vector_output():
    t = table()

    t.add_generated_column(
        "embedding_norm",
        values=t.embedding.row_transformer.norm(axis=0),
        dtype=blosc2.float64(),
        create_index=True,
    )
    np.testing.assert_allclose(t.embedding_norm[:], np.linalg.norm(t.embedding[:], axis=1))

    t.append((3, np.array([0, 3, 4], dtype=np.float32), np.zeros((2, 2, 3), dtype=np.float32)))
    np.testing.assert_allclose(t.embedding_norm[:], np.linalg.norm(t.embedding[:], axis=1))

    t.embedding[0] = np.array([0, 0, 0], dtype=np.float32)
    assert t._materialized_cols["embedding_norm"]["stale"] is True
    t.refresh_generated_column("embedding_norm")
    np.testing.assert_allclose(t.embedding_norm[:], np.linalg.norm(t.embedding[:], axis=1))

    t.add_generated_column(
        "image_mean_rgb",
        values=t.image.row_transformer.mean(axis=(0, 1)),
        dtype=blosc2.ndarray((3,), dtype=blosc2.float32()),
    )
    np.testing.assert_allclose(t.image_mean_rgb[:], t.image[:].mean(axis=(1, 2)))


def test_stale_generated_column_raises_and_read_stale_escape_hatch():
    t = table()
    t.add_generated_column(
        "embedding_sum",
        values=t.embedding.row_transformer.sum(axis=0),
        dtype=blosc2.float64(),
    )
    t.add_generated_column(
        "score",
        values="embedding_sum + sin(id)",
        dtype=blosc2.float64(),
    )

    old_sum = t.embedding_sum[:].copy()
    t.embedding[0] = np.array([10, 20, 30], dtype=np.float32)

    assert t._materialized_cols["embedding_sum"]["stale"] is True
    assert t._materialized_cols["score"]["stale"] is True
    with pytest.raises(ValueError, match="read_stale"):
        _ = t.embedding_sum[:]
    with pytest.raises(ValueError, match="read_stale"):
        _ = t.score.sum()
    np.testing.assert_array_equal(t.embedding_sum.read_stale(), old_sum)

    t.refresh_generated_columns(source="embedding")
    np.testing.assert_allclose(t.embedding_sum[:], t.embedding[:].sum(axis=1))
    np.testing.assert_allclose(t.score[:], t.embedding_sum[:] + np.sin(t.id[:]))
