#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# CTable N-dimensional columns: each cell can hold a full compressed
# multidimensional array — embeddings, thumbnails, time-series windows,
# or any per-row tensor payload.  This example covers construction,
# multi-axis slicing, per-row reductions, generated columns, filtering
# on item dimensions, and nullability.

from dataclasses import dataclass

import numpy as np

import blosc2

rng = np.random.default_rng(123)


@dataclass
class Product:
    product_id: int = blosc2.field(blosc2.int32())
    name: str = blosc2.field(blosc2.string(max_length=32))
    # 768-D text embedding (LLM style)
    embedding: object = blosc2.field(blosc2.ndarray((768,), dtype=blosc2.float32()))
    # 64×64×3 RGB product image
    thumbnail: object = blosc2.field(blosc2.ndarray((64, 64, 3), dtype=blosc2.float32()))
    # 168-hour (7-day) sales time-series, nullable: missing when product is new
    weekly_sales: object = blosc2.field(blosc2.ndarray((168,), dtype=blosc2.float32(), nullable=True))


# -- Build a small product catalog ------------------------------------------

N = 10
names = [
    "Widget",
    "Gadget",
    "Doodad",
    "Thingy",
    "Gizmo",
    "Whatsit",
    "Doohickey",
    "Contraption",
    "Gimmick",
    "Apparatus",
]

data = []
for i in range(N):
    emb = rng.normal(0, 0.1, size=(768,)).astype(np.float32)
    thumb = rng.random((64, 64, 3)).astype(np.float32)
    # First two products are new — no sales history yet
    sales = None if i < 2 else rng.poisson(lam=50, size=(168,)).astype(np.float32)
    data.append((i, names[i], emb, thumb, sales))

t = blosc2.CTable(Product, new_data=data)
print(f"Product catalog: {t.nrows} products, {t.weekly_sales.null_count()} without sales history\n")

# -- Basic properties -------------------------------------------------------

print(f"embedding shape  : {t.embedding.shape}     (nrows × item_shape)")
print(f"embedding item   : {t.embedding.item_shape}")
print(f"thumbnail shape  : {t.thumbnail.shape}")
print(f"weekly_sales     : {t.weekly_sales.item_shape}  ({t.weekly_sales.null_count()} null rows)")

# -- Multi-axis slicing -----------------------------------------------------

# Slice across rows AND item dimensions in a single operation
first_five_dims = t.embedding[0, :5]  # first 5 dims of product 0
three_dims_all = t.embedding[:3, :3]  # first 3 dims of first 3 products → (3, 3)
two_prods_channel = t.thumbnail[[0, -1], 0, 0, :]  # pixel (0,0) RGB for first & last → (2, 3)

print(f"\nembedding[0, :5]             → {first_five_dims}")
print(f"embedding[:3, :3]            → shape {three_dims_all.shape}, values:\n{three_dims_all}")
print(f"thumbnail[[0,-1], 0, 0, :]   → pixel RGB: {two_prods_channel[0]} | {two_prods_channel[1]}")

# -- Per-row reductions -----------------------------------------------------

# Column reductions with axis= (axis 0 is rows, item dims start at 1)
norms = t.embedding.norm(axis=1)  # L2 norm per product
strongest_dim = t.embedding.argmax(axis=1)  # strongest embedding coordinate per product
mean_rgb = t.thumbnail.mean(axis=(1, 2))  # mean RGB per thumbnail

print(f"\nEmbedding L2 norms : {np.round(norms[:], 4)}")
print(f"Strongest embedding dims : {strongest_dim[:]}")
print(f"Mean RGB (first 3) :\n{mean_rgb[:3]}")

# -- Generated columns: materialize transformations -------------------------
# (stays in sync when rows are appended or updated)

t.add_generated_column(
    "embedding_norm",
    values=t.embedding.row_transformer.norm(),
    dtype=blosc2.float64(),
)
t.add_generated_column(
    "dominant_embedding_dim",
    values=t.embedding.row_transformer.argmax(),
    dtype=blosc2.int64(),
)
t.add_generated_column(
    "thumbnail_mean_rgb",
    values=t.thumbnail.row_transformer.mean(axis=(0, 1)),
    dtype=blosc2.ndarray((3,), dtype=blosc2.float32()),
)
print(f"\nGenerated columns: {['embedding_norm', 'dominant_embedding_dim', 'thumbnail_mean_rgb']}")
print(f"embedding_norm[:3]            : {np.round(t.embedding_norm[:3], 4)}")
print(f"dominant_embedding_dim[:3]    : {t.dominant_embedding_dim[:3]}")
print(f"thumbnail_mean_rgb[0]         : {np.round(t.thumbnail_mean_rgb[0], 4)}")

# -- Filtering on item dimensions -------------------------------------------

# Products whose first embedding component is above zero
positive_first = t.where(t.embedding[:, 0] > 0)
print(f"\nProducts with embedding[:, 0] > 0 : {positive_first.nrows}")
print(positive_first[["product_id", "name"]])

# Products whose red channel mean exceeds green (brighter reds than greens)
red_over_green = t.where(t.thumbnail_mean_rgb[:, 0] > t.thumbnail_mean_rgb[:, 1])
print(f"\nProducts with mean R > mean G  : {red_over_green.nrows}")
print(red_over_green[["product_id", "name"]])

# -- Nullable ndarray columns -----------------------------------------------

# Weekly sales are nullable — new products have None
print(f"\nProducts with sales history : {t.weekly_sales.null_count()} null rows")
print(f"Product 0 sales is null     : {np.all(np.isnan(t.weekly_sales[0]))}")

# Skip null rows in reductions
sales_with_history = t.where(~t.weekly_sales.is_null())
# Total and peak-hour sales per product (operate along the 168-hour item axis)
total_sales = sales_with_history.weekly_sales.sum(axis=1)
peak_hour = sales_with_history.weekly_sales.argmax(axis=1)
print(f"Total sales per product     : {total_sales[:]}")
print(f"Peak sales hour per product : {peak_hour[:]}")
