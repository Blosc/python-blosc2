#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np

import blosc2
from blosc2 import where


def show(label, value):
    print(f"{label}: {value}")


@blosc2.dsl_kernel
def masked_energy(a, b, mask):
    return where(mask > 0, a * a + 2 * b, 0.0)


bundle_path = "example_b2o_bundle.b2z"
blosc2.remove_urlpath(bundle_path)

# Build a portable bundle with ordinary arrays plus persisted lazy recipes.
with blosc2.DictStore(bundle_path, mode="w", threshold=1) as store:
    store["/data/a"] = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    store["/data/b"] = np.linspace(1.0, 2.0, 10, dtype=np.float32)
    store["/data/mask"] = np.asarray([0, 1, 1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int8)

    # Reopen the array members through the store so operand refs point back to
    # the .b2z container via dictstore_key payloads.
    a = store["/data/a"]
    b = store["/data/b"]
    mask = store["/data/mask"]

    expr = blosc2.lazyexpr("(a - b) / (a + b + 1e-6)", operands={"a": a, "b": b})
    udf = blosc2.lazyudf(masked_energy, (a, b, mask), dtype=np.float32, shape=a.shape)

    # DictStore currently stores array-like external leaves, so persist the
    # logical lazy objects through their b2o NDArray carriers.
    store["/recipes/expr"] = blosc2.ndarray_from_cframe(expr.to_cframe())
    store["/recipes/udf"] = blosc2.ndarray_from_cframe(udf.to_cframe())

show("Bundle created", bundle_path)

# Reopen the bundle read-only.  The persisted LazyExpr and LazyUDF can be
# evaluated directly without re-saving, rebuilding, or re-deploying the .b2z.
with blosc2.open(bundle_path, mode="r") as store:
    show("Read-only keys", sorted(store.keys()))

    expr = store["/recipes/expr"]
    udf = store["/recipes/udf"]

    expr_result = expr.compute()
    udf_result = udf.compute()

    show("Reopened expr type", type(expr).__name__)
    show("Reopened udf type", type(udf).__name__)
    show("Expr operand refs", expr.array.schunk.vlmeta["b2o"]["operands"])
    show("UDF operand refs", udf.array.schunk.vlmeta["b2o"]["operands"])
    show("Expr values", np.round(expr_result[:], 4))
    show("UDF values", udf_result[:])

    expected_expr = (store["/data/a"][:] - store["/data/b"][:]) / (
        store["/data/a"][:] + store["/data/b"][:] + 1e-6
    )
    expected_udf = np.where(
        store["/data/mask"][:] > 0,
        store["/data/a"][:] ** 2 + 2 * store["/data/b"][:],
        0.0,
    ).astype(np.float32)
    np.testing.assert_allclose(expr_result[:], expected_expr, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(udf_result[:], expected_udf, rtol=1e-6, atol=1e-6)
    show("Read-only evaluation", "ok")
