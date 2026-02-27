#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Demonstrate malformed DSL diagnostics and enriched lazyexpr error feedback.

import importlib

import numpy as np

import blosc2
from blosc2.dsl_kernel import DSLSyntaxError


# --- 1) Malformed DSL syntax: validate_dsl() and lazyudf() diagnostics ---
@blosc2.dsl_kernel
def kernel_bad_ternary(x):
    return 1 if x else 0


report = blosc2.validate_dsl(kernel_bad_ternary)
print("validate_dsl valid:", report["valid"])
print("validate_dsl error:\n", report["error"])

try:
    x = blosc2.ones((8, 8), dtype=np.float32)
    _ = blosc2.lazyudf(kernel_bad_ternary, (x,), dtype=np.int32)
except DSLSyntaxError as e:
    print("\nlazyudf rejected malformed DSL kernel as expected:\n", e)


# --- 2) Force miniexpr backend failure to show enriched RuntimeError message ---
@blosc2.dsl_kernel
def kernel_ok(x, y):
    return x + y


lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
old_try_miniexpr = lazyexpr_mod.try_miniexpr
old_set_pref_expr = blosc2.NDArray._set_pref_expr


def failing_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc=None, jit=None):
    raise ValueError("forced backend failure from malformed_dsl.py demo")


try:
    # Keep miniexpr enabled so the failing hook is exercised.
    lazyexpr_mod.try_miniexpr = True
    blosc2.NDArray._set_pref_expr = failing_set_pref_expr

    a = blosc2.ones((16, 16), dtype=np.float32)
    b = blosc2.full((16, 16), 2.0, dtype=np.float32)
    expr = blosc2.lazyudf(kernel_ok, (a, b), dtype=np.float32)

    try:
        _ = expr.compute()
    except RuntimeError as e:
        print("\nRuntimeError from DSL miniexpr path (expected in this demo):")
        print(e)
finally:
    lazyexpr_mod.try_miniexpr = old_try_miniexpr
    blosc2.NDArray._set_pref_expr = old_set_pref_expr
