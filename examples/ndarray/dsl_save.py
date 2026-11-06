#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Demonstrate saving and reloading Menudet recipes.
#
# We blend a 2-D field with a source term. The portable artifact is
# saved to disk and reloaded without reconstructing a Python function.

import numpy as np

import blosc2

shape = (128, 128)

# ── operand arrays built with native Blosc2 constructors ─────────────
# Persist on disk so they can be referenced from the saved LazyUDF.
u = blosc2.linspace(0.0, 1.0, shape=shape, dtype=np.float64, urlpath="u.b2nd", mode="w")
vexpr = blosc2.sin(blosc2.linspace(0.0, 2 * np.pi, shape=shape, dtype=np.float64))
v = vexpr.compute(urlpath="v.b2nd", mode="w")


# ── DSL kernel: one explicit Jacobi-style stencil step ──────────────
# `u` holds the current temperature field; `v` is a source/sink term.
# The kernel operates element-wise on flat chunks; index-based
# neighbour access is intentionally avoided here so the expression
# stays in the simple DSL subset that miniexpr can JIT.
@blosc2.menudet
def heat_step(u, v):
    # Weighted blend: 0.25*(left+right+up+down) approximated element-wise
    # by mixing u and a scaled source term – keeps the kernel portable
    # while still exercising non-trivial arithmetic.
    alpha = 0.1
    return u + alpha * (v - u)


# ── build and save the lazy computation ────────────────────────────
lazy = heat_step(u, v)
lazy.save(urlpath="heat_step.b2nd")
print("Menudet recipe saved to heat_step.b2nd")

# ── reload in a 'fresh' context (no reference to heat_step) ─────────
reloaded = blosc2.open("heat_step.b2nd", mode="r")
assert isinstance(reloaded.kernel, blosc2.MenudetKernel)
print(f"Reloaded Menudet artifact:\n{reloaded.kernel.to_json()}\n")

# ── evaluate and verify ──────────────────────────────────────────────
result = reloaded.compute()
expected = u[()] + 0.1 * (v[()] - u[()])
assert np.allclose(result[()], expected), "Numerical mismatch after reload!"
print("Max absolute error vs NumPy reference:", np.max(np.abs(result[()] - expected)))

# ── chain two steps: save the first result and run a second step ─────
u2 = result.copy(urlpath="u2.b2nd", mode="w")

lazy2 = heat_step(u2, v)
lazy2.save(urlpath="heat_step2.b2nd")

reloaded2 = blosc2.open("heat_step2.b2nd", mode="r")
result2 = reloaded2.compute()
expected2 = u2[()] + 0.1 * (v[()] - u2[()])
assert np.allclose(result2[()], expected2)
print("Two-step heat diffusion matches NumPy reference. ✓")

# ── getitem also works on the reloaded kernel (full-array access) ────
full_result = reloaded[()]
assert np.allclose(full_result, expected)
print("Full-array getitem on reloaded LazyUDF works correctly. ✓")

# ── tidy up ─────────────────────────────────────────────────────────
for path in ["u.b2nd", "v.b2nd", "u2.b2nd", "heat_step.b2nd", "heat_step2.b2nd"]:
    blosc2.remove_urlpath(path)

print("\nDSL kernel save/reload demo completed successfully!")
