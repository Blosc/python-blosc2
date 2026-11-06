"""The same arithmetic as a LazyExpr and a decorated Menudet kernel."""

import numpy as np

import blosc2


@blosc2.menudet
def adjusted(x, y):
    return x + (y - 1)


@blosc2.menudet
def typed_adjusted(x: np.float64, y: np.float64) -> np.float64:
    return x + (y - 1)


x = np.array([1, 2, 3], dtype="float64")
y = np.array([4, 5, 6], dtype="float64")
expr = blosc2.lazyexpr("x + (y - 1)", {"x": x, "y": y})
print("LazyExpr:", expr[:])
print("Menudet:", adjusted(x, y)[:])
print("Annotated Menudet:", typed_adjusted(x, y)[:])
print("Menudet JIT:", adjusted.specialize(x, y).has_jit)
