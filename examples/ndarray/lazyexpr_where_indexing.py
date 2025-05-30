# Imports

import numpy as np

import blosc2

N = 1000
it = ((-x + 1, x - 2, 0.1 * x) for x in range(N))
sa = blosc2.fromiter(
    it, dtype=[("A", "i4"), ("B", "f4"), ("C", "f8")], shape=(N,), urlpath="sa-1M.b2nd", mode="w"
)
expr = sa["(A < B)"]
A = sa["A"][:]
B = sa["B"][:]
C = sa["C"][:]
temp = sa[:]
indices = A < B
idx = np.argmax(indices)

# One might think that expr[:10] gives the first 10 elements of the evaluated expression, but this is not the case.
# It actually computes the expression on the first 10 elements of the operands; since for some elements the condition
# is False, the result will be shorter than 10 elements.
# Returns less than 10 elements in general
sliced = expr.compute(slice(0, 10))
gotitem = expr[:10]
np.testing.assert_array_equal(sliced[:], gotitem)
np.testing.assert_array_equal(gotitem, temp[:10][indices[:10]])  # Equivalent syntax
# Actually this makes sense since one can understand this as a request to compute on a portion of operands.
# If one desires a portion of the result, one should compute the whole expression and then slice it.

# Get first element for which condition is true
sliced = expr.compute(idx)
gotitem = expr[idx]
# Arrays of one element
np.testing.assert_array_equal(sliced[()], gotitem)
np.testing.assert_array_equal(gotitem, temp[idx])

# Should return void arrays here.
sliced = expr.compute(0)
gotitem = expr[0]
np.testing.assert_array_equal(sliced[()], gotitem)
np.testing.assert_array_equal(gotitem, temp[0])
