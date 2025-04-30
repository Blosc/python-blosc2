Announcing Python-Blosc2 3.3.1
==============================

In our effort to better adapt to better adapt to the array API
(https://data-apis.org/array-api/latest/), we have introduced
permute_dims() and matrix_transpose() functions, and the .T property.
This replaces to previous transpose() function, which is now deprecated.
See PR #384.  Thanks to Ricardo Sales Piquer (@ricardosp4).

We have also reduced the memory footprint of constructors like ``arange()``,
``linspace()`` and ``fromiter()`` by a large factor. As an example, a 5 TB
array of 8-byte floats now uses less than 200 MB of memory instead of
170 GB previously.

You can think of Python-Blosc2 3.x as an extension of NumPy/numexpr that:

- Can deal with ndarrays compressed using first-class codecs & filters.
- Performs many kind of math expressions, including reductions, indexing...
- Supports broadcasting operations.
- Supports NumPy ufunc mechanism: mix and match NumPy and Blosc2 computations.
- Integrates with Numba and Cython via UDFs (User Defined Functions).
- Adheres to modern NumPy casting rules way better than numexpr.
- Supports linear algebra operations (like ``blosc2.matmul()``).

Install it with::

    pip install blosc2 --update   # if you prefer wheels
    conda install -c conda-forge python-blosc2 mkl  # if you prefer conda and MKL

For more info, you can have a look at the release notes in:

https://github.com/Blosc/python-blosc2/releases

Code example::

    from time import time
    import blosc2
    import numpy as np

    # Create some data operands
    N = 20_000
    a = blosc2.linspace(0, 1, N * N, dtype="float32", shape=(N, N))
    b = blosc2.linspace(1, 2, N * N, shape=(N, N))
    c = blosc2.linspace(-10, 10, N)  # broadcasting is supported

    # Expression
    t0 = time()
    expr = ((a**3 + blosc2.sin(c * 2)) < b) & (c > 0)
    print(f"Time to create expression: {time()-t0:.5f}")

    # Evaluate while reducing (yep, reductions are in) along axis 1
    t0 = time()
    out = blosc2.sum(expr, axis=1)
    t1 = time() - t0
    print(f"Time to compute with Blosc2: {t1:.5f}")

    # Evaluate using NumPy
    na, nb, nc = a[:], b[:], c[:]
    t0 = time()
    nout = np.sum(((na**3 + np.sin(nc * 2)) < nb) & (nc > 0), axis=1)
    t2 = time() - t0
    print(f"Time to compute with NumPy: {t2:.5f}")
    print(f"Speedup: {t2/t1:.2f}x")

    assert np.all(out == nout)
    print("All results are equal!")


This will output something like (using an Intel i9-13900X CPU here)::

    Time to create expression: 0.00033
    Time to compute with Blosc2: 0.46387
    Time to compute with NumPy: 2.57469
    Speedup: 5.55x
    All results are equal!

See a more in-depth example, explaining why Python-Blosc2 is so fast, at:

https://www.blosc.org/python-blosc2/getting_started/overview.html#operating-with-ndarrays

Sources repository
------------------

The sources and documentation are managed through github services at:

https://github.com/Blosc/python-blosc2

Python-Blosc2 is distributed using the BSD license, see
https://github.com/Blosc/python-blosc2/blob/main/LICENSE.txt
for details.

Mastodon feed
-------------

Follow https://fosstodon.org/@Blosc2 to get informed about the latest
developments.

Enjoy!

- Blosc Development Team
  Compress better, compute bigger
