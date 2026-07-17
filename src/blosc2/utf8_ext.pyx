#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
# cython: boundscheck=False, wraparound=False, initializedcheck=False

"""Bulk StringDType construction for utf8 columns.

NumPy provides no bulk constructor for a ``StringDType`` array from an
offsets+bytes buffer pair -- the only Python-level ways in are per-element
assignment or conversion from another array.  This kernel uses NumPy's C
API for ``StringDType`` (``NpyString_pack``) to fill every element of a
preallocated ``StringDType`` array directly from the raw offsets/bytes
representation, in a single C loop with no per-row Python object churn.
"""

import numpy as np
cimport numpy as cnp

from libc.stdint cimport int64_t, uint8_t

cnp.import_array()


# Declared here instead of relying on `cimport numpy`: the NpyString C API
# has been part of the NumPy headers since 2.0, but its Cython declarations
# only appear in the numpy/__init__.pxd of newer NumPy versions, and some
# build environments (e.g. the Pyodide cross-build) pin an older one.  The
# functions resolve through the API table populated by cnp.import_array().
cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct npy_string_allocator:
        pass
    ctypedef struct npy_packed_static_string:
        pass
    ctypedef struct PyArray_StringDTypeObject:
        pass

cdef extern from "numpy/arrayobject.h":
    npy_string_allocator* NpyString_acquire_allocator(
        const PyArray_StringDTypeObject* descr
    ) nogil
    void NpyString_release_allocator(npy_string_allocator* allocator) nogil
    int NpyString_pack(
        npy_string_allocator* allocator,
        npy_packed_static_string* packed_string,
        const char* buf,
        size_t size,
    ) nogil


def pack_utf8_span(cnp.ndarray rel not None, cnp.ndarray data not None, cnp.ndarray out not None):
    """Fill *out* in place with rows carved out of *data* using *rel*.

    *rel* is ``int64``, length ``len(out) + 1``, ``rel[0] == 0``: the
    relative byte offset of each row within *data*.  *data* is ``uint8``
    and holds valid UTF-8 bytes (this packs bytes, it does not validate
    the encoding).  *out* is a ``numpy.dtypes.StringDType`` array of length
    ``len(rel) - 1``, already allocated by the caller.
    """
    if rel.ndim != 1 or data.ndim != 1 or out.ndim != 1:
        raise ValueError("rel, data and out must be 1-D arrays")
    cdef Py_ssize_t n = out.shape[0]
    if rel.shape[0] != n + 1:
        raise ValueError("rel must have length len(out) + 1")
    if rel.dtype != np.dtype(np.int64):
        raise TypeError("rel must have dtype int64")
    if data.dtype != np.dtype(np.uint8):
        raise TypeError("data must have dtype uint8")
    if not (rel.flags["C_CONTIGUOUS"] and data.flags["C_CONTIGUOUS"] and out.flags["C_CONTIGUOUS"]):
        raise ValueError("rel, data and out must be C-contiguous")
    if n == 0:
        return

    cdef const int64_t* rel_ptr = <const int64_t*>cnp.PyArray_DATA(rel)
    cdef const uint8_t* data_ptr = <const uint8_t*>cnp.PyArray_DATA(data)
    cdef char* out_data = <char*>cnp.PyArray_DATA(out)
    cdef cnp.npy_intp itemsize = cnp.PyArray_ITEMSIZE(out)
    cdef npy_string_allocator* allocator = NpyString_acquire_allocator(
        <const PyArray_StringDTypeObject*>cnp.PyArray_DESCR(out)
    )
    if allocator == NULL:
        raise TypeError("out must be a StringDType array")

    cdef Py_ssize_t i
    cdef int64_t start, length
    cdef int ret = 0
    try:
        for i in range(n):
            start = rel_ptr[i]
            length = rel_ptr[i + 1] - start
            ret = NpyString_pack(
                allocator,
                <npy_packed_static_string*>(out_data + i * itemsize),
                <const char*>(data_ptr + start),
                <size_t>length,
            )
            if ret == -1:
                break
    finally:
        NpyString_release_allocator(allocator)

    if ret == -1:
        raise MemoryError("Failed to pack a UTF-8 row into the StringDType array")
