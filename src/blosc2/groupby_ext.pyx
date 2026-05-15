#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
# cython: boundscheck=False, wraparound=False, initializedcheck=False

"""Cython group-reduce kernels for CTable group_by()."""

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t, int64_t


# ----------------------------------------------------------------------
# Group-reduce kernels
# ----------------------------------------------------------------------

def groupby_dense_i32_f64_sum(
    np.ndarray keys,
    np.ndarray values,
    np.ndarray valid,
    np.ndarray sums,
    np.ndarray present,
    bint skip_key_null=False,
    int32_t key_null=0,
    bint skip_value_nan=False,
):
    """Accumulate ``sum(values)`` by dense int32 keys.

    This is a low-level CTable group-by helper.  *keys*, *values*, and *valid*
    are same-length 1-D chunk arrays.  *sums* and *present* are dense group
    state arrays indexed directly by key value.  Keys must be non-negative and
    already fit in the state arrays.
    """
    if keys.ndim != 1 or values.ndim != 1 or valid.ndim != 1:
        raise ValueError("keys, values and valid must be 1-D arrays")
    if keys.shape[0] != values.shape[0] or keys.shape[0] != valid.shape[0]:
        raise ValueError("keys, values and valid must have the same length")
    if sums.ndim != 1 or present.ndim != 1:
        raise ValueError("sums and present must be 1-D arrays")
    if keys.dtype != np.dtype(np.int32):
        raise TypeError("keys must have dtype int32")
    if values.dtype != np.dtype(np.float64):
        raise TypeError("values must have dtype float64")
    if valid.dtype != np.dtype(np.bool_):
        raise TypeError("valid must have dtype bool")
    if sums.dtype != np.dtype(np.float64):
        raise TypeError("sums must have dtype float64")
    if present.dtype != np.dtype(np.bool_):
        raise TypeError("present must have dtype bool")

    cdef int32_t[:] keys_view = keys
    cdef double[:] values_view = values
    cdef np.npy_bool[:] valid_view = valid
    cdef double[:] sums_view = sums
    cdef np.npy_bool[:] present_view = present
    cdef Py_ssize_t n = keys.shape[0]
    cdef Py_ssize_t nstates = sums.shape[0]
    cdef Py_ssize_t i
    cdef int32_t key
    cdef double value

    if present.shape[0] != sums.shape[0]:
        raise ValueError("present and sums must have the same length")

    with nogil:
        for i in range(n):
            if not valid_view[i]:
                continue
            key = keys_view[i]
            if skip_key_null and key == key_null:
                continue
            if key < 0 or key >= nstates:
                continue
            value = values_view[i]
            if skip_value_nan and value != value:
                continue
            sums_view[key] += value
            present_view[key] = 1
    return None


def groupby_dense_i32_f64_sum_checked(
    np.ndarray keys,
    np.ndarray values,
    np.ndarray valid,
    np.ndarray sums,
    np.ndarray present,
    bint skip_key_null=False,
    int32_t key_null=0,
    bint skip_value_nan=False,
):
    """Checked dense int32/float64 sum kernel.

    Returns ``0`` on success, ``-1`` if a negative non-null key is found, or
    ``max_key + 1`` when the dense state arrays need to be grown.  The state is
    not mutated unless the function returns ``0``.
    """
    if keys.ndim != 1 or values.ndim != 1 or valid.ndim != 1:
        raise ValueError("keys, values and valid must be 1-D arrays")
    if keys.shape[0] != values.shape[0] or keys.shape[0] != valid.shape[0]:
        raise ValueError("keys, values and valid must have the same length")
    if sums.ndim != 1 or present.ndim != 1:
        raise ValueError("sums and present must be 1-D arrays")
    if keys.dtype != np.dtype(np.int32):
        raise TypeError("keys must have dtype int32")
    if values.dtype != np.dtype(np.float64):
        raise TypeError("values must have dtype float64")
    if valid.dtype != np.dtype(np.bool_):
        raise TypeError("valid must have dtype bool")
    if sums.dtype != np.dtype(np.float64):
        raise TypeError("sums must have dtype float64")
    if present.dtype != np.dtype(np.bool_):
        raise TypeError("present must have dtype bool")
    if present.shape[0] != sums.shape[0]:
        raise ValueError("present and sums must have the same length")

    cdef int32_t[:] keys_view = keys
    cdef double[:] values_view = values
    cdef np.npy_bool[:] valid_view = valid
    cdef double[:] sums_view = sums
    cdef np.npy_bool[:] present_view = present
    cdef Py_ssize_t n = keys.shape[0]
    cdef Py_ssize_t nstates = sums.shape[0]
    cdef Py_ssize_t i
    cdef int32_t key
    cdef int32_t max_key = -1
    cdef int ret = 0
    cdef double value

    with nogil:
        for i in range(n):
            if not valid_view[i]:
                continue
            key = keys_view[i]
            if skip_key_null and key == key_null:
                continue
            if key < 0:
                ret = -1
                break
            if key > max_key:
                max_key = key
        if ret == 0:
            if max_key < 0:
                ret = 0
            elif max_key >= nstates:
                ret = <int>max_key + 1
            else:
                for i in range(n):
                    if not valid_view[i]:
                        continue
                    key = keys_view[i]
                    if skip_key_null and key == key_null:
                        continue
                    value = values_view[i]
                    if skip_value_nan and value != value:
                        continue
                    sums_view[key] += value
                    present_view[key] = 1
    return ret


def groupby_dense_f64_integral_key_f64_sum_checked(
    np.ndarray keys,
    np.ndarray values,
    np.ndarray valid,
    np.ndarray sums,
    np.ndarray present,
    bint skip_key_nan=True,
    bint skip_value_nan=False,
):
    """Checked dense float64-integral-key/float64 sum kernel.

    Fast path for float keys that are exactly integral, finite and
    non-negative.  Returns ``0`` on success, ``-1`` if a key cannot be handled,
    or ``max_key + 1`` when the dense state arrays need to be grown.  The state is
    not mutated unless the function returns ``0``.
    """
    if keys.ndim != 1 or values.ndim != 1 or valid.ndim != 1:
        raise ValueError("keys, values and valid must be 1-D arrays")
    if keys.shape[0] != values.shape[0] or keys.shape[0] != valid.shape[0]:
        raise ValueError("keys, values and valid must have the same length")
    if sums.ndim != 1 or present.ndim != 1:
        raise ValueError("sums and present must be 1-D arrays")
    if keys.dtype != np.dtype(np.float64):
        raise TypeError("keys must have dtype float64")
    if values.dtype != np.dtype(np.float64):
        raise TypeError("values must have dtype float64")
    if valid.dtype != np.dtype(np.bool_):
        raise TypeError("valid must have dtype bool")
    if sums.dtype != np.dtype(np.float64):
        raise TypeError("sums must have dtype float64")
    if present.dtype != np.dtype(np.bool_):
        raise TypeError("present must have dtype bool")
    if present.shape[0] != sums.shape[0]:
        raise ValueError("present and sums must have the same length")

    cdef double[:] keys_view = keys
    cdef double[:] values_view = values
    cdef np.npy_bool[:] valid_view = valid
    cdef double[:] sums_view = sums
    cdef np.npy_bool[:] present_view = present
    cdef Py_ssize_t n = keys.shape[0]
    cdef Py_ssize_t nstates = sums.shape[0]
    cdef Py_ssize_t i
    cdef double key_f
    cdef int64_t key_i
    cdef int64_t max_key = -1
    cdef int ret = 0
    cdef double value

    with nogil:
        for i in range(n):
            if not valid_view[i]:
                continue
            key_f = keys_view[i]
            if key_f != key_f:
                if skip_key_nan:
                    continue
                ret = -1
                break
            if key_f < 0.0 or key_f > 9223372036854774784.0:
                ret = -1
                break
            key_i = <int64_t>key_f
            if key_f != <double>key_i:
                ret = -1
                break
            if key_i > max_key:
                max_key = key_i
        if ret == 0:
            if max_key < 0:
                ret = 0
            elif max_key >= nstates:
                if max_key > 2147483646:
                    ret = -1
                else:
                    ret = <int>max_key + 1
            else:
                for i in range(n):
                    if not valid_view[i]:
                        continue
                    key_f = keys_view[i]
                    if key_f != key_f:
                        if skip_key_nan:
                            continue
                        ret = -1
                        break
                    key_i = <int64_t>key_f
                    value = values_view[i]
                    if skip_value_nan and value != value:
                        continue
                    sums_view[key_i] += value
                    present_view[key_i] = 1
    return ret


def groupby_dense_f32_integral_key_f64_sum_checked(
    np.ndarray keys,
    np.ndarray values,
    np.ndarray valid,
    np.ndarray sums,
    np.ndarray present,
    bint skip_key_nan=True,
    bint skip_value_nan=False,
):
    """Checked dense float32-integral-key/float64 sum kernel."""
    if keys.ndim != 1 or values.ndim != 1 or valid.ndim != 1:
        raise ValueError("keys, values and valid must be 1-D arrays")
    if keys.shape[0] != values.shape[0] or keys.shape[0] != valid.shape[0]:
        raise ValueError("keys, values and valid must have the same length")
    if sums.ndim != 1 or present.ndim != 1:
        raise ValueError("sums and present must be 1-D arrays")
    if keys.dtype != np.dtype(np.float32):
        raise TypeError("keys must have dtype float32")
    if values.dtype != np.dtype(np.float64):
        raise TypeError("values must have dtype float64")
    if valid.dtype != np.dtype(np.bool_):
        raise TypeError("valid must have dtype bool")
    if sums.dtype != np.dtype(np.float64):
        raise TypeError("sums must have dtype float64")
    if present.dtype != np.dtype(np.bool_):
        raise TypeError("present must have dtype bool")
    if present.shape[0] != sums.shape[0]:
        raise ValueError("present and sums must have the same length")

    cdef float[:] keys_view = keys
    cdef double[:] values_view = values
    cdef np.npy_bool[:] valid_view = valid
    cdef double[:] sums_view = sums
    cdef np.npy_bool[:] present_view = present
    cdef Py_ssize_t n = keys.shape[0]
    cdef Py_ssize_t nstates = sums.shape[0]
    cdef Py_ssize_t i
    cdef float key_f
    cdef int64_t key_i
    cdef int64_t max_key = -1
    cdef int ret = 0
    cdef double value

    with nogil:
        for i in range(n):
            if not valid_view[i]:
                continue
            key_f = keys_view[i]
            if key_f != key_f:
                if skip_key_nan:
                    continue
                ret = -1
                break
            if key_f < 0.0 or key_f > 16777216.0:
                ret = -1
                break
            key_i = <int64_t>key_f
            if key_f != <float>key_i:
                ret = -1
                break
            if key_i > max_key:
                max_key = key_i
        if ret == 0:
            if max_key < 0:
                ret = 0
            elif max_key >= nstates:
                if max_key > 2147483646:
                    ret = -1
                else:
                    ret = <int>max_key + 1
            else:
                for i in range(n):
                    if not valid_view[i]:
                        continue
                    key_f = keys_view[i]
                    if key_f != key_f:
                        if skip_key_nan:
                            continue
                        ret = -1
                        break
                    key_i = <int64_t>key_f
                    value = values_view[i]
                    if skip_value_nan and value != value:
                        continue
                    sums_view[key_i] += value
                    present_view[key_i] = 1
    return ret
