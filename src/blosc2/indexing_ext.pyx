#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
cimport numpy as np

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t


cdef inline Py_ssize_t _search_left_float32(np.float32_t[:] values, np.float32_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_right_float32(np.float32_t[:] values, np.float32_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_left_float64(np.float64_t[:] values, np.float64_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_right_float64(np.float64_t[:] values, np.float64_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_left_int8(np.int8_t[:] values, np.int8_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_right_int8(np.int8_t[:] values, np.int8_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_left_int16(np.int16_t[:] values, np.int16_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_right_int16(np.int16_t[:] values, np.int16_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_left_int32(np.int32_t[:] values, np.int32_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_right_int32(np.int32_t[:] values, np.int32_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_left_int64(np.int64_t[:] values, np.int64_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_right_int64(np.int64_t[:] values, np.int64_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_left_uint8(np.uint8_t[:] values, np.uint8_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_right_uint8(np.uint8_t[:] values, np.uint8_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_left_uint16(np.uint16_t[:] values, np.uint16_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_right_uint16(np.uint16_t[:] values, np.uint16_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_left_uint32(np.uint32_t[:] values, np.uint32_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_right_uint32(np.uint32_t[:] values, np.uint32_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_left_uint64(np.uint64_t[:] values, np.uint64_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline Py_ssize_t _search_right_uint64(np.uint64_t[:] values, np.uint64_t target) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline tuple _search_bounds_float32_impl(
    np.ndarray[np.float32_t, ndim=1] values,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef np.float32_t lower_v
    cdef np.float32_t upper_v
    if lower is not None:
        lower_v = lower
        lo = _search_left_float32(values, lower_v) if lower_inclusive else _search_right_float32(values, lower_v)
    if upper is not None:
        upper_v = upper
        hi = _search_right_float32(values, upper_v) if upper_inclusive else _search_left_float32(values, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_boundary_bounds_float32_impl(
    np.ndarray[np.float32_t, ndim=1] starts,
    np.ndarray[np.float32_t, ndim=1] ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = starts.shape[0]
    cdef np.float32_t lower_v
    cdef np.float32_t upper_v
    if lower is not None:
        lower_v = lower
        lo = _search_left_float32(ends, lower_v) if lower_inclusive else _search_right_float32(ends, lower_v)
    if upper is not None:
        upper_v = upper
        hi = _search_right_float32(starts, upper_v) if upper_inclusive else _search_left_float32(starts, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_bounds_float64_impl(
    np.ndarray[np.float64_t, ndim=1] values,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef np.float64_t lower_v
    cdef np.float64_t upper_v
    if lower is not None:
        lower_v = lower
        lo = _search_left_float64(values, lower_v) if lower_inclusive else _search_right_float64(values, lower_v)
    if upper is not None:
        upper_v = upper
        hi = _search_right_float64(values, upper_v) if upper_inclusive else _search_left_float64(values, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_boundary_bounds_float64_impl(
    np.ndarray[np.float64_t, ndim=1] starts,
    np.ndarray[np.float64_t, ndim=1] ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = starts.shape[0]
    cdef np.float64_t lower_v
    cdef np.float64_t upper_v
    if lower is not None:
        lower_v = lower
        lo = _search_left_float64(ends, lower_v) if lower_inclusive else _search_right_float64(ends, lower_v)
    if upper is not None:
        upper_v = upper
        hi = _search_right_float64(starts, upper_v) if upper_inclusive else _search_left_float64(starts, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_bounds_int8_impl(
    np.ndarray[np.int8_t, ndim=1] values,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef int lower_i
    cdef int upper_i
    cdef np.int8_t lower_v
    cdef np.int8_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 127:
            lo = hi
        elif lower_i >= -128:
            lower_v = <np.int8_t>lower_i
            lo = _search_left_int8(values, lower_v) if lower_inclusive else _search_right_int8(values, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < -128:
            hi = 0
        elif upper_i <= 127:
            upper_v = <np.int8_t>upper_i
            hi = _search_right_int8(values, upper_v) if upper_inclusive else _search_left_int8(values, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_boundary_bounds_int8_impl(
    np.ndarray[np.int8_t, ndim=1] starts,
    np.ndarray[np.int8_t, ndim=1] ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = starts.shape[0]
    cdef int lower_i
    cdef int upper_i
    cdef np.int8_t lower_v
    cdef np.int8_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 127:
            lo = hi
        elif lower_i >= -128:
            lower_v = <np.int8_t>lower_i
            lo = _search_left_int8(ends, lower_v) if lower_inclusive else _search_right_int8(ends, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < -128:
            hi = 0
        elif upper_i <= 127:
            upper_v = <np.int8_t>upper_i
            hi = _search_right_int8(starts, upper_v) if upper_inclusive else _search_left_int8(starts, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_bounds_int16_impl(
    np.ndarray[np.int16_t, ndim=1] values,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef int lower_i
    cdef int upper_i
    cdef np.int16_t lower_v
    cdef np.int16_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 32767:
            lo = hi
        elif lower_i >= -32768:
            lower_v = <np.int16_t>lower_i
            lo = _search_left_int16(values, lower_v) if lower_inclusive else _search_right_int16(values, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < -32768:
            hi = 0
        elif upper_i <= 32767:
            upper_v = <np.int16_t>upper_i
            hi = _search_right_int16(values, upper_v) if upper_inclusive else _search_left_int16(values, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_boundary_bounds_int16_impl(
    np.ndarray[np.int16_t, ndim=1] starts,
    np.ndarray[np.int16_t, ndim=1] ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = starts.shape[0]
    cdef int lower_i
    cdef int upper_i
    cdef np.int16_t lower_v
    cdef np.int16_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 32767:
            lo = hi
        elif lower_i >= -32768:
            lower_v = <np.int16_t>lower_i
            lo = _search_left_int16(ends, lower_v) if lower_inclusive else _search_right_int16(ends, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < -32768:
            hi = 0
        elif upper_i <= 32767:
            upper_v = <np.int16_t>upper_i
            hi = _search_right_int16(starts, upper_v) if upper_inclusive else _search_left_int16(starts, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_bounds_int32_impl(
    np.ndarray[np.int32_t, ndim=1] values,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef long long lower_i
    cdef long long upper_i
    cdef np.int32_t lower_v
    cdef np.int32_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 2147483647:
            lo = hi
        elif lower_i >= -2147483648:
            lower_v = <np.int32_t>lower_i
            lo = _search_left_int32(values, lower_v) if lower_inclusive else _search_right_int32(values, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < -2147483648:
            hi = 0
        elif upper_i <= 2147483647:
            upper_v = <np.int32_t>upper_i
            hi = _search_right_int32(values, upper_v) if upper_inclusive else _search_left_int32(values, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_boundary_bounds_int32_impl(
    np.ndarray[np.int32_t, ndim=1] starts,
    np.ndarray[np.int32_t, ndim=1] ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = starts.shape[0]
    cdef long long lower_i
    cdef long long upper_i
    cdef np.int32_t lower_v
    cdef np.int32_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 2147483647:
            lo = hi
        elif lower_i >= -2147483648:
            lower_v = <np.int32_t>lower_i
            lo = _search_left_int32(ends, lower_v) if lower_inclusive else _search_right_int32(ends, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < -2147483648:
            hi = 0
        elif upper_i <= 2147483647:
            upper_v = <np.int32_t>upper_i
            hi = _search_right_int32(starts, upper_v) if upper_inclusive else _search_left_int32(starts, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_bounds_int64_impl(
    np.ndarray[np.int64_t, ndim=1] values,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef object lower_i
    cdef object upper_i
    cdef np.int64_t lower_v
    cdef np.int64_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 9223372036854775807:
            lo = hi
        elif lower_i >= -9223372036854775808:
            lower_v = <np.int64_t>lower_i
            lo = _search_left_int64(values, lower_v) if lower_inclusive else _search_right_int64(values, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < -9223372036854775808:
            hi = 0
        elif upper_i <= 9223372036854775807:
            upper_v = <np.int64_t>upper_i
            hi = _search_right_int64(values, upper_v) if upper_inclusive else _search_left_int64(values, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_boundary_bounds_int64_impl(
    np.ndarray[np.int64_t, ndim=1] starts,
    np.ndarray[np.int64_t, ndim=1] ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = starts.shape[0]
    cdef object lower_i
    cdef object upper_i
    cdef np.int64_t lower_v
    cdef np.int64_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 9223372036854775807:
            lo = hi
        elif lower_i >= -9223372036854775808:
            lower_v = <np.int64_t>lower_i
            lo = _search_left_int64(ends, lower_v) if lower_inclusive else _search_right_int64(ends, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < -9223372036854775808:
            hi = 0
        elif upper_i <= 9223372036854775807:
            upper_v = <np.int64_t>upper_i
            hi = _search_right_int64(starts, upper_v) if upper_inclusive else _search_left_int64(starts, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_bounds_uint8_impl(
    np.ndarray[np.uint8_t, ndim=1] values,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef object lower_i
    cdef object upper_i
    cdef np.uint8_t lower_v
    cdef np.uint8_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 255:
            lo = hi
        elif lower_i >= 0:
            lower_v = <np.uint8_t>lower_i
            lo = _search_left_uint8(values, lower_v) if lower_inclusive else _search_right_uint8(values, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < 0:
            hi = 0
        elif upper_i <= 255:
            upper_v = <np.uint8_t>upper_i
            hi = _search_right_uint8(values, upper_v) if upper_inclusive else _search_left_uint8(values, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_boundary_bounds_uint8_impl(
    np.ndarray[np.uint8_t, ndim=1] starts,
    np.ndarray[np.uint8_t, ndim=1] ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = starts.shape[0]
    cdef object lower_i
    cdef object upper_i
    cdef np.uint8_t lower_v
    cdef np.uint8_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 255:
            lo = hi
        elif lower_i >= 0:
            lower_v = <np.uint8_t>lower_i
            lo = _search_left_uint8(ends, lower_v) if lower_inclusive else _search_right_uint8(ends, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < 0:
            hi = 0
        elif upper_i <= 255:
            upper_v = <np.uint8_t>upper_i
            hi = _search_right_uint8(starts, upper_v) if upper_inclusive else _search_left_uint8(starts, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_bounds_uint16_impl(
    np.ndarray[np.uint16_t, ndim=1] values,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef object lower_i
    cdef object upper_i
    cdef np.uint16_t lower_v
    cdef np.uint16_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 65535:
            lo = hi
        elif lower_i >= 0:
            lower_v = <np.uint16_t>lower_i
            lo = _search_left_uint16(values, lower_v) if lower_inclusive else _search_right_uint16(values, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < 0:
            hi = 0
        elif upper_i <= 65535:
            upper_v = <np.uint16_t>upper_i
            hi = _search_right_uint16(values, upper_v) if upper_inclusive else _search_left_uint16(values, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_boundary_bounds_uint16_impl(
    np.ndarray[np.uint16_t, ndim=1] starts,
    np.ndarray[np.uint16_t, ndim=1] ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = starts.shape[0]
    cdef object lower_i
    cdef object upper_i
    cdef np.uint16_t lower_v
    cdef np.uint16_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 65535:
            lo = hi
        elif lower_i >= 0:
            lower_v = <np.uint16_t>lower_i
            lo = _search_left_uint16(ends, lower_v) if lower_inclusive else _search_right_uint16(ends, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < 0:
            hi = 0
        elif upper_i <= 65535:
            upper_v = <np.uint16_t>upper_i
            hi = _search_right_uint16(starts, upper_v) if upper_inclusive else _search_left_uint16(starts, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_bounds_uint32_impl(
    np.ndarray[np.uint32_t, ndim=1] values,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef object lower_i
    cdef object upper_i
    cdef np.uint32_t lower_v
    cdef np.uint32_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 4294967295:
            lo = hi
        elif lower_i >= 0:
            lower_v = <np.uint32_t>lower_i
            lo = _search_left_uint32(values, lower_v) if lower_inclusive else _search_right_uint32(values, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < 0:
            hi = 0
        elif upper_i <= 4294967295:
            upper_v = <np.uint32_t>upper_i
            hi = _search_right_uint32(values, upper_v) if upper_inclusive else _search_left_uint32(values, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_boundary_bounds_uint32_impl(
    np.ndarray[np.uint32_t, ndim=1] starts,
    np.ndarray[np.uint32_t, ndim=1] ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = starts.shape[0]
    cdef object lower_i
    cdef object upper_i
    cdef np.uint32_t lower_v
    cdef np.uint32_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 4294967295:
            lo = hi
        elif lower_i >= 0:
            lower_v = <np.uint32_t>lower_i
            lo = _search_left_uint32(ends, lower_v) if lower_inclusive else _search_right_uint32(ends, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < 0:
            hi = 0
        elif upper_i <= 4294967295:
            upper_v = <np.uint32_t>upper_i
            hi = _search_right_uint32(starts, upper_v) if upper_inclusive else _search_left_uint32(starts, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_bounds_uint64_impl(
    np.ndarray[np.uint64_t, ndim=1] values,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = values.shape[0]
    cdef object lower_i
    cdef object upper_i
    cdef np.uint64_t lower_v
    cdef np.uint64_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 18446744073709551615:
            lo = hi
        elif lower_i >= 0:
            lower_v = <np.uint64_t>lower_i
            lo = _search_left_uint64(values, lower_v) if lower_inclusive else _search_right_uint64(values, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < 0:
            hi = 0
        elif upper_i <= 18446744073709551615:
            upper_v = <np.uint64_t>upper_i
            hi = _search_right_uint64(values, upper_v) if upper_inclusive else _search_left_uint64(values, upper_v)
    return int(lo), int(hi)


cdef inline tuple _search_boundary_bounds_uint64_impl(
    np.ndarray[np.uint64_t, ndim=1] starts,
    np.ndarray[np.uint64_t, ndim=1] ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = starts.shape[0]
    cdef object lower_i
    cdef object upper_i
    cdef np.uint64_t lower_v
    cdef np.uint64_t upper_v
    if lower is not None:
        lower_i = int(lower)
        if lower_i > 18446744073709551615:
            lo = hi
        elif lower_i >= 0:
            lower_v = <np.uint64_t>lower_i
            lo = _search_left_uint64(ends, lower_v) if lower_inclusive else _search_right_uint64(ends, lower_v)
    if upper is not None:
        upper_i = int(upper)
        if upper_i < 0:
            hi = 0
        elif upper_i <= 18446744073709551615:
            upper_v = <np.uint64_t>upper_i
            hi = _search_right_uint64(starts, upper_v) if upper_inclusive else _search_left_uint64(starts, upper_v)
    return int(lo), int(hi)


def index_search_bounds(np.ndarray values, object lower, bint lower_inclusive, object upper, bint upper_inclusive):
    cdef np.dtype dtype = values.dtype
    if dtype == np.dtype(np.float32):
        return _search_bounds_float32_impl(values, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.float64):
        return _search_bounds_float64_impl(values, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.int8):
        return _search_bounds_int8_impl(values, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.int16):
        return _search_bounds_int16_impl(values, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.int32):
        return _search_bounds_int32_impl(values, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.int64):
        return _search_bounds_int64_impl(values, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.uint8):
        return _search_bounds_uint8_impl(values, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.uint16):
        return _search_bounds_uint16_impl(values, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.uint32):
        return _search_bounds_uint32_impl(values, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.uint64):
        return _search_bounds_uint64_impl(values, lower, lower_inclusive, upper, upper_inclusive)
    raise TypeError("unsupported dtype for index_search_bounds")


def index_search_boundary_bounds(
    np.ndarray starts,
    np.ndarray ends,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef np.dtype dtype = starts.dtype
    if dtype != ends.dtype:
        raise TypeError("starts and ends must have the same dtype")
    if dtype == np.dtype(np.float32):
        return _search_boundary_bounds_float32_impl(starts, ends, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.float64):
        return _search_boundary_bounds_float64_impl(starts, ends, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.int8):
        return _search_boundary_bounds_int8_impl(starts, ends, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.int16):
        return _search_boundary_bounds_int16_impl(starts, ends, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.int32):
        return _search_boundary_bounds_int32_impl(starts, ends, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.int64):
        return _search_boundary_bounds_int64_impl(starts, ends, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.uint8):
        return _search_boundary_bounds_uint8_impl(starts, ends, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.uint16):
        return _search_boundary_bounds_uint16_impl(starts, ends, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.uint32):
        return _search_boundary_bounds_uint32_impl(starts, ends, lower, lower_inclusive, upper, upper_inclusive)
    if dtype == np.dtype(np.uint64):
        return _search_boundary_bounds_uint64_impl(starts, ends, lower, lower_inclusive, upper, upper_inclusive)
    raise TypeError("unsupported dtype for index_search_boundary_bounds")


cdef tuple _collect_chunk_positions_float32(
    np.ndarray[np.int64_t, ndim=1] offsets,
    np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar,
    object positions_sidecar,
    object l2_sidecar,
    np.ndarray l2_row,
    np.ndarray[np.float32_t, ndim=1] span_values,
    np.ndarray local_positions,
    int64_t chunk_len,
    int32_t nav_segment_len,
    int32_t nsegments_per_chunk,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef np.ndarray[np.float32_t, ndim=1] starts = l2_row["start"]
    cdef np.ndarray[np.float32_t, ndim=1] ends = l2_row["end"]
    cdef Py_ssize_t idx
    cdef int64_t chunk_id
    cdef int64_t chunk_items
    cdef int32_t segment_count
    cdef int seg_lo
    cdef int seg_hi
    cdef int64_t local_start
    cdef int64_t local_stop
    cdef int32_t span_items
    cdef int lo
    cdef int hi
    cdef int total_candidate_segments = 0
    cdef list parts = []
    cdef np.ndarray values_view
    cdef np.ndarray positions_view
    for idx in range(candidate_chunk_ids.shape[0]):
        chunk_id = candidate_chunk_ids[idx]
        chunk_items = offsets[chunk_id + 1] - offsets[chunk_id]
        segment_count = <int32_t>((chunk_items + nav_segment_len - 1) // nav_segment_len)
        l2_sidecar.get_1d_span_numpy(l2_row, chunk_id, 0, nsegments_per_chunk)
        seg_lo, seg_hi = _search_boundary_bounds_float32_impl(
            starts[:segment_count], ends[:segment_count], lower, lower_inclusive, upper, upper_inclusive
        )
        total_candidate_segments += seg_hi - seg_lo
        if seg_lo >= seg_hi:
            continue
        local_start = seg_lo * nav_segment_len
        local_stop = min(seg_hi * nav_segment_len, chunk_items)
        span_items = <int32_t>(local_stop - local_start)
        values_view = span_values[:span_items]
        values_sidecar.get_1d_span_numpy(values_view, chunk_id, <int32_t>local_start, span_items)
        lo, hi = _search_bounds_float32_impl(values_view, lower, lower_inclusive, upper, upper_inclusive)
        if lo >= hi:
            continue
        positions_view = local_positions[: hi - lo]
        positions_sidecar.get_1d_span_numpy(positions_view, chunk_id, <int32_t>(local_start + lo), hi - lo)
        parts.append(chunk_id * chunk_len + positions_view.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64), total_candidate_segments
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), total_candidate_segments


cdef tuple _collect_chunk_positions_float64(
    np.ndarray[np.int64_t, ndim=1] offsets,
    np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar,
    object positions_sidecar,
    object l2_sidecar,
    np.ndarray l2_row,
    np.ndarray[np.float64_t, ndim=1] span_values,
    np.ndarray local_positions,
    int64_t chunk_len,
    int32_t nav_segment_len,
    int32_t nsegments_per_chunk,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef np.ndarray[np.float64_t, ndim=1] starts = l2_row["start"]
    cdef np.ndarray[np.float64_t, ndim=1] ends = l2_row["end"]
    cdef Py_ssize_t idx
    cdef int64_t chunk_id
    cdef int64_t chunk_items
    cdef int32_t segment_count
    cdef int seg_lo
    cdef int seg_hi
    cdef int64_t local_start
    cdef int64_t local_stop
    cdef int32_t span_items
    cdef int lo
    cdef int hi
    cdef int total_candidate_segments = 0
    cdef list parts = []
    cdef np.ndarray values_view
    cdef np.ndarray positions_view
    for idx in range(candidate_chunk_ids.shape[0]):
        chunk_id = candidate_chunk_ids[idx]
        chunk_items = offsets[chunk_id + 1] - offsets[chunk_id]
        segment_count = <int32_t>((chunk_items + nav_segment_len - 1) // nav_segment_len)
        l2_sidecar.get_1d_span_numpy(l2_row, chunk_id, 0, nsegments_per_chunk)
        seg_lo, seg_hi = _search_boundary_bounds_float64_impl(
            starts[:segment_count], ends[:segment_count], lower, lower_inclusive, upper, upper_inclusive
        )
        total_candidate_segments += seg_hi - seg_lo
        if seg_lo >= seg_hi:
            continue
        local_start = seg_lo * nav_segment_len
        local_stop = min(seg_hi * nav_segment_len, chunk_items)
        span_items = <int32_t>(local_stop - local_start)
        values_view = span_values[:span_items]
        values_sidecar.get_1d_span_numpy(values_view, chunk_id, <int32_t>local_start, span_items)
        lo, hi = _search_bounds_float64_impl(values_view, lower, lower_inclusive, upper, upper_inclusive)
        if lo >= hi:
            continue
        positions_view = local_positions[: hi - lo]
        positions_sidecar.get_1d_span_numpy(positions_view, chunk_id, <int32_t>(local_start + lo), hi - lo)
        parts.append(chunk_id * chunk_len + positions_view.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64), total_candidate_segments
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), total_candidate_segments


cdef tuple _collect_chunk_positions_int8(
    np.ndarray[np.int64_t, ndim=1] offsets, np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar, object positions_sidecar, object l2_sidecar, np.ndarray l2_row,
    np.ndarray[np.int8_t, ndim=1] span_values, np.ndarray local_positions,
    int64_t chunk_len, int32_t nav_segment_len, int32_t nsegments_per_chunk,
    object lower, bint lower_inclusive, object upper, bint upper_inclusive,
):
    cdef np.ndarray[np.int8_t, ndim=1] starts = l2_row["start"]
    cdef np.ndarray[np.int8_t, ndim=1] ends = l2_row["end"]
    cdef Py_ssize_t idx
    cdef int64_t chunk_id
    cdef int64_t chunk_items
    cdef int32_t segment_count
    cdef int seg_lo
    cdef int seg_hi
    cdef int64_t local_start
    cdef int64_t local_stop
    cdef int32_t span_items
    cdef int lo
    cdef int hi
    cdef int total_candidate_segments = 0
    cdef list parts = []
    cdef np.ndarray values_view
    cdef np.ndarray positions_view
    for idx in range(candidate_chunk_ids.shape[0]):
        chunk_id = candidate_chunk_ids[idx]
        chunk_items = offsets[chunk_id + 1] - offsets[chunk_id]
        segment_count = <int32_t>((chunk_items + nav_segment_len - 1) // nav_segment_len)
        l2_sidecar.get_1d_span_numpy(l2_row, chunk_id, 0, nsegments_per_chunk)
        seg_lo, seg_hi = _search_boundary_bounds_int8_impl(starts[:segment_count], ends[:segment_count], lower, lower_inclusive, upper, upper_inclusive)
        total_candidate_segments += seg_hi - seg_lo
        if seg_lo >= seg_hi:
            continue
        local_start = seg_lo * nav_segment_len
        local_stop = min(seg_hi * nav_segment_len, chunk_items)
        span_items = <int32_t>(local_stop - local_start)
        values_view = span_values[:span_items]
        values_sidecar.get_1d_span_numpy(values_view, chunk_id, <int32_t>local_start, span_items)
        lo, hi = _search_bounds_int8_impl(values_view, lower, lower_inclusive, upper, upper_inclusive)
        if lo >= hi:
            continue
        positions_view = local_positions[: hi - lo]
        positions_sidecar.get_1d_span_numpy(positions_view, chunk_id, <int32_t>(local_start + lo), hi - lo)
        parts.append(chunk_id * chunk_len + positions_view.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64), total_candidate_segments
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), total_candidate_segments


cdef tuple _collect_chunk_positions_int16(
    np.ndarray[np.int64_t, ndim=1] offsets, np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar, object positions_sidecar, object l2_sidecar, np.ndarray l2_row,
    np.ndarray[np.int16_t, ndim=1] span_values, np.ndarray local_positions,
    int64_t chunk_len, int32_t nav_segment_len, int32_t nsegments_per_chunk,
    object lower, bint lower_inclusive, object upper, bint upper_inclusive,
):
    cdef np.ndarray[np.int16_t, ndim=1] starts = l2_row["start"]
    cdef np.ndarray[np.int16_t, ndim=1] ends = l2_row["end"]
    cdef Py_ssize_t idx
    cdef int64_t chunk_id
    cdef int64_t chunk_items
    cdef int32_t segment_count
    cdef int seg_lo
    cdef int seg_hi
    cdef int64_t local_start
    cdef int64_t local_stop
    cdef int32_t span_items
    cdef int lo
    cdef int hi
    cdef int total_candidate_segments = 0
    cdef list parts = []
    cdef np.ndarray values_view
    cdef np.ndarray positions_view
    for idx in range(candidate_chunk_ids.shape[0]):
        chunk_id = candidate_chunk_ids[idx]
        chunk_items = offsets[chunk_id + 1] - offsets[chunk_id]
        segment_count = <int32_t>((chunk_items + nav_segment_len - 1) // nav_segment_len)
        l2_sidecar.get_1d_span_numpy(l2_row, chunk_id, 0, nsegments_per_chunk)
        seg_lo, seg_hi = _search_boundary_bounds_int16_impl(starts[:segment_count], ends[:segment_count], lower, lower_inclusive, upper, upper_inclusive)
        total_candidate_segments += seg_hi - seg_lo
        if seg_lo >= seg_hi:
            continue
        local_start = seg_lo * nav_segment_len
        local_stop = min(seg_hi * nav_segment_len, chunk_items)
        span_items = <int32_t>(local_stop - local_start)
        values_view = span_values[:span_items]
        values_sidecar.get_1d_span_numpy(values_view, chunk_id, <int32_t>local_start, span_items)
        lo, hi = _search_bounds_int16_impl(values_view, lower, lower_inclusive, upper, upper_inclusive)
        if lo >= hi:
            continue
        positions_view = local_positions[: hi - lo]
        positions_sidecar.get_1d_span_numpy(positions_view, chunk_id, <int32_t>(local_start + lo), hi - lo)
        parts.append(chunk_id * chunk_len + positions_view.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64), total_candidate_segments
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), total_candidate_segments


cdef tuple _collect_chunk_positions_int32(
    np.ndarray[np.int64_t, ndim=1] offsets, np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar, object positions_sidecar, object l2_sidecar, np.ndarray l2_row,
    np.ndarray[np.int32_t, ndim=1] span_values, np.ndarray local_positions,
    int64_t chunk_len, int32_t nav_segment_len, int32_t nsegments_per_chunk,
    object lower, bint lower_inclusive, object upper, bint upper_inclusive,
):
    cdef np.ndarray[np.int32_t, ndim=1] starts = l2_row["start"]
    cdef np.ndarray[np.int32_t, ndim=1] ends = l2_row["end"]
    cdef Py_ssize_t idx
    cdef int64_t chunk_id
    cdef int64_t chunk_items
    cdef int32_t segment_count
    cdef int seg_lo
    cdef int seg_hi
    cdef int64_t local_start
    cdef int64_t local_stop
    cdef int32_t span_items
    cdef int lo
    cdef int hi
    cdef int total_candidate_segments = 0
    cdef list parts = []
    cdef np.ndarray values_view
    cdef np.ndarray positions_view
    for idx in range(candidate_chunk_ids.shape[0]):
        chunk_id = candidate_chunk_ids[idx]
        chunk_items = offsets[chunk_id + 1] - offsets[chunk_id]
        segment_count = <int32_t>((chunk_items + nav_segment_len - 1) // nav_segment_len)
        l2_sidecar.get_1d_span_numpy(l2_row, chunk_id, 0, nsegments_per_chunk)
        seg_lo, seg_hi = _search_boundary_bounds_int32_impl(starts[:segment_count], ends[:segment_count], lower, lower_inclusive, upper, upper_inclusive)
        total_candidate_segments += seg_hi - seg_lo
        if seg_lo >= seg_hi:
            continue
        local_start = seg_lo * nav_segment_len
        local_stop = min(seg_hi * nav_segment_len, chunk_items)
        span_items = <int32_t>(local_stop - local_start)
        values_view = span_values[:span_items]
        values_sidecar.get_1d_span_numpy(values_view, chunk_id, <int32_t>local_start, span_items)
        lo, hi = _search_bounds_int32_impl(values_view, lower, lower_inclusive, upper, upper_inclusive)
        if lo >= hi:
            continue
        positions_view = local_positions[: hi - lo]
        positions_sidecar.get_1d_span_numpy(positions_view, chunk_id, <int32_t>(local_start + lo), hi - lo)
        parts.append(chunk_id * chunk_len + positions_view.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64), total_candidate_segments
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), total_candidate_segments


cdef tuple _collect_chunk_positions_int64(
    np.ndarray[np.int64_t, ndim=1] offsets, np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar, object positions_sidecar, object l2_sidecar, np.ndarray l2_row,
    np.ndarray[np.int64_t, ndim=1] span_values, np.ndarray local_positions,
    int64_t chunk_len, int32_t nav_segment_len, int32_t nsegments_per_chunk,
    object lower, bint lower_inclusive, object upper, bint upper_inclusive,
):
    cdef np.ndarray[np.int64_t, ndim=1] starts = l2_row["start"]
    cdef np.ndarray[np.int64_t, ndim=1] ends = l2_row["end"]
    cdef Py_ssize_t idx
    cdef int64_t chunk_id
    cdef int64_t chunk_items
    cdef int32_t segment_count
    cdef int seg_lo
    cdef int seg_hi
    cdef int64_t local_start
    cdef int64_t local_stop
    cdef int32_t span_items
    cdef int lo
    cdef int hi
    cdef int total_candidate_segments = 0
    cdef list parts = []
    cdef np.ndarray values_view
    cdef np.ndarray positions_view
    for idx in range(candidate_chunk_ids.shape[0]):
        chunk_id = candidate_chunk_ids[idx]
        chunk_items = offsets[chunk_id + 1] - offsets[chunk_id]
        segment_count = <int32_t>((chunk_items + nav_segment_len - 1) // nav_segment_len)
        l2_sidecar.get_1d_span_numpy(l2_row, chunk_id, 0, nsegments_per_chunk)
        seg_lo, seg_hi = _search_boundary_bounds_int64_impl(starts[:segment_count], ends[:segment_count], lower, lower_inclusive, upper, upper_inclusive)
        total_candidate_segments += seg_hi - seg_lo
        if seg_lo >= seg_hi:
            continue
        local_start = seg_lo * nav_segment_len
        local_stop = min(seg_hi * nav_segment_len, chunk_items)
        span_items = <int32_t>(local_stop - local_start)
        values_view = span_values[:span_items]
        values_sidecar.get_1d_span_numpy(values_view, chunk_id, <int32_t>local_start, span_items)
        lo, hi = _search_bounds_int64_impl(values_view, lower, lower_inclusive, upper, upper_inclusive)
        if lo >= hi:
            continue
        positions_view = local_positions[: hi - lo]
        positions_sidecar.get_1d_span_numpy(positions_view, chunk_id, <int32_t>(local_start + lo), hi - lo)
        parts.append(chunk_id * chunk_len + positions_view.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64), total_candidate_segments
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), total_candidate_segments


cdef tuple _collect_chunk_positions_uint8(
    np.ndarray[np.int64_t, ndim=1] offsets, np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar, object positions_sidecar, object l2_sidecar, np.ndarray l2_row,
    np.ndarray[np.uint8_t, ndim=1] span_values, np.ndarray local_positions,
    int64_t chunk_len, int32_t nav_segment_len, int32_t nsegments_per_chunk,
    object lower, bint lower_inclusive, object upper, bint upper_inclusive,
):
    cdef np.ndarray[np.uint8_t, ndim=1] starts = l2_row["start"]
    cdef np.ndarray[np.uint8_t, ndim=1] ends = l2_row["end"]
    cdef Py_ssize_t idx
    cdef int64_t chunk_id
    cdef int64_t chunk_items
    cdef int32_t segment_count
    cdef int seg_lo
    cdef int seg_hi
    cdef int64_t local_start
    cdef int64_t local_stop
    cdef int32_t span_items
    cdef int lo
    cdef int hi
    cdef int total_candidate_segments = 0
    cdef list parts = []
    cdef np.ndarray values_view
    cdef np.ndarray positions_view
    for idx in range(candidate_chunk_ids.shape[0]):
        chunk_id = candidate_chunk_ids[idx]
        chunk_items = offsets[chunk_id + 1] - offsets[chunk_id]
        segment_count = <int32_t>((chunk_items + nav_segment_len - 1) // nav_segment_len)
        l2_sidecar.get_1d_span_numpy(l2_row, chunk_id, 0, nsegments_per_chunk)
        seg_lo, seg_hi = _search_boundary_bounds_uint8_impl(starts[:segment_count], ends[:segment_count], lower, lower_inclusive, upper, upper_inclusive)
        total_candidate_segments += seg_hi - seg_lo
        if seg_lo >= seg_hi:
            continue
        local_start = seg_lo * nav_segment_len
        local_stop = min(seg_hi * nav_segment_len, chunk_items)
        span_items = <int32_t>(local_stop - local_start)
        values_view = span_values[:span_items]
        values_sidecar.get_1d_span_numpy(values_view, chunk_id, <int32_t>local_start, span_items)
        lo, hi = _search_bounds_uint8_impl(values_view, lower, lower_inclusive, upper, upper_inclusive)
        if lo >= hi:
            continue
        positions_view = local_positions[: hi - lo]
        positions_sidecar.get_1d_span_numpy(positions_view, chunk_id, <int32_t>(local_start + lo), hi - lo)
        parts.append(chunk_id * chunk_len + positions_view.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64), total_candidate_segments
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), total_candidate_segments


cdef tuple _collect_chunk_positions_uint16(
    np.ndarray[np.int64_t, ndim=1] offsets, np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar, object positions_sidecar, object l2_sidecar, np.ndarray l2_row,
    np.ndarray[np.uint16_t, ndim=1] span_values, np.ndarray local_positions,
    int64_t chunk_len, int32_t nav_segment_len, int32_t nsegments_per_chunk,
    object lower, bint lower_inclusive, object upper, bint upper_inclusive,
):
    cdef np.ndarray[np.uint16_t, ndim=1] starts = l2_row["start"]
    cdef np.ndarray[np.uint16_t, ndim=1] ends = l2_row["end"]
    cdef Py_ssize_t idx
    cdef int64_t chunk_id
    cdef int64_t chunk_items
    cdef int32_t segment_count
    cdef int seg_lo
    cdef int seg_hi
    cdef int64_t local_start
    cdef int64_t local_stop
    cdef int32_t span_items
    cdef int lo
    cdef int hi
    cdef int total_candidate_segments = 0
    cdef list parts = []
    cdef np.ndarray values_view
    cdef np.ndarray positions_view
    for idx in range(candidate_chunk_ids.shape[0]):
        chunk_id = candidate_chunk_ids[idx]
        chunk_items = offsets[chunk_id + 1] - offsets[chunk_id]
        segment_count = <int32_t>((chunk_items + nav_segment_len - 1) // nav_segment_len)
        l2_sidecar.get_1d_span_numpy(l2_row, chunk_id, 0, nsegments_per_chunk)
        seg_lo, seg_hi = _search_boundary_bounds_uint16_impl(starts[:segment_count], ends[:segment_count], lower, lower_inclusive, upper, upper_inclusive)
        total_candidate_segments += seg_hi - seg_lo
        if seg_lo >= seg_hi:
            continue
        local_start = seg_lo * nav_segment_len
        local_stop = min(seg_hi * nav_segment_len, chunk_items)
        span_items = <int32_t>(local_stop - local_start)
        values_view = span_values[:span_items]
        values_sidecar.get_1d_span_numpy(values_view, chunk_id, <int32_t>local_start, span_items)
        lo, hi = _search_bounds_uint16_impl(values_view, lower, lower_inclusive, upper, upper_inclusive)
        if lo >= hi:
            continue
        positions_view = local_positions[: hi - lo]
        positions_sidecar.get_1d_span_numpy(positions_view, chunk_id, <int32_t>(local_start + lo), hi - lo)
        parts.append(chunk_id * chunk_len + positions_view.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64), total_candidate_segments
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), total_candidate_segments


cdef tuple _collect_chunk_positions_uint32(
    np.ndarray[np.int64_t, ndim=1] offsets, np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar, object positions_sidecar, object l2_sidecar, np.ndarray l2_row,
    np.ndarray[np.uint32_t, ndim=1] span_values, np.ndarray local_positions,
    int64_t chunk_len, int32_t nav_segment_len, int32_t nsegments_per_chunk,
    object lower, bint lower_inclusive, object upper, bint upper_inclusive,
):
    cdef np.ndarray[np.uint32_t, ndim=1] starts = l2_row["start"]
    cdef np.ndarray[np.uint32_t, ndim=1] ends = l2_row["end"]
    cdef Py_ssize_t idx
    cdef int64_t chunk_id
    cdef int64_t chunk_items
    cdef int32_t segment_count
    cdef int seg_lo
    cdef int seg_hi
    cdef int64_t local_start
    cdef int64_t local_stop
    cdef int32_t span_items
    cdef int lo
    cdef int hi
    cdef int total_candidate_segments = 0
    cdef list parts = []
    cdef np.ndarray values_view
    cdef np.ndarray positions_view
    for idx in range(candidate_chunk_ids.shape[0]):
        chunk_id = candidate_chunk_ids[idx]
        chunk_items = offsets[chunk_id + 1] - offsets[chunk_id]
        segment_count = <int32_t>((chunk_items + nav_segment_len - 1) // nav_segment_len)
        l2_sidecar.get_1d_span_numpy(l2_row, chunk_id, 0, nsegments_per_chunk)
        seg_lo, seg_hi = _search_boundary_bounds_uint32_impl(starts[:segment_count], ends[:segment_count], lower, lower_inclusive, upper, upper_inclusive)
        total_candidate_segments += seg_hi - seg_lo
        if seg_lo >= seg_hi:
            continue
        local_start = seg_lo * nav_segment_len
        local_stop = min(seg_hi * nav_segment_len, chunk_items)
        span_items = <int32_t>(local_stop - local_start)
        values_view = span_values[:span_items]
        values_sidecar.get_1d_span_numpy(values_view, chunk_id, <int32_t>local_start, span_items)
        lo, hi = _search_bounds_uint32_impl(values_view, lower, lower_inclusive, upper, upper_inclusive)
        if lo >= hi:
            continue
        positions_view = local_positions[: hi - lo]
        positions_sidecar.get_1d_span_numpy(positions_view, chunk_id, <int32_t>(local_start + lo), hi - lo)
        parts.append(chunk_id * chunk_len + positions_view.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64), total_candidate_segments
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), total_candidate_segments


cdef tuple _collect_chunk_positions_uint64(
    np.ndarray[np.int64_t, ndim=1] offsets, np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar, object positions_sidecar, object l2_sidecar, np.ndarray l2_row,
    np.ndarray[np.uint64_t, ndim=1] span_values, np.ndarray local_positions,
    int64_t chunk_len, int32_t nav_segment_len, int32_t nsegments_per_chunk,
    object lower, bint lower_inclusive, object upper, bint upper_inclusive,
):
    cdef np.ndarray[np.uint64_t, ndim=1] starts = l2_row["start"]
    cdef np.ndarray[np.uint64_t, ndim=1] ends = l2_row["end"]
    cdef Py_ssize_t idx
    cdef int64_t chunk_id
    cdef int64_t chunk_items
    cdef int32_t segment_count
    cdef int seg_lo
    cdef int seg_hi
    cdef int64_t local_start
    cdef int64_t local_stop
    cdef int32_t span_items
    cdef int lo
    cdef int hi
    cdef int total_candidate_segments = 0
    cdef list parts = []
    cdef np.ndarray values_view
    cdef np.ndarray positions_view
    for idx in range(candidate_chunk_ids.shape[0]):
        chunk_id = candidate_chunk_ids[idx]
        chunk_items = offsets[chunk_id + 1] - offsets[chunk_id]
        segment_count = <int32_t>((chunk_items + nav_segment_len - 1) // nav_segment_len)
        l2_sidecar.get_1d_span_numpy(l2_row, chunk_id, 0, nsegments_per_chunk)
        seg_lo, seg_hi = _search_boundary_bounds_uint64_impl(starts[:segment_count], ends[:segment_count], lower, lower_inclusive, upper, upper_inclusive)
        total_candidate_segments += seg_hi - seg_lo
        if seg_lo >= seg_hi:
            continue
        local_start = seg_lo * nav_segment_len
        local_stop = min(seg_hi * nav_segment_len, chunk_items)
        span_items = <int32_t>(local_stop - local_start)
        values_view = span_values[:span_items]
        values_sidecar.get_1d_span_numpy(values_view, chunk_id, <int32_t>local_start, span_items)
        lo, hi = _search_bounds_uint64_impl(values_view, lower, lower_inclusive, upper, upper_inclusive)
        if lo >= hi:
            continue
        positions_view = local_positions[: hi - lo]
        positions_sidecar.get_1d_span_numpy(positions_view, chunk_id, <int32_t>(local_start + lo), hi - lo)
        parts.append(chunk_id * chunk_len + positions_view.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64), total_candidate_segments
    return (np.concatenate(parts) if len(parts) > 1 else parts[0]), total_candidate_segments


def index_collect_reduced_chunk_nav_positions(
    np.ndarray[np.int64_t, ndim=1] offsets,
    np.ndarray[np.intp_t, ndim=1] candidate_chunk_ids,
    object values_sidecar,
    object positions_sidecar,
    object l2_sidecar,
    np.ndarray l2_row,
    np.ndarray span_values,
    np.ndarray local_positions,
    int64_t chunk_len,
    int32_t nav_segment_len,
    int32_t nsegments_per_chunk,
    object lower,
    bint lower_inclusive,
    object upper,
    bint upper_inclusive,
):
    cdef np.dtype dtype = span_values.dtype
    if dtype == np.dtype(np.float32):
        return _collect_chunk_positions_float32(
            offsets, candidate_chunk_ids, values_sidecar, positions_sidecar, l2_sidecar, l2_row,
            span_values, local_positions, chunk_len, nav_segment_len, nsegments_per_chunk,
            lower, lower_inclusive, upper, upper_inclusive
        )
    if dtype == np.dtype(np.float64):
        return _collect_chunk_positions_float64(
            offsets, candidate_chunk_ids, values_sidecar, positions_sidecar, l2_sidecar, l2_row,
            span_values, local_positions, chunk_len, nav_segment_len, nsegments_per_chunk,
            lower, lower_inclusive, upper, upper_inclusive
        )
    if dtype == np.dtype(np.int8):
        return _collect_chunk_positions_int8(
            offsets, candidate_chunk_ids, values_sidecar, positions_sidecar, l2_sidecar, l2_row,
            span_values, local_positions, chunk_len, nav_segment_len, nsegments_per_chunk,
            lower, lower_inclusive, upper, upper_inclusive
        )
    if dtype == np.dtype(np.int16):
        return _collect_chunk_positions_int16(
            offsets, candidate_chunk_ids, values_sidecar, positions_sidecar, l2_sidecar, l2_row,
            span_values, local_positions, chunk_len, nav_segment_len, nsegments_per_chunk,
            lower, lower_inclusive, upper, upper_inclusive
        )
    if dtype == np.dtype(np.int32):
        return _collect_chunk_positions_int32(
            offsets, candidate_chunk_ids, values_sidecar, positions_sidecar, l2_sidecar, l2_row,
            span_values, local_positions, chunk_len, nav_segment_len, nsegments_per_chunk,
            lower, lower_inclusive, upper, upper_inclusive
        )
    if dtype == np.dtype(np.int64):
        return _collect_chunk_positions_int64(
            offsets, candidate_chunk_ids, values_sidecar, positions_sidecar, l2_sidecar, l2_row,
            span_values, local_positions, chunk_len, nav_segment_len, nsegments_per_chunk,
            lower, lower_inclusive, upper, upper_inclusive
        )
    if dtype == np.dtype(np.uint8):
        return _collect_chunk_positions_uint8(
            offsets, candidate_chunk_ids, values_sidecar, positions_sidecar, l2_sidecar, l2_row,
            span_values, local_positions, chunk_len, nav_segment_len, nsegments_per_chunk,
            lower, lower_inclusive, upper, upper_inclusive
        )
    if dtype == np.dtype(np.uint16):
        return _collect_chunk_positions_uint16(
            offsets, candidate_chunk_ids, values_sidecar, positions_sidecar, l2_sidecar, l2_row,
            span_values, local_positions, chunk_len, nav_segment_len, nsegments_per_chunk,
            lower, lower_inclusive, upper, upper_inclusive
        )
    if dtype == np.dtype(np.uint32):
        return _collect_chunk_positions_uint32(
            offsets, candidate_chunk_ids, values_sidecar, positions_sidecar, l2_sidecar, l2_row,
            span_values, local_positions, chunk_len, nav_segment_len, nsegments_per_chunk,
            lower, lower_inclusive, upper, upper_inclusive
        )
    if dtype == np.dtype(np.uint64):
        return _collect_chunk_positions_uint64(
            offsets, candidate_chunk_ids, values_sidecar, positions_sidecar, l2_sidecar, l2_row,
            span_values, local_positions, chunk_len, nav_segment_len, nsegments_per_chunk,
            lower, lower_inclusive, upper, upper_inclusive
        )
    raise TypeError("unsupported dtype for index_collect_reduced_chunk_nav_positions")
