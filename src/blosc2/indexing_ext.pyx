#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
cimport numpy as np

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t


ctypedef fused sort_float_t:
    np.float32_t
    np.float64_t


ctypedef fused sort_ordered_t:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t


cdef inline bint _le_float_pair(
    sort_float_t left_value,
    uint64_t left_position,
    sort_float_t right_value,
    uint64_t right_position,
) noexcept nogil:
    cdef bint left_nan = left_value != left_value
    cdef bint right_nan = right_value != right_value
    if left_nan:
        if right_nan:
            return left_position <= right_position
        return False
    if right_nan:
        return True
    if left_value < right_value:
        return True
    if left_value > right_value:
        return False
    return left_position <= right_position


cdef inline bint _le_ordered_pair(
    sort_ordered_t left_value,
    uint64_t left_position,
    sort_ordered_t right_value,
    uint64_t right_position,
) noexcept nogil:
    if left_value < right_value:
        return True
    if left_value > right_value:
        return False
    return left_position <= right_position


cdef void _stable_mergesort_float(
    sort_float_t[:] values,
    uint64_t[:] positions,
    sort_float_t[:] tmp_values,
    uint64_t[:] tmp_positions,
) noexcept nogil:
    cdef Py_ssize_t n = values.shape[0]
    cdef Py_ssize_t width = 1
    cdef Py_ssize_t start
    cdef Py_ssize_t mid
    cdef Py_ssize_t stop
    cdef Py_ssize_t left
    cdef Py_ssize_t right
    cdef Py_ssize_t out
    cdef sort_float_t[:] src_values = values
    cdef uint64_t[:] src_positions = positions
    cdef sort_float_t[:] dst_values = tmp_values
    cdef uint64_t[:] dst_positions = tmp_positions
    cdef sort_float_t[:] swap_values
    cdef uint64_t[:] swap_positions
    cdef bint in_original = True
    while width < n:
        start = 0
        while start < n:
            mid = start + width
            if mid > n:
                mid = n
            stop = start + 2 * width
            if stop > n:
                stop = n
            left = start
            right = mid
            out = start
            while left < mid and right < stop:
                if _le_float_pair(
                    src_values[left], src_positions[left], src_values[right], src_positions[right]
                ):
                    dst_values[out] = src_values[left]
                    dst_positions[out] = src_positions[left]
                    left += 1
                else:
                    dst_values[out] = src_values[right]
                    dst_positions[out] = src_positions[right]
                    right += 1
                out += 1
            while left < mid:
                dst_values[out] = src_values[left]
                dst_positions[out] = src_positions[left]
                left += 1
                out += 1
            while right < stop:
                dst_values[out] = src_values[right]
                dst_positions[out] = src_positions[right]
                right += 1
                out += 1
            start = stop
        swap_values = src_values
        src_values = dst_values
        dst_values = swap_values
        swap_positions = src_positions
        src_positions = dst_positions
        dst_positions = swap_positions
        in_original = not in_original
        width <<= 1
    if not in_original:
        for start in range(n):
            values[start] = src_values[start]
            positions[start] = src_positions[start]


cdef void _stable_mergesort_ordered(
    sort_ordered_t[:] values,
    uint64_t[:] positions,
    sort_ordered_t[:] tmp_values,
    uint64_t[:] tmp_positions,
) noexcept nogil:
    cdef Py_ssize_t n = values.shape[0]
    cdef Py_ssize_t width = 1
    cdef Py_ssize_t start
    cdef Py_ssize_t mid
    cdef Py_ssize_t stop
    cdef Py_ssize_t left
    cdef Py_ssize_t right
    cdef Py_ssize_t out
    cdef sort_ordered_t[:] src_values = values
    cdef uint64_t[:] src_positions = positions
    cdef sort_ordered_t[:] dst_values = tmp_values
    cdef uint64_t[:] dst_positions = tmp_positions
    cdef sort_ordered_t[:] swap_values
    cdef uint64_t[:] swap_positions
    cdef bint in_original = True
    while width < n:
        start = 0
        while start < n:
            mid = start + width
            if mid > n:
                mid = n
            stop = start + 2 * width
            if stop > n:
                stop = n
            left = start
            right = mid
            out = start
            while left < mid and right < stop:
                if _le_ordered_pair(
                    src_values[left], src_positions[left], src_values[right], src_positions[right]
                ):
                    dst_values[out] = src_values[left]
                    dst_positions[out] = src_positions[left]
                    left += 1
                else:
                    dst_values[out] = src_values[right]
                    dst_positions[out] = src_positions[right]
                    right += 1
                out += 1
            while left < mid:
                dst_values[out] = src_values[left]
                dst_positions[out] = src_positions[left]
                left += 1
                out += 1
            while right < stop:
                dst_values[out] = src_values[right]
                dst_positions[out] = src_positions[right]
                right += 1
                out += 1
            start = stop
        swap_values = src_values
        src_values = dst_values
        dst_values = swap_values
        swap_positions = src_positions
        src_positions = dst_positions
        dst_positions = swap_positions
        in_original = not in_original
        width <<= 1
    if not in_original:
        for start in range(n):
            values[start] = src_values[start]
            positions[start] = src_positions[start]


cdef tuple _intra_chunk_sort_run_float32(np.ndarray values, Py_ssize_t run_start, np.dtype position_dtype):
    cdef np.ndarray[np.float32_t, ndim=1] sorted_values = np.array(values, copy=True, order="C")
    cdef np.ndarray[np.uint64_t, ndim=1] positions = np.empty(sorted_values.shape[0], dtype=np.uint64)
    cdef np.ndarray[np.float32_t, ndim=1] tmp_values = np.empty_like(sorted_values)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_positions = np.empty_like(positions)
    cdef np.float32_t[:] sorted_values_mv = sorted_values
    cdef np.uint64_t[:] positions_mv = positions
    cdef np.float32_t[:] tmp_values_mv = tmp_values
    cdef np.uint64_t[:] tmp_positions_mv = tmp_positions
    cdef Py_ssize_t idx
    with nogil:
        for idx in range(sorted_values.shape[0]):
            positions[idx] = <uint64_t>(run_start + idx)
        _stable_mergesort_float(sorted_values_mv, positions_mv, tmp_values_mv, tmp_positions_mv)
    return sorted_values, positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_sort_run_float64(np.ndarray values, Py_ssize_t run_start, np.dtype position_dtype):
    cdef np.ndarray[np.float64_t, ndim=1] sorted_values = np.array(values, copy=True, order="C")
    cdef np.ndarray[np.uint64_t, ndim=1] positions = np.empty(sorted_values.shape[0], dtype=np.uint64)
    cdef np.ndarray[np.float64_t, ndim=1] tmp_values = np.empty_like(sorted_values)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_positions = np.empty_like(positions)
    cdef np.float64_t[:] sorted_values_mv = sorted_values
    cdef np.uint64_t[:] positions_mv = positions
    cdef np.float64_t[:] tmp_values_mv = tmp_values
    cdef np.uint64_t[:] tmp_positions_mv = tmp_positions
    cdef Py_ssize_t idx
    with nogil:
        for idx in range(sorted_values.shape[0]):
            positions[idx] = <uint64_t>(run_start + idx)
        _stable_mergesort_float(sorted_values_mv, positions_mv, tmp_values_mv, tmp_positions_mv)
    return sorted_values, positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_sort_run_int8(np.ndarray values, Py_ssize_t run_start, np.dtype position_dtype):
    cdef np.ndarray[np.int8_t, ndim=1] sorted_values = np.array(values, copy=True, order="C")
    cdef np.ndarray[np.uint64_t, ndim=1] positions = np.empty(sorted_values.shape[0], dtype=np.uint64)
    cdef np.ndarray[np.int8_t, ndim=1] tmp_values = np.empty_like(sorted_values)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_positions = np.empty_like(positions)
    cdef np.int8_t[:] sorted_values_mv = sorted_values
    cdef np.uint64_t[:] positions_mv = positions
    cdef np.int8_t[:] tmp_values_mv = tmp_values
    cdef np.uint64_t[:] tmp_positions_mv = tmp_positions
    cdef Py_ssize_t idx
    with nogil:
        for idx in range(sorted_values.shape[0]):
            positions[idx] = <uint64_t>(run_start + idx)
        _stable_mergesort_ordered(sorted_values_mv, positions_mv, tmp_values_mv, tmp_positions_mv)
    return sorted_values, positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_sort_run_int16(np.ndarray values, Py_ssize_t run_start, np.dtype position_dtype):
    cdef np.ndarray[np.int16_t, ndim=1] sorted_values = np.array(values, copy=True, order="C")
    cdef np.ndarray[np.uint64_t, ndim=1] positions = np.empty(sorted_values.shape[0], dtype=np.uint64)
    cdef np.ndarray[np.int16_t, ndim=1] tmp_values = np.empty_like(sorted_values)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_positions = np.empty_like(positions)
    cdef np.int16_t[:] sorted_values_mv = sorted_values
    cdef np.uint64_t[:] positions_mv = positions
    cdef np.int16_t[:] tmp_values_mv = tmp_values
    cdef np.uint64_t[:] tmp_positions_mv = tmp_positions
    cdef Py_ssize_t idx
    with nogil:
        for idx in range(sorted_values.shape[0]):
            positions[idx] = <uint64_t>(run_start + idx)
        _stable_mergesort_ordered(sorted_values_mv, positions_mv, tmp_values_mv, tmp_positions_mv)
    return sorted_values, positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_sort_run_int32(np.ndarray values, Py_ssize_t run_start, np.dtype position_dtype):
    cdef np.ndarray[np.int32_t, ndim=1] sorted_values = np.array(values, copy=True, order="C")
    cdef np.ndarray[np.uint64_t, ndim=1] positions = np.empty(sorted_values.shape[0], dtype=np.uint64)
    cdef np.ndarray[np.int32_t, ndim=1] tmp_values = np.empty_like(sorted_values)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_positions = np.empty_like(positions)
    cdef np.int32_t[:] sorted_values_mv = sorted_values
    cdef np.uint64_t[:] positions_mv = positions
    cdef np.int32_t[:] tmp_values_mv = tmp_values
    cdef np.uint64_t[:] tmp_positions_mv = tmp_positions
    cdef Py_ssize_t idx
    with nogil:
        for idx in range(sorted_values.shape[0]):
            positions[idx] = <uint64_t>(run_start + idx)
        _stable_mergesort_ordered(sorted_values_mv, positions_mv, tmp_values_mv, tmp_positions_mv)
    return sorted_values, positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_sort_run_int64(np.ndarray values, Py_ssize_t run_start, np.dtype position_dtype):
    cdef np.ndarray[np.int64_t, ndim=1] sorted_values = np.array(values, copy=True, order="C")
    cdef np.ndarray[np.uint64_t, ndim=1] positions = np.empty(sorted_values.shape[0], dtype=np.uint64)
    cdef np.ndarray[np.int64_t, ndim=1] tmp_values = np.empty_like(sorted_values)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_positions = np.empty_like(positions)
    cdef np.int64_t[:] sorted_values_mv = sorted_values
    cdef np.uint64_t[:] positions_mv = positions
    cdef np.int64_t[:] tmp_values_mv = tmp_values
    cdef np.uint64_t[:] tmp_positions_mv = tmp_positions
    cdef Py_ssize_t idx
    with nogil:
        for idx in range(sorted_values.shape[0]):
            positions[idx] = <uint64_t>(run_start + idx)
        _stable_mergesort_ordered(sorted_values_mv, positions_mv, tmp_values_mv, tmp_positions_mv)
    return sorted_values, positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_sort_run_uint8(np.ndarray values, Py_ssize_t run_start, np.dtype position_dtype):
    cdef np.ndarray[np.uint8_t, ndim=1] sorted_values = np.array(values, copy=True, order="C")
    cdef np.ndarray[np.uint64_t, ndim=1] positions = np.empty(sorted_values.shape[0], dtype=np.uint64)
    cdef np.ndarray[np.uint8_t, ndim=1] tmp_values = np.empty_like(sorted_values)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_positions = np.empty_like(positions)
    cdef np.uint8_t[:] sorted_values_mv = sorted_values
    cdef np.uint64_t[:] positions_mv = positions
    cdef np.uint8_t[:] tmp_values_mv = tmp_values
    cdef np.uint64_t[:] tmp_positions_mv = tmp_positions
    cdef Py_ssize_t idx
    with nogil:
        for idx in range(sorted_values.shape[0]):
            positions[idx] = <uint64_t>(run_start + idx)
        _stable_mergesort_ordered(sorted_values_mv, positions_mv, tmp_values_mv, tmp_positions_mv)
    return sorted_values, positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_sort_run_uint16(np.ndarray values, Py_ssize_t run_start, np.dtype position_dtype):
    cdef np.ndarray[np.uint16_t, ndim=1] sorted_values = np.array(values, copy=True, order="C")
    cdef np.ndarray[np.uint64_t, ndim=1] positions = np.empty(sorted_values.shape[0], dtype=np.uint64)
    cdef np.ndarray[np.uint16_t, ndim=1] tmp_values = np.empty_like(sorted_values)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_positions = np.empty_like(positions)
    cdef np.uint16_t[:] sorted_values_mv = sorted_values
    cdef np.uint64_t[:] positions_mv = positions
    cdef np.uint16_t[:] tmp_values_mv = tmp_values
    cdef np.uint64_t[:] tmp_positions_mv = tmp_positions
    cdef Py_ssize_t idx
    with nogil:
        for idx in range(sorted_values.shape[0]):
            positions[idx] = <uint64_t>(run_start + idx)
        _stable_mergesort_ordered(sorted_values_mv, positions_mv, tmp_values_mv, tmp_positions_mv)
    return sorted_values, positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_sort_run_uint32(np.ndarray values, Py_ssize_t run_start, np.dtype position_dtype):
    cdef np.ndarray[np.uint32_t, ndim=1] sorted_values = np.array(values, copy=True, order="C")
    cdef np.ndarray[np.uint64_t, ndim=1] positions = np.empty(sorted_values.shape[0], dtype=np.uint64)
    cdef np.ndarray[np.uint32_t, ndim=1] tmp_values = np.empty_like(sorted_values)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_positions = np.empty_like(positions)
    cdef np.uint32_t[:] sorted_values_mv = sorted_values
    cdef np.uint64_t[:] positions_mv = positions
    cdef np.uint32_t[:] tmp_values_mv = tmp_values
    cdef np.uint64_t[:] tmp_positions_mv = tmp_positions
    cdef Py_ssize_t idx
    with nogil:
        for idx in range(sorted_values.shape[0]):
            positions[idx] = <uint64_t>(run_start + idx)
        _stable_mergesort_ordered(sorted_values_mv, positions_mv, tmp_values_mv, tmp_positions_mv)
    return sorted_values, positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_sort_run_uint64(np.ndarray values, Py_ssize_t run_start, np.dtype position_dtype):
    cdef np.ndarray[np.uint64_t, ndim=1] sorted_values = np.array(values, copy=True, order="C")
    cdef np.ndarray[np.uint64_t, ndim=1] positions = np.empty(sorted_values.shape[0], dtype=np.uint64)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_values = np.empty_like(sorted_values)
    cdef np.ndarray[np.uint64_t, ndim=1] tmp_positions = np.empty_like(positions)
    cdef np.uint64_t[:] sorted_values_mv = sorted_values
    cdef np.uint64_t[:] positions_mv = positions
    cdef np.uint64_t[:] tmp_values_mv = tmp_values
    cdef np.uint64_t[:] tmp_positions_mv = tmp_positions
    cdef Py_ssize_t idx
    with nogil:
        for idx in range(sorted_values.shape[0]):
            positions[idx] = <uint64_t>(run_start + idx)
        _stable_mergesort_ordered(sorted_values_mv, positions_mv, tmp_values_mv, tmp_positions_mv)
    return sorted_values, positions.astype(position_dtype, copy=False)


def intra_chunk_sort_run(np.ndarray values, Py_ssize_t run_start, object position_dtype):
    cdef np.dtype dtype = values.dtype
    cdef np.dtype pos_dtype = np.dtype(position_dtype)
    if dtype == np.dtype(np.float32):
        return _intra_chunk_sort_run_float32(values, run_start, pos_dtype)
    if dtype == np.dtype(np.float64):
        return _intra_chunk_sort_run_float64(values, run_start, pos_dtype)
    if dtype == np.dtype(np.int8):
        return _intra_chunk_sort_run_int8(values, run_start, pos_dtype)
    if dtype == np.dtype(np.int16):
        return _intra_chunk_sort_run_int16(values, run_start, pos_dtype)
    if dtype == np.dtype(np.int32):
        return _intra_chunk_sort_run_int32(values, run_start, pos_dtype)
    if dtype == np.dtype(np.int64):
        return _intra_chunk_sort_run_int64(values, run_start, pos_dtype)
    if dtype == np.dtype(np.uint8) or dtype == np.dtype(np.bool_):
        sorted_values, positions = _intra_chunk_sort_run_uint8(values.view(np.uint8), run_start, pos_dtype)
        if dtype == np.dtype(np.bool_):
            return sorted_values.view(np.bool_), positions
        return sorted_values, positions
    if dtype == np.dtype(np.uint16):
        return _intra_chunk_sort_run_uint16(values, run_start, pos_dtype)
    if dtype == np.dtype(np.uint32):
        return _intra_chunk_sort_run_uint32(values, run_start, pos_dtype)
    if dtype == np.dtype(np.uint64):
        return _intra_chunk_sort_run_uint64(values, run_start, pos_dtype)
    if dtype.kind in {"m", "M"}:
        sorted_values, positions = _intra_chunk_sort_run_int64(values.view(np.int64), run_start, pos_dtype)
        return sorted_values.view(dtype), positions
    raise TypeError("unsupported dtype for intra_chunk_sort_run")


cdef void _linear_merge_float(
    sort_float_t[:] left_values,
    uint64_t[:] left_positions,
    sort_float_t[:] right_values,
    uint64_t[:] right_positions,
    sort_float_t[:] out_values,
    uint64_t[:] out_positions,
) noexcept nogil:
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = 0
    cdef Py_ssize_t out = 0
    cdef Py_ssize_t left_n = left_values.shape[0]
    cdef Py_ssize_t right_n = right_values.shape[0]
    while left < left_n and right < right_n:
        if _le_float_pair(left_values[left], left_positions[left], right_values[right], right_positions[right]):
            out_values[out] = left_values[left]
            out_positions[out] = left_positions[left]
            left += 1
        else:
            out_values[out] = right_values[right]
            out_positions[out] = right_positions[right]
            right += 1
        out += 1
    while left < left_n:
        out_values[out] = left_values[left]
        out_positions[out] = left_positions[left]
        left += 1
        out += 1
    while right < right_n:
        out_values[out] = right_values[right]
        out_positions[out] = right_positions[right]
        right += 1
        out += 1


cdef void _linear_merge_ordered(
    sort_ordered_t[:] left_values,
    uint64_t[:] left_positions,
    sort_ordered_t[:] right_values,
    uint64_t[:] right_positions,
    sort_ordered_t[:] out_values,
    uint64_t[:] out_positions,
) noexcept nogil:
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = 0
    cdef Py_ssize_t out = 0
    cdef Py_ssize_t left_n = left_values.shape[0]
    cdef Py_ssize_t right_n = right_values.shape[0]
    while left < left_n and right < right_n:
        if _le_ordered_pair(
            left_values[left], left_positions[left], right_values[right], right_positions[right]
        ):
            out_values[out] = left_values[left]
            out_positions[out] = left_positions[left]
            left += 1
        else:
            out_values[out] = right_values[right]
            out_positions[out] = right_positions[right]
            right += 1
        out += 1
    while left < left_n:
        out_values[out] = left_values[left]
        out_positions[out] = left_positions[left]
        left += 1
        out += 1
    while right < right_n:
        out_values[out] = right_values[right]
        out_positions[out] = right_positions[right]
        right += 1
        out += 1


cdef tuple _intra_chunk_merge_float32(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, np.dtype position_dtype
):
    cdef Py_ssize_t total = left_values.shape[0] + right_values.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] merged_values = np.empty(total, dtype=np.float32)
    cdef np.ndarray[np.uint64_t, ndim=1] merged_positions = np.empty(total, dtype=np.uint64)
    cdef np.float32_t[:] left_values_mv = left_values
    cdef np.uint64_t[:] left_positions_mv = np.asarray(left_positions, dtype=np.uint64)
    cdef np.float32_t[:] right_values_mv = right_values
    cdef np.uint64_t[:] right_positions_mv = np.asarray(right_positions, dtype=np.uint64)
    cdef np.float32_t[:] merged_values_mv = merged_values
    cdef np.uint64_t[:] merged_positions_mv = merged_positions
    with nogil:
        _linear_merge_float(
            left_values_mv, left_positions_mv, right_values_mv, right_positions_mv, merged_values_mv, merged_positions_mv
        )
    return merged_values, merged_positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_merge_float64(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, np.dtype position_dtype
):
    cdef Py_ssize_t total = left_values.shape[0] + right_values.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] merged_values = np.empty(total, dtype=np.float64)
    cdef np.ndarray[np.uint64_t, ndim=1] merged_positions = np.empty(total, dtype=np.uint64)
    cdef np.float64_t[:] left_values_mv = left_values
    cdef np.uint64_t[:] left_positions_mv = np.asarray(left_positions, dtype=np.uint64)
    cdef np.float64_t[:] right_values_mv = right_values
    cdef np.uint64_t[:] right_positions_mv = np.asarray(right_positions, dtype=np.uint64)
    cdef np.float64_t[:] merged_values_mv = merged_values
    cdef np.uint64_t[:] merged_positions_mv = merged_positions
    with nogil:
        _linear_merge_float(
            left_values_mv, left_positions_mv, right_values_mv, right_positions_mv, merged_values_mv, merged_positions_mv
        )
    return merged_values, merged_positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_merge_int8(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, np.dtype position_dtype
):
    cdef Py_ssize_t total = left_values.shape[0] + right_values.shape[0]
    cdef np.ndarray[np.int8_t, ndim=1] merged_values = np.empty(total, dtype=np.int8)
    cdef np.ndarray[np.uint64_t, ndim=1] merged_positions = np.empty(total, dtype=np.uint64)
    cdef np.int8_t[:] left_values_mv = left_values
    cdef np.uint64_t[:] left_positions_mv = np.asarray(left_positions, dtype=np.uint64)
    cdef np.int8_t[:] right_values_mv = right_values
    cdef np.uint64_t[:] right_positions_mv = np.asarray(right_positions, dtype=np.uint64)
    cdef np.int8_t[:] merged_values_mv = merged_values
    cdef np.uint64_t[:] merged_positions_mv = merged_positions
    with nogil:
        _linear_merge_ordered(
            left_values_mv, left_positions_mv, right_values_mv, right_positions_mv, merged_values_mv, merged_positions_mv
        )
    return merged_values, merged_positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_merge_int16(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, np.dtype position_dtype
):
    cdef Py_ssize_t total = left_values.shape[0] + right_values.shape[0]
    cdef np.ndarray[np.int16_t, ndim=1] merged_values = np.empty(total, dtype=np.int16)
    cdef np.ndarray[np.uint64_t, ndim=1] merged_positions = np.empty(total, dtype=np.uint64)
    cdef np.int16_t[:] left_values_mv = left_values
    cdef np.uint64_t[:] left_positions_mv = np.asarray(left_positions, dtype=np.uint64)
    cdef np.int16_t[:] right_values_mv = right_values
    cdef np.uint64_t[:] right_positions_mv = np.asarray(right_positions, dtype=np.uint64)
    cdef np.int16_t[:] merged_values_mv = merged_values
    cdef np.uint64_t[:] merged_positions_mv = merged_positions
    with nogil:
        _linear_merge_ordered(
            left_values_mv, left_positions_mv, right_values_mv, right_positions_mv, merged_values_mv, merged_positions_mv
        )
    return merged_values, merged_positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_merge_int32(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, np.dtype position_dtype
):
    cdef Py_ssize_t total = left_values.shape[0] + right_values.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] merged_values = np.empty(total, dtype=np.int32)
    cdef np.ndarray[np.uint64_t, ndim=1] merged_positions = np.empty(total, dtype=np.uint64)
    cdef np.int32_t[:] left_values_mv = left_values
    cdef np.uint64_t[:] left_positions_mv = np.asarray(left_positions, dtype=np.uint64)
    cdef np.int32_t[:] right_values_mv = right_values
    cdef np.uint64_t[:] right_positions_mv = np.asarray(right_positions, dtype=np.uint64)
    cdef np.int32_t[:] merged_values_mv = merged_values
    cdef np.uint64_t[:] merged_positions_mv = merged_positions
    with nogil:
        _linear_merge_ordered(
            left_values_mv, left_positions_mv, right_values_mv, right_positions_mv, merged_values_mv, merged_positions_mv
        )
    return merged_values, merged_positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_merge_int64(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, np.dtype position_dtype
):
    cdef Py_ssize_t total = left_values.shape[0] + right_values.shape[0]
    cdef np.ndarray[np.int64_t, ndim=1] merged_values = np.empty(total, dtype=np.int64)
    cdef np.ndarray[np.uint64_t, ndim=1] merged_positions = np.empty(total, dtype=np.uint64)
    cdef np.int64_t[:] left_values_mv = left_values
    cdef np.uint64_t[:] left_positions_mv = np.asarray(left_positions, dtype=np.uint64)
    cdef np.int64_t[:] right_values_mv = right_values
    cdef np.uint64_t[:] right_positions_mv = np.asarray(right_positions, dtype=np.uint64)
    cdef np.int64_t[:] merged_values_mv = merged_values
    cdef np.uint64_t[:] merged_positions_mv = merged_positions
    with nogil:
        _linear_merge_ordered(
            left_values_mv, left_positions_mv, right_values_mv, right_positions_mv, merged_values_mv, merged_positions_mv
        )
    return merged_values, merged_positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_merge_uint8(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, np.dtype position_dtype
):
    cdef Py_ssize_t total = left_values.shape[0] + right_values.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] merged_values = np.empty(total, dtype=np.uint8)
    cdef np.ndarray[np.uint64_t, ndim=1] merged_positions = np.empty(total, dtype=np.uint64)
    cdef np.uint8_t[:] left_values_mv = left_values
    cdef np.uint64_t[:] left_positions_mv = np.asarray(left_positions, dtype=np.uint64)
    cdef np.uint8_t[:] right_values_mv = right_values
    cdef np.uint64_t[:] right_positions_mv = np.asarray(right_positions, dtype=np.uint64)
    cdef np.uint8_t[:] merged_values_mv = merged_values
    cdef np.uint64_t[:] merged_positions_mv = merged_positions
    with nogil:
        _linear_merge_ordered(
            left_values_mv, left_positions_mv, right_values_mv, right_positions_mv, merged_values_mv, merged_positions_mv
        )
    return merged_values, merged_positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_merge_uint16(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, np.dtype position_dtype
):
    cdef Py_ssize_t total = left_values.shape[0] + right_values.shape[0]
    cdef np.ndarray[np.uint16_t, ndim=1] merged_values = np.empty(total, dtype=np.uint16)
    cdef np.ndarray[np.uint64_t, ndim=1] merged_positions = np.empty(total, dtype=np.uint64)
    cdef np.uint16_t[:] left_values_mv = left_values
    cdef np.uint64_t[:] left_positions_mv = np.asarray(left_positions, dtype=np.uint64)
    cdef np.uint16_t[:] right_values_mv = right_values
    cdef np.uint64_t[:] right_positions_mv = np.asarray(right_positions, dtype=np.uint64)
    cdef np.uint16_t[:] merged_values_mv = merged_values
    cdef np.uint64_t[:] merged_positions_mv = merged_positions
    with nogil:
        _linear_merge_ordered(
            left_values_mv, left_positions_mv, right_values_mv, right_positions_mv, merged_values_mv, merged_positions_mv
        )
    return merged_values, merged_positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_merge_uint32(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, np.dtype position_dtype
):
    cdef Py_ssize_t total = left_values.shape[0] + right_values.shape[0]
    cdef np.ndarray[np.uint32_t, ndim=1] merged_values = np.empty(total, dtype=np.uint32)
    cdef np.ndarray[np.uint64_t, ndim=1] merged_positions = np.empty(total, dtype=np.uint64)
    cdef np.uint32_t[:] left_values_mv = left_values
    cdef np.uint64_t[:] left_positions_mv = np.asarray(left_positions, dtype=np.uint64)
    cdef np.uint32_t[:] right_values_mv = right_values
    cdef np.uint64_t[:] right_positions_mv = np.asarray(right_positions, dtype=np.uint64)
    cdef np.uint32_t[:] merged_values_mv = merged_values
    cdef np.uint64_t[:] merged_positions_mv = merged_positions
    with nogil:
        _linear_merge_ordered(
            left_values_mv, left_positions_mv, right_values_mv, right_positions_mv, merged_values_mv, merged_positions_mv
        )
    return merged_values, merged_positions.astype(position_dtype, copy=False)


cdef tuple _intra_chunk_merge_uint64(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, np.dtype position_dtype
):
    cdef Py_ssize_t total = left_values.shape[0] + right_values.shape[0]
    cdef np.ndarray[np.uint64_t, ndim=1] merged_values = np.empty(total, dtype=np.uint64)
    cdef np.ndarray[np.uint64_t, ndim=1] merged_positions = np.empty(total, dtype=np.uint64)
    cdef np.uint64_t[:] left_values_mv = left_values
    cdef np.uint64_t[:] left_positions_mv = np.asarray(left_positions, dtype=np.uint64)
    cdef np.uint64_t[:] right_values_mv = right_values
    cdef np.uint64_t[:] right_positions_mv = np.asarray(right_positions, dtype=np.uint64)
    cdef np.uint64_t[:] merged_values_mv = merged_values
    cdef np.uint64_t[:] merged_positions_mv = merged_positions
    with nogil:
        _linear_merge_ordered(
            left_values_mv, left_positions_mv, right_values_mv, right_positions_mv, merged_values_mv, merged_positions_mv
        )
    return merged_values, merged_positions.astype(position_dtype, copy=False)


def intra_chunk_merge_sorted_slices(
    np.ndarray left_values, np.ndarray left_positions, np.ndarray right_values, np.ndarray right_positions, object position_dtype
):
    cdef np.dtype dtype = left_values.dtype
    cdef np.dtype pos_dtype = np.dtype(position_dtype)
    if dtype != right_values.dtype:
        raise TypeError("left_values and right_values must have the same dtype")
    if dtype == np.dtype(np.float32):
        return _intra_chunk_merge_float32(left_values, left_positions, right_values, right_positions, pos_dtype)
    if dtype == np.dtype(np.float64):
        return _intra_chunk_merge_float64(left_values, left_positions, right_values, right_positions, pos_dtype)
    if dtype == np.dtype(np.int8):
        return _intra_chunk_merge_int8(left_values, left_positions, right_values, right_positions, pos_dtype)
    if dtype == np.dtype(np.int16):
        return _intra_chunk_merge_int16(left_values, left_positions, right_values, right_positions, pos_dtype)
    if dtype == np.dtype(np.int32):
        return _intra_chunk_merge_int32(left_values, left_positions, right_values, right_positions, pos_dtype)
    if dtype == np.dtype(np.int64):
        return _intra_chunk_merge_int64(left_values, left_positions, right_values, right_positions, pos_dtype)
    if dtype == np.dtype(np.uint8):
        return _intra_chunk_merge_uint8(left_values, left_positions, right_values, right_positions, pos_dtype)
    if dtype == np.dtype(np.uint16):
        return _intra_chunk_merge_uint16(left_values, left_positions, right_values, right_positions, pos_dtype)
    if dtype == np.dtype(np.uint32):
        return _intra_chunk_merge_uint32(left_values, left_positions, right_values, right_positions, pos_dtype)
    if dtype == np.dtype(np.uint64):
        return _intra_chunk_merge_uint64(left_values, left_positions, right_values, right_positions, pos_dtype)
    if dtype == np.dtype(np.bool_):
        merged_values, merged_positions = _intra_chunk_merge_uint8(
            left_values.view(np.uint8), left_positions, right_values.view(np.uint8), right_positions, pos_dtype
        )
        return merged_values.view(np.bool_), merged_positions
    if dtype.kind in {"m", "M"}:
        merged_values, merged_positions = _intra_chunk_merge_int64(
            left_values.view(np.int64), left_positions, right_values.view(np.int64), right_positions, pos_dtype
        )
        return merged_values.view(dtype), merged_positions
    raise TypeError("unsupported dtype for intra_chunk_merge_sorted_slices")


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
