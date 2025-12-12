/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2021  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  NumPy Integration Utilities for MiniExpr

  This file provides conversion functions between miniexpr dtypes
  and NumPy type numbers for Python bindings.
**********************************************************************/

#ifndef MINIEXPR_NUMPY_H
#define MINIEXPR_NUMPY_H

#include <stdio.h>
#include "miniexpr.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Convert miniexpr dtype to NumPy type number
 *
 * Returns the NumPy dtype.num value corresponding to a miniexpr dtype.
 * Returns -1 for ME_AUTO (which has no NumPy equivalent).
 *
 * Example:
 *   int numpy_num = me_dtype_to_numpy(ME_INT64);  // Returns 7
 */
static inline int me_dtype_to_numpy(me_dtype dtype) {
    static const int numpy_type_nums[] = {
        -1,  // ME_AUTO (0) -> No NumPy equivalent
        0,   // ME_BOOL (1) -> NPY_BOOL
        1,   // ME_INT8 (2) -> NPY_BYTE
        3,   // ME_INT16 (3) -> NPY_SHORT
        5,   // ME_INT32 (4) -> NPY_INT
        7,   // ME_INT64 (5) -> NPY_LONGLONG
        2,   // ME_UINT8 (6) -> NPY_UBYTE
        4,   // ME_UINT16 (7) -> NPY_USHORT
        6,   // ME_UINT32 (8) -> NPY_UINT
        8,   // ME_UINT64 (9) -> NPY_ULONGLONG
        11,  // ME_FLOAT32 (10) -> NPY_FLOAT
        12,  // ME_FLOAT64 (11) -> NPY_DOUBLE
        14,  // ME_COMPLEX64 (12) -> NPY_CFLOAT
        15   // ME_COMPLEX128 (13) -> NPY_CDOUBLE
    };

    if (dtype >= 0 && dtype <= ME_COMPLEX128) {
        return numpy_type_nums[dtype];
    }
    return -1;  // Invalid dtype
}

/* Get a string name for a NumPy type number (for error messages)
 *
 * Returns a human-readable name for common NumPy types.
 * Returns "unknown" for unsupported types.
 */
static inline const char* me_numpy_type_name(int numpy_type_num) {
    switch (numpy_type_num) {
        case 0:  return "bool";
        case 1:  return "int8";
        case 2:  return "uint8";
        case 3:  return "int16";
        case 4:  return "uint16";
        case 5:  return "int32";
        case 6:  return "uint32";
        case 7:  return "int64";
        case 8:  return "uint64";
        case 9:  return "float16";      // Not supported
        case 10: return "longdouble";   // Not supported
        case 11: return "float32";
        case 12: return "float64";
        case 13: return "clongdouble";  // Not supported
        case 14: return "complex64";
        case 15: return "complex128";
        default: return "unknown";
    }
}

/* Convert NumPy type number to miniexpr dtype
 *
 * Returns the miniexpr dtype corresponding to a NumPy dtype.num value.
 * Returns -1 and prints an error message for unsupported NumPy types.
 *
 * Example:
 *   me_dtype dtype = me_dtype_from_numpy(7);  // Returns ME_INT64
 *   if (dtype < 0) {
 *       // Unsupported type, error already printed
 *       return NULL;
 *   }
 *
 * Note: This function only supports the subset of NumPy types that
 * miniexpr implements. Other types (float16, longdouble, etc.) will
 * return -1 and print an error message to stderr.
 */
static inline me_dtype me_dtype_from_numpy(int numpy_type_num) {
    switch (numpy_type_num) {
        case 0:  return ME_BOOL;
        case 1:  return ME_INT8;
        case 2:  return ME_UINT8;
        case 3:  return ME_INT16;
        case 4:  return ME_UINT16;
        case 5:  return ME_INT32;
        case 6:  return ME_UINT32;
        case 7:  return ME_INT64;
        case 8:  return ME_UINT64;
        case 11: return ME_FLOAT32;
        case 12: return ME_FLOAT64;
        case 14: return ME_COMPLEX64;
        case 15: return ME_COMPLEX128;
        default:
            fprintf(stderr, "Error: Unsupported NumPy dtype.num = %d (%s)\n",
                    numpy_type_num, me_numpy_type_name(numpy_type_num));
            return -1;  // Return -1 to indicate error
    }
}

/* Check if a NumPy type is supported by miniexpr
 *
 * Returns 1 if the NumPy type number is supported, 0 otherwise.
 * This function does not print error messages.
 *
 * Example:
 *   if (me_numpy_type_supported(numpy_dtype_num)) {
 *       // Can use this type with miniexpr
 *   }
 */
static inline int me_numpy_type_supported(int numpy_type_num) {
    // Check directly without calling me_dtype_from_numpy to avoid error messages
    switch (numpy_type_num) {
        case 0:   // bool
        case 1:   // int8
        case 2:   // uint8
        case 3:   // int16
        case 4:   // uint16
        case 5:   // int32
        case 6:   // uint32
        case 7:   // int64
        case 8:   // uint64
        case 11:  // float32
        case 12:  // float64
        case 14:  // complex64
        case 15:  // complex128
            return 1;
        default:
            return 0;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* MINIEXPR_NUMPY_H */
