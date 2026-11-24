/*
 * Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Per-block scalar reductions for typed Menudet prefilters.
 *
 * Each evaluated block is reduced into its own partial slot; Python combines
 * the partials with NumPy. Accumulator semantics follow NumPy's reductions:
 * integer sums/products wrap (computed modulo 2**64 in uint64), floating sums
 * use NumPy's pairwise summation in the accumulator dtype, and min/max
 * propagate NaN. The accumulator is chosen by the caller (see
 * expression_evaluator.py); unsupported combinations return -1.
 * NATIVE_MIN/MAX are separate from these legacy operations: they preserve
 * the mapped dtype, prefer -0 for min / +0 for max, select the first NaN, and
 * delegate all floating blocks to miniexpr so numerical rules have one owner.
 */

#ifndef B2_BLOCK_REDUCTIONS_H
#define B2_BLOCK_REDUCTIONS_H

#include <stdint.h>
#include <math.h>
#include <string.h>

/* Optional native capability. Older builds retain the graph-reader route. */
#if defined(__has_include)
#if __has_include("miniexpr_reduce.h")
#include "miniexpr_reduce.h"
#if defined(ME_EXTREMA_REDUCE_API) && ME_EXTREMA_REDUCE_API >= 1
#define B2_HAVE_SHARED_MENUDET_REDUCE 1
#endif
#endif
#endif
#ifndef B2_HAVE_SHARED_MENUDET_REDUCE
#define B2_HAVE_SHARED_MENUDET_REDUCE 0
#endif

#if defined(ME_MOMENTS_REDUCE_API) && ME_MOMENTS_REDUCE_API >= 1
#define B2_HAVE_MOMENTS 1
#else
#define B2_HAVE_MOMENTS 0
#endif

static inline int b2_moments_available(void) {
  return B2_HAVE_MOMENTS;
}

static inline int b2_moments_combine(const double *partials, int64_t count,
                                   double ddof, int stddev, double *out) {
#if B2_HAVE_MOMENTS
  double state[4] = {0};
  for (int64_t i = 0; i < count; i++) {
    if (!me_moments_merge(state, partials + 4 * i)) return -1;
  }
  return me_moments_finish(state, ddof, stddev != 0, out) ? 0 : -1;
#else
  (void)partials; (void)count; (void)ddof; (void)stddev; (void)out;
  return -1;
#endif
}

static inline int b2_moments_merge_chunk(double *acc, const double *partials,
                                         int64_t count) {
#if B2_HAVE_MOMENTS
  for (int64_t i = 0; i < count; i++) {
    if (!me_moments_merge(acc, partials + 4 * i)) return -1;
  }
  return 0;
#else
  (void)acc; (void)partials; (void)count;
  return -1;
#endif
}

enum {
  B2_REDUCE_NONE = 0,
  B2_REDUCE_SUM,
  B2_REDUCE_PROD,
  B2_REDUCE_MIN,
  B2_REDUCE_MAX,
  B2_REDUCE_ANY,
  B2_REDUCE_ALL,
  B2_REDUCE_NATIVE_MIN,
  B2_REDUCE_NATIVE_MAX,
  B2_REDUCE_MOMENTS,
};

/* Element type codes, shared by sources and accumulators. */
enum {
  B2_RT_BOOL = 0,
  B2_RT_I8,
  B2_RT_I16,
  B2_RT_I32,
  B2_RT_I64,
  B2_RT_U8,
  B2_RT_U16,
  B2_RT_U32,
  B2_RT_U64,
  B2_RT_F32,
  B2_RT_F64,
};

#define B2_PW_BLOCKSIZE 128

/* NumPy's pairwise summation (loops_utils.h), loading ST and accumulating in AT. */
#define B2_DEFINE_PAIRWISE(NAME, ST, AT)                                              \
  static AT NAME(const ST *a, int64_t n) {                                          \
    if (n < 8) {                                                                    \
      AT res = 0;                                                                   \
      for (int64_t i = 0; i < n; i++) res += (AT)a[i];                              \
      return res;                                                                   \
    }                                                                               \
    if (n <= B2_PW_BLOCKSIZE) {                                                     \
      AT r[8];                                                                      \
      int64_t i;                                                                    \
      for (int j = 0; j < 8; j++) r[j] = (AT)a[j];                                  \
      for (i = 8; i < n - (n % 8); i += 8) {                                        \
        for (int j = 0; j < 8; j++) r[j] += (AT)a[i + j];                           \
      }                                                                             \
      AT res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));   \
      for (; i < n; i++) res += (AT)a[i];                                           \
      return res;                                                                   \
    }                                                                               \
    int64_t n2 = n / 2;                                                             \
    n2 -= n2 % 8;                                                                   \
    return NAME(a, n2) + NAME(a + n2, n - n2);                                      \
  }

/*
 * One reducer per source type. IS_FLOAT is a compile-time constant; integer
 * accumulators are rejected for floating sources (NumPy casts are lossy there).
 */
#define B2_DEFINE_REDUCER(SUFFIX, ST, IS_FLOAT, MINMAX_CODE, MINMAX_T)                      \
  B2_DEFINE_PAIRWISE(b2_pairwise_f32_##SUFFIX, ST, float)                                 \
  B2_DEFINE_PAIRWISE(b2_pairwise_f64_##SUFFIX, ST, double)                                \
  static int b2_reduce_##SUFFIX(int op, int acc, const ST *x, int64_t n, void *slot) {   \
    int64_t i;                                                                            \
    switch (op) {                                                                         \
      case B2_REDUCE_SUM:                                                                 \
        if (acc == B2_RT_F32) { *(float *)slot = b2_pairwise_f32_##SUFFIX(x, n); return 0; } \
        if (acc == B2_RT_F64) { *(double *)slot = b2_pairwise_f64_##SUFFIX(x, n); return 0; } \
        if (acc == B2_RT_U64 && !(IS_FLOAT)) {                                            \
          uint64_t s = 0;                                                                 \
          for (i = 0; i < n; i++) s += (uint64_t)x[i];                                    \
          *(uint64_t *)slot = s;                                                          \
          return 0;                                                                       \
        }                                                                                 \
        return -1;                                                                        \
      case B2_REDUCE_PROD:                                                                \
        if (acc == B2_RT_F32) {                                                           \
          float p = 1;                                                                    \
          for (i = 0; i < n; i++) p *= (float)x[i];                                       \
          *(float *)slot = p;                                                             \
          return 0;                                                                       \
        }                                                                                 \
        if (acc == B2_RT_F64) {                                                           \
          double p = 1;                                                                   \
          for (i = 0; i < n; i++) p *= (double)x[i];                                      \
          *(double *)slot = p;                                                            \
          return 0;                                                                       \
        }                                                                                 \
        if (acc == B2_RT_U64 && !(IS_FLOAT)) {                                            \
          uint64_t p = 1;                                                                 \
          for (i = 0; i < n; i++) p *= (uint64_t)x[i];                                    \
          *(uint64_t *)slot = p;                                                          \
          return 0;                                                                       \
        }                                                                                 \
        return -1;                                                                        \
      case B2_REDUCE_MIN:                                                                 \
      case B2_REDUCE_MAX: {                                                               \
        if (acc != (MINMAX_CODE) || n <= 0) return -1;                                    \
        ST m = x[0];                                                                      \
        if (op == B2_REDUCE_MIN) {                                                        \
          for (i = 1; i < n; i++) if (x[i] < m || x[i] != x[i]) m = x[i];                 \
        } else {                                                                          \
          for (i = 1; i < n; i++) if (x[i] > m || x[i] != x[i]) m = x[i];                 \
        }                                                                                 \
        *(MINMAX_T *)slot = (MINMAX_T)m;                                                  \
        return 0;                                                                         \
      }                                                                                   \
      case B2_REDUCE_ANY:                                                                 \
      case B2_REDUCE_ALL: {                                                               \
        if (acc != B2_RT_BOOL) return -1;                                                 \
        uint8_t r = op == B2_REDUCE_ALL;                                                  \
        if (r) { for (i = 0; i < n; i++) if (x[i] == 0) { r = 0; break; } }               \
        else { for (i = 0; i < n; i++) if (x[i] != 0) { r = 1; break; } }                 \
        *(uint8_t *)slot = r;                                                             \
        return 0;                                                                         \
      }                                                                                   \
      default:                                                                            \
        return -1;                                                                        \
    }                                                                                     \
  }

/* Booleans are bytes holding 0/1; min/max partials keep them as uint64. */
B2_DEFINE_REDUCER(bool, uint8_t, 0, B2_RT_U64, uint64_t)
B2_DEFINE_REDUCER(i8, int8_t, 0, B2_RT_I64, int64_t)
B2_DEFINE_REDUCER(i16, int16_t, 0, B2_RT_I64, int64_t)
B2_DEFINE_REDUCER(i32, int32_t, 0, B2_RT_I64, int64_t)
B2_DEFINE_REDUCER(i64, int64_t, 0, B2_RT_I64, int64_t)
B2_DEFINE_REDUCER(u8, uint8_t, 0, B2_RT_U64, uint64_t)
B2_DEFINE_REDUCER(u16, uint16_t, 0, B2_RT_U64, uint64_t)
B2_DEFINE_REDUCER(u32, uint32_t, 0, B2_RT_U64, uint64_t)
B2_DEFINE_REDUCER(u64, uint64_t, 0, B2_RT_U64, uint64_t)
B2_DEFINE_REDUCER(f32, float, 1, B2_RT_F64, double)
B2_DEFINE_REDUCER(f64, double, 1, B2_RT_F64, double)

#if B2_HAVE_SHARED_MENUDET_REDUCE || B2_HAVE_MOMENTS
/* Map a source code onto miniexpr's dtype enum, or ME_AUTO. */
static inline me_dtype b2_shared_dtype(int src) {
  switch (src) {
    case B2_RT_BOOL: return ME_BOOL;
    case B2_RT_I8: return ME_INT8;
    case B2_RT_I16: return ME_INT16;
    case B2_RT_I32: return ME_INT32;
    case B2_RT_I64: return ME_INT64;
    case B2_RT_U8: return ME_UINT8;
    case B2_RT_U16: return ME_UINT16;
    case B2_RT_U32: return ME_UINT32;
    case B2_RT_U64: return ME_UINT64;
    case B2_RT_F32: return ME_FLOAT32;
    case B2_RT_F64: return ME_FLOAT64;
    default: return ME_AUTO;
  }
}
#endif

/* Report which extrema rules this build uses, so callers can qualify a wheel. */
static inline int b2_shared_reduction_rules_available(void) {
  return B2_HAVE_SHARED_MENUDET_REDUCE;
}

/* All numerical behavior, including NaNs, belongs to miniexpr. */
static inline int b2_shared_extrema(int op, int src, int acc, const void *data, int64_t n, void *slot) {
#if B2_HAVE_SHARED_MENUDET_REDUCE
  me_dtype dtype = b2_shared_dtype(src);
  if (dtype == ME_AUTO || acc != src || n <= 0) return -1;
  uint64_t bits = 0;
  if (!me_extrema_float(dtype, data, (size_t)n, 0, false,
                           op == B2_REDUCE_NATIVE_MIN, &bits)) return -1;
  memcpy(slot, &bits, dtype == ME_FLOAT32 ? sizeof(float) : sizeof(double));
  return 0;
#else
  (void)op; (void)src; (void)acc; (void)data; (void)n; (void)slot;
  return -1;
#endif
}

/* Reduce n > 0 items of type `src` at `data` into the accumulator `slot`. */
static inline int b2_block_reduce(int op, int src, int acc, const void *data, int64_t n, void *slot) {
  if (op == B2_REDUCE_MOMENTS) {
#if B2_HAVE_MOMENTS
    if (acc != B2_RT_F64 || n < 0) return -1;
    return me_moments_span(b2_shared_dtype(src), data, (size_t)n, (double *)slot) ? 0 : -1;
#else
    return -1;
#endif
  }
  if (op == B2_REDUCE_NATIVE_MIN || op == B2_REDUCE_NATIVE_MAX) {
    return b2_shared_extrema(op, src, acc, data, n, slot);
  }
  switch (src) {
    case B2_RT_BOOL: return b2_reduce_bool(op, acc, (const uint8_t *)data, n, slot);
    case B2_RT_I8: return b2_reduce_i8(op, acc, (const int8_t *)data, n, slot);
    case B2_RT_I16: return b2_reduce_i16(op, acc, (const int16_t *)data, n, slot);
    case B2_RT_I32: return b2_reduce_i32(op, acc, (const int32_t *)data, n, slot);
    case B2_RT_I64: return b2_reduce_i64(op, acc, (const int64_t *)data, n, slot);
    case B2_RT_U8: return b2_reduce_u8(op, acc, (const uint8_t *)data, n, slot);
    case B2_RT_U16: return b2_reduce_u16(op, acc, (const uint16_t *)data, n, slot);
    case B2_RT_U32: return b2_reduce_u32(op, acc, (const uint32_t *)data, n, slot);
    case B2_RT_U64: return b2_reduce_u64(op, acc, (const uint64_t *)data, n, slot);
    case B2_RT_F32: return b2_reduce_f32(op, acc, (const float *)data, n, slot);
    case B2_RT_F64: return b2_reduce_f64(op, acc, (const double *)data, n, slot);
    default: return -1;
  }
}

/* Validate an (op, src, acc) combination without data. */
static inline int b2_block_reduce_supported(int op, int src, int acc) {
  int is_float = src == B2_RT_F32 || src == B2_RT_F64;
  int is_signed = src >= B2_RT_I8 && src <= B2_RT_I64;
  switch (op) {
    case B2_REDUCE_MOMENTS:
      return B2_HAVE_MOMENTS && is_float && acc == B2_RT_F64;
    case B2_REDUCE_NATIVE_MIN:
    case B2_REDUCE_NATIVE_MAX:
      return B2_HAVE_SHARED_MENUDET_REDUCE && is_float && acc == src;
    case B2_REDUCE_SUM:
    case B2_REDUCE_PROD:
      return acc == B2_RT_F32 || acc == B2_RT_F64 || (acc == B2_RT_U64 && !is_float);
    case B2_REDUCE_MIN:
    case B2_REDUCE_MAX:
      return acc == (is_float ? B2_RT_F64 : (is_signed ? B2_RT_I64 : B2_RT_U64));
    case B2_REDUCE_ANY:
    case B2_REDUCE_ALL:
      return acc == B2_RT_BOOL;
    default:
      return 0;
  }
}

static inline int b2_argextrema(int src, const void *data, int64_t n,
    int64_t origin, int minimum, void *value, int64_t *index) {
#if B2_HAVE_SHARED_MENUDET_REDUCE
  me_dtype dtype = b2_shared_dtype(src);
  me_argextrema_state state = {0};
  if (n <= 0 || !me_argextrema_update(dtype, data, (size_t)n, origin, minimum != 0, &state)) return -1;
  memcpy(value, &state.value, me_extrema_itemsize(dtype));
  *index = state.index;
  return 0;
#else
  (void)src; (void)data; (void)n; (void)origin; (void)minimum; (void)value; (void)index;
  return -1;
#endif
}

static inline int b2_argextrema_combine(int src, const void *values,
    const int64_t *indices, int64_t n, int minimum, int64_t *index) {
#if B2_HAVE_SHARED_MENUDET_REDUCE
  me_dtype dtype = b2_shared_dtype(src);
  size_t width = me_extrema_itemsize(dtype);
  if (!width || n <= 0) return -1;
  me_argextrema_state state = {0};
  for (int64_t i = 0; i < n; i++) {
    uint64_t bits = 0;
    memcpy(&bits, (const unsigned char *)values + i * width, width);
    if (!me_argextrema_merge(dtype, &state, bits, indices[i], minimum != 0)) return -1;
  }
  *index = state.index;
  return 0;
#else
  (void)src; (void)values; (void)indices; (void)n; (void)minimum; (void)index;
  return -1;
#endif
}

#endif /* B2_BLOCK_REDUCTIONS_H */
