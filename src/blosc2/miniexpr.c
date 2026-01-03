/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

// Loosely based on https://github.com/CodePlea/tinyexpr. License follows:
// SPDX-License-Identifier: Zlib
/*
 * TINYEXPR - Tiny recursive descent parser and evaluation engine in C
 *
 * Copyright (c) 2015-2020 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgement in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* COMPILE TIME OPTIONS */

/* Exponentiation associativity:
For a**b**c = (a**b)**c and -a**b = (-a)**b do nothing.
For a**b**c = a**(b**c) and -a**b = -(a**b) uncomment the next line.*/
/* #define ME_POW_FROM_RIGHT */

/* Logarithms
For log = natural log do nothing (NumPy compatible)
For log = base 10 log comment the next line. */
#define ME_NAT_LOG

#include "miniexpr.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#if defined(__SSE2__) || defined(__SSE__) || defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#define IVDEP
#else
#define IVDEP _Pragma("GCC ivdep")
#endif

#include <complex.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define float_complex _Fcomplex
#define double_complex _Dcomplex
// And it doesn't support standard operators for them in C
static inline _Fcomplex add_c64(_Fcomplex a, _Fcomplex b) {
    return _FCbuild(crealf(a) + crealf(b), cimagf(a) + cimagf(b));
}
static inline _Fcomplex sub_c64(_Fcomplex a, _Fcomplex b) {
    return _FCbuild(crealf(a) - crealf(b), cimagf(a) - cimagf(b));
}
static inline _Fcomplex neg_c64(_Fcomplex a) { return _FCbuild(-crealf(a), -cimagf(a)); }
static inline _Fcomplex mul_c64(_Fcomplex a, _Fcomplex b) {
    return _FCbuild(crealf(a) * crealf(b) - cimagf(a) * cimagf(b), crealf(a) * cimagf(b) + cimagf(a) * crealf(b));
}
static inline _Fcomplex div_c64(_Fcomplex a, _Fcomplex b) {
    float denom = crealf(b) * crealf(b) + cimagf(b) * cimagf(b);
    return _FCbuild((crealf(a) * crealf(b) + cimagf(a) * cimagf(b)) / denom,
                    (cimagf(a) * crealf(b) - crealf(a) * cimagf(b)) / denom);
}
static inline _Dcomplex add_c128(_Dcomplex a, _Dcomplex b) { return _Cbuild(creal(a) + creal(b), cimag(a) + cimag(b)); }
static inline _Dcomplex sub_c128(_Dcomplex a, _Dcomplex b) { return _Cbuild(creal(a) - creal(b), cimag(a) - cimag(b)); }
static inline _Dcomplex neg_c128(_Dcomplex a) { return _Cbuild(-creal(a), -cimag(a)); }
static inline _Dcomplex mul_c128(_Dcomplex a, _Dcomplex b) {
    return _Cbuild(creal(a) * creal(b) - cimag(a) * cimag(b), creal(a) * cimag(b) + cimag(a) * creal(b));
}
static inline _Dcomplex div_c128(_Dcomplex a, _Dcomplex b) {
    double denom = creal(b) * creal(b) + cimag(b) * cimag(b);
    return _Cbuild((creal(a) * creal(b) + cimag(a) * cimag(b)) / denom,
                   (cimag(a) * creal(b) - creal(a) * cimag(b)) / denom);
}
#else
#define float_complex float _Complex
#define double_complex double _Complex
#define add_c64(a, b) ((a) + (b))
#define sub_c64(a, b) ((a) - (b))
#define neg_c64(a) (-(a))
#define mul_c64(a, b) ((a) * (b))
#define div_c64(a, b) ((a) / (b))
#define add_c128(a, b) ((a) + (b))
#define sub_c128(a, b) ((a) - (b))
#define neg_c128(a) (-(a))
#define mul_c128(a, b) ((a) * (b))
#define div_c128(a, b) ((a) / (b))
#endif

#if defined(_MSC_VER) && !defined(__clang__)
/* Wrappers for complex functions to handle MSVC's _Fcomplex/_Dcomplex */
static inline float _Complex me_cpowf(float _Complex a, float _Complex b) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua, ub, ur;
    ua.c = a;
    ub.c = b;
    ur.m = cpowf(ua.m, ub.m);
    return ur.c;
}
static inline double _Complex me_cpow(double _Complex a, double _Complex b) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua, ub, ur;
    ua.c = a;
    ub.c = b;
    ur.m = cpow(ua.m, ub.m);
    return ur.c;
}
static inline float _Complex me_csqrtf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = csqrtf(ua.m);
    return ur.c;
}
static inline double _Complex me_csqrt(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = csqrt(ua.m);
    return ur.c;
}
static inline float _Complex me_cexpf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = cexpf(ua.m);
    return ur.c;
}
static inline double _Complex me_cexp(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = cexp(ua.m);
    return ur.c;
}
static inline float _Complex me_clogf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = clogf(ua.m);
    return ur.c;
}
static inline double _Complex me_clog(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = clog(ua.m);
    return ur.c;
}
static inline float me_cabsf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua;
    ua.c = a;
    return cabsf(ua.m);
}
static inline double me_cabs(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua;
    ua.c = a;
    return cabs(ua.m);
}
static inline float me_cimagf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua;
    ua.c = a;
    return cimagf(ua.m);
}
static inline double me_cimag(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua;
    ua.c = a;
    return cimag(ua.m);
}
static inline float me_crealf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua;
    ua.c = a;
    return crealf(ua.m);
}
static inline double me_creal(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua;
    ua.c = a;
    return creal(ua.m);
}
static inline float _Complex me_conjf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = conjf(ua.m);
    return ur.c;
}
static inline double _Complex me_conj(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = conj(ua.m);
    return ur.c;
}
#else
#if defined(_MSC_VER) && defined(__clang__)
#define me_cimagf __builtin_cimagf
#define me_cimag __builtin_cimag
#define me_crealf __builtin_crealf
#define me_creal __builtin_creal
#define me_conjf __builtin_conjf
#define me_conj __builtin_conj
#define me_cpowf __builtin_cpowf
#define me_cpow __builtin_cpow
#define me_csqrtf __builtin_csqrtf
#define me_csqrt __builtin_csqrt
#define me_cexpf __builtin_cexpf
#define me_cexp __builtin_cexp
#define me_clogf __builtin_clogf
#define me_clog __builtin_clog
#define me_cabsf __builtin_cabsf
#define me_cabs __builtin_cabs
#else
#define me_cpowf cpowf
#define me_cpow cpow
#define me_csqrtf csqrtf
#define me_csqrt csqrt
#define me_cexpf cexpf
#define me_cexp cexp
#define me_clogf clogf
#define me_clog clog
#define me_cabsf cabsf
#define me_cabs cabs
#define me_cimagf cimagf
#define me_cimag cimag
#define me_crealf crealf
#define me_creal creal
#define me_conjf conjf
#define me_conj conj
#endif
#endif

/* Type-specific cast and comparison macros to handle MSVC complex structs */
#define TO_TYPE_bool(x) (bool)(x)
#define TO_TYPE_i8(x) (int8_t)(x)
#define TO_TYPE_i16(x) (int16_t)(x)
#define TO_TYPE_i32(x) (int32_t)(x)
#define TO_TYPE_i64(x) (int64_t)(x)
#define TO_TYPE_u8(x) (uint8_t)(x)
#define TO_TYPE_u16(x) (uint16_t)(x)
#define TO_TYPE_u32(x) (uint32_t)(x)
#define TO_TYPE_u64(x) (uint64_t)(x)
#define TO_TYPE_f32(x) (float)(x)
#define TO_TYPE_f64(x) (double)(x)

#define FROM_TYPE_bool(x) (double)(x)
#define FROM_TYPE_i8(x) (double)(x)
#define FROM_TYPE_i16(x) (double)(x)
#define FROM_TYPE_i32(x) (double)(x)
#define FROM_TYPE_i64(x) (double)(x)
#define FROM_TYPE_u8(x) (double)(x)
#define FROM_TYPE_u16(x) (double)(x)
#define FROM_TYPE_u32(x) (double)(x)
#define FROM_TYPE_u64(x) (double)(x)
#define FROM_TYPE_f32(x) (double)(x)
#define FROM_TYPE_f64(x) (double)(x)

#define IS_NONZERO_bool(x) (x)
#define IS_NONZERO_i8(x) ((x) != 0)
#define IS_NONZERO_i16(x) ((x) != 0)
#define IS_NONZERO_i32(x) ((x) != 0)
#define IS_NONZERO_i64(x) ((x) != 0)
#define IS_NONZERO_u8(x) ((x) != 0)
#define IS_NONZERO_u16(x) ((x) != 0)
#define IS_NONZERO_u32(x) ((x) != 0)
#define IS_NONZERO_u64(x) ((x) != 0)
#define IS_NONZERO_f32(x) ((x) != 0.0f)
#define IS_NONZERO_f64(x) ((x) != 0.0)

#if defined(_MSC_VER) && !defined(__clang__)
#define TO_TYPE_c64(x) _FCbuild((float)(x), 0.0f)
#define TO_TYPE_c128(x) _Cbuild((double)(x), 0.0)
#define FROM_TYPE_c64(x) (double)crealf(x)
#define FROM_TYPE_c128(x) (double)creal(x)
#define IS_NONZERO_c64(x) (crealf(x) != 0.0f || cimagf(x) != 0.0f)
#define IS_NONZERO_c128(x) (creal(x) != 0.0 || cimag(x) != 0.0)

/* Helper macros for complex-to-complex conversions */
#define CONV_c64_to_c128(x) _Cbuild((double)crealf(x), (double)cimagf(x))
#define TO_TYPE_c128_from_c64(x) CONV_c64_to_c128(x)
#else
#define TO_TYPE_c64(x) (float_complex)(x)
#define TO_TYPE_c128(x) (double_complex)(x)
#define FROM_TYPE_c64(x) (double)me_crealf(x)
#define FROM_TYPE_c128(x) (double)me_creal(x)
#define IS_NONZERO_c64(x) (me_crealf(x) != 0.0f || me_cimagf(x) != 0.0f)
#define IS_NONZERO_c128(x) (me_creal(x) != 0.0 || me_cimag(x) != 0.0)
#define TO_TYPE_c128_from_c64(x) (double_complex)(x)
#endif

#include <assert.h>

#ifndef NAN
#define NAN (0.0/0.0)
#endif

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif


typedef double (*me_fun2)(double, double);

#if defined(_WIN32) || defined(_WIN64)
static bool has_complex_node(const me_expr* n);
static bool has_complex_input(const me_expr* n);
#endif

enum {
    TOK_NULL = ME_CLOSURE7 + 1, TOK_ERROR, TOK_END, TOK_SEP,
    TOK_OPEN, TOK_CLOSE, TOK_NUMBER, TOK_VARIABLE, TOK_INFIX,
    TOK_BITWISE, TOK_SHIFT, TOK_COMPARE, TOK_POW
};

/* Internal definition of me_expr (opaque to users) */
struct me_expr {
    int type;

    union {
        double value;
        const void* bound;
        const void* function;
    };

    /* Vector operation info */
    void* output; // Generic pointer (can be float* or double*)
    int nitems;
    me_dtype dtype; // Data type for this expression (result type after promotion)
    me_dtype input_dtype; // Original input type (for variables/constants)
    /* Bytecode info (for fused evaluation) */
    void* bytecode; // Pointer to compiled bytecode
    int ncode; // Number of instructions
    void* parameters[1]; // Must be last (flexible array member)
};


/* Type promotion table following NumPy rules */
/* Note: ME_AUTO (0) should never appear in type promotion, so we index from 1 */
static const me_dtype type_promotion_table[13][13] = {
    /* Rows: left operand, Columns: right operand */
    /* BOOL,  INT8,    INT16,   INT32,   INT64,   UINT8,   UINT16,  UINT32,  UINT64,  FLOAT32, FLOAT64, COMPLEX64, COMPLEX128 */
    {
        ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* BOOL */
    {
        ME_INT8, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_INT16, ME_INT32, ME_INT64, ME_FLOAT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* INT8 */
    {
        ME_INT16, ME_INT16, ME_INT16, ME_INT32, ME_INT64, ME_INT32, ME_INT32, ME_INT64, ME_FLOAT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* INT16 */
    {
        ME_INT32, ME_INT32, ME_INT32, ME_INT32, ME_INT64, ME_INT64, ME_INT64, ME_INT64, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* INT32 */
    {
        ME_INT64, ME_INT64, ME_INT64, ME_INT64, ME_INT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* INT64 */
    {
        ME_UINT8, ME_INT16, ME_INT32, ME_INT64, ME_FLOAT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* UINT8 */
    {
        ME_UINT16, ME_INT32, ME_INT32, ME_INT64, ME_FLOAT64, ME_UINT16, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* UINT16 */
    {
        ME_UINT32, ME_INT64, ME_INT64, ME_INT64, ME_FLOAT64, ME_UINT32, ME_UINT32, ME_UINT32, ME_UINT64, ME_FLOAT64,
        ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* UINT32 */
    {
        ME_UINT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_UINT64, ME_UINT64, ME_UINT64, ME_UINT64,
        ME_FLOAT64, ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* UINT64 */
    {
        ME_FLOAT32, ME_FLOAT32, ME_FLOAT32, ME_FLOAT64, ME_FLOAT64, ME_FLOAT32, ME_FLOAT32, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT32, ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* FLOAT32 */
    {
        ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT64, ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* FLOAT64 */
    {
        ME_COMPLEX64, ME_COMPLEX64, ME_COMPLEX64, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX64, ME_COMPLEX64,
        ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX64, ME_COMPLEX128, ME_COMPLEX64, ME_COMPLEX128
    }, /* COMPLEX64 */
    {
        ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128,
        ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128
    } /* COMPLEX128 */
};

/* Promote two types according to NumPy rules */
static me_dtype promote_types(me_dtype a, me_dtype b) {
    // ME_AUTO should have been resolved during compilation
    if (a == ME_AUTO || b == ME_AUTO) {
        fprintf(stderr, "FATAL: ME_AUTO in type promotion (a=%d, b=%d). This is a bug.\n", a, b);
#ifdef NDEBUG
        abort(); // Release build: terminate immediately
#else
        assert(0 && "ME_AUTO should be resolved during compilation"); // Debug: trigger debugger
#endif
    }

    // Adjust indices since table starts at ME_BOOL (index 1), not ME_AUTO (index 0)
    int a_idx = a - 1;
    int b_idx = b - 1;
    if (a_idx >= 0 && a_idx < 13 && b_idx >= 0 && b_idx < 13) {
        return type_promotion_table[a_idx][b_idx];
    }
    fprintf(stderr, "WARNING: Invalid dtype in type promotion (a=%d, b=%d). Falling back to FLOAT64.\n", a, b);
    return ME_FLOAT64; // Fallback for out-of-range types
}

static bool is_integer_dtype(me_dtype dt) {
    return dt >= ME_INT8 && dt <= ME_UINT64;
}

static bool is_float_dtype(me_dtype dt) {
    return dt == ME_FLOAT32 || dt == ME_FLOAT64;
}

static bool is_complex_dtype(me_dtype dt) {
    return dt == ME_COMPLEX64 || dt == ME_COMPLEX128;
}

static double sum_reduce(double x);
static double prod_reduce(double x);
static double any_reduce(double x);
static double all_reduce(double x);

static me_dtype reduction_output_dtype(me_dtype dt, const void* func) {
    if (func == (void*)any_reduce || func == (void*)all_reduce) {
        return ME_BOOL;
    }
    if (func == (void*)sum_reduce || func == (void*)prod_reduce) {
        if (dt == ME_BOOL) {
            return ME_INT64;
        }
        if (dt >= ME_UINT8 && dt <= ME_UINT64) {
            return ME_UINT64;
        }
        if (dt >= ME_INT8 && dt <= ME_INT64) {
            return ME_INT64;
        }
    }
    return dt;
}

/* Get size of a type in bytes */
static size_t dtype_size(me_dtype dtype) {
    switch (dtype) {
    case ME_BOOL: return sizeof(bool);
    case ME_INT8: return sizeof(int8_t);
    case ME_INT16: return sizeof(int16_t);
    case ME_INT32: return sizeof(int32_t);
    case ME_INT64: return sizeof(int64_t);
    case ME_UINT8: return sizeof(uint8_t);
    case ME_UINT16: return sizeof(uint16_t);
    case ME_UINT32: return sizeof(uint32_t);
    case ME_UINT64: return sizeof(uint64_t);
    case ME_FLOAT32: return sizeof(float);
    case ME_FLOAT64: return sizeof(double);
    case ME_COMPLEX64: return sizeof(float _Complex);
    case ME_COMPLEX128: return sizeof(double _Complex);
    default: return 0;
    }
}


enum { ME_CONSTANT = 1 };


typedef struct state {
    const char* start;
    const char* next;
    int type;

    union {
        double value;
        const double* bound;
        const void* function;
    };

    void* context;
    me_dtype dtype; // Type of current token
    me_dtype target_dtype; // Target dtype for the overall expression

    const me_variable* lookup;
    int lookup_len;
} state;


#define TYPE_MASK(TYPE) ((TYPE)&0x0000001F)

#define IS_PURE(TYPE) (((TYPE) & ME_FLAG_PURE) != 0)
#define IS_FUNCTION(TYPE) (((TYPE) & ME_FUNCTION0) != 0)
#define IS_CLOSURE(TYPE) (((TYPE) & ME_CLOSURE0) != 0)
#define ARITY(TYPE) ( ((TYPE) & (ME_FUNCTION0 | ME_CLOSURE0)) ? ((TYPE) & 0x00000007) : 0 )
#define NEW_EXPR(type, ...) new_expr((type), (const me_expr*[]){__VA_ARGS__})
#define CHECK_NULL(ptr, ...) if ((ptr) == NULL) { __VA_ARGS__; return NULL; }

/* Forward declarations */
static me_expr* new_expr(const int type, const me_expr* parameters[]);
static me_dtype infer_output_type(const me_expr* n);
static void private_eval(const me_expr* n);
static void eval_reduction(const me_expr* n, int output_nitems);
static double conj_wrapper(double x);
static double imag_wrapper(double x);
static double real_wrapper(double x);
static double round_wrapper(double x);
static double sum_reduce(double x);
static double prod_reduce(double x);
static double any_reduce(double x);
static double all_reduce(double x);
static double min_reduce(double x);
static double max_reduce(double x);
static double sign(double x);
static double square(double x);
static double trunc_wrapper(double x);
static double where_scalar(double c, double x, double y);

static bool is_reduction_function(const void* func) {
    return func == (void*)sum_reduce || func == (void*)prod_reduce ||
        func == (void*)min_reduce || func == (void*)max_reduce ||
        func == (void*)any_reduce || func == (void*)all_reduce;
}

static bool is_reduction_node(const me_expr* n) {
    return n && IS_FUNCTION(n->type) && ARITY(n->type) == 1 &&
        is_reduction_function(n->function);
}

static bool contains_reduction(const me_expr* n) {
    if (!n) return false;
    if (is_reduction_node(n)) return true;

    switch (TYPE_MASK(n->type)) {
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (contains_reduction((const me_expr*)n->parameters[i])) {
                    return true;
                }
            }
            return false;
        }
    default:
        return false;
    }
}

static bool reduction_usage_is_valid(const me_expr* n) {
    if (!n) return true;
    if (is_reduction_node(n)) {
        me_expr* arg = (me_expr*)n->parameters[0];
        if (!arg) return false;
        if (contains_reduction(arg)) return false;
        me_dtype arg_type = infer_output_type(arg);
        if (n->function == (void*)min_reduce || n->function == (void*)max_reduce) {
            if (arg_type == ME_COMPLEX64 || arg_type == ME_COMPLEX128) {
                return false;
            }
        }
        return true;
    }

    switch (TYPE_MASK(n->type)) {
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (!reduction_usage_is_valid((const me_expr*)n->parameters[i])) {
                    return false;
                }
            }
            return true;
        }
    default:
        return true;
    }
}

/* Infer computation type from expression tree (for evaluation) */
static me_dtype infer_result_type(const me_expr* n) {
    if (!n) return ME_FLOAT64;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        return n->dtype;

    case ME_VARIABLE:
        return n->dtype;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            if (is_reduction_node(n)) {
                me_dtype param_type = infer_result_type((const me_expr*)n->parameters[0]);
                return reduction_output_dtype(param_type, n->function);
            }
            // Special case: imag() and real() return real type from complex input
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1) {
                if (n->function == (void*)imag_wrapper || n->function == (void*)real_wrapper) {
                    me_dtype param_type = infer_result_type((const me_expr*)n->parameters[0]);
                    if (param_type == ME_COMPLEX64) {
                        return ME_FLOAT32;
                    }
                    else if (param_type == ME_COMPLEX128) {
                        return ME_FLOAT64;
                    }
                    // If input is not complex, return as-is (shouldn't happen, but be safe)
                    return param_type;
                }
            }

            // For comparisons with ME_BOOL output, we still need to infer the
            // computation type from operands (e.g., float64 for float inputs).
            // Don't return ME_BOOL early - let the operand types determine
            // the computation type.

            const int arity = ARITY(n->type);
            me_dtype result = ME_BOOL;

            for (int i = 0; i < arity; i++) {
                me_dtype param_type = infer_result_type((const me_expr*)n->parameters[i]);
                result = promote_types(result, param_type);
            }

            return result;
        }
    }

    return ME_FLOAT64;
}

/* Infer logical output type from expression tree (for compilation with ME_AUTO) */
static me_dtype infer_output_type(const me_expr* n) {
    if (!n) return ME_FLOAT64;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        return n->dtype;

    case ME_VARIABLE:
        return n->dtype;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            if (is_reduction_node(n)) {
                me_dtype param_type = infer_output_type((const me_expr*)n->parameters[0]);
                return reduction_output_dtype(param_type, n->function);
            }
            // Special case: imag() and real() return real type from complex input
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1) {
                if (n->function == (void*)imag_wrapper || n->function == (void*)real_wrapper) {
                    me_dtype param_type = infer_output_type((const me_expr*)n->parameters[0]);
                    if (param_type == ME_COMPLEX64) {
                        return ME_FLOAT32;
                    }
                    else if (param_type == ME_COMPLEX128) {
                        return ME_FLOAT64;
                    }
                    // If input is not complex, return as-is (shouldn't happen, but be safe)
                    return param_type;
                }
            }

            // Special case: where(cond, x, y) -> promote(x, y), regardless of cond type.
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 3 &&
                n->function == (void*)where_scalar) {
                me_dtype x_type = infer_output_type((const me_expr*)n->parameters[1]);
                me_dtype y_type = infer_output_type((const me_expr*)n->parameters[2]);
                return promote_types(x_type, y_type);
            }

            // If this node is a comparison (dtype == ME_BOOL set during parsing),
            // the output type is ME_BOOL
            if (n->dtype == ME_BOOL) {
                return ME_BOOL;
            }

            // Otherwise, infer from operands
            const int arity = ARITY(n->type);
            me_dtype result = ME_BOOL;

            for (int i = 0; i < arity; i++) {
                me_dtype param_type = infer_output_type((const me_expr*)n->parameters[i]);
                result = promote_types(result, param_type);
            }

            return result;
        }
    }

    return ME_FLOAT64;
}

/* Apply type promotion to a binary operation node */
static me_expr* create_conversion_node(me_expr* source, me_dtype target_dtype) {
    /* Create a unary conversion node that converts source to target_dtype */
    me_expr* conv = NEW_EXPR(ME_FUNCTION1 | ME_FLAG_PURE, source);
    if (conv) {
        conv->function = NULL; // Mark as conversion
        conv->dtype = target_dtype;
        conv->input_dtype = source->dtype;
    }
    return conv;
}

static void apply_type_promotion(me_expr* node) {
    if (!node || ARITY(node->type) < 2) return;

    me_expr* left = (me_expr*)node->parameters[0];
    me_expr* right = (me_expr*)node->parameters[1];

    if (left && right) {
        me_dtype left_type = left->dtype;
        me_dtype right_type = right->dtype;
        me_dtype promoted = promote_types(left_type, right_type);

        // Store the promoted output type
        node->dtype = promoted;

        // TODO: Conversion nodes not fully implemented yet
        // See TYPE_PROMOTION_IMPLEMENTATION.md for details
        /*
        // Insert conversion nodes if needed
        if (left_type != promoted) {
            me_expr *conv_left = creame_conversion_node(left, promoted);
            if (conv_left) {
                node->parameters[0] = conv_left;
            }
        }

        if (right_type != promoted) {
            me_expr *conv_right = creame_conversion_node(right, promoted);
            if (conv_right) {
                node->parameters[1] = conv_right;
            }
        }
        */
    }
}

static me_expr* new_expr(const int type, const me_expr* parameters[]) {
    const int arity = ARITY(type);
    const int psize = sizeof(void*) * arity;
    const int size = (sizeof(me_expr) - sizeof(void*)) + psize + (IS_CLOSURE(type) ? sizeof(void*) : 0);
    me_expr* ret = malloc(size);
    CHECK_NULL(ret);

    memset(ret, 0, size);
    if (arity && parameters) {
        memcpy(ret->parameters, parameters, psize);
    }
    ret->type = type;
    ret->bound = 0;
    ret->output = NULL;
    ret->nitems = 0;
    ret->dtype = ME_FLOAT64; // Default to double
    ret->bytecode = NULL;
    ret->ncode = 0;
    return ret;
}


void me_free_parameters(me_expr* n) {
    if (!n) return;
    switch (TYPE_MASK(n->type)) {
    case ME_FUNCTION7:
    case ME_CLOSURE7:
        if (n->parameters[6] && ((me_expr*)n->parameters[6])->output &&
            ((me_expr*)n->parameters[6])->output != n->output) {
            free(((me_expr*)n->parameters[6])->output);
        }
        me_free(n->parameters[6]);
    case ME_FUNCTION6:
    case ME_CLOSURE6:
        if (n->parameters[5] && ((me_expr*)n->parameters[5])->output &&
            ((me_expr*)n->parameters[5])->output != n->output) {
            free(((me_expr*)n->parameters[5])->output);
        }
        me_free(n->parameters[5]);
    case ME_FUNCTION5:
    case ME_CLOSURE5:
        if (n->parameters[4] && ((me_expr*)n->parameters[4])->output &&
            ((me_expr*)n->parameters[4])->output != n->output) {
            free(((me_expr*)n->parameters[4])->output);
        }
        me_free(n->parameters[4]);
    case ME_FUNCTION4:
    case ME_CLOSURE4:
        if (n->parameters[3] && ((me_expr*)n->parameters[3])->output &&
            ((me_expr*)n->parameters[3])->output != n->output) {
            free(((me_expr*)n->parameters[3])->output);
        }
        me_free(n->parameters[3]);
    case ME_FUNCTION3:
    case ME_CLOSURE3:
        if (n->parameters[2] && ((me_expr*)n->parameters[2])->output &&
            ((me_expr*)n->parameters[2])->output != n->output) {
            free(((me_expr*)n->parameters[2])->output);
        }
        me_free(n->parameters[2]);
    case ME_FUNCTION2:
    case ME_CLOSURE2:
        if (n->parameters[1] && ((me_expr*)n->parameters[1])->output &&
            ((me_expr*)n->parameters[1])->output != n->output) {
            free(((me_expr*)n->parameters[1])->output);
        }
        me_free(n->parameters[1]);
    case ME_FUNCTION1:
    case ME_CLOSURE1:
        if (n->parameters[0] && ((me_expr*)n->parameters[0])->output &&
            ((me_expr*)n->parameters[0])->output != n->output) {
            free(((me_expr*)n->parameters[0])->output);
        }
        me_free(n->parameters[0]);
    }
}


void me_free(me_expr* n) {
    if (!n) return;
    me_free_parameters(n);
    if (n->bytecode) {
        free(n->bytecode);
    }
    free(n);
}


static double pi(void) { return 3.14159265358979323846; }
static double e(void) { return 2.71828182845904523536; }

/* Wrapper for expm1: exp(x) - 1, more accurate for small x */
static double expm1_wrapper(double x) { return expm1(x); }

/* Wrapper for log1p: log(1 + x), more accurate for small x */
static double log1p_wrapper(double x) { return log1p(x); }

/* Wrapper for log2: base-2 logarithm */
static double log2_wrapper(double x) { return log2(x); }

/* logaddexp: log(exp(a) + exp(b)), numerically stable */
static double logaddexp(double a, double b) {
    if (a == b) {
        return a + log1p(1.0); // log(2*exp(a)) = a + log(2)
    }
    double max_val = (a > b) ? a : b;
    double min_val = (a > b) ? b : a;
    return max_val + log1p(exp(min_val - max_val));
}

/* Forward declarations for complex operations */
/* (Already declared above) */

/* Wrapper functions for complex operations (for function pointer compatibility) */
/* These are placeholders - actual implementation is in vector functions */
static double conj_wrapper(double x) { return x; }

static double imag_wrapper(double x) {
    (void)x;
    return 0.0;
}

/* Wrapper for round: round to nearest integer */
static double round_wrapper(double x) { return round(x); }

/* sign: returns -1.0, 0.0, or 1.0 based on sign of x */
static double sign(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

/* square: x * x */
static double square(double x) { return x * x; }

/* Wrapper for trunc: truncate towards zero */
static double trunc_wrapper(double x) { return trunc(x); }

/* Scalar helper for where(), used only in generic slow path */
static double where_scalar(double c, double x, double y) {
    return (c != 0.0) ? x : y;
}

static double real_wrapper(double x) { return x; }

static double fac(double a) {
    /* simplest version of fac */
    if (a < 0.0)
        return NAN;
    if (a > UINT_MAX)
        return INFINITY;
    unsigned int ua = (unsigned int)(a);
    unsigned long int result = 1, i;
    for (i = 1; i <= ua; i++) {
        if (i > ULONG_MAX / result)
            return INFINITY;
        result *= i;
    }
    return (double)result;
}

static double ncr(double n, double r) {
    if (n < 0.0 || r < 0.0 || n < r) return NAN;
    if (n > UINT_MAX || r > UINT_MAX) return INFINITY;
    unsigned long int un = (unsigned int)(n), ur = (unsigned int)(r), i;
    unsigned long int result = 1;
    if (ur > un / 2) ur = un - ur;
    for (i = 1; i <= ur; i++) {
        if (result > ULONG_MAX / (un - ur + i))
            return INFINITY;
        result *= un - ur + i;
        result /= i;
    }
    return result;
}

static double npr(double n, double r) { return ncr(n, r) * fac(r); }

#ifdef _MSC_VER
#pragma function (ceil)
#pragma function (floor)
#endif

static const me_variable functions[] = {
    /* must be in alphabetical order */
    /* Format: {name, dtype, address, type, context} */
    {"abs", 0, fabs, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"acos", 0, acos, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"acosh", 0, acosh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"all", 0, all_reduce, ME_FUNCTION1, 0},
    {"any", 0, any_reduce, ME_FUNCTION1, 0},
    {"arccos", 0, acos, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arccosh", 0, acosh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arcsin", 0, asin, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arcsinh", 0, asinh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arctan", 0, atan, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arctan2", 0, atan2, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"arctanh", 0, atanh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"asin", 0, asin, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"asinh", 0, asinh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"atan", 0, atan, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"atan2", 0, atan2, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"atanh", 0, atanh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"ceil", 0, ceil, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"conj", 0, conj_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"cos", 0, cos, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"cosh", 0, cosh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"e", 0, e, ME_FUNCTION0 | ME_FLAG_PURE, 0},
    {"exp", 0, exp, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"expm1", 0, expm1_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"fac", 0, fac, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"floor", 0, floor, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"imag", 0, imag_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"ln", 0, log, ME_FUNCTION1 | ME_FLAG_PURE, 0},
#ifdef ME_NAT_LOG
    {"log", 0, log, ME_FUNCTION1 | ME_FLAG_PURE, 0},
#else
    {"log", 0, log10, ME_FUNCTION1 | ME_FLAG_PURE, 0},
#endif
    {"log10", 0, log10, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"log1p", 0, log1p_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"log2", 0, log2_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"logaddexp", 0, logaddexp, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"max", 0, max_reduce, ME_FUNCTION1, 0},
    {"min", 0, min_reduce, ME_FUNCTION1, 0},
    {"ncr", 0, ncr, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"npr", 0, npr, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"pi", 0, pi, ME_FUNCTION0 | ME_FLAG_PURE, 0},
    {"pow", 0, pow, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"prod", 0, prod_reduce, ME_FUNCTION1, 0},
    {"real", 0, real_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"round", 0, round_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sign", 0, sign, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sin", 0, sin, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sinh", 0, sinh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sqrt", 0, sqrt, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"square", 0, square, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sum", 0, sum_reduce, ME_FUNCTION1, 0},
    {"tan", 0, tan, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"tanh", 0, tanh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"trunc", 0, trunc_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"where", 0, where_scalar, ME_FUNCTION3 | ME_FLAG_PURE, 0},
    {0, 0, 0, 0, 0}
};

static const me_variable* find_builtin(const char* name, int len) {
    int imin = 0;
    int imax = sizeof(functions) / sizeof(me_variable) - 2;

    /*Binary search.*/
    while (imax >= imin) {
        const int i = (imin + ((imax - imin) / 2));
        int c = strncmp(name, functions[i].name, len);
        if (!c) c = '\0' - functions[i].name[len];
        if (c == 0) {
            return functions + i;
        }
        else if (c > 0) {
            imin = i + 1;
        }
        else {
            imax = i - 1;
        }
    }

    return 0;
}

static const me_variable* find_lookup(const state* s, const char* name, int len) {
    int iters;
    const me_variable* var;
    if (!s->lookup) return 0;

    for (var = s->lookup, iters = s->lookup_len; iters; ++var, --iters) {
        if (strncmp(name, var->name, len) == 0 && var->name[len] == '\0') {
            return var;
        }
    }
    return 0;
}


static double add(double a, double b) { return a + b; }
static double sub(double a, double b) { return a - b; }
static double mul(double a, double b) { return a * b; }
static double divide(double a, double b) { return a / b; }
static double negate(double a) { return -a; }
static volatile double sum_salt = 0.0;
static volatile double prod_salt = 1.0;
static volatile double min_salt = 0.0;
static volatile double max_salt = 0.0;
static volatile double any_salt = 0.0;
static volatile double all_salt = 0.0;
static double sum_reduce(double x) { return x + sum_salt; }
static double prod_reduce(double x) { return x * prod_salt; }
static double any_reduce(double x) { return x + any_salt; }
static double all_reduce(double x) { return x * (1.0 + all_salt); }
static double min_reduce(double x) { return x + min_salt; }
static double max_reduce(double x) { return x - max_salt; }

static float reduce_min_float32_nan_safe(const float* data, int nitems) {
    if (nitems <= 0) return INFINITY;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256 vmin = _mm256_set1_ps(INFINITY);
    __m256 vnan = _mm256_setzero_ps();
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        vnan = _mm256_or_ps(vnan, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        vmin = _mm256_min_ps(vmin, v);
    }
    __m128 low = _mm256_castps256_ps128(vmin);
    __m128 high = _mm256_extractf128_ps(vmin, 1);
    __m128 min128 = _mm_min_ps(low, high);
    __m128 tmp = _mm_min_ps(min128, _mm_movehl_ps(min128, min128));
    tmp = _mm_min_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1));
    float acc = _mm_cvtss_f32(tmp);
    if (_mm256_movemask_ps(vnan)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#elif defined(__SSE__)
    int i = 0;
    __m128 vmin = _mm_set1_ps(INFINITY);
    __m128 vnan = _mm_setzero_ps();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m128 v = _mm_loadu_ps(data + i);
        vnan = _mm_or_ps(vnan, _mm_cmpunord_ps(v, v));
        vmin = _mm_min_ps(vmin, v);
    }
    __m128 tmp = _mm_min_ps(vmin, _mm_movehl_ps(vmin, vmin));
    tmp = _mm_min_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1));
    float acc = _mm_cvtss_f32(tmp);
    if (_mm_movemask_ps(vnan)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    float32x4_t vmin = vdupq_n_f32(INFINITY);
    uint32x4_t vnan = vdupq_n_u32(0);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        uint32x4_t eq = vceqq_f32(v, v);
        vnan = vorrq_u32(vnan, vmvnq_u32(eq));
        vmin = vminq_f32(vmin, v);
    }
#if defined(__aarch64__)
    float acc = vminvq_f32(vmin);
#else
    float32x2_t min2 = vmin_f32(vget_low_f32(vmin), vget_high_f32(vmin));
    min2 = vpmin_f32(min2, min2);
    float acc = vget_lane_f32(min2, 0);
#endif
    uint32x2_t nan2 = vorr_u32(vget_low_u32(vnan), vget_high_u32(vnan));
    nan2 = vpadd_u32(nan2, nan2);
    if (vget_lane_u32(nan2, 0)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#else
    float acc = data[0];
    for (int i = 0; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#endif
}

static float reduce_max_float32_nan_safe(const float* data, int nitems) {
    if (nitems <= 0) return -INFINITY;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    __m256 vnan = _mm256_setzero_ps();
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        vnan = _mm256_or_ps(vnan, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        vmax = _mm256_max_ps(vmax, v);
    }
    __m128 low = _mm256_castps256_ps128(vmax);
    __m128 high = _mm256_extractf128_ps(vmax, 1);
    __m128 max128 = _mm_max_ps(low, high);
    __m128 tmp = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    tmp = _mm_max_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1));
    float acc = _mm_cvtss_f32(tmp);
    if (_mm256_movemask_ps(vnan)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#elif defined(__SSE__)
    int i = 0;
    __m128 vmax = _mm_set1_ps(-INFINITY);
    __m128 vnan = _mm_setzero_ps();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m128 v = _mm_loadu_ps(data + i);
        vnan = _mm_or_ps(vnan, _mm_cmpunord_ps(v, v));
        vmax = _mm_max_ps(vmax, v);
    }
    __m128 tmp = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
    tmp = _mm_max_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1));
    float acc = _mm_cvtss_f32(tmp);
    if (_mm_movemask_ps(vnan)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    float32x4_t vmax = vdupq_n_f32(-INFINITY);
    uint32x4_t vnan = vdupq_n_u32(0);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        uint32x4_t eq = vceqq_f32(v, v);
        vnan = vorrq_u32(vnan, vmvnq_u32(eq));
        vmax = vmaxq_f32(vmax, v);
    }
#if defined(__aarch64__)
    float acc = vmaxvq_f32(vmax);
#else
    float32x2_t max2 = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    max2 = vpmax_f32(max2, max2);
    float acc = vget_lane_f32(max2, 0);
#endif
    uint32x2_t nan2 = vorr_u32(vget_low_u32(vnan), vget_high_u32(vnan));
    nan2 = vpadd_u32(nan2, nan2);
    if (vget_lane_u32(nan2, 0)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#else
    float acc = data[0];
    for (int i = 0; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#endif
}

static double reduce_min_float64_nan_safe(const double* data, int nitems) {
    if (nitems <= 0) return INFINITY;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256d vmin = _mm256_set1_pd(INFINITY);
    __m256d vnan = _mm256_setzero_pd();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        vnan = _mm256_or_pd(vnan, _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
        vmin = _mm256_min_pd(vmin, v);
    }
    __m128d low = _mm256_castpd256_pd128(vmin);
    __m128d high = _mm256_extractf128_pd(vmin, 1);
    __m128d min128 = _mm_min_pd(low, high);
    min128 = _mm_min_sd(min128, _mm_unpackhi_pd(min128, min128));
    double acc = _mm_cvtsd_f64(min128);
    if (_mm256_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#elif defined(__SSE2__)
    int i = 0;
    __m128d vmin = _mm_set1_pd(INFINITY);
    __m128d vnan = _mm_setzero_pd();
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        __m128d v = _mm_loadu_pd(data + i);
        vnan = _mm_or_pd(vnan, _mm_cmpunord_pd(v, v));
        vmin = _mm_min_pd(vmin, v);
    }
    vmin = _mm_min_sd(vmin, _mm_unpackhi_pd(vmin, vmin));
    double acc = _mm_cvtsd_f64(vmin);
    if (_mm_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    float64x2_t vmin = vdupq_n_f64(INFINITY);
    uint64x2_t vnan = vdupq_n_u64(0);
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        uint64x2_t eq = vceqq_f64(v, v);
        vnan = vorrq_u64(vnan, veorq_u64(eq, vdupq_n_u64(~0ULL)));
        vmin = vminq_f64(vmin, v);
    }
    double acc = vminvq_f64(vmin);
    uint64x2_t nan_or = vorrq_u64(vnan, vextq_u64(vnan, vnan, 1));
    if (vgetq_lane_u64(nan_or, 0)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#else
    double acc = data[0];
    for (int i = 0; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#endif
}

static double reduce_max_float64_nan_safe(const double* data, int nitems) {
    if (nitems <= 0) return -INFINITY;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256d vmax = _mm256_set1_pd(-INFINITY);
    __m256d vnan = _mm256_setzero_pd();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        vnan = _mm256_or_pd(vnan, _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
        vmax = _mm256_max_pd(vmax, v);
    }
    __m128d low = _mm256_castpd256_pd128(vmax);
    __m128d high = _mm256_extractf128_pd(vmax, 1);
    __m128d max128 = _mm_max_pd(low, high);
    max128 = _mm_max_sd(max128, _mm_unpackhi_pd(max128, max128));
    double acc = _mm_cvtsd_f64(max128);
    if (_mm256_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#elif defined(__SSE2__)
    int i = 0;
    __m128d vmax = _mm_set1_pd(-INFINITY);
    __m128d vnan = _mm_setzero_pd();
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        __m128d v = _mm_loadu_pd(data + i);
        vnan = _mm_or_pd(vnan, _mm_cmpunord_pd(v, v));
        vmax = _mm_max_pd(vmax, v);
    }
    vmax = _mm_max_sd(vmax, _mm_unpackhi_pd(vmax, vmax));
    double acc = _mm_cvtsd_f64(vmax);
    if (_mm_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    float64x2_t vmax = vdupq_n_f64(-INFINITY);
    uint64x2_t vnan = vdupq_n_u64(0);
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        uint64x2_t eq = vceqq_f64(v, v);
        vnan = vorrq_u64(vnan, veorq_u64(eq, vdupq_n_u64(~0ULL)));
        vmax = vmaxq_f64(vmax, v);
    }
    double acc = vmaxvq_f64(vmax);
    uint64x2_t nan_or = vorrq_u64(vnan, vextq_u64(vnan, vnan, 1));
    if (vgetq_lane_u64(nan_or, 0)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#else
    double acc = data[0];
    for (int i = 0; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#endif
}

static int32_t reduce_min_int32(const int32_t* data, int nitems) {
    if (nitems <= 0) return INT32_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi32(INT32_MAX);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epi32(vmin, v);
    }
    int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    int32_t acc = tmp[0];
    for (int j = 1; j < 8; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__SSE4_1__)
    int i = 0;
    __m128i vmin = _mm_set1_epi32(INT32_MAX);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i*)(data + i));
        vmin = _mm_min_epi32(vmin, v);
    }
    int32_t tmp[4];
    _mm_storeu_si128((__m128i*)tmp, vmin);
    int32_t acc = tmp[0];
    for (int j = 1; j < 4; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int32x4_t vmin = vdupq_n_s32(INT32_MAX);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        int32x4_t v = vld1q_s32(data + i);
        vmin = vminq_s32(vmin, v);
    }
#if defined(__aarch64__)
    int32_t acc = vminvq_s32(vmin);
#else
    int32x2_t min2 = vmin_s32(vget_low_s32(vmin), vget_high_s32(vmin));
    min2 = vpmin_s32(min2, min2);
    int32_t acc = vget_lane_s32(min2, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    int32_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static int32_t reduce_max_int32(const int32_t* data, int nitems) {
    if (nitems <= 0) return INT32_MIN;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_set1_epi32(INT32_MIN);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epi32(vmax, v);
    }
    int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    int32_t acc = tmp[0];
    for (int j = 1; j < 8; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__SSE4_1__)
    int i = 0;
    __m128i vmax = _mm_set1_epi32(INT32_MIN);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i*)(data + i));
        vmax = _mm_max_epi32(vmax, v);
    }
    int32_t tmp[4];
    _mm_storeu_si128((__m128i*)tmp, vmax);
    int32_t acc = tmp[0];
    for (int j = 1; j < 4; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int32x4_t vmax = vdupq_n_s32(INT32_MIN);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        int32x4_t v = vld1q_s32(data + i);
        vmax = vmaxq_s32(vmax, v);
    }
#if defined(__aarch64__)
    int32_t acc = vmaxvq_s32(vmax);
#else
    int32x2_t max2 = vmax_s32(vget_low_s32(vmax), vget_high_s32(vmax));
    max2 = vpmax_s32(max2, max2);
    int32_t acc = vget_lane_s32(max2, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    int32_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static int8_t reduce_min_int8(const int8_t* data, int nitems) {
    if (nitems <= 0) return INT8_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi8(INT8_MAX);
    const int limit = nitems & ~31;
    for (; i < limit; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epi8(vmin, v);
    }
    int8_t tmp[32];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    int8_t acc = tmp[0];
    for (int j = 1; j < 32; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int8x16_t vmin = vdupq_n_s8(INT8_MAX);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        int8x16_t v = vld1q_s8(data + i);
        vmin = vminq_s8(vmin, v);
    }
#if defined(__aarch64__)
    int8_t acc = vminvq_s8(vmin);
#else
    int8x8_t min8 = vmin_s8(vget_low_s8(vmin), vget_high_s8(vmin));
    min8 = vpmin_s8(min8, min8);
    min8 = vpmin_s8(min8, min8);
    int8_t acc = vget_lane_s8(min8, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    int8_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static int8_t reduce_max_int8(const int8_t* data, int nitems) {
    if (nitems <= 0) return INT8_MIN;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_set1_epi8(INT8_MIN);
    const int limit = nitems & ~31;
    for (; i < limit; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epi8(vmax, v);
    }
    int8_t tmp[32];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    int8_t acc = tmp[0];
    for (int j = 1; j < 32; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int8x16_t vmax = vdupq_n_s8(INT8_MIN);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        int8x16_t v = vld1q_s8(data + i);
        vmax = vmaxq_s8(vmax, v);
    }
#if defined(__aarch64__)
    int8_t acc = vmaxvq_s8(vmax);
#else
    int8x8_t max8 = vmax_s8(vget_low_s8(vmax), vget_high_s8(vmax));
    max8 = vpmax_s8(max8, max8);
    max8 = vpmax_s8(max8, max8);
    int8_t acc = vget_lane_s8(max8, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    int8_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static int16_t reduce_min_int16(const int16_t* data, int nitems) {
    if (nitems <= 0) return INT16_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi16(INT16_MAX);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epi16(vmin, v);
    }
    int16_t tmp[16];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    int16_t acc = tmp[0];
    for (int j = 1; j < 16; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int16x8_t vmin = vdupq_n_s16(INT16_MAX);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        int16x8_t v = vld1q_s16(data + i);
        vmin = vminq_s16(vmin, v);
    }
#if defined(__aarch64__)
    int16_t acc = vminvq_s16(vmin);
#else
    int16x4_t min4 = vmin_s16(vget_low_s16(vmin), vget_high_s16(vmin));
    min4 = vpmin_s16(min4, min4);
    min4 = vpmin_s16(min4, min4);
    int16_t acc = vget_lane_s16(min4, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    int16_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static int16_t reduce_max_int16(const int16_t* data, int nitems) {
    if (nitems <= 0) return INT16_MIN;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_set1_epi16(INT16_MIN);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epi16(vmax, v);
    }
    int16_t tmp[16];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    int16_t acc = tmp[0];
    for (int j = 1; j < 16; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int16x8_t vmax = vdupq_n_s16(INT16_MIN);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        int16x8_t v = vld1q_s16(data + i);
        vmax = vmaxq_s16(vmax, v);
    }
#if defined(__aarch64__)
    int16_t acc = vmaxvq_s16(vmax);
#else
    int16x4_t max4 = vmax_s16(vget_low_s16(vmax), vget_high_s16(vmax));
    max4 = vpmax_s16(max4, max4);
    max4 = vpmax_s16(max4, max4);
    int16_t acc = vget_lane_s16(max4, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    int16_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static int64_t reduce_min_int64(const int64_t* data, int nitems) {
    if (nitems <= 0) return INT64_MAX;
    int64_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
}

static int64_t reduce_max_int64(const int64_t* data, int nitems) {
    if (nitems <= 0) return INT64_MIN;
    int64_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
}

static uint8_t reduce_min_uint8(const uint8_t* data, int nitems) {
    if (nitems <= 0) return UINT8_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi8((char)UINT8_MAX);
    const int limit = nitems & ~31;
    for (; i < limit; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epu8(vmin, v);
    }
    uint8_t tmp[32];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    uint8_t acc = tmp[0];
    for (int j = 1; j < 32; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint8x16_t vmin = vdupq_n_u8(UINT8_MAX);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        uint8x16_t v = vld1q_u8(data + i);
        vmin = vminq_u8(vmin, v);
    }
#if defined(__aarch64__)
    uint8_t acc = vminvq_u8(vmin);
#else
    uint8x8_t min8 = vmin_u8(vget_low_u8(vmin), vget_high_u8(vmin));
    min8 = vpmin_u8(min8, min8);
    min8 = vpmin_u8(min8, min8);
    uint8_t acc = vget_lane_u8(min8, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    uint8_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static uint8_t reduce_max_uint8(const uint8_t* data, int nitems) {
    if (nitems <= 0) return 0;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_setzero_si256();
    const int limit = nitems & ~31;
    for (; i < limit; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epu8(vmax, v);
    }
    uint8_t tmp[32];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    uint8_t acc = tmp[0];
    for (int j = 1; j < 32; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint8x16_t vmax = vdupq_n_u8(0);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        uint8x16_t v = vld1q_u8(data + i);
        vmax = vmaxq_u8(vmax, v);
    }
#if defined(__aarch64__)
    uint8_t acc = vmaxvq_u8(vmax);
#else
    uint8x8_t max8 = vmax_u8(vget_low_u8(vmax), vget_high_u8(vmax));
    max8 = vpmax_u8(max8, max8);
    max8 = vpmax_u8(max8, max8);
    uint8_t acc = vget_lane_u8(max8, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    uint8_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static uint16_t reduce_min_uint16(const uint16_t* data, int nitems) {
    if (nitems <= 0) return UINT16_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi16((short)UINT16_MAX);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epu16(vmin, v);
    }
    uint16_t tmp[16];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    uint16_t acc = tmp[0];
    for (int j = 1; j < 16; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint16x8_t vmin = vdupq_n_u16(UINT16_MAX);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        uint16x8_t v = vld1q_u16(data + i);
        vmin = vminq_u16(vmin, v);
    }
#if defined(__aarch64__)
    uint16_t acc = vminvq_u16(vmin);
#else
    uint16x4_t min4 = vmin_u16(vget_low_u16(vmin), vget_high_u16(vmin));
    min4 = vpmin_u16(min4, min4);
    min4 = vpmin_u16(min4, min4);
    uint16_t acc = vget_lane_u16(min4, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    uint16_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static uint16_t reduce_max_uint16(const uint16_t* data, int nitems) {
    if (nitems <= 0) return 0;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_setzero_si256();
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epu16(vmax, v);
    }
    uint16_t tmp[16];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    uint16_t acc = tmp[0];
    for (int j = 1; j < 16; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint16x8_t vmax = vdupq_n_u16(0);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        uint16x8_t v = vld1q_u16(data + i);
        vmax = vmaxq_u16(vmax, v);
    }
#if defined(__aarch64__)
    uint16_t acc = vmaxvq_u16(vmax);
#else
    uint16x4_t max4 = vmax_u16(vget_low_u16(vmax), vget_high_u16(vmax));
    max4 = vpmax_u16(max4, max4);
    max4 = vpmax_u16(max4, max4);
    uint16_t acc = vget_lane_u16(max4, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    uint16_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static uint32_t reduce_min_uint32(const uint32_t* data, int nitems) {
    if (nitems <= 0) return UINT32_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi32((int)UINT32_MAX);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epu32(vmin, v);
    }
    uint32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    uint32_t acc = tmp[0];
    for (int j = 1; j < 8; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint32x4_t vmin = vdupq_n_u32(UINT32_MAX);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        uint32x4_t v = vld1q_u32(data + i);
        vmin = vminq_u32(vmin, v);
    }
#if defined(__aarch64__)
    uint32_t acc = vminvq_u32(vmin);
#else
    uint32x2_t min2 = vmin_u32(vget_low_u32(vmin), vget_high_u32(vmin));
    min2 = vpmin_u32(min2, min2);
    uint32_t acc = vget_lane_u32(min2, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    uint32_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static uint32_t reduce_max_uint32(const uint32_t* data, int nitems) {
    if (nitems <= 0) return 0;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_setzero_si256();
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epu32(vmax, v);
    }
    uint32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    uint32_t acc = tmp[0];
    for (int j = 1; j < 8; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint32x4_t vmax = vdupq_n_u32(0);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        uint32x4_t v = vld1q_u32(data + i);
        vmax = vmaxq_u32(vmax, v);
    }
#if defined(__aarch64__)
    uint32_t acc = vmaxvq_u32(vmax);
#else
    uint32x2_t max2 = vmax_u32(vget_low_u32(vmax), vget_high_u32(vmax));
    max2 = vpmax_u32(max2, max2);
    uint32_t acc = vget_lane_u32(max2, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    uint32_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static uint64_t reduce_min_uint64(const uint64_t* data, int nitems) {
    if (nitems <= 0) return UINT64_MAX;
    uint64_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
}

static uint64_t reduce_max_uint64(const uint64_t* data, int nitems) {
    if (nitems <= 0) return 0;
    uint64_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
}

static double reduce_prod_float32_nan_safe(const float* data, int nitems) {
    if (nitems <= 0) return 1.0;
    double acc = 1.0;
    for (int i = 0; i < nitems; i++) {
        double v = (double)data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
}

static double reduce_prod_float64_nan_safe(const double* data, int nitems) {
    if (nitems <= 0) return 1.0;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256d vprod = _mm256_set1_pd(1.0);
    __m256d vnan = _mm256_setzero_pd();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        vnan = _mm256_or_pd(vnan, _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
        vprod = _mm256_mul_pd(vprod, v);
    }
    __m128d low = _mm256_castpd256_pd128(vprod);
    __m128d high = _mm256_extractf128_pd(vprod, 1);
    __m128d prod128 = _mm_mul_pd(low, high);
    prod128 = _mm_mul_sd(prod128, _mm_unpackhi_pd(prod128, prod128));
    double acc = _mm_cvtsd_f64(prod128);
    if (_mm256_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#elif defined(__SSE2__)
    int i = 0;
    __m128d vprod = _mm_set1_pd(1.0);
    __m128d vnan = _mm_setzero_pd();
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        __m128d v = _mm_loadu_pd(data + i);
        vnan = _mm_or_pd(vnan, _mm_cmpunord_pd(v, v));
        vprod = _mm_mul_pd(vprod, v);
    }
    vprod = _mm_mul_sd(vprod, _mm_unpackhi_pd(vprod, vprod));
    double acc = _mm_cvtsd_f64(vprod);
    if (_mm_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    float64x2_t vprod = vdupq_n_f64(1.0);
    uint64x2_t vnan = vdupq_n_u64(0);
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        uint64x2_t eq = vceqq_f64(v, v);
        vnan = vorrq_u64(vnan, veorq_u64(eq, vdupq_n_u64(~0ULL)));
        vprod = vmulq_f64(vprod, v);
    }
    double acc = vgetq_lane_f64(vprod, 0) * vgetq_lane_f64(vprod, 1);
    uint64x2_t nan_or = vorrq_u64(vnan, vextq_u64(vnan, vnan, 1));
    if (vgetq_lane_u64(nan_or, 0)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#else
    double acc = 1.0;
    for (int i = 0; i < nitems; i++) {
        double v = data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#endif
}

static double reduce_sum_float32_nan_safe(const float* data, int nitems) {
    if (nitems <= 0) return 0.0;
    double acc = 0.0;
    for (int i = 0; i < nitems; i++) {
        double v = (double)data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
}

static double reduce_sum_float64_nan_safe(const double* data, int nitems) {
    if (nitems <= 0) return 0.0;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256d vsum = _mm256_setzero_pd();
    __m256d vnan = _mm256_setzero_pd();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        vnan = _mm256_or_pd(vnan, _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
        vsum = _mm256_add_pd(vsum, v);
    }
    __m128d low = _mm256_castpd256_pd128(vsum);
    __m128d high = _mm256_extractf128_pd(vsum, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    sum128 = _mm_add_sd(sum128, _mm_unpackhi_pd(sum128, sum128));
    double acc = _mm_cvtsd_f64(sum128);
    if (_mm256_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#elif defined(__SSE2__)
    int i = 0;
    __m128d vsum = _mm_setzero_pd();
    __m128d vnan = _mm_setzero_pd();
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        __m128d v = _mm_loadu_pd(data + i);
        vnan = _mm_or_pd(vnan, _mm_cmpunord_pd(v, v));
        vsum = _mm_add_pd(vsum, v);
    }
    vsum = _mm_add_sd(vsum, _mm_unpackhi_pd(vsum, vsum));
    double acc = _mm_cvtsd_f64(vsum);
    if (_mm_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    float64x2_t vsum = vdupq_n_f64(0.0);
    uint64x2_t vnan = vdupq_n_u64(0);
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        uint64x2_t eq = vceqq_f64(v, v);
        vnan = vorrq_u64(vnan, veorq_u64(eq, vdupq_n_u64(~0ULL)));
        vsum = vaddq_f64(vsum, v);
    }
    double acc = vaddvq_f64(vsum);
    uint64x2_t nan_or = vorrq_u64(vnan, vextq_u64(vnan, vnan, 1));
    if (vgetq_lane_u64(nan_or, 0)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#else
    double acc = 0.0;
    for (int i = 0; i < nitems; i++) {
        double v = data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#endif
}

static double comma(double a, double b) {
    (void)a;
    return b;
}

/* Bitwise operators (for integer types) */
static double bit_and(double a, double b) { return (double)((int64_t)a & (int64_t)b); }
static double bit_or(double a, double b) { return (double)((int64_t)a | (int64_t)b); }
static double bit_xor(double a, double b) { return (double)((int64_t)a ^ (int64_t)b); }
static double bit_not(double a) { return (double)(~(int64_t)a); }
static double bit_shl(double a, double b) { return (double)((int64_t)a << (int64_t)b); }
static double bit_shr(double a, double b) { return (double)((int64_t)a >> (int64_t)b); }

/* Comparison operators (return 1.0 for true, 0.0 for false) */
static double cmp_eq(double a, double b) { return a == b ? 1.0 : 0.0; }
static double cmp_ne(double a, double b) { return a != b ? 1.0 : 0.0; }
static double cmp_lt(double a, double b) { return a < b ? 1.0 : 0.0; }
static double cmp_le(double a, double b) { return a <= b ? 1.0 : 0.0; }
static double cmp_gt(double a, double b) { return a > b ? 1.0 : 0.0; }
static double cmp_ge(double a, double b) { return a >= b ? 1.0 : 0.0; }

/* Logical operators (for bool type) - short-circuit via OR/AND */
static double logical_and(double a, double b) { return ((int)a) && ((int)b) ? 1.0 : 0.0; }
static double logical_or(double a, double b) { return ((int)a) || ((int)b) ? 1.0 : 0.0; }
static double logical_not(double a) { return !(int)a ? 1.0 : 0.0; }
static double logical_xor(double a, double b) { return ((int)a) != ((int)b) ? 1.0 : 0.0; }

static bool is_identifier_start(char c) {
    return isalpha((unsigned char)c) || c == '_';
}

static bool is_identifier_char(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

static void skip_whitespace(state* s) {
    while (*s->next && isspace((unsigned char)*s->next)) {
        s->next++;
    }
}

static void read_number_token(state* s) {
    const char* start = s->next;
    s->value = strtod(s->next, (char**)&s->next);
    s->type = TOK_NUMBER;

    // Determine if it is a floating point or integer constant
    bool is_float = false;
    for (const char* p = start; p < s->next; p++) {
        if (*p == '.' || *p == 'e' || *p == 'E') {
            is_float = true;
            break;
        }
    }

    if (is_float) {
        // Match NumPy conventions: float constants match target_dtype when it's a float type
        // This ensures FLOAT32 arrays + float constants -> FLOAT32 (NumPy behavior)
        if (s->target_dtype == ME_FLOAT32) {
            s->dtype = ME_FLOAT32;
        }
        else {
            s->dtype = ME_FLOAT64;
        }
    }
    else {
        // For integers, we use a heuristic
        if (s->value > INT_MAX || s->value < INT_MIN) {
            s->dtype = ME_INT64;
        }
        else {
            // Use target_dtype if it's an integer type, otherwise default to INT32
            if (is_integer_dtype(s->target_dtype)) {
                s->dtype = s->target_dtype;
            }
            else {
                s->dtype = ME_INT32;
            }
        }
    }
}

static void read_identifier_token(state* s) {
    const char* start = s->next;
    while (is_identifier_char(*s->next)) {
        s->next++;
    }

    const me_variable* var = find_lookup(s, start, s->next - start);
    if (!var) {
        var = find_builtin(start, s->next - start);
    }

    if (!var) {
        s->type = TOK_ERROR;
        return;
    }

    switch (TYPE_MASK(var->type)) {
    case ME_VARIABLE:
        s->type = TOK_VARIABLE;
        s->bound = var->address;
        s->dtype = var->dtype;
        break;

    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        s->context = var->context;
    /* Falls through. */
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
        s->type = var->type;
        s->function = var->address;
        break;
    }
}

typedef struct {
    const char* literal;
    int token_type;
    me_fun2 function;
} operator_spec;

static bool handle_multi_char_operator(state* s) {
    static const operator_spec multi_ops[] = {
        {"**", TOK_POW, pow},
        {"<<", TOK_SHIFT, bit_shl},
        {">>", TOK_SHIFT, bit_shr},
        {"==", TOK_COMPARE, cmp_eq},
        {"!=", TOK_COMPARE, cmp_ne},
        {"<=", TOK_COMPARE, cmp_le},
        {">=", TOK_COMPARE, cmp_ge},
    };

    for (size_t i = 0; i < sizeof(multi_ops) / sizeof(multi_ops[0]); i++) {
        const operator_spec* op = &multi_ops[i];
        size_t len = strlen(op->literal);
        if (strncmp(s->next, op->literal, len) == 0) {
            s->type = op->token_type;
            s->function = op->function;
            s->next += len;
            return true;
        }
    }
    return false;
}

static void handle_single_char_operator(state* s, char c) {
    s->next++;
    switch (c) {
    case '+': s->type = TOK_INFIX;
        s->function = add;
        break;
    case '-': s->type = TOK_INFIX;
        s->function = sub;
        break;
    case '*': s->type = TOK_INFIX;
        s->function = mul;
        break;
    case '/': s->type = TOK_INFIX;
        s->function = divide;
        break;
    case '%': s->type = TOK_INFIX;
        s->function = fmod;
        break;
    case '&': s->type = TOK_BITWISE;
        s->function = bit_and;
        break;
    case '|': s->type = TOK_BITWISE;
        s->function = bit_or;
        break;
    case '^': s->type = TOK_BITWISE;
        s->function = bit_xor;
        break;
    case '~': s->type = TOK_BITWISE;
        s->function = bit_not;
        break;
    case '<': s->type = TOK_COMPARE;
        s->function = cmp_lt;
        break;
    case '>': s->type = TOK_COMPARE;
        s->function = cmp_gt;
        break;
    case '(': s->type = TOK_OPEN;
        break;
    case ')': s->type = TOK_CLOSE;
        break;
    case ',': s->type = TOK_SEP;
        break;
    default: s->type = TOK_ERROR;
        break;
    }
}

static void read_operator_token(state* s) {
    if (handle_multi_char_operator(s)) {
        return;
    }

    if (!*s->next) {
        s->type = TOK_END;
        return;
    }

    handle_single_char_operator(s, *s->next);
}

void next_token(state* s) {
    s->type = TOK_NULL;

    do {
        skip_whitespace(s);

        if (!*s->next) {
            s->type = TOK_END;
            return;
        }

        if ((s->next[0] >= '0' && s->next[0] <= '9') || s->next[0] == '.') {
            read_number_token(s);
        }
        else if (is_identifier_start(s->next[0])) {
            read_identifier_token(s);
        }
        else {
            read_operator_token(s);
        }
    }
    while (s->type == TOK_NULL);
}


static me_expr* list(state* s);

static me_expr* expr(state* s);

static me_expr* power(state* s);

static me_expr* shift_expr(state* s);

static me_expr* bitwise_and(state* s);

static me_expr* bitwise_xor(state* s);

static me_expr* bitwise_or(state* s);

static me_expr* comparison(state* s);


static me_expr* base(state* s) {
    /* <base>      =    <constant> | <variable> | <function-0> {"(" ")"} | <function-1> <power> | <function-X> "(" <expr> {"," <expr>} ")" | "(" <list> ")" */
    me_expr* ret;
    int arity;

    switch (TYPE_MASK(s->type)) {
    case TOK_NUMBER:
        ret = new_expr(ME_CONSTANT, 0);
        CHECK_NULL(ret);

        ret->value = s->value;
        // Use inferred type for constants (floating point vs integer)
        if (s->target_dtype == ME_AUTO) {
            ret->dtype = s->dtype;
        }
        else {
            // If target_dtype is integer but constant is float/complex, we must use float/complex
            if (is_integer_dtype(s->target_dtype)) {
                if (is_float_dtype(s->dtype) || is_complex_dtype(s->dtype)) {
                    ret->dtype = s->dtype;
                }
                else if (is_integer_dtype(s->dtype) && dtype_size(s->dtype) > dtype_size(s->target_dtype)) {
                    // Use larger integer type if needed
                    ret->dtype = s->dtype;
                }
                else {
                    ret->dtype = s->target_dtype;
                }
            }
            else {
                // For float/complex target types, use target_dtype to match NumPy conventions
                // Float constants are typed based on target_dtype (FLOAT32 or FLOAT64)
                // This ensures FLOAT32 arrays + float constants -> FLOAT32 (NumPy behavior)
                ret->dtype = s->target_dtype;
            }
        }
        next_token(s);
        break;

    case TOK_VARIABLE:
        ret = new_expr(ME_VARIABLE, 0);
        CHECK_NULL(ret);

        ret->bound = s->bound;
        ret->dtype = s->dtype; // Set the variable's type
        ret->input_dtype = s->dtype;
        next_token(s);
        break;

    case ME_FUNCTION0:
    case ME_CLOSURE0:
        ret = new_expr(s->type, 0);
        CHECK_NULL(ret);

        ret->function = s->function;
        if (IS_CLOSURE(s->type)) ret->parameters[0] = s->context;
        next_token(s);
        if (s->type == TOK_OPEN) {
            next_token(s);
            if (s->type != TOK_CLOSE) {
                s->type = TOK_ERROR;
            }
            else {
                next_token(s);
            }
        }
        break;

    case ME_FUNCTION1:
    case ME_CLOSURE1:
        ret = new_expr(s->type, 0);
        CHECK_NULL(ret);

        ret->function = s->function;
        if (IS_CLOSURE(s->type)) ret->parameters[1] = s->context;
        next_token(s);
        ret->parameters[0] = power(s);
        CHECK_NULL(ret->parameters[0], me_free(ret));
        break;

    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        arity = ARITY(s->type);

        ret = new_expr(s->type, 0);
        CHECK_NULL(ret);

        ret->function = s->function;
        if (IS_CLOSURE(s->type)) ret->parameters[arity] = s->context;
        next_token(s);

        if (s->type != TOK_OPEN) {
            s->type = TOK_ERROR;
        }
        else {
            int i;
            for (i = 0; i < arity; i++) {
                next_token(s);
                ret->parameters[i] = expr(s);
                CHECK_NULL(ret->parameters[i], me_free(ret));

                if (s->type != TOK_SEP) {
                    break;
                }
            }
            if (s->type != TOK_CLOSE || i != arity - 1) {
                s->type = TOK_ERROR;
            }
            else {
                next_token(s);
            }
        }

        break;

    case TOK_OPEN:
        next_token(s);
        ret = list(s);
        CHECK_NULL(ret);

        if (s->type != TOK_CLOSE) {
            s->type = TOK_ERROR;
        }
        else {
            next_token(s);
        }
        break;

    default:
        ret = new_expr(0, 0);
        CHECK_NULL(ret);

        s->type = TOK_ERROR;
        ret->value = NAN;
        break;
    }

    return ret;
}


static me_expr* power(state* s) {
    /* <power>     =    {("-" | "+")} <base> */
    int sign = 1;
    while (s->type == TOK_INFIX && (s->function == add || s->function == sub)) {
        if (s->function == sub) sign = -sign;
        next_token(s);
    }

    me_expr* ret;

    if (sign == 1) {
        ret = base(s);
    }
    else {
        me_expr* b = base(s);
        CHECK_NULL(b);

        ret = NEW_EXPR(ME_FUNCTION1 | ME_FLAG_PURE, b);
        CHECK_NULL(ret, me_free(b));

        ret->function = negate;
    }

    return ret;
}

#ifdef ME_POW_FROM_RIGHT
static me_expr* factor(state* s) {
    /* <factor>    =    <power> {"**" <factor>}  (right associative) */
    me_expr* ret = power(s);
    CHECK_NULL(ret);

    if (s->type == TOK_POW) {
        me_fun2 t = s->function;
        next_token(s);
        me_expr* f = factor(s); /* Right associative: recurse */
        CHECK_NULL(f, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, f);
        CHECK_NULL(ret, me_free(f), me_free(prev));

        ret->function = t;
        apply_type_promotion(ret);
    }

    return ret;
}
#else
static me_expr* factor(state* s) {
    /* <factor>    =    <power> {"**" <power>}  (left associative) */
    me_expr* ret = power(s);
    CHECK_NULL(ret);

    while (s->type == TOK_POW) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* f = power(s);
        CHECK_NULL(f, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, f);
        CHECK_NULL(ret, me_free(f), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret);
    }

    return ret;
}
#endif


static me_expr* term(state* s) {
    /* <term>      =    <factor> {("*" | "/" | "%") <factor>} */
    me_expr* ret = factor(s);
    CHECK_NULL(ret);

    while (s->type == TOK_INFIX && (s->function == mul || s->function == divide || s->function == fmod)) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* f = factor(s);
        CHECK_NULL(f, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, f);
        CHECK_NULL(ret, me_free(f), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr* expr(state* s) {
    /* <expr>      =    <term> {("+" | "-") <term>} */
    me_expr* ret = term(s);
    CHECK_NULL(ret);

    while (s->type == TOK_INFIX && (s->function == add || s->function == sub)) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* te = term(s);
        CHECK_NULL(te, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, te);
        CHECK_NULL(ret, me_free(te), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret); // Apply type promotion
    }

    return ret;
}


static me_expr* shift_expr(state* s) {
    /* <shift_expr> =    <expr> {("<<" | ">>") <expr>} */
    me_expr* ret = expr(s);
    CHECK_NULL(ret);

    while (s->type == TOK_SHIFT) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* e = expr(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr* bitwise_and(state* s) {
    /* <bitwise_and> =    <shift_expr> {"&" <shift_expr>} */
    me_expr* ret = shift_expr(s);
    CHECK_NULL(ret);

    while (s->type == TOK_BITWISE && s->function == bit_and) {
        next_token(s);
        me_expr* e = shift_expr(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = bit_and;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr* bitwise_xor(state* s) {
    /* <bitwise_xor> =    <bitwise_and> {"^" <bitwise_and>} */
    /* Note: ^ is XOR for integers/bools. Use ** for power */
    me_expr* ret = bitwise_and(s);
    CHECK_NULL(ret);

    while (s->type == TOK_BITWISE && s->function == bit_xor) {
        next_token(s);
        me_expr* e = bitwise_and(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = bit_xor;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr* bitwise_or(state* s) {
    /* <bitwise_or> =    <bitwise_xor> {"|" <bitwise_xor>} */
    me_expr* ret = bitwise_xor(s);
    CHECK_NULL(ret);

    while (s->type == TOK_BITWISE && (s->function == bit_or)) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* e = bitwise_xor(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr* comparison(state* s) {
    /* <comparison> =    <bitwise_or> {("<" | ">" | "<=" | ">=" | "==" | "!=") <bitwise_or>} */
    me_expr* ret = bitwise_or(s);
    CHECK_NULL(ret);

    while (s->type == TOK_COMPARE) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* e = bitwise_or(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret);
        /* Comparisons always return bool */
        ret->dtype = ME_BOOL;
    }

    return ret;
}


static me_expr* list(state* s) {
    /* <list>      =    <comparison> {"," <comparison>} */
    me_expr* ret = comparison(s);
    CHECK_NULL(ret);

    while (s->type == TOK_SEP) {
        next_token(s);
        me_expr* e = comparison(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = comma;
        apply_type_promotion(ret);
    }

    return ret;
}


#define ME_FUN(...) ((double(*)(__VA_ARGS__))n->function)
#define M(e) me_eval_scalar(n->parameters[e])

static double me_eval_scalar(const me_expr* n) {
    if (!n) return NAN;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT: return n->value;
    case ME_VARIABLE: return *(const double*)n->bound;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
        switch (ARITY(n->type)) {
        case 0: return ME_FUN(void)();
        case 1: return ME_FUN(double)(M(0));
        case 2: return ME_FUN(double, double)(M(0), M(1));
        case 3: return ME_FUN(double, double, double)(M(0), M(1), M(2));
        case 4: return ME_FUN(double, double, double, double)(M(0), M(1), M(2), M(3));
        case 5: return ME_FUN(double, double, double, double, double)(M(0), M(1), M(2), M(3), M(4));
        case 6: return ME_FUN(double, double, double, double, double, double)(
                M(0), M(1), M(2), M(3), M(4), M(5));
        case 7: return ME_FUN(double, double, double, double, double, double, double)(
                M(0), M(1), M(2), M(3), M(4), M(5), M(6));
        default: return NAN;
        }

    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        switch (ARITY(n->type)) {
        case 0: return ME_FUN(void*)(n->parameters[0]);
        case 1: return ME_FUN(void*, double)(n->parameters[1], M(0));
        case 2: return ME_FUN(void*, double, double)(n->parameters[2], M(0), M(1));
        case 3: return ME_FUN(void*, double, double, double)(n->parameters[3], M(0), M(1), M(2));
        case 4: return ME_FUN(void*, double, double, double, double)(n->parameters[4], M(0), M(1), M(2), M(3));
        case 5: return ME_FUN(void*, double, double, double, double, double)(
                n->parameters[5], M(0), M(1), M(2), M(3), M(4));
        case 6: return ME_FUN(void*, double, double, double, double, double, double)(
                n->parameters[6], M(0), M(1), M(2), M(3), M(4), M(5));
        case 7: return ME_FUN(void*, double, double, double, double, double, double, double)(
                n->parameters[7], M(0), M(1), M(2), M(3), M(4), M(5), M(6));
        default: return NAN;
        }

    default: return NAN;
    }
}

#undef ME_FUN
#undef M

/* Specialized vector operations for better performance */
static void vec_add(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static void vec_sub(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

static void vec_mul(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

static void vec_div(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] / b[i];
    }
}

static void vec_add_scalar(const double* a, double b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b;
    }
}

static void vec_mul_scalar(const double* a, double b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b;
    }
}

static void vec_pow(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = pow(a[i], b[i]);
    }
}

static void vec_pow_scalar(const double* a, double b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = pow(a[i], b);
    }
}

static void vec_sqrt(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sqrt(a[i]);
    }
}

static void vec_sin(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sin(a[i]);
    }
}

static void vec_cos(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = cos(a[i]);
    }
}

static void vec_negate(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = -a[i];
    }
}

/* ============================================================================
 * FLOAT32 VECTOR OPERATIONS
 * ============================================================================ */

static void vec_add_f32(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static void vec_sub_f32(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

static void vec_mul_f32(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

static void vec_div_f32(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] / b[i];
    }
}

static void vec_add_scalar_f32(const float* a, float b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b;
    }
}

static void vec_mul_scalar_f32(const float* a, float b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b;
    }
}

static void vec_pow_f32(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = powf(a[i], b[i]);
    }
}

static void vec_pow_scalar_f32(const float* a, float b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = powf(a[i], b);
    }
}

static void vec_sqrt_f32(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sqrtf(a[i]);
    }
}

static void vec_sin_f32(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sinf(a[i]);
    }
}

static void vec_cos_f32(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = cosf(a[i]);
    }
}

static void vec_negame_f32(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = -a[i];
    }
}

/* ============================================================================
 * INTEGER VECTOR OPERATIONS (int8_t through uint64_t)
 * ============================================================================ */

/* Macros to generate integer vector operations */
#define DEFINE_INT_VEC_OPS(SUFFIX, TYPE) \
static void vec_add_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] + b[i]; \
} \
static void vec_sub_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] - b[i]; \
} \
static void vec_mul_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] * b[i]; \
} \
static void vec_div_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (b[i] != 0) ? (a[i] / b[i]) : 0; \
} \
static void vec_add_scalar_##SUFFIX(const TYPE *a, TYPE b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] + b; \
} \
static void vec_mul_scalar_##SUFFIX(const TYPE *a, TYPE b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] * b; \
} \
static void vec_pow_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (TYPE)pow((double)a[i], (double)b[i]); \
} \
static void vec_pow_scalar_##SUFFIX(const TYPE *a, TYPE b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (TYPE)pow((double)a[i], (double)b); \
} \
static void vec_sqrt_##SUFFIX(const TYPE *a, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (TYPE)sqrt((double)a[i]); \
} \
static void vec_negame_##SUFFIX(const TYPE *a, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = -a[i]; \
} \
static void vec_and_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] & b[i]; \
} \
static void vec_or_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] | b[i]; \
} \
static void vec_xor_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] ^ b[i]; \
} \
static void vec_not_##SUFFIX(const TYPE *a, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = ~a[i]; \
} \
static void vec_shl_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] << b[i]; \
} \
static void vec_shr_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] >> b[i]; \
}

/* Generate ops for all integer types */
DEFINE_INT_VEC_OPS(i8, int8_t)
DEFINE_INT_VEC_OPS(i16, int16_t)
DEFINE_INT_VEC_OPS(i32, int32_t)
DEFINE_INT_VEC_OPS(i64, int64_t)
DEFINE_INT_VEC_OPS(u8, uint8_t)
DEFINE_INT_VEC_OPS(u16, uint16_t)
DEFINE_INT_VEC_OPS(u32, uint32_t)
DEFINE_INT_VEC_OPS(u64, uint64_t)

/* Boolean logical operations */
static void vec_and_bool(const bool* a, const bool* b, bool* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = a[i] && b[i];
}

static void vec_or_bool(const bool* a, const bool* b, bool* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = a[i] || b[i];
}

static void vec_xor_bool(const bool* a, const bool* b, bool* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = a[i] != b[i];
}

static void vec_not_bool(const bool* a, bool* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = !a[i];
}

/* Comparison operations - generate for all numeric types */
/* Note: These return bool arrays, but we'll store them as the same type for simplicity */
#define DEFINE_COMPARE_OPS(SUFFIX, TYPE) \
static void vec_cmp_eq_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] == b[i]) ? 1 : 0; \
} \
static void vec_cmp_ne_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] != b[i]) ? 1 : 0; \
} \
static void vec_cmp_lt_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] < b[i]) ? 1 : 0; \
} \
static void vec_cmp_le_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] <= b[i]) ? 1 : 0; \
} \
static void vec_cmp_gt_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] > b[i]) ? 1 : 0; \
} \
static void vec_cmp_ge_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] >= b[i]) ? 1 : 0; \
}

/* Generate comparison ops for all types */
DEFINE_COMPARE_OPS(i8, int8_t)
DEFINE_COMPARE_OPS(i16, int16_t)
DEFINE_COMPARE_OPS(i32, int32_t)
DEFINE_COMPARE_OPS(i64, int64_t)
DEFINE_COMPARE_OPS(u8, uint8_t)
DEFINE_COMPARE_OPS(u16, uint16_t)
DEFINE_COMPARE_OPS(u32, uint32_t)
DEFINE_COMPARE_OPS(u64, uint64_t)
DEFINE_COMPARE_OPS(f32, float)
DEFINE_COMPARE_OPS(f64, double)

/* Complex operations */
static void vec_add_c64(const float _Complex* a, const float _Complex* b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = add_c64(a[i], b[i]);
}

static void vec_sub_c64(const float _Complex* a, const float _Complex* b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = sub_c64(a[i], b[i]);
}

static void vec_mul_c64(const float _Complex* a, const float _Complex* b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = mul_c64(a[i], b[i]);
}

static void vec_div_c64(const float _Complex* a, const float _Complex* b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = div_c64(a[i], b[i]);
}

static void vec_add_scalar_c64(const float _Complex* a, float _Complex b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = add_c64(a[i], b);
}

static void vec_mul_scalar_c64(const float _Complex* a, float _Complex b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = mul_c64(a[i], b);
}

static void vec_pow_c64(const float _Complex* a, const float _Complex* b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cpowf(a[i], b[i]);
}

static void vec_pow_scalar_c64(const float _Complex* a, float _Complex b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cpowf(a[i], b);
}

static void vec_sqrt_c64(const float _Complex* a, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_csqrtf(a[i]);
}

static void vec_negame_c64(const float _Complex* a, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = neg_c64(a[i]);
}

static void vec_conj_c64(const float _Complex* a, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_conjf(a[i]);
}

static void vec_imag_c64(const float _Complex* a, float* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cimagf(a[i]);
}

static void vec_add_c128(const double _Complex* a, const double _Complex* b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = add_c128(a[i], b[i]);
}

static void vec_sub_c128(const double _Complex* a, const double _Complex* b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = sub_c128(a[i], b[i]);
}

static void vec_mul_c128(const double _Complex* a, const double _Complex* b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = mul_c128(a[i], b[i]);
}

static void vec_div_c128(const double _Complex* a, const double _Complex* b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = div_c128(a[i], b[i]);
}

static void vec_add_scalar_c128(const double _Complex* a, double _Complex b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = add_c128(a[i], b);
}

static void vec_mul_scalar_c128(const double _Complex* a, double _Complex b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = mul_c128(a[i], b);
}

static void vec_pow_c128(const double _Complex* a, const double _Complex* b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cpow(a[i], b[i]);
}

static void vec_pow_scalar_c128(const double _Complex* a, double _Complex b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cpow(a[i], b);
}

static void vec_sqrt_c128(const double _Complex* a, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_csqrt(a[i]);
}

static void vec_negame_c128(const double _Complex* a, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = neg_c128(a[i]);
}

static void vec_conj_c128(const double _Complex* a, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_conj(a[i]);
}

static void vec_imag_c128(const double _Complex* a, double* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cimag(a[i]);
}

/* ============================================================================
 * TYPE CONVERSION FUNCTIONS
 * ============================================================================
 * These functions convert between different data types for mixed-type expressions.
 */

#define DEFINE_VEC_CONVERT(FROM_SUFFIX, TO_SUFFIX, FROM_TYPE, TO_TYPE) \
static void vec_convert_##FROM_SUFFIX##_to_##TO_SUFFIX(const FROM_TYPE *in, TO_TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = TO_TYPE_##TO_SUFFIX(in[i]); \
}


/* Generate all conversion functions */
/* Conversions FROM bool TO other types */
DEFINE_VEC_CONVERT(bool, i8, bool, int8_t)
DEFINE_VEC_CONVERT(bool, i16, bool, int16_t)
DEFINE_VEC_CONVERT(bool, i32, bool, int32_t)
DEFINE_VEC_CONVERT(bool, i64, bool, int64_t)
DEFINE_VEC_CONVERT(bool, u8, bool, uint8_t)
DEFINE_VEC_CONVERT(bool, u16, bool, uint16_t)
DEFINE_VEC_CONVERT(bool, u32, bool, uint32_t)
DEFINE_VEC_CONVERT(bool, u64, bool, uint64_t)
DEFINE_VEC_CONVERT(bool, f32, bool, float)
DEFINE_VEC_CONVERT(bool, f64, bool, double)

/* Conversions FROM other types TO bool */
DEFINE_VEC_CONVERT(i8, bool, int8_t, bool)
DEFINE_VEC_CONVERT(i16, bool, int16_t, bool)
DEFINE_VEC_CONVERT(i32, bool, int32_t, bool)
DEFINE_VEC_CONVERT(i64, bool, int64_t, bool)
DEFINE_VEC_CONVERT(u8, bool, uint8_t, bool)
DEFINE_VEC_CONVERT(u16, bool, uint16_t, bool)
DEFINE_VEC_CONVERT(u32, bool, uint32_t, bool)
DEFINE_VEC_CONVERT(u64, bool, uint64_t, bool)
DEFINE_VEC_CONVERT(f32, bool, float, bool)
DEFINE_VEC_CONVERT(f64, bool, double, bool)
DEFINE_VEC_CONVERT(f64, f32, double, float)

DEFINE_VEC_CONVERT(i8, i16, int8_t, int16_t)
DEFINE_VEC_CONVERT(i8, i32, int8_t, int32_t)
DEFINE_VEC_CONVERT(i8, i64, int8_t, int64_t)
DEFINE_VEC_CONVERT(i8, f32, int8_t, float)
DEFINE_VEC_CONVERT(i8, f64, int8_t, double)

DEFINE_VEC_CONVERT(i16, i32, int16_t, int32_t)
DEFINE_VEC_CONVERT(i16, i64, int16_t, int64_t)
DEFINE_VEC_CONVERT(i16, f32, int16_t, float)
DEFINE_VEC_CONVERT(i16, f64, int16_t, double)

DEFINE_VEC_CONVERT(i32, i64, int32_t, int64_t)
DEFINE_VEC_CONVERT(i32, f32, int32_t, float)
DEFINE_VEC_CONVERT(i32, f64, int32_t, double)

DEFINE_VEC_CONVERT(i64, f64, int64_t, double)

DEFINE_VEC_CONVERT(u8, u16, uint8_t, uint16_t)
DEFINE_VEC_CONVERT(u8, u32, uint8_t, uint32_t)
DEFINE_VEC_CONVERT(u8, u64, uint8_t, uint64_t)
DEFINE_VEC_CONVERT(u8, i16, uint8_t, int16_t)
DEFINE_VEC_CONVERT(u8, i32, uint8_t, int32_t)
DEFINE_VEC_CONVERT(u8, i64, uint8_t, int64_t)
DEFINE_VEC_CONVERT(u8, f32, uint8_t, float)
DEFINE_VEC_CONVERT(u8, f64, uint8_t, double)

DEFINE_VEC_CONVERT(u16, u32, uint16_t, uint32_t)
DEFINE_VEC_CONVERT(u16, u64, uint16_t, uint64_t)
DEFINE_VEC_CONVERT(u16, i32, uint16_t, int32_t)
DEFINE_VEC_CONVERT(u16, i64, uint16_t, int64_t)
DEFINE_VEC_CONVERT(u16, f32, uint16_t, float)
DEFINE_VEC_CONVERT(u16, f64, uint16_t, double)

DEFINE_VEC_CONVERT(u32, u64, uint32_t, uint64_t)
DEFINE_VEC_CONVERT(u32, i64, uint32_t, int64_t)
DEFINE_VEC_CONVERT(u32, f64, uint32_t, double)

DEFINE_VEC_CONVERT(u64, f64, uint64_t, double)

DEFINE_VEC_CONVERT(f32, f64, float, double)
DEFINE_VEC_CONVERT(f32, c64, float, float _Complex)
DEFINE_VEC_CONVERT(f32, c128, float, double _Complex)

DEFINE_VEC_CONVERT(f64, c128, double, double _Complex)

DEFINE_VEC_CONVERT(c64, c128, float _Complex, double _Complex)

/* Function to get conversion function pointer */
typedef void (*convert_func_t)(const void*, void*, int);

static convert_func_t get_convert_func(me_dtype from, me_dtype to) {
    /* Return conversion function for a specific type pair */
    if (from == to) return NULL; // No conversion needed

#define CONV_CASE(FROM, TO, FROM_S, TO_S) \
        if (from == FROM && to == TO) return (convert_func_t)vec_convert_##FROM_S##_to_##TO_S;

    CONV_CASE(ME_BOOL, ME_INT8, bool, i8)
    CONV_CASE(ME_BOOL, ME_INT16, bool, i16)
    CONV_CASE(ME_BOOL, ME_INT32, bool, i32)
    CONV_CASE(ME_BOOL, ME_INT64, bool, i64)
    CONV_CASE(ME_BOOL, ME_UINT8, bool, u8)
    CONV_CASE(ME_BOOL, ME_UINT16, bool, u16)
    CONV_CASE(ME_BOOL, ME_UINT32, bool, u32)
    CONV_CASE(ME_BOOL, ME_UINT64, bool, u64)
    CONV_CASE(ME_BOOL, ME_FLOAT32, bool, f32)
    CONV_CASE(ME_BOOL, ME_FLOAT64, bool, f64)

    CONV_CASE(ME_INT8, ME_BOOL, i8, bool)
    CONV_CASE(ME_INT16, ME_BOOL, i16, bool)
    CONV_CASE(ME_INT32, ME_BOOL, i32, bool)
    CONV_CASE(ME_INT64, ME_BOOL, i64, bool)
    CONV_CASE(ME_UINT8, ME_BOOL, u8, bool)
    CONV_CASE(ME_UINT16, ME_BOOL, u16, bool)
    CONV_CASE(ME_UINT32, ME_BOOL, u32, bool)
    CONV_CASE(ME_UINT64, ME_BOOL, u64, bool)
    CONV_CASE(ME_FLOAT32, ME_BOOL, f32, bool)
    CONV_CASE(ME_FLOAT64, ME_BOOL, f64, bool)

    CONV_CASE(ME_INT8, ME_INT16, i8, i16)
    CONV_CASE(ME_INT8, ME_INT32, i8, i32)
    CONV_CASE(ME_INT8, ME_INT64, i8, i64)
    CONV_CASE(ME_INT8, ME_FLOAT32, i8, f32)
    CONV_CASE(ME_INT8, ME_FLOAT64, i8, f64)

    CONV_CASE(ME_INT16, ME_INT32, i16, i32)
    CONV_CASE(ME_INT16, ME_INT64, i16, i64)
    CONV_CASE(ME_INT16, ME_FLOAT32, i16, f32)
    CONV_CASE(ME_INT16, ME_FLOAT64, i16, f64)

    CONV_CASE(ME_INT32, ME_INT64, i32, i64)
    CONV_CASE(ME_INT32, ME_FLOAT32, i32, f32)
    CONV_CASE(ME_INT32, ME_FLOAT64, i32, f64)

    CONV_CASE(ME_INT64, ME_FLOAT64, i64, f64)

    CONV_CASE(ME_UINT8, ME_UINT16, u8, u16)
    CONV_CASE(ME_UINT8, ME_UINT32, u8, u32)
    CONV_CASE(ME_UINT8, ME_UINT64, u8, u64)
    CONV_CASE(ME_UINT8, ME_INT16, u8, i16)
    CONV_CASE(ME_UINT8, ME_INT32, u8, i32)
    CONV_CASE(ME_UINT8, ME_INT64, u8, i64)
    CONV_CASE(ME_UINT8, ME_FLOAT32, u8, f32)
    CONV_CASE(ME_UINT8, ME_FLOAT64, u8, f64)

    CONV_CASE(ME_UINT16, ME_UINT32, u16, u32)
    CONV_CASE(ME_UINT16, ME_UINT64, u16, u64)
    CONV_CASE(ME_UINT16, ME_INT32, u16, i32)
    CONV_CASE(ME_UINT16, ME_INT64, u16, i64)
    CONV_CASE(ME_UINT16, ME_FLOAT32, u16, f32)
    CONV_CASE(ME_UINT16, ME_FLOAT64, u16, f64)

    CONV_CASE(ME_UINT32, ME_UINT64, u32, u64)
    CONV_CASE(ME_UINT32, ME_INT64, u32, i64)
    CONV_CASE(ME_UINT32, ME_FLOAT64, u32, f64)

    CONV_CASE(ME_UINT64, ME_FLOAT64, u64, f64)

    CONV_CASE(ME_FLOAT32, ME_FLOAT64, f32, f64)
    CONV_CASE(ME_FLOAT32, ME_COMPLEX64, f32, c64)
    CONV_CASE(ME_FLOAT32, ME_COMPLEX128, f32, c128)

    CONV_CASE(ME_FLOAT64, ME_FLOAT32, f64, f32)
    CONV_CASE(ME_FLOAT64, ME_COMPLEX128, f64, c128)

    CONV_CASE(ME_COMPLEX64, ME_COMPLEX128, c64, c128)

#undef CONV_CASE

    return NULL; // Unsupported conversion
}


typedef double (*me_fun1)(double);

typedef float (*me_fun1_f32)(float);

/* Template for type-specific evaluator */
#define DEFINE_ME_EVAL(SUFFIX, TYPE, VEC_ADD, VEC_SUB, VEC_MUL, VEC_DIV, VEC_POW, \
    VEC_ADD_SCALAR, VEC_MUL_SCALAR, VEC_POW_SCALAR, \
    VEC_SQRT, VEC_SIN, VEC_COS, VEC_NEGATE, \
    SQRT_FUNC, SIN_FUNC, COS_FUNC, EXP_FUNC, LOG_FUNC, FABS_FUNC, POW_FUNC, \
    VEC_CONJ) \
static void me_eval_##SUFFIX(const me_expr *n) { \
    if (!n || !n->output) return; \
    if (is_reduction_node(n)) { \
        eval_reduction(n, n->nitems); \
        return; \
    } \
    if (n->nitems <= 0) return; \
    \
    int i, j; \
    const int arity = ARITY(n->type); \
    TYPE *output = (TYPE*)n->output; \
    \
    switch(TYPE_MASK(n->type)) { \
        case ME_CONSTANT: \
            { \
                TYPE val = TO_TYPE_##SUFFIX(n->value); \
                for (i = 0; i < n->nitems; i++) { \
                    output[i] = val; \
                } \
            } \
            break; \
            \
        case ME_VARIABLE: \
            { \
                const TYPE *src = (const TYPE*)n->bound; \
                for (i = 0; i < n->nitems; i++) { \
                    output[i] = src[i]; \
                } \
            } \
            break; \
        \
        case ME_FUNCTION0: case ME_FUNCTION1: case ME_FUNCTION2: case ME_FUNCTION3: \
        case ME_FUNCTION4: case ME_FUNCTION5: case ME_FUNCTION6: case ME_FUNCTION7: \
        case ME_CLOSURE0: case ME_CLOSURE1: case ME_CLOSURE2: case ME_CLOSURE3: \
        case ME_CLOSURE4: case ME_CLOSURE5: case ME_CLOSURE6: case ME_CLOSURE7: \
            for (j = 0; j < arity; j++) { \
                me_expr *param = (me_expr*)n->parameters[j]; \
                if (param->type != ME_CONSTANT && param->type != ME_VARIABLE) { \
                    if (!param->output) { \
                        param->output = malloc(n->nitems * sizeof(TYPE)); \
                        param->nitems = n->nitems; \
                        param->dtype = n->dtype; \
                    } \
                    me_eval_##SUFFIX(param); \
                } \
            } \
            \
            if (arity == 2 && IS_FUNCTION(n->type)) { \
                me_expr *left = (me_expr*)n->parameters[0]; \
                me_expr *right = (me_expr*)n->parameters[1]; \
                \
                const TYPE *ldata = (left->type == ME_CONSTANT) ? NULL : \
                                   (left->type == ME_VARIABLE) ? (const TYPE*)left->bound : (const TYPE*)left->output; \
                const TYPE *rdata = (right->type == ME_CONSTANT) ? NULL : \
                                    (right->type == ME_VARIABLE) ? (const TYPE*)right->bound : (const TYPE*)right->output; \
                \
                me_fun2 func = (me_fun2)n->function; \
                \
                if (func == add) { \
                    if (ldata && rdata) { \
                        VEC_ADD(ldata, rdata, output, n->nitems); \
                    } else if (ldata && right->type == ME_CONSTANT) { \
                        VEC_ADD_SCALAR(ldata, TO_TYPE_##SUFFIX(right->value), output, n->nitems); \
                    } else if (left->type == ME_CONSTANT && rdata) { \
                        VEC_ADD_SCALAR(rdata, TO_TYPE_##SUFFIX(left->value), output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == sub) { \
                    if (ldata && rdata) { \
                        VEC_SUB(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == mul) { \
                    if (ldata && rdata) { \
                        VEC_MUL(ldata, rdata, output, n->nitems); \
                    } else if (ldata && right->type == ME_CONSTANT) { \
                        VEC_MUL_SCALAR(ldata, TO_TYPE_##SUFFIX(right->value), output, n->nitems); \
                    } else if (left->type == ME_CONSTANT && rdata) { \
                        VEC_MUL_SCALAR(rdata, TO_TYPE_##SUFFIX(left->value), output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == divide) { \
                    if (ldata && rdata) { \
                        VEC_DIV(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)pow) { \
                    if (ldata && rdata) { \
                        VEC_POW(ldata, rdata, output, n->nitems); \
                    } else if (ldata && right->type == ME_CONSTANT) { \
                        VEC_POW_SCALAR(ldata, TO_TYPE_##SUFFIX(right->value), output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else { \
                    general_case_binary_##SUFFIX: \
                    for (i = 0; i < n->nitems; i++) { \
                        double a = (left->type == ME_CONSTANT) ? left->value : \
                                  FROM_TYPE_##SUFFIX(ldata[i]); \
                        double b = (right->type == ME_CONSTANT) ? right->value : \
                                  FROM_TYPE_##SUFFIX(rdata[i]); \
                        output[i] = TO_TYPE_##SUFFIX(func(a, b)); \
                    } \
                } \
            } else if (arity == 3 && IS_FUNCTION(n->type) && n->function == (void*)where_scalar) { \
                /* where(cond, x, y) – NumPy-like semantics: cond != 0 selects x else y */ \
                me_expr *cond = (me_expr*)n->parameters[0]; \
                me_expr *xexpr = (me_expr*)n->parameters[1]; \
                me_expr *yexpr = (me_expr*)n->parameters[2]; \
                \
                const TYPE *cdata = (const TYPE*)((cond->type == ME_VARIABLE) ? cond->bound : cond->output); \
                const TYPE *xdata = (const TYPE*)((xexpr->type == ME_VARIABLE) ? xexpr->bound : xexpr->output); \
                const TYPE *ydata = (const TYPE*)((yexpr->type == ME_VARIABLE) ? yexpr->bound : yexpr->output); \
                \
                for (i = 0; i < n->nitems; i++) { \
                    output[i] = (IS_NONZERO_##SUFFIX(cdata[i])) ? xdata[i] : ydata[i]; \
                } \
            } \
            else if (arity == 1 && IS_FUNCTION(n->type)) { \
                me_expr *arg = (me_expr*)n->parameters[0]; \
                \
                const TYPE *adata = (arg->type == ME_CONSTANT) ? NULL : \
                                   (arg->type == ME_VARIABLE) ? (const TYPE*)arg->bound : (const TYPE*)arg->output; \
                \
                const void *func_ptr = n->function; \
                \
                if (func_ptr == (void*)sqrt) { \
                    if (adata) VEC_SQRT(adata, output, n->nitems); \
                } else if (func_ptr == (void*)sin) { \
                    if (adata) VEC_SIN(adata, output, n->nitems); \
                } else if (func_ptr == (void*)cos) { \
                    if (adata) VEC_COS(adata, output, n->nitems); \
                } else if (func_ptr == (void*)negate) { \
                    if (adata) VEC_NEGATE(adata, output, n->nitems); \
                } else if (func_ptr == (void*)imag_wrapper) { \
                    /* NumPy semantics: imag(real) == 0 with same dtype */ \
                    if (adata) { \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = TO_TYPE_##SUFFIX(0); \
                        } \
                    } \
                } else if (func_ptr == (void*)real_wrapper) { \
                    /* NumPy semantics: real(real) == real with same dtype */ \
                    if (adata) { \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = adata[i]; \
                        } \
                    } \
                } else if (func_ptr == (void*)conj_wrapper) { \
                    if (adata) VEC_CONJ(adata, output, n->nitems); \
                } else { \
                    me_fun1 func = (me_fun1)func_ptr; \
                    if (arg->type == ME_CONSTANT) { \
                        TYPE val = TO_TYPE_##SUFFIX(func(arg->value)); \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = val; \
                        } \
                    } else { \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = TO_TYPE_##SUFFIX(func(FROM_TYPE_##SUFFIX(adata[i]))); \
                        } \
                    } \
                } \
            } \
            else { \
                for (i = 0; i < n->nitems; i++) { \
                    double args[7]; \
                    \
                    for (j = 0; j < arity; j++) { \
                        me_expr *param = (me_expr*)n->parameters[j]; \
                        const TYPE *pdata = (const TYPE*)((param->type == ME_VARIABLE) ? param->bound : param->output); \
                        if (param->type == ME_CONSTANT) { \
                            args[j] = param->value; \
                        } else { \
                            args[j] = FROM_TYPE_##SUFFIX(pdata[i]); \
                        } \
                    } \
                    \
                    if (IS_FUNCTION(n->type)) { \
                        switch(arity) { \
                            case 0: output[i] = TO_TYPE_##SUFFIX(((double(*)(void))n->function)()); break; \
                            case 3: output[i] = TO_TYPE_##SUFFIX(((double(*)(double,double,double))n->function)(args[0], args[1], args[2])); break; \
                            case 4: output[i] = TO_TYPE_##SUFFIX(((double(*)(double,double,double,double))n->function)(args[0], args[1], args[2], args[3])); break; \
                            case 5: output[i] = TO_TYPE_##SUFFIX(((double(*)(double,double,double,double,double))n->function)(args[0], args[1], args[2], args[3], args[4])); break; \
                            case 6: output[i] = TO_TYPE_##SUFFIX(((double(*)(double,double,double,double,double,double))n->function)(args[0], args[1], args[2], args[3], args[4], args[5])); break; \
                            case 7: output[i] = TO_TYPE_##SUFFIX(((double(*)(double,double,double,double,double,double,double))n->function)(args[0], args[1], args[2], args[3], args[4], args[5], args[6])); break; \
                        } \
                    } else if (IS_CLOSURE(n->type)) { \
                        void *context = n->parameters[arity]; \
                        switch(arity) { \
                            case 0: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*))n->function)(context)); break; \
                            case 1: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double))n->function)(context, args[0])); break; \
                            case 2: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double))n->function)(context, args[0], args[1])); break; \
                            case 3: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double,double))n->function)(context, args[0], args[1], args[2])); break; \
                            case 4: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3])); break; \
                            case 5: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3], args[4])); break; \
                            case 6: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3], args[4], args[5])); break; \
                            case 7: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double,double,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3], args[4], args[5], args[6])); break; \
                        } \
                    } \
                } \
            } \
            break; \
        \
        default: \
            for (i = 0; i < n->nitems; i++) { \
                output[i] = TO_TYPE_##SUFFIX(NAN); \
            } \
            break; \
    } \
}

/* Vector operation macros - expand to inline loops */
#define vec_add(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = pow((a)[_i], (b)); } while(0)
#define vec_sqrt(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = sqrt((a)[_i]); } while(0)
#define vec_sin(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = sin((a)[_i]); } while(0)
#define vec_cos(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cos((a)[_i]); } while(0)
#define vec_negate(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)
#define vec_copy(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i]; } while(0)

#define vec_add_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = powf((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = powf((a)[_i], (b)); } while(0)
#define vec_sqrt_f32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = sqrtf((a)[_i]); } while(0)
#define vec_sin_f32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = sinf((a)[_i]); } while(0)
#define vec_cos_f32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cosf((a)[_i]); } while(0)
#define vec_negame_f32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int8_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int8_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_i8(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int8_t)sqrt((a)[_i]); } while(0)
#define vec_negame_i8(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int16_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int16_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_i16(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int16_t)sqrt((a)[_i]); } while(0)
#define vec_negame_i16(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int32_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int32_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_i32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int32_t)sqrt((a)[_i]); } while(0)
#define vec_negame_i32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int64_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int64_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_i64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int64_t)sqrt((a)[_i]); } while(0)
#define vec_negame_i64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint8_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint8_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_u8(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint8_t)sqrt((a)[_i]); } while(0)
#define vec_negame_u8(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint16_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint16_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_u16(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint16_t)sqrt((a)[_i]); } while(0)
#define vec_negame_u16(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint32_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint32_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_u32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint32_t)sqrt((a)[_i]); } while(0)
#define vec_negame_u32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint64_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint64_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_u64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint64_t)sqrt((a)[_i]); } while(0)
#define vec_negame_u64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#if defined(_MSC_VER) && !defined(__clang__)
#define vec_add_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = add_c64((a)[_i], (b)[_i]); } while(0)
#define vec_sub_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = sub_c64((a)[_i], (b)[_i]); } while(0)
#define vec_mul_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = mul_c64((a)[_i], (b)[_i]); } while(0)
#define vec_div_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = div_c64((a)[_i], (b)[_i]); } while(0)
#define vec_pow_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpowf((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = add_c64((a)[_i], (b)); } while(0)
#define vec_mul_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = mul_c64((a)[_i], (b)); } while(0)
#define vec_pow_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpowf((a)[_i], (b)); } while(0)
#define vec_sqrt_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = csqrtf((a)[_i]); } while(0)
#define vec_negame_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = neg_c64((a)[_i]); } while(0)
#define vec_conj_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = conjf((a)[_i]); } while(0)
#define vec_imag_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cimagf((a)[_i]); } while(0)
#define vec_real_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_crealf((a)[_i]); } while(0)
#define vec_conj_noop(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i]; } while(0)

#define vec_add_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = add_c128((a)[_i], (b)[_i]); } while(0)
#define vec_sub_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = sub_c128((a)[_i], (b)[_i]); } while(0)
#define vec_mul_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = mul_c128((a)[_i], (b)[_i]); } while(0)
#define vec_div_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = div_c128((a)[_i], (b)[_i]); } while(0)
#define vec_pow_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = add_c128((a)[_i], (b)); } while(0)
#define vec_mul_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = mul_c128((a)[_i], (b)); } while(0)
#define vec_pow_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpow((a)[_i], (b)); } while(0)
#define vec_sqrt_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = csqrt((a)[_i]); } while(0)
#define vec_negame_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = neg_c128((a)[_i]); } while(0)
#define vec_conj_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = conj((a)[_i]); } while(0)
#define vec_imag_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cimag((a)[_i]); } while(0)
#define vec_real_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_creal((a)[_i]); } while(0)
#else
#define vec_add_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cpowf((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cpowf((a)[_i], (b)); } while(0)
#define vec_sqrt_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_csqrtf((a)[_i]); } while(0)
#define vec_negame_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)
#define vec_conj_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_conjf((a)[_i]); } while(0)
#define vec_imag_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cimagf((a)[_i]); } while(0)
#define vec_real_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_crealf((a)[_i]); } while(0)
#define vec_conj_noop(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i]; } while(0)

#define vec_add_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cpow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cpow((a)[_i], (b)); } while(0)
#define vec_sqrt_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_csqrt((a)[_i]); } while(0)
#define vec_negame_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)
#define vec_conj_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_conj((a)[_i]); } while(0)
#define vec_imag_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cimag((a)[_i]); } while(0)
#define vec_real_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_creal((a)[_i]); } while(0)
#endif

/* Generate float32 evaluator */
DEFINE_ME_EVAL(f32, float,
               vec_add_f32, vec_sub_f32, vec_mul_f32, vec_div_f32, vec_pow_f32,
               vec_add_scalar_f32, vec_mul_scalar_f32, vec_pow_scalar_f32,
               vec_sqrt_f32, vec_sin_f32, vec_cos_f32, vec_negame_f32,
               sqrtf, sinf, cosf, expf, logf, fabsf, powf,
               vec_copy)

/* Generate float64 (double) evaluator */
DEFINE_ME_EVAL(f64, double,
               vec_add, vec_sub, vec_mul, vec_div, vec_pow,
               vec_add_scalar, vec_mul_scalar, vec_pow_scalar,
               vec_sqrt, vec_sin, vec_cos, vec_negate,
               sqrt, sin, cos, exp, log, fabs, pow,
               vec_copy)

/* Generate integer evaluators - sin/cos cast to double and back */
DEFINE_ME_EVAL(i8, int8_t,
               vec_add_i8, vec_sub_i8, vec_mul_i8, vec_div_i8, vec_pow_i8,
               vec_add_scalar_i8, vec_mul_scalar_i8, vec_pow_scalar_i8,
               vec_sqrt_i8, vec_sqrt_i8, vec_sqrt_i8, vec_negame_i8,
               sqrt, sin, cos, exp, log, fabs, pow,
               vec_conj_noop)

DEFINE_ME_EVAL(i16, int16_t,
               vec_add_i16, vec_sub_i16, vec_mul_i16, vec_div_i16, vec_pow_i16,
               vec_add_scalar_i16, vec_mul_scalar_i16, vec_pow_scalar_i16,
               vec_sqrt_i16, vec_sqrt_i16, vec_sqrt_i16, vec_negame_i16,
               sqrt, sin, cos, exp, log, fabs, pow,
               vec_conj_noop)

DEFINE_ME_EVAL(i32, int32_t,
               vec_add_i32, vec_sub_i32, vec_mul_i32, vec_div_i32, vec_pow_i32,
               vec_add_scalar_i32, vec_mul_scalar_i32, vec_pow_scalar_i32,
               vec_sqrt_i32, vec_sqrt_i32, vec_sqrt_i32, vec_negame_i32,
               sqrt, sin, cos, exp, log, fabs, pow,
               vec_conj_noop)

DEFINE_ME_EVAL(i64, int64_t,
               vec_add_i64, vec_sub_i64, vec_mul_i64, vec_div_i64, vec_pow_i64,
               vec_add_scalar_i64, vec_mul_scalar_i64, vec_pow_scalar_i64,
               vec_sqrt_i64, vec_sqrt_i64, vec_sqrt_i64, vec_negame_i64,
               sqrt, sin, cos, exp, log, fabs, pow,
               vec_conj_noop)

DEFINE_ME_EVAL(u8, uint8_t,
               vec_add_u8, vec_sub_u8, vec_mul_u8, vec_div_u8, vec_pow_u8,
               vec_add_scalar_u8, vec_mul_scalar_u8, vec_pow_scalar_u8,
               vec_sqrt_u8, vec_sqrt_u8, vec_sqrt_u8, vec_negame_u8,
               sqrt, sin, cos, exp, log, fabs, pow,
               vec_conj_noop)

DEFINE_ME_EVAL(u16, uint16_t,
               vec_add_u16, vec_sub_u16, vec_mul_u16, vec_div_u16, vec_pow_u16,
               vec_add_scalar_u16, vec_mul_scalar_u16, vec_pow_scalar_u16,
               vec_sqrt_u16, vec_sqrt_u16, vec_sqrt_u16, vec_negame_u16,
               sqrt, sin, cos, exp, log, fabs, pow,
               vec_conj_noop)

DEFINE_ME_EVAL(u32, uint32_t,
               vec_add_u32, vec_sub_u32, vec_mul_u32, vec_div_u32, vec_pow_u32,
               vec_add_scalar_u32, vec_mul_scalar_u32, vec_pow_scalar_u32,
               vec_sqrt_u32, vec_sqrt_u32, vec_sqrt_u32, vec_negame_u32,
               sqrt, sin, cos, exp, log, fabs, pow,
               vec_conj_noop)

DEFINE_ME_EVAL(u64, uint64_t,
               vec_add_u64, vec_sub_u64, vec_mul_u64, vec_div_u64, vec_pow_u64,
               vec_add_scalar_u64, vec_mul_scalar_u64, vec_pow_scalar_u64,
               vec_sqrt_u64, vec_sqrt_u64, vec_sqrt_u64, vec_negame_u64,
               sqrt, sin, cos, exp, log, fabs, pow,
               vec_conj_noop)

/* Generate complex evaluators */
DEFINE_ME_EVAL(c64, float _Complex,
               vec_add_c64, vec_sub_c64, vec_mul_c64, vec_div_c64, vec_pow_c64,
               vec_add_scalar_c64, vec_mul_scalar_c64, vec_pow_scalar_c64,
               vec_sqrt_c64, vec_sqrt_c64, vec_sqrt_c64, vec_negame_c64,
               me_csqrtf, me_csqrtf, me_csqrtf, me_cexpf, me_clogf, me_cabsf, me_cpowf,
               vec_conj_c64)

DEFINE_ME_EVAL(c128, double _Complex,
               vec_add_c128, vec_sub_c128, vec_mul_c128, vec_div_c128, vec_pow_c128,
               vec_add_scalar_c128, vec_mul_scalar_c128, vec_pow_scalar_c128,
               vec_sqrt_c128, vec_sqrt_c128, vec_sqrt_c128, vec_negame_c128,
               me_csqrt, me_csqrt, me_csqrt, me_cexp, me_clog, me_cabs, me_cpow,
               vec_conj_c128)

/* Public API - dispatches to correct type-specific evaluator */
/* Structure to track promoted variables */
typedef struct {
    void* promoted_data; // Temporary buffer for promoted data
    me_dtype original_type;
    bool needs_free;
} promoted_var_t;

/* Helper to save original variable bindings */
static void save_variable_bindings(const me_expr* node,
                                   const void** original_bounds,
                                   me_dtype* original_types,
                                   int* save_idx) {
    if (!node) return;
    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        original_bounds[*save_idx] = node->bound;
        original_types[*save_idx] = node->dtype;
        (*save_idx)++;
        break;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                save_variable_bindings((const me_expr*)node->parameters[i],
                                       original_bounds, original_types, save_idx);
            }
            break;
        }
    }
}

/* Recursively promote variables in expression tree */
static void promote_variables_in_tree(me_expr* n, me_dtype target_type,
                                      promoted_var_t* promotions, int* promo_count,
                                      int nitems) {
    if (!n) return;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        // Constants are promoted on-the-fly during evaluation
        break;

    case ME_VARIABLE:
        if (n->dtype != target_type) {
            // Need to promote this variable
            void* promoted = malloc(nitems * dtype_size(target_type));
            if (promoted) {
                convert_func_t conv = get_convert_func(n->dtype, target_type);
                if (conv) {
                    conv(n->bound, promoted, nitems);

                    // Track this promotion for later cleanup
                    promotions[*promo_count].promoted_data = promoted;
                    promotions[*promo_count].original_type = n->dtype;
                    promotions[*promo_count].needs_free = true;
                    (*promo_count)++;

                    // Temporarily replace bound pointer
                    n->bound = promoted;
                    n->dtype = target_type;
                }
                else {
                    free(promoted);
                }
            }
        }
        break;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                promote_variables_in_tree((me_expr*)n->parameters[i], target_type,
                                          promotions, promo_count, nitems);
            }
            break;
        }
    }
}

/* Restore original variable bindings after promotion */
static void restore_variables_in_tree(me_expr* n, const void** original_bounds,
                                      const me_dtype* original_types, int* restore_idx) {
    if (!n) return;

    switch (TYPE_MASK(n->type)) {
    case ME_VARIABLE:
        if (original_bounds[*restore_idx] != NULL) {
            n->bound = original_bounds[*restore_idx];
            n->dtype = original_types[*restore_idx];
            (*restore_idx)++;
        }
        break;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                restore_variables_in_tree((me_expr*)n->parameters[i], original_bounds, original_types, restore_idx);
            }
            break;
        }
    }
}

/* Check if all variables in tree match target type */
static bool all_variables_match_type(const me_expr* n, me_dtype target_type) {
    if (!n) return true;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        return true; // Constants are always OK

    case ME_VARIABLE:
        return n->dtype == target_type;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (!all_variables_match_type((const me_expr*)n->parameters[i], target_type)) {
                    return false;
                }
            }
            return true;
        }
    }

    return true;
}

static void broadcast_reduction_output(void* output, me_dtype dtype, int output_nitems) {
    if (!output || output_nitems <= 1) return;
    switch (dtype) {
    case ME_BOOL:
        {
            bool val = ((bool*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((bool*)output)[i] = val;
            }
            break;
        }
    case ME_INT8:
        {
            int8_t val = ((int8_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((int8_t*)output)[i] = val;
            }
            break;
        }
    case ME_INT16:
        {
            int16_t val = ((int16_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((int16_t*)output)[i] = val;
            }
            break;
        }
    case ME_INT32:
        {
            int32_t val = ((int32_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((int32_t*)output)[i] = val;
            }
            break;
        }
    case ME_INT64:
        {
            int64_t val = ((int64_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((int64_t*)output)[i] = val;
            }
            break;
        }
    case ME_UINT8:
        {
            uint8_t val = ((uint8_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((uint8_t*)output)[i] = val;
            }
            break;
        }
    case ME_UINT16:
        {
            uint16_t val = ((uint16_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((uint16_t*)output)[i] = val;
            }
            break;
        }
    case ME_UINT32:
        {
            uint32_t val = ((uint32_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((uint32_t*)output)[i] = val;
            }
            break;
        }
    case ME_UINT64:
        {
            uint64_t val = ((uint64_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((uint64_t*)output)[i] = val;
            }
            break;
        }
    case ME_FLOAT32:
        {
            float val = ((float*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((float*)output)[i] = val;
            }
            break;
        }
    case ME_FLOAT64:
        {
            double val = ((double*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((double*)output)[i] = val;
            }
            break;
        }
    case ME_COMPLEX64:
        {
            float _Complex val = ((float _Complex*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((float _Complex*)output)[i] = val;
            }
            break;
        }
    case ME_COMPLEX128:
        {
            double _Complex val = ((double _Complex*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((double _Complex*)output)[i] = val;
            }
            break;
        }
    default:
        break;
    }
}

static void eval_reduction(const me_expr* n, int output_nitems) {
    if (!n || !n->output || !is_reduction_node(n)) return;
    if (output_nitems <= 0) return;

    me_expr* arg = (me_expr*)n->parameters[0];
    if (!arg) return;

    const int nitems = n->nitems;
    me_dtype arg_type = arg->dtype;
    if (arg->type != ME_CONSTANT && arg->type != ME_VARIABLE) {
        arg_type = infer_output_type(arg);
        if (nitems > 0) {
            if (!arg->output) {
                arg->output = malloc((size_t)nitems * dtype_size(arg_type));
                if (!arg->output) return;
            }
            arg->nitems = nitems;
            arg->dtype = arg_type;
            private_eval(arg);
        }
    }
    me_dtype result_type = reduction_output_dtype(arg_type, n->function);
    me_dtype output_type = n->dtype;
    bool is_prod = n->function == (void*)prod_reduce;
    bool is_min = n->function == (void*)min_reduce;
    bool is_max = n->function == (void*)max_reduce;
    bool is_any = n->function == (void*)any_reduce;
    bool is_all = n->function == (void*)all_reduce;

    void* write_ptr = n->output;
    void* temp_output = NULL;
    if (output_type != result_type) {
        temp_output = malloc((size_t)output_nitems * dtype_size(result_type));
        if (!temp_output) return;
        write_ptr = temp_output;
    }

    if (arg->type == ME_CONSTANT) {
        double val = arg->value;
        if (is_any || is_all) {
            bool acc = is_all;
            if (nitems == 0) {
                acc = is_all;
            }
            else {
                switch (arg_type) {
                case ME_BOOL:
                    acc = val != 0.0;
                    break;
                case ME_INT8:
                case ME_INT16:
                case ME_INT32:
                case ME_INT64:
                case ME_UINT8:
                case ME_UINT16:
                case ME_UINT32:
                case ME_UINT64:
                case ME_FLOAT32:
                case ME_FLOAT64:
                    acc = val != 0.0;
                    break;
                case ME_COMPLEX64:
                case ME_COMPLEX128:
                    acc = val != 0.0;
                    break;
                default:
                    acc = false;
                    break;
                }
            }
            ((bool*)write_ptr)[0] = acc;
        }
        else if (is_min || is_max) {
            switch (arg_type) {
            case ME_BOOL:
                {
                    bool acc = is_min;
                    if (nitems > 0) {
                        acc = (bool)val;
                    }
                    ((bool*)write_ptr)[0] = acc;
                    break;
                }
            case ME_INT8:
                {
                    int8_t acc = (int8_t)(is_min ? INT8_MAX : INT8_MIN);
                    if (nitems > 0) acc = (int8_t)val;
                    ((int8_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_INT16:
                {
                    int16_t acc = (int16_t)(is_min ? INT16_MAX : INT16_MIN);
                    if (nitems > 0) acc = (int16_t)val;
                    ((int16_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_INT32:
                {
                    int32_t acc = (int32_t)(is_min ? INT32_MAX : INT32_MIN);
                    if (nitems > 0) acc = (int32_t)val;
                    ((int32_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_INT64:
                {
                    int64_t acc = is_min ? INT64_MAX : INT64_MIN;
                    if (nitems > 0) acc = (int64_t)val;
                    ((int64_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_UINT8:
                {
                    uint8_t acc = is_min ? UINT8_MAX : 0;
                    if (nitems > 0) acc = (uint8_t)val;
                    ((uint8_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_UINT16:
                {
                    uint16_t acc = is_min ? UINT16_MAX : 0;
                    if (nitems > 0) acc = (uint16_t)val;
                    ((uint16_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_UINT32:
                {
                    uint32_t acc = is_min ? UINT32_MAX : 0;
                    if (nitems > 0) acc = (uint32_t)val;
                    ((uint32_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_UINT64:
                {
                    uint64_t acc = is_min ? UINT64_MAX : 0;
                    if (nitems > 0) acc = (uint64_t)val;
                    ((uint64_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_FLOAT32:
                {
                    float acc = is_min ? INFINITY : -INFINITY;
                    if (nitems > 0) acc = (float)val;
                    ((float*)write_ptr)[0] = acc;
                    break;
                }
            case ME_FLOAT64:
                {
                    double acc = is_min ? INFINITY : -INFINITY;
                    if (nitems > 0) acc = val;
                    ((double*)write_ptr)[0] = acc;
                    break;
                }
            case ME_COMPLEX64:
                {
                    ((float _Complex*)write_ptr)[0] = (float _Complex)0.0f;
                    break;
                }
            case ME_COMPLEX128:
                {
                    ((double _Complex*)write_ptr)[0] = (double _Complex)0.0;
                    break;
                }
            default:
                break;
            }
        }
        else {
            switch (arg_type) {
            case ME_BOOL:
            case ME_INT8:
            case ME_INT16:
            case ME_INT32:
            case ME_INT64:
                {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        int64_t v = (int64_t)val;
                        for (int i = 0; i < nitems; i++) acc *= v;
                    }
                    else {
                        acc = (int64_t)val * (int64_t)nitems;
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_UINT8:
            case ME_UINT16:
            case ME_UINT32:
            case ME_UINT64:
                {
                    uint64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        uint64_t v = (uint64_t)val;
                        for (int i = 0; i < nitems; i++) acc *= v;
                    }
                    else {
                        acc = (uint64_t)val * (uint64_t)nitems;
                    }
                    ((uint64_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_FLOAT32:
                {
                    float acc = is_prod ? 1.0f : 0.0f;
                    if (nitems == 0) {
                        acc = is_prod ? 1.0f : 0.0f;
                    }
                    else if (is_prod) {
                        float v = (float)val;
                        for (int i = 0; i < nitems; i++) acc *= v;
                    }
                    else {
                        acc = (float)val * (float)nitems;
                    }
                    ((float*)write_ptr)[0] = acc;
                    break;
                }
            case ME_FLOAT64:
                {
                    double acc = is_prod ? 1.0 : 0.0;
                    if (nitems == 0) {
                        acc = is_prod ? 1.0 : 0.0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= val;
                    }
                    else {
                        acc = val * (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                    break;
                }
            case ME_COMPLEX64:
                {
                    float _Complex acc = is_prod ? (float _Complex)1.0f : (float _Complex)0.0f;
                    float _Complex v = (float _Complex)val;
                    if (nitems == 0) {
                        acc = is_prod ? (float _Complex)1.0f : (float _Complex)0.0f;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= v;
                    }
                    else {
                        acc = v * (float)nitems;
                    }
                    ((float _Complex*)write_ptr)[0] = acc;
                    break;
                }
            case ME_COMPLEX128:
                {
                    double _Complex acc = is_prod ? (double _Complex)1.0 : (double _Complex)0.0;
                    double _Complex v = (double _Complex)val;
                    if (nitems == 0) {
                        acc = is_prod ? (double _Complex)1.0 : (double _Complex)0.0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= v;
                    }
                    else {
                        acc = v * (double)nitems;
                    }
                    ((double _Complex*)write_ptr)[0] = acc;
                    break;
                }
            default:
                break;
            }
        }
    }
    else {
        const void* saved_bound = arg->bound;
        int saved_type = arg->type;
        if (arg->type != ME_VARIABLE) {
            ((me_expr*)arg)->bound = arg->output;
            ((me_expr*)arg)->type = ME_VARIABLE;
        }
        switch (arg_type) {
        case ME_BOOL:
            {
                const bool* data = (const bool*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i]) { acc = true; break; }
                            }
                            else {
                                if (!data[i]) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    bool acc = is_min;
                    if (nitems > 0) {
                        acc = data[0];
                        for (int i = 1; i < nitems; i++) {
                            acc = is_min ? (acc && data[i]) : (acc || data[i]);
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i] ? 1 : 0;
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i] ? 1 : 0;
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_INT8:
            {
                const int8_t* data = (const int8_t*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    int8_t acc = is_min ? reduce_min_int8(data, nitems) :
                        reduce_max_int8(data, nitems);
                    ((int8_t*)write_ptr)[0] = acc;
                }
                else {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_INT16:
            {
                const int16_t* data = (const int16_t*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    int16_t acc = is_min ? reduce_min_int16(data, nitems) :
                        reduce_max_int16(data, nitems);
                    ((int16_t*)write_ptr)[0] = acc;
                }
                else {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_INT32:
            {
                const int32_t* data = (const int32_t*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    int32_t acc = is_min ? reduce_min_int32(data, nitems) :
                        reduce_max_int32(data, nitems);
                    ((int32_t*)write_ptr)[0] = acc;
                }
                else {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_INT64:
            {
                const int64_t* data = (const int64_t*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    int64_t acc = is_min ? reduce_min_int64(data, nitems) :
                        reduce_max_int64(data, nitems);
                    ((int64_t*)write_ptr)[0] = acc;
                }
                else {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_UINT8:
            {
                const uint8_t* data = (const uint8_t*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    uint8_t acc = is_min ? reduce_min_uint8(data, nitems) :
                        reduce_max_uint8(data, nitems);
                    ((uint8_t*)write_ptr)[0] = acc;
                }
                else {
                    uint64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((uint64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_UINT16:
            {
                const uint16_t* data = (const uint16_t*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    uint16_t acc = is_min ? reduce_min_uint16(data, nitems) :
                        reduce_max_uint16(data, nitems);
                    ((uint16_t*)write_ptr)[0] = acc;
                }
                else {
                    uint64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((uint64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_UINT32:
            {
                const uint32_t* data = (const uint32_t*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    uint32_t acc = is_min ? reduce_min_uint32(data, nitems) :
                        reduce_max_uint32(data, nitems);
                    ((uint32_t*)write_ptr)[0] = acc;
                }
                else {
                    uint64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((uint64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_UINT64:
            {
                const uint64_t* data = (const uint64_t*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    uint64_t acc = is_min ? reduce_min_uint64(data, nitems) :
                        reduce_max_uint64(data, nitems);
                    ((uint64_t*)write_ptr)[0] = acc;
                }
                else {
                    uint64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((uint64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_FLOAT32:
            {
                const float* data = (const float*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0.0f) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0.0f) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else {
                    if (nitems == 0) {
                        float acc = 0.0f;
                        if (is_min) acc = INFINITY;
                        else if (is_max) acc = -INFINITY;
                        else acc = is_prod ? 1.0f : 0.0f;
                        ((float*)write_ptr)[0] = acc;
                    }
                    else if (is_min) {
                        float acc = reduce_min_float32_nan_safe(data, nitems);
                        ((float*)write_ptr)[0] = acc;
                    }
                    else if (is_max) {
                        float acc = reduce_max_float32_nan_safe(data, nitems);
                        ((float*)write_ptr)[0] = acc;
                    }
                    else if (is_prod) {
                        /* Accumulate float32 sum/prod in float64 for better precision. */
                        double acc = reduce_prod_float32_nan_safe(data, nitems);
                        ((float*)write_ptr)[0] = (float)acc;
                    }
                    else {
                        double acc = reduce_sum_float32_nan_safe(data, nitems);
                        ((float*)write_ptr)[0] = (float)acc;
                    }
                }
                break;
            }
        case ME_FLOAT64:
            {
                const double* data = (const double*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0.0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0.0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else {
                    double acc = 0.0;
                    if (nitems == 0) {
                        if (is_min) acc = INFINITY;
                        else if (is_max) acc = -INFINITY;
                        else acc = is_prod ? 1.0 : 0.0;
                    }
                    else if (is_min) {
                        acc = reduce_min_float64_nan_safe(data, nitems);
                    }
                    else if (is_max) {
                        acc = reduce_max_float64_nan_safe(data, nitems);
                    }
                    else if (is_prod) {
                        acc = reduce_prod_float64_nan_safe(data, nitems);
                    }
                    else {
                        acc = reduce_sum_float64_nan_safe(data, nitems);
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_COMPLEX64:
            {
                const float _Complex* data = (const float _Complex*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            bool nonzero = IS_NONZERO_c64(data[i]);
                            if (is_any) {
                                if (nonzero) { acc = true; break; }
                            }
                            else {
                                if (!nonzero) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                    break;
                }
                if (is_min || is_max) {
                    ((float _Complex*)write_ptr)[0] = (float _Complex)0.0f;
                    break;
                }
                float _Complex acc = is_prod ? (float _Complex)1.0f : (float _Complex)0.0f;
                if (nitems == 0) {
                    acc = is_prod ? (float _Complex)1.0f : (float _Complex)0.0f;
                }
                else if (is_prod) {
                    for (int i = 0; i < nitems; i++) acc *= data[i];
                }
                else {
                    for (int i = 0; i < nitems; i++) acc += data[i];
                }
                ((float _Complex*)write_ptr)[0] = acc;
                break;
            }
        case ME_COMPLEX128:
            {
                const double _Complex* data = (const double _Complex*)arg->bound;
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            bool nonzero = IS_NONZERO_c128(data[i]);
                            if (is_any) {
                                if (nonzero) { acc = true; break; }
                            }
                            else {
                                if (!nonzero) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                    break;
                }
                if (is_min || is_max) {
                    ((double _Complex*)write_ptr)[0] = (double _Complex)0.0;
                    break;
                }
                double _Complex acc = is_prod ? (double _Complex)1.0 : (double _Complex)0.0;
                if (nitems == 0) {
                    acc = is_prod ? (double _Complex)1.0 : (double _Complex)0.0;
                }
                else if (is_prod) {
                    for (int i = 0; i < nitems; i++) acc *= data[i];
                }
                else {
                    for (int i = 0; i < nitems; i++) acc += data[i];
                }
                ((double _Complex*)write_ptr)[0] = acc;
                break;
            }
        default:
            break;
        }
        if (saved_type != ME_VARIABLE) {
            ((me_expr*)arg)->bound = saved_bound;
            ((me_expr*)arg)->type = saved_type;
        }
    }

    {
        me_dtype write_type = temp_output ? result_type : output_type;
        broadcast_reduction_output(write_ptr, write_type, output_nitems);
    }

    if (temp_output) {
        convert_func_t conv = get_convert_func(result_type, output_type);
        if (conv) {
            conv(temp_output, n->output, output_nitems);
        }
        free(temp_output);
    }
}

static void private_eval(const me_expr* n) {
    if (!n) return;

    if (is_reduction_node(n)) {
        eval_reduction(n, 1);
        return;
    }

    // Special case: imag() and real() functions return real from complex input
    if (IS_FUNCTION(n->type) && ARITY(n->type) == 1) {
        if (n->function == (void*)imag_wrapper || n->function == (void*)real_wrapper) {
            me_expr* arg = (me_expr*)n->parameters[0];
            me_dtype arg_type = infer_result_type(arg);

            if (arg_type == ME_COMPLEX64) {
                // Evaluate argument as complex64
                if (!arg->output) {
                    arg->output = malloc(n->nitems * sizeof(float _Complex));
                    arg->nitems = n->nitems;
                    ((me_expr*)arg)->dtype = ME_COMPLEX64;
                }
                me_eval_c64(arg);

                // Extract real/imaginary part to float32 output
                const float _Complex* cdata = (const float _Complex*)arg->output;
                float* output = (float*)n->output;
                if (n->function == (void*)imag_wrapper) {
                    for (int i = 0; i < n->nitems; i++) {
#if defined(_MSC_VER) && defined(__clang__)
                        output[i] = __builtin_cimagf(cdata[i]);
#else
                        output[i] = cimagf(cdata[i]);
#endif
                    }
                }
                else { // real_wrapper
                    for (int i = 0; i < n->nitems; i++) {
#if defined(_MSC_VER) && defined(__clang__)
                        output[i] = __builtin_crealf(cdata[i]);
#else
                        output[i] = crealf(cdata[i]);
#endif
                    }
                }
                return;
            }
            else if (arg_type == ME_COMPLEX128) {
                // Evaluate argument as complex128
                if (!arg->output) {
                    arg->output = malloc(n->nitems * sizeof(double _Complex));
                    arg->nitems = n->nitems;
                    ((me_expr*)arg)->dtype = ME_COMPLEX128;
                }
                me_eval_c128(arg);

                // Extract real/imaginary part to float64 output
                const double _Complex* cdata = (const double _Complex*)arg->output;
                double* output = (double*)n->output;
                if (n->function == (void*)imag_wrapper) {
                    for (int i = 0; i < n->nitems; i++) {
#if defined(_MSC_VER) && defined(__clang__)
                        output[i] = __builtin_cimag(cdata[i]);
#else
                        output[i] = cimag(cdata[i]);
#endif
                    }
                }
                else { // real_wrapper
                    for (int i = 0; i < n->nitems; i++) {
#if defined(_MSC_VER) && defined(__clang__)
                        output[i] = __builtin_creal(cdata[i]);
#else
                        output[i] = creal(cdata[i]);
#endif
                    }
                }
                return;
            }
            // If not complex, fall through to normal evaluation
        }
    }

    // Infer the result type from the expression tree
    me_dtype result_type = infer_result_type(n);

    // If all variables already match result type, use fast path
    bool all_match = all_variables_match_type(n, result_type);
    if (result_type == n->dtype && all_match) {
        // Fast path: no promotion needed
        if (n->dtype == ME_AUTO) {
            fprintf(stderr, "FATAL: ME_AUTO dtype in evaluation. This is a bug.\n");
#ifdef NDEBUG
            abort(); // Release build: terminate immediately
#else
            assert(0 && "ME_AUTO should be resolved during compilation"); // Debug: trigger debugger
#endif
        }
        switch (n->dtype) {
        case ME_BOOL: me_eval_i8(n);
            break;
        case ME_INT8: me_eval_i8(n);
            break;
        case ME_INT16: me_eval_i16(n);
            break;
        case ME_INT32: me_eval_i32(n);
            break;
        case ME_INT64: me_eval_i64(n);
            break;
        case ME_UINT8: me_eval_u8(n);
            break;
        case ME_UINT16: me_eval_u16(n);
            break;
        case ME_UINT32: me_eval_u32(n);
            break;
        case ME_UINT64: me_eval_u64(n);
            break;
        case ME_FLOAT32: me_eval_f32(n);
            break;
        case ME_FLOAT64: me_eval_f64(n);
            break;
        case ME_COMPLEX64: me_eval_c64(n);
            break;
        case ME_COMPLEX128: me_eval_c128(n);
            break;
        default:
            fprintf(stderr, "FATAL: Invalid dtype %d in evaluation.\n", n->dtype);
#ifdef NDEBUG
            abort(); // Release build: terminate immediately
#else
            assert(0 && "Invalid dtype"); // Debug: trigger debugger
#endif
        }
        return;
    }

    // Slow path: need to promote variables
    // Allocate tracking structures (max ME_MAX_VARS variables)
    promoted_var_t promotions[ME_MAX_VARS];
    int promo_count = 0;

    // Save original variable bindings
    const void* original_bounds[ME_MAX_VARS];
    me_dtype original_types[ME_MAX_VARS];
    int save_idx = 0;

    save_variable_bindings(n, original_bounds, original_types, &save_idx);

    // Promote variables
    promote_variables_in_tree((me_expr*)n, result_type, promotions, &promo_count, n->nitems);

    // Check if we need output type conversion (e.g., computation in float64, output in bool)
    me_dtype saved_dtype = n->dtype;
    void* original_output = n->output;
    void* temp_output = NULL;

    if (saved_dtype != result_type) {
        // Allocate temp buffer for computation
        temp_output = malloc(n->nitems * dtype_size(result_type));
        if (temp_output) {
            ((me_expr*)n)->output = temp_output;
        }
    }

    // Update expression type for evaluation
    ((me_expr*)n)->dtype = result_type;

    // Evaluate with promoted types
    if (result_type == ME_AUTO) {
        fprintf(stderr, "FATAL: ME_AUTO result type in evaluation. This is a bug.\n");
#ifdef NDEBUG
        abort(); // Release build: terminate immediately
#else
        assert(0 && "ME_AUTO should be resolved during compilation"); // Debug: trigger debugger
#endif
    }
    switch (result_type) {
    case ME_BOOL: me_eval_i8(n);
        break;
    case ME_INT8: me_eval_i8(n);
        break;
    case ME_INT16: me_eval_i16(n);
        break;
    case ME_INT32: me_eval_i32(n);
        break;
    case ME_INT64: me_eval_i64(n);
        break;
    case ME_UINT8: me_eval_u8(n);
        break;
    case ME_UINT16: me_eval_u16(n);
        break;
    case ME_UINT32: me_eval_u32(n);
        break;
    case ME_UINT64: me_eval_u64(n);
        break;
    case ME_FLOAT32: me_eval_f32(n);
        break;
    case ME_FLOAT64: me_eval_f64(n);
        break;
    case ME_COMPLEX64: me_eval_c64(n);
        break;
    case ME_COMPLEX128: me_eval_c128(n);
        break;
    default:
        fprintf(stderr, "FATAL: Invalid result type %d in evaluation.\n", result_type);
#ifdef NDEBUG
        abort(); // Release build: terminate immediately
#else
        assert(0 && "Invalid dtype"); // Debug: trigger debugger
#endif
    }

    // If we used a temp buffer, convert to final output type
    if (temp_output) {
        convert_func_t conv = get_convert_func(result_type, saved_dtype);
        if (conv) {
            conv(temp_output, original_output, n->nitems);
        }
        // Restore original output pointer
        ((me_expr*)n)->output = original_output;
        free(temp_output);
    }

    // Restore original variable bindings
    int restore_idx = 0;
    restore_variables_in_tree((me_expr*)n, original_bounds, original_types, &restore_idx);

    // Restore expression type
    ((me_expr*)n)->dtype = saved_dtype;

    // Free promoted buffers
    for (int i = 0; i < promo_count; i++) {
        if (promotions[i].needs_free) {
            free(promotions[i].promoted_data);
        }
    }
}

/* Helper to update variable bindings and nitems in tree */
static void save_nitems_in_tree(const me_expr* node, int* nitems_array, int* idx) {
    if (!node) return;
    nitems_array[(*idx)++] = node->nitems;

    switch (TYPE_MASK(node->type)) {
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                save_nitems_in_tree((const me_expr*)node->parameters[i], nitems_array, idx);
            }
            break;
        }
    default:
        break;
    }
}

static void restore_nitems_in_tree(me_expr* node, const int* nitems_array, int* idx) {
    if (!node) return;
    node->nitems = nitems_array[(*idx)++];

    switch (TYPE_MASK(node->type)) {
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                restore_nitems_in_tree((me_expr*)node->parameters[i], nitems_array, idx);
            }
            break;
        }
    default:
        break;
    }
}

/* Helper to free intermediate output buffers */
static void free_intermediate_buffers(me_expr* node) {
    if (!node) return;

    switch (TYPE_MASK(node->type)) {
    case ME_CONSTANT:
    case ME_VARIABLE:
        // These don't have intermediate buffers
        break;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                me_expr* param = (me_expr*)node->parameters[i];
                free_intermediate_buffers(param);

                // Free intermediate buffer (but not for root or variables/constants)
                if (param->type != ME_CONSTANT && param->type != ME_VARIABLE && param->output) {
                    free(param->output);
                    param->output = NULL;
                }
            }
            break;
        }
    }
}

/* Helper to save original variable bindings with their pointers */
static void save_variable_metadata(const me_expr* node, const void** var_pointers, size_t* var_sizes, int* var_count) {
    if (!node) return;
    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        // Check if this pointer is already in the list
        for (int i = 0; i < *var_count; i++) {
            if (var_pointers[i] == node->bound) return; // Already saved
        }
        var_pointers[*var_count] = node->bound;
        var_sizes[*var_count] = dtype_size(node->input_dtype);
        (*var_count)++;
        break;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                save_variable_metadata((const me_expr*)node->parameters[i], var_pointers, var_sizes, var_count);
            }
            break;
        }
    }
}

static int count_variable_nodes(const me_expr* node) {
    if (!node) return 0;
    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        return 1;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            int count = 0;
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                count += count_variable_nodes((const me_expr*)node->parameters[i]);
            }
            return count;
        }
    }
    return 0;
}

static void collect_variable_nodes(me_expr* node, const void** var_pointers, int n_vars,
                                   me_expr** var_nodes, int* var_indices, int* node_count) {
    if (!node) return;
    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        {
            int idx = -1;
            for (int i = 0; i < n_vars; i++) {
                if (node->bound == var_pointers[i]) {
                    idx = i;
                    break;
                }
            }
            if (idx >= 0) {
                var_nodes[*node_count] = node;
                var_indices[*node_count] = idx;
                (*node_count)++;
            }
            break;
        }
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                collect_variable_nodes((me_expr*)node->parameters[i], var_pointers, n_vars,
                                       var_nodes, var_indices, node_count);
            }
            break;
        }
    }
}

/* Helper to update variable bindings by matching original pointers */
static void update_vars_by_pointer(me_expr* node, const void** old_pointers, const void** new_pointers, int n_vars) {
    if (!node) return;
    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        // Find which variable this is and update to new pointer
        for (int i = 0; i < n_vars; i++) {
            if (node->bound == old_pointers[i]) {
                node->bound = new_pointers[i];
                break;
            }
        }
        break;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                update_vars_by_pointer((me_expr*)node->parameters[i], old_pointers, new_pointers, n_vars);
            }
            break;
        }
    }
}

/* Helper to update variable bindings and nitems in tree */
static void update_variable_bindings(me_expr* node, const void** new_bounds, int* var_idx, int new_nitems) {
    if (!node) return;

    // Update nitems for all nodes to handle intermediate buffers
    if (new_nitems > 0) {
        node->nitems = new_nitems;
    }

    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        if (new_bounds && *var_idx >= 0) {
            node->bound = new_bounds[*var_idx];
            (*var_idx)++;
        }
        break;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                update_variable_bindings((me_expr*)node->parameters[i], new_bounds, var_idx, new_nitems);
            }
            break;
        }
    }
}

/* Evaluate compiled expression with new variable and output pointers */
static me_expr* clone_expr(const me_expr* src) {
    if (!src) return NULL;

    const int arity = ARITY(src->type);
    const int psize = sizeof(void*) * arity;
    const int size = (sizeof(me_expr) - sizeof(void*)) + psize + (IS_CLOSURE(src->type) ? sizeof(void*) : 0);
    me_expr* clone = malloc(size);
    if (!clone) return NULL;

    // Copy the entire structure
    memcpy(clone, src, size);

    // Clone children recursively
    if (arity > 0) {
        for (int i = 0; i < arity; i++) {
            clone->parameters[i] = clone_expr((const me_expr*)src->parameters[i]);
            if (src->parameters[i] && !clone->parameters[i]) {
                // Clone failed, clean up
                for (int j = 0; j < i; j++) {
                    me_free((me_expr*)clone->parameters[j]);
                }
                free(clone);
                return NULL;
            }
        }
    }

    // Don't clone output buffer - it will be set by caller
    // Don't clone bytecode - not needed for clones
    clone->output = NULL;
    clone->bytecode = NULL;
    clone->ncode = 0;

    return clone;
}

/* Thread-safe chunked evaluation using expression cloning.
 * This function is safe to call from multiple threads simultaneously,
 * even on the same expression object. Each call creates a temporary
 * clone of the expression tree to avoid race conditions. */
int me_eval(const me_expr* expr, const void** vars_chunk,
            int n_vars, void* output_chunk, int chunk_nitems) {
    if (!expr) return ME_EVAL_ERR_NULL_EXPR;

    // Verify variable count matches
    const void* original_var_pointers[ME_MAX_VARS];
    size_t var_sizes[ME_MAX_VARS];
    int actual_var_count = 0;
    save_variable_metadata(expr, original_var_pointers, var_sizes, &actual_var_count);
    if (actual_var_count > ME_MAX_VARS) {
        fprintf(stderr, "Error: Expression uses %d variables, exceeds ME_MAX_VARS=%d\n",
                actual_var_count, ME_MAX_VARS);
        return ME_EVAL_ERR_TOO_MANY_VARS;
    }

    if (actual_var_count != n_vars) {
        return ME_EVAL_ERR_VAR_MISMATCH;
    }

    // Clone the expression tree
    me_expr* clone = clone_expr(expr);
    if (!clone) return ME_EVAL_ERR_OOM;

    const int block_nitems = ME_EVAL_BLOCK_NITEMS;
    int status = ME_EVAL_SUCCESS;

    if (!ME_EVAL_ENABLE_BLOCKING || chunk_nitems <= block_nitems) {
        // Update clone's variable bindings
        update_vars_by_pointer(clone, original_var_pointers, vars_chunk, n_vars);

        // Update clone's nitems throughout the tree
        int update_idx = 0;
        update_variable_bindings(clone, NULL, &update_idx, chunk_nitems);

        // Set output pointer
        clone->output = output_chunk;

        // Evaluate the clone
        private_eval(clone);
    }
    else if (is_reduction_node(clone)) {
        // Reductions operate on the full chunk; avoid block processing.
        update_vars_by_pointer(clone, original_var_pointers, vars_chunk, n_vars);

        int update_idx = 0;
        update_variable_bindings(clone, NULL, &update_idx, chunk_nitems);

        clone->output = output_chunk;
        private_eval(clone);
    }
    else {
        const size_t output_item_size = dtype_size(clone->dtype);
        const int max_var_nodes = count_variable_nodes(clone);
        me_expr** var_nodes = NULL;
        int* var_indices = NULL;
        int var_node_count = 0;

        if (max_var_nodes > 0) {
            var_nodes = malloc((size_t)max_var_nodes * sizeof(*var_nodes));
            var_indices = malloc((size_t)max_var_nodes * sizeof(*var_indices));
            if (!var_nodes || !var_indices) {
                free(var_nodes);
                free(var_indices);
                status = ME_EVAL_ERR_OOM;
                goto cleanup;
            }
            collect_variable_nodes(clone, original_var_pointers, n_vars,
                                   var_nodes, var_indices, &var_node_count);
        }

#if defined(__clang__)
#pragma clang loop unroll_count(4)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 4
#endif
        for (int offset = 0; offset < chunk_nitems; offset += block_nitems) {
            int current = block_nitems;
            if (offset + current > chunk_nitems) {
                current = chunk_nitems - offset;
            }

            const void* block_vars[ME_MAX_VARS];
            for (int i = 0; i < n_vars; i++) {
                const unsigned char* base = (const unsigned char*)vars_chunk[i];
                block_vars[i] = base + (size_t)offset * var_sizes[i];
            }

            for (int i = 0; i < var_node_count; i++) {
                var_nodes[i]->bound = block_vars[var_indices[i]];
            }

            int update_idx = 0;
            update_variable_bindings(clone, NULL, &update_idx, current);

            clone->output = (unsigned char*)output_chunk + (size_t)offset * output_item_size;
            private_eval(clone);
        }

        free(var_nodes);
        free(var_indices);
    }

cleanup:
    // Free the clone (including any intermediate buffers it allocated)
    me_free(clone);
    return status;
}


static void optimize(me_expr* n) {
    /* Evaluates as much as possible. */
    if (!n) return;
    if (n->type == ME_CONSTANT) return;
    if (n->type == ME_VARIABLE) return;

    /* Only optimize out functions flagged as pure. */
    if (IS_PURE(n->type)) {
        const int arity = ARITY(n->type);
        int known = 1;
        int i;
        for (i = 0; i < arity; ++i) {
            optimize(n->parameters[i]);
            if (((me_expr*)(n->parameters[i]))->type != ME_CONSTANT) {
                known = 0;
            }
        }
        if (known) {
            const double value = me_eval_scalar(n);
            me_free_parameters(n);
            n->type = ME_CONSTANT;
            n->value = value;
        }
    }
}

#if defined(_WIN32) || defined(_WIN64)
static bool has_complex_node(const me_expr* n) {
    if (!n) return false;
    if (n->dtype == ME_COMPLEX64 || n->dtype == ME_COMPLEX128) return true;
    const int arity = ARITY(n->type);
    for (int i = 0; i < arity; i++) {
        if (has_complex_node((const me_expr*)n->parameters[i])) return true;
    }
    return false;
}

static bool has_complex_input(const me_expr* n) {
    if (!n) return false;
    if (n->input_dtype == ME_COMPLEX64 || n->input_dtype == ME_COMPLEX128) return true;
    const int arity = ARITY(n->type);
    for (int i = 0; i < arity; i++) {
        if (has_complex_input((const me_expr*)n->parameters[i])) return true;
    }
    return false;
}
#endif


static int private_compile(const char* expression, const me_variable* variables, int var_count,
                           void* output, int nitems, me_dtype dtype, int* error, me_expr** out) {
    if (out) *out = NULL;
    if (!expression || !out || var_count < 0) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }

    // Validate dtype usage: either all vars are ME_AUTO (use dtype), or dtype is ME_AUTO (use var dtypes)
    if (variables && var_count > 0) {
        int auto_count = 0;
        int specified_count = 0;

        for (int i = 0; i < var_count; i++) {
            if (variables[i].dtype == ME_AUTO) {
                auto_count++;
            }
            else {
                specified_count++;
            }
        }

        // Check the two valid modes
        if (dtype == ME_AUTO) {
            // Mode 1: Output dtype is ME_AUTO, all variables must have explicit dtypes
            if (auto_count > 0) {
                fprintf(
                    stderr,
                    "Error: When output dtype is ME_AUTO, all variable dtypes must be specified (not ME_AUTO)\n");
                if (error) *error = -1;
                return ME_COMPILE_ERR_VAR_UNSPECIFIED;
            }
        }
        else {
            // Mode 2: Output dtype is specified
            // Two sub-modes: all ME_AUTO (homogeneous), or all explicit (heterogeneous with conversion)
            if (auto_count > 0 && specified_count > 0) {
                // Mixed mode not allowed
                fprintf(stderr, "Error: Variable dtypes must be all ME_AUTO or all explicitly specified\n");
                if (error) *error = -1;
                return ME_COMPILE_ERR_VAR_MIXED;
            }
        }
    }

    // Create a copy of variables with dtype filled in (if not already set)
    me_variable* vars_copy = NULL;
    if (variables && var_count > 0) {
        vars_copy = malloc(var_count * sizeof(me_variable));
        if (!vars_copy) {
            if (error) *error = -1;
            return ME_COMPILE_ERR_OOM;
        }
        for (int i = 0; i < var_count; i++) {
            vars_copy[i] = variables[i];
            // If dtype not set (ME_AUTO), use the provided dtype
            if (vars_copy[i].dtype == ME_AUTO && vars_copy[i].type == 0) {
                vars_copy[i].dtype = dtype;
                vars_copy[i].type = ME_VARIABLE;
            }
        }
    }

    state s;
    s.start = s.next = expression;
    s.lookup = vars_copy ? vars_copy : variables;
    s.lookup_len = var_count;
    // When dtype is ME_AUTO, infer target dtype from variables to avoid type mismatch
    if (dtype != ME_AUTO) {
        s.target_dtype = dtype;
    }
    else if (variables && var_count > 0) {
        // Use the first variable's dtype as the target for constants
        // This prevents type promotion issues when mixing float32 vars with float64 constants
        s.target_dtype = variables[0].dtype;
    }
    else {
        s.target_dtype = ME_AUTO;
    }

    next_token(&s);
    me_expr* root = list(&s);

    if (root == NULL) {
        if (error) *error = -1;
        if (vars_copy) free(vars_copy);
        return ME_COMPILE_ERR_OOM;
    }

    if (contains_reduction(root) && !reduction_usage_is_valid(root)) {
        me_free(root);
        if (error) *error = -1;
        if (vars_copy) free(vars_copy);
        return ME_COMPILE_ERR_REDUCTION_INVALID;
    }

#if defined(_WIN32) || defined(_WIN64)
    {
        const me_variable* vars_check = vars_copy ? vars_copy : variables;
        bool complex_vars = false;
        if (vars_check) {
            for (int i = 0; i < var_count; i++) {
                if (vars_check[i].dtype == ME_COMPLEX64 || vars_check[i].dtype == ME_COMPLEX128) {
                    complex_vars = true;
                    break;
                }
            }
        }
        if (complex_vars ||
            dtype == ME_COMPLEX64 || dtype == ME_COMPLEX128 ||
            has_complex_node(root) || has_complex_input(root)) {
            fprintf(stderr, "Error: Complex expressions are not supported on Windows (no C99 complex ABI)\n");
            me_free(root);
            if (error) *error = -1;
            if (vars_copy) free(vars_copy);
            return ME_COMPILE_ERR_COMPLEX_UNSUPPORTED;
        }
    }
#endif

    if (s.type != TOK_END) {
        me_free(root);
        if (error) {
            *error = (s.next - s.start);
            if (*error == 0) *error = 1;
        }
        if (vars_copy) free(vars_copy);
        return ME_COMPILE_ERR_PARSE;
    }
    else {
        optimize(root);
        root->output = output;
        root->nitems = nitems;

        // If dtype is ME_AUTO, infer from expression; otherwise use provided dtype
        if (dtype == ME_AUTO) {
            root->dtype = infer_output_type(root);
        }
        else {
            // User explicitly requested a dtype - use it (will cast if needed)
            root->dtype = dtype;
        }

        if (error) *error = 0;
        if (vars_copy) free(vars_copy);
        *out = root;
        return ME_COMPILE_SUCCESS;
    }
}

// Synthetic addresses for ordinal matching (when user provides NULL addresses)
static char synthetic_var_addresses[ME_MAX_VARS];

int me_compile(const char* expression, const me_variable* variables,
               int var_count, me_dtype dtype, int* error, me_expr** out) {
    if (out) *out = NULL;
    if (!out) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }

    // For chunked evaluation, we compile without specific output/nitems
    // If variables have NULL addresses, assign synthetic unique addresses for ordinal matching
    me_variable* vars_copy = NULL;
    int needs_synthetic = 0;

    if (variables && var_count > 0) {
        // Check if any variables have NULL addresses
        for (int i = 0; i < var_count; i++) {
            if (variables[i].address == NULL) {
                needs_synthetic = 1;
                break;
            }
        }

        if (needs_synthetic) {
            // Create copy with synthetic addresses
            vars_copy = malloc(var_count * sizeof(me_variable));
            if (!vars_copy) {
                if (error) *error = -1;
                return ME_COMPILE_ERR_OOM;
            }

            for (int i = 0; i < var_count; i++) {
                vars_copy[i] = variables[i];
                if (vars_copy[i].address == NULL) {
                    // Use address in synthetic array (each index is unique)
                    vars_copy[i].address = &synthetic_var_addresses[i];
                }
            }

            int status = private_compile(expression, vars_copy, var_count, NULL, 0, dtype, error, out);
            free(vars_copy);
            return status;
        }
    }

    // No NULL addresses, use variables as-is
    return private_compile(expression, variables, var_count, NULL, 0, dtype, error, out);
}

static void pn(const me_expr* n, int depth) {
    int i, arity;
    printf("%*s", depth, "");

    if (!n) {
        printf("NULL\n");
        return;
    }

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT: printf("%f\n", n->value);
        break;
    case ME_VARIABLE: printf("bound %p\n", n->bound);
        break;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        arity = ARITY(n->type);
        printf("f%d", arity);
        for (i = 0; i < arity; i++) {
            printf(" %p", n->parameters[i]);
        }
        printf("\n");
        for (i = 0; i < arity; i++) {
            pn(n->parameters[i], depth + 1);
        }
        break;
    }
}

void me_print(const me_expr* n) {
    pn(n, 0);
}

me_dtype me_get_dtype(const me_expr* expr) {
    return expr ? expr->dtype : ME_AUTO;
}
