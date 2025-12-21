/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2021  Blosc Development Team <blosc@blosc.org>
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
#include <complex.h>
#include <assert.h>

#ifndef NAN
#define NAN (0.0/0.0)
#endif

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif


typedef double (*me_fun2)(double, double);

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
        const void *bound;
        const void *function;
    };

    /* Vector operation info */
    void *output; // Generic pointer (can be float* or double*)
    int nitems;
    me_dtype dtype; // Data type for this expression (result type after promotion)
    me_dtype input_dtype; // Original input type (for variables/constants)
    /* Bytecode info (for fused evaluation) */
    void *bytecode; // Pointer to compiled bytecode
    int ncode; // Number of instructions
    void *parameters[1]; // Must be last (flexible array member)
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
        case ME_COMPLEX64: return sizeof(float complex);
        case ME_COMPLEX128: return sizeof(double complex);
        default: return 0;
    }
}


enum { ME_CONSTANT = 1 };


typedef struct state {
    const char *start;
    const char *next;
    int type;

    union {
        double value;
        const double *bound;
        const void *function;
    };

    void *context;
    me_dtype dtype; // Type of current token
    me_dtype target_dtype; // Target dtype for the overall expression

    const me_variable *lookup;
    int lookup_len;
} state;


#define TYPE_MASK(TYPE) ((TYPE)&0x0000001F)

#define IS_PURE(TYPE) (((TYPE) & ME_FLAG_PURE) != 0)
#define IS_FUNCTION(TYPE) (((TYPE) & ME_FUNCTION0) != 0)
#define IS_CLOSURE(TYPE) (((TYPE) & ME_CLOSURE0) != 0)
#define ARITY(TYPE) ( ((TYPE) & (ME_FUNCTION0 | ME_CLOSURE0)) ? ((TYPE) & 0x00000007) : 0 )
#define NEW_EXPR(type, ...) new_expr((type), (const me_expr*[]){__VA_ARGS__})
#define CHECK_NULL(ptr, ...) if ((ptr) == NULL) { __VA_ARGS__; return NULL; }

/* Forward declaration */
static me_expr *new_expr(const int type, const me_expr *parameters[]);

/* Infer computation type from expression tree (for evaluation) */
static me_dtype infer_result_type(const me_expr *n) {
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
        case ME_CLOSURE7: {
            // For comparisons with ME_BOOL output, we still need to infer the
            // computation type from operands (e.g., float64 for float inputs).
            // Don't return ME_BOOL early - let the operand types determine
            // the computation type.

            const int arity = ARITY(n->type);
            me_dtype result = ME_BOOL;

            for (int i = 0; i < arity; i++) {
                me_dtype param_type = infer_result_type((const me_expr *) n->parameters[i]);
                result = promote_types(result, param_type);
            }

            return result;
        }
    }

    return ME_FLOAT64;
}

/* Infer logical output type from expression tree (for compilation with ME_AUTO) */
static me_dtype infer_output_type(const me_expr *n) {
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
        case ME_CLOSURE7: {
            // If this node is a comparison (dtype == ME_BOOL set during parsing),
            // the output type is ME_BOOL
            if (n->dtype == ME_BOOL) {
                return ME_BOOL;
            }

            // Otherwise, infer from operands
            const int arity = ARITY(n->type);
            me_dtype result = ME_BOOL;

            for (int i = 0; i < arity; i++) {
                me_dtype param_type = infer_output_type((const me_expr *) n->parameters[i]);
                result = promote_types(result, param_type);
            }

            return result;
        }
    }

    return ME_FLOAT64;
}

/* Apply type promotion to a binary operation node */
static me_expr *create_conversion_node(me_expr *source, me_dtype target_dtype) {
    /* Create a unary conversion node that converts source to target_dtype */
    me_expr *conv = NEW_EXPR(ME_FUNCTION1 | ME_FLAG_PURE, source);
    if (conv) {
        conv->function = NULL; // Mark as conversion
        conv->dtype = target_dtype;
        conv->input_dtype = source->dtype;
    }
    return conv;
}

static void apply_type_promotion(me_expr *node) {
    if (!node || ARITY(node->type) < 2) return;

    me_expr *left = (me_expr *) node->parameters[0];
    me_expr *right = (me_expr *) node->parameters[1];

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

static me_expr *new_expr(const int type, const me_expr *parameters[]) {
    const int arity = ARITY(type);
    const int psize = sizeof(void *) * arity;
    const int size = (sizeof(me_expr) - sizeof(void *)) + psize + (IS_CLOSURE(type) ? sizeof(void *) : 0);
    me_expr *ret = malloc(size);
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


void me_free_parameters(me_expr *n) {
    if (!n) return;
    switch (TYPE_MASK(n->type)) {
        case ME_FUNCTION7:
        case ME_CLOSURE7:
            if (n->parameters[6] && ((me_expr *) n->parameters[6])->output &&
                ((me_expr *) n->parameters[6])->output != n->output) {
                free(((me_expr *) n->parameters[6])->output);
            }
            me_free(n->parameters[6]);
        case ME_FUNCTION6:
        case ME_CLOSURE6:
            if (n->parameters[5] && ((me_expr *) n->parameters[5])->output &&
                ((me_expr *) n->parameters[5])->output != n->output) {
                free(((me_expr *) n->parameters[5])->output);
            }
            me_free(n->parameters[5]);
        case ME_FUNCTION5:
        case ME_CLOSURE5:
            if (n->parameters[4] && ((me_expr *) n->parameters[4])->output &&
                ((me_expr *) n->parameters[4])->output != n->output) {
                free(((me_expr *) n->parameters[4])->output);
            }
            me_free(n->parameters[4]);
        case ME_FUNCTION4:
        case ME_CLOSURE4:
            if (n->parameters[3] && ((me_expr *) n->parameters[3])->output &&
                ((me_expr *) n->parameters[3])->output != n->output) {
                free(((me_expr *) n->parameters[3])->output);
            }
            me_free(n->parameters[3]);
        case ME_FUNCTION3:
        case ME_CLOSURE3:
            if (n->parameters[2] && ((me_expr *) n->parameters[2])->output &&
                ((me_expr *) n->parameters[2])->output != n->output) {
                free(((me_expr *) n->parameters[2])->output);
            }
            me_free(n->parameters[2]);
        case ME_FUNCTION2:
        case ME_CLOSURE2:
            if (n->parameters[1] && ((me_expr *) n->parameters[1])->output &&
                ((me_expr *) n->parameters[1])->output != n->output) {
                free(((me_expr *) n->parameters[1])->output);
            }
            me_free(n->parameters[1]);
        case ME_FUNCTION1:
        case ME_CLOSURE1:
            if (n->parameters[0] && ((me_expr *) n->parameters[0])->output &&
                ((me_expr *) n->parameters[0])->output != n->output) {
                free(((me_expr *) n->parameters[0])->output);
            }
            me_free(n->parameters[0]);
    }
}


void me_free(me_expr *n) {
    if (!n) return;
    me_free_parameters(n);
    if (n->bytecode) {
        free(n->bytecode);
    }
    free(n);
}


static double pi(void) { return 3.14159265358979323846; }
static double e(void) { return 2.71828182845904523536; }

static double fac(double a) {
    /* simplest version of fac */
    if (a < 0.0)
        return NAN;
    if (a > UINT_MAX)
        return INFINITY;
    unsigned int ua = (unsigned int) (a);
    unsigned long int result = 1, i;
    for (i = 1; i <= ua; i++) {
        if (i > ULONG_MAX / result)
            return INFINITY;
        result *= i;
    }
    return (double) result;
}

static double ncr(double n, double r) {
    if (n < 0.0 || r < 0.0 || n < r) return NAN;
    if (n > UINT_MAX || r > UINT_MAX) return INFINITY;
    unsigned long int un = (unsigned int) (n), ur = (unsigned int) (r), i;
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
    {"arccos", 0, acos, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arcsin", 0, asin, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arctan", 0, atan, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arctan2", 0, atan2, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"asin", 0, asin, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"atan", 0, atan, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"atan2", 0, atan2, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"ceil", 0, ceil, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"cos", 0, cos, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"cosh", 0, cosh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"e", 0, e, ME_FUNCTION0 | ME_FLAG_PURE, 0},
    {"exp", 0, exp, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"fac", 0, fac, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"floor", 0, floor, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"ln", 0, log, ME_FUNCTION1 | ME_FLAG_PURE, 0},
#ifdef ME_NAT_LOG
    {"log", 0, log, ME_FUNCTION1 | ME_FLAG_PURE, 0},
#else
    {"log", 0, log10, ME_FUNCTION1 | ME_FLAG_PURE, 0},
#endif
    {"log10", 0, log10, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"ncr", 0, ncr, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"npr", 0, npr, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"pi", 0, pi, ME_FUNCTION0 | ME_FLAG_PURE, 0},
    {"pow", 0, pow, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"sin", 0, sin, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sinh", 0, sinh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sqrt", 0, sqrt, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"tan", 0, tan, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"tanh", 0, tanh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {0, 0, 0, 0, 0}
};

static const me_variable *find_builtin(const char *name, int len) {
    int imin = 0;
    int imax = sizeof(functions) / sizeof(me_variable) - 2;

    /*Binary search.*/
    while (imax >= imin) {
        const int i = (imin + ((imax - imin) / 2));
        int c = strncmp(name, functions[i].name, len);
        if (!c) c = '\0' - functions[i].name[len];
        if (c == 0) {
            return functions + i;
        } else if (c > 0) {
            imin = i + 1;
        } else {
            imax = i - 1;
        }
    }

    return 0;
}

static const me_variable *find_lookup(const state *s, const char *name, int len) {
    int iters;
    const me_variable *var;
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

static double comma(double a, double b) {
    (void) a;
    return b;
}

/* Bitwise operators (for integer types) */
static double bit_and(double a, double b) { return (double) ((int64_t) a & (int64_t) b); }
static double bit_or(double a, double b) { return (double) ((int64_t) a | (int64_t) b); }
static double bit_xor(double a, double b) { return (double) ((int64_t) a ^ (int64_t) b); }
static double bit_not(double a) { return (double) (~(int64_t) a); }
static double bit_shl(double a, double b) { return (double) ((int64_t) a << (int64_t) b); }
static double bit_shr(double a, double b) { return (double) ((int64_t) a >> (int64_t) b); }

/* Comparison operators (return 1.0 for true, 0.0 for false) */
static double cmp_eq(double a, double b) { return a == b ? 1.0 : 0.0; }
static double cmp_ne(double a, double b) { return a != b ? 1.0 : 0.0; }
static double cmp_lt(double a, double b) { return a < b ? 1.0 : 0.0; }
static double cmp_le(double a, double b) { return a <= b ? 1.0 : 0.0; }
static double cmp_gt(double a, double b) { return a > b ? 1.0 : 0.0; }
static double cmp_ge(double a, double b) { return a >= b ? 1.0 : 0.0; }

/* Logical operators (for bool type) - short-circuit via OR/AND */
static double logical_and(double a, double b) { return ((int) a) && ((int) b) ? 1.0 : 0.0; }
static double logical_or(double a, double b) { return ((int) a) || ((int) b) ? 1.0 : 0.0; }
static double logical_not(double a) { return !(int) a ? 1.0 : 0.0; }
static double logical_xor(double a, double b) { return ((int) a) != ((int) b) ? 1.0 : 0.0; }

static bool is_identifier_start(char c) {
    return isalpha((unsigned char) c) || c == '_';
}

static bool is_identifier_char(char c) {
    return isalnum((unsigned char) c) || c == '_';
}

static void skip_whitespace(state *s) {
    while (*s->next && isspace((unsigned char) *s->next)) {
        s->next++;
    }
}

static void read_number_token(state *s) {
    s->value = strtod(s->next, (char **) &s->next);
    s->type = TOK_NUMBER;
}

static void read_identifier_token(state *s) {
    const char *start = s->next;
    while (is_identifier_char(*s->next)) {
        s->next++;
    }

    const me_variable *var = find_lookup(s, start, s->next - start);
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
    const char *literal;
    int token_type;
    me_fun2 function;
} operator_spec;

static bool handle_multi_char_operator(state *s) {
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
        const operator_spec *op = &multi_ops[i];
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

static void handle_single_char_operator(state *s, char c) {
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

static void read_operator_token(state *s) {
    if (handle_multi_char_operator(s)) {
        return;
    }

    if (!*s->next) {
        s->type = TOK_END;
        return;
    }

    handle_single_char_operator(s, *s->next);
}

void next_token(state *s) {
    s->type = TOK_NULL;

    do {
        skip_whitespace(s);

        if (!*s->next) {
            s->type = TOK_END;
            return;
        }

        if ((s->next[0] >= '0' && s->next[0] <= '9') || s->next[0] == '.') {
            read_number_token(s);
        } else if (is_identifier_start(s->next[0])) {
            read_identifier_token(s);
        } else {
            read_operator_token(s);
        }
    } while (s->type == TOK_NULL);
}


static me_expr *list(state *s);

static me_expr *expr(state *s);

static me_expr *power(state *s);

static me_expr *shift_expr(state *s);

static me_expr *bitwise_and(state *s);

static me_expr *bitwise_xor(state *s);

static me_expr *bitwise_or(state *s);

static me_expr *comparison(state *s);


static me_expr *base(state *s) {
    /* <base>      =    <constant> | <variable> | <function-0> {"(" ")"} | <function-1> <power> | <function-X> "(" <expr> {"," <expr>} ")" | "(" <list> ")" */
    me_expr *ret;
    int arity;

    switch (TYPE_MASK(s->type)) {
        case TOK_NUMBER:
            ret = new_expr(ME_CONSTANT, 0);
            CHECK_NULL(ret);

            ret->value = s->value;
            ret->dtype = s->target_dtype; // Use target dtype for constants
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
                } else {
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
            } else {
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
                } else {
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
            } else {
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


static me_expr *power(state *s) {
    /* <power>     =    {("-" | "+")} <base> */
    int sign = 1;
    while (s->type == TOK_INFIX && (s->function == add || s->function == sub)) {
        if (s->function == sub) sign = -sign;
        next_token(s);
    }

    me_expr *ret;

    if (sign == 1) {
        ret = base(s);
    } else {
        me_expr *b = base(s);
        CHECK_NULL(b);

        ret = NEW_EXPR(ME_FUNCTION1 | ME_FLAG_PURE, b);
        CHECK_NULL(ret, me_free(b));

        ret->function = negate;
    }

    return ret;
}

#ifdef ME_POW_FROM_RIGHT
static me_expr *factor(state *s) {
    /* <factor>    =    <power> {"**" <factor>}  (right associative) */
    me_expr *ret = power(s);
    CHECK_NULL(ret);

    if (s->type == TOK_POW) {
        me_fun2 t = s->function;
        next_token(s);
        me_expr *f = factor(s); /* Right associative: recurse */
        CHECK_NULL(f, me_free(ret));

        me_expr *prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, f);
        CHECK_NULL(ret, me_free(f), me_free(prev));

        ret->function = t;
        apply_type_promotion(ret);
    }

    return ret;
}
#else
static me_expr *factor(state *s) {
    /* <factor>    =    <power> {"**" <power>}  (left associative) */
    me_expr *ret = power(s);
    CHECK_NULL(ret);

    while (s->type == TOK_POW) {
        me_fun2 t = s->function;
        next_token(s);
        me_expr *f = power(s);
        CHECK_NULL(f, me_free(ret));

        me_expr *prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, f);
        CHECK_NULL(ret, me_free(f), me_free(prev));

        ret->function = t;
        apply_type_promotion(ret);
    }

    return ret;
}
#endif


static me_expr *term(state *s) {
    /* <term>      =    <factor> {("*" | "/" | "%") <factor>} */
    me_expr *ret = factor(s);
    CHECK_NULL(ret);

    while (s->type == TOK_INFIX && (s->function == mul || s->function == divide || s->function == fmod)) {
        me_fun2 t = s->function;
        next_token(s);
        me_expr *f = factor(s);
        CHECK_NULL(f, me_free(ret));

        me_expr *prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, f);
        CHECK_NULL(ret, me_free(f), me_free(prev));

        ret->function = t;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr *expr(state *s) {
    /* <expr>      =    <term> {("+" | "-") <term>} */
    me_expr *ret = term(s);
    CHECK_NULL(ret);

    while (s->type == TOK_INFIX && (s->function == add || s->function == sub)) {
        me_fun2 t = s->function;
        next_token(s);
        me_expr *te = term(s);
        CHECK_NULL(te, me_free(ret));

        me_expr *prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, te);
        CHECK_NULL(ret, me_free(te), me_free(prev));

        ret->function = t;
        apply_type_promotion(ret); // Apply type promotion
    }

    return ret;
}


static me_expr *shift_expr(state *s) {
    /* <shift_expr> =    <expr> {("<<" | ">>") <expr>} */
    me_expr *ret = expr(s);
    CHECK_NULL(ret);

    while (s->type == TOK_SHIFT) {
        me_fun2 t = s->function;
        next_token(s);
        me_expr *e = expr(s);
        CHECK_NULL(e, me_free(ret));

        me_expr *prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = t;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr *bitwise_and(state *s) {
    /* <bitwise_and> =    <shift_expr> {"&" <shift_expr>} */
    me_expr *ret = shift_expr(s);
    CHECK_NULL(ret);

    while (s->type == TOK_BITWISE && s->function == bit_and) {
        next_token(s);
        me_expr *e = shift_expr(s);
        CHECK_NULL(e, me_free(ret));

        me_expr *prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = bit_and;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr *bitwise_xor(state *s) {
    /* <bitwise_xor> =    <bitwise_and> {"^" <bitwise_and>} */
    /* Note: ^ is XOR for integers/bools. Use ** for power */
    me_expr *ret = bitwise_and(s);
    CHECK_NULL(ret);

    while (s->type == TOK_BITWISE && s->function == bit_xor) {
        next_token(s);
        me_expr *e = bitwise_and(s);
        CHECK_NULL(e, me_free(ret));

        me_expr *prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = bit_xor;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr *bitwise_or(state *s) {
    /* <bitwise_or> =    <bitwise_xor> {"|" <bitwise_xor>} */
    me_expr *ret = bitwise_xor(s);
    CHECK_NULL(ret);

    while (s->type == TOK_BITWISE && (s->function == bit_or)) {
        me_fun2 t = s->function;
        next_token(s);
        me_expr *e = bitwise_xor(s);
        CHECK_NULL(e, me_free(ret));

        me_expr *prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = t;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr *comparison(state *s) {
    /* <comparison> =    <bitwise_or> {("<" | ">" | "<=" | ">=" | "==" | "!=") <bitwise_or>} */
    me_expr *ret = bitwise_or(s);
    CHECK_NULL(ret);

    while (s->type == TOK_COMPARE) {
        me_fun2 t = s->function;
        next_token(s);
        me_expr *e = bitwise_or(s);
        CHECK_NULL(e, me_free(ret));

        me_expr *prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = t;
        apply_type_promotion(ret);
        /* Comparisons always return bool */
        ret->dtype = ME_BOOL;
    }

    return ret;
}


static me_expr *list(state *s) {
    /* <list>      =    <comparison> {"," <comparison>} */
    me_expr *ret = comparison(s);
    CHECK_NULL(ret);

    while (s->type == TOK_SEP) {
        next_token(s);
        me_expr *e = comparison(s);
        CHECK_NULL(e, me_free(ret));

        me_expr *prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = comma;
        apply_type_promotion(ret);
    }

    return ret;
}


#define ME_FUN(...) ((double(*)(__VA_ARGS__))n->function)
#define M(e) me_eval_scalar(n->parameters[e])

static double me_eval_scalar(const me_expr *n) {
    if (!n) return NAN;

    switch (TYPE_MASK(n->type)) {
        case ME_CONSTANT: return n->value;
        case ME_VARIABLE: return *(const double *) n->bound;

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
static void vec_add(const double *a, const double *b, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static void vec_sub(const double *a, const double *b, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

static void vec_mul(const double *a, const double *b, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

static void vec_div(const double *a, const double *b, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] / b[i];
    }
}

static void vec_add_scalar(const double *a, double b, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b;
    }
}

static void vec_mul_scalar(const double *a, double b, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b;
    }
}

static void vec_pow(const double *a, const double *b, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = pow(a[i], b[i]);
    }
}

static void vec_pow_scalar(const double *a, double b, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = pow(a[i], b);
    }
}

static void vec_sqrt(const double *a, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sqrt(a[i]);
    }
}

static void vec_sin(const double *a, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sin(a[i]);
    }
}

static void vec_cos(const double *a, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = cos(a[i]);
    }
}

static void vec_negate(const double *a, double *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = -a[i];
    }
}

/* ============================================================================
 * FLOAT32 VECTOR OPERATIONS
 * ============================================================================ */

static void vec_add_f32(const float *a, const float *b, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static void vec_sub_f32(const float *a, const float *b, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

static void vec_mul_f32(const float *a, const float *b, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

static void vec_div_f32(const float *a, const float *b, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] / b[i];
    }
}

static void vec_add_scalar_f32(const float *a, float b, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b;
    }
}

static void vec_mul_scalar_f32(const float *a, float b, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b;
    }
}

static void vec_pow_f32(const float *a, const float *b, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = powf(a[i], b[i]);
    }
}

static void vec_pow_scalar_f32(const float *a, float b, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = powf(a[i], b);
    }
}

static void vec_sqrt_f32(const float *a, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sqrtf(a[i]);
    }
}

static void vec_sin_f32(const float *a, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sinf(a[i]);
    }
}

static void vec_cos_f32(const float *a, float *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = cosf(a[i]);
    }
}

static void vec_negame_f32(const float *a, float *out, int n) {
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
static void vec_and_bool(const bool *a, const bool *b, bool *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] && b[i];
}

static void vec_or_bool(const bool *a, const bool *b, bool *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] || b[i];
}

static void vec_xor_bool(const bool *a, const bool *b, bool *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] != b[i];
}

static void vec_not_bool(const bool *a, bool *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = !a[i];
}

/* Comparison operations - generate for all numeric types */
/* Note: These return bool arrays, but we'll store them as the same type for simplicity */
#define DEFINE_COMPARE_OPS(SUFFIX, TYPE) \
static void vec_cmp_eq_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (a[i] == b[i]) ? 1 : 0; \
} \
static void vec_cmp_ne_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (a[i] != b[i]) ? 1 : 0; \
} \
static void vec_cmp_lt_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (a[i] < b[i]) ? 1 : 0; \
} \
static void vec_cmp_le_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (a[i] <= b[i]) ? 1 : 0; \
} \
static void vec_cmp_gt_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (a[i] > b[i]) ? 1 : 0; \
} \
static void vec_cmp_ge_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
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
static void vec_add_c64(const float complex *a, const float complex *b, float complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static void vec_sub_c64(const float complex *a, const float complex *b, float complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] - b[i];
}

static void vec_mul_c64(const float complex *a, const float complex *b, float complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] * b[i];
}

static void vec_div_c64(const float complex *a, const float complex *b, float complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] / b[i];
}

static void vec_add_scalar_c64(const float complex *a, float complex b, float complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] + b;
}

static void vec_mul_scalar_c64(const float complex *a, float complex b, float complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] * b;
}

static void vec_pow_c64(const float complex *a, const float complex *b, float complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = cpowf(a[i], b[i]);
}

static void vec_pow_scalar_c64(const float complex *a, float complex b, float complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = cpowf(a[i], b);
}

static void vec_sqrt_c64(const float complex *a, float complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = csqrtf(a[i]);
}

static void vec_negame_c64(const float complex *a, float complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = -a[i];
}

static void vec_add_c128(const double complex *a, const double complex *b, double complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static void vec_sub_c128(const double complex *a, const double complex *b, double complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] - b[i];
}

static void vec_mul_c128(const double complex *a, const double complex *b, double complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] * b[i];
}

static void vec_div_c128(const double complex *a, const double complex *b, double complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] / b[i];
}

static void vec_add_scalar_c128(const double complex *a, double complex b, double complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] + b;
}

static void vec_mul_scalar_c128(const double complex *a, double complex b, double complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = a[i] * b;
}

static void vec_pow_c128(const double complex *a, const double complex *b, double complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = cpow(a[i], b[i]);
}

static void vec_pow_scalar_c128(const double complex *a, double complex b, double complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = cpow(a[i], b);
}

static void vec_sqrt_c128(const double complex *a, double complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = csqrt(a[i]);
}

static void vec_negame_c128(const double complex *a, double complex *out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) out[i] = -a[i];
}

/* ============================================================================
 * TYPE CONVERSION FUNCTIONS
 * ============================================================================
 * These functions convert between different data types for mixed-type expressions.
 */

#define DEFINE_VEC_CONVERT(FROM_SUFFIX, TO_SUFFIX, FROM_TYPE, TO_TYPE) \
static void vec_convert_##FROM_SUFFIX##_to_##TO_SUFFIX(const FROM_TYPE *in, TO_TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (TO_TYPE)in[i]; \
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
DEFINE_VEC_CONVERT(f32, c64, float, float complex)
DEFINE_VEC_CONVERT(f32, c128, float, double complex)

DEFINE_VEC_CONVERT(f64, c128, double, double complex)

DEFINE_VEC_CONVERT(c64, c128, float complex, double complex)

/* Function to get conversion function pointer */
typedef void (*convert_func_t)(const void *, void *, int);

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
    SQRT_FUNC, SIN_FUNC, COS_FUNC, EXP_FUNC, LOG_FUNC, FABS_FUNC, POW_FUNC) \
static void me_eval_##SUFFIX(const me_expr *n) { \
    if (!n || !n->output || n->nitems <= 0) return; \
    \
    int i, j; \
    const int arity = ARITY(n->type); \
    TYPE *output = (TYPE*)n->output; \
    \
    switch(TYPE_MASK(n->type)) { \
        case ME_CONSTANT: \
            { \
                TYPE val = (TYPE)n->value; \
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
                        VEC_ADD_SCALAR(ldata, (TYPE)right->value, output, n->nitems); \
                    } else if (left->type == ME_CONSTANT && rdata) { \
                        VEC_ADD_SCALAR(rdata, (TYPE)left->value, output, n->nitems); \
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
                        VEC_MUL_SCALAR(ldata, (TYPE)right->value, output, n->nitems); \
                    } else if (left->type == ME_CONSTANT && rdata) { \
                        VEC_MUL_SCALAR(rdata, (TYPE)left->value, output, n->nitems); \
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
                        VEC_POW_SCALAR(ldata, (TYPE)right->value, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else { \
                    general_case_binary_##SUFFIX: \
                    for (i = 0; i < n->nitems; i++) { \
                        double a = (left->type == ME_CONSTANT) ? left->value : \
                                  (left->type == ME_VARIABLE) ? (double)ldata[i] : (double)ldata[i]; \
                        double b = (right->type == ME_CONSTANT) ? right->value : \
                                  (right->type == ME_VARIABLE) ? (double)rdata[i] : (double)rdata[i]; \
                        output[i] = (TYPE)func(a, b); \
                    } \
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
                } else { \
                    me_fun1 func = (me_fun1)func_ptr; \
                    if (arg->type == ME_CONSTANT) { \
                        TYPE val = (TYPE)func(arg->value); \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = val; \
                        } \
                    } else { \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = (TYPE)func((double)adata[i]); \
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
                            args[j] = (double)pdata[i]; \
                        } \
                    } \
                    \
                    if (IS_FUNCTION(n->type)) { \
                        switch(arity) { \
                            case 0: output[i] = (TYPE)((double(*)(void))n->function)(); break; \
                            case 3: output[i] = (TYPE)((double(*)(double,double,double))n->function)(args[0], args[1], args[2]); break; \
                            case 4: output[i] = (TYPE)((double(*)(double,double,double,double))n->function)(args[0], args[1], args[2], args[3]); break; \
                            case 5: output[i] = (TYPE)((double(*)(double,double,double,double,double))n->function)(args[0], args[1], args[2], args[3], args[4]); break; \
                            case 6: output[i] = (TYPE)((double(*)(double,double,double,double,double,double))n->function)(args[0], args[1], args[2], args[3], args[4], args[5]); break; \
                            case 7: output[i] = (TYPE)((double(*)(double,double,double,double,double,double,double))n->function)(args[0], args[1], args[2], args[3], args[4], args[5], args[6]); break; \
                        } \
                    } else if (IS_CLOSURE(n->type)) { \
                        void *context = n->parameters[arity]; \
                        switch(arity) { \
                            case 0: output[i] = (TYPE)((double(*)(void*))n->function)(context); break; \
                            case 1: output[i] = (TYPE)((double(*)(void*,double))n->function)(context, args[0]); break; \
                            case 2: output[i] = (TYPE)((double(*)(void*,double,double))n->function)(context, args[0], args[1]); break; \
                            case 3: output[i] = (TYPE)((double(*)(void*,double,double,double))n->function)(context, args[0], args[1], args[2]); break; \
                            case 4: output[i] = (TYPE)((double(*)(void*,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3]); break; \
                            case 5: output[i] = (TYPE)((double(*)(void*,double,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3], args[4]); break; \
                            case 6: output[i] = (TYPE)((double(*)(void*,double,double,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3], args[4], args[5]); break; \
                            case 7: output[i] = (TYPE)((double(*)(void*,double,double,double,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3], args[4], args[5], args[6]); break; \
                        } \
                    } \
                } \
            } \
            break; \
        \
        default: \
            for (i = 0; i < n->nitems; i++) { \
                output[i] = (TYPE)NAN; \
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

#define vec_add_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpowf((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpowf((a)[_i], (b)); } while(0)
#define vec_sqrt_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = csqrtf((a)[_i]); } while(0)
#define vec_negame_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpow((a)[_i], (b)); } while(0)
#define vec_sqrt_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = csqrt((a)[_i]); } while(0)
#define vec_negame_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

/* Generate float32 evaluator */
DEFINE_ME_EVAL(f32, float,
               vec_add_f32, vec_sub_f32, vec_mul_f32, vec_div_f32, vec_pow_f32,
               vec_add_scalar_f32, vec_mul_scalar_f32, vec_pow_scalar_f32,
               vec_sqrt_f32, vec_sin_f32, vec_cos_f32, vec_negame_f32,
               sqrtf, sinf, cosf, expf, logf, fabsf, powf)

/* Generate float64 (double) evaluator */
DEFINE_ME_EVAL(f64, double,
               vec_add, vec_sub, vec_mul, vec_div, vec_pow,
               vec_add_scalar, vec_mul_scalar, vec_pow_scalar,
               vec_sqrt, vec_sin, vec_cos, vec_negate,
               sqrt, sin, cos, exp, log, fabs, pow)

/* Generate integer evaluators - sin/cos cast to double and back */
DEFINE_ME_EVAL(i8, int8_t,
               vec_add_i8, vec_sub_i8, vec_mul_i8, vec_div_i8, vec_pow_i8,
               vec_add_scalar_i8, vec_mul_scalar_i8, vec_pow_scalar_i8,
               vec_sqrt_i8, vec_sqrt_i8, vec_sqrt_i8, vec_negame_i8,
               sqrt, sin, cos, exp, log, fabs, pow)

DEFINE_ME_EVAL(i16, int16_t,
               vec_add_i16, vec_sub_i16, vec_mul_i16, vec_div_i16, vec_pow_i16,
               vec_add_scalar_i16, vec_mul_scalar_i16, vec_pow_scalar_i16,
               vec_sqrt_i16, vec_sqrt_i16, vec_sqrt_i16, vec_negame_i16,
               sqrt, sin, cos, exp, log, fabs, pow)

DEFINE_ME_EVAL(i32, int32_t,
               vec_add_i32, vec_sub_i32, vec_mul_i32, vec_div_i32, vec_pow_i32,
               vec_add_scalar_i32, vec_mul_scalar_i32, vec_pow_scalar_i32,
               vec_sqrt_i32, vec_sqrt_i32, vec_sqrt_i32, vec_negame_i32,
               sqrt, sin, cos, exp, log, fabs, pow)

DEFINE_ME_EVAL(i64, int64_t,
               vec_add_i64, vec_sub_i64, vec_mul_i64, vec_div_i64, vec_pow_i64,
               vec_add_scalar_i64, vec_mul_scalar_i64, vec_pow_scalar_i64,
               vec_sqrt_i64, vec_sqrt_i64, vec_sqrt_i64, vec_negame_i64,
               sqrt, sin, cos, exp, log, fabs, pow)

DEFINE_ME_EVAL(u8, uint8_t,
               vec_add_u8, vec_sub_u8, vec_mul_u8, vec_div_u8, vec_pow_u8,
               vec_add_scalar_u8, vec_mul_scalar_u8, vec_pow_scalar_u8,
               vec_sqrt_u8, vec_sqrt_u8, vec_sqrt_u8, vec_negame_u8,
               sqrt, sin, cos, exp, log, fabs, pow)

DEFINE_ME_EVAL(u16, uint16_t,
               vec_add_u16, vec_sub_u16, vec_mul_u16, vec_div_u16, vec_pow_u16,
               vec_add_scalar_u16, vec_mul_scalar_u16, vec_pow_scalar_u16,
               vec_sqrt_u16, vec_sqrt_u16, vec_sqrt_u16, vec_negame_u16,
               sqrt, sin, cos, exp, log, fabs, pow)

DEFINE_ME_EVAL(u32, uint32_t,
               vec_add_u32, vec_sub_u32, vec_mul_u32, vec_div_u32, vec_pow_u32,
               vec_add_scalar_u32, vec_mul_scalar_u32, vec_pow_scalar_u32,
               vec_sqrt_u32, vec_sqrt_u32, vec_sqrt_u32, vec_negame_u32,
               sqrt, sin, cos, exp, log, fabs, pow)

DEFINE_ME_EVAL(u64, uint64_t,
               vec_add_u64, vec_sub_u64, vec_mul_u64, vec_div_u64, vec_pow_u64,
               vec_add_scalar_u64, vec_mul_scalar_u64, vec_pow_scalar_u64,
               vec_sqrt_u64, vec_sqrt_u64, vec_sqrt_u64, vec_negame_u64,
               sqrt, sin, cos, exp, log, fabs, pow)

/* Generate complex evaluators */
DEFINE_ME_EVAL(c64, float complex,
               vec_add_c64, vec_sub_c64, vec_mul_c64, vec_div_c64, vec_pow_c64,
               vec_add_scalar_c64, vec_mul_scalar_c64, vec_pow_scalar_c64,
               vec_sqrt_c64, vec_sqrt_c64, vec_sqrt_c64, vec_negame_c64,
               csqrtf, csqrtf, csqrtf, cexpf, clogf, cabsf, cpowf)

DEFINE_ME_EVAL(c128, double complex,
               vec_add_c128, vec_sub_c128, vec_mul_c128, vec_div_c128, vec_pow_c128,
               vec_add_scalar_c128, vec_mul_scalar_c128, vec_pow_scalar_c128,
               vec_sqrt_c128, vec_sqrt_c128, vec_sqrt_c128, vec_negame_c128,
               csqrt, csqrt, csqrt, cexp, clog, cabs, cpow)

/* Public API - dispatches to correct type-specific evaluator */
/* Structure to track promoted variables */
typedef struct {
    void *promoted_data; // Temporary buffer for promoted data
    me_dtype original_type;
    bool needs_free;
} promoted_var_t;

/* Helper to save original variable bindings */
static void save_variable_bindings(const me_expr *node,
                                   const void **original_bounds,
                                   me_dtype *original_types,
                                   int *save_idx) {
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
        case ME_CLOSURE7: {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                save_variable_bindings((const me_expr *) node->parameters[i],
                                       original_bounds, original_types, save_idx);
            }
            break;
        }
    }
}

/* Recursively promote variables in expression tree */
static void promote_variables_in_tree(me_expr *n, me_dtype target_type,
                                      promoted_var_t *promotions, int *promo_count,
                                      int nitems) {
    if (!n) return;

    switch (TYPE_MASK(n->type)) {
        case ME_CONSTANT:
            // Constants are promoted on-the-fly during evaluation
            break;

        case ME_VARIABLE:
            if (n->dtype != target_type) {
                // Need to promote this variable
                void *promoted = malloc(nitems * dtype_size(target_type));
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
                    } else {
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
        case ME_CLOSURE7: {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                promote_variables_in_tree((me_expr *) n->parameters[i], target_type,
                                          promotions, promo_count, nitems);
            }
            break;
        }
    }
}

/* Restore original variable bindings after promotion */
static void restore_variables_in_tree(me_expr *n, const void **original_bounds,
                                      const me_dtype *original_types, int *restore_idx) {
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
        case ME_CLOSURE7: {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                restore_variables_in_tree((me_expr *) n->parameters[i], original_bounds, original_types, restore_idx);
            }
            break;
        }
    }
}

/* Check if all variables in tree match target type */
static bool all_variables_match_type(const me_expr *n, me_dtype target_type) {
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
        case ME_CLOSURE7: {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (!all_variables_match_type((const me_expr *) n->parameters[i], target_type)) {
                    return false;
                }
            }
            return true;
        }
    }

    return true;
}

static void private_eval(const me_expr *n) {
    if (!n) return;

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
    // Allocate tracking structures (max 100 variables)
    promoted_var_t promotions[100];
    int promo_count = 0;

    // Save original variable bindings
    const void *original_bounds[100];
    me_dtype original_types[100];
    int save_idx = 0;

    save_variable_bindings(n, original_bounds, original_types, &save_idx);

    // Promote variables
    promote_variables_in_tree((me_expr *) n, result_type, promotions, &promo_count, n->nitems);

    // Check if we need output type conversion (e.g., computation in float64, output in bool)
    me_dtype saved_dtype = n->dtype;
    void *original_output = n->output;
    void *temp_output = NULL;

    if (saved_dtype != result_type) {
        // Allocate temp buffer for computation
        temp_output = malloc(n->nitems * dtype_size(result_type));
        if (temp_output) {
            ((me_expr *) n)->output = temp_output;
        }
    }

    // Update expression type for evaluation
    ((me_expr *) n)->dtype = result_type;

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
        ((me_expr *) n)->output = original_output;
        free(temp_output);
    }

    // Restore original variable bindings
    int restore_idx = 0;
    restore_variables_in_tree((me_expr *) n, original_bounds, original_types, &restore_idx);

    // Restore expression type
    ((me_expr *) n)->dtype = saved_dtype;

    // Free promoted buffers
    for (int i = 0; i < promo_count; i++) {
        if (promotions[i].needs_free) {
            free(promotions[i].promoted_data);
        }
    }
}

/* Helper to update variable bindings and nitems in tree */
static void save_nitems_in_tree(const me_expr *node, int *nitems_array, int *idx) {
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
        case ME_CLOSURE7: {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                save_nitems_in_tree((const me_expr *) node->parameters[i], nitems_array, idx);
            }
            break;
        }
        default:
            break;
    }
}

static void restore_nitems_in_tree(me_expr *node, const int *nitems_array, int *idx) {
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
        case ME_CLOSURE7: {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                restore_nitems_in_tree((me_expr *) node->parameters[i], nitems_array, idx);
            }
            break;
        }
        default:
            break;
    }
}

/* Helper to free intermediate output buffers */
static void free_intermediate_buffers(me_expr *node) {
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
        case ME_CLOSURE7: {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                me_expr *param = (me_expr *) node->parameters[i];
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
static void save_variable_pointers(const me_expr *node, const void **var_pointers, int *var_count) {
    if (!node) return;
    switch (TYPE_MASK(node->type)) {
        case ME_VARIABLE:
            // Check if this pointer is already in the list
            for (int i = 0; i < *var_count; i++) {
                if (var_pointers[i] == node->bound) return; // Already saved
            }
            var_pointers[*var_count] = node->bound;
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
        case ME_CLOSURE7: {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                save_variable_pointers((const me_expr *) node->parameters[i], var_pointers, var_count);
            }
            break;
        }
    }
}

/* Helper to update variable bindings by matching original pointers */
static void update_vars_by_pointer(me_expr *node, const void **old_pointers, const void **new_pointers, int n_vars) {
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
        case ME_CLOSURE7: {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                update_vars_by_pointer((me_expr *) node->parameters[i], old_pointers, new_pointers, n_vars);
            }
            break;
        }
    }
}

/* Helper to update variable bindings and nitems in tree */
static void update_variable_bindings(me_expr *node, const void **new_bounds, int *var_idx, int new_nitems) {
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
        case ME_CLOSURE7: {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                update_variable_bindings((me_expr *) node->parameters[i], new_bounds, var_idx, new_nitems);
            }
            break;
        }
    }
}

/* Evaluate compiled expression with new variable and output pointers */
static me_expr *clone_expr(const me_expr *src) {
    if (!src) return NULL;

    const int arity = ARITY(src->type);
    const int psize = sizeof(void *) * arity;
    const int size = (sizeof(me_expr) - sizeof(void *)) + psize + (IS_CLOSURE(src->type) ? sizeof(void *) : 0);
    me_expr *clone = malloc(size);
    if (!clone) return NULL;

    // Copy the entire structure
    memcpy(clone, src, size);

    // Clone children recursively
    if (arity > 0) {
        for (int i = 0; i < arity; i++) {
            clone->parameters[i] = clone_expr((const me_expr *) src->parameters[i]);
            if (src->parameters[i] && !clone->parameters[i]) {
                // Clone failed, clean up
                for (int j = 0; j < i; j++) {
                    me_free((me_expr *) clone->parameters[j]);
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
void me_eval(const me_expr *expr, const void **vars_chunk,
             int n_vars, void *output_chunk, int chunk_nitems) {
    if (!expr) return;

    // Verify variable count matches
    const void *original_var_pointers[100];
    int actual_var_count = 0;
    save_variable_pointers(expr, original_var_pointers, &actual_var_count);

    if (actual_var_count != n_vars) {
        return;
    }

    // Clone the expression tree
    me_expr *clone = clone_expr(expr);
    if (!clone) return;

    // Update clone's variable bindings
    update_vars_by_pointer(clone, original_var_pointers, vars_chunk, n_vars);

    // Update clone's nitems throughout the tree
    int update_idx = 0;
    update_variable_bindings(clone, NULL, &update_idx, chunk_nitems);

    // Set output pointer
    clone->output = output_chunk;

    // Evaluate the clone
    private_eval(clone);

    // Free the clone (including any intermediate buffers it allocated)
    me_free(clone);
}


static void optimize(me_expr *n) {
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
            if (((me_expr *) (n->parameters[i]))->type != ME_CONSTANT) {
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


static me_expr *private_compile(const char *expression, const me_variable *variables, int var_count,
                                void *output, int nitems, me_dtype dtype, int *error) {
    // Validate dtype usage: either all vars are ME_AUTO (use dtype), or dtype is ME_AUTO (use var dtypes)
    if (variables && var_count > 0) {
        int auto_count = 0;
        int specified_count = 0;

        for (int i = 0; i < var_count; i++) {
            if (variables[i].dtype == ME_AUTO) {
                auto_count++;
            } else {
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
                return NULL;
            }
        } else {
            // Mode 2: Output dtype is specified
            // Two sub-modes: all ME_AUTO (homogeneous), or all explicit (heterogeneous with conversion)
            if (auto_count > 0 && specified_count > 0) {
                // Mixed mode not allowed
                fprintf(stderr, "Error: Variable dtypes must be all ME_AUTO or all explicitly specified\n");
                if (error) *error = -1;
                return NULL;
            }
        }
    }

    // Create a copy of variables with dtype filled in (if not already set)
    me_variable *vars_copy = NULL;
    if (variables && var_count > 0) {
        vars_copy = malloc(var_count * sizeof(me_variable));
        if (!vars_copy) {
            if (error) *error = -1;
            return NULL;
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
    } else if (variables && var_count > 0) {
        // Use the first variable's dtype as the target for constants
        // This prevents type promotion issues when mixing float32 vars with float64 constants
        s.target_dtype = variables[0].dtype;
    } else {
        s.target_dtype = ME_FLOAT64; // Fallback to double
    }

    next_token(&s);
    me_expr *root = list(&s);

    if (vars_copy) free(vars_copy);

    if (root == NULL) {
        if (error) *error = -1;
        return NULL;
    }

    if (s.type != TOK_END) {
        me_free(root);
        if (error) {
            *error = (s.next - s.start);
            if (*error == 0) *error = 1;
        }
        return 0;
    } else {
        optimize(root);
        root->output = output;
        root->nitems = nitems;

        // If dtype is ME_AUTO, infer from expression; otherwise use provided dtype
        if (dtype == ME_AUTO) {
            root->dtype = infer_output_type(root);
        } else {
            // User explicitly requested a dtype - use it (will cast if needed)
            root->dtype = dtype;
        }

        if (error) *error = 0;
        return root;
    }
}

// Synthetic addresses for ordinal matching (when user provides NULL addresses)
static char synthetic_var_addresses[100];

me_expr *me_compile(const char *expression, const me_variable *variables,
                    int var_count, me_dtype dtype, int *error) {
    // For chunked evaluation, we compile without specific output/nitems
    // If variables have NULL addresses, assign synthetic unique addresses for ordinal matching
    me_variable *vars_copy = NULL;
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
                return NULL;
            }

            for (int i = 0; i < var_count; i++) {
                vars_copy[i] = variables[i];
                if (vars_copy[i].address == NULL) {
                    // Use address in synthetic array (each index is unique)
                    vars_copy[i].address = &synthetic_var_addresses[i];
                }
            }

            me_expr *result = private_compile(expression, vars_copy, var_count, NULL, 0, dtype, error);
            free(vars_copy);
            return result;
        }
    }

    // No NULL addresses, use variables as-is
    return private_compile(expression, variables, var_count, NULL, 0, dtype, error);
}

static void pn(const me_expr *n, int depth) {
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

void me_print(const me_expr *n) {
    pn(n, 0);
}

me_dtype me_get_dtype(const me_expr *expr) {
    return expr ? expr->dtype : ME_AUTO;
}
