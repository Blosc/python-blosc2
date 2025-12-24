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

#ifndef MINIEXPR_H
#define MINIEXPR_H


#ifdef __cplusplus
extern "C" {


#endif

/* Internal eval block size (elements). Compile-time fixed. */
#ifndef ME_EVAL_BLOCK_NITEMS
#define ME_EVAL_BLOCK_NITEMS 1024
#endif

/* Maximum number of variables supported in a single expression. */
#ifndef ME_MAX_VARS
#define ME_MAX_VARS 128
#endif

/* Enable internal eval blocking for large chunks (1 = on, 0 = off). */
#ifndef ME_EVAL_ENABLE_BLOCKING
#define ME_EVAL_ENABLE_BLOCKING 1
#endif


/* Data type enumeration - Full C99 support */
typedef enum {
    /* Automatic type inference */
    ME_AUTO,

    /* Boolean */
    ME_BOOL,

    /* Signed integers */
    ME_INT8,
    ME_INT16,
    ME_INT32,
    ME_INT64,

    /* Unsigned integers */
    ME_UINT8,
    ME_UINT16,
    ME_UINT32,
    ME_UINT64,

    /* Floating point */
    ME_FLOAT32,
    ME_FLOAT64,

    /* Complex (C99) */
    ME_COMPLEX64, /* float complex */
    ME_COMPLEX128 /* double complex */
} me_dtype;

/* Opaque type for compiled expressions */
typedef struct me_expr me_expr;


enum {
    ME_VARIABLE = 0,

    ME_FUNCTION0 = 8, ME_FUNCTION1, ME_FUNCTION2, ME_FUNCTION3,
    ME_FUNCTION4, ME_FUNCTION5, ME_FUNCTION6, ME_FUNCTION7,

    ME_CLOSURE0 = 16, ME_CLOSURE1, ME_CLOSURE2, ME_CLOSURE3,
    ME_CLOSURE4, ME_CLOSURE5, ME_CLOSURE6, ME_CLOSURE7,

    ME_FLAG_PURE = 32
};

typedef struct me_variable {
    const char *name;
    me_dtype dtype; // Data type of this variable (ME_AUTO = use output dtype)
    const void *address; // Pointer to data (NULL for me_compile)
    int type; // ME_VARIABLE for user variables (0 = auto-set to ME_VARIABLE)
    void *context; // For closures/functions (NULL for normal variables)
} me_variable;

/* Note: When initializing variables, only name/dtype/address are typically needed.
 * Unspecified fields default to 0/NULL, which is correct for normal use:
 *   {"varname"}                          → defaults all fields
 *   {"varname", ME_FLOAT64}              → for me_compile with mixed types
 *   {"varname", ME_FLOAT64, var_array}   → for me_compile with address
 * Advanced users can specify type for closures/functions if needed.
 */


/* Compile expression for chunked evaluation.
 * This function is optimized for use with me_eval(),
 * where variable and output pointers are provided later during evaluation.
 *
 * Parameters:
 *   expression: The expression string to compile
 *   variables: Array of variable definitions. Only the 'name' field is required.
 *              Variables will be matched by position (ordinal order) during me_eval().
 *   var_count: Number of variables
 *   dtype: Data type handling:
 *          - ME_AUTO: All variables must specify their dtypes, output is inferred
 *          - Specific type: Either all variables are ME_AUTO (homogeneous, all use this type),
 *            OR all variables have explicit dtypes (heterogeneous, result cast to this type)
 *   error: Optional pointer to receive error position (0 on success, >0 on error)
 *
 * Returns: Compiled expression ready for chunked evaluation, or NULL on error
 *
 * Example 1 (simple - all same type):
 *   me_variable vars[] = {{"x"}, {"y"}};  // Both ME_AUTO
 *   me_expr *expr = me_compile("x + y", vars, 2, ME_FLOAT64, &err);
 *
 * Example 2 (mixed types with ME_AUTO):
 *   me_variable vars[] = {{"x", ME_INT32}, {"y", ME_FLOAT64}};
 *   me_expr *expr = me_compile("x + y", vars, 2, ME_AUTO, &err);
 *
 * Example 3 (mixed types with explicit output):
 *   me_variable vars[] = {{"x", ME_INT32}, {"y", ME_FLOAT64}};
 *   me_expr *expr = me_compile("x + y", vars, 2, ME_FLOAT32, &err);
 *   // Variables keep their types, result is cast to FLOAT32
 *
 *   // Later, provide data in same order as variable definitions
 *   const void *data[] = {x_array, y_array};  // x first, y second
 *   me_eval(expr, data, 2, output, nitems);
 */
me_expr *me_compile(const char *expression, const me_variable *variables,
                    int var_count, me_dtype dtype, int *error);

/* Evaluates compiled expression with variable and output pointers.
 * This function can be safely called from multiple threads simultaneously on the
 * same compiled expression. It creates a temporary clone of the expression tree
 * for each call, eliminating race conditions at the cost of some memory allocation.
 *
 * Parameters:
 *   expr: Compiled expression (from me_compile)
 *   vars_chunk: Array of pointers to variable data chunks (same order as in me_compile)
 *   n_vars: Number of variables (must match the number used in me_compile)
 *   output_chunk: Pointer to output buffer for this chunk
 *   chunk_nitems: Number of elements in this chunk
 *
 * Use this function for both serial and parallel evaluation. It is thread-safe
 * and can be used from multiple threads to process different chunks simultaneously.
 */
void me_eval(const me_expr *expr, const void **vars_chunk,
             int n_vars, void *output_chunk, int chunk_nitems);

/* Prints the expression tree for debugging purposes. */
void me_print(const me_expr *n);

/* Frees the expression. */
/* This is safe to call on NULL pointers. */
void me_free(me_expr *n);

/* Get the result data type of a compiled expression.
 * Returns the dtype that will be used for the output of me_eval().
 */
me_dtype me_get_dtype(const me_expr *expr);


#ifdef __cplusplus
}
#endif

#endif /*MINIEXPR_H*/
