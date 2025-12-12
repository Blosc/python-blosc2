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

#ifndef MINIEXPR_H
#define MINIEXPR_H


#ifdef __cplusplus
extern "C" {

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

typedef struct me_expr {
    int type;

    union {
        double value;
        const double *bound;
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
} me_expr;


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
    me_dtype dtype;      // Data type of this variable (ME_AUTO = use output dtype)
    const void *address; // Pointer to data (NULL for me_compile_chunk)
    int type;            // ME_VARIABLE for user variables (0 = auto-set to ME_VARIABLE)
    void *context;       // For closures/functions (NULL for normal variables)
} me_variable;

/* Note: When initializing variables, only name/dtype/address are typically needed.
 * Unspecified fields default to 0/NULL, which is correct for normal use:
 *   {"varname"}                          → defaults all fields
 *   {"varname", ME_FLOAT64}              → for me_compile_chunk with mixed types
 *   {"varname", ME_FLOAT64, var_array}   → for me_compile with address
 * Advanced users can specify type for closures/functions if needed.
 */


/* Parses the input expression and binds variables. */
/* Returns NULL on error. */
/*
 * The dtype parameter controls variable type handling:
 *   - If dtype is ME_AUTO: All variables must have explicit dtypes (not ME_AUTO).
 *                          Output dtype is inferred from the expression.
 *   - If dtype is specified: All variables must be ME_AUTO.
 *                            Both variables and output use this dtype.
 * The actual result type is available in expr->dtype after compilation.
 */
me_expr *me_compile(const char *expression, const me_variable *variables, int var_count,
                    void *output, int nitems, me_dtype dtype, int *error);

/* Compile expression for chunked evaluation.
 * This variant is optimized for use with me_eval_chunk() and me_eval_chunk_threadsafe(),
 * where variable and output pointers are provided later during evaluation.
 *
 * Parameters:
 *   expression: The expression string to compile
 *   variables: Array of variable definitions. Only the 'name' field is required.
 *              Variables will be matched by position (ordinal order) during me_eval_chunk().
 *   var_count: Number of variables
 *   dtype: Data type handling (same rules as me_compile):
 *          - ME_AUTO: All variables must specify their dtypes, output is inferred
 *          - Specific type: All variables must be ME_AUTO, this type is used for all
 *   error: Optional pointer to receive error position (0 on success, >0 on error)
 *
 * Returns: Compiled expression ready for chunked evaluation, or NULL on error
 *
 * Example 1 (simple - all same type):
 *   me_variable vars[] = {{"x"}, {"y"}};  // Both ME_AUTO
 *   me_expr *expr = me_compile_chunk("x + y", vars, 2, ME_FLOAT64, &err);
 *
 * Example 2 (mixed types):
 *   me_variable vars[] = {{"x", ME_INT32}, {"y", ME_FLOAT64}};
 *   me_expr *expr = me_compile_chunk("x + y", vars, 2, ME_AUTO, &err);
 *
 *   // Later, provide data in same order as variable definitions
 *   const void *data[] = {x_array, y_array};  // x first, y second
 *   me_eval_chunk(expr, data, 2, output, nitems);
 */
me_expr *me_compile_chunk(const char *expression, const me_variable *variables,
                          int var_count, me_dtype dtype, int *error);

/* Evaluates the expression on vectors. */
void me_eval(const me_expr *n);

/* Evaluates using fused bytecode (faster for complex expressions). */
void me_eval_fused(const me_expr *n);

/* Evaluates compiled expression with new variable and output pointers.
 * This allows processing large arrays in chunks without recompiling.
 *
 * Parameters:
 *   expr: Compiled expression (from me_compile)
 *   vars_chunk: Array of pointers to variable data chunks (same order as in me_compile)
 *   n_vars: Number of variables (must match the number used in me_compile)
 *   output_chunk: Pointer to output buffer for this chunk
 *   chunk_nitems: Number of elements in this chunk
 *
 * Note: The chunks must have the same data types as the original variables.
 * WARNING: This function is NOT thread-safe. Use me_eval_chunk_threadsafe() for
 *          concurrent evaluation from multiple threads.
 */
void me_eval_chunk(const me_expr *expr, const void **vars_chunk, int n_vars,
                   void *output_chunk, int chunk_nitems);

/* Thread-safe version of me_eval_chunk.
 * This function can be safely called from multiple threads simultaneously on the
 * same compiled expression. It creates a temporary clone of the expression tree
 * for each call, eliminating race conditions at the cost of some memory allocation.
 *
 * Use this when you need to evaluate the same expression in parallel across
 * different chunks from multiple threads.
 */
void me_eval_chunk_threadsafe(const me_expr *expr, const void **vars_chunk,
                              int n_vars, void *output_chunk, int chunk_nitems);

/* Prints debugging information on the syntax tree. */
void me_print(const me_expr *n);

/* Frees the expression. */
/* This is safe to call on NULL pointers. */
void me_free(me_expr *n);


#ifdef __cplusplus
}
#endif

#endif /*MINIEXPR_H*/
