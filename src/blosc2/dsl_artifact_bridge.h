/* Copyright (c) 2026, Blosc Development Team <blosc@blosc.org>
 * SPDX-License-Identifier: BSD-3-Clause
 * Optional native artifact support; old miniexpr builds remain usable. */
#ifndef B2_DSL_ARTIFACT_BRIDGE_H
#define B2_DSL_ARTIFACT_BRIDGE_H
#include "miniexpr.h"
#include "miniexpr_kernel.h"
#include "miniexpr_grid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Formula discovery uses the same typed contract as formula execution. Never
 * wrap a formula in a different language merely to ask for its string width. */
static int b2_formula_describe(const char *source, const me_input_spec *inputs,
    size_t ninputs, me_dtype *dtype, size_t *width) {
    me_signature signature = ME_SIGNATURE_INITIALIZER;
    signature.inputs = inputs;
    signature.ninputs = ninputs;
    me_kernel *kernel = NULL;
    int rc = me_prepare_formula(source, &signature, NULL, &kernel, NULL);
    if (rc == ME_SUCCESS) {
        me_kernel_info info = ME_KERNEL_INFO_INITIALIZER;
        rc = me_describe(kernel, &info);
        if (rc == ME_SUCCESS) {
            *dtype = info.output_dtype;
            *width = info.output_itemsize;
        }
    }
    me_kernel_free(kernel);
    return rc;
}

typedef struct {
    me_kernel *kernel;
    me_grid *grid;
    size_t ninputs;
    size_t widths[ME_MAX_VARS];
    size_t output_width;
} b2_formula_grid;

static void b2_formula_grid_free(void *handle) {
    b2_formula_grid *state = (b2_formula_grid *)handle;
    if (!state) return;
    me_kernel_free(state->kernel);
    me_grid_free(state->grid);
    free(state);
}

static int b2_formula_grid_prepare(const char *source, const me_variable *inputs,
    int ninputs, me_dtype dtype, size_t width, int ndim, const int64_t *shape,
    const int32_t *chunks, const int32_t *blocks, void **out) {
    *out = NULL;
    if (ninputs < 0 || ninputs > ME_MAX_VARS || !width) return ME_ERR_INVALID_ARG;
    b2_formula_grid *state = (b2_formula_grid *)calloc(1, sizeof(*state));
    if (!state) return ME_ERR_OOM;
    me_input_spec slots[ME_MAX_VARS];
    for (int i = 0; i < ninputs; i++) {
        slots[i] = (me_input_spec){inputs[i].name, inputs[i].dtype, inputs[i].itemsize};
        state->widths[i] = inputs[i].itemsize;
        if (!state->widths[i]) {
            switch (inputs[i].dtype) {
            case ME_BOOL: case ME_INT8: case ME_UINT8: state->widths[i] = 1; break;
            case ME_INT16: case ME_UINT16: state->widths[i] = 2; break;
            case ME_INT32: case ME_UINT32: case ME_FLOAT32: state->widths[i] = 4; break;
            case ME_INT64: case ME_UINT64: case ME_FLOAT64: case ME_COMPLEX64: state->widths[i] = 8; break;
            case ME_COMPLEX128: state->widths[i] = 16; break;
            default: break;
            }
        }
    }
    state->ninputs = (size_t)ninputs;
    state->output_width = width;
    me_signature signature = ME_SIGNATURE_INITIALIZER;
    signature.inputs = slots;
    signature.ninputs = state->ninputs;
    signature.output_dtype = dtype;
    signature.output_itemsize = width;
    signature.cardinality = ME_ELEMENTWISE;
    int rc = me_prepare_formula(source, &signature, NULL, &state->kernel, NULL);
    me_grid_spec grid = ME_GRID_SPEC_INITIALIZER;
    grid.ndim = ndim;
    grid.shape = shape;
    grid.chunkshape = chunks;
    grid.blockshape = blocks;
    if (rc == ME_SUCCESS) rc = me_grid_create(&grid, &state->grid, NULL);
    if (rc != ME_SUCCESS) b2_formula_grid_free(state);
    else *out = state;
    return rc;
}

static int b2_formula_grid_valid(const void *handle, int64_t chunk, int64_t block,
    int64_t *valid) {
    const b2_formula_grid *state = (const b2_formula_grid *)handle;
    me_grid_block_info info = ME_GRID_BLOCK_INFO_INITIALIZER;
    int rc = me_grid_describe_block(state->grid, chunk, block, &info, NULL);
    if (rc == ME_SUCCESS) *valid = (int64_t)info.valid_nitems;
    return rc;
}

static int b2_formula_grid_eval(const void *handle, const void **inputs, size_t nitems,
    int64_t chunk, int64_t block, void *output) {
    const b2_formula_grid *state = (const b2_formula_grid *)handle;
    me_buffer buffers[ME_MAX_VARS];
    for (size_t i = 0; i < state->ninputs; i++) {
        if (!state->widths[i] || nitems > SIZE_MAX / state->widths[i]) return ME_ERR_INVALID_ARG;
        buffers[i] = (me_buffer){inputs[i], nitems * state->widths[i]};
    }
    if (nitems > SIZE_MAX / state->output_width) return ME_ERR_INVALID_ARG;
    return me_grid_eval_block(state->grid, state->kernel, chunk, block, buffers,
        state->ninputs, (me_mut_buffer){output, nitems * state->output_width}, NULL, NULL);
}

#ifdef B2_HAVE_MENUDET_ARTIFACT
#include "miniexpr_artifact.h"
#endif

static int b2_artifact_options_available(void) {
#ifdef ME_ARTIFACT_LOAD_OPTIONS_VERSION
    return 1;
#else
    return 0;
#endif
}
static int b2_artifact_auto_jit_mode(void) {
#ifdef ME_ARTIFACT_EXECUTION_KIND_VERSION
    return 3;
#else
    return 0;
#endif
}
static int b2_core_prepare_available(void) {
#ifdef ME_ARTIFACT_DIRECT_PREPARE_VERSION
    return 1;
#else
    return 0;
#endif
}
static const char *b2_artifact_execution_kind(const void *handle) {
#ifdef ME_ARTIFACT_EXECUTION_KIND_VERSION
    return me_artifact_execution_kind((const me_artifact *)handle);
#else
    (void)handle;
    return "unknown";
#endif
}
#if defined(B2_HAVE_MENUDET_ARTIFACT) && defined(ME_ARTIFACT_MENUDET_SCHEMA_VERSION)
typedef me_artifact_input b2_artifact_input;
typedef me_artifact_error b2_artifact_error;
static int b2_artifact_available(void) { return !strcmp(ME_ARTIFACT_SCHEMA_VERSION, "0.1"); }
static int b2_artifact_load(const char *json, size_t size, int jit, void **out, b2_artifact_error *error) {
    if (!b2_artifact_available()) {
        *out = NULL;
        memset(error, 0, sizeof(*error));
        snprintf(error->message, sizeof(error->message), "Build with miniexpr Menudet 0.1 beta artifact support");
        return -100;
    }
    me_artifact *handle = NULL;
    int rc = me_artifact_load(json, size, (me_jit_mode)jit, &handle, error);
    *out = handle;
    return rc;
}
static int b2_artifact_eval(const void *handle, const b2_artifact_input *inputs, int ninputs,
                          void *output, size_t count, b2_artifact_error *error) {
    return me_artifact_eval((const me_artifact *)handle, inputs, ninputs, output, count, error);
}
static void b2_artifact_free(void *handle) { me_artifact_free((me_artifact *)handle); }
static const char *b2_artifact_source(const void *handle) { return me_artifact_source(handle); }
static const char *b2_artifact_entry(const void *handle) { return me_artifact_entry_point(handle); }
static int b2_artifact_ninputs(const void *handle) { return me_artifact_ninputs(handle); }
static const char *b2_artifact_name(const void *handle, int index) { return me_artifact_input_name(handle, index); }
static me_dtype b2_artifact_dtype(const void *handle, int index) { return me_artifact_input_dtype(handle, index); }
static me_dtype b2_artifact_output(const void *handle) { return me_artifact_output_dtype(handle); }
static int b2_artifact_jit(const void *handle) { return me_artifact_has_jit(handle); }
#else
typedef struct {
    const char *name;
    me_dtype dtype;
    const void *data;
    size_t nitems;
} b2_artifact_input;
typedef struct {
    int native_status;
    int line;
    int column;
    char message[256];
} b2_artifact_error;
static int b2_artifact_available(void) { return 0; }
static int b2_artifact_load(const char *json, size_t size, int jit, void **out, b2_artifact_error *error) {
    (void)json; (void)size; (void)jit;
    *out = NULL;
    memset(error, 0, sizeof(*error));
    snprintf(error->message, sizeof(error->message), "Build with miniexpr Menudet artifact support");
    return -100;
}
static int b2_artifact_eval(const void *handle, const b2_artifact_input *inputs, int ninputs,
                          void *output, size_t count, b2_artifact_error *error) {
    (void)handle; (void)inputs; (void)ninputs; (void)output; (void)count; (void)error;
    return -100;
}
static void b2_artifact_free(void *handle) { (void)handle; }
static const char *b2_artifact_source(const void *handle) { (void)handle; return NULL; }
static const char *b2_artifact_entry(const void *handle) { (void)handle; return NULL; }
static int b2_artifact_ninputs(const void *handle) { (void)handle; return 0; }
static const char *b2_artifact_name(const void *handle, int index) { (void)handle; (void)index; return NULL; }
static me_dtype b2_artifact_dtype(const void *handle, int index) { (void)handle; (void)index; return ME_AUTO; }
static me_dtype b2_artifact_output(const void *handle) { (void)handle; return ME_AUTO; }
static int b2_artifact_jit(const void *handle) { (void)handle; return 0; }
#endif

static int b2_artifact_prepare_source(const char *version, const char *source,
    const me_signature *signature, int jit, const me_jit_options *options,
    void **out, b2_artifact_error *error) {
#ifdef ME_ARTIFACT_DIRECT_PREPARE_VERSION
    me_artifact *handle = NULL;
    int rc = me_artifact_prepare_menudet(version, source, signature, jit, options, &handle, error);
    *out = handle;
    return rc;
#else
    (void)version; (void)source; (void)signature; (void)jit; (void)options;
    *out = NULL;
    memset(error, 0, sizeof(*error));
    snprintf(error->message, sizeof(error->message), "Build with direct Menudet preparation support");
    return -100;
#endif
}

static int b2_artifact_load_options(const char *json, size_t size, int jit,
                                  const me_jit_options *options, void **out, b2_artifact_error *error) {
#ifdef ME_ARTIFACT_LOAD_OPTIONS_VERSION
    me_artifact *handle = NULL;
    int rc = me_artifact_load_options(json, size, (me_jit_mode)jit, options, &handle, error);
    *out = handle;
    return rc;
#else
    (void)options;
    return b2_artifact_load(json, size, jit, out, error);
#endif
}
static int b2_artifact_load_inferred_available(void) {
#ifdef ME_ARTIFACT_INFER_OUTPUT_VERSION
    return 1;
#else
    return 0;
#endif
}
static int b2_artifact_load_inferred(const char *json, size_t size, int jit,
                                     const me_jit_options *options, void **out,
                                     b2_artifact_error *error) {
#ifdef ME_ARTIFACT_INFER_OUTPUT_VERSION
    me_artifact *handle = NULL;
    int rc = me_artifact_load_inferred_options(json, size, (me_jit_mode)jit, options, &handle, error);
    *out = handle;
    return rc;
#else
    (void)options;
    return b2_artifact_load(json, size, jit, out, error);
#endif
}
#if defined(B2_HAVE_MENUDET_ARTIFACT) && defined(ME_ARTIFACT_MENUDET_SCHEMA_VERSION)
typedef me_artifact_buffer b2_artifact_buffer;
typedef me_artifact_eval_descriptor b2_artifact_descriptor;
static int b2_artifact_descriptor_available(void) { return b2_artifact_available(); }
static int b2_artifact_cardinality(const void *handle) { return me_artifact_result_cardinality(handle); }
static size_t b2_artifact_width(const void *handle, int index) {
    return index < 0 ? me_artifact_output_itemsize(handle) : me_artifact_input_itemsize(handle, index);
}
static int b2_artifact_rank(const void *handle) { return me_artifact_context_ndim(handle); }
static int b2_artifact_partitionable(const void *handle) {
#ifdef ME_ARTIFACT_PARTITIONABLE_VERSION
    return me_artifact_is_partitionable(handle);
#else
    (void)handle;
    return -1;
#endif
}
static const char *b2_artifact_version(const void *handle) { return me_artifact_schema_version(handle); }
#ifdef ME_ARTIFACT_MATH_POLICY_VERSION
static const char *b2_artifact_math(const void *handle) { return me_artifact_math_backend(handle); }
static const char *b2_artifact_math_ulp(const void *handle) { return me_artifact_sleef_accuracy(handle); }
static const char *b2_artifact_fp(const void *handle) { return me_artifact_fp_mode(handle); }
#else
static const char *b2_artifact_math(const void *handle) { (void)handle; return "unknown"; }
static const char *b2_artifact_math_ulp(const void *handle) { (void)handle; return ""; }
static const char *b2_artifact_fp(const void *handle) { (void)handle; return "unknown"; }
#endif
static int b2_artifact_eval_ex(const void *handle, const b2_artifact_buffer *inputs, int ninputs,
                             void *output, const b2_artifact_descriptor *descriptor, b2_artifact_error *error) {
    return me_artifact_eval_ex(handle, inputs, ninputs, output, descriptor, error);
}
#else
typedef struct {
    const char *name;
    me_dtype dtype;
    size_t itemsize;
    const void *data;
    size_t capacity;
} b2_artifact_buffer;
typedef struct {
    size_t struct_size;
    unsigned int version;
    size_t nitems;
    size_t output_capacity;
    const uint8_t *valid_mask;
    size_t valid_mask_capacity;
    int ndim;
    const int64_t *logical_shape;
    const int64_t *block_origin;
    const int64_t *block_extent;
} b2_artifact_descriptor;
static int b2_artifact_descriptor_available(void) { return 0; }
static int b2_artifact_cardinality(const void *handle) { (void)handle; return 0; }
static size_t b2_artifact_width(const void *handle, int index) { (void)handle; (void)index; return 0; }
static int b2_artifact_rank(const void *handle) { (void)handle; return 0; }
static int b2_artifact_partitionable(const void *handle) { (void)handle; return -1; }
static const char *b2_artifact_version(const void *handle) { (void)handle; return "unsupported"; }
static const char *b2_artifact_math(const void *handle) { (void)handle; return "unknown"; }
static const char *b2_artifact_math_ulp(const void *handle) { (void)handle; return ""; }
static const char *b2_artifact_fp(const void *handle) { (void)handle; return "unknown"; }
static int b2_artifact_eval_ex(const void *handle, const b2_artifact_buffer *inputs, int ninputs,
                             void *output, const b2_artifact_descriptor *descriptor, b2_artifact_error *error) {
    (void)handle; (void)inputs; (void)ninputs; (void)output; (void)descriptor;
    memset(error, 0, sizeof(*error));
    snprintf(error->message, sizeof(error->message), "Build with Menudet 0.1 beta descriptor support");
    return -2;
}
#endif
#ifdef ME_ARTIFACT_NUMPY_SCHEMA_VERSION
static me_dtype b2_artifact_inferred(const void *handle) { return me_artifact_inferred_dtype(handle); }
#else
static me_dtype b2_artifact_inferred(const void *handle) { (void)handle; return ME_AUTO; }
#endif
#ifdef ME_ARTIFACT_FP_STATUS_VERSION
typedef me_artifact_fp_status b2_artifact_fp_status;
static int b2_artifact_eval_status(const void *handle, const b2_artifact_buffer *inputs, int ninputs,
    void *output, const b2_artifact_descriptor *descriptor, unsigned mask,
    b2_artifact_fp_status *status, b2_artifact_error *error) {
    return me_artifact_eval_status(handle, inputs, ninputs, output, descriptor, mask, status, error);
}
#else
typedef struct { unsigned flags; unsigned supported; } b2_artifact_fp_status;
static int b2_artifact_eval_status(const void *handle, const b2_artifact_buffer *inputs, int ninputs,
    void *output, const b2_artifact_descriptor *descriptor, unsigned mask,
    b2_artifact_fp_status *status, b2_artifact_error *error) {
    (void)handle; (void)inputs; (void)ninputs; (void)output; (void)descriptor; (void)mask;
    status->flags = status->supported = 0;
    memset(error, 0, sizeof(*error));
    snprintf(error->message, sizeof(error->message), "Native runtime does not support floating status");
    return -2;
}
#endif
#endif
