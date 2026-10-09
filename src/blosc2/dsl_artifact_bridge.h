/* Copyright (c) 2026, Blosc Development Team <blosc@blosc.org>
 * SPDX-License-Identifier: BSD-3-Clause
 * Optional native artifact support; old miniexpr builds remain usable. */
#ifndef B2_DSL_ARTIFACT_BRIDGE_H
#define B2_DSL_ARTIFACT_BRIDGE_H
#include "miniexpr.h"
#include <stdio.h>
#include <string.h>

#ifdef B2_HAVE_PORTABLE_ARTIFACT
#include "miniexpr_artifact.h"
#endif
#if defined(B2_HAVE_PORTABLE_ARTIFACT) && defined(ME_ARTIFACT_DRAFT_SCHEMA_VERSION)
typedef me_artifact_input b2_artifact_input;
typedef me_artifact_error b2_artifact_error;
static int b2_artifact_available(void) { return !strcmp(ME_ARTIFACT_SCHEMA_VERSION, "1.0"); }
static int b2_artifact_load(const char *json, size_t size, int jit, void **out, b2_artifact_error *error) {
    if (!b2_artifact_available()) {
        *out = NULL;
        memset(error, 0, sizeof(*error));
        snprintf(error->message, sizeof(error->message), "Build with miniexpr draft 1.0 artifact support");
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
    snprintf(error->message, sizeof(error->message), "Build with miniexpr portable artifact support");
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
#if defined(B2_HAVE_PORTABLE_ARTIFACT) && defined(ME_ARTIFACT_DRAFT_SCHEMA_VERSION)
typedef me_artifact_buffer b2_artifact_buffer;
typedef me_artifact_eval_descriptor b2_artifact_descriptor;
static int b2_artifact_descriptor_available(void) { return b2_artifact_available(); }
static int b2_artifact_cardinality(const void *handle) { return me_artifact_result_cardinality(handle); }
static size_t b2_artifact_width(const void *handle, int index) {
    return index < 0 ? me_artifact_output_itemsize(handle) : me_artifact_input_itemsize(handle, index);
}
static int b2_artifact_rank(const void *handle) { return me_artifact_context_ndim(handle); }
static const char *b2_artifact_version(const void *handle) { return me_artifact_schema_version(handle); }
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
static const char *b2_artifact_version(const void *handle) { (void)handle; return "unsupported"; }
static int b2_artifact_eval_ex(const void *handle, const b2_artifact_buffer *inputs, int ninputs,
                             void *output, const b2_artifact_descriptor *descriptor, b2_artifact_error *error) {
    (void)handle; (void)inputs; (void)ninputs; (void)output; (void)descriptor;
    memset(error, 0, sizeof(*error));
    snprintf(error->message, sizeof(error->message), "Build with draft miniexpr 1.0 descriptor support");
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
