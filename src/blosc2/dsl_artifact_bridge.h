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
typedef me_artifact_input b2_artifact_input;
typedef me_artifact_error b2_artifact_error;
static int b2_artifact_available(void) { return 1; }
static int b2_artifact_load(const char *json, size_t size, int jit, void **out, b2_artifact_error *error) {
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
#endif
