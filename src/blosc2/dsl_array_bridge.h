#ifndef BLOSC2_DSL_ARRAY_BRIDGE_H
#define BLOSC2_DSL_ARRAY_BRIDGE_H
#include "dsl_artifact_bridge.h"
#ifdef ME_ARTIFACT_ARRAY_VERSION
typedef me_array_view b2_array_view;
typedef me_array_options b2_array_options;
typedef me_array_report b2_array_report;
static int b2_array_shape(const void *handle, int rank, const int64_t *shape,
    const b2_array_options *options, int *out_rank, int64_t *out_shape,
    me_dtype *dtype, b2_artifact_error *error) {
    return me_array_result_shape(handle,rank,shape,options,out_rank,out_shape,dtype,error);
}
static int b2_array_eval(const void *handle, const b2_array_view *inputs, int ninputs,
    int rank, const int64_t *shape, const b2_array_options *options, void *output,
    size_t capacity, b2_array_report *report, b2_artifact_error *error) {
    return me_artifact_eval_array(handle,inputs,ninputs,rank,shape,options,output,capacity,report,error);
}
#else
typedef struct {
    const char *name; me_dtype dtype; const void *base;
    size_t capacity; size_t offset; int rank;
    int64_t shape[16]; int64_t strides[16]; unsigned byte_order;
} b2_array_view;
typedef struct {
    unsigned version; int reduction; int naxes; int axes[16]; bool keepdims;
    me_dtype accumulator; const void *initial; size_t tile_items; const b2_array_view *where;
} b2_array_options;
typedef struct {
    size_t temporary_bytes; size_t gathered_bytes; size_t zero_copy_tiles;
    size_t evaluated_tiles; unsigned fp_flags; unsigned fp_supported;
} b2_array_report;
static int b2_array_shape(const void *handle, int rank, const int64_t *shape,
    const b2_array_options *options, int *out_rank, int64_t *out_shape,
    me_dtype *dtype, b2_artifact_error *error) {
    (void)handle; (void)rank; (void)shape; (void)options; (void)out_rank; (void)out_shape; (void)dtype;
    memset(error,0,sizeof(*error)); snprintf(error->message,sizeof(error->message),"Native runtime does not support logical arrays"); return -2;
}
static int b2_array_eval(const void *handle, const b2_array_view *inputs, int ninputs,
    int rank, const int64_t *shape, const b2_array_options *options, void *output,
    size_t capacity, b2_array_report *report, b2_artifact_error *error) {
    (void)handle; (void)inputs; (void)ninputs; (void)rank; (void)shape; (void)options; (void)output; (void)capacity; (void)report;
    memset(error,0,sizeof(*error)); snprintf(error->message,sizeof(error->message),"Native runtime does not support logical arrays"); return -2;
}
#endif
#endif
