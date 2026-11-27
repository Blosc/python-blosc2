/* Private optional typed-plan bridge. Standard builds retain the adapter;
 * this bridge never invokes it, including on preparation/execution failure. */
#ifndef B2_EXEC_BRIDGE_H
#define B2_EXEC_BRIDGE_H
#include "miniexpr_kernel.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    int op, dtype, slot, args[2], nargs, weak;
    size_t itemsize, scalar_size;
    unsigned char scalar[16];
    int has_axes, keepdims, has_initial, initial_dtype, initial_weak;
    size_t naxes, initial_size;
    int axes[16];
    unsigned char initial_scalar[16];
} b2_exec_node;

typedef struct {
    int dtype, ndim;
    size_t itemsize;
    int64_t shape[16];
} b2_exec_meta;

typedef struct {
    int dtype, reduction, identity, map_dtype, reducer, partitionable, has_jit;
    size_t itemsize, ninputs, nitems, tile_items, mapped_bytes, reducer_bytes, output_items;
    size_t plan_metadata_bytes, schedule_metadata_bytes;
    size_t ndim, broadcast_gather_bytes;
    int64_t shape[16];
    int has_axes, keepdims, has_initial, pairwise_sum;
    size_t naxes, output_ndim, reduction_groups, reduction_lanes;
    int axes[16];
    int64_t output_shape[16];
} b2_exec_info;

typedef struct {
    size_t tiles, map_calls, direct_tiles, mapped_bytes;
    me_fp_status fp;
    size_t tasks, worker_bytes, gather_bytes, gathered_bytes;
    size_t staging_bytes;
    size_t interpreter_calls, jit_calls;
} b2_exec_report;

typedef struct {
    const void *base;
    size_t capacity, offset;
    int64_t strides[16];
    unsigned byte_order;
} b2_exec_view;

typedef struct {
    void *base;
    size_t capacity, offset;
    int64_t strides[16];
} b2_exec_output_view;

#ifdef B2_HAVE_B2ND_EXEC
#include "b2nd-exec.h"
typedef me_fp_scope b2_exec_fp_scope;

static int b2_exec_fp_begin(b2_exec_fp_scope *scope, me_fp_status *status, me_error *error) {
    return me_fp_scope_begin(scope, status, error);
}
static int b2_exec_fp_end(b2_exec_fp_scope *scope, me_error *error) {
    return me_fp_scope_end(scope, error);
}

static int b2_exec_available(void) { return 1; }
static int b2_exec_prepare(const b2_exec_node *nodes, size_t count, size_t ninputs,
                           uint32_t root, int execution, void **out, me_error *error) {
    *out = NULL;
    if (!count || count > B2ND_EXEC_MAX_NODES) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_exec_node *native = calloc(count, sizeof(*native));
    if (!native) {
        return ME_ERR_OOM;
    }
    for (size_t i = 0; i < count; i++) {
        native[i].struct_size = sizeof(*native);
        native[i].id = (uint32_t)i;
        native[i].op = (b2nd_exec_op)nodes[i].op;
        native[i].dtype = (me_dtype)nodes[i].dtype;
        native[i].slot = (uint32_t)nodes[i].slot;
        native[i].nargs = (size_t)nodes[i].nargs;
        native[i].args[0] = (uint32_t)nodes[i].args[0];
        native[i].args[1] = (uint32_t)nodes[i].args[1];
        native[i].itemsize = nodes[i].itemsize;
        native[i].scalar_size = nodes[i].scalar_size;
        native[i].category = nodes[i].weak ? ME_CAPTURE_WEAK : ME_CAPTURE_TYPED;
        memcpy(native[i].scalar, nodes[i].scalar, sizeof(native[i].scalar));
        native[i].has_axes = nodes[i].has_axes != 0;
        native[i].naxes = nodes[i].naxes;
        if (nodes[i].naxes && nodes[i].naxes <= B2ND_EXEC_MAX_RANK) {
            memcpy(native[i].axes, nodes[i].axes, nodes[i].naxes * sizeof(int));
        }
        native[i].keepdims = nodes[i].keepdims != 0;
        native[i].has_initial = nodes[i].has_initial != 0;
        native[i].initial_dtype = (me_dtype)nodes[i].initial_dtype;
        native[i].initial_category = nodes[i].initial_weak ? ME_CAPTURE_WEAK : ME_CAPTURE_TYPED;
        native[i].initial_size = nodes[i].initial_size;
        memcpy(native[i].initial_scalar, nodes[i].initial_scalar, sizeof(native[i].initial_scalar));
    }
    b2nd_exec_descriptor descriptor = {sizeof(descriptor), native, count, ninputs, root};
    b2nd_exec_options options = B2ND_EXEC_OPTIONS_INITIALIZER;
    options.execution = (me_execution_policy)execution;
    b2nd_exec_plan *plan = NULL;
    int rc = b2nd_exec_prepare(&descriptor, &options, &plan, error);
    free(native);
    *out = plan;
    return rc;
}
static void b2_exec_free(void *plan) { b2nd_exec_plan_free(plan); }
static void b2_exec_schedule_free(void *schedule) { b2nd_exec_schedule_free(schedule); }
static int b2_exec_describe(const void *plan, b2_exec_info *out) {
    b2nd_exec_plan_info info;
    int rc = b2nd_exec_describe(plan, &info);
    if (!rc) {
        memset(out, 0, sizeof(*out));
        out->dtype = info.dtype;
        out->map_dtype = info.map_dtype;
        out->itemsize = info.itemsize;
        out->ninputs = info.ninputs;
        out->identity = info.identity;
        out->reduction = info.reduction;
        out->reducer = (int)info.reducer;
        out->partitionable = info.partitionable;
        out->has_jit = info.has_jit;
        out->plan_metadata_bytes = info.plan_metadata_bytes;
        out->has_axes = info.has_axes;
        out->naxes = info.naxes;
        if (info.naxes && info.naxes <= B2ND_EXEC_MAX_RANK) {
            memcpy(out->axes, info.axes, info.naxes * sizeof(int));
        }
        out->keepdims = info.keepdims;
        out->has_initial = info.has_initial;
    }
    return rc;
}
static int b2_exec_specialize(const void *plan, const b2_exec_meta *inputs, size_t ninputs,
                              size_t tile, size_t budget, int pairwise_sum, void **out,
                              me_error *error) {
    *out = NULL;
    if (ninputs > B2ND_EXEC_MAX_INPUTS) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_exec_input_meta metadata[B2ND_EXEC_MAX_INPUTS] = {0};
    for (size_t i = 0; i < ninputs; i++) {
        metadata[i].dtype = (me_dtype)inputs[i].dtype;
        metadata[i].itemsize = inputs[i].itemsize;
        metadata[i].ndim = (size_t)inputs[i].ndim;
        memcpy(metadata[i].shape, inputs[i].shape, sizeof(metadata[i].shape));
    }
    b2nd_exec_schedule *schedule = NULL;
    int rc = b2nd_exec_specialize_ex(plan, metadata, ninputs, tile, budget, pairwise_sum != 0,
                                     &schedule, error);
    *out = schedule;
    return rc;
}
static int b2_exec_schedule_describe(const void *schedule, b2_exec_info *out) {
    b2nd_exec_schedule_info info;
    int rc = b2nd_exec_describe_schedule(schedule, &info);
    if (!rc) {
        out->nitems = info.nitems;
        out->tile_items = info.tile_items;
        out->mapped_bytes = info.mapped_bytes;
        out->reducer_bytes = info.reducer_bytes;
        out->output_items = info.output_items;
        out->schedule_metadata_bytes = info.schedule_metadata_bytes;
        out->ndim = info.ndim;
        out->broadcast_gather_bytes = info.broadcast_gather_bytes;
        memcpy(out->shape, info.shape, sizeof(out->shape));
        out->output_ndim = info.output_ndim;
        memcpy(out->output_shape, info.output_shape, sizeof(out->output_shape));
        out->reduction_groups = info.reduction_groups;
        out->reduction_lanes = info.reduction_lanes;
        out->pairwise_sum = info.pairwise_sum;
    }
    return rc;
}
static void b2_exec_fill_views(b2nd_exec_view *views, const b2_exec_view *inputs, size_t ninputs,
                               b2nd_exec_output_view *out_view, const b2_exec_output_view *output) {
    for (size_t i = 0; i < ninputs; i++) {
        views[i].struct_size = sizeof(views[i]);
        views[i].base = inputs[i].base;
        views[i].capacity = inputs[i].capacity;
        views[i].offset = inputs[i].offset;
        views[i].byte_order = inputs[i].byte_order;
        memcpy(views[i].strides, inputs[i].strides, sizeof(views[i].strides));
    }
    out_view->struct_size = sizeof(*out_view);
    out_view->base = output->base;
    out_view->capacity = output->capacity;
    out_view->offset = output->offset;
    memcpy(out_view->strides, output->strides, sizeof(out_view->strides));
}
static void b2_exec_fill_report(b2_exec_report *report, const b2nd_exec_report *native) {
    report->tiles = native->tiles;
    report->map_calls = native->map_calls;
    report->direct_tiles = native->direct_tiles;
    report->mapped_bytes = native->mapped_bytes;
    report->fp = native->fp;
    report->tasks = native->tasks;
    report->worker_bytes = native->worker_bytes;
    report->gather_bytes = native->gather_bytes;
    report->gathered_bytes = native->gathered_bytes;
    report->staging_bytes = native->staging_bytes;
    report->interpreter_calls = native->interpreter_calls;
    report->jit_calls = native->jit_calls;
}
static int b2_exec_execute_into(const void *schedule, const b2_exec_view *inputs, size_t ninputs,
                                const b2_exec_output_view *output, unsigned raise_mask,
                                size_t max_gather_bytes, size_t workers, size_t worker_scratch,
                                b2_exec_report *report, me_error *error) {
    if (ninputs > B2ND_EXEC_MAX_INPUTS || !output) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_exec_view views[B2ND_EXEC_MAX_INPUTS] = {0};
    b2nd_exec_output_view out_view = {0};
    b2_exec_fill_views(views, inputs, ninputs, &out_view, output);
    b2nd_exec_report native = {0};
    b2nd_exec_jobs jobs = {sizeof(jobs), workers ? workers : 1, worker_scratch};
    int rc = b2nd_exec_execute_into(schedule, views, ninputs, &out_view, raise_mask,
                                    max_gather_bytes, workers > 1 ? &jobs : NULL, &native, error);
    b2_exec_fill_report(report, &native);
    return rc;
}
static int b2_exec_execute_alias(const void *schedule, const b2_exec_view *inputs, size_t ninputs,
                                 const b2_exec_output_view *output, size_t staging_budget,
                                 unsigned raise_mask, size_t max_gather_bytes,
                                 size_t workers, size_t worker_scratch,
                                 b2_exec_report *report, me_error *error) {
    if (ninputs > B2ND_EXEC_MAX_INPUTS || !output) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_exec_view views[B2ND_EXEC_MAX_INPUTS] = {0};
    b2nd_exec_output_view out_view = {0};
    b2_exec_fill_views(views, inputs, ninputs, &out_view, output);
    b2nd_exec_report native = {0};
    b2nd_exec_jobs jobs = {sizeof(jobs), workers ? workers : 1, worker_scratch};
    int rc = b2nd_exec_execute_alias(schedule, views, ninputs, &out_view, staging_budget, raise_mask,
                                     max_gather_bytes, workers > 1 ? &jobs : NULL, &native, error);
    b2_exec_fill_report(report, &native);
    return rc;
}
#else
typedef union {
    long double alignment;
    unsigned char storage[256];
} b2_exec_fp_scope;
static int b2_exec_fp_begin(b2_exec_fp_scope *scope, me_fp_status *status, me_error *error) {
    (void)scope; (void)status; (void)error;
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_fp_end(b2_exec_fp_scope *scope, me_error *error) {
    (void)scope; (void)error;
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_available(void) { return 0; }
static int b2_exec_prepare(const b2_exec_node *nodes, size_t count, size_t ninputs,
                           uint32_t root, int execution, void **out, me_error *error) {
    (void)nodes; (void)count; (void)ninputs; (void)root; (void)execution; (void)error;
    *out = NULL;
    return ME_ERR_UNSUPPORTED;
}
static void b2_exec_free(void *plan) { (void)plan; }
static void b2_exec_schedule_free(void *schedule) { (void)schedule; }
static int b2_exec_describe(const void *plan, b2_exec_info *out) {
    (void)plan; (void)out; return ME_ERR_UNSUPPORTED;
}
static int b2_exec_specialize(const void *plan, const b2_exec_meta *inputs, size_t ninputs,
                              size_t tile, size_t budget, int pairwise_sum, void **out,
                              me_error *error) {
    (void)plan; (void)inputs; (void)ninputs; (void)tile; (void)budget;
    (void)pairwise_sum; (void)error;
    *out = NULL;
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_schedule_describe(const void *schedule, b2_exec_info *out) {
    (void)schedule; (void)out; return ME_ERR_UNSUPPORTED;
}
static int b2_exec_execute_into(const void *schedule, const b2_exec_view *inputs, size_t ninputs,
                                const b2_exec_output_view *output, unsigned raise_mask,
                                size_t max_gather_bytes, size_t workers, size_t worker_scratch,
                                b2_exec_report *report, me_error *error) {
    (void)schedule; (void)inputs; (void)ninputs; (void)output; (void)raise_mask;
    (void)max_gather_bytes; (void)workers; (void)worker_scratch; (void)report; (void)error;
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_execute_alias(const void *schedule, const b2_exec_view *inputs, size_t ninputs,
                                 const b2_exec_output_view *output, size_t staging_budget,
                                 unsigned raise_mask, size_t max_gather_bytes,
                                 size_t workers, size_t worker_scratch,
                                 b2_exec_report *report, me_error *error) {
    (void)schedule; (void)inputs; (void)ninputs; (void)output; (void)staging_budget;
    (void)raise_mask; (void)max_gather_bytes; (void)workers; (void)worker_scratch;
    (void)report; (void)error;
    return ME_ERR_UNSUPPORTED;
}
#endif
#endif
