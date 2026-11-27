/* Private optional typed-plan bridge. Standard builds retain the adapter;
 * this bridge never invokes it, including on preparation/execution failure. */
#ifndef B2_EXEC_BRIDGE_H
#define B2_EXEC_BRIDGE_H
#include "miniexpr_kernel.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    int op, dtype, slot, args[3], nargs, weak;
    size_t itemsize, scalar_size;
    unsigned char scalar[16];
    int has_axes, keepdims, has_initial, initial_dtype, initial_weak;
    size_t naxes, initial_size;
    int axes[16];
    unsigned char initial_scalar[16];
    double ddof;
    char function[32];
    int has_where, where;
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
    int has_axes, keepdims, has_initial, pairwise_sum, has_where;
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
    int has_axes, keepdims, has_initial, initial_dtype, initial_weak;
    size_t naxes, initial_size;
    int axes[16];
    unsigned char initial_scalar[16];
} b2_exec_reduce_opts;

typedef struct {
    size_t ndim;
    int64_t domain[16];
    int64_t partitions[16];
    int has_selection;
    int64_t sel_start[16];
    int64_t sel_stop[16];
} b2_exec_partition_spec;

typedef struct {
    b2_exec_report execution;
    size_t decode_bytes, storage_metadata_bytes, decodes, decoded_bytes, cache_hits, read_spans;
    size_t evaluated_groups, max_group_items;
} b2_exec_storage_report;

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

typedef int (*b2_exec_source_read_fn)(void *context, const size_t *indices, size_t count,
                                      void *dst, size_t dst_capacity, me_error *error);
typedef struct {
    int dtype, ndim;
    size_t itemsize;
    int64_t shape[16];
    void *context;
    b2_exec_source_read_fn read;
} b2_exec_source;

typedef int (*b2_exec_sink_write_fn)(void *context, size_t offset_items, size_t count,
                                      const void *src, size_t src_capacity, me_error *error);
typedef struct {
    int dtype, ndim;
    size_t itemsize;
    int64_t shape[16];
    void *context;
    b2_exec_sink_write_fn write;
} b2_exec_sink;

typedef struct {
    b2_exec_report execution;
    size_t read_calls, read_items, max_read_items, read_bytes;
    size_t write_calls, write_items, max_write_items, written_bytes;
    size_t index_buffer_bytes, output_buffer_bytes;
} b2_exec_callback_report;

#ifdef B2_HAVE_B2ND_EXEC
#include "b2nd-exec.h"
#include "b2nd-exec-storage.h"
#include "b2nd-exec-stages.h"
#ifdef B2_HAVE_MENUDET_ARTIFACT
#include "miniexpr_artifact.h"
#endif
typedef me_fp_scope b2_exec_fp_scope;

static int b2_exec_fp_begin(b2_exec_fp_scope *scope, me_fp_status *status, me_error *error) {
    return me_fp_scope_begin(scope, status, error);
}
static int b2_exec_fp_end(b2_exec_fp_scope *scope, me_error *error) {
    return me_fp_scope_end(scope, error);
}

static int b2_exec_available(void) { return 1; }
static int b2_exec_prepare_impl(const b2_exec_node *nodes, size_t count, size_t ninputs,
                            uint32_t root, int execution, int compiler, int math_backend,
                            int integer_powers, void **out, size_t *failed_stage,
                            uint32_t *failed_node, me_error *error, int staged) {
    *out = NULL;
    if (failed_stage) *failed_stage = SIZE_MAX;
    if (failed_node) *failed_node = UINT32_MAX;
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
        native[i].args[2] = (uint32_t)nodes[i].args[2];
        memcpy(native[i].function, nodes[i].function, sizeof(native[i].function));
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
        native[i].ddof = nodes[i].ddof;
        native[i].has_where = nodes[i].has_where != 0;
        native[i].where = (uint32_t)nodes[i].where;
    }
    b2nd_exec_descriptor descriptor = {sizeof(descriptor), native, count, ninputs, root};
    b2nd_exec_options options = B2ND_EXEC_OPTIONS_INITIALIZER;
    options.execution = (me_execution_policy)execution;
    options.jit_compiler = (me_jit_compiler)compiler;
    options.math_backend = (me_math_backend)math_backend;
    options.integer_powers = integer_powers != 0;
    int rc;
    if (staged) {
        b2nd_exec_staged_plan *plan = NULL;
        b2nd_exec_stage_diagnostic diag = {SIZE_MAX, UINT32_MAX};
        rc = b2nd_exec_staged_prepare_ex(&descriptor, &options, &plan, &diag, error);
        if (failed_stage) *failed_stage = diag.stage;
        if (failed_node) *failed_node = diag.node;
        *out = plan;
    } else {
        b2nd_exec_plan *plan = NULL;
        rc = b2nd_exec_prepare(&descriptor, &options, &plan, error);
        *out = plan;
    }
    free(native);
    return rc;
}
static int b2_exec_prepare(const b2_exec_node *nodes, size_t count, size_t ninputs,
                           uint32_t root, int execution, int compiler, int math_backend,
                           int integer_powers, void **out, me_error *error) {
    return b2_exec_prepare_impl(nodes, count, ninputs, root, execution, compiler, math_backend,
                                integer_powers, out, NULL, NULL, error, 0);
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
        out->has_where = info.has_where;
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
static void b2_exec_fill_storage_report(b2_exec_storage_report *report,
                                        const b2nd_exec_storage_report *native) {
    b2_exec_fill_report(&report->execution, &native->execution);
    report->decode_bytes = native->decode_bytes;
    report->storage_metadata_bytes = native->storage_metadata_bytes;
    report->decodes = native->decodes;
    report->decoded_bytes = native->decoded_bytes;
    report->cache_hits = native->cache_hits;
    report->read_spans = native->read_spans;
    report->evaluated_groups = native->evaluated_groups;
    report->max_group_items = native->max_group_items;
}
static int b2_exec_prepare_storage_views(const b2_exec_view *views, size_t ninputs,
                                         b2nd_exec_view *native_views) {
    if (ninputs > B2ND_EXEC_MAX_INPUTS) {
        return ME_ERR_INVALID_ARG;
    }
    if (views) {
        for (size_t i = 0; i < ninputs; i++) {
            native_views[i].struct_size = sizeof(native_views[i]);
            native_views[i].base = views[i].base;
            native_views[i].capacity = views[i].capacity;
            native_views[i].offset = views[i].offset;
            native_views[i].byte_order = views[i].byte_order;
            memcpy(native_views[i].strides, views[i].strides, sizeof(native_views[i].strides));
        }
    }
    return ME_SUCCESS;
}
static int b2_exec_execute_arrays(const void *schedule, const void *const *inputs,
                                  const b2_exec_view *views, size_t ninputs,
                                  void *output, size_t output_bytes, unsigned raise_mask,
                                  size_t decode_budget, size_t gather_budget,
                                  size_t workers, int has_worker_scratch, size_t worker_scratch,
                                  b2_exec_storage_report *report, me_error *error) {
    b2nd_exec_view native_views[B2ND_EXEC_MAX_INPUTS] = {0};
    int prep = b2_exec_prepare_storage_views(views, ninputs, native_views);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_exec_storage_report native = {0};
    b2nd_exec_jobs jobs = {sizeof(jobs), workers ? workers : 1, worker_scratch};
    int rc = b2nd_exec_execute_arrays_ex(
        schedule, (const b2nd_array_t *const *)inputs, views ? native_views : NULL, ninputs,
        (me_mut_buffer){output, output_bytes}, raise_mask, decode_budget, gather_budget,
        (workers > 1 || has_worker_scratch) ? &jobs : NULL, &native, error);
    b2_exec_fill_storage_report(report, &native);
    return rc;
}
static int b2_exec_execute_to_array(const void *schedule, const void *const *inputs,
                                    const b2_exec_view *views, size_t ninputs,
                                    void *output_array, unsigned raise_mask,
                                    size_t decode_budget, size_t gather_budget,
                                    size_t workers, int has_worker_scratch, size_t worker_scratch,
                                    b2_exec_storage_report *report, me_error *error) {
    b2nd_exec_view native_views[B2ND_EXEC_MAX_INPUTS] = {0};
    int prep = b2_exec_prepare_storage_views(views, ninputs, native_views);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_exec_storage_report native = {0};
    b2nd_exec_jobs jobs = {sizeof(jobs), workers ? workers : 1, worker_scratch};
    int rc = b2nd_exec_execute_to_array(
        schedule, (const b2nd_array_t *const *)inputs, views ? native_views : NULL, ninputs,
        (b2nd_array_t *)output_array, raise_mask, decode_budget, gather_budget,
        (workers > 1 || has_worker_scratch) ? &jobs : NULL, &native, error);
    b2_exec_fill_storage_report(report, &native);
    return rc;
}
static int b2_exec_execute_into(const void *schedule, const b2_exec_view *inputs, size_t ninputs,
                                const b2_exec_output_view *output, unsigned raise_mask,
                                size_t max_gather_bytes, size_t workers,
                                int has_worker_scratch, size_t worker_scratch,
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
                                    max_gather_bytes,
                                    (workers > 1 || has_worker_scratch) ? &jobs : NULL,
                                    &native, error);
    b2_exec_fill_report(report, &native);
    return rc;
}
static int b2_exec_execute_alias(const void *schedule, const b2_exec_view *inputs, size_t ninputs,
                                 const b2_exec_output_view *output, size_t staging_budget,
                                 unsigned raise_mask, size_t max_gather_bytes,
                                 size_t workers, int has_worker_scratch, size_t worker_scratch,
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
                                     max_gather_bytes,
                                     (workers > 1 || has_worker_scratch) ? &jobs : NULL,
                                     &native, error);
    b2_exec_fill_report(report, &native);
    return rc;
}
typedef struct {
    const b2_exec_source *src;
    const b2_exec_sink *sink;
} b2_exec_cb_wrap;
static me_status b2_exec_cb_source_trampoline(void *context, const size_t *indices, size_t count,
                                              me_mut_buffer dst, me_error *error) {
    const b2_exec_source *src = context;
    return (me_status)src->read(src->context, indices, count, dst.data, dst.capacity, error);
}
static me_status b2_exec_cb_sink_trampoline(void *context, size_t offset_items, size_t count,
                                            me_buffer src, me_error *error) {
    const b2_exec_sink *sink = context;
    return (me_status)sink->write(sink->context, offset_items, count, src.data, src.capacity, error);
}
static void b2_exec_fill_cb_report(b2_exec_callback_report *report,
                                   const b2nd_exec_callback_report *native) {
    b2_exec_fill_report(&report->execution, &native->execution);
    report->read_calls = native->read_calls;
    report->read_items = native->read_items;
    report->max_read_items = native->max_read_items;
    report->read_bytes = native->read_bytes;
    report->write_calls = native->write_calls;
    report->write_items = native->write_items;
    report->max_write_items = native->max_write_items;
    report->written_bytes = native->written_bytes;
    report->index_buffer_bytes = native->index_buffer_bytes;
    report->output_buffer_bytes = native->output_buffer_bytes;
}
static int b2_exec_prepare_cb_args(const b2_exec_source *sources, const b2_exec_view *views,
                                   size_t ninputs, b2nd_exec_source *native_sources,
                                   b2nd_exec_view *native_views) {
    if ((!sources && ninputs) || ninputs > B2ND_EXEC_MAX_INPUTS) {
        return ME_ERR_INVALID_ARG;
    }
    int prep = b2_exec_prepare_storage_views(views, ninputs, native_views);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    for (size_t i = 0; i < ninputs; i++) {
        native_sources[i].struct_size = sizeof(native_sources[i]);
        native_sources[i].dtype = (me_dtype)sources[i].dtype;
        native_sources[i].itemsize = sources[i].itemsize;
        native_sources[i].ndim = (size_t)sources[i].ndim;
        if (native_sources[i].ndim <= B2ND_EXEC_MAX_RANK) {
            memcpy(native_sources[i].shape, sources[i].shape, sizeof(native_sources[i].shape));
        }
        if (sources[i].read) {
            native_sources[i].context = (void *)&sources[i];
            native_sources[i].read = b2_exec_cb_source_trampoline;
        }
    }
    return ME_SUCCESS;
}
static int b2_exec_execute_sources(const void *schedule, const b2_exec_source *sources,
                                   const b2_exec_view *views, size_t ninputs,
                                   void *output, size_t output_bytes,
                                   const b2_exec_sink *sink, unsigned raise_mask,
                                   size_t max_gather_bytes, size_t max_output_bytes,
                                   b2_exec_callback_report *report, me_error *error) {
    b2nd_exec_view native_views[B2ND_EXEC_MAX_INPUTS] = {0};
    b2nd_exec_source native_sources[B2ND_EXEC_MAX_INPUTS] = {0};
    int prep = b2_exec_prepare_cb_args(sources, views, ninputs, native_sources, native_views);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_exec_sink native_sink = {0};
    if (sink) {
        native_sink.struct_size = sizeof(native_sink);
        native_sink.dtype = (me_dtype)sink->dtype;
        native_sink.itemsize = sink->itemsize;
        native_sink.ndim = (size_t)sink->ndim;
        if (native_sink.ndim <= B2ND_EXEC_MAX_RANK) {
            memcpy(native_sink.shape, sink->shape, sizeof(native_sink.shape));
        }
        native_sink.context = (void *)sink;
        native_sink.write = sink->write ? b2_exec_cb_sink_trampoline : NULL;
    }
    b2nd_exec_callback_report native = {0};
    int rc = b2nd_exec_execute_sources(
        schedule, native_sources, views ? native_views : NULL, ninputs,
        (me_mut_buffer){output, output_bytes}, sink ? &native_sink : NULL,
        raise_mask, max_gather_bytes, max_output_bytes, &native, error);
    b2_exec_fill_cb_report(report, &native);
    return rc;
}
static int b2_exec_execute_sources_into(const void *schedule, const b2_exec_source *sources,
                                        const b2_exec_view *views, size_t ninputs,
                                        const b2_exec_output_view *output, unsigned raise_mask,
                                        size_t max_gather_bytes,
                                        b2_exec_callback_report *report, me_error *error) {
    if (!output) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_exec_view native_views[B2ND_EXEC_MAX_INPUTS] = {0};
    b2nd_exec_source native_sources[B2ND_EXEC_MAX_INPUTS] = {0};
    int prep = b2_exec_prepare_cb_args(sources, views, ninputs, native_sources, native_views);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_exec_output_view out_view = {sizeof(out_view), output->base, output->capacity, output->offset, {0}};
    memcpy(out_view.strides, output->strides, sizeof(out_view.strides));
    b2nd_exec_callback_report native = {0};
    int rc = b2nd_exec_execute_sources_into(
        schedule, native_sources, views ? native_views : NULL, ninputs,
        &out_view, raise_mask, max_gather_bytes, &native, error);
    b2_exec_fill_cb_report(report, &native);
    return rc;
}
static int b2_exec_artifact_has_core(const void *artifact) {
#ifdef B2_HAVE_MENUDET_ARTIFACT
    return me_artifact_borrow_core((const me_artifact *)artifact) != NULL;
#else
    (void)artifact;
    return 0;
#endif
}
static int b2_exec_prepare_artifact_kernel(const void *artifact, int reducer,
                                           const b2_exec_reduce_opts *opts,
                                           void **out, me_error *error) {
    *out = NULL;
#ifdef B2_HAVE_MENUDET_ARTIFACT
    const me_kernel *kernel = me_artifact_borrow_core((const me_artifact *)artifact);
    if (!kernel) {
        return ME_ERR_UNSUPPORTED;
    }
    b2nd_exec_reduce_opts native_opts = {0};
    if (opts) {
        native_opts.struct_size = sizeof(native_opts);
        native_opts.has_axes = opts->has_axes != 0;
        native_opts.naxes = opts->naxes;
        if (opts->naxes && opts->naxes <= B2ND_EXEC_MAX_RANK) {
            memcpy(native_opts.axes, opts->axes, opts->naxes * sizeof(int));
        }
        native_opts.keepdims = opts->keepdims != 0;
        native_opts.has_initial = opts->has_initial != 0;
        native_opts.initial_dtype = (me_dtype)opts->initial_dtype;
        native_opts.initial_category = opts->initial_weak ? ME_CAPTURE_WEAK : ME_CAPTURE_TYPED;
        native_opts.initial_size = opts->initial_size;
        memcpy(native_opts.initial_scalar, opts->initial_scalar, sizeof(native_opts.initial_scalar));
    }
    b2nd_exec_plan *plan = NULL;
    int rc = b2nd_exec_prepare_borrowed_kernel(kernel, (me_reduction_kind)reducer,
                                               opts ? &native_opts : NULL, &plan, error);
    *out = plan;
    return rc;
#else
    (void)artifact; (void)reducer; (void)opts; (void)error;
    return ME_ERR_UNSUPPORTED;
#endif
}
static int b2_exec_specialize_partitions(const void *plan, const b2_exec_meta *inputs,
                                         size_t ninputs, const b2_exec_partition_spec *spec,
                                         size_t max_group_bytes, void **out, me_error *error) {
    *out = NULL;
    if (!spec || ninputs > B2ND_EXEC_MAX_INPUTS) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_exec_input_meta metadata[B2ND_EXEC_MAX_INPUTS] = {0};
    for (size_t i = 0; i < ninputs; i++) {
        metadata[i].dtype = (me_dtype)inputs[i].dtype;
        metadata[i].itemsize = inputs[i].itemsize;
        metadata[i].ndim = (size_t)inputs[i].ndim;
        if (metadata[i].ndim <= B2ND_EXEC_MAX_RANK) {
            memcpy(metadata[i].shape, inputs[i].shape, sizeof(metadata[i].shape));
        }
    }
    b2nd_exec_partition_spec native_spec = {0};
    native_spec.struct_size = sizeof(native_spec);
    native_spec.ndim = spec->ndim;
    if (spec->ndim <= B2ND_EXEC_MAX_RANK) {
        memcpy(native_spec.domain, spec->domain, sizeof(native_spec.domain));
        memcpy(native_spec.partitions, spec->partitions, sizeof(native_spec.partitions));
        memcpy(native_spec.sel_start, spec->sel_start, sizeof(native_spec.sel_start));
        memcpy(native_spec.sel_stop, spec->sel_stop, sizeof(native_spec.sel_stop));
    }
    native_spec.has_selection = spec->has_selection != 0;
    b2nd_exec_schedule *schedule = NULL;
    int rc = b2nd_exec_specialize_partitions(
        plan, ninputs ? metadata : NULL, ninputs, &native_spec, max_group_bytes, &schedule, error);
    *out = schedule;
    return rc;
}
static int b2_exec_partitions_impl(const void *schedule, const void *const *arrays,
                                      const b2_exec_view *views, const b2_exec_source *sources,
                                      size_t ninputs, void *output, size_t output_bytes,
                                      unsigned raise_mask, size_t max_decode_bytes,
                                      size_t max_gather_bytes, b2_exec_storage_report *report,
                                      me_error *error, int *admitted) {
    if (ninputs > B2ND_EXEC_MAX_INPUTS) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_exec_view native_views[B2ND_EXEC_MAX_INPUTS] = {0};
    int prep = b2_exec_prepare_storage_views(views, ninputs, native_views);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_exec_source native_sources[B2ND_EXEC_MAX_INPUTS] = {0};
    if (sources) {
        for (size_t i = 0; i < ninputs; i++) {
            native_sources[i].struct_size = sizeof(native_sources[i]);
            native_sources[i].dtype = (me_dtype)sources[i].dtype;
            native_sources[i].itemsize = sources[i].itemsize;
            native_sources[i].ndim = (size_t)sources[i].ndim;
            if (native_sources[i].ndim <= B2ND_EXEC_MAX_RANK) {
                memcpy(native_sources[i].shape, sources[i].shape, sizeof(native_sources[i].shape));
            }
            if (sources[i].read) {
                native_sources[i].context = (void *)&sources[i];
                native_sources[i].read = b2_exec_cb_source_trampoline;
            }
        }
    }
    if (admitted) {
        bool native_admitted = false;
        int rc = b2nd_exec_admit_partitions(
            schedule, (const b2nd_array_t *const *)arrays, views ? native_views : NULL,
            sources ? native_sources : NULL, ninputs, &native_admitted, error);
        *admitted = native_admitted;
        return rc;
    }
    b2nd_exec_storage_report native = {0};
    int rc = b2nd_exec_execute_partitions(
        schedule, (const b2nd_array_t *const *)arrays, views ? native_views : NULL,
        sources ? native_sources : NULL, ninputs, (me_mut_buffer){output, output_bytes},
        raise_mask, max_decode_bytes, max_gather_bytes, &native, error);
    b2_exec_fill_storage_report(report, &native);
    return rc;
}
static int b2_exec_admit_partitions(const void *schedule, const void *const *arrays,
                                    const b2_exec_view *views, const b2_exec_source *sources,
                                    size_t ninputs, int *admitted, me_error *error) {
    return b2_exec_partitions_impl(schedule, arrays, views, sources, ninputs,
                                   NULL, 0, 0, 0, 0, NULL, error, admitted);
}
static int b2_exec_execute_partitions(const void *schedule, const void *const *arrays,
                                      const b2_exec_view *views, const b2_exec_source *sources,
                                      size_t ninputs, void *output, size_t output_bytes,
                                      unsigned raise_mask, size_t max_decode_bytes,
                                      size_t max_gather_bytes, b2_exec_storage_report *report,
                                      me_error *error) {
    return b2_exec_partitions_impl(schedule, arrays, views, sources, ninputs, output,
                                   output_bytes, raise_mask, max_decode_bytes, max_gather_bytes,
                                   report, error, NULL);
}
#else
static int b2_exec_admit_partitions(const void *schedule, const void *const *arrays,
                                    const b2_exec_view *views, const b2_exec_source *sources,
                                    size_t ninputs, int *admitted, me_error *error) {
    (void)schedule; (void)arrays; (void)views; (void)sources; (void)ninputs; (void)error;
    *admitted = 0;
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_execute_arrays(const void *schedule, const void *const *inputs,
                                  const b2_exec_view *views, size_t ninputs,
                                  void *output, size_t output_bytes, unsigned raise_mask,
                                  size_t decode_budget, size_t gather_budget,
                                  size_t workers, int has_worker_scratch, size_t worker_scratch,
                                  b2_exec_storage_report *report, me_error *error) {
    (void)schedule; (void)inputs; (void)views; (void)ninputs; (void)output; (void)output_bytes;
    (void)raise_mask; (void)decode_budget; (void)gather_budget; (void)workers;
    (void)has_worker_scratch; (void)worker_scratch; (void)error;
    memset(report, 0, sizeof(*report));
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_execute_to_array(const void *schedule, const void *const *inputs,
                                    const b2_exec_view *views, size_t ninputs,
                                    void *output_array, unsigned raise_mask,
                                    size_t decode_budget, size_t gather_budget,
                                    size_t workers, int has_worker_scratch, size_t worker_scratch,
                                    b2_exec_storage_report *report, me_error *error) {
    (void)schedule; (void)inputs; (void)views; (void)ninputs; (void)output_array;
    (void)raise_mask; (void)decode_budget; (void)gather_budget; (void)workers;
    (void)has_worker_scratch; (void)worker_scratch; (void)error;
    memset(report, 0, sizeof(*report));
    return ME_ERR_UNSUPPORTED;
}
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
                           uint32_t root, int execution, int compiler, int math_backend,
                           int integer_powers, void **out, me_error *error) {
    (void)nodes; (void)count; (void)ninputs; (void)root; (void)execution; (void)error;
    (void)compiler; (void)math_backend; (void)integer_powers;
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
                                size_t max_gather_bytes, size_t workers,
                                int has_worker_scratch, size_t worker_scratch,
                                b2_exec_report *report, me_error *error) {
    (void)schedule; (void)inputs; (void)ninputs; (void)output; (void)raise_mask;
    (void)max_gather_bytes; (void)workers; (void)has_worker_scratch;
    (void)worker_scratch; (void)report; (void)error;
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_execute_alias(const void *schedule, const b2_exec_view *inputs, size_t ninputs,
                                 const b2_exec_output_view *output, size_t staging_budget,
                                 unsigned raise_mask, size_t max_gather_bytes,
                                 size_t workers, int has_worker_scratch, size_t worker_scratch,
                                 b2_exec_report *report, me_error *error) {
    (void)schedule; (void)inputs; (void)ninputs; (void)output; (void)staging_budget;
    (void)raise_mask; (void)max_gather_bytes; (void)workers;
    (void)has_worker_scratch; (void)worker_scratch;
    (void)report; (void)error;
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_execute_sources(const void *schedule, const b2_exec_source *sources,
                                   const b2_exec_view *views, size_t ninputs,
                                   void *output, size_t output_bytes,
                                   const b2_exec_sink *sink, unsigned raise_mask,
                                   size_t max_gather_bytes, size_t max_output_bytes,
                                   b2_exec_callback_report *report, me_error *error) {
    (void)schedule; (void)sources; (void)views; (void)ninputs; (void)output; (void)output_bytes;
    (void)sink; (void)raise_mask; (void)max_gather_bytes; (void)max_output_bytes; (void)error;
    memset(report, 0, sizeof(*report));
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_execute_sources_into(const void *schedule, const b2_exec_source *sources,
                                        const b2_exec_view *views, size_t ninputs,
                                        const b2_exec_output_view *output, unsigned raise_mask,
                                        size_t max_gather_bytes,
                                        b2_exec_callback_report *report, me_error *error) {
    (void)schedule; (void)sources; (void)views; (void)ninputs; (void)output;
    (void)raise_mask; (void)max_gather_bytes; (void)error;
    memset(report, 0, sizeof(*report));
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_artifact_has_core(const void *artifact) {
    (void)artifact;
    return 0;
}
static int b2_exec_prepare_artifact_kernel(const void *artifact, int reducer,
                                           const b2_exec_reduce_opts *opts,
                                           void **out, me_error *error) {
    (void)artifact; (void)reducer; (void)opts; (void)error;
    *out = NULL;
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_specialize_partitions(const void *plan, const b2_exec_meta *inputs,
                                         size_t ninputs, const b2_exec_partition_spec *spec,
                                         size_t max_group_bytes, void **out, me_error *error) {
    (void)plan; (void)inputs; (void)ninputs; (void)spec; (void)max_group_bytes; (void)error;
    *out = NULL;
    return ME_ERR_UNSUPPORTED;
}
static int b2_exec_execute_partitions(const void *schedule, const void *const *arrays,
                                      const b2_exec_view *views, const b2_exec_source *sources,
                                      size_t ninputs, void *output, size_t output_bytes,
                                      unsigned raise_mask, size_t max_decode_bytes,
                                      size_t max_gather_bytes, b2_exec_storage_report *report,
                                      me_error *error) {
    (void)schedule; (void)arrays; (void)views; (void)sources; (void)ninputs;
    (void)output; (void)output_bytes; (void)raise_mask; (void)max_decode_bytes;
    (void)max_gather_bytes; (void)error;
    memset(report, 0, sizeof(*report));
    return ME_ERR_UNSUPPORTED;
}
#endif
#include "b2nd_exec_stages_bridge.h"
#endif
