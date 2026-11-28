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
    const char *fp_mode, *math_backend, *sleef_accuracy;
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
    int has_where, accumulator;
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

#include "dsl_artifact_bridge.h"
#ifdef B2_HAVE_B2ND_EXEC
#include "b2nd_exec.h"
#include "b2nd_exec_kernel.h"
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

static int b2_exec_available(void) {
    return b2nd_exec_available() ? 1 : 0;
}

static void b2_exec_copy_error(me_error *dst, const b2nd_exec_error_t *src) {
    if (!dst || !src) {
        return;
    }
    dst->line = src->line;
    dst->column = src->column;
    snprintf(dst->message, sizeof(dst->message), "%s", src->message);
}

static void b2_exec_fill_plan_info(b2_exec_info *out, const b2nd_plan_info_t *info) {
    memset(out, 0, sizeof(*out));
    out->dtype = (int)info->dtype;
    out->map_dtype = (int)info->map_dtype;
    out->itemsize = info->itemsize;
    out->ninputs = info->ninputs;
    out->identity = info->identity ? 1 : 0;
    out->reduction = info->reduction ? 1 : 0;
    out->reducer = info->reducer <= B2ND_REDUCTION_ALL ? (int)info->reducer : 0;
    out->partitionable = info->partitionable ? 1 : 0;
    out->has_jit = info->has_jit ? 1 : 0;
    out->fp_mode = info->fp_mode;
    out->math_backend = info->math_backend;
    out->sleef_accuracy = info->sleef_accuracy;
    out->plan_metadata_bytes = info->plan_metadata_bytes;
    out->has_axes = info->has_axes ? 1 : 0;
    out->naxes = info->naxes;
    if (info->naxes && info->naxes <= B2ND_EXEC_MAX_RANK) {
        memcpy(out->axes, info->axes, info->naxes * sizeof(int));
    }
    out->keepdims = info->keepdims ? 1 : 0;
    out->has_initial = info->has_initial ? 1 : 0;
    out->has_where = info->has_where ? 1 : 0;
}

static void b2_exec_fill_schedule_info(b2_exec_info *out, const b2nd_schedule_info_t *info) {
    out->nitems = info->nitems;
    out->tile_items = info->tile_items;
    out->mapped_bytes = info->mapped_bytes;
    out->reducer_bytes = info->reducer_bytes;
    out->output_items = info->output_items;
    out->schedule_metadata_bytes = info->schedule_metadata_bytes;
    out->ndim = info->ndim;
    out->broadcast_gather_bytes = info->broadcast_gather_bytes;
    memcpy(out->shape, info->shape, sizeof(out->shape));
    out->output_ndim = info->output_ndim;
    memcpy(out->output_shape, info->output_shape, sizeof(out->output_shape));
    out->reduction_groups = info->reduction_groups;
    out->reduction_lanes = info->reduction_lanes;
    out->pairwise_sum = info->pairwise_sum ? 1 : 0;
}

static int b2_exec_prepare_impl(const b2_exec_node *nodes, size_t count, size_t ninputs,
                                uint32_t root, int execution, int compiler, int math_backend,
                                int integer_powers, int fp_mode, int accuracy, void **out,
                                size_t *failed_stage, uint32_t *failed_node,
                                me_error *error, int staged) {
    *out = NULL;
    if (failed_stage) {
        *failed_stage = SIZE_MAX;
    }
    if (failed_node) {
        *failed_node = UINT32_MAX;
    }
    if (!count || count > B2ND_EXEC_MAX_NODES) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_exec_node_t *native = calloc(count, sizeof(*native));
    if (!native) {
        return ME_ERR_OOM;
    }
    for (size_t i = 0; i < count; i++) {
        native[i] = (b2nd_exec_node_t)B2ND_EXEC_NODE_INITIALIZER;
        native[i].id = (uint32_t)i;
        native[i].op = (b2nd_exec_op_t)nodes[i].op;
        native[i].dtype = (b2nd_exec_dtype_t)nodes[i].dtype;
        native[i].slot = (uint32_t)nodes[i].slot;
        native[i].nargs = (size_t)nodes[i].nargs;
        native[i].args[0] = (uint32_t)nodes[i].args[0];
        native[i].args[1] = (uint32_t)nodes[i].args[1];
        native[i].args[2] = (uint32_t)nodes[i].args[2];
        memcpy(native[i].function, nodes[i].function, sizeof(native[i].function));
        native[i].itemsize = nodes[i].itemsize;
        native[i].scalar_size = nodes[i].scalar_size;
        native[i].category = nodes[i].weak ? B2ND_CAPTURE_WEAK : B2ND_CAPTURE_TYPED;
        memcpy(native[i].scalar, nodes[i].scalar, sizeof(native[i].scalar));
        native[i].has_axes = nodes[i].has_axes != 0;
        native[i].naxes = nodes[i].naxes;
        if (nodes[i].naxes && nodes[i].naxes <= B2ND_EXEC_MAX_RANK) {
            memcpy(native[i].axes, nodes[i].axes, nodes[i].naxes * sizeof(int));
        }
        native[i].keepdims = nodes[i].keepdims != 0;
        native[i].has_initial = nodes[i].has_initial != 0;
        native[i].initial_dtype = (b2nd_exec_dtype_t)nodes[i].initial_dtype;
        native[i].initial_category = nodes[i].initial_weak
            ? B2ND_CAPTURE_WEAK : B2ND_CAPTURE_TYPED;
        native[i].initial_size = nodes[i].initial_size;
        memcpy(native[i].initial_scalar, nodes[i].initial_scalar,
               sizeof(native[i].initial_scalar));
        native[i].ddof = nodes[i].ddof;
        native[i].has_where = nodes[i].has_where != 0;
        native[i].where = (uint32_t)nodes[i].where;
    }
    b2nd_exec_graph_t graph = B2ND_EXEC_GRAPH_INITIALIZER;
    graph.nodes = native;
    graph.nnodes = count;
    graph.ninputs = ninputs;
    graph.root = root;
    graph.disable_auto_stages = staged == 0;
    b2nd_exec_options_t options = B2ND_EXEC_OPTIONS_INITIALIZER;
    options.execution = (b2nd_exec_policy_t)execution;
    options.compiler = (b2nd_compiler_t)compiler;
    options.math_backend = (b2nd_math_backend_t)math_backend;
    options.integer_powers = integer_powers != 0;
    options.fp_mode = (b2nd_fp_mode_t)fp_mode;
    options.math_accuracy = (b2nd_math_accuracy_t)accuracy;
    b2nd_plan_t *plan = NULL;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_plan_from_graph(&graph, &options, &plan, &pub_err);
    free(native);
    if (st != B2ND_EXEC_SUCCESS) {
        if (failed_stage) {
            *failed_stage = pub_err.stage;
        }
        if (failed_node) {
            *failed_node = pub_err.node;
        }
        b2_exec_copy_error(error, &pub_err);
    }
    *out = plan;
    return (int)st;
}

static int b2_exec_prepare(const b2_exec_node *nodes, size_t count, size_t ninputs,
                           uint32_t root, int execution, int compiler, int math_backend,
                           int integer_powers, int fp_mode, int accuracy,
                           void **out, me_error *error) {
    return b2_exec_prepare_impl(nodes, count, ninputs, root, execution, compiler, math_backend,
                                integer_powers, fp_mode, accuracy, out, NULL, NULL, error, 0);
}

static void b2_exec_free(void *plan) {
    b2nd_plan_free((b2nd_plan_t *)plan);
}

static void b2_exec_schedule_free(void *schedule) {
    b2nd_schedule_free((b2nd_schedule_t *)schedule);
}

static int b2_exec_describe(const void *plan, b2_exec_info *out) {
    b2nd_plan_info_t info = B2ND_PLAN_INFO_INITIALIZER;
    b2nd_exec_status_t st = b2nd_plan_describe((const b2nd_plan_t *)plan, &info);
    if (st == B2ND_EXEC_SUCCESS) {
        b2_exec_fill_plan_info(out, &info);
    }
    return (int)st;
}

static int b2_exec_convert_meta(const b2_exec_meta *inputs, size_t ninputs,
                                b2nd_exec_input_meta_t *metadata) {
    if (ninputs > B2ND_EXEC_MAX_INPUTS) {
        return ME_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < ninputs; i++) {
        metadata[i] = (b2nd_exec_input_meta_t)B2ND_EXEC_INPUT_META_INITIALIZER;
        metadata[i].dtype = (b2nd_exec_dtype_t)inputs[i].dtype;
        metadata[i].itemsize = inputs[i].itemsize;
        metadata[i].ndim = (size_t)inputs[i].ndim;
        if (metadata[i].ndim <= B2ND_EXEC_MAX_RANK) {
            memcpy(metadata[i].shape, inputs[i].shape, sizeof(metadata[i].shape));
        }
    }
    return ME_SUCCESS;
}

static int b2_exec_specialize(const void *plan, const b2_exec_meta *inputs, size_t ninputs,
                              size_t tile, size_t budget, int pairwise_sum, void **out,
                              me_error *error) {
    *out = NULL;
    b2nd_exec_input_meta_t metadata[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_convert_meta(inputs, ninputs, metadata);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_schedule_options_t sopts = B2ND_SCHEDULE_OPTIONS_INITIALIZER;
    sopts.tile_items = tile;
    sopts.max_mapped_bytes = budget;
    sopts.pairwise_sum = pairwise_sum != 0;
    b2nd_schedule_t *schedule = NULL;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_schedule_create(
        (const b2nd_plan_t *)plan, ninputs ? metadata : NULL, ninputs, &sopts,
        &schedule, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    *out = schedule;
    return (int)st;
}

static int b2_exec_schedule_describe(const void *schedule, b2_exec_info *out) {
    b2nd_schedule_info_t info = B2ND_SCHEDULE_INFO_INITIALIZER;
    b2nd_exec_status_t st = b2nd_schedule_describe((const b2nd_schedule_t *)schedule, &info, NULL, 0);
    if (st == B2ND_EXEC_SUCCESS) {
        b2_exec_fill_schedule_info(out, &info);
    }
    return (int)st;
}

typedef struct {
    const b2_exec_source *src;
    const b2_exec_sink *sink;
} b2_exec_cb_wrap;

static b2nd_exec_status_t b2_exec_cb_source_trampoline(void *context, const size_t *indices,
                                                      size_t count, void *dst, size_t dst_capacity,
                                                      b2nd_exec_error_t *error) {
    const b2_exec_cb_wrap *wrap = context;
    me_error me_err = {0};
    int rc = wrap->src->read(wrap->src->context, indices, count, dst, dst_capacity, &me_err);
    if (rc != ME_SUCCESS && error) {
        error->status = (b2nd_exec_status_t)rc;
        error->line = me_err.line;
        error->column = me_err.column;
        snprintf(error->message, sizeof(error->message), "%s", me_err.message);
    }
    return (b2nd_exec_status_t)rc;
}

static b2nd_exec_status_t b2_exec_cb_sink_trampoline(void *context, size_t offset_items,
                                                    size_t count, const void *src,
                                                    size_t src_capacity,
                                                    b2nd_exec_error_t *error) {
    const b2_exec_cb_wrap *wrap = context;
    me_error me_err = {0};
    int rc = wrap->sink->write(wrap->sink->context, offset_items, count, src, src_capacity, &me_err);
    if (rc != ME_SUCCESS && error) {
        error->status = (b2nd_exec_status_t)rc;
        error->line = me_err.line;
        error->column = me_err.column;
        snprintf(error->message, sizeof(error->message), "%s", me_err.message);
    }
    return (b2nd_exec_status_t)rc;
}

static void b2_exec_fill_report(b2_exec_report *report, const b2nd_exec_report_t *native) {
    report->tiles = native->tiles;
    report->map_calls = native->map_calls;
    report->direct_tiles = native->direct_tiles;
    report->mapped_bytes = native->mapped_bytes;
    report->fp.flags = native->fp.flags;
    report->fp.supported = native->fp.supported;
    report->tasks = native->tasks;
    report->worker_bytes = native->worker_bytes;
    report->gather_bytes = native->gather_bytes;
    report->gathered_bytes = native->gathered_bytes;
    report->staging_bytes = native->staging_bytes;
    report->interpreter_calls = native->interpreter_calls;
    report->jit_calls = native->jit_calls;
}

static void b2_exec_fill_storage_report(b2_exec_storage_report *report,
                                        const b2nd_exec_report_t *native) {
    b2_exec_fill_report(&report->execution, native);
    report->decode_bytes = native->decode_bytes;
    report->storage_metadata_bytes = native->storage_metadata_bytes;
    report->decodes = native->decodes;
    report->decoded_bytes = native->decoded_bytes;
    report->cache_hits = native->cache_hits;
    report->read_spans = native->read_spans;
    report->evaluated_groups = native->evaluated_groups;
    report->max_group_items = native->max_group_items;
}

static void b2_exec_fill_cb_report(b2_exec_callback_report *report,
                                   const b2nd_exec_report_t *native) {
    b2_exec_fill_report(&report->execution, native);
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

static void b2_exec_fill_view_operand(const b2_exec_view *v, b2nd_operand_t *op) {
    *op = (b2nd_operand_t)B2ND_OPERAND_INITIALIZER;
    op->kind = B2ND_OPERAND_VIEW;
    op->as.view.base = v->base;
    op->as.view.capacity = v->capacity;
    op->as.view.offset = v->offset;
    op->as.view.byte_order = (b2nd_byte_order_t)v->byte_order;
    memcpy(op->as.view.strides, v->strides, sizeof(op->as.view.strides));
}

static int b2_exec_fill_operands(const void *const *arrays, const b2_exec_view *views,
                                 const b2_exec_source *sources, size_t ninputs,
                                 b2nd_operand_t *ops, b2_exec_cb_wrap *wraps) {
    if (ninputs > B2ND_EXEC_MAX_INPUTS) {
        return ME_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < ninputs; i++) {
        ops[i] = (b2nd_operand_t)B2ND_OPERAND_INITIALIZER;
        if (arrays && arrays[i]) {
            ops[i].kind = B2ND_OPERAND_ARRAY;
            ops[i].as.array = (const b2nd_array_t *)arrays[i];
        } else if (sources && sources[i].read) {
            ops[i].kind = B2ND_OPERAND_SOURCE;
            ops[i].as.source.dtype = (b2nd_exec_dtype_t)sources[i].dtype;
            ops[i].as.source.itemsize = sources[i].itemsize;
            ops[i].as.source.ndim = (size_t)sources[i].ndim;
            if (ops[i].as.source.ndim <= B2ND_EXEC_MAX_RANK) {
                memcpy(ops[i].as.source.shape, sources[i].shape,
                       ops[i].as.source.ndim * sizeof(int64_t));
            }
            wraps[i].src = &sources[i];
            wraps[i].sink = NULL;
            ops[i].as.source.context = &wraps[i];
            ops[i].as.source.read = b2_exec_cb_source_trampoline;
        } else if (views) {
            b2_exec_fill_view_operand(&views[i], &ops[i]);
        } else {
            return ME_ERR_INVALID_ARG;
        }
    }
    return ME_SUCCESS;
}

static void b2_exec_init_view_output(const b2_exec_output_view *output, size_t staging_budget,
                                     int allow_alias, b2nd_output_t *out_op) {
    *out_op = (b2nd_output_t)B2ND_OUTPUT_INITIALIZER;
    out_op->kind = B2ND_OUTPUT_VIEW;
    out_op->as.view.base = output->base;
    out_op->as.view.capacity = output->capacity;
    out_op->as.view.offset = output->offset;
    memcpy(out_op->as.view.strides, output->strides, sizeof(out_op->as.view.strides));
    out_op->as.view.staging_budget = staging_budget;
    out_op->as.view.allow_alias = allow_alias != 0;
}

static int b2_exec_init_buffer_output(const void *schedule, void *output, size_t output_bytes,
                                      b2nd_output_t *out_op) {
    b2nd_schedule_info_t sinfo = B2ND_SCHEDULE_INFO_INITIALIZER;
    if (b2nd_schedule_describe((const b2nd_schedule_t *)schedule, &sinfo, NULL, 0) != B2ND_EXEC_SUCCESS) {
        return ME_ERR_INVALID_ARG;
    }
    if (!sinfo.output_itemsize || sinfo.output_ndim > B2ND_EXEC_MAX_RANK) {
        return ME_ERR_INVALID_ARG;
    }
    *out_op = (b2nd_output_t)B2ND_OUTPUT_INITIALIZER;
    out_op->kind = B2ND_OUTPUT_VIEW;
    out_op->as.view.base = output;
    out_op->as.view.capacity = output_bytes;
    out_op->as.view.offset = 0;
    size_t stride = sinfo.output_itemsize;
    for (size_t d = sinfo.output_ndim; d-- > 0;) {
        out_op->as.view.strides[d] = (int64_t)stride;
        if (sinfo.output_shape[d] > 0) {
            stride *= (size_t)sinfo.output_shape[d];
        }
    }
    return ME_SUCCESS;
}

static b2nd_run_options_t b2_exec_make_run_options(unsigned raise_mask, size_t decode_budget,
                                                   size_t gather_budget, size_t output_budget,
                                                   size_t workers, int has_worker_scratch,
                                                   size_t worker_scratch) {
    b2nd_run_options_t ropts = B2ND_RUN_OPTIONS_INITIALIZER;
    ropts.workers = workers ? workers : 1;
    ropts.has_worker_scratch = has_worker_scratch != 0;
    ropts.max_worker_scratch = worker_scratch;
    ropts.max_decode_bytes = decode_budget;
    ropts.max_gather_bytes = gather_budget;
    ropts.max_output_bytes = output_budget;
    ropts.raise_mask = raise_mask;
    return ropts;
}

static int b2_exec_execute_arrays(const void *schedule, const void *const *inputs,
                                  const b2_exec_view *views, size_t ninputs,
                                  void *output, size_t output_bytes, unsigned raise_mask,
                                  size_t decode_budget, size_t gather_budget,
                                  size_t workers, int has_worker_scratch, size_t worker_scratch,
                                  b2_exec_storage_report *report, me_error *error) {
    b2nd_operand_t ops[B2ND_EXEC_MAX_INPUTS];
    b2_exec_cb_wrap wraps[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_fill_operands(inputs, views, NULL, ninputs, ops, wraps);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_output_t out_op = B2ND_OUTPUT_INITIALIZER;
    prep = b2_exec_init_buffer_output(schedule, output, output_bytes, &out_op);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_run_options_t ropts = b2_exec_make_run_options(
        raise_mask, decode_budget, gather_budget, 0, workers, has_worker_scratch, worker_scratch);
    b2nd_exec_report_t pub_rep = B2ND_EXEC_REPORT_INITIALIZER;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_execute(
        (const b2nd_schedule_t *)schedule, ninputs ? ops : NULL, ninputs, &out_op,
        &ropts, &pub_rep, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    b2_exec_fill_storage_report(report, &pub_rep);
    return (int)st;
}

static int b2_exec_execute_to_array(const void *schedule, const void *const *inputs,
                                    const b2_exec_view *views, size_t ninputs,
                                    void *output_array, unsigned raise_mask,
                                    size_t decode_budget, size_t gather_budget,
                                    size_t workers, int has_worker_scratch, size_t worker_scratch,
                                    b2_exec_storage_report *report, me_error *error) {
    b2nd_operand_t ops[B2ND_EXEC_MAX_INPUTS];
    b2_exec_cb_wrap wraps[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_fill_operands(inputs, views, NULL, ninputs, ops, wraps);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_output_t out_op = B2ND_OUTPUT_INITIALIZER;
    out_op.kind = B2ND_OUTPUT_ARRAY;
    out_op.as.array = (b2nd_array_t *)output_array;
    b2nd_run_options_t ropts = b2_exec_make_run_options(
        raise_mask, decode_budget, gather_budget, 0, workers, has_worker_scratch, worker_scratch);
    b2nd_exec_report_t pub_rep = B2ND_EXEC_REPORT_INITIALIZER;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_execute(
        (const b2nd_schedule_t *)schedule, ninputs ? ops : NULL, ninputs, &out_op,
        &ropts, &pub_rep, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    b2_exec_fill_storage_report(report, &pub_rep);
    return (int)st;
}

static int b2_exec_execute_view_common(const void *schedule, const b2_exec_view *inputs,
                                       size_t ninputs, const b2_exec_output_view *output,
                                       size_t staging_budget, int allow_alias, unsigned raise_mask,
                                       size_t max_gather_bytes, size_t workers,
                                       int has_worker_scratch, size_t worker_scratch,
                                       b2_exec_report *report, me_error *error) {
    if (!output) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_operand_t ops[B2ND_EXEC_MAX_INPUTS];
    b2_exec_cb_wrap wraps[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_fill_operands(NULL, inputs, NULL, ninputs, ops, wraps);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_output_t out_op = B2ND_OUTPUT_INITIALIZER;
    b2_exec_init_view_output(output, staging_budget, allow_alias, &out_op);
    b2nd_run_options_t ropts = b2_exec_make_run_options(
        raise_mask, 0, max_gather_bytes, 0, workers, has_worker_scratch, worker_scratch);
    b2nd_exec_report_t pub_rep = B2ND_EXEC_REPORT_INITIALIZER;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_execute(
        (const b2nd_schedule_t *)schedule, ninputs ? ops : NULL, ninputs, &out_op,
        &ropts, &pub_rep, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    b2_exec_fill_report(report, &pub_rep);
    return (int)st;
}

static int b2_exec_execute_into(const void *schedule, const b2_exec_view *inputs, size_t ninputs,
                                const b2_exec_output_view *output, unsigned raise_mask,
                                size_t max_gather_bytes, size_t workers,
                                int has_worker_scratch, size_t worker_scratch,
                                b2_exec_report *report, me_error *error) {
    return b2_exec_execute_view_common(
        schedule, inputs, ninputs, output, 0, 0, raise_mask, max_gather_bytes,
        workers, has_worker_scratch, worker_scratch, report, error);
}

static int b2_exec_execute_alias(const void *schedule, const b2_exec_view *inputs, size_t ninputs,
                                 const b2_exec_output_view *output, size_t staging_budget,
                                 unsigned raise_mask, size_t max_gather_bytes,
                                 size_t workers, int has_worker_scratch, size_t worker_scratch,
                                 b2_exec_report *report, me_error *error) {
    return b2_exec_execute_view_common(
        schedule, inputs, ninputs, output, staging_budget, 1, raise_mask, max_gather_bytes,
        workers, has_worker_scratch, worker_scratch, report, error);
}

static int b2_exec_execute_sources(const void *schedule, const b2_exec_source *sources,
                                   const b2_exec_view *views, size_t ninputs,
                                   void *output, size_t output_bytes,
                                   const b2_exec_sink *sink, unsigned raise_mask,
                                   size_t max_gather_bytes, size_t max_output_bytes,
                                   b2_exec_callback_report *report, me_error *error) {
    if ((!sources && !views && ninputs) || (!sink && !output && output_bytes)) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_operand_t ops[B2ND_EXEC_MAX_INPUTS];
    b2_exec_cb_wrap wraps[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_fill_operands(NULL, views, sources, ninputs, ops, wraps);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2_exec_cb_wrap sink_wrap = {NULL, sink};
    b2nd_output_t out_op = B2ND_OUTPUT_INITIALIZER;
    if (sink && sink->write) {
        out_op.kind = B2ND_OUTPUT_SINK;
        out_op.as.sink.dtype = (b2nd_exec_dtype_t)sink->dtype;
        out_op.as.sink.itemsize = sink->itemsize;
        out_op.as.sink.ndim = (size_t)sink->ndim;
        if (out_op.as.sink.ndim <= B2ND_EXEC_MAX_RANK) {
            memcpy(out_op.as.sink.shape, sink->shape, out_op.as.sink.ndim * sizeof(int64_t));
        }
        out_op.as.sink.context = &sink_wrap;
        out_op.as.sink.write = b2_exec_cb_sink_trampoline;
    } else if (sink) {
        return ME_ERR_INVALID_ARG;
    } else {
        prep = b2_exec_init_buffer_output(schedule, output, output_bytes, &out_op);
        if (prep != ME_SUCCESS) {
            return prep;
        }
    }
    b2nd_run_options_t ropts = b2_exec_make_run_options(
        raise_mask, 0, max_gather_bytes, max_output_bytes, 1, 0, 0);
    b2nd_exec_report_t pub_rep = B2ND_EXEC_REPORT_INITIALIZER;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_execute(
        (const b2nd_schedule_t *)schedule, ninputs ? ops : NULL, ninputs, &out_op,
        &ropts, &pub_rep, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    b2_exec_fill_cb_report(report, &pub_rep);
    return (int)st;
}

static int b2_exec_execute_sources_into(const void *schedule, const b2_exec_source *sources,
                                        const b2_exec_view *views, size_t ninputs,
                                        const b2_exec_output_view *output, unsigned raise_mask,
                                        size_t max_gather_bytes,
                                        b2_exec_callback_report *report, me_error *error) {
    if (!output) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_operand_t ops[B2ND_EXEC_MAX_INPUTS];
    b2_exec_cb_wrap wraps[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_fill_operands(NULL, views, sources, ninputs, ops, wraps);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_output_t out_op = B2ND_OUTPUT_INITIALIZER;
    b2_exec_init_view_output(output, 0, 0, &out_op);
    b2nd_run_options_t ropts = b2_exec_make_run_options(
        raise_mask, 0, max_gather_bytes, 0, 1, 0, 0);
    b2nd_exec_report_t pub_rep = B2ND_EXEC_REPORT_INITIALIZER;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_execute(
        (const b2nd_schedule_t *)schedule, ninputs ? ops : NULL, ninputs, &out_op,
        &ropts, &pub_rep, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    b2_exec_fill_cb_report(report, &pub_rep);
    return (int)st;
}

static int b2_exec_artifact_has_core(const void *artifact) {
#ifdef B2_HAVE_MENUDET_ARTIFACT
    return me_artifact_borrow_core((const me_artifact *)artifact) != NULL;
#else
    (void)artifact;
    return 0;
#endif
}

static int b2_exec_prepare_borrowed_common(const me_kernel *kernel, int reducer,
                                             const b2_exec_reduce_opts *opts,
                                             void **out, me_error *error) {
    *out = NULL;
    if (!kernel) {
        return ME_ERR_UNSUPPORTED;
    }
    me_kernel *borrowed = (me_kernel *)kernel;
    b2nd_exec_reduce_opts_t ropts = B2ND_EXEC_REDUCE_OPTS_INITIALIZER;
    const b2nd_exec_reduce_opts_t *ropts_ptr = NULL;
    if (reducer != 0 || opts != NULL) {
        ropts.reduction = (b2nd_reduction_kind_t)reducer;
        if (opts) {
            ropts.has_axes = opts->has_axes != 0;
            ropts.keepdims = opts->keepdims != 0;
            ropts.has_initial = opts->has_initial != 0;
            ropts.has_where_slot = opts->has_where != 0;
            ropts.naxes = opts->naxes;
            if (opts->naxes <= B2ND_EXEC_MAX_RANK) {
                memcpy(ropts.axes, opts->axes, opts->naxes * sizeof(int));
            }
            ropts.initial_dtype = (b2nd_exec_dtype_t)opts->initial_dtype;
            ropts.initial_category = opts->initial_weak
                ? B2ND_CAPTURE_WEAK : B2ND_CAPTURE_TYPED;
            ropts.initial_size = opts->initial_size;
            memcpy(ropts.initial_scalar, opts->initial_scalar, sizeof(ropts.initial_scalar));
            ropts.accumulator = (b2nd_exec_dtype_t)opts->accumulator;
        }
        ropts_ptr = &ropts;
    }
    b2nd_plan_t *plan = NULL;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_plan_from_kernel(&borrowed, false, ropts_ptr, &plan, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    *out = plan;
    return (int)st;
}

static int b2_exec_prepare_core_kernel(const void *core_handle, int reducer,
                                       const b2_exec_reduce_opts *opts,
                                       void **out, me_error *error) {
    return b2_exec_prepare_borrowed_common(b2_core_kernel_unwrap(core_handle), reducer, opts, out, error);
}

static int b2_exec_prepare_artifact_kernel(const void *artifact, int reducer,
                                           const b2_exec_reduce_opts *opts,
                                           void **out, me_error *error) {
#ifdef B2_HAVE_MENUDET_ARTIFACT
    return b2_exec_prepare_borrowed_common(me_artifact_borrow_core((const me_artifact *)artifact),
                                           reducer, opts, out, error);
#else
    (void)artifact; (void)reducer; (void)opts; (void)error;
    *out = NULL;
    return ME_ERR_UNSUPPORTED;
#endif
}

static int b2_exec_specialize_partitions(const void *plan, const b2_exec_meta *inputs,
                                         size_t ninputs, const b2_exec_partition_spec *spec,
                                         size_t max_group_bytes, void **out, me_error *error) {
    *out = NULL;
    if (!spec) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_exec_input_meta_t metadata[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_convert_meta(inputs, ninputs, metadata);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_partition_spec_t pspec = B2ND_PARTITION_SPEC_INITIALIZER;
    pspec.ndim = spec->ndim;
    if (spec->ndim <= B2ND_EXEC_MAX_RANK) {
        memcpy(pspec.domain, spec->domain, sizeof(pspec.domain));
        memcpy(pspec.partitions, spec->partitions, sizeof(pspec.partitions));
        memcpy(pspec.sel_start, spec->sel_start, sizeof(pspec.sel_start));
        memcpy(pspec.sel_stop, spec->sel_stop, sizeof(pspec.sel_stop));
    }
    pspec.has_selection = spec->has_selection != 0;
    b2nd_schedule_options_t sopts = B2ND_SCHEDULE_OPTIONS_INITIALIZER;
    sopts.max_mapped_bytes = max_group_bytes;
    sopts.partition_spec = &pspec;
    b2nd_schedule_t *schedule = NULL;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_schedule_create(
        (const b2nd_plan_t *)plan, ninputs ? metadata : NULL, ninputs, &sopts,
        &schedule, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    *out = schedule;
    return (int)st;
}

static int b2_exec_admit_partitions(const void *schedule, const void *const *arrays,
                                    const b2_exec_view *views, const b2_exec_source *sources,
                                    size_t ninputs, int *admitted, me_error *error) {
    b2nd_operand_t ops[B2ND_EXEC_MAX_INPUTS];
    b2_exec_cb_wrap wraps[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_fill_operands(arrays, views, sources, ninputs, ops, wraps);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    bool native_admitted = false;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_schedule_admit(
        (const b2nd_schedule_t *)schedule, ninputs ? ops : NULL, ninputs, NULL,
        &native_admitted, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    if (admitted) {
        *admitted = native_admitted ? 1 : 0;
    }
    return (int)st;
}

static int b2_exec_execute_partitions(const void *schedule, const void *const *arrays,
                                      const b2_exec_view *views, const b2_exec_source *sources,
                                      size_t ninputs, void *output, size_t output_bytes,
                                      unsigned raise_mask, size_t max_decode_bytes,
                                      size_t max_gather_bytes, b2_exec_storage_report *report,
                                      me_error *error) {
    b2nd_operand_t ops[B2ND_EXEC_MAX_INPUTS];
    b2_exec_cb_wrap wraps[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_fill_operands(arrays, views, sources, ninputs, ops, wraps);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_output_t out_op = B2ND_OUTPUT_INITIALIZER;
    prep = b2_exec_init_buffer_output(schedule, output, output_bytes, &out_op);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_run_options_t ropts = b2_exec_make_run_options(
        raise_mask, max_decode_bytes, max_gather_bytes, 0, 1, 0, 0);
    b2nd_exec_report_t pub_rep = B2ND_EXEC_REPORT_INITIALIZER;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_execute(
        (const b2nd_schedule_t *)schedule, ninputs ? ops : NULL, ninputs, &out_op,
        &ropts, &pub_rep, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    b2_exec_fill_storage_report(report, &pub_rep);
    return (int)st;
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
                           int integer_powers, int fp_mode, int accuracy, void **out, me_error *error) {
    (void)nodes; (void)count; (void)ninputs; (void)root; (void)execution; (void)error;
    (void)compiler; (void)math_backend; (void)integer_powers;
    (void)fp_mode; (void)accuracy;
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
static int b2_exec_prepare_core_kernel(const void *core_handle, int reducer,
                                       const b2_exec_reduce_opts *opts,
                                       void **out, me_error *error) {
    (void)core_handle; (void)reducer; (void)opts; (void)error;
    *out = NULL;
    return ME_ERR_UNSUPPORTED;
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
