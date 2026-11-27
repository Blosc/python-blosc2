/* Private staged bridge. No adapter or serialization is involved. */
#ifndef B2_EXEC_STAGES_BRIDGE_H
#define B2_EXEC_STAGES_BRIDGE_H
typedef struct {
    size_t stage;
    uint32_t node;
} b2_st_diag;
typedef struct {
    b2_exec_info info;
    size_t node, last_consumer, output_bytes;
} b2_st_info;
typedef struct {
    b2_exec_callback_report stages[32];
    size_t entered_stages, intermediate_bytes, live_intermediate_bytes;
    size_t failed_stage, failed_node;
    me_fp_status fp;
} b2_st_report;

#ifdef B2_HAVE_B2ND_EXEC
static void b2_st_fill_plan(b2_exec_info *out, const b2nd_exec_plan_info *info) {
    memset(out, 0, sizeof(*out));
    out->dtype = info->dtype; out->map_dtype = info->map_dtype;
    out->itemsize = info->itemsize; out->ninputs = info->ninputs;
    out->identity = info->identity; out->reduction = info->reduction;
    out->reducer = info->reducer; out->has_jit = info->has_jit;
    out->partitionable = info->partitionable;
    out->plan_metadata_bytes = info->plan_metadata_bytes;
    out->has_axes = info->has_axes; out->naxes = info->naxes;
    memcpy(out->axes, info->axes, sizeof(out->axes));
    out->keepdims = info->keepdims; out->has_initial = info->has_initial;
    out->has_where = info->has_where;
}
static int b2_st_prepare(const b2_exec_node *nodes, size_t count, size_t ninputs,
                         uint32_t root, int execution, int compiler, int math_backend,
                         void **out, b2_st_diag *diag, me_error *error) {
    return b2_exec_prepare_impl(nodes, count, ninputs, root, execution, compiler, math_backend,
                                out, diag ? &diag->stage : NULL, diag ? &diag->node : NULL, error, 1);
}
static void b2_st_free(void *plan) { b2nd_exec_staged_plan_free(plan); }
static void b2_st_schedule_free(void *schedule) { b2nd_exec_staged_schedule_free(schedule); }
static size_t b2_st_count(const void *plan) { return b2nd_exec_staged_count(plan); }
static int b2_st_describe(const void *plan, b2_exec_info *out) {
    b2nd_exec_plan_info info;
    int rc = b2nd_exec_staged_plan_describe(plan, &info);
    if (!rc) b2_st_fill_plan(out, &info);
    return rc;
}
static int b2_st_assemble(const void *const *children, const int *bindings, const size_t *offsets,
                          size_t count, size_t ninputs, void **out, b2_st_diag *diag,
                          me_error *error) {
    *out = NULL;
    if (diag) {
        diag->stage = SIZE_MAX;
        diag->node = UINT32_MAX;
    }
    if (!count || count > B2ND_EXEC_MAX_STAGES) return ME_ERR_INVALID_ARG;
    b2nd_exec_stage_spec stages[B2ND_EXEC_MAX_STAGES] = {{0}};
    for (size_t s = 0; s < count; s++) {
        stages[s].plan = children[s];
        stages[s].sources = bindings + offsets[s];
        stages[s].node = (uint32_t)s;
    }
    b2nd_exec_staged_plan *plan = NULL;
    b2nd_exec_stage_diagnostic native_diag = {SIZE_MAX, UINT32_MAX};
    int rc = b2nd_exec_staged_assemble_ex(stages, count, ninputs, &plan,
                                          diag ? &native_diag : NULL, error);
    if (diag) {
        diag->stage = native_diag.stage;
        diag->node = native_diag.node;
    }
    *out = plan;
    return rc;
}
static int b2_st_child(const void *plan, size_t stage, void **out, int *bindings,
                       size_t *count, size_t *node) {
    const b2nd_exec_plan *child = NULL;
    const int *sources = NULL;
    uint32_t origin;
    int rc = b2nd_exec_staged_child(plan, stage, &child, &sources, count, &origin);
    if (rc) return rc;
    *out = (void *)child;
    *node = origin;
    if (*count) memcpy(bindings, sources, *count * sizeof(int));
    return 0;
}
static int b2_st_source(const char *version, const char *source, const me_signature *signature,
                        int execution, int compiler, int math_backend, void **out, me_error *error) {
    *out = NULL;
    me_prepare_options options = ME_PREPARE_OPTIONS_INITIALIZER;
    options.execution = (me_execution_policy)execution;
    options.jit_compiler = (me_jit_compiler)compiler;
    options.math_backend = (me_math_backend)math_backend;
    options.menudet_forbidden_features = ME_MENUDET_FEATURE_CONTROL_FLOW |
        ME_MENUDET_FEATURE_BLOCK_REDUCTIONS | ME_MENUDET_FEATURE_FIXED_STRINGS;
    me_kernel *kernel = NULL;
    int rc = me_prepare_menudet(version, source, signature, &options, &kernel, error);
    if (rc == ME_SUCCESS) {
        b2nd_exec_plan *plan = NULL;
        rc = b2nd_exec_prepare_kernel(&kernel, ME_REDUCTION_NONE, &plan, error);
        *out = plan;
    }
    me_kernel_free(kernel);
    return rc;
}
static int b2_st_specialize(const void *plan, const b2_exec_meta *inputs, size_t ninputs,
                            size_t tile, size_t mapped, size_t intermediate,
                            void **out, b2_st_diag *diag, me_error *error) {
    *out = NULL;
    if (diag) {
        diag->stage = SIZE_MAX;
        diag->node = UINT32_MAX;
    }
    if (ninputs > B2ND_EXEC_MAX_INPUTS) return ME_ERR_INVALID_ARG;
    b2nd_exec_input_meta metadata[B2ND_EXEC_MAX_INPUTS] = {{0}};
    for (size_t i = 0; i < ninputs; i++) {
        metadata[i].dtype = (me_dtype)inputs[i].dtype;
        metadata[i].itemsize = inputs[i].itemsize;
        metadata[i].ndim = (size_t)inputs[i].ndim;
        memcpy(metadata[i].shape, inputs[i].shape, sizeof(metadata[i].shape));
    }
    b2nd_exec_staged_schedule *schedule = NULL;
    b2nd_exec_stage_diagnostic native_diag = {SIZE_MAX, UINT32_MAX};
    int rc = b2nd_exec_staged_specialize_ex(plan, metadata, ninputs, tile, mapped, intermediate,
                                            &schedule, diag ? &native_diag : NULL, error);
    if (diag) {
        diag->stage = native_diag.stage;
        diag->node = native_diag.node;
    }
    *out = schedule;
    return rc;
}
static int b2_st_stage(const void *schedule, size_t stage, b2_st_info *out) {
    b2nd_exec_stage_info info;
    int rc = b2nd_exec_staged_describe(schedule, stage, &info);
    if (rc) return rc;
    b2_st_fill_plan(&out->info, &info.plan);
    b2_exec_info *o = &out->info;
    const b2nd_exec_schedule_info *s = &info.schedule;
    o->nitems = s->nitems; o->tile_items = s->tile_items;
    o->mapped_bytes = s->mapped_bytes; o->reducer_bytes = s->reducer_bytes;
    o->output_items = s->output_items; o->schedule_metadata_bytes = s->schedule_metadata_bytes;
    o->ndim = s->ndim; memcpy(o->shape, s->shape, sizeof(o->shape));
    o->output_ndim = s->output_ndim;
    memcpy(o->output_shape, s->output_shape, sizeof(o->output_shape));
    o->reduction_groups = s->reduction_groups; o->reduction_lanes = s->reduction_lanes;
    o->broadcast_gather_bytes = s->broadcast_gather_bytes;
    out->node = info.node; out->last_consumer = info.last_consumer; out->output_bytes = info.output_bytes;
    return 0;
}
static int b2_st_execute(const void *schedule, const b2_exec_source *sources,
                         const b2_exec_view *views, size_t ninputs,
                         void *output, size_t capacity, unsigned raise_mask,
                         size_t gather, b2_st_report *report, me_error *error) {
    if (ninputs > B2ND_EXEC_MAX_INPUTS) return ME_ERR_INVALID_ARG;
    b2nd_exec_view native_views[B2ND_EXEC_MAX_INPUTS] = {{0}};
    int prep = b2_exec_prepare_storage_views(views, ninputs, native_views);
    if (prep != ME_SUCCESS) return prep;
    b2nd_exec_source native_sources[B2ND_EXEC_MAX_INPUTS] = {{0}};
    for (size_t i = 0; i < ninputs; i++) {
        native_sources[i].struct_size = sizeof(native_sources[i]);
        native_sources[i].dtype = (me_dtype)sources[i].dtype;
        native_sources[i].itemsize = sources[i].itemsize;
        native_sources[i].ndim = (size_t)sources[i].ndim;
        memcpy(native_sources[i].shape, sources[i].shape, sizeof(native_sources[i].shape));
        if (sources[i].read) {
            native_sources[i].context = (void *)&sources[i];
            native_sources[i].read = b2_exec_cb_source_trampoline;
        }
    }
    b2nd_exec_staged_report native = {0};
    int rc = b2nd_exec_staged_execute_sources(schedule, native_sources, native_views,
        ninputs, (me_mut_buffer){output, capacity}, raise_mask, gather, &native, error);
    memset(report, 0, sizeof(*report));
    for (size_t s = 0; s < B2ND_EXEC_MAX_STAGES; s++) b2_exec_fill_cb_report(&report->stages[s], &native.stages[s]);
    report->entered_stages = native.entered_stages;
    report->intermediate_bytes = native.intermediate_bytes;
    report->live_intermediate_bytes = native.live_intermediate_bytes;
    report->failed_stage = native.failed_stage; report->failed_node = native.failed_node;
    report->fp = native.fp;
    return rc;
}
#else
static int b2_st_prepare(const b2_exec_node *nodes, size_t count, size_t ninputs,
                         uint32_t root, int execution, int compiler, int math_backend,
                         void **out, b2_st_diag *diag, me_error *error) {
    (void)nodes; (void)count; (void)ninputs; (void)root; (void)execution;
    (void)compiler; (void)math_backend; (void)error;
    if (diag) {
        diag->stage = (size_t)-1;
        diag->node = (uint32_t)-1;
    }
    *out = NULL; return ME_ERR_UNSUPPORTED;
}
static void b2_st_free(void *plan) { (void)plan; }
static void b2_st_schedule_free(void *schedule) { (void)schedule; }
static size_t b2_st_count(const void *plan) { (void)plan; return 0; }
static int b2_st_describe(const void *plan, b2_exec_info *out) {
    (void)plan; (void)out; return ME_ERR_UNSUPPORTED;
}
static int b2_st_assemble(const void *const *children, const int *bindings, const size_t *offsets,
                          size_t count, size_t ninputs, void **out, b2_st_diag *diag,
                          me_error *error) {
    (void)children; (void)bindings; (void)offsets; (void)count; (void)ninputs;
    (void)error;
    if (diag) {
        diag->stage = (size_t)-1;
        diag->node = (uint32_t)-1;
    }
    *out = NULL; return ME_ERR_UNSUPPORTED;
}
static int b2_st_child(const void *plan, size_t stage, void **out, int *bindings,
                       size_t *count, size_t *node) {
    (void)plan; (void)stage; (void)bindings; (void)count; (void)node;
    *out = NULL; return ME_ERR_UNSUPPORTED;
}
static int b2_st_source(const char *version, const char *source, const me_signature *signature,
                        int execution, int compiler, int math_backend, void **out, me_error *error) {
    (void)version; (void)source; (void)signature; (void)execution;
    (void)compiler; (void)math_backend; (void)error; *out = NULL; return ME_ERR_UNSUPPORTED;
}
static int b2_st_specialize(const void *plan, const b2_exec_meta *inputs, size_t ninputs,
                            size_t tile, size_t mapped, size_t intermediate,
                            void **out, b2_st_diag *diag, me_error *error) {
    (void)plan; (void)inputs; (void)ninputs; (void)tile; (void)mapped; (void)intermediate;
    (void)error;
    if (diag) {
        diag->stage = (size_t)-1;
        diag->node = (uint32_t)-1;
    }
    *out = NULL; return ME_ERR_UNSUPPORTED;
}
static int b2_st_stage(const void *schedule, size_t stage, b2_st_info *out) {
    (void)schedule; (void)stage; (void)out; return ME_ERR_UNSUPPORTED;
}
static int b2_st_execute(const void *schedule, const b2_exec_source *sources,
                         const b2_exec_view *views, size_t ninputs,
                         void *output, size_t capacity, unsigned raise_mask,
                         size_t gather, b2_st_report *report, me_error *error) {
    (void)schedule; (void)sources; (void)views; (void)ninputs; (void)output; (void)capacity;
    (void)raise_mask; (void)gather; (void)report; (void)error; return ME_ERR_UNSUPPORTED;
}
#endif
#endif
