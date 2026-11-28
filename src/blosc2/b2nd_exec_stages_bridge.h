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
static int b2_st_prepare(const b2_exec_node *nodes, size_t count, size_t ninputs,
                         uint32_t root, int execution, int compiler, int math_backend,
                         int integer_powers, int fp_mode, int accuracy, void **out,
                         b2_st_diag *diag, me_error *error) {
    return b2_exec_prepare_impl(nodes, count, ninputs, root, execution, compiler, math_backend,
                                integer_powers, fp_mode, accuracy, out, diag ? &diag->stage : NULL,
                                diag ? &diag->node : NULL, error, 1);
}

static void b2_st_free(void *plan) {
    b2nd_plan_free((b2nd_plan_t *)plan);
}

static void b2_st_schedule_free(void *schedule) {
    b2nd_schedule_free((b2nd_schedule_t *)schedule);
}

static size_t b2_st_count(const void *plan) {
    b2nd_plan_info_t info = B2ND_PLAN_INFO_INITIALIZER;
    if (b2nd_plan_describe((const b2nd_plan_t *)plan, &info) != B2ND_EXEC_SUCCESS) {
        return 0;
    }
    return info.nstages;
}

static int b2_st_describe(const void *plan, b2_exec_info *out) {
    return b2_exec_describe(plan, out);
}

static int b2_st_assemble(const void *const *children, const int *bindings, const size_t *offsets,
                          size_t count, size_t ninputs, void **out, b2_st_diag *diag,
                          me_error *error) {
    *out = NULL;
    if (diag) {
        diag->stage = SIZE_MAX;
        diag->node = UINT32_MAX;
    }
    if (!count || count > B2ND_EXEC_MAX_STAGES) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_stage_binding_t stages[B2ND_EXEC_MAX_STAGES];
    for (size_t s = 0; s < count; s++) {
        stages[s] = (b2nd_stage_binding_t)B2ND_STAGE_BINDING_INITIALIZER;
        stages[s].plan = (const b2nd_plan_t *)children[s];
        stages[s].sources = bindings + offsets[s];
        stages[s].node = (uint32_t)s;
    }
    b2nd_plan_t *plan = NULL;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_plan_compose(stages, count, ninputs, &plan, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        if (diag) {
            diag->stage = pub_err.stage;
            diag->node = pub_err.node;
        }
        b2_exec_copy_error(error, &pub_err);
    }
    *out = plan;
    return (int)st;
}

static int b2_st_child_info(const void *plan, size_t stage, b2_exec_info *out, int *bindings,
                            size_t *count, size_t *node) {
    if (!out || !count || !node) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_plan_info_t info = B2ND_PLAN_INFO_INITIALIZER;
    uint32_t origin = UINT32_MAX;
    b2nd_exec_status_t st = b2nd_plan_describe_stage(
        (const b2nd_plan_t *)plan, stage, &info, bindings, B2ND_EXEC_MAX_INPUTS, count, &origin);
    if (st != B2ND_EXEC_SUCCESS) {
        return (int)st;
    }
    b2_exec_fill_plan_info(out, &info);
    *node = origin;
    return 0;
}

static int b2_st_source(const char *version, const char *source, const me_signature *signature,
                        int execution, int compiler, int math_backend, int fp_mode, int accuracy,
                        void **out, me_error *error) {
    *out = NULL;
    me_prepare_options options = ME_PREPARE_OPTIONS_INITIALIZER;
    options.execution = (me_execution_policy)execution;
    options.jit_compiler = (me_jit_compiler)compiler;
    options.math_backend = (me_math_backend)math_backend;
    options.fp_mode = (me_fp_mode)fp_mode;
    options.math_accuracy = (me_math_accuracy)accuracy;
    options.menudet_forbidden_features = ME_MENUDET_FEATURE_CONTROL_FLOW |
        ME_MENUDET_FEATURE_BLOCK_REDUCTIONS | ME_MENUDET_FEATURE_FIXED_STRINGS;
    me_kernel *kernel = NULL;
    int rc = me_prepare_menudet(version, source, signature, &options, &kernel, error);
    if (rc == ME_SUCCESS) {
        b2nd_plan_t *plan = NULL;
        b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
        b2nd_exec_status_t st = b2nd_plan_from_kernel(&kernel, true, NULL, &plan, &pub_err);
        if (st != B2ND_EXEC_SUCCESS) {
            b2_exec_copy_error(error, &pub_err);
        }
        *out = plan;
        rc = (int)st;
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
    b2nd_exec_input_meta_t metadata[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_convert_meta(inputs, ninputs, metadata);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_schedule_options_t sopts = B2ND_SCHEDULE_OPTIONS_INITIALIZER;
    sopts.tile_items = tile;
    sopts.max_mapped_bytes = mapped;
    sopts.max_intermediate_bytes = intermediate;
    b2nd_schedule_t *schedule = NULL;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_schedule_create(
        (const b2nd_plan_t *)plan, ninputs ? metadata : NULL, ninputs, &sopts,
        &schedule, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        if (diag) {
            diag->stage = pub_err.stage;
            diag->node = pub_err.node;
        }
        b2_exec_copy_error(error, &pub_err);
    }
    *out = schedule;
    return (int)st;
}

static int b2_st_stage(const void *schedule, size_t stage, b2_st_info *out) {
    if (!schedule || !out || stage >= B2ND_EXEC_MAX_STAGES) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_schedule_info_t sinfo = B2ND_SCHEDULE_INFO_INITIALIZER;
    b2nd_stage_info_t stages[B2ND_EXEC_MAX_STAGES];
    for (size_t i = 0; i <= stage; i++) {
        stages[i] = (b2nd_stage_info_t)B2ND_STAGE_INFO_INITIALIZER;
    }
    if (b2nd_schedule_describe((const b2nd_schedule_t *)schedule, &sinfo, stages, stage + 1) != B2ND_EXEC_SUCCESS ||
        stage >= sinfo.nstages) {
        return ME_ERR_INVALID_ARG;
    }
    b2_exec_fill_plan_info(&out->info, &stages[stage].plan);
    b2_exec_fill_schedule_info(&out->info, &stages[stage].schedule);
    out->node = stages[stage].node;
    out->last_consumer = stages[stage].last_consumer;
    out->output_bytes = stages[stage].output_bytes;
    return 0;
}

static void b2_st_fill_stage_report(b2_exec_callback_report *dst, const b2nd_stage_report_t *src) {
    dst->execution.tiles = src->tiles;
    dst->execution.map_calls = src->map_calls;
    dst->execution.direct_tiles = src->direct_tiles;
    dst->execution.mapped_bytes = src->mapped_bytes;
    dst->execution.fp.flags = src->fp.flags;
    dst->execution.fp.supported = src->fp.supported;
    dst->execution.gather_bytes = src->gather_bytes;
    dst->execution.gathered_bytes = src->gathered_bytes;
    dst->execution.interpreter_calls = src->interpreter_calls;
    dst->execution.jit_calls = src->jit_calls;
    dst->read_calls = src->read_calls;
    dst->read_items = src->read_items;
    dst->max_read_items = src->max_read_items;
    dst->read_bytes = src->read_bytes;
    dst->write_calls = src->write_calls;
    dst->write_items = src->write_items;
    dst->max_write_items = src->max_write_items;
    dst->written_bytes = src->written_bytes;
    dst->index_buffer_bytes = src->index_buffer_bytes;
    dst->output_buffer_bytes = src->output_buffer_bytes;
}

static void b2_st_fill_report(b2_st_report *report, const b2nd_exec_report_t *pub,
                              const b2nd_stage_report_t *stage_reps, int status) {
    memset(report, 0, sizeof(*report));
    report->failed_stage = pub->failed_stage;
    report->failed_node = pub->failed_node;
    report->fp.flags = pub->fp.flags;
    report->fp.supported = pub->fp.supported;
    if (pub->entered_stages > 0) {
        size_t lim = pub->entered_stages <= B2ND_EXEC_MAX_STAGES
            ? pub->entered_stages : B2ND_EXEC_MAX_STAGES;
        for (size_t s = 0; s < lim; s++) {
            b2_st_fill_stage_report(&report->stages[s], &stage_reps[s]);
        }
        report->entered_stages = pub->entered_stages;
        report->intermediate_bytes = pub->intermediate_bytes;
        report->live_intermediate_bytes = pub->live_intermediate_bytes;
    } else if (status == ME_SUCCESS) {
        b2_exec_fill_cb_report(&report->stages[0], pub);
        report->entered_stages = 1;
    }
}

static int b2_st_execute_common(const void *schedule, const b2_exec_source *sources,
                                const b2_exec_view *views, size_t ninputs,
                                const b2nd_output_t *out_op, unsigned raise_mask,
                                size_t gather, b2_st_report *report, me_error *error) {
    b2nd_operand_t ops[B2ND_EXEC_MAX_INPUTS];
    b2_exec_cb_wrap wraps[B2ND_EXEC_MAX_INPUTS];
    int prep = b2_exec_fill_operands(NULL, views, sources, ninputs, ops, wraps);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    b2nd_stage_report_t stage_reps[B2ND_EXEC_MAX_STAGES];
    for (size_t s = 0; s < B2ND_EXEC_MAX_STAGES; s++) {
        stage_reps[s] = (b2nd_stage_report_t)B2ND_STAGE_REPORT_INITIALIZER;
    }
    b2nd_run_options_t ropts = b2_exec_make_run_options(raise_mask, 0, gather, 0, 1, 0, 0);
    b2nd_exec_report_t pub_rep = B2ND_EXEC_REPORT_INITIALIZER;
    pub_rep.stage_reports = stage_reps;
    pub_rep.stage_reports_cap = B2ND_EXEC_MAX_STAGES;
    b2nd_exec_error_t pub_err = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_exec_status_t st = b2nd_execute(
        (const b2nd_schedule_t *)schedule, ninputs ? ops : NULL, ninputs, out_op,
        &ropts, &pub_rep, &pub_err);
    if (st != B2ND_EXEC_SUCCESS) {
        b2_exec_copy_error(error, &pub_err);
    }
    b2_st_fill_report(report, &pub_rep, stage_reps, (int)st);
    return (int)st;
}

static int b2_st_execute(const void *schedule, const b2_exec_source *sources,
                         const b2_exec_view *views, size_t ninputs,
                         void *output, size_t capacity, unsigned raise_mask,
                         size_t gather, b2_st_report *report, me_error *error) {
    b2nd_output_t out_op = B2ND_OUTPUT_INITIALIZER;
    int prep = b2_exec_init_buffer_output(schedule, output, capacity, &out_op);
    if (prep != ME_SUCCESS) {
        return prep;
    }
    return b2_st_execute_common(schedule, sources, views, ninputs, &out_op,
                                raise_mask, gather, report, error);
}

static int b2_st_execute_into(const void *schedule, const b2_exec_source *sources,
                              const b2_exec_view *views, size_t ninputs,
                              const b2_exec_output_view *output, unsigned raise_mask,
                              size_t gather, b2_st_report *report, me_error *error) {
    if (!output) {
        return ME_ERR_INVALID_ARG;
    }
    b2nd_output_t out_op = B2ND_OUTPUT_INITIALIZER;
    b2_exec_init_view_output(output, 0, 0, &out_op);
    return b2_st_execute_common(schedule, sources, views, ninputs, &out_op,
                                raise_mask, gather, report, error);
}
#else
static int b2_st_prepare(const b2_exec_node *nodes, size_t count, size_t ninputs,
                         uint32_t root, int execution, int compiler, int math_backend,
                         int integer_powers, int fp_mode, int accuracy, void **out, b2_st_diag *diag, me_error *error) {
    (void)nodes; (void)count; (void)ninputs; (void)root; (void)execution;
    (void)compiler; (void)math_backend; (void)integer_powers; (void)error;
    (void)fp_mode; (void)accuracy;
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
static int b2_st_child_info(const void *plan, size_t stage, b2_exec_info *out, int *bindings,
                            size_t *count, size_t *node) {
    (void)plan; (void)stage; (void)out; (void)bindings; (void)count; (void)node;
    return ME_ERR_UNSUPPORTED;
}
static int b2_st_source(const char *version, const char *source, const me_signature *signature,
                        int execution, int compiler, int math_backend, int fp_mode, int accuracy,
                        void **out, me_error *error) {
    (void)version; (void)source; (void)signature; (void)execution;
    (void)compiler; (void)math_backend; (void)fp_mode; (void)accuracy;
    (void)error; *out = NULL; return ME_ERR_UNSUPPORTED;
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
static int b2_st_execute_into(const void *schedule, const b2_exec_source *sources,
                              const b2_exec_view *views, size_t ninputs,
                              const b2_exec_output_view *output, unsigned raise_mask,
                              size_t gather, b2_st_report *report, me_error *error) {
    (void)schedule; (void)sources; (void)views; (void)ninputs; (void)output;
    (void)raise_mask; (void)gather; (void)report; (void)error; return ME_ERR_UNSUPPORTED;
}
#endif
#endif
