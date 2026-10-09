#ifndef BLOSC2_DSL_GRAPH_BRIDGE_H
#define BLOSC2_DSL_GRAPH_BRIDGE_H
#include "dsl_array_bridge.h"
typedef struct {
    const char *name;
    me_dtype dtype;
    int rank;
    int64_t shape[16];
} b2_graph_metadata;
typedef struct {
    int node, stage;
    b2_artifact_error native;
} b2_graph_error;
typedef struct {
    b2_array_report array;
    size_t stages, jit_stages, interpreter_stages;
} b2_graph_report;
#ifdef B2_HAVE_NATIVE_GRAPH
#include "miniexpr_graph.h"
static int b2_graph_prepare(const char *json, size_t n, int jit, int required, void **out, b2_graph_error *e) {
    me_graph_prepare_options options = {sizeof(options), ME_GRAPH_VERSION, (me_jit_mode)jit, required != 0, true};
    me_graph_plan *p = NULL;
    me_graph_error error;
    int rc = me_graph_prepare_json(json, n, &options, &p, &error);
    *out = p; e->node = error.node; e->stage = error.stage; e->native = error.native;
    return rc;
}
static int b2_graph_specialize(const void *p, const b2_graph_metadata *inputs, int n, size_t tile, size_t budget,
    void **out, b2_graph_error *e) {
    me_graph_input_metadata metadata[ME_MAX_VARS];
    if (n < 0 || n > ME_MAX_VARS) {
        *out = NULL; memset(e, 0, sizeof(*e)); e->node = e->stage = -1;
        snprintf(e->native.message, sizeof(e->native.message), "Native graph binding limit exceeded");
        return -5;
    }
    for (int i = 0; i < n; i++) {
        metadata[i].name = inputs[i].name; metadata[i].dtype = inputs[i].dtype; metadata[i].rank = inputs[i].rank;
        memcpy(metadata[i].shape, inputs[i].shape, sizeof(metadata[i].shape));
    }
    me_graph_specialize_options options = {sizeof(options), ME_GRAPH_VERSION, tile, budget};
    me_graph_schedule *s = NULL; me_graph_error error;
    int rc = me_graph_specialize(p, metadata, n, &options, &s, &error);
    *out = s; e->node = error.node; e->stage = error.stage; e->native = error.native;
    return rc;
}
static int b2_graph_prepare_text(const char *text, size_t length, const b2_graph_metadata *inputs,
    int n, int jit, int required, void **out, b2_graph_error *e) {
    me_graph_input_metadata metadata[ME_MAX_VARS] = {0};
    if (n < 0 || n > ME_MAX_VARS) {
        *out = NULL; memset(e, 0, sizeof(*e)); e->node = e->stage = -1;
        snprintf(e->native.message, sizeof(e->native.message), "Native graph binding limit exceeded");
        return -5;
    }
    for (int i = 0; i < n; i++) { metadata[i].name = inputs[i].name; metadata[i].dtype = inputs[i].dtype; }
    me_graph_prepare_options options = {sizeof(options), ME_GRAPH_VERSION, (me_jit_mode)jit, required != 0, true};
    me_graph_plan *p = NULL; me_graph_error error;
    int rc = me_graph_prepare_expression(text, length, metadata, n, &options, &p, &error);
    *out = p; e->node = error.node; e->stage = error.stage; e->native = error.native;
    return rc;
}
static int b2_graph_execute(const void *s, const b2_array_view *inputs, int n, void *out, size_t bytes,
    unsigned raise_mask, b2_graph_report *report, b2_graph_error *e) {
    me_graph_execute_options options = {sizeof(options), ME_GRAPH_VERSION, raise_mask};
    me_graph_report result; me_graph_error error;
    int rc = me_graph_execute(s, inputs, n, out, bytes, &options, &result, &error);
    report->array = result.array;
    report->stages = result.stages;
#ifdef ME_GRAPH_STAGED_FORMAT
    report->jit_stages = result.jit_stages;
    report->interpreter_stages = result.interpreter_stages;
#else
    report->jit_stages = result.has_jit ? 1 : 0;
    report->interpreter_stages = result.stages - report->jit_stages;
#endif
    e->node = error.node; e->stage = error.stage; e->native = error.native;
    return rc;
}
static void b2_graph_free(void *p) { me_graph_plan_free(p); }
static void b2_graph_schedule_free(void *s) { me_graph_schedule_free(s); }
static int b2_graph_ninputs(const void *p) { return me_graph_ninputs(p); }
static const char *b2_graph_name(const void *p, int i) { return me_graph_input_name(p, i); }
static me_dtype b2_graph_dtype(const void *p, int i) { return me_graph_input_dtype(p, i); }
static me_dtype b2_graph_inferred(const void *p) { return me_graph_inferred_dtype(p); }
static int b2_graph_jit(const void *p) { return me_graph_has_jit(p); }
static const char *b2_graph_json(const void *p) { return me_graph_export_json(p, NULL); }
static const char *b2_graph_map_json(const void *p) { return me_graph_export_map_json(p); }
static size_t b2_graph_stages(const void *p) { return me_graph_stage_count(p); }
static size_t b2_graph_intermediate(const void *s) { return me_graph_intermediate_bytes(s); }
static size_t b2_graph_plan_bytes(const void *p) { return me_graph_plan_bytes(p); }
static size_t b2_graph_schedule_bytes(const void *s) { return me_graph_schedule_bytes(s); }
static size_t b2_graph_scratch(const void *s) { return me_graph_scratch_bytes(s); }
static int b2_graph_stage_info(const void *s, int stage, int64_t *shape, me_dtype *dtype,
    size_t *bytes, int *last) {
#ifdef ME_GRAPH_STAGED_FORMAT
    int rank = me_graph_stage_output_rank(s, stage);
    if (rank >= 0) memcpy(shape, me_graph_stage_output_shape(s, stage), rank * sizeof(*shape));
    *dtype = me_graph_stage_output_dtype(s, stage);
    *bytes = me_graph_stage_output_bytes(s, stage);
    /* Last consumer is queried through the retaining schedule's plan below. */
    *last = me_graph_schedule_stage_last_consumer(s, stage);
    return rank;
#else
    (void)s; (void)stage; (void)shape; (void)dtype; (void)bytes; (void)last;
    return -1;
#endif
}
static int b2_graph_shape(const void *s, int64_t *shape, me_dtype *dtype) {
    int rank = me_graph_output_rank(s);
    if (rank >= 0) memcpy(shape, me_graph_output_shape(s), rank * sizeof(*shape));
    *dtype = me_graph_output_dtype(s); return rank;
}
static int b2_graph_map_shape(const void *s, int64_t *shape) {
    int rank; const int64_t *value = me_graph_map_shape(s, &rank);
    if (rank >= 0) memcpy(shape, value, rank * sizeof(*shape));
    return rank;
}
#else
static int b2_graph_prepare(const char *json, size_t n, int jit, int required, void **out, b2_graph_error *e) {
    (void)json; (void)n; (void)jit; (void)required; *out = NULL; memset(e, 0, sizeof(*e));
    snprintf(e->native.message, sizeof(e->native.message), "Build with miniexpr native graph preparation support"); return -100;
}
static int b2_graph_specialize(const void *p, const b2_graph_metadata *inputs, int n, size_t tile, size_t budget, void **out, b2_graph_error *e) {
    (void)p; (void)inputs; (void)n; (void)tile; (void)budget; (void)e; *out = NULL; return -100;
}
static int b2_graph_prepare_text(const char *text, size_t length, const b2_graph_metadata *inputs,
    int n, int jit, int required, void **out, b2_graph_error *e) {
    (void)text; (void)length; (void)inputs; (void)n; (void)jit; (void)required; (void)e; *out = NULL; return -100;
}
static int b2_graph_execute(const void *s, const b2_array_view *inputs, int n, void *out, size_t bytes,
    unsigned raise_mask, b2_graph_report *report, b2_graph_error *e) {
    (void)s; (void)inputs; (void)n; (void)out; (void)bytes; (void)raise_mask; (void)report; (void)e; return -100;
}
static void b2_graph_free(void *p) { (void)p; }
static void b2_graph_schedule_free(void *s) { (void)s; }
static int b2_graph_ninputs(const void *p) { (void)p; return 0; }
static const char *b2_graph_name(const void *p, int i) { (void)p; (void)i; return NULL; }
static me_dtype b2_graph_dtype(const void *p, int i) { (void)p; (void)i; return ME_AUTO; }
static me_dtype b2_graph_inferred(const void *p) { (void)p; return ME_AUTO; }
static int b2_graph_jit(const void *p) { (void)p; return 0; }
static const char *b2_graph_json(const void *p) { (void)p; return NULL; }
static const char *b2_graph_map_json(const void *p) { (void)p; return NULL; }
static size_t b2_graph_stages(const void *p) { (void)p; return 0; }
static size_t b2_graph_intermediate(const void *s) { (void)s; return 0; }
static size_t b2_graph_plan_bytes(const void *p) { (void)p; return 0; }
static size_t b2_graph_schedule_bytes(const void *s) { (void)s; return 0; }
static size_t b2_graph_scratch(const void *s) { (void)s; return 0; }
static int b2_graph_stage_info(const void *s, int stage, int64_t *shape, me_dtype *dtype,
    size_t *bytes, int *last) {
    (void)s; (void)stage; (void)shape; (void)dtype; (void)bytes; (void)last; return -1;
}
static int b2_graph_shape(const void *s, int64_t *shape, me_dtype *dtype) { (void)s; (void)shape; *dtype = ME_AUTO; return -1; }
static int b2_graph_map_shape(const void *s, int64_t *shape) { (void)s; (void)shape; return -1; }
#endif
#endif
