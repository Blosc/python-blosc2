#undef NDEBUG
#include <b2nd_exec.h>
#include <assert.h>
#include <stdio.h>

int main(void) {
    b2nd_exec_error_t error = B2ND_EXEC_ERROR_INITIALIZER;
    b2nd_plan_t *plan = NULL;
    b2nd_exec_options_t options = B2ND_EXEC_OPTIONS_INITIALIZER;
    options.struct_size = B2ND_EXEC_OPTIONS_V1_SIZE;
    options.execution = B2ND_EXEC_POLICY_INTERPRET;
    assert(b2nd_exec_available() == EXPECT_EXECUTION);
    blosc2_init();
    b2nd_exec_status_t status = b2nd_plan_from_expr(
        "1.0", NULL, 0, &options, NULL, &plan, &error);
    if (EXPECT_EXECUTION) {
        if(status != B2ND_EXEC_SUCCESS) {
            fprintf(stderr, "prepare failed (%d): %s\n", status, error.message);
            return 1;
        }
        assert(plan != NULL);
    }
    else {
        assert(status == B2ND_EXEC_ERR_UNSUPPORTED);
        assert(plan == NULL);
    }
    b2nd_plan_free(plan);
    blosc2_destroy();
    return 0;
}
