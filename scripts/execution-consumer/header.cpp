#include <b2nd_exec.h>

static_assert(B2ND_EXEC_API_VERSION == 1, "Unexpected execution API version");

int execution_cpp_header_check() {
    b2nd_exec_options_t options = B2ND_EXEC_OPTIONS_INITIALIZER;
    return options.struct_size >= B2ND_EXEC_OPTIONS_V1_SIZE ? 0 : 1;
}
