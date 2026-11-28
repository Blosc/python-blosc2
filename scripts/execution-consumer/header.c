#include <b2nd_exec.h>

typedef char execution_api_version_check[B2ND_EXEC_API_VERSION == 1 ? 1 : -1];

int execution_header_check(void) {
    b2nd_exec_options_t options = B2ND_EXEC_OPTIONS_INITIALIZER;
    return options.struct_size >= B2ND_EXEC_OPTIONS_V1_SIZE ? 0 : 1;
}
