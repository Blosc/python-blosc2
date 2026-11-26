/* Optional bridge to the C-Blosc2 scheduler pilot; not a public dispatch path. */
#ifndef B2_ND_KERNEL_PILOT_H
#define B2_ND_KERNEL_PILOT_H
#include "b2nd.h"
#include "miniexpr_kernel.h"

#ifdef B2_HAVE_B2ND_KERNEL_PILOT
#include "b2nd-kernel.h"
#include <string.h>

static int b2_pilot_read(void *context, size_t input, int64_t chunk, int64_t block,
                         const me_grid_block_info *info, me_mut_buffer out) {
    (void)input;
    (void)chunk;
    (void)block;
    int64_t stop = info->origin[0] + info->extent[0];
    int64_t padded = (int64_t)info->padded_nitems;
    return b2nd_get_slice_cbuffer(context, info->origin, &stop, out.data, &padded, (int64_t)out.capacity);
}

static int b2_pilot_write(void *context, int64_t chunk, int64_t block,
                          const me_grid_block_info *info, me_buffer result) {
    (void)chunk;
    (void)block;
    if (info->valid_nitems) {
        double *output = context;
        memcpy(output + info->origin[0], result.data, info->valid_nitems * sizeof(*output));
    }
    return 0;
}

static int b2_pilot_map(b2nd_array_t *input, const char *source, double *output,
                        size_t scratch, me_error *error) {
    const size_t width = sizeof(double);
    if (!input || input->ndim != 1 || !input->sc || input->sc->typesize != width) {
        return BLOSC2_ERROR_INVALID_PARAM;
    }
    me_input_spec slot = {"x", ME_FLOAT64, sizeof(double)};
    me_signature signature = ME_SIGNATURE_INITIALIZER;
    signature.inputs = &slot;
    signature.ninputs = 1;
    signature.cardinality = ME_ELEMENTWISE;
    me_kernel *kernel = NULL;
    if (me_prepare_menudet("0.2", source, &signature, NULL, &kernel, error) != ME_SUCCESS) {
        return BLOSC2_ERROR_INVALID_PARAM;
    }
    me_kernel_info info = ME_KERNEL_INFO_INITIALIZER;
    int rc = BLOSC2_ERROR_INVALID_PARAM;
    if (me_describe(kernel, &info) == ME_SUCCESS && info.cardinality == ME_ELEMENTWISE &&
        !info.context_ndim && info.output_dtype == ME_FLOAT64) {
        rc = b2nd_eval_kernel_grid(input, kernel, 1, &width, scratch, b2_pilot_read, input,
                                   b2_pilot_write, output, error);
    }
    me_kernel_free(kernel);
    return rc;
}

static int b2_pilot_available(void) { return 1; }
#else
static int b2_pilot_available(void) { return 0; }
static int b2_pilot_map(b2nd_array_t *input, const char *source, double *output,
                        size_t scratch, me_error *error) {
    (void)input; (void)source; (void)output; (void)scratch; (void)error;
    return BLOSC2_ERROR_INVALID_PARAM;
}
#endif
#endif
