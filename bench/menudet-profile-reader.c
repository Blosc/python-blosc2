/* Profiling-only resident compressed-chunk decoder. Not a storage adapter:
 * snapshots omit SChunk dictionary/postfilter/lazy-chunk integration. */
#include <blosc2.h>

void *menudet_profile_context(void) {
  blosc2_dparams params = BLOSC2_DPARAMS_DEFAULTS;
  params.nthreads = 1;
  params.typesize = 8;
  return blosc2_create_dctx(params);
}

int menudet_profile_decode(void *context, const void *source, int source_bytes,
                          int start_bytes, int size_bytes, void *output) {
  return blosc2_getitem_bytes_ctx(context, source, source_bytes, start_bytes,
                                size_bytes, output, size_bytes);
}

void menudet_profile_free(void *context) {
  blosc2_free_ctx(context);
}
