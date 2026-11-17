#ifndef BLOSC2_PAIRWISE_SUM_BRIDGE_H
#define BLOSC2_PAIRWISE_SUM_BRIDGE_H
#include "dsl_artifact_bridge.h"
#include <fenv.h>
#if defined(FE_INVALID) && defined(FE_DIVBYZERO) && defined(FE_OVERFLOW) && \
    defined(FE_UNDERFLOW) && defined(FE_ALL_EXCEPT)
#define B2_PAIRWISE_FENV_AVAILABLE 1
static unsigned b2_pairwise_flags(void) {
    int f = fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW);
    return ((f & FE_INVALID) ? 1u : 0u) | ((f & FE_DIVBYZERO) ? 2u : 0u) |
           ((f & FE_OVERFLOW) ? 4u : 0u) | ((f & FE_UNDERFLOW) ? 8u : 0u);
}
static void b2_pairwise_clear_flags(void) { feclearexcept(FE_ALL_EXCEPT); }
#else
/* Emscripten does not expose hardware floating-point exception tracking.
 * Keep inactive bridge calls compilable without advertising this route. */
#define B2_PAIRWISE_FENV_AVAILABLE 0
static unsigned b2_pairwise_flags(void) { return 0; }
static void b2_pairwise_clear_flags(void) {}
#endif
#ifdef ME_PAIRWISE_SUM_API
enum { B2_PAIRWISE_LEAF_ITEMS = ME_PAIRWISE_SUM_LEAF_ITEMS };
typedef me_pairwise_sum_state b2_pairwise_state;
#define b2_pairwise_init me_pairwise_sum_init
#define b2_pairwise_update me_pairwise_sum_update
#define b2_pairwise_append me_pairwise_sum_append
#define b2_pairwise_leaves me_pairwise_sum_leaves
#define b2_pairwise_finish me_pairwise_sum_finish
static int b2_pairwise_available(void) { return B2_PAIRWISE_FENV_AVAILABLE; }
#else
/* Nonzero placeholder for compiling inactive arithmetic on older runtimes. */
enum { B2_PAIRWISE_LEAF_ITEMS = 1 };
typedef struct { int unused; } b2_pairwise_state;
static int b2_pairwise_available(void) { return 0; }
static int b2_pairwise_init(b2_pairwise_state *s, me_dtype d) { (void)s; (void)d; return -2; }
static int b2_pairwise_update(b2_pairwise_state *s, const void *p, size_t n) {
    (void)s; (void)p; (void)n; return -2;
}
static int b2_pairwise_append(b2_pairwise_state *s, const void *p, size_t n) {
    (void)s; (void)p; (void)n; return -2;
}
static int b2_pairwise_leaves(me_dtype d, const void *p, size_t n, void *out, size_t cap) {
    (void)d; (void)p; (void)n; (void)out; (void)cap; return -2;
}
static int b2_pairwise_finish(b2_pairwise_state *s, void *out) { (void)s; (void)out; return -2; }
#endif
#endif
