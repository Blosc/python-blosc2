#ifndef BLOSC2_PAIRWISE_SUM_BRIDGE_H
#define BLOSC2_PAIRWISE_SUM_BRIDGE_H
#include "dsl_artifact_bridge.h"
#include <fenv.h>
#include <stdint.h>

typedef struct {
    fenv_t env;
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t fpsr;
    int fast;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    uint32_t mxcsr;
    int fast;
#endif
} b2_fp_env;

/* Fast ARM64 FPCR/FPSR and x86-64 MXCSR guard: avoids pipeline-serializing
 * `msr fpcr` or x87 `fnstenv`/`fldenv` writes on every block when floating-point
 * exception traps are already disabled. */
static inline int b2_fp_hold(b2_fp_env *saved) {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr) :: "memory");
    if ((fpcr & 0x00009F00ULL) == 0) {
        uint64_t fpsr;
        __asm__ __volatile__("mrs %0, fpsr" : "=r"(fpsr) :: "memory");
        saved->fpsr = fpsr;
        saved->fast = 1;
        if ((fpsr & 0x9FULL) != 0) {
            __asm__ __volatile__("msr fpsr, %0" :: "r"(fpsr & ~0x9FULL) : "memory");
        }
        return 0;
    }
    saved->fast = 0;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    uint32_t mxcsr = 0;
    __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr) :: "memory");
    if ((mxcsr & 0x1F80u) == 0x1F80u) {
        saved->mxcsr = mxcsr;
        saved->fast = 1;
        if ((mxcsr & 0x3Fu) != 0) {
            uint32_t cleared = mxcsr & ~0x3Fu;
            __asm__ __volatile__("ldmxcsr %0" :: "m"(cleared) : "memory");
        }
        return 0;
    }
    saved->fast = 0;
#endif
    return feholdexcept(&saved->env);
}

static inline int b2_fp_restore(const b2_fp_env *saved) {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    if (saved->fast) {
        uint64_t cur_fpsr;
        __asm__ __volatile__("mrs %0, fpsr" : "=r"(cur_fpsr) :: "memory");
        if (cur_fpsr != saved->fpsr) {
            __asm__ __volatile__("msr fpsr, %0" :: "r"(saved->fpsr) : "memory");
        }
        return 0;
    }
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (saved->fast) {
        uint32_t cur_mxcsr = 0;
        __asm__ __volatile__("stmxcsr %0" : "=m"(cur_mxcsr) :: "memory");
        if (cur_mxcsr != saved->mxcsr) {
            uint32_t orig = saved->mxcsr;
            __asm__ __volatile__("ldmxcsr %0" :: "m"(orig) : "memory");
        }
        return 0;
    }
#endif
    return fesetenv(&saved->env);
}

#if defined(FE_INVALID) && defined(FE_DIVBYZERO) && defined(FE_OVERFLOW) && \
    defined(FE_UNDERFLOW) && defined(FE_ALL_EXCEPT)
#define B2_PAIRWISE_FENV_AVAILABLE 1
static unsigned b2_pairwise_flags(void) {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t fpsr;
    __asm__ __volatile__("mrs %0, fpsr" : "=r"(fpsr) :: "memory");
    return (unsigned)(fpsr & 0x0FULL);
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    uint32_t mxcsr = 0;
    __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr) :: "memory");
    return ((mxcsr & 0x01u) ? 1u : 0u) |
           ((mxcsr & 0x04u) ? 2u : 0u) |
           ((mxcsr & 0x08u) ? 4u : 0u) |
           ((mxcsr & 0x10u) ? 8u : 0u);
#else
    int f = fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW);
    return ((f & FE_INVALID) ? 1u : 0u) | ((f & FE_DIVBYZERO) ? 2u : 0u) |
           ((f & FE_OVERFLOW) ? 4u : 0u) | ((f & FE_UNDERFLOW) ? 8u : 0u);
#endif
}
static void b2_pairwise_clear_flags(void) {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t fpsr;
    __asm__ __volatile__("mrs %0, fpsr" : "=r"(fpsr) :: "memory");
    if ((fpsr & 0x9FULL) != 0) {
        __asm__ __volatile__("msr fpsr, %0" :: "r"(fpsr & ~0x9FULL) : "memory");
    }
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    uint32_t mxcsr = 0;
    __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr) :: "memory");
    if ((mxcsr & 0x3Fu) != 0) {
        uint32_t cleared = mxcsr & ~0x3Fu;
        __asm__ __volatile__("ldmxcsr %0" :: "m"(cleared) : "memory");
    }
#else
    feclearexcept(FE_ALL_EXCEPT);
#endif
}
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
