/*
 * Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* This module hosts the backend-selection and accelerator-specific GEMM
 * wrappers used by the matmul fast path. The portable naive kernel stays in
 * blosc2_ext.pyx, and external BLAS-style integrations live here.
 */

#include "matmul_kernels.h"
static int g_b2_matmul_backend = B2_MATMUL_BACKEND_AUTO;

int b2_has_accelerate(void) {
#if defined(__APPLE__)
    return 1;
#else
    return 0;
#endif
}

void b2_set_matmul_backend(int backend) {
    switch (backend) {
        case B2_MATMUL_BACKEND_AUTO:
        case B2_MATMUL_BACKEND_NAIVE:
        case B2_MATMUL_BACKEND_ACCELERATE:
            g_b2_matmul_backend = backend;
            break;
        default:
            g_b2_matmul_backend = B2_MATMUL_BACKEND_AUTO;
            break;
    }
}

int b2_get_matmul_backend(void) {
    return g_b2_matmul_backend;
}

int b2_get_selected_matmul_backend(void) {
    if (g_b2_matmul_backend == B2_MATMUL_BACKEND_ACCELERATE && !b2_has_accelerate()) {
        return B2_MATMUL_BACKEND_NAIVE;
    }
    if (g_b2_matmul_backend == B2_MATMUL_BACKEND_AUTO) {
        return b2_has_accelerate() ? B2_MATMUL_BACKEND_ACCELERATE : B2_MATMUL_BACKEND_NAIVE;
    }
    return g_b2_matmul_backend;
}

const char *b2_get_matmul_backend_name(void) {
    switch (g_b2_matmul_backend) {
        case B2_MATMUL_BACKEND_NAIVE:
            return "naive";
        case B2_MATMUL_BACKEND_ACCELERATE:
            return "accelerate";
        case B2_MATMUL_BACKEND_AUTO:
        default:
            return "auto";
    }
}

const char *b2_get_selected_matmul_backend_name(void) {
    switch (b2_get_selected_matmul_backend()) {
        case B2_MATMUL_BACKEND_ACCELERATE:
            return "accelerate";
        case B2_MATMUL_BACKEND_NAIVE:
        default:
            return "naive";
    }
}

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>

int b2_gemm_accelerate_f32(const float *A, const float *B, float *C, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 1.0f, C, N);
    return 0;
}

int b2_gemm_accelerate_f64(const double *A, const double *B, double *C, int M, int K, int N) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 1.0, C, N);
    return 0;
}
#else
int b2_gemm_accelerate_f32(const float *A, const float *B, float *C, int M, int K, int N) {
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)K;
    (void)N;
    return -1;
}

int b2_gemm_accelerate_f64(const double *A, const double *B, double *C, int M, int K, int N) {
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)K;
    (void)N;
    return -1;
}
#endif
