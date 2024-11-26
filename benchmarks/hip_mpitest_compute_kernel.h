/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/
#ifndef __HIP_MPITEST_COMPUTE_KERNEL__
#define __HIP_MPITEST_COMPUTE_KERNEL__

#include <hip/hip_runtime_api.h>

typedef struct hip_mpitest_compute_params_s {
    int         N, K, Kthresh, Rthresh, niter;
    long       *Ahost, *Adevice;
    double     *Afhost, *Afdevice;
    double      est_runtime;
    hipStream_t stream;
} hip_mpitest_compute_params_t;

int  hip_mpitest_compute_init(hip_mpitest_compute_params_t &params);
void hip_mpitest_compute_set_params(hip_mpitest_compute_params_t &params, double runtime);
int hip_mpitest_compute_launch (hip_mpitest_compute_params_t &params);
void hip_mpitest_compute_fini(hip_mpitest_compute_params_t &params);

#endif
