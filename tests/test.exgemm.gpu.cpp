/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "blas3.hpp"
#include "common.hpp"

#include <iostream>
#include <limits>
#include <string.h>

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#ifdef EXBLAS_VS_MPFR
#include <cstddef>
#include <mpfr.h>

static double exgemmVsMPFR(double *exgemm, uint m, uint n, uint k, double alpha, double *a, uint lda, double *b, uint ldb, double*c, uint ldc) {
    double *exgemm_mpfr;
    mpfr_t sum, dot, op1;

    exgemm_mpfr = (double *) malloc(m * n * sizeof(double));

    mpfr_init2(op1, 64);
    mpfr_init2(dot, 128);
    mpfr_init2(sum, 2098);
    mpfr_set_d(dot, 0.0, MPFR_RNDN);

    //Produce a result matrix of DGEMM using MPFR
    for(uint i = 0; i < m; i++) {
        for(uint j = 0; j < n; j++) {
            mpfr_set_d(sum, 0.0, MPFR_RNDN);
            for(uint l = 0; l < k; l++) {
                mpfr_set_d(op1, a[i * k + l], MPFR_RNDN);
                mpfr_mul_d(dot, op1, b[l * n + j], MPFR_RNDN);
                mpfr_add(sum, sum, dot, MPFR_RNDN);
            }
            //exgemm_mpfr[i * n + j] = mpfr_get_d(sum, MPFR_RNDD);
            exgemm_mpfr[i * n + j] = c[i * n + j] + mpfr_get_d(sum, MPFR_RNDD);
        }
    }

    //Compare the GPU and MPFR results
#if 0
    //Frobenius Norm
    double norm = 0.0, val = 0.0;
    for (uint i = 0; i < m * n; i++) {
        norm += pow(exgemm[i] - exgemm_mpfr[i], 2);
        val += pow(exgemm_mpfr[i], 2);
    }
    norm = ::sqrt(norm) / ::sqrt(val);
#else
    //Inf norm -- maximum absolute row sum norm
    double norm = 0.0, val = 0.0;
    for(uint i = 0; i < m; i++) {
        double rowsum = 0.0, valrowsum = 0.0;
        for(uint j = 0; j < n; j++) {
            rowsum += fabs(exgemm[i * n + j] - exgemm_mpfr[i * n + j]);
            valrowsum += fabs(exgemm_mpfr[i * n + j]);
        }
        val = std::max(val, valrowsum);
        norm = std::max(norm, rowsum);
    }
    norm = norm / val;
#endif

    free(exgemm_mpfr);
    mpfr_free_cache();

    return norm;
}
#else
static double exgemmVsSuperacc(double *exgemm, double *superacc, uint m, uint n, uint k) {
#if 0
    //Frobenius Norm
    double norm = 0.0, val = 0.0;
    for (uint i = 0; i < m * n; i++) {
        norm += pow(exgemm[i] - superacc[i], 2);
        val += pow(superacc[i], 2);
    }
    norm = ::sqrt(norm) / ::sqrt(val);
#else
    //Inf norm -- maximum absolute row sum norm
    double norm = 0.0, val = 0.0;
    for(uint i = 0; i < m; i++) {
        double rowsum = 0.0, valrowsum = 0.0;
        for(uint j = 0; j < n; j++) {
            rowsum += fabs(exgemm[i * n + j] - superacc[i * n + j]);
            valrowsum += fabs(superacc[i * n + j]);
        }
        val = std::max(val, valrowsum);
        norm = std::max(norm, rowsum);
    }
    norm = norm / val;
#endif

    return norm;
}
#endif

static inline void copyMatrix(uint m, uint n, double* c, double* c_orig){
    for(uint i = 0; i < m; i++)
        for(uint j = 0; j < n; j++)
	    c[i * n + j] = c_orig[i * n + j];
}

static inline void printMatrix(uint m, uint n, double* c){
    for(uint i = 0; i < m; i++) {
        for(uint j = 0; j < n; j++)
	    printf("%f ", c[i * n + j]);
	printf("\n");
    }
}


int main(int argc, char *argv[]) {
    int m = 64, n = 64, k = 64;
    bool lognormal = false;
    if(argc > 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }
    if(argc > 6) {
        if(argv[6][0] == 'n') {
            lognormal = true;
        }
    }

    int range = 1;
    int emax = 0;
    double mean = 1., stddev = 1.;
    if(lognormal) {
        stddev = strtod(argv[4], 0);
        mean = strtod(argv[5], 0);
    }
    else {
        if(argc > 4) {
            range = atoi(argv[4]);
        }
        if(argc > 5) {
            emax = atoi(argv[5]);
        }
    }

    double eps = 1e-15;
    double *a, *b, *c, *c_orig;
    int err = posix_memalign((void **) &a, 64, m * k * sizeof(double));
    err &= posix_memalign((void **) &b, 64, k * n * sizeof(double));
    err &= posix_memalign((void **) &c, 64, m * n * sizeof(double));
    err &= posix_memalign((void **) &c_orig, 64, m * n * sizeof(double));
    if ((!a) || (!b) || (!c) || (!c_orig) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");
    if(lognormal) {
        init_lognormal(a, m * k, mean, stddev);
        init_lognormal(b, k * n, mean, stddev);
        init_lognormal(c, m * n, mean, stddev);
    } else if ((argc > 6) && (argv[6][0] == 'i')) {
        init_ill_cond(a, m * k, range);
        init_ill_cond(b, k * n, range);
        init_ill_cond(c, m * n, range);
    } else {
        if(range == 1){
            init_naive(a, m * k);
            init_naive(b, k * n);
            init_naive(c, m * n);
        } else {
            init_fpuniform(a, m * k, range, emax);
            init_fpuniform(b, k * n, range, emax);
            init_fpuniform(c, m * n, range, emax);
        }
    }
    copyMatrix(m, n, c_orig, c);

    fprintf(stderr, "%d %d %d ", m, n, k);

    if(lognormal) {
        fprintf(stderr, "%f ", stddev);
    } else {
        fprintf(stderr, "%d ", range);
    }

    bool is_pass = true;
    double *superacc;
    double norm;
    err = posix_memalign((void **) &superacc, 64, m * n * sizeof(double));
    if ((!superacc) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");
    copyMatrix(m, n, superacc, c);

    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 1.0, superacc, n, 1);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(superacc, m, n, k, 1.0, a, k, b, n, c_orig, n);
    printf("Superacc error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#endif

    copyMatrix(m, n, c, c_orig);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 1.0, c, n, 3);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n, c_orig, n);
    printf("FPE3 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif

    copyMatrix(m, n, c, c_orig);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 1.0, c, n, 4);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n, c_orig, n);
    printf("FPE4 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif

    copyMatrix(m, n, c, c_orig);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 1.0, c, n, 8);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n, c_orig, n);
    printf("FPE8 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif

    copyMatrix(m, n, c, c_orig);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 1.0, c, n, 4, true);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n, c_orig, n);
    printf("FPE4EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif

    copyMatrix(m, n, c, c_orig);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 1.0, c, n, 6, true);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n, c_orig, n);
    printf("FPE6EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif

    copyMatrix(m, n, c, c_orig);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 1.0, c, n, 8, true);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n, c_orig, n);
    printf("FPE8EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif
    fprintf(stderr, "\n");

    if (is_pass)
        printf("TestPassed; ALL OK!\n");
    else
        printf("TestFailed!\n");

    return 0;
}

