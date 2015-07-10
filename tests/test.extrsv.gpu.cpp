/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "blas2.hpp"
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


static void copyVector(uint n, double *x, double *y) {
    for (uint i = 0; i < n; i++)
        x[i] = y[i];
}

#ifdef EXBLAS_VS_MPFR
#include <cstddef>
#include <mpfr.h>

static double extrsvVsMPFR(char uplo, double *extrsv, int n, double *a, uint lda, double *x, uint incx) {
#if 1
    // Compare to the results from Matlab
    FILE *pFilex;
    size_t resx;
    if (uplo == 'L') {
        pFilex = fopen("matrices/lnn/x_test_trsv_lnn_e08_64.bin", "rb");
        //pFilex = fopen("matrices/lnn/x_test_trsv_e21_64.bin", "rb");
        //pFilex = fopen("matrices/lnn/x_test_trsv_e40_64.bin", "rb");
    } else if (uplo == 'U') {
        //pFilex = fopen("matrices/unn/xe_unn_100_3.08e+07.bin", "rb");
        pFilex = fopen("matrices/unn/x_test_trsv_unn_e07_64.bin", "rb");
    }
    if (pFilex == NULL) {
        fprintf(stderr, "Cannot open files to read matrix and vector\n");
        exit(1);
    }

    double *xmatlab = (double *) malloc(n * sizeof(double));
    resx = fread(xmatlab, sizeof(double), n, pFilex);
    if (resx != n) {
        fprintf(stderr, "Cannot read matrix and vector from files\n");
        exit(1);
    }
    fclose(pFilex);

    //Inf norm
    double nrm2 = 0.0, val2 = 0.0;
    /*for(int i = 0; i < n; i++) {
        val2 = std::max(val2, fabs(xmatlab[i]));
        nrm2 = std::max(nrm2, fabs(extrsv[i] - xmatlab[i]));
        if (fabs(extrsv[i] - xmatlab[i]) != 0.0)
            printf("\n %d \t", i);
        printf("%.16g\t", fabs(extrsv[i] - xmatlab[i]));
    }
    printf("\n\n");
    printf("ExTRSV vs Matlab = %.16g \t %.16g\n", nrm2, val2);
    nrm2 = nrm2 / val2;
    printf("ExTRSV vs Matlab = %.16g\n", nrm2);*/

    mpfr_t sum, dot;

    // mpfr
    double *extrsv_mpfr = (double *) malloc(n * sizeof(double));
    copyVector(n, extrsv_mpfr, x);

    mpfr_init2(dot, 192);
    mpfr_init2(sum, 2098);

    //Produce a result matrix of TRSV using MPFR
    if (uplo == 'L') {
        for(int i = 0; i < n; i++) {
            // sum += a[i,j] * x[j], j < i
            mpfr_set_d(sum, extrsv_mpfr[i], MPFR_RNDN);
            for(int j = 0; j < i; j++) {
                mpfr_set_d(dot, a[j * n + i], MPFR_RNDN);
                mpfr_mul_d(dot, dot, extrsv_mpfr[j], MPFR_RNDN);
                mpfr_sub(sum, sum, dot, MPFR_RNDN);
            }
            extrsv_mpfr[i] = mpfr_get_d(sum, MPFR_RNDN);
            extrsv_mpfr[i] = extrsv_mpfr[i] / a[i * (n + 1)];
        }
    } else if (uplo == 'U') {
        for(int i = n-1; i >= 0; i--) {
            // sum += a[i,j] * x[j], j < i
            mpfr_set_d(sum, extrsv_mpfr[i], MPFR_RNDN);
            for(int j = i + 1; j < n; j++) {
                mpfr_set_d(dot, a[j * n + i], MPFR_RNDN);
                mpfr_mul_d(dot, dot, extrsv_mpfr[j], MPFR_RNDN);
                mpfr_sub(sum, sum, dot, MPFR_RNDN);
            }
            extrsv_mpfr[i] = mpfr_get_d(sum, MPFR_RNDN);
            extrsv_mpfr[i] = extrsv_mpfr[i] / a[i * (n + 1)];
        }
    }

    //Inf norm
    double nrm = 0.0, val = 0.0;
    for(int i = 0; i < n; i++) {
        printf("%.16g\t", xmatlab[i]);
    }
    printf("\n\n");
    for(int i = 0; i < n; i++) {
        printf("%.16g\t", extrsv_mpfr[i]);
    }
    printf("\n\n");
    for(int i = 0; i < n; i++) {
        val = std::max(val, fabs(extrsv_mpfr[i]));
        //nrm = std::max(nrm, fabs(extrsv[i] - extrsv_mpfr[i]));
        //printf("%.16g\t", fabs(extrsv[i] - extrsv_mpfr[i]));
        nrm = std::max(nrm, fabs(xmatlab[i] - extrsv_mpfr[i]));
        printf("%.16g\t", fabs(xmatlab[i] - extrsv_mpfr[i]));
    }
    printf("\n\n");
    printf("ExTRSV vs MPFR = %.16g \t %.16g\n", nrm, val);
    nrm = nrm / val;
    printf("ExTRSV vs MPFR = %.16g\n", nrm);

    return nrm2;
#else

    mpfr_t sum, dot;

    double *extrsv_mpfr = (double *) malloc(n * sizeof(double));
    copyVector(n, extrsv_mpfr, x);

    mpfr_init2(dot, 128);
    mpfr_init2(sum, 2098);

    //Produce a result matrix of TRSV using MPFR
    if (uplo == 'L') {
        for(int i = 0; i < n; i++) {
            // sum += a[i,j] * x[j], j < i
            mpfr_set_d(sum, 0.0, MPFR_RNDN);
            for(int j = 0; j < i; j++) {
                mpfr_set_d(dot, a[j * n + i], MPFR_RNDN);
                mpfr_mul_d(dot, dot, -extrsv_mpfr[j], MPFR_RNDN);
                mpfr_add(sum, sum, dot, MPFR_RNDN);
            }
            mpfr_add_d(sum, sum, extrsv_mpfr[i], MPFR_RNDN);
            mpfr_div_d(sum, sum, a[i * (n + 1)], MPFR_RNDN);
            extrsv_mpfr[i] = mpfr_get_d(sum, MPFR_RNDN);
        }
    } else if (uplo == 'U') {
        for(int i = n-1; i >= 0; i--) {
            // sum += a[i,j] * x[j], j < i
            mpfr_set_d(sum, 0.0, MPFR_RNDN);
            for(int j = i+1; j < n; j++) {
                mpfr_set_d(dot, a[j * n + i], MPFR_RNDN);
                mpfr_mul_d(dot, dot, -extrsv_mpfr[j], MPFR_RNDN);
                mpfr_add(sum, sum, dot, MPFR_RNDN);
            }
            mpfr_add_d(sum, sum, extrsv_mpfr[i], MPFR_RNDN);
            mpfr_div_d(sum, sum, a[i * (n + 1)], MPFR_RNDN);
            extrsv_mpfr[i] = mpfr_get_d(sum, MPFR_RNDN);
        }
    }

    //compare the GPU and MPFR results
#if 0
    //L2 norm
    double nrm = 0.0, val = 0.0;
    for(uint i = 0; i < n; i++) {
        nrm += pow(fabs(extrsv[i] - extrsv_mpfr[i]), 2);
        val += pow(fabs(extrsv_mpfr[i]), 2);
    }
    nrm = ::sqrt(nrm) / ::sqrt(val);
#else
    //Inf norm
    double nrm = 0.0, val = 0.0;
    for(int i = 0; i < n; i++) {
        val = std::max(val, fabs(extrsv_mpfr[i]));
        nrm = std::max(nrm, fabs(extrsv[i] - extrsv_mpfr[i]));
        //printf("%.16g\t", fabs(extrsv[i] - extrsv_mpfr[i]));
    }
    nrm = nrm / val;
#endif

    free(extrsv_mpfr);
    mpfr_free_cache();

    return nrm;
#endif
}

#else
static double extrsvVsSuperacc(uint n, double *extrsv, double *superacc) {
    double nrm = 0.0, val = 0.0;
    for (uint i = 0; i < n; i++) {
        nrm += pow(fabs(extrsv[i] - superacc[i]), 2);
        val += pow(fabs(superacc[i]), 2);
    }
    nrm = ::sqrt(nrm) / ::sqrt(val);

    return nrm;
}
#endif


int main(int argc, char *argv[]) {
    char uplo = 'U';
    uint n = 64;
    bool lognormal = false;
    if(argc > 1)
        uplo = argv[1][0];
    if(argc > 2)
        n = atoi(argv[2]);
    if(argc > 5) {
        if(argv[5][0] == 'n') {
            lognormal = true;
        }
    }

    int range = 1;
    int emax = 0;
    double mean = 1., stddev = 1.;
    if(lognormal) {
        stddev = strtod(argv[3], 0);
        mean = strtod(argv[4], 0);
    }
    else {
        if(argc > 3) {
            range = atoi(argv[3]);
        }
        if(argc > 4) {
            emax = atoi(argv[4]);
        }
    }

    double eps = 1e-13;
    double *a, *x, *xorig;
    int err = posix_memalign((void **) &a, 64, n * n * sizeof(double));
    err &= posix_memalign((void **) &x, 64, n * sizeof(double));
    err &= posix_memalign((void **) &xorig, 64, n * sizeof(double));
    if ((!a) || (!x) || (!xorig) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");

#if 1
    //Reading matrix A and vector b from files
    FILE *pFileA, *pFileb;
    size_t resA, resb;
    if (uplo == 'L') {
        pFileA = fopen("matrices/lnn/A_lnn_64_9.76e+08.bin", "rb");
        pFileb = fopen("matrices/lnn/b_lnn_64_9.76e+08.bin", "rb");
        //pFileA = fopen("matrices/lnn/A_lnn_64_9.30e+13.bin", "rb");
        //pFileb = fopen("matrices/lnn/b_lnn_64_9.30e+13.bin", "rb");
        //pFileA = fopen("matrices/lnn/A_lnn_64_9.53e+21.bin", "rb");
        //pFileb = fopen("matrices/lnn/b_lnn_64_9.53e+21.bin", "rb");
        //pFileA = fopen("matrices/lnn/A_lnn_64_7.58e+40.bin", "rb");
        //pFileb = fopen("matrices/lnn/b_lnn_64_7.58e+40.bin", "rb");
    } else if (uplo == 'U') {
        pFileA = fopen("matrices/unn/A_unn_100_3.08e+07.bin", "rb");
        pFileb = fopen("matrices/unn/b_unn_100_3.08e+07.bin", "rb");
    }
    if ((pFileA == NULL) || (pFileb == NULL)) {
        fprintf(stderr, "Cannot open files to read matrix and vector\n");
        exit(1);
    }

    resA = fread(a, sizeof(double), n * n, pFileA);
    resb = fread(xorig, sizeof(double), n, pFileb);
    if ((resA != n * n) || (resb != n)) {
        fprintf(stderr, "Cannot read matrix and vector from files\n");
        exit(1);
    }

    //for (int i = 0; i < n; i++)
    //    a[i * (n + 1)] = 1.0;

    fclose(pFileA);
    fclose(pFileb);
#else
    if(lognormal) {
        printf("init_lognormal_matrix\n");
        init_lognormal_matrix(uplo, 'N', a, n, mean, stddev);
        init_lognormal(xorig, n, mean, stddev);
    } else if ((argc > 5) && (argv[5][0] == 'i')) {
        printf("init_ill_cond\n");
        init_ill_cond(a, n * n, range);
        init_ill_cond(xorig, n, range);
    } else {
        printf("init_fpuniform_matrix\n");
        init_fpuniform_matrix(uplo, 'N', a, n, range, emax);
        init_fpuniform(xorig, n, range, emax);
    }
#endif
    copyVector(n, x, xorig);

    fprintf(stderr, "%d x %d\n", n, n);

    bool is_pass = true;
    double *superacc;
    double norm;
    err = posix_memalign((void **) &superacc, 64, n * sizeof(double));
    if ((!superacc) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");

    /*copyVector(n, superacc, xorig);
    extrsv(uplo, 'N', 'N', n, a, n, superacc, 1, 0);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, superacc, n, a, n, xorig, 1);
    printf("Superacc error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#endif
    */

    copyVector(n, x, xorig);
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
    exit(0);

    copyVector(n, x, xorig);
    extrsv(uplo, 'N', 'N', n, a, n, x, 1, 1);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
    printf("FPE IR error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#endif
    exit(0);

    copyVector(n, x, xorig);
    extrsv(uplo, 'N', 'N', n, a, n, x, 1, 3);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
    printf("FPE3 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv(uplo, 'N', 'N', n, a, n, x, 1, 4);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
    printf("FPE4 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv(uplo, 'N', 'N', n, a, n, x, 1, 8);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
    printf("FPE8 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv(uplo, 'N', 'N', n, a, n, x, 1, 4, true);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
    printf("FPE4EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv(uplo, 'N', 'N', n, a, n, x, 1, 6, true);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
    printf("FPE6EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv(uplo, 'N', 'N', n, a, n, x, 1, 8, true);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
    printf("FPE8EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
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

