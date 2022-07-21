/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>

#include "UDTypes.h"

#include "blas1.hpp"
#include "common.hpp"

#define max(x, y) ((x < y) ? y : x)
#define min(x, y) ((x > y) ? y : x)

#define PI 3.14159265359

float kernel_value_CPU(float v)
{
    float rValue = 0;

    const float z = v * v;

    // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
    float num = (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * (z * 0.210580722890567e-22f +
                                                                                  0.380715242345326e-19f) +
                                                                             0.479440257548300e-16f) +
                                                                        0.435125971262668e-13f) +
                                                                   0.300931127112960e-10f) +
                                                              0.160224679395361e-7f) +
                                                         0.654858370096785e-5f) +
                                                    0.202591084143397e-2f) +
                                               0.463076284721000e0f) +
                                          0.754337328948189e2f) +
                                     0.830792541809429e4f) +
                                0.571661130563785e6f) +
                           0.216415572361227e8f) +
                      0.356644482244025e9f) +
                 0.144048298227235e10f);

    float den = (z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f) - 0.144048298227235e10f);

    rValue = -num / den;

    return rValue;
}

void calculate_LUT_seq(float beta, float width, float **LUT, unsigned int *sizeLUT)
{
    float v;
    float cutoff2 = (width * width) / 4.0;

    unsigned int size;

    if (width > 0) {
        // compute size of LUT based on kernel width
        size = (unsigned int) (10000 * width);

        // allocate memory
        (*LUT) = (float *) malloc(size * sizeof(float));

        unsigned int k;
        for (k = 0; k < size; ++k) {
            // compute value to evaluate kernel at
            // v in the range 0:(_width/2)^2
            v = (((float) k) / ((float) size)) * cutoff2;

            // compute kernel value and store
            (*LUT)[k] = kernel_value_CPU(beta * sqrt(1.0 - (v / cutoff2)));
        }
        (*sizeLUT) = size;
    }
}

void calculate_LUT_omp(float beta, float width, float **LUT, unsigned int *sizeLUT)
{
    float v;
    float cutoff2 = (width * width) / 4.0;

    unsigned int size;

    if (width > 0) {
        // compute size of LUT based on kernel width
        size = (unsigned int) (10000 * width);

        // allocate memory
        (*LUT) = (float *) malloc(size * sizeof(float));

#pragma omp parallel for default(none) \
            private(v) \
            shared(size, cutoff2, LUT, beta)
        for (unsigned int k = 0; k < size; ++k) {
            // compute value to evaluate kernel at
            // v in the range 0:(_width/2)^2
            v = (((float) k) / ((float) size)) * cutoff2;

            // compute kernel value and store
            (*LUT)[k] = kernel_value_CPU(beta * sqrt(1.0 - (v / cutoff2)));
        }
        (*sizeLUT) = size;
    }
}

float kernel_value_LUT(float v, float *LUT, int sizeLUT, float _1overCutoff2)
{
    unsigned int k0;
    float v0;

    v *= (float) sizeLUT;
    k0 = (unsigned int) (v * _1overCutoff2);
    v0 = ((float) k0) / _1overCutoff2;
    return LUT[k0] + ((v - v0) * (LUT[k0 + 1] - LUT[k0]) / _1overCutoff2);
}

void gridding_seq(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT, unsigned int sizeLUT,
                  cmplx *gridData, float *sampleDensity, bool reproducible, double *time,
                  const int fpe = 0, const bool early_exit = false)
{
    printf("Sequential (gold) implementation: %s!\n", reproducible ? "reproducible" : "non-reproducible");

    unsigned int NxL, NxH;
    unsigned int NyL, NyH;
    unsigned int NzL, NzH;

    int nx;
    int ny;
    int nz;

    float w;
    unsigned int idx;
    unsigned int idx0;

    unsigned int idxZ;
    unsigned int idxY;

    float Dx2[100];
    float Dy2[100];
    float Dz2[100];
    float *dx2 = NULL;
    float *dy2 = NULL;
    float *dz2 = NULL;

    float dy2dz2;
    float v;

    unsigned int size_x = params.gridSize[0];
    unsigned int size_y = params.gridSize[1];
    unsigned int size_z = params.gridSize[2];

    float cutoff = ((float) (params.kernelWidth)) / 2.0; // cutoff radius
    float cutoff2 = cutoff * cutoff;                     // square of cutoff radius
    float _1overCutoff2 = 1 / cutoff2;                   // 1 over square of cutoff radius

    double tstart = omp_get_wtime();

    float beta = PI * sqrt(4 * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - .5) * (params.oversample - .5) - .8);

    unsigned int gridNumElems = size_x * size_y * size_z;

    cmplx_dbl_elements *dbl_gridDataElements = NULL;

    if (reproducible) {
        dbl_gridDataElements = (cmplx_dbl_elements *) calloc(size_x * size_y * size_z, sizeof(cmplx_dbl_elements));
    }

    // reproducible implementation needs a precomputation step
    // to allocate temporary arrays for storing precomputed data before summation
    //
    int steps = reproducible ? 2 : 1;

    for (int step = 0; step < steps; ++step)
    {
        for (int i = 0; i < n; i++) {
            ReconstructionSample pt = sample[i];

            float kx = pt.kX;
            float ky = pt.kY;
            float kz = pt.kZ;

            NxL = max((kx - cutoff), 0.0);
            NxH = min((kx + cutoff), size_x - 1.0);

            NyL = max((ky - cutoff), 0.0);
            NyH = min((ky + cutoff), size_y - 1.0);

            NzL = max((kz - cutoff), 0.0);
            NzH = min((kz + cutoff), size_z - 1.0);

            if ((pt.real != 0.0 || pt.imag != 0.0) && pt.sdc != 0.0) {
                for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2) {
                    *dz2 = ((kz - nz) * (kz - nz));
                }
                for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2) {
                    *dx2 = ((kx - nx) * (kx - nx));
                }
                for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2) {
                    *dy2 = ((ky - ny) * (ky - ny));
                }

                idxZ = (NzL - 1) * size_x * size_y;
                for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2) {
                    /* linear offset into 3-D matrix to get to zposition */
                    idxZ += size_x * size_y;

                    idxY = (NyL - 1) * size_x;

                    /* loop over x indexes, but only if curent distance is close enough (distance will increase by adding
                    * x&y distance) */
                    if ((*dz2) < cutoff2) {
                        for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2) {
                            /* linear offset IN ADDITION to idxZ to get to Y position */
                            idxY += size_x;

                            dy2dz2 = (*dz2) + (*dy2);

                            idx0 = idxY + idxZ;

                            /* loop over y indexes, but only if curent distance is close enough (distance will increase by
                            * adding y distance) */
                            if (dy2dz2 < cutoff2) {
                                for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2) {
                                    /* value to evaluate kernel at */
                                    v = dy2dz2 + (*dx2);

                                    if (v < cutoff2) {
                                        /* linear index of (x,y,z) point */
                                        idx = nx + idx0;

                                        if (!reproducible || step == 1)
                                        {
                                            /* kernel weighting value */
                                            if (params.useLUT) {
                                                w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt.sdc;
                                            } else {
                                                w = kernel_value_CPU(beta * sqrt(1.0 - (v * _1overCutoff2))) * pt.sdc;
                                            }
                                        }

                                        /* grid data */
                                        if (!reproducible)
                                        {
                                            gridData[idx].real += (w * pt.real);
                                            gridData[idx].imag += (w * pt.imag);
                                        }
                                        else if (step == 1)
                                        {
                                            dbl_gridDataElements[idx].real_elements[(size_t) sampleDensity[idx]] = (w * pt.real);
                                            dbl_gridDataElements[idx].imag_elements[(size_t) sampleDensity[idx]] = (w * pt.imag);
                                        }

                                        /* estimate sample density */
                                        sampleDensity[idx] += 1.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (reproducible && step == 0)
        {
            for (idx = 0; idx < gridNumElems; ++idx)
            {
                // allocate elements for summation only for points that have samples
                if (sampleDensity[idx] > 0) {
                    dbl_gridDataElements[idx].real_elements = (double *) malloc((size_t) sampleDensity[idx] * sizeof(double));
                    dbl_gridDataElements[idx].imag_elements = (double *) malloc((size_t) sampleDensity[idx] * sizeof(double));
                }

                // reset sample density to keep track of elements below
                sampleDensity[idx] = 0;
            }
        }
    }

    if (reproducible)
    {
        // compute grid data by summing calculated contributions
        for (idx = 0; idx < gridNumElems; ++idx)
        {
            if (sampleDensity[idx] > 0) {
                gridData[idx].real = static_cast<float> (exsum(
                    /* Ng */            (int) sampleDensity[idx],
                    /* ag */            dbl_gridDataElements[idx].real_elements,
                    /* inca */          1,
                    /* offset */        0,
                    /* fpe */           fpe,
                    /* early_exit */    early_exit,
                    /* parallel */      false));

                gridData[idx].imag = static_cast<float> (exsum(
                    /* Ng */            (int) sampleDensity[idx],
                    /* ag */            dbl_gridDataElements[idx].imag_elements,
                    /* inca */          1,
                    /* offset */        0,
                    /* fpe */           fpe,
                    /* early_exit */    early_exit,
                    /* parallel */      false));

                // cleanup
                free(dbl_gridDataElements[idx].real_elements);
                free(dbl_gridDataElements[idx].imag_elements);
            }
        }

        // cleanup
        free(dbl_gridDataElements);
    }

    *time = omp_get_wtime() - tstart;
}

/**
 * This implementation uses thread-local copies of output arrays and parallel reduction therefore avoiding thread
 * synchronization at the cost of higher memory consumption. Since the CPUs do not have thousands of threads like GPUs
 * and output arrays are relatively small for the provided example (~150MB), memory consumption may not be a problem.
 * On the other hand, for bigger data sets or a much greater amount of CPU threads this probably is not an acceptable
 * solution. Regarding numerical reproducibility, this solution exibits weak reproducibility, i.e. for same thread
 * count it returns the same result every time. However, the returned result may not be accurate.
 */
void gridding_omp_mem(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT, unsigned int sizeLUT,
                      cmplx *gridData, float *sampleDensity, double *time)
{
    printf("Parallel implementation using thread-local output buffers: non-reproducible!\n");

    int numThreads = omp_get_max_threads();

    if (numThreads == 1) {
        gridding_seq(n, params, sample, LUT, sizeLUT, gridData, sampleDensity, /* reproducible */ false, time);
        return;
    }

    unsigned int NxL, NxH;
    unsigned int NyL, NyH;
    unsigned int NzL, NzH;

    int nx;
    int ny;
    int nz;

    float w;
    unsigned int idx;
    unsigned int idx0;

    unsigned int idxZ;
    unsigned int idxY;

    float Dx2[100];
    float Dy2[100];
    float Dz2[100];
    float *dx2 = NULL;
    float *dy2 = NULL;
    float *dz2 = NULL;

    float dy2dz2;
    float v;

    unsigned int size_x = params.gridSize[0];
    unsigned int size_y = params.gridSize[1];
    unsigned int size_z = params.gridSize[2];

    float cutoff = ((float) (params.kernelWidth)) / 2.0; // cutoff radius
    float cutoff2 = cutoff * cutoff;                     // square of cutoff radius
    float _1overCutoff2 = 1 / cutoff2;                   // 1 over square of cutoff radius

    float beta = PI * sqrt(4 * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - .5) * (params.oversample - .5) - .8);

    int i;

    /* initialize thread-private copies of output data */

    ReconstructionSample pt;

    int tid;
    int chunkSize = (n + numThreads - 1) / numThreads;
    int start;
    int end;

    int gridNumElems = size_x * size_y * size_z;

    cmplx **tGridData;
    float **tSampleDensity;

    tGridData = (cmplx **) calloc(numThreads, sizeof(cmplx *));

    /* no need to allocate output buffers for thread with id 0, use already allocated buffers */
    tGridData[0] = gridData;

    /* other threads require output buffer allocation */
    for (i = 1; i < numThreads; ++i) {
        tGridData[i] = (cmplx *) calloc(gridNumElems, sizeof(cmplx));
    }

    tSampleDensity = (float **) calloc(numThreads, sizeof(float *));

    /* no need to allocate output buffers for thread with id 0, use already allocated buffers */
    tSampleDensity[0] = sampleDensity;

    /* other threads require output buffer allocation */
    for (i = 1; i < numThreads; ++i) {
        tSampleDensity[i] = (float *) calloc(gridNumElems, sizeof(float));
    }

    double tstart = omp_get_wtime();

#pragma omp parallel default(none) \
        private(i, tid, pt, NxL, NxH, NyL, NyH, NzL, NzH, nx, ny, nz, w, idx, idx0, idxZ, idxY, Dx2, Dy2, Dz2, dx2, dy2, dz2, dy2dz2, v, start, end) \
        shared(n, params, sample, LUT, sizeLUT, size_x, size_y, size_z, cutoff, cutoff2, _1overCutoff2, beta, tGridData, tSampleDensity, chunkSize, numThreads)
{
    tid = omp_get_thread_num();
    start = tid * chunkSize;
    end = start + chunkSize > n ? n : start + chunkSize;

    for (i = start; i < end; i++) {
        pt = sample[i];

        NxL = max((pt.kX - cutoff), 0.0);
        NxH = min((pt.kX + cutoff), size_x - 1.0);

        NyL = max((pt.kY - cutoff), 0.0);
        NyH = min((pt.kY + cutoff), size_y - 1.0);

        NzL = max((pt.kZ - cutoff), 0.0);
        NzH = min((pt.kZ + cutoff), size_z - 1.0);

        if ((pt.real != 0.0 || pt.imag != 0.0) && pt.sdc != 0.0) {
            for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2) {
                *dz2 = ((pt.kZ - nz) * (pt.kZ - nz));
            }
            for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2) {
                *dx2 = ((pt.kX - nx) * (pt.kX - nx));
            }
            for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2) {
                *dy2 = ((pt.kY - ny) * (pt.kY - ny));
            }

            idxZ = (NzL - 1) * size_x * size_y;
            for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2) {
                /* linear offset into 3-D matrix to get to zposition */
                idxZ += size_x * size_y;

                idxY = (NyL - 1) * size_x;

                /* loop over x indexes, but only if curent distance is close enough (distance will increase by adding
                 * x&y distance) */
                if ((*dz2) < cutoff2) {
                    for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2) {
                        /* linear offset IN ADDITION to idxZ to get to Y position */
                        idxY += size_x;

                        dy2dz2 = (*dz2) + (*dy2);

                        idx0 = idxY + idxZ;

                        /* loop over y indexes, but only if curent distance is close enough (distance will increase by
                         * adding y distance) */
                        if (dy2dz2 < cutoff2) {
                            for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2) {
                                /* value to evaluate kernel at */
                                v = dy2dz2 + (*dx2);

                                if (v < cutoff2) {
                                    /* linear index of (x,y,z) point */
                                    idx = nx + idx0;

                                    /* kernel weighting value */
                                    if (params.useLUT) {
                                        w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt.sdc;
                                    } else {
                                        w = kernel_value_CPU(beta * sqrt(1.0 - (v * _1overCutoff2))) * pt.sdc;
                                    }

                                    /* grid data */
                                    tGridData[tid][idx].real += (w * pt.real);
                                    tGridData[tid][idx].imag += (w * pt.imag);

                                    /* estimate sample density */
                                    tSampleDensity[tid][idx] += 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

    *time = omp_get_wtime() - tstart;

    /* parallel reduction, thread with id 0 uses function output buffers */

    chunkSize = (gridNumElems + numThreads - 1) / numThreads;
    int j;
#pragma omp parallel default(none) \
            private(i, j, tid, start, end) \
            shared(gridNumElems, numThreads, gridData, tGridData, sampleDensity, tSampleDensity, chunkSize)
{
    tid = omp_get_thread_num();
    start = tid * chunkSize;
    end = start + chunkSize > gridNumElems ? gridNumElems : start + chunkSize;

    for (i = start; i < end; ++i) {
        for (j = 1; j < numThreads; ++j) {
            gridData[i].real += tGridData[j][i].real;
            gridData[i].imag += tGridData[j][i].imag;
            sampleDensity[i] += tSampleDensity[j][i];
        }
    }
}

    /* free allocated memory */
    for (i = 1; i < numThreads; ++i) {
        free(tGridData[i]);
    }
    free(tGridData);
    for (i = 1; i < numThreads; ++i) {
        free(tSampleDensity[i]);
    }
    free(tSampleDensity);
}

constexpr int MAX_LOCK_COUNT = 10000;

/**
 * This implementation uses locks that serve as mutexes for multiple output elements. The number of output elements is
 * very large therefore they need to share locks to maintain performance and control memory usage. Again, scheduling is
 * static because it incurs the least performance penalty. The load is not balanced but using OpenMP to balance the load
 * can only incur more performance penalty. Due to the nature of locks and thread contention, this solution is not
 * reproducible and multiple runs with same thread count do not guarantee numerical reproducibility.
 */
void gridding_omp_locks(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT, unsigned int sizeLUT,
                        cmplx *gridData, float *sampleDensity, double *time)
{
    printf("Parallel implementation using locks: non-reproducible.\n");

    int numThreads = omp_get_max_threads();

    if (numThreads == 1) {
        gridding_seq(n, params, sample, LUT, sizeLUT, gridData, sampleDensity, /* reproducible */ false, time);
        return;
    }

    unsigned int NxL, NxH;
    unsigned int NyL, NyH;
    unsigned int NzL, NzH;

    int nx;
    int ny;
    int nz;

    float w;
    unsigned int idx;
    unsigned int idx0;

    unsigned int idxZ;
    unsigned int idxY;

    float Dx2[100];
    float Dy2[100];
    float Dz2[100];
    float *dx2 = NULL;
    float *dy2 = NULL;
    float *dz2 = NULL;

    float dy2dz2;
    float v;

    unsigned int size_x = params.gridSize[0];
    unsigned int size_y = params.gridSize[1];
    unsigned int size_z = params.gridSize[2];

    float cutoff = ((float) (params.kernelWidth)) / 2.0; // cutoff radius
    float cutoff2 = cutoff * cutoff;                     // square of cutoff radius
    float _1overCutoff2 = 1 / cutoff2;                   // 1 over square of cutoff radius

    double tstart = omp_get_wtime();

    float beta = PI * sqrt(4 * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - .5) * (params.oversample - .5) - .8);

    int i;

    /* initialize required variables - locks, etc. */

    ReconstructionSample pt;

    float tempGridDataReal;
    float tempGridDataImag;

    unsigned int gridNumElems = size_x * size_y * size_z;
    unsigned int lockNum = MAX_LOCK_COUNT > gridNumElems ? gridNumElems : MAX_LOCK_COUNT;

    omp_lock_t *locks = (omp_lock_t *) calloc(lockNum, sizeof(omp_lock_t));
    if (locks == NULL) {
        printf("ERROR: Unable to allocate memory for locks\n");
        exit(1);
    }

    for (i = 0; i < lockNum; ++i) {
        omp_init_lock(&locks[i]);
    }

#pragma omp parallel default(none) \
            private(i, pt, NxL, NxH, NyL, NyH, NzL, NzH, nx, ny, nz, w, tempGridDataReal, tempGridDataImag, idx, idx0, idxZ, idxY, Dx2, Dy2, Dz2, dx2, dy2, dz2, dy2dz2, v) \
            shared(n, params, sample, LUT, sizeLUT, gridData, sampleDensity, size_x, size_y, size_z, cutoff, cutoff2, _1overCutoff2, beta, lockNum, locks)
{

#pragma omp for
    for (i = 0; i < n; i++) {
        pt = sample[i];

        NxL = max((pt.kX - cutoff), 0.0);
        NxH = min((pt.kX + cutoff), size_x - 1.0);

        NyL = max((pt.kY - cutoff), 0.0);
        NyH = min((pt.kY + cutoff), size_y - 1.0);

        NzL = max((pt.kZ - cutoff), 0.0);
        NzH = min((pt.kZ + cutoff), size_z - 1.0);

        if ((pt.real != 0.0 || pt.imag != 0.0) && pt.sdc != 0.0) {
            for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2) {
                *dz2 = ((pt.kZ - nz) * (pt.kZ - nz));
            }
            for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2) {
                *dx2 = ((pt.kX - nx) * (pt.kX - nx));
            }
            for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2) {
                *dy2 = ((pt.kY - ny) * (pt.kY - ny));
            }

            idxZ = (NzL - 1) * size_x * size_y;
            for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2) {
                /* linear offset into 3-D matrix to get to zposition */
                idxZ += size_x * size_y;

                idxY = (NyL - 1) * size_x;

                /* loop over x indexes, but only if curent distance is close enough (distance will increase by adding
                 * x&y distance) */
                if ((*dz2) < cutoff2) {
                    for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2) {
                        /* linear offset IN ADDITION to idxZ to get to Y position */
                        idxY += size_x;

                        dy2dz2 = (*dz2) + (*dy2);

                        idx0 = idxY + idxZ;

                        /* loop over y indexes, but only if curent distance is close enough (distance will increase by
                         * adding y distance) */
                        if (dy2dz2 < cutoff2) {
                            for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2) {
                                /* value to evaluate kernel at */
                                v = dy2dz2 + (*dx2);

                                if (v < cutoff2) {
                                    /* linear index of (x,y,z) point */
                                    idx = nx + idx0;

                                    /* kernel weighting value */
                                    if (params.useLUT) {
                                        w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt.sdc;
                                    } else {
                                        w = kernel_value_CPU(beta * sqrt(1.0 - (v * _1overCutoff2))) * pt.sdc;
                                    }

                                    /* multiply outside critical region to minimize time spent inside */
                                    tempGridDataReal = w * pt.real;
                                    tempGridDataImag = w * pt.imag;

                                    /* enter critical region */
                                    omp_set_lock(&locks[idx % lockNum]);

                                    /* grid data */
                                    gridData[idx].real += tempGridDataReal;
                                    gridData[idx].imag += tempGridDataImag;

                                    /* estimate sample density */
                                    sampleDensity[idx] += 1.0;

                                    /* exit critical region */
                                    omp_unset_lock(&locks[idx % lockNum]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

    *time = omp_get_wtime() - tstart;

    /* free allocated memory */
    for (i = 0; i < lockNum; ++i) {
        omp_destroy_lock(&locks[i]);
    }
    free(locks);
}

void gridding_omp(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT, unsigned int sizeLUT,
                  cmplx *gridData, float *sampleDensity, bool reproducible, double *time,
                  const int fpe = 0, const bool early_exit = false)
{
    printf("Parallel implementation: %s!\n", reproducible ? "reproducible" : "non-reproducible");

    unsigned int numThreads = omp_get_max_threads();

    if (numThreads == 1) {
        gridding_seq(n, params, sample, LUT, sizeLUT, gridData, sampleDensity, reproducible, time, fpe, early_exit);
        return;
    }

    unsigned int chunkSize = (n + numThreads - 1) / numThreads;

    unsigned int size_x = params.gridSize[0];
    unsigned int size_y = params.gridSize[1];
    unsigned int size_z = params.gridSize[2];

    unsigned int gridNumElems = size_x * size_y * size_z;

    float cutoff = ((float) (params.kernelWidth)) / 2.0; // cutoff radius
    float cutoff2 = cutoff * cutoff;                     // square of cutoff radius
    float _1overCutoff2 = 1 / cutoff2;                   // 1 over square of cutoff radius

    float beta = PI * sqrt(4 * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - .5) * (params.oversample - .5) - .8);

    double tstart = omp_get_wtime();

    cmplx_elements *gridDataElements = NULL;
    unsigned int **tSampleDensity = NULL;

    /* reproducible */
    cmplx_dbl_elements *dbl_gridDataElements = NULL;

    if (reproducible) {
        dbl_gridDataElements = (cmplx_dbl_elements *) calloc(size_x * size_y * size_z, sizeof(cmplx_dbl_elements));
    } else {
        gridDataElements = (cmplx_elements *) calloc(size_x * size_y * size_z, sizeof(cmplx_elements));
    }

    tSampleDensity = (unsigned int **) malloc(gridNumElems * sizeof(unsigned int *));
    for (unsigned int idx = 0; idx < gridNumElems; ++idx) {
        tSampleDensity[idx] = (unsigned int *) calloc(numThreads, sizeof(unsigned int));
    }

#pragma omp parallel default(none) \
            shared(n, params, sample, LUT, sizeLUT, gridData, sampleDensity, fpe, early_exit, numThreads, chunkSize, size_x, size_y, size_z, gridNumElems, cutoff, cutoff2, _1overCutoff2, beta, reproducible, gridDataElements, tSampleDensity, dbl_gridDataElements)
{
    unsigned int NxL, NxH;
    unsigned int NyL, NyH;
    unsigned int NzL, NzH;

    int nx;
    int ny;
    int nz;

    float w;
    unsigned int idx;
    unsigned int idx0;

    unsigned int idxZ;
    unsigned int idxY;

    float Dx2[100];
    float Dy2[100];
    float Dz2[100];
    float *dx2 = NULL;
    float *dy2 = NULL;
    float *dz2 = NULL;

    float dy2dz2;
    float v;

    ReconstructionSample pt;

    unsigned int tid = omp_get_thread_num();;
    unsigned int start = tid * chunkSize;
    unsigned int end = start + chunkSize;

    unsigned int sum_prev;
    unsigned int sum;

    if (end > n) {
        end = n;
    }

    for (int step = 0; step < 2; ++step)
    {
        for (int i = start; i < end; i++) {
            pt = sample[i];

            NxL = max((pt.kX - cutoff), 0.0);
            NxH = min((pt.kX + cutoff), size_x - 1.0);

            NyL = max((pt.kY - cutoff), 0.0);
            NyH = min((pt.kY + cutoff), size_y - 1.0);

            NzL = max((pt.kZ - cutoff), 0.0);
            NzH = min((pt.kZ + cutoff), size_z - 1.0);

            if ((pt.real != 0.0 || pt.imag != 0.0) && pt.sdc != 0.0) {
                for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2) {
                    *dz2 = ((pt.kZ - nz) * (pt.kZ - nz));
                }
                for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2) {
                    *dx2 = ((pt.kX - nx) * (pt.kX - nx));
                }
                for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2) {
                    *dy2 = ((pt.kY - ny) * (pt.kY - ny));
                }

                idxZ = (NzL - 1) * size_x * size_y;
                for (dz2 = Dz2, nz = NzL; nz <= NzH; ++nz, ++dz2) {
                    /* linear offset into 3-D matrix to get to zposition */
                    idxZ += size_x * size_y;

                    idxY = (NyL - 1) * size_x;

                    /* loop over x indexes, but only if curent distance is close enough (distance will increase by adding
                    * x&y distance) */
                    if ((*dz2) < cutoff2) {
                        for (dy2 = Dy2, ny = NyL; ny <= NyH; ++ny, ++dy2) {
                            /* linear offset IN ADDITION to idxZ to get to Y position */
                            idxY += size_x;

                            dy2dz2 = (*dz2) + (*dy2);

                            idx0 = idxY + idxZ;

                            /* loop over y indexes, but only if curent distance is close enough (distance will increase by
                            * adding y distance) */
                            if (dy2dz2 < cutoff2) {
                                for (dx2 = Dx2, nx = NxL; nx <= NxH; ++nx, ++dx2) {
                                    /* value to evaluate kernel at */
                                    v = dy2dz2 + (*dx2);

                                    if (v < cutoff2) {
                                        /* linear index of (x,y,z) point */
                                        idx = nx + idx0;

                                        if (step == 1)
                                        {
                                            /* kernel weighting value */
                                            if (params.useLUT) {
                                                w = kernel_value_LUT(v, LUT, sizeLUT, _1overCutoff2) * pt.sdc;
                                            } else {
                                                w = kernel_value_CPU(beta * sqrt(1.0 - (v * _1overCutoff2))) * pt.sdc;
                                            }

                                            /* grid data */
                                            if (reproducible) {
                                                dbl_gridDataElements[idx].real_elements[tSampleDensity[idx][tid]] = w * pt.real;
                                                dbl_gridDataElements[idx].imag_elements[tSampleDensity[idx][tid]] = w * pt.imag;
                                            } else {
                                                gridDataElements[idx].real_elements[tSampleDensity[idx][tid]] = w * pt.real;
                                                gridDataElements[idx].imag_elements[tSampleDensity[idx][tid]] = w * pt.imag;
                                            }
                                        }

                                        /* estimate sample density */
                                        tSampleDensity[idx][tid]++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

#pragma omp barrier

        if (step == 0)
        {
#pragma omp for
            for (idx = 0; idx < gridNumElems; ++idx) {
                sum_prev = 0;
                sum = 0;
                for (int tidx = 0; tidx < numThreads; ++tidx) {
                    sum += tSampleDensity[idx][tidx];
                    tSampleDensity[idx][tidx] = sum_prev;
                    sum_prev = sum;
                }

                /* set sample density now */
                sampleDensity[idx] = (float) sum;

                // allocate elements for summation only for points that have samples
                if (sum > 0) {
                    if (reproducible) {
                        dbl_gridDataElements[idx].real_elements = (double *) malloc(sum * sizeof(double));
                        dbl_gridDataElements[idx].imag_elements = (double *) malloc(sum * sizeof(double));
                    } else {
                        gridDataElements[idx].real_elements = (float *) malloc(sum * sizeof(float));
                        gridDataElements[idx].imag_elements = (float *) malloc(sum * sizeof(float));
                    }
                }
            }
        }
        else
        {
#pragma omp for
            for (idx = 0; idx < gridNumElems; ++idx) {
                if (sampleDensity[idx] > 0) {
                    if (reproducible) {
                        gridData[idx].real = static_cast<float> (exsum(
                            /* Ng */            (int) sampleDensity[idx],
                            /* ag */            dbl_gridDataElements[idx].real_elements,
                            /* inca */          1,
                            /* offset */        0,
                            /* fpe */           fpe,
                            /* early_exit */    early_exit,
                            /* parallel */      false));

                        gridData[idx].imag = static_cast<float> (exsum(
                            /* Ng */            (int) sampleDensity[idx],
                            /* ag */            dbl_gridDataElements[idx].imag_elements,
                            /* inca */          1,
                            /* offset */        0,
                            /* fpe */           fpe,
                            /* early_exit */    early_exit,
                            /* parallel */      false));

                        // cleanup
                        free(dbl_gridDataElements[idx].real_elements);
                        free(dbl_gridDataElements[idx].imag_elements);
                    } else {
                        for (int i = 0; i < (int) sampleDensity[idx]; ++i) {
                            gridData[idx].real += gridDataElements[idx].real_elements[i];
                            gridData[idx].imag += gridDataElements[idx].imag_elements[i];
                        }

                        // cleanup
                        free(gridDataElements[idx].real_elements);
                        free(gridDataElements[idx].imag_elements);
                    }
                }
            }
        }
    }
}

    if (reproducible) {
        free(dbl_gridDataElements);
    } else {
        free(gridDataElements);
    }

    *time = omp_get_wtime() - tstart;
}