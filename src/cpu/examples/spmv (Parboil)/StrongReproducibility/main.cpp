/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "parboil.h"

#include <cstdio>
#include <cstring>
#include <random>

#include "file.h"
#include "convert_dataset.h"

#include <omp.h>
#include "../../../../config.h"
#include <binned.h>

using namespace std;

constexpr uint32_t DEFAULT_SEED = 1549813198;
constexpr uint32_t DEFAULT_NUMBER_OF_RUNS = 50;

bool generate_vector (float *x_vector, int dim, uint32_t seed)
{
    if (nullptr == x_vector) {
        return false;
    }

    mt19937 gen(seed);
    uniform_real_distribution<float> float_dist(0.f, 1.f);

    for (int i = 0; i < dim; i++) {
        x_vector[i] = float_dist(gen);
    }

    return true;
}

bool diff(int dim, float *h_Ax_vector_1, float *h_Ax_vector_2)
{
    for (int i = 0; i < dim; i++)
        if (h_Ax_vector_1[i] != h_Ax_vector_2[i])
            return true;
    return false;
}

void spmv_seq (bool reproducible, int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
               float *h_x_vector, int *h_perm, float *h_Ax_vector)
{
    float sum = 0.0f;
    float_binned* sum_binned = binned_sballoc(SIDEFAULTFOLD);

    // Consider creating a random map by creating an array 0..dim - 1 and randomly shuffling it
    // for each execution. This should provide required randomness given the order of operations
    // is sequential at the moment.
    //
    for (int i = 0; i < dim; i++) {
        if (reproducible) {
            binned_sbsetzero(SIDEFAULTFOLD, sum_binned);
        } else {
            sum = 0.0f;
        }

        int bound = h_nzcnt[i];

        for (int k = 0; k < bound; k++) {
            int j = h_ptr[k] + i;
            int in = h_indices[j];

            float d = h_data[j];
            float t = h_x_vector[in];

            if (reproducible) {
                binned_sbsadd(SIDEFAULTFOLD, d * t, sum_binned);
            } else {
                sum += d * t;
            }
        }

        if (reproducible) {
            h_Ax_vector[h_perm[i]] = binned_ssbconv(SIDEFAULTFOLD, sum_binned);
        } else {
            h_Ax_vector[h_perm[i]] = sum;
        }
    }

    free(sum_binned);
}

void spmv_omp (bool reproducible, int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
               float *h_x_vector, int *h_perm, float *h_Ax_vector)
{
#pragma omp parallel
{
    float sum = 0.0f;
    float_binned* sum_binned = binned_sballoc(SIDEFAULTFOLD);

    // Consider creating a random map by creating an array 0..dim - 1 and randomly shuffling it
    // for each execution. This should provide required randomness given the order of operations
    // is sequential at the moment.
    //
#pragma omp for
    for (int i = 0; i < dim; i++) {
        if (reproducible) {
            binned_sbsetzero(SIDEFAULTFOLD, sum_binned);
        } else {
            sum = 0.0f;
        }

        int bound = h_nzcnt[i];

        for (int k = 0; k < bound; k++) {
            int j = h_ptr[k] + i;
            int in = h_indices[j];

            float d = h_data[j];
            float t = h_x_vector[in];

            if (reproducible) {
                binned_sbsadd(SIDEFAULTFOLD, d * t, sum_binned);
            } else {
                sum += d * t;
            }
        }

        if (reproducible) {
            h_Ax_vector[h_perm[i]] = binned_ssbconv(SIDEFAULTFOLD, sum_binned);
        } else {
            h_Ax_vector[h_perm[i]] = sum;
        }
    }

    free(sum_binned);
}
}

void execute (uint32_t nruns, bool parallel, bool reproducible, int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
              float *h_x_vector, int *h_perm, float *h_Ax_vector, double &time)
{
    printf("Running %s (%sreproducible) implementation...\n", parallel ? "parallel" : "sequential", reproducible ? "" : "non-");

    time = 0.0f;

    float *tmp_h_Ax_vector = new float[dim];

    for (int i = 0; i < nruns; ++i) {
        if (i == 0)
            time = omp_get_wtime();
        if (parallel)
            spmv_omp(reproducible, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, tmp_h_Ax_vector);
        else
            spmv_seq(reproducible, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, tmp_h_Ax_vector);
        if (i == 0) {
            time = omp_get_wtime() - time;
            memcpy (h_Ax_vector, tmp_h_Ax_vector, dim * sizeof (float));
        } else if (diff(dim, h_Ax_vector, tmp_h_Ax_vector)) {
            printf("%s (%sreproducible) implementation not reproducible after %d runs!\n",
                    parallel ? "Parallel" : "Sequential", reproducible ? "" : "non-", i);
            break;
        }
    }

    delete[] tmp_h_Ax_vector;
}

int main (int argc, char** argv)
{
    struct pb_TimerSet timers;
    struct pb_Parameters *parameters;
    
    printf ("CPU-based sparse matrix vector multiplication****\n");
    printf ("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
    printf ("This version maintained by Chris Rodrigues  ***********\n");

    parameters = pb_ReadParameters (&argc, argv);
    if (parameters->inpFiles[0] == NULL) {
        fprintf (stderr, "Expecting one input filename.\n");
        exit (-1);
    }

    pb_InitializeTimerSet (&timers);
    pb_SwitchToTimer (&timers, pb_TimerID_COMPUTE);

    // Parameters declaration
    //
    int len;
    int depth;
    int dim;
    int pad=1;
    int nzcnt_len;

    // Host memory allocation
    // Matrix
    //
    float *h_data;
    int *h_indices;
    int *h_ptr;
    int *h_perm;
    int *h_nzcnt;

    // Vector
    //
    float *h_Ax_vector_seq, *h_Ax_vector_seq_rep, *h_Ax_vector_omp, *h_Ax_vector_omp_rep;
    float *h_x_vector;

    // Load matrix from files
    //
    pb_SwitchToTimer (&timers, pb_TimerID_IO);

    int col_count;
    coo_to_jds(
        parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
        1, // row padding
        pad, // warp size
        1, // pack size
        1, // debug level [0:2]
        &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
        &col_count, &dim, &len, &nzcnt_len, &depth
    );

    h_Ax_vector_seq = new float[dim];
    h_Ax_vector_seq_rep = new float[dim];
    h_Ax_vector_omp = new float[dim];
    h_Ax_vector_omp_rep = new float[dim];

    h_x_vector = new float[dim];

    uint32_t seed = parameters->seed == 0 ? DEFAULT_SEED : parameters->seed;
    uint32_t nruns = parameters->nruns == 0 ? DEFAULT_NUMBER_OF_RUNS : parameters->nruns;

    if (!generate_vector(h_x_vector, dim, seed)) {
        fprintf(stderr, "Failed to generate dense vector.\n");
        exit(-1);
    }

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    double time_seq, time_omp, time_seq_rep, time_omp_rep;

    execute (nruns, false, false, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_seq, time_seq);
    execute (nruns, false, true, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_seq_rep, time_seq_rep);
    execute (nruns, true, false, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_omp, time_omp);
    execute (nruns, true, true, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_omp_rep, time_omp_rep);

    printf("Non-reproducible sequential and parallel results %smatch!\n",
           diff(dim, h_Ax_vector_seq, h_Ax_vector_omp) ? "do not " : "");

    printf("Reproducible sequential and parallel results %smatch!\n",
           diff(dim, h_Ax_vector_seq_rep, h_Ax_vector_omp_rep) ? "do not " : "");

    if (diff(dim, h_Ax_vector_seq, h_Ax_vector_seq_rep)) {
        printf("Non-reproducible and reproducible sequential results do not match!\n");
    }

    if (diff(dim, h_Ax_vector_omp, h_Ax_vector_omp_rep)) {
        printf("Non-reproducible and reproducible parallel results do not match!\n");
    }

    printf("Sequential implementation time: %.3f\n", time_seq);
    printf("Parallel implementation time: %.3f\n", time_omp);
    printf("Speedup: %.3f\n", time_seq / time_omp);

    printf("Sequential implementation time (reproducible): %.3f\n", time_seq_rep);
    printf("Parallel implementation time (reproducible): %.3f\n", time_omp_rep);
    printf("Speedup (reproducible): %.3f\n", time_seq_rep / time_omp_rep);

    printf("Time sequential reproducible / non-reproducible: %.3f\n", time_seq_rep / time_seq);
    printf("Time parallel reproducible / non-reproducible: %.3f\n", time_omp_rep / time_omp);

    delete[] h_data;
    delete[] h_indices;
    delete[] h_ptr;
    delete[] h_perm;
    delete[] h_nzcnt;
    delete[] h_Ax_vector_seq;
    delete[] h_Ax_vector_seq_rep;
    delete[] h_Ax_vector_omp;
    delete[] h_Ax_vector_omp_rep;
    delete[] h_x_vector;

    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);

    pb_FreeParameters(parameters);

    return 0;
}