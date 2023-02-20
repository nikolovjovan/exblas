/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

#include "parboil.h"
#include "file.h"
#include "gpu_info.h"
#include "ocl.h"
#include "convert_dataset.h"

#include "blas1.hpp"
#include "common.hpp"

using namespace std;

constexpr uint32_t DEFAULT_SEED = 1549813198;
constexpr uint32_t DEFAULT_NUMBER_OF_RUNS = 50;

constexpr char* input_files[] = { "fidapm05.mtx", "jgl009.mtx" }; // "bcsstk32.mtx" - takes > ~40 minutes to complete, ignoring this one

constexpr char* algorithm[] = { "accumulator-only", "fpe2", "fpe4", "fpe8ee" };
constexpr int fpe[] = { 0, 2, 4, 8 };
constexpr bool early_exit[] = { false, false, false, true };

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

void spmv_exsum (const int fpe, const bool early_exit,
                 int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
                 float *h_x_vector, int *h_perm, float *h_Ax_vector)
{
    double* dbl_products = (double*) malloc(dim * sizeof(double));

    // Consider creating a random map by creating an array 0..dim - 1 and randomly shuffling it
    // for each execution. This should provide required randomness given the order of operations
    // is sequential at the moment.
    //
    for (int i = 0; i < dim; i++) {
        int bound = h_nzcnt[i];

        for (int k = 0; k < bound; k++) {
            int j = h_ptr[k] + i;
            int in = h_indices[j];

            float d = h_data[j];
            float t = h_x_vector[in];

            dbl_products[k] = static_cast<double> (d) * t;
        }

        if (bound > 0)
        {
            // cout << "i: " << i << "\tbound = " << bound << '\n';

            h_Ax_vector[h_perm[i]] = static_cast<float> (exsum(
            /* Ng */            bound,
            /* ag */            dbl_products,
            /* inca */          1,
            /* offset */        0,
            /* fpe */           fpe,
            /* early_exit */    early_exit,
            /* parallel */      false));
        }
    }

    free(dbl_products);
}

void spmv_exdot (const int fpe, const bool early_exit,
                 int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
                 float *h_x_vector, int *h_perm, float *h_Ax_vector)
{
    double* dbl_row = (double*) malloc(dim * sizeof(double));
    double* dbl_vec = (double*) malloc(dim * sizeof(double));

    // Consider creating a random map by creating an array 0..dim - 1 and randomly shuffling it
    // for each execution. This should provide required randomness given the order of operations
    // is sequential at the moment.
    //
    for (int i = 0; i < dim; i++) {
        int bound = h_nzcnt[i];

        for (int k = 0; k < bound; k++) {
            int j = h_ptr[k] + i;
            int in = h_indices[j];

            float d = h_data[j];
            float t = h_x_vector[in];

            dbl_row[k] = static_cast<double> (d);
            dbl_vec[k] = static_cast<double> (t);
        }

        if (bound > 0)
        {
            // cout << "i: " << i << "\tbound = " << bound << '\n';

            h_Ax_vector[h_perm[i]] = static_cast<float> (exdot(
            /* Ng */            bound,
            /* ag */            dbl_row,
            /* inca */          1,
            /* offseta */       0,
            /* bg */            dbl_vec,
            /* incb */          1,
            /* offsetb */       0,
            /* fpe */           fpe,
            /* early_exit */    early_exit));
        }
    }

    free(dbl_row);
    free(dbl_vec);
}

int main(int argc, char **argv)
{
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
    float *h_Ax_vector;
    float *h_x_vector;

    const int exe_path_len = strrchr(argv[0], '/') - argv[0] + 1;
    char exe_path[256];
    strncpy(exe_path, argv[0], exe_path_len);
    exe_path[exe_path_len] = '\0';

    char input_file_path[256];

    cout << "unit: [ms]\n\n";

    chrono::steady_clock::time_point start;
    uint64_t time_first_setup = 0, time;

    double dummy_array[2] = { 1.0, 2.0 };

    start = chrono::steady_clock::now();
    exsum(2, dummy_array, 1, 0, 0, false, false);
    time_first_setup =
        chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

    cout << "OCL first (dummy) setup time : " << fixed << setprecision(10) << (float) time_first_setup / 1000.0 << endl << endl; // ms

    cout << "unit: [ms]\n\n";
    
    int filecnt = sizeof(input_files) / sizeof(char*);
    
    for (int i = 0; i < filecnt; ++i)
    {
        strncpy(input_file_path, exe_path, exe_path_len + 1);
        strcat(input_file_path, "data/");
        strcat(input_file_path, input_files[i]);

        cout << input_files[i] << "\n\n";

        int col_count;
        coo_to_jds(
            input_file_path,
            1, // row padding
            pad, // warp size
            1, // pack size
            0, // debug level [0:2]
            &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
            &col_count, &dim, &len, &nzcnt_len, &depth
        );

        h_x_vector = new float[dim];

        if (!generate_vector(h_x_vector, dim, DEFAULT_SEED)) {
            fprintf(stderr, "Failed to generate dense vector.\n");
            exit(-1);
        }

        h_Ax_vector = new float[dim];
        
        int algcnt = sizeof(algorithm) / sizeof(char*);

        cout << "\nexsum:\n\n";

        for (int algidx = 0; algidx < algcnt; ++algidx)
        {
            cout << algorithm[algidx] << "\n\n";

            for (int run = 0; run < 3; ++run) {
                start = chrono::steady_clock::now();
                spmv_exsum (fpe[algidx], early_exit[algidx], dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector);
                time = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
                cout << fixed << setprecision(10) << (float) time / 1000.0 << '\t'; // ms
            }
            cout << "\n\n";
        }

        cout << "exdot:\n\n";

        for (int algidx = 0; algidx < algcnt; ++algidx)
        {
            cout << algorithm[algidx] << "\n\n";

            for (int run = 0; run < 3; ++run) {
                start = chrono::steady_clock::now();
                spmv_exdot (fpe[algidx], early_exit[algidx], dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector);
                time = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
                cout << fixed << setprecision(10) << (float) time / 1000.0 << '\t'; // ms
            }
            cout << '\n';
        }

        delete[] h_data;
        delete[] h_indices;
        delete[] h_ptr;
        delete[] h_perm;
        delete[] h_nzcnt;
        delete[] h_x_vector;
        delete[] h_Ax_vector;
    }

    return 0;
}