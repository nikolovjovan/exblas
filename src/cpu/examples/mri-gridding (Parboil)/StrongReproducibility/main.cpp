/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define PI 3.14159265

extern void calculate_LUT_seq(float beta, float width, float **LUT, unsigned int *sizeLUT);

extern void calculate_LUT_omp(float beta, float width, float **LUT, unsigned int *sizeLUT);

extern void gridding_seq(unsigned int n, parameters params, ReconstructionSample *sample,
                         float *LUT, unsigned int sizeLUT, cmplx *gridData, float *sampleDensity,
                         const int fpe, const bool early_exit);

extern void gridding_omp(unsigned int n, parameters params, ReconstructionSample *sample,
                         float *LUT, unsigned int sizeLUT, cmplx *gridData, float *sampleDensity,
                         const int fpe, const bool early_exit);

#include <iostream>
#include <iomanip>

using namespace std;

constexpr char* input_files[] = { "small/small.uks" };

constexpr char* algorithm[] = { "accumulator-only", "fpe2", "fpe4", "fpe8ee" };
constexpr int fpe[] = { 0, 2, 4, 8 };
constexpr bool early_exit[] = { false, false, false, true };

/************************************************************
 * This function reads the parameters from the file provided
 * as a comman line argument.
 ************************************************************/
void setParameters(FILE *file, parameters *p)
{
    fscanf(file, "aquisition.numsamples=%d\n", &(p->numSamples));
    fscanf(file, "aquisition.kmax=%f %f %f\n", &(p->kMax[0]), &(p->kMax[1]), &(p->kMax[2]));
    fscanf(file, "aquisition.matrixSize=%d %d %d\n", &(p->aquisitionMatrixSize[0]), &(p->aquisitionMatrixSize[1]),
           &(p->aquisitionMatrixSize[2]));
    fscanf(file, "reconstruction.matrixSize=%d %d %d\n", &(p->reconstructionMatrixSize[0]),
           &(p->reconstructionMatrixSize[1]), &(p->reconstructionMatrixSize[2]));
    fscanf(file, "gridding.matrixSize=%d %d %d\n", &(p->gridSize[0]), &(p->gridSize[1]), &(p->gridSize[2]));
    fscanf(file, "gridding.oversampling=%f\n", &(p->oversample));
    fscanf(file, "kernel.width=%f\n", &(p->kernelWidth));
    fscanf(file, "kernel.useLUT=%d\n", &(p->useLUT));
}

/************************************************************
 * This function reads the sample point data from the kspace
 * and klocation files (and sdc file if provided) into the
 * sample array.
 * Returns the number of samples read successfully.
 ************************************************************/
unsigned int readSampleData(parameters params, FILE *uksdata_f, ReconstructionSample *samples)
{
    unsigned int i;

    for (i = 0; i < params.numSamples; i++) {
        if (feof(uksdata_f)) {
            break;
        }
        fread((void *) &(samples[i]), sizeof(ReconstructionSample), 1, uksdata_f);
    }

    float kScale[3];
    kScale[0] = (float) (params.aquisitionMatrixSize[0]) /
                ((float) (params.reconstructionMatrixSize[0]) * (float) (params.kMax[0]));
    kScale[1] = (float) (params.aquisitionMatrixSize[1]) /
                ((float) (params.reconstructionMatrixSize[1]) * (float) (params.kMax[1]));
    kScale[2] = (float) (params.aquisitionMatrixSize[2]) /
                ((float) (params.reconstructionMatrixSize[2]) * (float) (params.kMax[2]));

    int size_x = params.gridSize[0];
    int size_y = params.gridSize[1];
    int size_z = params.gridSize[2];

    float ax = (kScale[0] * (size_x - 1)) / 2.0;
    float bx = (float) (size_x - 1) / 2.0;

    float ay = (kScale[1] * (size_y - 1)) / 2.0;
    float by = (float) (size_y - 1) / 2.0;

    float az = (kScale[2] * (size_z - 1)) / 2.0;
    float bz = (float) (size_z - 1) / 2.0;

    int n;
    for (n = 0; n < i; n++) {
        samples[n].kX = floor((samples[n].kX * ax) + bx);
        samples[n].kY = floor((samples[n].kY * ay) + by);
        samples[n].kZ = floor((samples[n].kZ * az) + bz);
    }

    return i;
}

void execute(bool parallel, parameters &params, unsigned int number_of_samples, ReconstructionSample *samples, const int fpe, const bool early_exit)
{
    float *LUT;             // use look-up table for faster execution on CPU (intermediate data)
    unsigned int sizeLUT;   // set in the function calculateLUT (intermediate data)

    int gridNumElems = params.gridSize[0] * params.gridSize[1] * params.gridSize[2];

    // Output Data
    cmplx *gridData = (cmplx *) calloc(gridNumElems, sizeof(cmplx));
    float *sampleDensity = (float *) calloc(gridNumElems, sizeof(float));

    if (gridData == nullptr || sampleDensity == nullptr) {
        printf("ERROR: Unable to allocate memory for output data\n");
        exit(1);
    }

    double start, time = 0;

    if (params.useLUT) {
        float beta = PI * sqrt(4 * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - .5) * (params.oversample - .5) - .8);

        start = omp_get_wtime();
        if (parallel) {
            calculate_LUT_omp(beta, params.kernelWidth, &LUT, &sizeLUT);
        } else {
            calculate_LUT_seq(beta, params.kernelWidth, &LUT, &sizeLUT);
        }
        time += omp_get_wtime() - start;
    }

    start = omp_get_wtime();
    if (parallel) {
        gridding_omp(number_of_samples, params, samples, LUT, sizeLUT, gridData, sampleDensity, fpe, early_exit);
    } else {
        gridding_seq(number_of_samples, params, samples, LUT, sizeLUT, gridData, sampleDensity, fpe, early_exit);
    }
    time += omp_get_wtime() - start;

    cout << fixed << setprecision(10) << (float) time * 1000.0 << '\t'; // ms

    if (params.useLUT) {
        free(LUT);
    }
    free(gridData);
    free(sampleDensity);
}

int main(int argc, char *argv[])
{
    parameters params;
    ReconstructionSample *samples;
    unsigned int number_of_samples;

    const int exe_path_len = strrchr(argv[0], '/') - argv[0] + 1;
    char exe_path[256];
    strncpy(exe_path, argv[0], exe_path_len);
    exe_path[exe_path_len] = '\0';

    char uksfile[256];
    char uksdata[256];

    cout << "unit: [ms]\n\n";

    int nfiles = sizeof(input_files) / sizeof(input_files[0]);

    for (int file_idx = 0; file_idx < nfiles; ++file_idx)
    {
        strncpy(uksfile, exe_path, exe_path_len + 1);
        strcat(uksfile, "data/");
        strcat(uksfile, input_files[file_idx]);

        cout << input_files[file_idx] << "\n";

        {
            FILE *uksfile_f = nullptr;
            FILE *uksdata_f = nullptr;

            strcpy(uksdata, uksfile);
            strcat(uksdata, ".data");

            uksfile_f = fopen(uksfile, "r");
            if (uksfile_f == nullptr) {
                printf("ERROR: Could not open %s\n", uksfile);
                exit(1);
            }

            setParameters(uksfile_f, &params);

            samples = (ReconstructionSample *) malloc(params.numSamples * sizeof(ReconstructionSample)); // Input Data

            if (samples == nullptr) {
                printf("ERROR: Unable to allocate memory for input data\n");
                exit(1);
            }

            uksdata_f = fopen(uksdata, "rb");

            if (uksdata_f == nullptr) {
                printf("ERROR: Could not open data file\n");
                exit(1);
            }

            number_of_samples = readSampleData(params, uksdata_f, samples);
            fclose(uksdata_f);
        }

        int algcnt = sizeof(algorithm) / sizeof(char*);

        for (int algidx = 0; algidx < algcnt; ++algidx)
        {
            cout << algorithm[algidx] << "\n\n";

            cout << "\nseq\t";
            for (int run = 0; run < 3; ++run) execute (false, params, number_of_samples, samples, fpe[algidx], early_exit[algidx]);
            cout << '\n';
            
            for (int thread_count = 1; thread_count <= 128; thread_count <<= 1)
            {
                omp_set_dynamic(0);                 // Explicitly disable dynamic teams
                omp_set_num_threads(thread_count);  // Use  thread_count threads for all consecutive parallel regions

                #pragma omp parallel
                #pragma omp single
                {
                    cout << omp_get_num_threads() << '\t';
                }

                for (int run = 0; run < 3; ++run) execute (true, params, number_of_samples, samples, fpe[algidx], early_exit[algidx]);
                cout << '\n';
            }

            cout << '\n';
        }
            
        free(samples);
    }

    return 0;
}