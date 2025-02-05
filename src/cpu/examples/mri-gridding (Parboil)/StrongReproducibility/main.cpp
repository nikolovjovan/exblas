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

#define CALLOC_AND_CHECK(name, type, size, errCode) \
    type *name = (type *) calloc(size, sizeof(type)); \
    if (name == NULL) { \
        printf("ERROR: Unable to allocate memory for " #name ".\n"); \
        exit(errCode); \
    }

extern void calculate_LUT_seq(float beta, float width, float **LUT, unsigned int *sizeLUT);

extern void calculate_LUT_omp(float beta, float width, float **LUT, unsigned int *sizeLUT);

extern void gridding_seq(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT,
                         unsigned int sizeLUT, cmplx *gridData, float *sampleDensity,
                         bool reproducible, double *time,
                         const int fpe = 0, const bool early_exit = false);

extern void gridding_omp(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT,
                         unsigned int sizeLUT, cmplx *gridData, float *sampleDensity,
                         bool reproducible, double *time,
                         const int fpe = 0, const bool early_exit = false);

extern void gridding_omp_mem(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT,
                             unsigned int sizeLUT, cmplx *gridData, float *sampleDensity, double *time);

extern void gridding_omp_locks(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT,
                               unsigned int sizeLUT, cmplx *gridData, float *sampleDensity, double *time);

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

    printf("  Number of samples = %d\n", p->numSamples);
    printf("  Grid Size = %dx%dx%d\n", p->gridSize[0], p->gridSize[1], p->gridSize[2]);
    printf("  Input Matrix Size = %dx%dx%d\n", p->aquisitionMatrixSize[0], p->aquisitionMatrixSize[1],
           p->aquisitionMatrixSize[2]);
    printf("  Recon Matrix Size = %dx%dx%d\n", p->reconstructionMatrixSize[0], p->reconstructionMatrixSize[1],
           p->reconstructionMatrixSize[2]);
    printf("  Kernel Width = %f\n", p->kernelWidth);
    printf("  KMax = %.2f %.2f %.2f\n", p->kMax[0], p->kMax[1], p->kMax[2]);
    printf("  Oversampling = %f\n", p->oversample);
    printf("  GPU Binsize = %d\n", p->binsize);
    printf("  Use LUT = %s\n", (p->useLUT) ? "Yes" : "No");
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

const float ACCURACY = 0.1;

/**
 * "Nastelovana" funkcija za poredjenje float vrednosti.
 * Naime, posto u racunanju vrednosti jednog elementa nizova gridData i sampleDensity postoji nekoliko hiljada operacija, preciznost operacija drasticno opada.
 * Za jednu operaciju sabiranja float garantuje preciznost od 6-7 znacajnih cifara. Kada se to ponovi nekoliko hiljada puta koliko je ovde i slucaj
 * (vidi se u nizu sampleDensity), preciznost pada na 1-2 znacajne cifre.
 * Ovaj test brojeve deli sa 10 dok ne postanu manji od 10 pa proverava da li je razlika manja od 0.1 sto efektivno znaci da proverava na 2 znacajne cifre.
 * Kada se proba sa 3 znacajne cifre ne prolazi iako brojevi "lice". Upravo to je rezultat inherentne greske kod floating point operacija.
 */
int compare_significant_digits(float a, float b)
{
    while (fabs(a) > 10.f && fabs(b) > 10.f) {
        a /= 10;
        b /= 10;
    }
    if (a - b > ACCURACY) return 1;
    if (a - b < -ACCURACY) return -1;
    return 0;
}

int compare(float a, float b)
{
    if (a > b) return 1;
    if (a < b) return -1;
    return 0;
}

int diff(int sizeLUT, float *LUT, int sizeLUT_omp, float *LUT_omp, int gridNumElems, cmplx *gridData, float *sampleDensity, cmplx *gridData_omp, float *sampleDensity_omp)
{
    int i;

    // Compare LUT
    if (sizeLUT != sizeLUT_omp) {
        printf("LUT sizes mismatch!\n");
        return 1;
    }
    for (i = 0; i < sizeLUT; ++i) {
        if (fabs(LUT[i] - LUT_omp[i]) > ACCURACY) {
            printf("LUT[%d] mismatch: %f <-> %f\n", i, LUT[i], LUT_omp[i]);
            return 2;
        }
    }

    // Compare gridData and sampleDensity
    for (i = 0; i < gridNumElems; ++i) {
        if (compare(gridData[i].real, gridData_omp[i].real) || compare(gridData[i].imag, gridData_omp[i].imag)) {
            printf("gridData[%d] mismatch: %f, %f <-> %f, %f\n", i, gridData[i].real, gridData[i].imag, gridData_omp[i].real, gridData_omp[i].imag);
            return 3;
        }
        if (fabs(sampleDensity[i] - sampleDensity_omp[i]) > ACCURACY) {
            printf("sampleDensity[%d] mismatch: %f <-> %f\n", i, sampleDensity[i], sampleDensity_omp[i]);
            return 4;
        }
    }

    // Results same
    return 0;
}

int main(int argc, char *argv[])
{
    char uksfile[256];
    char uksdata[256];
    parameters params;
    params.binsize = 128;

    FILE *uksfile_f = NULL;
    FILE *uksdata_f = NULL;

    int fpe = 0;
    bool early_exit = false;

    if (argc < 2) {
        printf("ERROR: Not enough arguments!\n");
        exit(1);
    }

    strcpy(uksfile, argv[1]);
    strcpy(uksdata, argv[1]);
    strcat(uksdata, ".data");

    uksfile_f = fopen(uksfile, "r");
    if (uksfile_f == NULL) {
        printf("ERROR: Could not open %s\n", uksfile);
        exit(2);
    }

    printf("Reading parameters\n");

    if (argc > 2) {
        params.binsize = atoi(argv[2]);
    }

    if (argc > 3) {
        fpe = atoi(argv[3]);
    }

    if (argc > 4) {
        early_exit = (bool) atoi(argv[4]);
    }

    setParameters(uksfile_f, &params);

    printf("  Number of allocated threads = %d\n", omp_get_max_threads());

    ReconstructionSample *samples =
        (ReconstructionSample *) malloc(params.numSamples * sizeof(ReconstructionSample)); // Input Data

    if (samples == NULL) {
        printf("ERROR: Unable to allocate memory for input data\n");
        exit(1);
    }

    float *LUT_seq, *LUT_omp;              // use look-up table for faster execution on CPU (intermediate data)
    unsigned int sizeLUT_seq, sizeLUT_omp; // set in the function calculateLUT (intermediate data)

    int gridNumElems = params.gridSize[0] * params.gridSize[1] * params.gridSize[2];

    // Output Data
    CALLOC_AND_CHECK (/* name */ gridData_seq,                  /* type */ cmplx,   /* size */ gridNumElems,    /* errCode */ 1);
    CALLOC_AND_CHECK (/* name */ gridData_omp_mem,              /* type */ cmplx,   /* size */ gridNumElems,    /* errCode */ 2);
    CALLOC_AND_CHECK (/* name */ gridData_omp_locks,            /* type */ cmplx,   /* size */ gridNumElems,    /* errCode */ 3);
    CALLOC_AND_CHECK (/* name */ gridData_omp_2step,            /* type */ cmplx,   /* size */ gridNumElems,    /* errCode */ 4);
    CALLOC_AND_CHECK (/* name */ gridData_seq_rep,              /* type */ cmplx,   /* size */ gridNumElems,    /* errCode */ 5);
    CALLOC_AND_CHECK (/* name */ gridData_omp_2step_rep,        /* type */ cmplx,   /* size */ gridNumElems,    /* errCode */ 6);
    CALLOC_AND_CHECK (/* name */ sampleDensity_seq,             /* type */ float,   /* size */ gridNumElems,    /* errCode */ 7);
    CALLOC_AND_CHECK (/* name */ sampleDensity_omp_mem,         /* type */ float,   /* size */ gridNumElems,    /* errCode */ 8);
    CALLOC_AND_CHECK (/* name */ sampleDensity_omp_locks,       /* type */ float,   /* size */ gridNumElems,    /* errCode */ 9);
    CALLOC_AND_CHECK (/* name */ sampleDensity_omp_2step,       /* type */ float,   /* size */ gridNumElems,    /* errCode */ 10);
    CALLOC_AND_CHECK (/* name */ sampleDensity_seq_rep,         /* type */ float,   /* size */ gridNumElems,    /* errCode */ 11);
    CALLOC_AND_CHECK (/* name */ sampleDensity_omp_2step_rep,   /* type */ float,   /* size */ gridNumElems,    /* errCode */ 12);

    uksdata_f = fopen(uksdata, "rb");

    if (uksdata_f == NULL) {
        printf("ERROR: Could not open data file\n");
        exit(1);
    }

    printf("Reading input data from files\n");

    unsigned int n = readSampleData(params, uksdata_f, samples);
    fclose(uksdata_f);

    double tstart, time_loop_seq, time_loop_omp_mem, time_loop_omp_locks, time_loop_omp_2step, time_loop_seq_rep, time_loop_omp_2step_rep;
    double time_seq = 0, time_omp_mem = 0, time_omp_locks = 0, time_omp_2step = 0, time_seq_rep = 0, time_omp_2step_rep = 0;

    if (params.useLUT) {
        printf("Generating Look-Up Table\n");
        float beta = PI * sqrt(4 * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - .5) * (params.oversample - .5) - .8);

        tstart = omp_get_wtime();
        calculate_LUT_seq(beta, params.kernelWidth, &LUT_seq, &sizeLUT_seq);
        time_seq += omp_get_wtime() - tstart;

        tstart = omp_get_wtime();
        calculate_LUT_omp(beta, params.kernelWidth, &LUT_omp, &sizeLUT_omp);
        time_omp_mem += omp_get_wtime() - tstart;
    }

    printf("\nSequential implementation LUT generation execution time: %.3f\n", time_seq);
    printf("Parallel implementation LUT generation execution time: %.3f\n", time_omp_mem);
    printf("LUT generation speedup: %.3f\n\n", time_seq / time_omp_mem);

    /* add LUT calculation time to reproducible time, no need to calculate LUT twice */
    time_omp_locks += time_omp_mem;
    time_omp_2step += time_omp_mem;
    time_seq_rep += time_seq;
    time_omp_2step_rep += time_omp_mem;

    tstart = omp_get_wtime();
    gridding_seq(n, params, samples, LUT_seq, sizeLUT_seq, gridData_seq, sampleDensity_seq, false, &time_loop_seq);
    time_seq += omp_get_wtime() - tstart;

    tstart = omp_get_wtime();
    gridding_omp_mem(n, params, samples, LUT_seq, sizeLUT_seq, gridData_omp_mem, sampleDensity_omp_mem, &time_loop_omp_mem);
    time_omp_mem += omp_get_wtime() - tstart;

    tstart = omp_get_wtime();
    gridding_omp_locks(n, params, samples, LUT_seq, sizeLUT_seq, gridData_omp_locks, sampleDensity_omp_locks, &time_loop_omp_locks);
    time_omp_locks += omp_get_wtime() - tstart;

    tstart = omp_get_wtime();
    gridding_omp(n, params, samples, LUT_seq, sizeLUT_seq, gridData_omp_2step, sampleDensity_omp_2step, /* reproducible */ false, &time_loop_omp_2step, fpe, early_exit);
    time_omp_2step += omp_get_wtime() - tstart;

    tstart = omp_get_wtime();
    gridding_seq(n, params, samples, LUT_seq, sizeLUT_seq, gridData_seq_rep, sampleDensity_seq_rep, true, &time_loop_seq_rep);
    time_seq_rep += omp_get_wtime() - tstart;

    tstart = omp_get_wtime();
    gridding_omp(n, params, samples, LUT_omp, sizeLUT_omp, gridData_omp_2step_rep, sampleDensity_omp_2step_rep, /* reproducible */ true, &time_loop_omp_2step_rep, fpe, early_exit);
    time_omp_2step_rep += omp_get_wtime() - tstart;

    printf("\n");

    if (diff(sizeLUT_seq, LUT_seq, sizeLUT_seq, LUT_seq, gridNumElems,
        gridData_seq, sampleDensity_seq, gridData_seq_rep, sampleDensity_seq_rep)) {
        printf("Non-reproducible and reproducible sequential results do not match!\n");
    }

    if (diff(sizeLUT_omp, LUT_omp, sizeLUT_omp, LUT_omp, gridNumElems,
        gridData_omp_2step, sampleDensity_omp_2step, gridData_omp_2step_rep, sampleDensity_omp_2step_rep)) {
        printf("Non-reproducible and reproducible parallel results do not match!\n");
    }

    printf("\nNon-reproducible sequential and parallel results ");
    if (diff(sizeLUT_seq, LUT_seq, sizeLUT_omp, LUT_omp, gridNumElems,
        gridData_seq, sampleDensity_seq, gridData_omp_2step, sampleDensity_omp_2step)) {
        printf("do not ");
    }
    printf("match!\n");

    printf("Reproducible sequential and parallel results ");
    if (diff(sizeLUT_seq, LUT_seq, sizeLUT_omp, LUT_omp, gridNumElems,
        gridData_seq_rep, sampleDensity_seq_rep, gridData_omp_2step_rep, sampleDensity_omp_2step_rep)) {
        printf("do not ");
    }
    printf("match!\n");

    printf("\nSequential implementation loop execution time: %.3f\n", time_loop_seq);
    printf("Parallel implementation loop execution time (mem): %.3f\n", time_loop_omp_mem);
    printf("Loop speedup: %.3f\n", time_loop_seq / time_loop_omp_mem);
    printf("Parallel implementation loop execution time (locks): %.3f\n", time_loop_omp_locks);
    printf("Loop speedup: %.3f\n", time_loop_seq / time_loop_omp_locks);
    printf("Parallel implementation loop execution time (2step): %.3f\n", time_loop_omp_2step);
    printf("Loop speedup: %.3f\n", time_loop_seq / time_loop_omp_2step);

    printf("\nReproducible sequential implementation loop execution time: %.3f\n", time_loop_seq_rep);
    printf("Reproducible parallel implementation loop execution time: %.3f\n", time_loop_omp_2step_rep);
    printf("Loop speedup (reproducible): %.3f\n", time_loop_seq_rep / time_loop_omp_2step_rep);

    printf("\nSequential implementation execution time: %.3f\n", time_seq);
    printf("Parallel implementation execution time (mem): %.3f\n", time_omp_mem);
    printf("Speedup: %.3f\n", time_seq / time_omp_mem);
    printf("Parallel implementation execution time (locks): %.3f\n", time_omp_locks);
    printf("Speedup: %.3f\n", time_seq / time_omp_locks);
    printf("Parallel implementation execution time (2step): %.3f\n", time_omp_2step);
    printf("Speedup: %.3f\n", time_seq / time_omp_2step);

    printf("\nReproducible sequential implementation execution time: %.3f\n", time_seq_rep);
    printf("Reproducible parallel implementation execution time: %.3f\n", time_omp_2step_rep);
    printf("Speedup (reproducible): %.3f\n", time_seq_rep / time_omp_2step_rep);

    printf("\nTime sequential reproducible / non-reproducible: %.3f\n", time_seq_rep / time_seq);
    printf("Time parallel reproducible / non-reproducible: %.3f\n", time_omp_2step_rep / time_omp_2step);

    if (params.useLUT) {
        free(LUT_seq);
        free(LUT_omp);
    }

    free(samples);
    free(gridData_seq);
    free(gridData_omp_mem);
    free(gridData_omp_locks);
    free(gridData_omp_2step);
    free(gridData_seq_rep);
    free(gridData_omp_2step_rep);
    free(sampleDensity_seq);
    free(sampleDensity_omp_mem);
    free(sampleDensity_omp_locks);
    free(sampleDensity_omp_2step);
    free(sampleDensity_seq_rep);
    free(sampleDensity_omp_2step_rep);

    return 0;
}