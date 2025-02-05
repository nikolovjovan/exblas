/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <cstdio>

#include "UDTypes.h"

void calculateLUT(float beta, float width, float **LUT, unsigned int *sizeLUT);

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample *sample, float *LUT, unsigned int sizeLUT, cmplx *gridData, float *sampleDensity);

int gridding_CPU(unsigned int n, parameters params, ReconstructionSample *sample, int *CPUbin, int CPUbin_size,
                 float *LUT, int sizeLUT, cmplx *gridData[], float *sampleDensity[], int *indeces[]);