/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"

#include "blas1.hpp"
#include "common.hpp"

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);

int find_nearest_point(float *pt,                  /* [nfeatures] */
                       int nfeatures, float **pts, /* [npts][nfeatures] */
                       int npts)
{
    int index, i;
    float min_dist = FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i = 0; i < npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures); /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return (index);
}

/* multi-dimensional spatial Euclid distance square */
__inline float euclid_dist_2(float *pt1, float *pt2, int numdims)
{
    int i;
    float ans = 0.0;

    for (i = 0; i < numdims; i++)
        ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);

    return (ans);
}

double **float_feature_to_double(float **feature, /* in: [npoints][nfeatures] */
                                 int nfeatures, int npoints)
{    
    double **dbl_feature;     /* [npoints][nfeatures] */

    /* allocate space for double feature vector */
    dbl_feature = (double **) malloc(npoints * sizeof(double *));
    dbl_feature[0] = (double *) malloc(npoints * nfeatures * sizeof(double));

    /* copy data */
    for (int i = 1; i < npoints; ++i) {
        dbl_feature[i] = dbl_feature[i - 1] + nfeatures;
        for (int j = 0; j < nfeatures; ++j) {
            dbl_feature[i][j] = feature[i][j];
        }
    }

    return dbl_feature;
}

void free_feature(double **dbl_feature)
{
    free(dbl_feature[0]);
    free(dbl_feature);
}

float **kmeans_clustering_seq(bool reproducible,
                              float **feature, /* in: [npoints][nfeatures] */
                              int nfeatures, int npoints, int nclusters, float threshold,
                              int *membership, /* out: [npoints] */
                              const int fpe, const bool early_exit)
{
    int i, j, n = 0, index, loop = 0;
    int *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float **new_centers; /* [nclusters][nfeatures] */
    float **clusters;    /* out: [nclusters][nfeatures] */
    float delta;
    double *dbl_new_features;   // array of feature f of points nearest to current cluster
                                // used to sum all features of points belonging to a cluster at once

    /* allocate space for returning variable clusters[] */
    clusters = (float **) malloc(nclusters * sizeof(float *));
    clusters[0] = (float *) malloc(nclusters * nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        // n = (int) rand() % npoints;
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int *) calloc(nclusters, sizeof(int));

    new_centers = (float **) malloc(nclusters * sizeof(float *));
    new_centers[0] = (float *) calloc(nclusters * nfeatures, sizeof(float));
    for (i = 1; i < nclusters; i++)
        new_centers[i] = new_centers[i - 1] + nfeatures;

    if (reproducible) {
        /* allocate space for temporary feature array */
        dbl_new_features = (double *) malloc(npoints * sizeof(double));
    }

    do {
        delta = 0.0;

        for (i = 0; i < npoints; i++) {
            /* find the index of nearest cluster centers */
            index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
            /* if membership changes, increase delta by 1 */
            if (membership[i] != index)
                delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;
            new_centers_len[index]++;

            if (!reproducible) {
                /* update new cluster centers : sum of objects located within */
                for (j = 0; j < nfeatures; j++)
                    new_centers[index][j] += feature[i][j];
            }
        }

        if (reproducible) {
            for (index = 0; index < nclusters; index++) {
                /* update new cluster centers : sum of objects located within */
                for (j = 0; j < nfeatures; j++) {
                    n = 0; // number of features in current dbl_new_features array (goes to new_centers_len[index])
                    for (i = 0; i < npoints; i++) {
                        if (membership[i] == index) { // point i belongs to cluster index
                            dbl_new_features[n++] = feature[i][j];
                        }
                    }
                    new_centers[index][j] = static_cast<float> (exsum(
                        /* Ng */            new_centers_len[index],
                        /* ag */            dbl_new_features,
                        /* inca */          1,
                        /* offset */        0,
                        /* fpe */           fpe,
                        /* early_exit */    early_exit,
                        /* parallel */      false));
                }
            }
        }

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            for (j = 0; j < nfeatures; j++) {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0; /* set back to 0 */
            }
            new_centers_len[i] = 0; /* set back to 0 */
        }
    } while (delta > threshold);

    if (reproducible) {
        free(dbl_new_features);
    }

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}

float **kmeans_clustering_omp(bool reproducible,
                              float **feature, /* in: [npoints][nfeatures] */
                              int nfeatures, int npoints, int nclusters, float threshold,
                              int *membership, /* out: [npoints] */
                              const int fpe, const bool early_exit)
{
    int i, j, k, l, n = 0, index, loop = 0;
    int *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float **new_centers; /* [nclusters][nfeatures] */
    float **clusters;    /* out: [nclusters][nfeatures] */
    float delta;

    int nthreads = omp_get_max_threads();
    int **partial_new_centers_len;
    float ***partial_new_centers;
    double **dbl_new_features;  // array of feature f of points nearest to current cluster
                                        // used to sum all features of points belonging to a cluster at once

    /* allocate space for returning variable clusters[] */
    clusters = (float **) malloc(nclusters * sizeof(float *));
    clusters[0] = (float *) malloc(nclusters * nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        // n = (int) rand() % npoints;
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int *) calloc(nclusters, sizeof(int));

    new_centers = (float **) malloc(nclusters * sizeof(float *));
    new_centers[0] = (float *) calloc(nclusters * nfeatures, sizeof(float));
    for (i = 1; i < nclusters; i++)
        new_centers[i] = new_centers[i - 1] + nfeatures;

    partial_new_centers_len = (int **) malloc(nthreads * sizeof(int *));
    partial_new_centers_len[0] = (int *) calloc(nthreads * nclusters, sizeof(int));
    for (i = 1; i < nthreads; ++i)
        partial_new_centers_len[i] = partial_new_centers_len[i - 1] + nclusters;

    partial_new_centers = (float ***) malloc(nthreads * sizeof(float **));
    partial_new_centers[0] = (float **) malloc(nthreads * nclusters * sizeof(float *));
    for (i = 1; i < nthreads; ++i)
        partial_new_centers[i] = partial_new_centers[i - 1] + nclusters;

    for (i = 0; i < nthreads; ++i)
        for (j = 0; j < nclusters; ++j)
            partial_new_centers[i][j] = (float *) calloc(nfeatures, sizeof(float));

    if (reproducible) {
        /* allocate space for temporary feature array */
        dbl_new_features = (double **) malloc(nthreads * sizeof(double *));
        dbl_new_features[0] = (double *) malloc(nthreads * npoints * sizeof(double));
        for (i = 1; i < nthreads; ++i)
            dbl_new_features[i] = dbl_new_features[i - 1] + npoints;
    }

    do {
        delta = 0.0;

#pragma omp parallel \
        private(i, j, k, l, n, index) \
        firstprivate(npoints, nclusters, nfeatures, fpe, early_exit) \ 
        shared(feature, clusters, membership, partial_new_centers, partial_new_centers_len, dbl_new_features)
{
        int tid = omp_get_thread_num();

#pragma omp for \
        schedule(static) \
        reduction(+:delta)

        for (i = 0; i < npoints; i++) {
            /* find the index of nearest cluster centers */
            index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
            /* if membership changes, increase delta by 1 */
            if (membership[i] != index)
                delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;
            partial_new_centers_len[tid][index]++;

            /* update new cluster centers : sum of objects located within */
            if (!reproducible) {
                for (j = 0; j < nfeatures; j++)
                    partial_new_centers[tid][index][j] += feature[i][j];
            }
        }

        if (reproducible) {

#pragma omp for \
        schedule(static)

            /* reduce partial_new_centers_len */
            for (index = 0; index < nclusters; index++) {
                for (i = 0; i < nthreads; i++) {
                    new_centers_len[index] += partial_new_centers_len[i][index];
                    partial_new_centers_len[i][index] = 0;
                }
            }

            /* improve paralelization below since thread number > nclusters in the provided example */
            n = nclusters * nfeatures;

#pragma omp for \
        schedule(static)

            for (k = 0; k < n; k++) {
                index = k / nfeatures;
                j = k % nfeatures;
                l = 0; // number of points/features in current dbl_new_features array
                for (i = 0; i < npoints && l < new_centers_len[index]; i++) {
                    if (membership[i] == index) {
                        dbl_new_features[tid][l++] = feature[i][j];
                    }
                }
                new_centers[index][j] = static_cast<float> (exsum(
                    /* Ng */            new_centers_len[index],
                    /* ag */            dbl_new_features[tid],
                    /* inca */          1,
                    /* offset */        0,
                    /* fpe */           fpe,
                    /* early_exit */    early_exit,
                    /* parallel */      false));
            }
        }
} /* end of #pragma omp parallel */

        if (!reproducible) {
            /* let the main thread perform the array reduction */
            for (index = 0; index < nclusters; index++) {
                for (i = 0; i < nthreads; i++) {
                    new_centers_len[index] += partial_new_centers_len[i][index];
                    partial_new_centers_len[i][index] = 0.0;
                    for (j = 0; j < nfeatures; j++) {
                        new_centers[index][j] += partial_new_centers[i][index][j];
                        partial_new_centers[i][index][j] = 0.0;
                    }
                }
            }
        }

        /* replace old cluster centers with new_centers */
        for (index = 0; index < nclusters; index++) {
            for (j = 0; j < nfeatures; j++) {
                if (new_centers_len[index] > 0)
                    clusters[index][j] = new_centers[index][j] / new_centers_len[index];
                new_centers[index][j] = 0.0; /* set back to 0 */
            }
            new_centers_len[index] = 0; /* set back to 0 */
        }
    } while (delta > threshold && loop++ < 500);

    if (reproducible) {
        free(dbl_new_features[0]);
        free(dbl_new_features);
    }

    for (i = 0; i < nthreads; ++i)
        for (index = 0; index < nclusters; ++index)
            free(partial_new_centers[i][index]);
    free(partial_new_centers[0]);
    free(partial_new_centers);

    free(partial_new_centers_len[0]);
    free(partial_new_centers_len);

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}