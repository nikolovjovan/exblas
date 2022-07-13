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
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee					**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation crietria; instead		**/
/**					calculate root mean squared error.					**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include "kmeans.h"

#include "blas1.hpp"
#include "common.hpp"

#define RANDOM_MAX 2147483647

extern double wtime(void);

/*----< kmeans_clustering() >---------------------------------------------*/
float **kmeans_clustering(bool reproducible, /* use reproducible implementation */
						  float **feature, /* in: [npoints][nfeatures] */
						  int nfeatures,
						  int npoints,
						  int nclusters,
						  float threshold,
						  int *membership, /* out: [npoints] */
						  const int fpe, const bool early_exit)
{
	int i, j, index, n = 0; /* counters */
	int loop = 0, temp;

	int *new_membership;  /* [npoints] new membership of each point */
	int *new_centers_len; /* [nclusters]: no. of points in each cluster */
	float delta;		  /* if the point moved */
	float **clusters;	  /* out: [nclusters][nfeatures] */
	float **new_centers;  /* [nclusters][nfeatures] */
    double *dbl_new_features;   // array of feature f of points nearest to current cluster
                                // used to sum all features of points belonging to a cluster at once

	int *initial; /* used to hold the index of points not yet selected
					 prevents the "birthday problem" of dual selection (?)
					 considered holding initial cluster indices, but changed due to
					 possible, though unlikely, infinite loops */
	int initial_points;
	int c = 0;

	/* nclusters should never be > npoints
	   that would guarantee a cluster without points */
	if (nclusters > npoints)
		nclusters = npoints;

	/* allocate space for and initialize returning variable clusters[] */
	clusters = (float **)malloc(nclusters * sizeof(float *));
	clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
	for (i = 1; i < nclusters; i++)
		clusters[i] = clusters[i - 1] + nfeatures;

	/* initialize the random clusters */
	initial = (int *)malloc(npoints * sizeof(int));
	for (i = 0; i < npoints; i++)
	{
		initial[i] = i;
	}

	initial_points = npoints;

	/* randomly pick cluster centers */
	for (i = 0; i < nclusters && initial_points >= 0; i++)
	{
		// n = (int)rand() % initial_points;

		for (j = 0; j < nfeatures; j++)
			clusters[i][j] = feature[initial[n]][j]; // remapped

		/* swap the selected index to the end (not really necessary,
		   could just move the end up) */
		temp = initial[n];
		initial[n] = initial[initial_points - 1];
		initial[initial_points - 1] = temp;
		initial_points--;
		n++;
	}

	/* initialize the membership to -1 for all */
	for (i = 0; i < npoints; i++)
		membership[i] = -1;

	/* allocate space for and initialize new_centers_len and new_centers */
	new_centers_len = (int *)calloc(nclusters, sizeof(int));

	new_centers = (float **)malloc(nclusters * sizeof(float *));
	new_centers[0] = (float *)calloc(nclusters * nfeatures, sizeof(float));
	for (i = 1; i < nclusters; i++)
		new_centers[i] = new_centers[i - 1] + nfeatures;

    if (reproducible) {
        /* allocate space for temporary feature array */
        dbl_new_features = (double *) malloc(npoints * sizeof(double));
    }

	/* iterate until convergence */
	do
	{
		delta = 0.0;

		new_membership = calculate_membership(nfeatures,	/* number of attributes for each point */
											  npoints,		/* number of data points */
											  nclusters,	/* number of clusters */
											  clusters);	/* out: [nclusters][nfeatures] */

		for (i = 0; i < npoints; i++)
		{
			int cluster_id = new_membership[i];
			new_centers_len[cluster_id]++;
		
			if (new_membership[i] != membership[i])
			{
				delta++;
				membership[i] = new_membership[i];
			}

			if (!reproducible)
			{
				for (j = 0; j < nfeatures; j++)
				{
					new_centers[cluster_id][j] += feature[i][j];
				}
			}
		}

		if (reproducible)
		{
			for (index = 0; index < nclusters; index++)
			{
				/* update new cluster centers : sum of objects located within */
				for (j = 0; j < nfeatures; j++)
				{
					n = 0; // number of features in current dbl_new_features array (goes to new_centers_len[index])
					for (i = 0; i < npoints; i++)
					{
						if (membership[i] == index)
						{ // point i belongs to cluster index
							dbl_new_features[n++] = feature[i][j];
						}
					}
					// printf("exsum run\n");
					new_centers[index][j] = static_cast<float>(exsum(
						/* Ng */ new_centers_len[index],
						/* ag */ dbl_new_features,
						/* inca */ 1,
						/* offset */ 0,
						/* fpe */ fpe,
						/* early_exit */ early_exit,
						/* parallel */ false));
				}
			}
		}

		/* replace old cluster centers with new_centers */
		/* CPU side of reduction */
		for (i = 0; i < nclusters; i++)
		{
			for (j = 0; j < nfeatures; j++)
			{
				if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers[i][j] / new_centers_len[i]; /* take average i.e. sum/n */
				new_centers[i][j] = 0.0;									 /* set back to 0 */
			}
			new_centers_len[i] = 0; /* set back to 0 */
		}

		c++;
	} while ((delta > threshold) && (loop++ < 500)); /* makes sure loop terminates */

	printf("iterated %d times\n", c);

    if (reproducible) {
        free(dbl_new_features);
    }

	free(new_centers[0]);
	free(new_centers);
	free(new_centers_len);

	return clusters;
}