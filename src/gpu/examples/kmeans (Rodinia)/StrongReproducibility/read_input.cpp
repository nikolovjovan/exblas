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
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
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
#define _CRT_SECURE_NO_DEPRECATE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "kmeans.h"
#include <unistd.h>

#include "blas1.hpp"
#include "common.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std;

constexpr char* input_files[] = { "100" }; //, "204800.txt", "819200.txt", "kdd_cup" }; - too slow

constexpr char* algorithm[] = { "accumulator-only", "fpe2", "fpe4", "fpe8ee" };
constexpr int fpe[] = { 0, 2, 4, 8 };
constexpr bool early_exit[] = { false, false, false, true };

/*---< main() >-------------------------------------------------------------*/
int setup(int argc, char **argv)
{
	float *buf;
	char line[1024];

	float threshold = 0.001; /* default value */
	int max_nclusters = 5;	 /* default value */
	int min_nclusters = 5;	 /* default value */
	int best_nclusters = 0;
	int nfeatures = 0;
	int npoints = 0;

	float **features;
	float **cluster_centres = NULL;
	int nloops = 1; /* default value */

	uint64_t time_setup[3];
	uint64_t time_run[3];

	int isRMSE = 0;
	float rmse;

    const int exe_path_len = strrchr(argv[0], '/') - argv[0] + 1;
    char exe_path[256];
    strncpy(exe_path, argv[0], exe_path_len);
    exe_path[exe_path_len] = '\0';

    char input_file_path[256];

    chrono::steady_clock::time_point start;
    uint64_t time_first_setup = 0, time;

    double dummy_array[2] = { 1.0, 2.0 };

    start = chrono::steady_clock::now();
    exsum(2, dummy_array, 1, 0, 0, false, false);
    time_first_setup =
        chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();

    cout << "OCL first (dummy) setup time : " << fixed << setprecision(10) << (float) time_first_setup / 1000.0 << endl << endl; // ms

    cout << "unit: [ms]\n\n";

    int nfiles = sizeof(input_files) / sizeof(input_files[0]);

    for (int file_idx = 0; file_idx < nfiles; ++file_idx)
    {
        strncpy(input_file_path, exe_path, exe_path_len + 1);
        strcat(input_file_path, "data/");
        strcat(input_file_path, input_files[file_idx]);

        cout << input_files[file_idx] << "\n\n";

		/* ============== I/O begin ==============*/
		/* get nfeatures and npoints */
		{
			npoints = 0;
			nfeatures = 0;

			FILE *infile;
			if ((infile = fopen(input_file_path, "r")) == NULL)
			{
				fprintf(stderr, "Error: no such file (%s)\n", input_file_path);
				exit(1);
			}

			while (fgets(line, 1024, infile) != NULL)
				if (strtok(line, " \t\n") != 0)
					npoints++;

			rewind(infile);
			while (fgets(line, 1024, infile) != NULL)
			{
				if (strtok(line, " \t\n") != 0)
				{
					/* ignore the id (first attribute): nfeatures = 1; */
					while (strtok(NULL, " ,\t\n") != NULL)
						nfeatures++;
					break;
				}
			}

			/* allocate space for features[] and read attributes of all objects */
			buf = (float *)malloc(npoints * nfeatures * sizeof(float));
			features = (float **)malloc(npoints * sizeof(float *));
			features[0] = (float *)malloc(npoints * nfeatures * sizeof(float));

			int i;
			for (i = 1; i < npoints; i++)
				features[i] = features[i - 1] + nfeatures;

			rewind(infile);
			i = 0;

			while (fgets(line, 1024, infile) != NULL)
			{
				if (strtok(line, " \t\n") == NULL)
					continue;

				for (int j = 0; j < nfeatures; j++)
				{
					buf[i] = atof(strtok(NULL, " ,\t\n"));
					i++;
				}
			}
			fclose(infile);
		}

		srand(7);													   /* seed for future random number generator */
		memcpy(features[0], buf, npoints * nfeatures * sizeof(float)); /* now features holds 2-dimensional array of features */
		free(buf);

        int algcnt = sizeof(algorithm) / sizeof(char*);

        for (int algidx = 0; algidx < algcnt; ++algidx)
        {
            cout << algorithm[algidx] << "\n\n";

            for (int run = 0; run < 3; ++run)
			{
				cluster_centres = NULL;
				cluster(npoints,	   /* number of data points */
						nfeatures,	   /* number of features for each point */
						features,	   /* array: [npoints][nfeatures] */
						min_nclusters, /* range of min to max number of clusters */
						max_nclusters,
						threshold,		  /* loop termination factor */
						&best_nclusters,  /* return: number between min and max */
						&cluster_centres, /* return: [best_nclusters][nfeatures] */
						&rmse,			  /* Root Mean Squared Error */
						isRMSE,			  /* calculate RMSE */
						nloops,			  /* number of iteration for each number of clusters */
						time_setup[run],
						time_run[run],
						fpe[algidx],
						early_exit[algidx]);
				free(cluster_centres);
			}

			for (int run = 0; run < 3; ++run) {
				cout << fixed << setprecision(10) << (float) time_setup[run] / 1000.0 << '\t'; // ms
			}
			cout << '\n';

			for (int run = 0; run < 3; ++run) {
				cout << fixed << setprecision(10) << (float) time_run[run] / 1000.0 << '\t'; // ms
			}
			cout << "\n\n";
        }

		/* free up memory */
		free(features[0]);
		free(features);
	}

	return 0;
}