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
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "kmeans.h"

#include <iomanip>
#include <iostream>

using namespace std;

constexpr char* input_files[] = { "100", "204800.txt", "819200.txt", "kdd_cup" };

constexpr char* algorithm[] = { "accumulator-only", "fpe2", "fpe4", "fpe8ee" };
constexpr int fpe[] = { 0, 2, 4, 8 };
constexpr bool early_exit[] = { false, false, false, true };

void execute(bool parallel, bool reproducible, int numObjects, int numAttributes,
         float **attributes, int nclusters, float threshold,
         const int fpe, const bool early_exit)
{
    float **cluster_centres = NULL;
    double time = omp_get_wtime();
    cluster(parallel, reproducible, numObjects, numAttributes, attributes, /* [numObjects][numAttributes] */
            nclusters, threshold, &cluster_centres, fpe, early_exit);
    time = omp_get_wtime() - time;
    cout << fixed << setprecision(10) << (float) time * 1000.0 << '\t'; // ms
    free(cluster_centres);
}

int main(int argc, char **argv)
{
    int opt;
    extern char *optarg;
    extern int optind;
    int nclusters = 5;
    float *buf;
    float **attributes;

    int numAttributes = 0;
    int numObjects = 0;
    char line[1024];
    int nrepeats = 0;
    float threshold = 0.001;

    const int exe_path_len = strrchr(argv[0], '/') - argv[0] + 1;
    char exe_path[256];
    strncpy(exe_path, argv[0], exe_path_len);
    exe_path[exe_path_len] = '\0';

    char input_file_path[256];

    cout << "unit: [ms]\n\n";

    int nfiles = sizeof(input_files) / sizeof(input_files[0]);

    for (int file_idx = 0; file_idx < nfiles; ++file_idx)
    {
        strncpy(input_file_path, exe_path, exe_path_len + 1);
        strcat(input_file_path, "data/");
        strcat(input_file_path, input_files[file_idx]);

        cout << input_files[file_idx] << "\n\n";

        /* from the input file, get the numAttributes and numObjects ------------*/
        {
            numObjects = 0;
            numAttributes = 0;

            FILE *infile;
            if ((infile = fopen(input_file_path, "r")) == NULL) {
                fprintf(stderr, "Error: no such file (%s)\n", input_file_path);
                exit(1);
            }
            while (fgets(line, 1024, infile) != NULL)
                if (strtok(line, " \t\n") != 0)
                    numObjects++;
            rewind(infile);
            while (fgets(line, 1024, infile) != NULL) {
                if (strtok(line, " \t\n") != 0) {
                    /* ignore the id (first attribute): numAttributes = 1; */
                    while (strtok(NULL, " ,\t\n") != NULL)
                        numAttributes++;
                    break;
                }
            }

            /* allocate space for attributes[] and read attributes of all objects */
            buf = (float *) malloc(numObjects * numAttributes * sizeof(float));
            attributes = (float **) malloc(numObjects * sizeof(float *));
            attributes[0] = (float *) malloc(numObjects * numAttributes * sizeof(float));
            int i;
            for (i = 1; i < numObjects; i++)
                attributes[i] = attributes[i - 1] + numAttributes;
            rewind(infile);
            i = 0;
            while (fgets(line, 1024, infile) != NULL) {
                if (strtok(line, " \t\n") == NULL)
                    continue;
                for (int j = 0; j < numAttributes; j++) {
                    buf[i] = atof(strtok(NULL, " ,\t\n"));
                    i++;
                }
            }
            fclose(infile);
        }

        memcpy(attributes[0], buf, numObjects * numAttributes * sizeof(float));
        free(buf);

        int algcnt = sizeof(algorithm) / sizeof(char*);

        for (int algidx = 0; algidx < algcnt; ++algidx)
        {
            cout << algorithm[algidx] << "\n\n";

            cout << "seq\t";
            for (int run = 0; run < 3; ++run) execute (false, true, numObjects, numAttributes, attributes, nclusters, threshold, fpe[algidx], early_exit[algidx]);
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

                for (int run = 0; run < 3; ++run) execute (true, true, numObjects, numAttributes, attributes, nclusters, threshold, fpe[algidx], early_exit[algidx]);
                cout << '\n';
            }

            cout << '\n';
        }

        free(attributes);
    }

    return 0;
}