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

void usage(char *argv0)
{
    char help[] = "Usage: %s [switches] -i filename\n"
                  "       -i filename     : file containing data to be clustered\n"
                  "       -b              : input file is in binary format\n"
                  "       -k              : number of clusters (default is 8)\n"
                  "       -r              : number of repeats for reproducibilty study (default is 0)\n"
                  "       -t threshold    : threshold value\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

bool diff(int nclusters, int numAttributes, float **cluster_centres_1, float **cluster_centres_2)
{
    for (int i = 0; i < nclusters; i++)
        for (int j = 0; j < numAttributes; j++)
            if (cluster_centres_1[i][j] != cluster_centres_2[i][j])
                return true;
    return false;
}

void run(int nrepeats, bool parallel, bool reproducible, int numObjects, int numAttributes,
         float **attributes, int nclusters, float threshold, float** &cluster_centres, double &time)
{
    printf("Running %s (%sreproducible) implementation...\n", parallel ? "parallel" : "sequential", reproducible ? "" : "non-");
    float **tmp_cluster_centres = NULL;
    for (int i = 0; i <= nrepeats; i++) {
        if (i == 0)
            time = omp_get_wtime();
        cluster(parallel, reproducible, numObjects, numAttributes, attributes, /* [numObjects][numAttributes] */
                nclusters, threshold, &tmp_cluster_centres);
        if (i == 0) {
            time = omp_get_wtime() - time;
            cluster_centres = tmp_cluster_centres;
            tmp_cluster_centres = NULL;
        } else if (diff(nclusters, numAttributes, cluster_centres, tmp_cluster_centres)) {
            printf("%s (%sreproducible) implementation not reproducible after %d runs!\n",
                   parallel ? "Parallel" : "Sequential", reproducible ? "" : "non-", i);
            break;
        }
    }
}

int main(int argc, char **argv)
{
    int opt;
    extern char *optarg;
    extern int optind;
    int nclusters = 5;
    char *filename = 0;
    float *buf;
    float **attributes;
    float **cluster_centres_seq = NULL;
    float **cluster_centres_omp = NULL;
    float **cluster_centres_seq_rep = NULL;
    float **cluster_centres_omp_rep = NULL;
    int i, j;

    int numAttributes;
    int numObjects;
    char line[1024];
    int isBinaryFile = 0;
    int nrepeats = 0;
    float threshold = 0.001;
    double time_seq;
    double time_omp;
    double time_seq_rep;
    double time_omp_rep;

    while ((opt = getopt(argc, argv, "i:k:r:t:b")) != EOF) {
        switch (opt) {
        case 'i':
            filename = optarg;
            break;
        case 'b':
            isBinaryFile = 1;
            break;
        case 't':
            threshold = atof(optarg);
            break;
        case 'k':
            nclusters = atoi(optarg);
            break;
        case 'r':
            nrepeats = atoi(optarg);
            if (nrepeats < 0) {
                printf("Invalid number of repeats: %d!", nrepeats);
                exit(1);
            }
            break;
        case '?':
            usage(argv[0]);
            break;
        default:
            usage(argv[0]);
            break;
        }
    }

    if (filename == 0)
        usage(argv[0]);

    numAttributes = numObjects = 0;

    /* from the input file, get the numAttributes and numObjects ------------*/

    if (isBinaryFile) {
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        read(infile, &numObjects, sizeof(int));
        read(infile, &numAttributes, sizeof(int));

        /* allocate space for attributes[] and read attributes of all objects */
        buf = (float *) malloc(numObjects * numAttributes * sizeof(float));
        attributes = (float **) malloc(numObjects * sizeof(float *));
        attributes[0] = (float *) malloc(numObjects * numAttributes * sizeof(float));
        for (i = 1; i < numObjects; i++)
            attributes[i] = attributes[i - 1] + numAttributes;

        read(infile, buf, numObjects * numAttributes * sizeof(float));

        close(infile);
    } else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
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
        for (i = 1; i < numObjects; i++)
            attributes[i] = attributes[i - 1] + numAttributes;
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL)
                continue;
            for (j = 0; j < numAttributes; j++) {
                buf[i] = atof(strtok(NULL, " ,\t\n"));
                i++;
            }
        }
        fclose(infile);
    }

    printf("I/O completed\n");
    printf("  Number of clusters: %d\n", nclusters);
    printf("  Number of objects: %d\n", numObjects);
    printf("  Number of attributes: %d\n", numAttributes);
    printf("  Number of available threads: %d\n", omp_get_max_threads());
    printf("  Number of repeats: %d\n", nrepeats);

    memcpy(attributes[0], buf, numObjects * numAttributes * sizeof(float));

    run(nrepeats, false, false, numObjects, numAttributes, attributes, nclusters, threshold, cluster_centres_seq, time_seq);
    run(nrepeats, false, true, numObjects, numAttributes, attributes, nclusters, threshold, cluster_centres_seq_rep, time_seq_rep);
    run(nrepeats, true, false, numObjects, numAttributes, attributes, nclusters, threshold, cluster_centres_omp, time_omp);
    run(nrepeats, true, true, numObjects, numAttributes, attributes, nclusters, threshold, cluster_centres_omp_rep, time_omp_rep);

    printf("Non-reproducible sequential and parallel results %smatch!\n",
           diff(nclusters, numAttributes, cluster_centres_seq, cluster_centres_omp) ? "do not " : "");

    printf("Reproducible sequential and parallel results %smatch!\n",
           diff(nclusters, numAttributes, cluster_centres_seq_rep, cluster_centres_omp_rep) ? "do not " : "");

    if (diff(nclusters, numAttributes, cluster_centres_seq, cluster_centres_seq_rep)) {
        printf("Non-reproducible and reproducible sequential results do not match!\n");
    }

    if (diff(nclusters, numAttributes, cluster_centres_omp, cluster_centres_omp_rep)) {
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

    free(attributes);
    free(cluster_centres_seq[0]);
    free(cluster_centres_seq);
    free(cluster_centres_omp[0]);
    free(cluster_centres_omp);
    free(cluster_centres_seq_rep[0]);
    free(cluster_centres_seq_rep);
    free(cluster_centres_omp_rep[0]);
    free(cluster_centres_omp_rep);
    free(buf);

    return 0;
}