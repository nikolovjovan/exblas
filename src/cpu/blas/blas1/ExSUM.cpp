/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "ExSUM.hpp"
#include "blas1.hpp"

#ifdef EXBLAS_TIMING
    #define iterations 50
#endif

/*
 * Singlethreaded summation
 * If fpe < 2, use superaccumulators only,
 * Otherwise, use floating-point expansions of size FPE with superaccumulators when needed
 * early_exit corresponds to the early-exit technique
 */
double exsumsingle(int Ng, double *ag, int inca, int offset, int fpe, bool early_exit, bool parallel) {
    if (fpe < 0) {
        fprintf(stderr, "Size of floating-point expansion should be a positive number. Preferably, it should be in the interval [2, 8]\n");
        exit(1);
    }

    int N;
    double *a;

#ifdef EXBLAS_MPI
    if (parallel)
    {
        int np = 1, p, err;
        MPI_Comm_rank(MPI_COMM_WORLD, &p);
        MPI_Comm_size(MPI_COMM_WORLD, &np);

        Superaccumulator acc, acc_fin;
        N = Ng / np + Ng % np;

        a = (double *)_mm_malloc(N * sizeof(double), 32);
        if (!a)
            fprintf(stderr, "Cannot allocate memory for per process array\n");

        int i;
        if (p == 0) {
            //distribute
            a = ag;
            ag = ag + N;
            for (i = 1; i < np; i++) {
                err = MPI_Send(ag + (i - 1)  * (N - Ng % np), N - Ng % np, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                if (err != MPI_SUCCESS)
                    fprintf(stderr, "MPI_Send does not word properly %d\n", err);
            }
        } else {
            MPI_Status status;
            err = MPI_Recv(a, N - Ng % np, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            if (err != MPI_SUCCESS)
                fprintf(stderr, "MPI_Recv does not word properly %d\n", err);
        }
    }
    else
#endif
    {
        N = Ng;
        a = ag;
    }

    // with superaccumulators only
    if (fpe < 2)
        return ExSUMSuperacc(N, a, inca, offset, parallel);

    if (early_exit) {
        if (fpe <= 4)
            return (ExSUMFPE<FPExpansionVect<Vec4d, 4, FPExpansionTraits<true> > >)(N, a, inca, offset, parallel);
        if (fpe <= 6)
            return (ExSUMFPE<FPExpansionVect<Vec4d, 6, FPExpansionTraits<true> > >)(N, a, inca, offset, parallel);
        if (fpe <= 8)
            return (ExSUMFPE<FPExpansionVect<Vec4d, 8, FPExpansionTraits<true> > >)(N, a, inca, offset, parallel);
    } else { // ! early_exit
        if (fpe == 2) 
        return (ExSUMFPE<FPExpansionVect<Vec4d, 2> >)(N, a, inca, offset, parallel);
        if (fpe == 3) 
        return (ExSUMFPE<FPExpansionVect<Vec4d, 3> >)(N, a, inca, offset, parallel);
        if (fpe == 4) 
        return (ExSUMFPE<FPExpansionVect<Vec4d, 4> >)(N, a, inca, offset, parallel);
        if (fpe == 5) 
        return (ExSUMFPE<FPExpansionVect<Vec4d, 5> >)(N, a, inca, offset, parallel);
        if (fpe == 6) 
        return (ExSUMFPE<FPExpansionVect<Vec4d, 6> >)(N, a, inca, offset, parallel);
        if (fpe == 7) 
        return (ExSUMFPE<FPExpansionVect<Vec4d, 7> >)(N, a, inca, offset, parallel);
        if (fpe == 8) 
        return (ExSUMFPE<FPExpansionVect<Vec4d, 8> >)(N, a, inca, offset, parallel);
    }

    return 0.0;
}

/*
 * Parallel summation using our algorithm
 * If fpe < 2, use superaccumulators only,
 * Otherwise, use floating-point expansions of size FPE with superaccumulators when needed
 * early_exit corresponds to the early-exit technique
 */
double exsum(int Ng, double *ag, int inca, int offset, int fpe, bool early_exit, bool parallel) {
    if (parallel)
    {
        oneapi::tbb::task_arena arena;
        return arena.execute([&fpe, &Ng, &ag, &inca, &offset, &early_exit, &parallel] {
            return exsumsingle(Ng, ag, inca, offset, fpe, early_exit, parallel);
        });
    }
    return exsumsingle(Ng, ag, inca, offset, fpe, early_exit, parallel);
}

/*
 * Our alg with superaccumulators only
 */
double ExSUMSuperacc(int N, double *a, int inca, int offset, bool parallel) {
    double dacc;
#ifdef EXBLAS_TIMING
    double t, mint = 10000;
    uint64_t tstart, tend;
    for(int iter = 0; iter != iterations; ++iter) {
    	tstart = rdtsc();
#endif

        TBBlongsum tbbsum(a);

        if (parallel)
        {
            tbb::parallel_reduce(tbb::blocked_range<size_t>(offset, N, inca), tbbsum);
        }
        else
        {
            tbbsum(tbb::blocked_range<size_t>(offset, N, inca));
        }

#ifdef EXBLAS_MPI
        if (parallel)
        {
            tbbsum.acc.Normalize();
            std::vector<int64_t> result(tbbsum.acc.get_f_words() + tbbsum.acc.get_e_words(), 0);
            //MPI_Reduce((int64_t *) &tbbsum.acc.accumulator[0], (int64_t *) &acc_fin.accumulator[0], get_f_words() + get_e_words(), MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&(tbbsum.acc.get_accumulator()[0]), &(result[0]), tbbsum.acc.get_f_words() + tbbsum.acc.get_e_words(), MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            Superaccumulator acc_fin(result);
            dacc = acc_fin.Round();
        }
        else
#endif
        {
            dacc = tbbsum.acc.Round();
        }

#ifdef EXBLAS_TIMING
        tend = rdtsc();
        t = double(tend - tstart) / N;
        mint = std::min(mint, t);
    }
    fprintf(stderr, "%f ", mint);
#endif

    return dacc;
}

/**
 * \brief Parallel reduction step
 *
 * \param step step among threads
 * \param tid1 id of the first thread
 * \param tid2 id of the second thread
 * \param acc1 superaccumulator of the first thread
 * \param acc2 superaccumulator of the second thread
 */
inline static void ReductionStep(int step, int tid1, int tid2, Superaccumulator * acc1, Superaccumulator * acc2,
    int volatile * ready1, int volatile * ready2)
{
    _mm_prefetch((char const*)ready2, _MM_HINT_T0);
    // Wait for thread 2
    while(*ready2 < step) {
        // wait
        _mm_pause();
    }
    acc1->Accumulate(*acc2);
}

/**
 * \brief Final step of summation -- Parallel reduction among threads
 *
 * \param tid thread ID
 * \param tnum number of threads
 * \param acc superaccumulator
 */
inline static void Reduction(unsigned int tid, unsigned int tnum, std::vector<int32_t>& ready,
    std::vector<Superaccumulator>& acc, int const linesize)
{
    // Custom reduction
    for(unsigned int s = 1; (1 << (s-1)) < tnum; ++s) 
    {
        int32_t volatile * c = &ready[tid * linesize];
        ++*c;
        if(tid % (1 << s) == 0) {
            unsigned int tid2 = tid | (1 << (s-1));
            if(tid2 < tnum) {
                //acc[tid2].Prefetch(); // No effect...
                ReductionStep(s, tid, tid2, &acc[tid], &acc[tid2],
                    &ready[tid * linesize], &ready[tid2 * linesize]);
            }
        }
    }
}

// TODO: Add support for inca (probably best to copy required elements to a new array and call this without inca).
//
template<typename CACHE> double ExSUMFPE(int N, double *a, int inca, int offset, bool parallel) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int maxthreads = parallel ? omp_get_max_threads() : 1;
    double dacc;
#ifdef EXBLAS_TIMING
    double t, mint = 10000;
    uint64_t tstart, tend;
    for(int iter = 0; iter != iterations; ++iter) {
        tstart = rdtsc();
#endif
        if (maxthreads > 1)
        {
            std::vector<Superaccumulator> acc(maxthreads);
            std::vector<int32_t> ready(maxthreads * linesize);

            #pragma omp parallel
            {
                unsigned int tid = omp_get_thread_num();
                unsigned int tnum = omp_get_num_threads();

                CACHE cache(acc[tid]);
                *(int32_t volatile *)(&ready[tid * linesize]) = 0;  // Race here, who cares?

                int l = offset + ((tid * int64_t(N - offset)) / tnum) & ~7ul;
                int r = offset + ((((tid+1) * int64_t(N - offset)) / tnum) & ~7ul) - 1;

                for (int i = l; i < r; i += 8) {
                    asm ("# myloop");
                    cache.Accumulate(Vec4d().load(a + i), Vec4d().load(a + i + 4));
                }
                cache.Flush();

                if (tid == tnum - 1)
                {
                    // Accumulate the rest of the elements on the last thread.
                    //
                    for (int i = r + 1; i < N; ++i) {
                        acc[tid].Accumulate(a[i]);
                    }
                }

                acc[tid].Normalize();

                Reduction(tid, tnum, ready, acc, linesize);
            }

#ifdef EXBLAS_MPI
            acc[0].Normalize();
            std::vector<int64_t> result(acc[0].get_f_words() + acc[0].get_e_words(), 0);
            MPI_Reduce(&(acc[0].get_accumulator()[0]), &(result[0]), acc[0].get_f_words() + acc[0].get_e_words(), MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            //MPI_Reduce((int64_t *) &acc[0].accumulator[0], (int64_t *) &acc_fin.accumulator[0], get_f_words() + get_e_words(), MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            Superaccumulator acc_fin(result);
            dacc = acc_fin.Round();
#else
            dacc = acc[0].Round();
#endif
        }
        else
        {
            Superaccumulator acc;
            CACHE cache(acc);

            int l = offset;
            int r = offset + (int64_t(N - offset) & ~7ul) - 1;

            for (int i = l; i < r; i += 8) {
                asm ("# myloop");
                cache.Accumulate(Vec4d().load(a + i), Vec4d().load(a + i + 4));
            }
            cache.Flush();

            // Accumulate the rest of the elements.
            //
            for (int i = r + 1; i < N; ++i) {
                acc.Accumulate(a[i]);
            }

            acc.Normalize();
            dacc = acc.Round();
        }

#ifdef EXBLAS_TIMING
        tend = rdtsc();
        t = double(tend - tstart) / N;
        mint = std::min(mint, t);
    }
    fprintf(stderr, "%f ", mint);
#endif

    return dacc;
}

