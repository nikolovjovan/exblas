/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma OPENCL EXTENSION cl_khr_fp64                   : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  // For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif

typedef double data_t;

#define BIN_COUNT      39
#define K              8                    // High-radix carry-save bits
#define digits         56
#define deltaScale     72057594037927936.0  // Assumes K>0
#define f_words        20 
#define TSAFE          0

#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
#ifdef USE_KNUTH
    double Knuth2Sum(double a, double b, double *s) {
        double r = a + b;
        double z = r - a;
        *s = (a - (r - z)) + (b - z);
        return r;
    }
#else
    //twosum
    double Knuth2Sum(double a, double b, double *s) {
        double r = a + b;
        int doswap = fabs(b) > fabs(a);
        double a2 = doswap ? b : a;
        double b2 = doswap ? a : b;
        *s = (a2 - r) + b2;
        return r;
    }
#endif

double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

// signedcarry in {-1, 0, 1}
long xadd(__global volatile long *sa, long x, uchar *of) {
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    long y = atom_add(sa, x);
    long z = y + x; // since the value sa->accumulator[i] can be changed by another work item

    // TODO: cover also underflow
    *of = 0;
    if(x > 0 && y > 0 && z < 0)
        *of = 1;
    if(x < 0 && y < 0 && z > 0)
        *of = 1;

    return y;
}

double OddRoundSumNonnegative(double th, double tl) {
    union {
        double d;
        long l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}


////////////////////////////////////////////////////////////////////////////////
// Rounding functions
////////////////////////////////////////////////////////////////////////////////
int Normalize(__global volatile long *accumulator, int *imin, int *imax) {
  if (*imin > *imax) {
    return 0;
  }
  long carry_in = accumulator[*imin] >> digits;
  accumulator[*imin] -= carry_in << digits;
  int i;
  // Sign-extend all the way
  for (i = *imin + 1; i < BIN_COUNT; ++i) {
#if 1
    long carry_out = accumulator[i] >> digits;    // Arithmetic shift
    accumulator[i] += carry_in - (carry_out << digits);
#else
    // BUGGY
    // get carry of accumulator[i] + carry_in
    unsigned char overflow;
    long oldword = xadd(&accumulator[i], carry_in, &overflow);
    int s = oldword > 0;
    long carrybit = (s ? 1ll << K : -1ll << K);

    long carry_out = (accumulator[i] >> digits) + carrybit;// Arithmetic shift
    accumulator[i] -= carry_out << digits;
#endif
    carry_in = carry_out;
  }
  imax = i - 1;

  if (carry_in != 0 && carry_in != -1) {
    //TODO: handle overflow
    //status = Overflow;
  }
  return carry_in < 0;
}

double Round(__global volatile long *accumulator) {
  int imin = 38; 
  int imax = 0;
  int negative = Normalize(accumulator, &imin, &imax);

  //Find leading word
  int i;
  //Skip zeroes
  for (i = imax; accumulator[i] == 0 && i >= imin; --i) {
  }
  if (negative) {
    //Skip ones
    for (; accumulator[i] == ((1ll << digits) - 1) && i >= imin; --i) {
    }
  }
  if (i < 0) {
    //TODO: should we preserve sign of zero?
    return 0.;
  }

  long hiword = negative ? (1ll << digits) - accumulator[i] : accumulator[i];
  double rounded = (double) hiword;
  double hi = ldexp(rounded, (i - f_words) * digits);
  if (i == 0) {
    return negative ? -hi : hi;  // Correct rounding achieved
  }
  hiword -= (long) rint(rounded);
  double mid = ldexp((double) hiword, (i - f_words) * digits);

  //Compute sticky
  long sticky = 0;
  for (int j = imin; j != i - 1; ++j) {
    sticky |= negative ? (1ll << digits) - accumulator[j] : accumulator[j];
  }

  long loword = negative ? (1ll << digits) - accumulator[i - 1] : accumulator[i - 1];
  loword |= !!sticky;
  double lo = ldexp((double) loword, (i - 1 - f_words) * digits);

  //Now add3(hi, mid, lo)
  //No overlap, we have already normalized
  if (mid != 0) {
    lo = OddRoundSumNonnegative(mid, lo);
  }
  //Final rounding
  hi = hi + lo;
  return negative ? -hi : hi;
}


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial accumulators
////////////////////////////////////////////////////////////////////////////////
void AccumulateWord(__global volatile long *sa, int i, long x) {
  // With atomic accumulator updates
  // accumulation and carry propagation can happen in any order
  long carry = x;
  long carrybit;
  uchar overflow;
  long oldword = xadd(&sa[i], x, &overflow);

  // To propagate over- or underflow 
  while (overflow) {
    // Carry or borrow
    // oldword has sign S
    // x has sign S
    // accumulator[i] has sign !S (just after update)
    // carry has sign !S
    // carrybit has sign S
    carry = (oldword + carry) >> digits;
    bool s = oldword > 0;
    carrybit = (s ? 1l << K : -1l << K);

    // Cancel carry-save bits
    xadd(&sa[i], (long) -(carry << digits), &overflow);
    if (TSAFE && (s ^ overflow)) {
      carrybit *= 2;
    }
    carry += carrybit;

    ++i;
    if (i >= BIN_COUNT) {
      return;
    }
    oldword = xadd(&sa[i], carry, &overflow);
  }
}

void Accumulate(__global volatile long *sa, double x) {
  if (x == 0)
    return;

  int e;
  frexp(x, &e);
  int exp_word = e / digits;  // Word containing MSbit
  int iup = exp_word + f_words;

  double xscaled = ldexp(x, -digits * exp_word);

  int i;
  for (i = iup; xscaled != 0; --i) {
    double xrounded = rint(xscaled);
    long xint = (long) xrounded;
 
    AccumulateWord(sa, i, xint);

    xscaled -= xrounded;
    xscaled *= deltaScale;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Matrix multiplication on the device: C = A * B
// uiWA is A's width and uiWB is B's width
////////////////////////////////////////////////////////////////////////////////
__kernel void matrixMul(
    __global long* Accus,
    __global data_t* C,
    __global data_t* A,
    __global data_t* B, 
    int uiWA,
    int uiWB,
    __local data_t* As,
    __local data_t* Bs
) {
    //Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    //Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    //Index of the first sub-matrix of A processed by the block
    int aBegin = uiWA * BLOCK_SIZE * by;

    //Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + uiWA - 1;

    //Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    //Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * uiWB;

    //A superaccumulator that corresponds to a single value in the matrix C
    int c = uiWB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    __global long *g_workingBase = Accus + (c + uiWB * ty + tx);

    //for floating-point expansion
    double sum[NBFPE] = {0};

    //Loop over all the sub-matrices of A and B
    //required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        //Load the matrices from device memory to shared memory;
        //each thread loads one element of each matrix
        AS(ty, tx) = A[a + uiWA * ty + tx];
        BS(ty, tx) = B[b + uiWB * ty + tx];
	
        //Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        //Multiply the two matrices together;
        //each thread computes one element of the block sub-matrix
        #ifdef NVIDIA
          #pragma unroll
        #endif
        for (int k = 0; k < BLOCK_SIZE; ++k) {
	    double r; //residual of multiplication
            double x = TwoProductFMA(AS(ty, k), BS(k, tx), &r);
            #ifdef NVIDIA
                #pragma unroll
            #endif
            for(uint i = 0; i != NBFPE; ++i) {
                double s; //residual of addition
                sum[i] = Knuth2Sum(sum[i], x, &s);
                x = s + r;
            }
            if(x != 0.0) {
	        Accumulate(g_workingBase, x);
            }
	}

        //Synchronize to make sure that the preceding computation is done before 
        //loading two new sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //Flush to the accumulator
#ifdef NVIDIA
    #pragma unroll
#endif
    for(uint i = 0; i != NBFPE; ++i) {
	Accumulate(g_workingBase, sum[i]);
    }

    //TODO: Round the results back


    //int c = uiWB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    //TODO: the first non-zero from rigth
    C[c + uiWB * ty + tx] = sum[0];
}

