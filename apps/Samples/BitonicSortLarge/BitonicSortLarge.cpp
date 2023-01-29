// Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.

// Modified for NoCL, November 2022.

#include <NoCL.h>
#include <Rand.h>
#include <FastZero.h>

// Size of arrays being sorted
#define LOCAL_SIZE_LIMIT 4096

// Sort two key/value pairs
inline void twoSort(unsigned *keyA, unsigned* valA,
                    unsigned *keyB, unsigned* valB, unsigned dir)
{
  if ((*keyA > *keyB) == dir) {
    unsigned t;
    t = *keyA; *keyA = *keyB; *keyB = t;
    t = *valA; *valA = *valB; *valB = t;
  }
  noclConverge();
}

// Bottom-level bitonic sort
// Even / odd subarrays (of LOCAL_SIZE_LIMIT points) are
// sorted in opposite directions
struct BitonicSortLocal : Kernel {
  unsigned *d_DstKey_arg;
  unsigned *d_DstVal_arg;
  unsigned *d_SrcKey_arg;
  unsigned *d_SrcVal_arg;

  void kernel() {
    unsigned* l_key = shared.array<unsigned, LOCAL_SIZE_LIMIT>();
    unsigned* l_val = shared.array<unsigned, LOCAL_SIZE_LIMIT>();

    // Offset to the beginning of subbatch and load data
    unsigned* d_SrcKey =
      d_SrcKey_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    unsigned* d_SrcVal =
      d_SrcVal_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    unsigned* d_DstKey =
      d_DstKey_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    unsigned* d_DstVal =
      d_DstVal_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    l_key[threadIdx.x + 0] = d_SrcKey[0];
    l_val[threadIdx.x + 0] = d_SrcVal[0];
    l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] =
      d_SrcKey[(LOCAL_SIZE_LIMIT / 2)];
    l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] =
      d_SrcVal[(LOCAL_SIZE_LIMIT / 2)];

    unsigned global_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned comparatorI = global_id & ((LOCAL_SIZE_LIMIT / 2) - 1);

    for (unsigned size = 2; size < LOCAL_SIZE_LIMIT; size <<= 1){ 
      // Bitonic merge
      unsigned dir = (comparatorI & (size / 2)) != 0;
      for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        unsigned pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        twoSort(
          &l_key[pos + 0], &l_val[pos + 0],
          &l_key[pos + stride], &l_val[pos + stride], dir);
      }
    }

    // Odd / even arrays of LOCAL_SIZE_LIMIT elements
    // sorted in opposite directions  
    {
      unsigned dir = (blockIdx.x & 1);
      for(unsigned stride = LOCAL_SIZE_LIMIT / 2; stride > 0; stride >>= 1){
        __syncthreads();
        unsigned pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        twoSort(
          &l_key[pos + 0], &l_val[pos + 0],
          &l_key[pos + stride], &l_val[pos + stride], dir);
      }
    }

    __syncthreads();
    d_DstKey[0] = l_key[threadIdx.x + 0];
    d_DstVal[0] = l_val[threadIdx.x + 0];
    d_DstKey[(LOCAL_SIZE_LIMIT / 2)] =
      l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
    d_DstVal[(LOCAL_SIZE_LIMIT / 2)] =
      l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
  }
};

// Bitonic merge iteration for 'stride' >= LOCAL_SIZE_LIMIT
struct BitonicMergeGlobal : Kernel {
  unsigned* d_DstKey;
  unsigned* d_DstVal;
  unsigned* d_SrcKey;
  unsigned* d_SrcVal;
  unsigned arrayLength;
  unsigned size;
  unsigned stride;
  unsigned sortDir;

  void kernel() {
    unsigned global_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned global_comparatorI = global_id;
    unsigned comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    // Bitonic merge
    unsigned dir = sortDir ^ ( (comparatorI & (size / 2)) != 0 );
    unsigned pos =
      2 * global_comparatorI - (global_comparatorI & (stride - 1));

    unsigned keyA = d_SrcKey[pos + 0];
    unsigned valA = d_SrcVal[pos + 0];
    unsigned keyB = d_SrcKey[pos + stride];
    unsigned valB = d_SrcVal[pos + stride];

    twoSort(&keyA, &valA, &keyB, &valB, dir);

    d_DstKey[pos + 0] = keyA;
    d_DstVal[pos + 0] = valA;
    d_DstKey[pos + stride] = keyB;
    d_DstVal[pos + stride] = valB;
  }
};

//Combined bitonic merge steps for
//'size' > LOCAL_SIZE_LIMIT and 'stride' = [1 .. LOCAL_SIZE_LIMIT / 2]
struct BitonicMergeLocal : Kernel {
  unsigned* d_DstKey_arg;
  unsigned* d_DstVal_arg;
  unsigned* d_SrcKey_arg;
  unsigned* d_SrcVal_arg;
  unsigned arrayLength;
  unsigned stride_arg;
  unsigned size;
  unsigned sortDir;

  void kernel() {
    unsigned* l_key = shared.array<unsigned, LOCAL_SIZE_LIMIT>();
    unsigned* l_val = shared.array<unsigned, LOCAL_SIZE_LIMIT>();

    unsigned* d_SrcKey =
      d_SrcKey_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    unsigned* d_SrcVal =
      d_SrcVal_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    unsigned* d_DstKey =
      d_DstKey_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    unsigned* d_DstVal =
      d_DstVal_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    l_key[threadIdx.x + 0] = d_SrcKey[0];
    l_val[threadIdx.x + 0] = d_SrcVal[0];
    l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] =
      d_SrcKey[(LOCAL_SIZE_LIMIT / 2)];
    l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] =
      d_SrcVal[(LOCAL_SIZE_LIMIT / 2)];

    // Bitonic merge
    unsigned global_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned comparatorI = global_id & ((arrayLength / 2) - 1);
    unsigned dir = sortDir ^ ( (comparatorI & (size / 2)) != 0 );
    unsigned stride = stride_arg;
    for(; stride > 0; stride >>= 1){
      __syncthreads();
      unsigned pos =
        2 * threadIdx.x - (threadIdx.x & (stride - 1));
      twoSort(&l_key[pos + 0], &l_val[pos + 0],
              &l_key[pos + stride], &l_val[pos + stride], dir);
    }

    __syncthreads();
    d_DstKey[0] = l_key[threadIdx.x + 0];
    d_DstVal[0] = l_val[threadIdx.x + 0];
    d_DstKey[(LOCAL_SIZE_LIMIT / 2)] =
      l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
    d_DstVal[(LOCAL_SIZE_LIMIT / 2)] =
      l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Array size and number of arrays for benchmarking
  int N = 1 << (isSim ? 13 : 18);

  // Input and output vectors
  simt_aligned unsigned srcKeys[N], srcVals[N];
  simt_aligned unsigned dstKeys[N], dstVals[N];

  // fast zero the blocks of memory
  #if EnableFastZeroing
  fastZero(srcKeys, N * sizeof(unsigned)); 
  fastZero(srcVals, N * sizeof(unsigned));
  fastZero(dstKeys, N * sizeof(unsigned));
  fastZero(dstVals, N * sizeof(unsigned));
  #endif

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < N; i++) {
    srcKeys[i] = rand15(&seed);
    srcVals[i] = rand15(&seed);
  }

  // Instantiate kernels
  BitonicSortLocal sortLocal;
  BitonicMergeLocal mergeLocal;
  BitonicMergeGlobal mergeGlobal;

  // Launch BitonicSortLocal
  sortLocal.d_SrcKey_arg = srcKeys;
  sortLocal.d_SrcVal_arg = srcVals;
  sortLocal.d_DstKey_arg = dstKeys;
  sortLocal.d_DstVal_arg = dstVals;
  sortLocal.blockDim.x = LOCAL_SIZE_LIMIT / 2;
  sortLocal.gridDim.x = N / LOCAL_SIZE_LIMIT;
  noclRunKernelAndDumpStats(&sortLocal);

  for (unsigned size = 2 * LOCAL_SIZE_LIMIT; size <= N; size <<= 1) {
    for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
      if (stride >= LOCAL_SIZE_LIMIT) {
        // Launch BitonicMergeGlobal
        mergeGlobal.d_SrcKey = dstKeys;
        mergeGlobal.d_SrcVal = dstVals;
        mergeGlobal.d_DstKey = dstKeys;
        mergeGlobal.d_DstVal = dstVals;
        mergeGlobal.arrayLength = N;
        mergeGlobal.size = size;
        mergeGlobal.stride = stride;
        mergeGlobal.sortDir = 1;
        mergeGlobal.blockDim.x = LOCAL_SIZE_LIMIT / 2;
        mergeGlobal.gridDim.x = N / LOCAL_SIZE_LIMIT;
        noclRunKernelAndDumpStats(&mergeGlobal);
      }
      else {
        // Launch BitonicMergeLocal
        mergeLocal.d_SrcKey_arg = dstKeys;
        mergeLocal.d_SrcVal_arg = dstVals;
        mergeLocal.d_DstKey_arg = dstKeys;
        mergeLocal.d_DstVal_arg = dstVals;
        mergeLocal.arrayLength = N;
        mergeLocal.size = size;
        mergeLocal.stride_arg = stride;
        mergeLocal.sortDir = 1;
        mergeLocal.blockDim.x = LOCAL_SIZE_LIMIT / 2;
        mergeLocal.gridDim.x = N / LOCAL_SIZE_LIMIT;
        noclRunKernelAndDumpStats(&mergeLocal);
        break;
      }
    }
  }

  // Check result
  bool ok = true;
  for (int i = 0; i < N-1; i++)
    ok = ok && dstKeys[i] <= dstKeys[i+1];

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
