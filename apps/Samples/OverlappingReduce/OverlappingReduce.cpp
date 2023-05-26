#include <NoCL.h>
#include <Rand.h>

// Kernel for vector summation
template <int BlockSize> struct Reduce : Kernel {
  int len;
  int *in, *sum;
  
  void kernel() {
    int* block = shared.array<int, BlockSize>();

    // Sum global memory
    block[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
      block[threadIdx.x] += in[i];

    __syncthreads();

    // Sum shared local memory
    for(int i = blockDim.x >> 1; i > 0; i >>= 1)  {
      if (threadIdx.x < i)
        block[threadIdx.x] += block[threadIdx.x + i];
      __syncthreads();
    }

    // Write sum to global memory
    if (threadIdx.x == 0) *sum = block[0];
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Vector size for benchmarking
  int N = isSim ? 3000 : 1000;

  // Input and outputs
  simt_aligned int inFirst[N];
  simt_aligned int inSecond[N];
  int sumFirst, sumSecond;

  // Initialise inputs
  uint32_t seed = 1;
  int acc = 0;
  for (int i = 0; i < N; i++) {
    int r = rand15(&seed);
    inFirst[i] = r;
    inSecond[i] = r;
    acc += r;
  }

  // Instantiate first kernel
  Reduce<SIMTWarps * SIMTLanes> k1;

  // Use a single block of threads
  k1.blockDim.x = SIMTWarps * SIMTLanes / 2;
  k1.gridDim.x = 10;
  // Assign parameters
  k1.len = N;
  k1.in = inFirst;
  k1.sum = &sumFirst;

  // Instantiate second kernel
  Reduce<SIMTWarps * SIMTLanes> k2;

  // Use a single block of threads
  k2.blockDim.x = SIMTWarps * SIMTLanes / 2;
  k2.gridDim.x = 10;

  // Assign parameters
  k2.len = N;
  k2.in = inSecond;
  k2.sum = &sumSecond;

  // Invoke kernel
  uint64_t cycle1 = pebblesCycleCount();
  #if EnableOverlapping
  noclRunOverlappingKernel(&k1, &k2);
  #else
  noclRunKernel(&k1);
  noclRunKernel(&k2);
  #endif
  uint64_t cycle2 = pebblesCycleCount() - cycle1;
  puts("Cycle count: "); puthex(cycle2 >> 32); puthex(cycle2); putchar('\n');

  // Check result
  bool ok = sumFirst == acc;

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
