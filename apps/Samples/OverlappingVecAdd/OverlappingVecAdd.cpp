#include <NoCL.h>
#include <Rand.h>

// Kernel for adding vectors
struct VecAdd : Kernel {
  int len;
  int *a, *b, *result;

  void kernel() {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len)
      result[i] = a[i] + b[i];
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // #if !EnableOverlapping
  // puts("Overlapping is not enabled!\n");
  // return 1;
  // #endif

  // Vector size for benchmarking
  int N = isSim ? 3000 : 10000;

  // Input and output vectors
  simt_aligned int a[N], b[N], resultFirst[N], resultSecond[N];

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < N; i++) {
    a[i] = rand15(&seed);
    b[i] = rand15(&seed);
  }

  // Instantiate the first kernel 
  VecAdd k1;
  k1.blockDim.x = (SIMTWarps * SIMTLanes) >> 1;
  k1.gridDim.x = 10;
  k1.len = N;
  k1.a = a;
  k1.b = b;
  k1.result = resultFirst;

  // Instantiate the first kernel 
  VecAdd k2;
  k2.blockDim.x = (SIMTWarps * SIMTLanes) >> 1;
  k2.gridDim.x = 10;
  k2.len = N;
  k2.a = a;
  k2.b = b;
  k2.result = resultSecond;

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
  bool okFirst = true, okSecond = true;
  for (int i = 0; i < N; i++)
  {
    okFirst = okFirst && resultFirst[i] == a[i] + b[i];
    okSecond = okSecond && resultSecond[i] == a[i] + b[i];
  }

  // Display result
  puts("Self test: ");
  puts(okSecond && okFirst? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
