#include <NoCL.h>
#include <Rand.h>

// Kernel for adding vectors
struct VecAdd : Kernel {
  int len;
  int *a, *b, *result;

  void kernel() {
    // for (int i = threadIdx.x; i < len; i += blockDim.x)
    //   result[i] = a[i] + b[i];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len)
      result[i] = a[i] + b[i];
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  #if !EnableOverlapping
  puts("Overlapping is not enabled!\n");
  return 1;
  #endif

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
  noclRunOverlappingKernelAndDumpStats(&k1, &k2);

  // Check result
  bool ok_first = true;
  bool ok_second = true;
  for (int i = 0; i < N; i++)
  {
    ok_first = ok_first && resultFirst[i] == a[i] + b[i];
    ok_second = ok_second && resultFirst[i] == a[i] + b[i];
  }

  // Display result
  puts("Self test: ");
  puts(ok_first && ok_second ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
