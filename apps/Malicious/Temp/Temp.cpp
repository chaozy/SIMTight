#include <NoCL.h>
#include <Rand.h>

// Malicious kernels that ignore bound checking
struct MaliciousAccess : Kernel {
  int *a, *b, *result;

  void kernel() {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    result[i] = a[i] + b[i];
  }
};

// Kernel with bound checking
struct VecAdd : Kernel 
{
  int len;
  int *a, *b, *result;

  void kernel() {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len){
      result[i] = a[i] + b[i];
    }
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Vector size for benchmarking
  int N = isSim ? 3000 : 10000;

  // Input and output vectors
  simt_aligned int a[N], b[N], result[N];
  simt_aligned int a_ma[N], b_ma[N], result_ma[N];

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < N; i++) {
    a[i] = rand15(&seed);
    b[i] = rand15(&seed);
  }

  // Instantiate kernel
  VecAdd k;

  // Use a single block of threads
  k.blockDim.x = SIMTWarps * SIMTLanes;
  // 1024 * 10 is enough to cover the data
  k.gridDim.x = 10;
  // Assign parameters
  k.len = N;
  k.a = a;
  k.b = b;
  k.result = result;

  MaliciousAccess ma;
  ma.a = a_ma;
  ma.b = b_ma;
  ma.result = result_ma;

  // Invoke kernel
  uint64_t cycleCount1 = pebblesCycleCount(); 
  #if UseKernelQueue
  noclMapKernel(&k); 
  QueueNode<Kernel> node(&k);
  QueueNode<Kernel> *nodes[] = {&node};
  KernelQueue<Kernel> queue(nodes, 1);
  noclRunQueue(queue);
  #else
  // noclRunKernelAndDumpStats(&k);
  noclRunKernel(&k);
  #endif
  uint64_t cycleCount2 = pebblesCycleCount(); 
  uint64_t cycles = cycleCount2 - cycleCount1;
  puts("Cycles: "); puthex(cycles >> 32);  puthex(cycles); putchar('\n');
  
  // Check result
  bool ok = true;
  bool ok_ma = true;
  for (int i = 0; i < N; i++)
  {
    ok = ok && result[i] == a[i] + b[i];
    ok_ma = ok_ma && result_ma[i] == a[i] + b[i];
  }

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  
  putchar('\n');
  puts(ok_ma? "Malicious kernel passed" : "Malicious Kernel failed");
  putchar('\n');

  return 0;
}
