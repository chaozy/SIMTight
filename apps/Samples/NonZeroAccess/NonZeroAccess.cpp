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

template <int BlockSize> struct NonZeroAccess : Kernel {
	int *res;
	
	void kernel()
	{
		int* block = shared.array<int, BlockSize>();
		int size = sizeof(int) & BlockSize;
		block -= size;

		*res = block[0];
	}
};

int main()
{
  
  bool isSim = getchar();
  int N = isSim ? 3000 : 1000000;

  // Input and outputs
  simt_aligned int in[N];
  int sum;

  // Initialise inputs
  uint32_t seed = 1;
  int acc = 0;
  for (int i = 0; i < N; i++) {
    int r = rand15(&seed);
    in[i] = r;
    acc += r;
  }

  // Instantiate kernel
  Reduce<SIMTWarps * SIMTLanes> k;

  // Instantiate NonZeroAccess
	NonZeroAccess<SIMTWarps> ma;
	int res;
	ma.res = &res;
  ma.blockDim.x = SIMTWarps * SIMTLanes;

  // Use a single block of threads
  k.blockDim.x = SIMTWarps * SIMTLanes;
  // k.gridDim.x = 2;

  // Assign parameters
  k.len = N;
  k.in = in;
  k.sum = &sum;

  // Invoke kernel
  uint64_t cycleCount1 = pebblesCycleCount(); 
  #if UseKernelQueue
  noclMapKernel(&k);
  noclMapKernel(&ma); 
  QueueNode<Kernel> node1(&k);
  QueueNode<Kernel> node2(&ma);
  QueueNode<Kernel> *nodes[] = {&node, &ma};
  KernelQueue<Kernel> queue(nodes, 2);
  noclRunQueue(queue);
  #else
  noclRunKernel(&k);
	noclRunKernel(&ma);
  #endif
  // uint64_t cycleCount2 = pebblesCycleCount(); 
  // uint64_t cycles = cycleCount2 - cycleCount1;
  // puts("Cycles: "); puthex(cycles >> 32);  puthex(cycles); putchar('\n');

  // Check result
  bool ok = sum == acc;
	ok = (sum == res) & ok;

  printf("Sum from Reduce: %x, from MalicousAccess: %x\n", sum, res);
  puthex(res);

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
