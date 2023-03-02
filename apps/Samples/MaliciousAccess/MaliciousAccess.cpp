#include <NoCL.h>
#include <Rand.h>

// Kernel for vector summation
template <int BlockSize> struct Reduce : Kernel {
  int len;
  int *in, *sum, *sec;
  
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
    if (threadIdx.x == 1000) *sec = block[1000];
  }
};

template <int BlockSize> struct MaliciousAccess : Kernel {
	int *res;
	
	void kernel()
	{
		int* block = shared.array<int, BlockSize>();
		// int size = sizeof(int) * BlockSize;
		// block -= size;
    //block[threadIdx.x] = 1;
    // __syncthreads();
    // if (threadIdx.x == 2000) block[0] = 1;
    
		if (threadIdx.x == 1000) *res = block[1000];
    // if (threadIdx.x == 1) block[0] = 1;
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
	MaliciousAccess<SIMTWarps> ma;
	int res;
  int sec;
	ma.res = &res;
  ma.blockDim.x = SIMTWarps * SIMTLanes;
  ma.gridDim.x = 3;

  k.blockDim.x = SIMTLanes * SIMTLanes;
  k.gridDim.x = 4;

  // Assign parameters
  k.len = N;
  k.in = in;
  k.sum = &sum;
  k.sec = &sec;

  // Invoke kernel
  uint64_t cycleCount1 = pebblesCycleCount(); 
  #if UseKernelQueue
  noclMapKernel(&k);
  noclMapKernel(&ma); 
  QueueNode<Kernel> node1(&k);
  QueueNode<Kernel> node2(&ma);
  QueueNode<Kernel> *nodes[] = {&node1, &node2};
  KernelQueue<Kernel> queue(nodes, 2);
  noclRunQueue(queue);
  queue.print_cycle_wait();
  #else
  noclRunKernel(&k);
	noclRunKernel(&ma);
  #endif
  // uint64_t cycleCount2 = pebblesCycleCount(); 
  // uint64_t cycles = cycleCount2 - cycleCount1;
  // puts("Cycles: "); puthex(cycles >> 32);  puthex(cycles); putchar('\n');

  // Check result
  bool ok = sum == acc;
	ok = (sec == res) & ok;

  printf("Sum from Reduce: %x, from MalicousAccess: %x\n", sec, res);

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
