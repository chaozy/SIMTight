#include <NoCL.h>
#include <Rand.h>

INLINE int readReg()
{
  int x;
  asm volatile("csrrw %0, 0x1, zero" : "=r"(x));
  return x;
}

// Kernel for vector summation
template <int BlockSize> struct Reduce : Kernel {
  int len;
  int *in, *sum, *shared_in, *reg;
  
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

    // Change the stack memory
    if (threadIdx.x == 1000) shared_in[1000] = 0xdeadbeaf;

    if (threadIdx.x == 64)  *reg = readReg();
	}

};

template <int BlockSize> struct NonZeroAccess : Kernel {
	int *res, *read, *shared_in, *reg;
	
	void kernel()
	{
    // The top pointer is reset after every block is finished,  
    // hence this address is the same as the one in previous kernel
		int* block = shared.array<int, BlockSize>();

    // Read the shared memory
		if (threadIdx.x == 0) *res = block[0];

    // Read the stack memory
    if (threadIdx.x == 1000) *read = shared_in[1000];

    // Read the register
    if (threadIdx.x == 64)  *reg = readReg();
	}
};

int main()
{
  
  bool isSim = getchar();
  int N = isSim ? 3000 : 1000000;

  // Input and outputs
  simt_aligned int in[N];
  simt_aligned int shared_in[N];


  uint32_t seed = 1;
  int acc = 0;
  for (int i = 0; i < N; i++) {
    int r = rand15(&seed);
    in[i] = r;
    shared_in[i] = r;
    acc += r;
  }

  // Instantiate kernel
  Reduce<SIMTWarps * SIMTLanes> k;

  // Instantiate NonZeroAccess
	NonZeroAccess<SIMTWarps> na;
	int res, read, na_reg;
	na.res = &res;
  na.read = &read;
  na.reg = &na_reg;
  na.shared_in = shared_in;

  na.blockDim.x = SIMTWarps * SIMTLanes;
  na.gridDim.x = 3;

    int sum, k_reg;
  k.len = N;
  k.in = in;
  k.sum = &sum;
  k.reg = &k_reg;
  k.shared_in = shared_in;
  k.blockDim.x = SIMTWarps * SIMTLanes;
  k.gridDim.x = 4;

  // Invoke kernel
  uint64_t cycleCount1 = pebblesCycleCount(); 
  #if UseKernelQueue
  noclMapKernel(&k);
  noclMapKernel(&na); 
  QueueNode<Kernel> node1(&k);
  QueueNode<Kernel> node2(&na);
  QueueNode<Kernel> *nodes[] = {&node1, &node2};
  KernelQueue<Kernel> queue(nodes, 2);
  noclRunQueue(queue);
  #else
  noclRunKernel(&k);
	// noclRunKernel(&na);
  #endif
  // uint64_t cycleCount2 = pebblesCycleCount(); 
  // uint64_t cycles = cycleCount2 - cycleCount1;
  // puts("Cycles: "); puthex(cycles >> 32);  puthex(cycles); putchar('\n');

  // Check result
  bool ok = sum == acc;
	ok = (sum == res) & ok;
  ok = (read == 0xdeadbeaf);
  // ok = (reg == 3); 

  printf("Sum from Reduce: %x, from NonZerodAccess: %x\n", sum, res);
  puthex(k_reg); putchar('\n');
  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
