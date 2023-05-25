#include <NoCL.h>
#include <Rand.h>

INLINE int readReg()
{
  int x;
  asm volatile("mv %0, x4" : "=r"(x));
  return x;
}

INLINE void writeReg()
{
  asm volatile("mv x4, 0xdeadbeaf");
}

// Kernel for vector summation
template <int BlockSize> struct Reduce : Kernel {
  int len;
  int *in, *sum, *k_stack, *reg;
  
  void kernel() 
  {
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

    // Set a stack address to k_stack
    int local = 0xdeadbeaf;
    k_stack = &local;

    // if (threadIdx.x == 64)  *reg = readReg();
	}

};

template <int BlockSize> struct NonZeroAccess : Kernel {
	int *res, *na_stack, *reg, *na_stack_val;
	
	void kernel()
	{
    // The top pointer is reset after every block is finished,  
    // hence this address is the same as the one in previous kernel
		int* block = shared.array<int, BlockSize>();

    // Read the shared memory
		if (threadIdx.x == 0) *res = block[0];
    

    // Read the val stored in the stack of the other kernel
    *na_stack_val = *na_stack;

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

  uint32_t seed = 1;
  int acc = 0;
  for (int i = 0; i < N; i++) {
    int r = rand15(&seed);
    in[i] = r;
    acc += r;
  }

  // Instantiate kernels
  Reduce<SIMTWarps * SIMTLanes> k;
	NonZeroAccess<SIMTWarps> na;

  int sum, k_reg;
  int* k_stack;
  k.len = N;
  k.in = in;
  k.sum = &sum;
  k.reg = &k_reg;
  k.k_stack = k_stack;
  k.blockDim.x = SIMTWarps * SIMTLanes;
  k.gridDim.x = 4;

  int res, na_reg, na_stack_val;
  int *na_stack;
	na.res = &res;
  na.reg = &na_reg;
  na.na_stack = k_stack;
  na.na_stack_val = &na_stack_val;
  na.blockDim.x = SIMTWarps * SIMTLanes;
  na.gridDim.x = 3;

  // Invoke kernels
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
	noclRunKernel(&na);
  #endif

  // Check results
  bool ok = sum == acc;
	ok = (sum == res) & ok;
  ok = (na_stack_val == 0xdeadbeaf);

  printf("Sum from Reduce: %x, from NonZerodAccess: %x\n", sum, res);
  printf("Read leftover value in register from NonZeroedAccess: %x\n", na_reg);
  printf("Print out a local variable from the stack of Reduce: %x\n", na_stack_val);

  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
