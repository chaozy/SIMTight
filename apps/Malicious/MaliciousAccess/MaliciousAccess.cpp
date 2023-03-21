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

struct MaliciousAccess : Kernel {
  int *in, *res, *offset;
  int target;

	void kernel()
	{
    for (int i = 0; i < 10000; i++)
    {
      if ((in - sizeof(int) * i)[0] == target )
      {
        *offset = i;
        *res = (in - sizeof(int) * i)[0] ;
        break;
      }
    }
	}
};

int main()
{
  
  bool isSim = getchar();
  int N = isSim ? 3000 : 100;

  // Input and outputs
  simt_aligned int in[N];
  simt_aligned int in_ma[1];

  // Initialise inputs
  uint32_t seed = 1;
  int acc = 0;
  for (int i = 0; i < N; i++) {
    int r = rand15(&seed);
    in[i] = r;
    acc += r;
  }
  

  Reduce<SIMTWarps * SIMTLanes> k;
	MaliciousAccess ma;
  int res_ma, offset;
  ma.blockDim.x = SIMTWarps * SIMTLanes;
  ma.gridDim.x = 3;
  ma.in = in_ma;
  ma.res = &res_ma;
  ma.offset = &offset; 

  // Set the first element of the input array to a unique value
  acc = acc - in[0] + 0xdeadbeaf;
  in[0] = 0xdeadbeaf;
  ma.target = in[0];


  int sum;
  k.in = in;
  k.len = N;
  k.sum = &sum;
  k.blockDim.x = SIMTLanes * SIMTLanes;
  k.gridDim.x = 4;

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
  printf("in[0] from Reduce: %x, read from Malicious kernel: %x, offset is %x\n",
                            in[0], res_ma, offset);

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
