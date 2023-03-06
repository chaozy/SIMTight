#include <NoCL.h>
#include <Rand.h>

// Kernel for vector summation
struct Reduce : Kernel {
  int *in, *res;
  
  void kernel() {
    if (threadIdx.x == 0) in[0] = 0xdeadbeaf;
  }
};

struct MaliciousAccess : Kernel {
	int *in, *res;
	
	void kernel()
	{
		if (threadIdx.x == 0) *res = in[0];
	}
};

int main()
{
  
  bool isSim = getchar();
  int N = isSim ? 3000 : 1000000;

  // Input and outputs
  simt_aligned int in[N];

  // Initialise inputs
  uint32_t seed = 1;
  int acc = 0;
  for (int i = 0; i < N; i++) {
    int r = rand15(&seed);
    in[i] = r;
    acc += r;
  }

  int reduceRes;
  int maRes;
  Reduce k;
	MaliciousAccess ma;
	ma.in = in;
  ma.res = &maRes;
  ma.blockDim.x = SIMTWarps * SIMTLanes;
  ma.gridDim.x = 3;

  k.in = in;
  k.res = &reduceRes;
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
  bool ok = maRes == 0xdeadbeaf;

  // printf("Sum from Reduce: %x, from MalicousAccess: %x\n", sec, res);

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
