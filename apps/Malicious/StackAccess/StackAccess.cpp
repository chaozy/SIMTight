#include <NoCL.h>
#include <Rand.h>

// Kernel for adding vectors
struct VecAdd : Kernel {
  int *k_stack, *random;
  void kernel() {
    // Set a stack address to k_stack
    int local = 0xdeadbeaf;
    k_stack = &local;
  }
};

struct StackAccess : Kernel {
	int *na_stack, *na_stack_val;
	
	void kernel()
	{

    // Read the val stored in the stack of the other kernel
    int local;
    *na_stack_val = *na_stack;
	}
};

int main()
{
  
  bool isSim = getchar();

  // Instantiate kernels
  VecAdd k;
	StackAccess na;

  int* k_stack;
  k.k_stack = k_stack;
  k.blockDim.x = SIMTWarps * SIMTLanes;
  k.gridDim.x = 4;

  int res, na_reg, na_stack_val;
  int *na_stack;
  na.na_stack = k_stack;
  na.na_stack_val = &na_stack_val;
  na.blockDim.x = SIMTWarps * SIMTLanes;
  na.gridDim.x = 1;

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

  // Check results;

  printf("Print out a local variable from the stack of VecAdd: %x\n", na_stack_val);
  putchar('\n');

  return 0;
}
