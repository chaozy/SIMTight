#include <NoCL.h>
#include <Rand.h>

// Kernel for adding vectors
struct VecAdd : Kernel {
  int len;
  int *a, *b, *result;
  void kernel() {
    for (int i = threadIdx.x; i < len; i += blockDim.x)
      result[i] = a[i] + b[i];
  }
};

// Kernel for vector summation
template <int BlockSize> struct Reduce : Kernel {
  int len;
  int *in, *sum;
  void kernel() {
    int* block = shared.array<int, BlockSize>();
    block[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
      block[threadIdx.x] += in[i];
    __syncthreads();
    for(int i = blockDim.x >> 1; i > 0; i >>= 1)  {
      if (threadIdx.x < i)
        block[threadIdx.x] += block[threadIdx.x + i];
      __syncthreads();
    }
    if (threadIdx.x == 0) *sum = block[0];
  }
};

// Kernel for computing 256-bin histograms
struct Histogram : Kernel {
  int len;
  unsigned char* input;
  int* bins;
  void kernel() {
    int* histo = shared.array<int, 256>();
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
      histo[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < len; i += blockDim.x)
      atomicAdd(&histo[input[i]], 1);
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
      bins[i] = histo[i];
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

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < N; i++) {
    a[i] = rand15(&seed);
    b[i] = rand15(&seed);
  }

  // Instantiate VecAdd
  VecAdd kernel1;
  kernel1.blockDim.x = SIMTWarps * SIMTLanes;
  kernel1.len = N;
  kernel1.a = a;
  kernel1.b = b;
  kernel1.result = result;

  // Instantiate Reduce 
  simt_aligned int in[N];
  int sum;
  int acc = 0;
  for (int i = 0; i < N; i++) {
    int r = rand15(&seed);
    in[i] = r;
    acc += r;
  }
  Reduce<SIMTWarps * SIMTLanes> kernel2;
  kernel2.blockDim.x = SIMTWarps * SIMTLanes;
  kernel2.len = N;
  kernel2.in = in;
  kernel2.sum = &sum;

  // Instantiate Histogram
  nocl_aligned unsigned char input[N];
  nocl_aligned int bins[256];
  for (int i = 0; i < N; i++)
    input[i] = rand15(&seed) & 0xff;
  Histogram kernel3;
  kernel3.blockDim.x = SIMTLanes * SIMTWarps;
  kernel3.len = N;
  kernel3.input = input;
  kernel3.bins = bins;

  // Map hardware threads to CUDA thread
  noclMapKernel(&kernel1); 
  noclMapKernel(&kernel2); 
  noclMapKernel(&kernel3);
  
  // Init the nodes and the queue 
  QueueNode<Kernel> node1(&kernel1);
  QueueNode<Kernel> node2(&kernel2);
  QueueNode<Kernel> node3(&kernel3);
  QueueNode<Kernel> *nodes[] = {&node1, &node2, &node3};
  KernelQueue<Kernel> queue(nodes, 3);

  noclRunQueue(&queue);


  // Check VecAdd result
  bool ok_k1 = true;
  for (int i = 0; i < N; i++)
    ok_k1 = ok_k1 && result[i] == a[i] + b[i];

  // Check Reduce result
  bool ok_k2 = sum == acc;

  // Check Histogram result
  bool ok_k3 = true;
  int goldenBins[256];
  for (int i = 0; i < 256; i++) goldenBins[i] = 0;
  for (int i = 0; i < N; i++) goldenBins[input[i]]++;
  for (int i = 0; i < 256; i++)
    ok_k3 = ok_k3 && bins[i] == goldenBins[i];
  // puts("ARE U SURE?\n");
  // Display result
  printf("Results: %x, %x, %x\n", ok_k1, ok_k2, ok_k3);
  puts("Self test: ");
  puts(ok_k1 & ok_k2 & ok_k3? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
