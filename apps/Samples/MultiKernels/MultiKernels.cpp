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

// Kernel for matrix transposition
// One sub-square at a time
template <int SquareSize> struct Transpose : Kernel {
  Array2D<int> in, out;
  
  void kernel() {
    auto square = shared.array<int, SquareSize, SquareSize+1>();
    
    // Origin of square within matrix
    int originX = blockIdx.x * blockDim.x;
    int originY = blockIdx.y * blockDim.x;
    
    // Load square
    for (int y = threadIdx.y; y < blockDim.x; y += blockDim.y)
      square[y][threadIdx.x] = in[originY + y][originX + threadIdx.x];
    
    __syncthreads();
    
    // Store square
    for (int y = threadIdx.y; y < blockDim.x; y += blockDim.y)
      out[originX + y][originY + threadIdx.x] = square[threadIdx.x][y];
  }
};


int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Vector size for benchmarking
  int N = isSim ? 30 : 10000;

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

  // Matrix size for benchmarking
  int width = isSim ? 256 : 512;
  int height = isSim ? 64 : 512;

  // Input and output matrix data
  nocl_aligned int matInData[width*height];
  nocl_aligned int matOutData[width*height];

  // Friendly array wrappers
  Array2D<int> matIn(matInData, height, width);
  Array2D<int> matOut(matOutData, width, height);

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      matIn[i][j] = rand15(&seed);

  // Number of loop iterations per block.  The number of iterations
  // times the block Y dimension must equal the block X dimension.
  const int itersPerBlock = 4;

  // Instantiate kernel
  Transpose<SIMTLanes> kernel4;

  // Set block/grid dimensions
  kernel4.blockDim.x = SIMTLanes;
  kernel4.blockDim.y = SIMTLanes / itersPerBlock;
  kernel4.gridDim.x = width / kernel4.blockDim.x;
  kernel4.gridDim.y = height / (itersPerBlock * kernel4.blockDim.y);

  // Assign parameters
  kernel4.in = matIn;
  kernel4.out = matOut;

  uint64_t cycleCount1 = pebblesCycleCount(); 
  #if UseKernelQueue
  // Map hardware threads to CUDA thread
  noclMapKernel(&kernel1); 
  noclMapKernel(&kernel2); 
  noclMapKernel(&kernel3);
  noclMapKernel(&kernel4);

  // Init the nodes and the queue 
  QueueNode<Kernel> node1(&kernel1);
  QueueNode<Kernel> node2(&kernel2);
  QueueNode<Kernel> node3(&kernel3);
  QueueNode<Kernel> node4(&kernel4);
  QueueNode<Kernel> *nodes[] = {&node1, &node4, &node3, &node2};
  KernelQueue<Kernel> queue(nodes, 4);
  noclRunQueue(queue);

  queue.print_cycle_wait();
  #else  
  noclRunKernel(&kernel1);
  noclRunKernel(&kernel4);
  noclRunKernel(&kernel3);
  noclRunKernel(&kernel2);
  
  #endif 
  uint64_t cycles = pebblesCycleCount() - cycleCount1;
  puts("Cycle count: "); puthex(cycles >> 32); puthex(cycles); putchar('\n');

  
  

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
  
  // Check result
  bool ok_k4 = true;
  for (int i = 0; i < width; i++)
    for (int j = 0; j < height; j++)
      ok_k4 = ok_k4 && matOut[i][j] == matIn[j][i];

  // Display result
  // printf("Results: %x, %x, %x, %x\n", ok_k1, ok_k2, ok_k3, ok_k4);
  puts("Self test: ");
  puts(ok_k1 & ok_k2 & ok_k3 & ok_k4? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
