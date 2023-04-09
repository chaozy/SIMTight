#include <NoCL.h>
#include <Rand.h>

// Kernel for adding vectors
struct VecAdd : Kernel {
  int len;
  int *a, *b, *result;

  void kernel() {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len)
      result[i] = a[i] + b[i];
  }
};

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

  #if EnableOverlapping
  puts("Overlapping should be turned off!\n");
  return 1;
  #endif

  // Vector size for benchmarking
  int N = isSim ? 3000 : 10000;
  int width = isSim ? 256 : 512;
  int height = isSim ? 64 : 512;

  // Input and output matrix data
  nocl_aligned int matInData[width*height];
  nocl_aligned int matOutData[width*height];

  // Friendly array wrappers
  Array2D<int> matIn(matInData, height, width);
  Array2D<int> matOut(matOutData, width, height);

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      matIn[i][j] = rand15(&seed);

  const int itersPerBlock = 4;

  // Instantiate kernel
  Transpose<SIMTLanes> k1;

  // Set block/grid dimensions
  k1.blockDim.x = SIMTLanes;
  k1.blockDim.y = SIMTLanes / itersPerBlock;
  k1.gridDim.x = width / k1.blockDim.x;
  k1.gridDim.y = height / (itersPerBlock * k1.blockDim.y);

  // Assign parameters
  k1.in = matIn;
  k1.out = matOut;


  // Input and output vectors
  simt_aligned int a[N], b[N], result[N];

  // Initialise inputs
  for (int i = 0; i < N; i++) {
    a[i] = rand15(&seed);
    b[i] = rand15(&seed);
  }

  // Instantiate kernel
  VecAdd k2;

  // Use a single block of threads
  k2.blockDim.x = (SIMTWarps * SIMTLanes) >> 1;
  k2.gridDim.x = 10;

  // Assign parameters
  k2.len = N;
  k2.a = a;
  k2.b = b;
  k2.result = result;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k1);
  noclRunKernelAndDumpStats(&k2);

  // Check result
  bool ok = true;
  for (int i = 0; i < N; i++)
    ok = ok && result[i] == a[i] + b[i];

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
