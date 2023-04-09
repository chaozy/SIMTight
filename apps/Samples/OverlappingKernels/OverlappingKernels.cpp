#include <NoCL.h>
#include <Rand.h>

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

// Kernel for adding vectors
struct VecAdd : Kernel {
  int len;
  int *a, *b, *result;

  void kernel() {
    // for (int i = threadIdx.x; i < len; i += blockDim.x)
    //   result[i] = a[i] + b[i];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len)
      result[i] = a[i] + b[i];
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  #if !EnableOverlapping
  puts("Overlapping is not enabled!\n");
  return 1;
  #endif

  // Matrix size for benchmarking
  int width = isSim ? 256 : 512;
  int height = isSim ? 64 : 512;

  // Init VecAdd Kernels
  int N = isSim ? 3000 : 10000;

  simt_aligned int a[N], b[N], resultFirst[N];

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < N; i++) {
    a[i] = rand15(&seed);
    b[i] = rand15(&seed);
  }

  // Instantiate the first kernel 
  VecAdd k1;
  k1.blockDim.x = (SIMTWarps * SIMTLanes) >> 1;
  k1.gridDim.x = 10;
  k1.len = N;
  k1.a = a;
  k1.b = b;
  k1.result = resultFirst;

  // Input and output matrix data
  nocl_aligned int matInData[width*height];
  nocl_aligned int matOutData[width*height];

  // Friendly array wrappers
  Array2D<int> matIn(matInData, height, width);
  Array2D<int> matOut(matOutData, width, height);

  // Initialise inputs
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      matIn[i][j] = rand15(&seed);

  // Number of loop iterations per block.  The number of iterations
  // times the block Y dimension must equal the block X dimension.
  const int itersPerBlock = 4;

  // Instantiate Tranpose
  Transpose<SIMTLanes> k2;

  // Set block/grid dimensions for the second kernel 
  k2.blockDim.x = SIMTLanes;
  k2.blockDim.y = SIMTLanes / itersPerBlock;
  k2.gridDim.x = width / k2.blockDim.x;
  k2.gridDim.y = height / (itersPerBlock * k2.blockDim.y);

  // Assign parameters
  k2.in = matIn;
  k2.out = matOut;

  // Invoke kernel
  noclRunOverlappingKernelAndDumpStats(&k1, &k2);

  // Check result
  bool ok_first = true, ok_second = true;
  for (int i = 0; i < N; i++)
    ok_first = ok_first && resultFirst[i] == a[i] + b[i];
    
  for (int i = 0; i < width; i++)
    for (int j = 0; j < height; j++)
    {
      ok_second = ok_second && matOut[i][j] == matIn[j][i];
      
    }
  // Display result
  puts("Self test: ");
  puts(ok_first && ok_second? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
