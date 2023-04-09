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

  // Input and output matrix data
  nocl_aligned int matInData[width*height];
  nocl_aligned int matOutData[width*height];

  // Friendly array wrappers
  Array2D<int> matInFirst(matInData, height, width);
  Array2D<int> matInSecond(matInData, height, width);
  Array2D<int> matOutFirst(matOutData, width, height);
  Array2D<int> matOutSecond(matOutData, width, height);

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
    {
      matInFirst[i][j] = rand15(&seed);
      matInSecond[i][j] = rand15(&seed);
    }
  // Number of loop iterations per block.  The number of iterations
  // times the block Y dimension must equal the block X dimension.
  const int itersPerBlock = 4;

  // Instantiate kernel
  Transpose<SIMTLanes> k1;
  Transpose<SIMTLanes> k2;

  // Set block/grid dimensions for the first kernel 
  k1.blockDim.x = SIMTLanes;
  k1.blockDim.y = SIMTLanes / itersPerBlock;
  k1.gridDim.x = width / k1.blockDim.x;
  k1.gridDim.y = height / (itersPerBlock * k1.blockDim.y);

  // Assign parameters
  k1.in = matInFirst;
  k1.out = matOutFirst;

  // Set block/grid dimensions for the second kernel 
  k2.blockDim.x = SIMTLanes;
  k2.blockDim.y = SIMTLanes / itersPerBlock;
  k2.gridDim.x = width / k2.blockDim.x;
  k2.gridDim.y = height / (itersPerBlock * k2.blockDim.y);

  // Assign parameters
  k2.in = matInSecond;
  k2.out = matOutSecond;

  // Invoke kernel
  noclRunOverlappingKernelAndDumpStats(&k1, &k2);

  // Check result
  bool okFirst = true, okSecond = true;
  int cnt = 0;
  // for (int i = 0; i < width; i++)
  //   for (int j = 0; j < height; j++)
  //   {
  //     if (matOutSecond[i][j] != matInSecond[j][i])
  //     printf("first: %x, firstout: %x\n", matInSecond[j][i],matOutSecond[i][j]);
  //   }

  for (int i = 0; i < width; i++)
    for (int j = 0; j < height; j++)
    {
      okFirst = okFirst && matOutFirst[i][j] == matInFirst[j][i];
      okSecond = okSecond && matOutSecond[i][j] == matInSecond[j][i];
    }

  // Display result
  puts("Self test: ");
  puts(okFirst && okSecond ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
