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
      // square[y][threadIdx.x]  = 1;
    
    __syncthreads();
    if (pebblesHartId() == 1) pebblesSimEmit((unsigned int)square);
    // if (pebblesHartId() == 300) pebblesSimEmit((unsigned int)square);
    // Store square
    for (int y = threadIdx.y; y < blockDim.x; y += blockDim.y)
      out[originX + y][originY + threadIdx.x] = square[threadIdx.x][y];
      // out[originX + y][originY + threadIdx.x] = 1;
  }
};

template <int SquareSize> struct TransposeSecond : Kernel {
  Array2D<int> in, out;
  
  void kernel() {
    auto square = shared.array<int, SquareSize, SquareSize+1>();
    
    // Origin of square within matrix
    int originX = blockIdx.x * blockDim.x;
    int originY = blockIdx.y * blockDim.x;
    
    // Load square
    for (int y = threadIdx.y; y < blockDim.x; y += blockDim.y)
      square[y][threadIdx.x] = in[originY + y][originX + threadIdx.x];
      //square[y][threadIdx.x] = 1;
    
    __syncthreads();
    if (pebblesHartId() == 1025) pebblesSimEmit((unsigned int)square);
    // Store square
    for (int y = threadIdx.y; y < blockDim.x; y += blockDim.y)
      out[originX + y][originY + threadIdx.x] = square[threadIdx.x][y];
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // #if !EnableOverlapping
  // puts("Overlapping is not enabled!\n");
  // return 1;
  // #endif

  // Matrix size for benchmarking
  int width = isSim ? 256 : 512;
  int height = isSim ? 64 : 512;

  // Input and output matrix data
  nocl_aligned int matInDataFirst[width*height];
  nocl_aligned int matInDataSecond[width*height];
  nocl_aligned int matOutDataFirst[width*height];
  nocl_aligned int matOutDataSecond[width*height];

  // Friendly array wrappers
  Array2D<int> matInFirst(matInDataFirst, height, width);
  Array2D<int> matInSecond(matInDataSecond, height, width);
  Array2D<int> matOutFirst(matOutDataFirst, width, height);
  Array2D<int> matOutSecond(matOutDataSecond, width, height);

  // Initialise inputs
  uint32_t seed = 1;
  uint32_t seed_second = 6;
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
    {
      matInFirst[i][j] = rand15(&seed);
      matInSecond[i][j] = rand15(&seed_second);
      // printf("first: %x, second: %x\n", matInFirst[i][j], matInSecond[i][j]);
    }
  // Number of loop iterations per block.  The number of iterations
  // times the block Y dimension must equal the block X dimension.
  const int itersPerBlock = 4;

  // Instantiate kernel
  Transpose<SIMTLanes> k1;
  TransposeSecond<SIMTLanes> k2;

  // Set block/grid dimensions for the first kernel 
  k1.blockDim.x = SIMTLanes;
  k1.blockDim.y = SIMTLanes / itersPerBlock;
  k1.gridDim.x = width / k1.blockDim.x;
  k1.gridDim.y = height / (itersPerBlock * k1.blockDim.y);

  // Assign parameters
  k1.in = matInFirst;
  k1.out = matOutFirst;
  // printf("first: %x, second: %x\n", matInFirst[256][60], matInSecond[256][60]); 
  // for (int i = 0; i < 3; i++)
  //   for (int j = 0; j < 3; j++)
  //   {
  //     printf("first: %x, second: %x\n", matInFirst[i][j], matInSecond[i][j]);
  //   }
  
  // Set block/grid dimensions for the second kernel 
  k2.blockDim.x = SIMTLanes;
  k2.blockDim.y = SIMTLanes / itersPerBlock;
  k2.gridDim.x = width / k2.blockDim.x;
  k2.gridDim.y = height / (itersPerBlock * k2.blockDim.y);

  // Assign parameters
  k2.in = matInSecond;
  k2.out = matOutSecond;
  // for (int i = 0; i < 3; i++)
  //   for (int j = 0; j < 3; j++)
  //   {
  //     printf("first: %x, second: %x\n", matInFirst[i][j], matInSecond[i][j]);
  //   }
// printf("first: %x, second: %x\n", matInFirst[256][60], matInSecond[256][60]); 
//   for (int i = 0; i < height; i++)
//     for (int j = 0; j < width; j++)
//     {
//       // if (matOutSecond[i][j] != matInSecond[j][i])
//       printf("first: %x, firstout: %x\n", matInFirst[i][j],matInSecond[i][j]);
//     }
  // Invoke kernel
  uint64_t cycle1 = pebblesCycleCount();
  #if EnableOverlapping
  noclRunOverlappingKernel(&k1, &k2);
  #else
  noclRunKernel(&k1);
  noclRunKernel(&k2);
  #endif
  uint64_t cycle2 = pebblesCycleCount() - cycle1;
  puts("Cycle count: "); puthex(cycle2 >> 32); puthex(cycle2); putchar('\n');

  // Check result
  bool okFirst = true, okSecond = true;
  int cnt = 0;

  // for (int i = 0; i < 3; i++)
  //   for (int j = 0; j < 3; j++)
  //     printf("firstout: %x, secondout: %x\n", matOutFirst[i][j], matOutSecond[i][j]);


  for (int i = 0; i < width; i++)
    for (int j = 0; j < height; j++)
    {
      // printf("firstout: %x, secondout: %x\n", matOutFirst[i][j], matOutSecond[i][j]);
      // printf("firstin: %x, secondin: %x\n", matInFirst[j][i], matInSecond[j][i]);
      // printf("first: %x, second: %x\n", matInFirst[j][i], matOutFirst[i][j]);
      // printf("first: %x, second: %x\n", matInSecond[j][i], matOutSecond[i][j]);
      // if (matOutSecond[i][j] != matInSecond[j][i])
      //   printf("firstout: %x, secondout: %x\n", matOutSecond[i][j], matInSecond[j][i]);

      okFirst = okFirst && matOutFirst[i][j] == matInFirst[j][i];
      okSecond = okSecond && matOutSecond[i][j] == matInSecond[j][i];
    }

  // Display result
  puts("Self test: ");
  puts(okFirst && okSecond ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
