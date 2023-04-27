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

// Kernel for sparse matrix vector multipliation on ELLPACK format
// One thread per matrix row
struct SparseMatVecMul : Kernel {
  int num_rows;
  int num_cols;
  int num_cols_per_row;
  int* indices;
  int* data;
  int* x;
  int* y;

  void kernel() {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
      int dot = 0;
      for (int n = 0; n < num_cols_per_row; n++) {
        int col = indices[num_rows * n + row];
        int val = data[num_rows * n + row];
        dot += val * x[col];
      }
      y[row] = dot;
    }
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


  // Vector and matrix dimensions for benchmarking
  // Should be powers of two
  int width = isSim ? 256 : 2048;
  int height = isSim ? 64 : 2048;

  // Sparsity of matrix (power of two)
  int sparsity = 8;
  int samplesPerRow = width / sparsity;

  // Input and outputs
  simt_aligned int data[samplesPerRow * height],
                   indices[samplesPerRow * height],
                   dataT[samplesPerRow * height],
                   indicesT[samplesPerRow * height],
                   vecIn[width*2], vecOut[height], vecOutSecond[height];

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < width; i++)
    vecIn[i] = rand15(&seed) & 0xff;
  for (int r = 0; r < height; r++) {
    vecOut[r] = 0;
    vecOutSecond[r] = 0;
    int offset = rand15(&seed) & (2*sparsity - 1);
    int n = 0;
    while (n < samplesPerRow) {
      data[r*samplesPerRow + n] = rand15(&seed) & 0xff;
      indices[r*samplesPerRow + n] = offset;
      n++;
      offset += rand15(&seed) & (2*sparsity-1);
      if (offset >= width) break;
    }
    while (n < samplesPerRow) {
      data[r*samplesPerRow + n] = 0;
      indices[r*samplesPerRow + n] = 0;
      n++;
    }
  }

  // Get matrix in column-major order
  for (int r = 0; r < height; r++)
    for (int n = 0; n < samplesPerRow; n++) {
      dataT[n * height + r] = data[r * samplesPerRow + n];
      indicesT[n * height + r] = indices[r * samplesPerRow + n];
    }

  // Instantiate kernel
  SparseMatVecMul k1;

  // One thread per row
  int groups = height / SIMTLanes;
  k1.blockDim.x = SIMTLanes;
  k1.gridDim.x = groups < SIMTWarps ? SIMTWarps : groups;
  // Increase the blockDim to reduce gridDim
  k1.blockDim.x = 256;
  k1.gridDim.x = height / 256;

  // Assign parameters
  k1.num_rows = height;
  k1.num_cols = width;
  k1.num_cols_per_row = samplesPerRow;
  k1.indices = indicesT;
  k1.data = dataT;
  k1.x = vecIn;
  k1.y = vecOut;

  //   // Instantiate kernel
  // SparseMatVecMul k2;

  // // One thread per row
  // k2.blockDim.x = SIMTLanes;
  // k2.gridDim.x = groups < SIMTWarps ? SIMTWarps : groups;

  // // Assign parameters
  // k2.num_rows = height;
  // k2.num_cols = width;
  // k2.num_cols_per_row = samplesPerRow;
  // k2.indices = indicesT;
  // k2.data = dataT;
  // k2.x = vecIn;
  // k2.y = vecOutSecond;



  int wid = isSim ? 256 : 512;
  int hei = isSim ? 64 : 512;
  // Input and output matrix data
  nocl_aligned int matInData[wid*hei];
  nocl_aligned int matOutData[wid*hei];

  // Friendly array wrappers
  Array2D<int> matIn(matInData, hei, wid);
  Array2D<int> matOut(matOutData, wid, hei);

  // Initialise inputs
  for (int i = 0; i < hei; i++)
    for (int j = 0; j < wid; j++)
      matIn[i][j] = rand15(&seed);

  // Number of loop iterations per block.  The number of iterations
  // times the block Y dimension must equal the block X dimension.
  const int itersPerBlock = 4;

  // Instantiate Tranpose
  Transpose<SIMTLanes> k2;

  // Set block/grid dimensions for the second kernel 
  k2.blockDim.x = SIMTLanes;
  k2.blockDim.y = SIMTLanes / itersPerBlock;
  k2.gridDim.x = wid / k2.blockDim.x;
  k2.gridDim.y = hei / (itersPerBlock * k2.blockDim.y);

  // Assign parameters
  k2.in = matIn;
  k2.out = matOut;

  // Invoke kernel
  noclRunOverlappingKernelAndDumpStats(&k1, &k2);

  // Check result
  bool ok_first = true, ok_second = true;
  for (int r = 0; r < height; r++) {
    int sum = 0;
    for (int n = 0; n < samplesPerRow; n++) {
      int i = r*samplesPerRow + n;
      if (data[i] != 0) sum += data[i] * vecIn[indices[i]];
    }
    ok_first = ok_first && sum == vecOut[r];
    // ok_second = ok_second && sum == vecOutSecond[r];
  }
    
  for (int i = 0; i < wid; i++)
    for (int j = 0; j < hei; j++)
    {
      ok_second = ok_second && matOut[i][j] == matIn[j][i];
      // if (matOut[i][j] != matIn[j][i])
      // {
      //   // printf("matOut: %x, matIn: %x\n", matOut[i][j], matIn[j][i]);
      //   // printf("matOut: %x, matIn: %x\n", i, j);
      // }
    }
  // Display result
  puts("Self test: ");
  puts(ok_first && ok_second? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
