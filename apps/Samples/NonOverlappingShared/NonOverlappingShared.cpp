#include <NoCL.h>
#include <Rand.h>

// Matrix multiplication C = A * B
// (wA is A's width and wB is B's width)
template <int BlockSize> struct MatMul : Kernel {
  int *A, *B, *C;
  int wA, wB;

  void kernel() {
    auto As = shared.array<int, BlockSize, BlockSize>();
    auto Bs = shared.array<int, BlockSize, BlockSize>();

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BlockSize * by;

    int aEnd   = aBegin + wA - 1;

    int aStep  = BlockSize;

    int bBegin = BlockSize * bx;
    int bStep  = BlockSize * wB;

    int Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep) {


      As[ty][tx] = A[a + wA * ty + tx];
      Bs[ty][tx] = B[b + wB * ty + tx];

      __syncthreads();

      for (int k = 0; k < BlockSize; ++k) {
        Csub += As[ty][k] * Bs[k][tx];
      }
      __syncthreads();
    }

    int c = wB * BlockSize * by + BlockSize * bx;
    C[c + wB * ty + tx] = Csub;
  }
};

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

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  #if !EnableOverlapping
  puts("Overlapping is not enabled!\n");
  return 1;
  #endif

  int size = isSim ? 64 : 256;

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

    // Input and outputs
  simt_aligned int matA[size*size], matB[size*size],
                   matC[size*size], matCheck[size*size];

  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++) {
      matA[i*size+j] = rand15(&seed) & 0xff;
      matB[i*size+j] = rand15(&seed) & 0xff;
      matCheck[i*size+j] = 0;
    }

  // Instantiate kernel
  MatMul<SIMTLanes> k2;

  // One block of threads per matrix tile
  k2.blockDim.x = SIMTLanes;
  k2.blockDim.y = SIMTLanes;
  k2.gridDim.x = size / SIMTLanes;
  k2.gridDim.y = size / SIMTLanes;

  // Assign parameters
  k2.wA = size;
  k2.wB = size;
  k2.A = matA;
  k2.B = matB;
  k2.C = matC;

  // Invoke kernel
  noclRunOverlappingKernelAndDumpStats(&k1, &k2);

  // Check result
  bool okFirst = true, okFecond = true;

  for (int i = 0; i < N; i++)
    okFirst = okFirst && resultFirst[i] == a[i] + b[i];

  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      for (int k = 0; k < size; k++)
        matCheck[i*size+j] += matA[i*size+k] * matB[k*size+j];
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      okFecond = okFecond && matCheck[i*size+j] == matC[i*size+j];

  // Display result
  puts("Self test: ");
  puts(okFirst && okFecond? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
