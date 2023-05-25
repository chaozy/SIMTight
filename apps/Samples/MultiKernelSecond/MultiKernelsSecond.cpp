#include <NoCL.h>
#include <Rand.h>

// Kernel for adding vectors
struct VecAdd : Kernel {
  int len;
  int *a, *b, *result;
  void kernel() {
    // for (int i = threadIdx.x; i < len; i += blockDim.x)
    //   result[i] = a[i] + b[i];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) result[i] = a[i] + b[i];
  }
};

template <int BlockSize> struct MatMul : Kernel {
  int *A, *B, *C;
  int wA, wB;

  void kernel() {
    auto As = shared.array<int, BlockSize, BlockSize>();
    auto Bs = shared.array<int, BlockSize, BlockSize>();

    int bx = blockIdx.x;
    int by = blockIdx.y;
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

template <int BlockSize> struct Scan : Kernel {
  int len;
  int *in, *out;

  void kernel() {
    // Shared arrays
    int* tempIn = shared.array<int, BlockSize>();
    int* tempOut = shared.array<int, BlockSize>();

    // Shorthand for local thread id
    int t = threadIdx.x;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x < len) {

    // for (int x = 0; x < len; x += blockDim.x) {
      // Load data
      tempOut[t] = in[x+t];
      __syncthreads();

      // Local scan
      for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        swap(tempIn, tempOut);
        if (t >= offset)
          tempOut[t] = tempIn[t] + tempIn[t - offset];
        else
          tempOut[t] = tempIn[t];
        __syncthreads();
      }

      // Store data
      int acc = x > 0 ? out[x-1] : 0;
      out[x+t] = tempOut[t] + acc;
    }
  }
};


int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Vector size for benchmarking
  int N = isSim ? 30 : 1000;

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
  kernel1.gridDim.x = 1;
  kernel1.len = N;
  kernel1.a = a;
  kernel1.b = b;
  kernel1.result = result;

  // Init matmul
  int size = isSim ? 64 : 256;
  // Input and outputs
  simt_aligned int matA[size*size], matB[size*size],
                   matC[size*size], matCheck[size*size];

  // Initialise matrices
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++) {
      matA[i*size+j] = rand15(&seed) & 0xff;
      matB[i*size+j] = rand15(&seed) & 0xff;
      matCheck[i*size+j] = 0;
    }

  // Instantiate kernel
  MatMul<SIMTLanes> kernel3;

  // One block of threads per matrix tile
  kernel3.blockDim.x = SIMTLanes;
  kernel3.blockDim.y = SIMTLanes;
  kernel3.gridDim.x = size / SIMTLanes;
  kernel3.gridDim.y = size / SIMTLanes;

  // Assign parameters
  kernel3.wA = size;
  kernel3.wB = size;
  kernel3.A = matA;
  kernel3.B = matB;
  kernel3.C = matC;
  
  int L = isSim ? 4096 : 1024000;

  // Input and output vectors
  simt_aligned int in[L], out[L];

  // Initialise inputs
  for (int i = 0; i < L; i++) {
    in[i] = rand15(&seed);
  }

  // Instantiate kernel
  Scan<SIMTWarps * SIMTLanes> kernel4;

  // Use a single block of threads
  kernel4.blockDim.x = SIMTWarps * SIMTLanes;

  // Assign parameters
  kernel4.len = L;
  kernel4.in = in;
  kernel4.out = out;

  uint64_t cycleCount1 = pebblesCycleCount(); 
  #if UseKernelQueue
  // Map hardware threads to CUDA thread
  noclMapKernel(&kernel4); 
  noclMapKernel(&kernel3);
  noclMapKernel(&kernel1); 

  // Init the nodes and the queue 
  QueueNode<Kernel> node1(&kernel1);
  QueueNode<Kernel> node3(&kernel3);
  QueueNode<Kernel> node4(&kernel4);
  QueueNode<Kernel> *nodes[] = {&node1, &node3, &node4};
  KernelQueue<Kernel> queue(nodes, 3);
  noclRunQueue(queue);

  queue.print_cycle_wait();
  #else  
  noclRunKernel(&kernel1);
  noclRunKernel(&kernel3);
  noclRunKernel(&kernel4);
  
  #endif 
  uint64_t cycles = pebblesCycleCount() - cycleCount1;
  puts("Cycle count: "); puthex(cycles >> 32); puthex(cycles); putchar('\n');


  // Check VecAdd result
  bool ok_k1 = true;
  for (int i = 0; i < N; i++){
    //printf("%x %x %x\n", result[i],a[i],b[i] );
    ok_k1 = ok_k1 && result[i] == a[i] + b[i];
  }

  // Check Matmul result
  bool ok_k3 = true;
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      for (int k = 0; k < size; k++)
        matCheck[i*size+j] += matA[i*size+k] * matB[k*size+j];
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      ok_k3 = ok_k3 && matCheck[i*size+j] == matC[i*size+j];

  
  // Check result
  bool ok_k4 = true;
  int acc = 0;
  for (int i = 0; i < L; i++) {
    acc += in[i];
    ok_k4 = ok_k4 && out[i] == acc;
  }

  // Display result
  // printf("Results: %x, %x, %x, %x\n", ok_k1, ok_k2, ok_k3, ok_k4);
  puts("Self test: ");
  puts((ok_k1 & ok_k3 & ok_k4)? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
