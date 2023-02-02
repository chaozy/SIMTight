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
  int N = isSim ? 3000 : 1000000;

  // Input and output vectors
  simt_aligned int a[N], b[N], result[N];

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < N; i++) {
    a[i] = rand15(&seed);
    b[i] = rand15(&seed);
  }

  // Instantiate VecAdd
  VecAdd k1;
  k1.blockDim.x = SIMTWarps * SIMTLanes;
  k1.len = N;
  k1.a = a;
  k1.b = b;
  k1.result = result;

  // Instantiate Reduce 
  simt_aligned int in[N];
  int sum;
  int acc = 0;
  for (int i = 0; i < N; i++) {
    int r = rand15(&seed);
    in[i] = r;
    acc += r;
  }
  Reduce<SIMTWarps * SIMTLanes> k2;
  k2.blockDim.x = SIMTWarps * SIMTLanes;

  // Assign parameters
  k2.len = N;
  k2.in = in;
  k2.sum = &sum;

  // Input and output vectors
  nocl_aligned unsigned char input[N];
  nocl_aligned int bins[256];
  for (int i = 0; i < N; i++)
    input[i] = rand15(&seed) & 0xff;
  Histogram k3;
  k3.blockDim.x = SIMTLanes * SIMTWarps;
  k3.len = N;
  k3.input = input;
  k3.bins = bins;

  Queue queue;
  queue.enqueue(&k1); queue.enqueue(&k2); queue.enqueue(&k3);
  // noclRunQueueAndDumpStats(&queue);
  noclRunKernelAndDumpStats(&k1);

  // Invoke kernel
  // noclRunKernelAndDumpStats(&k2);

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
