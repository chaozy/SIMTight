#include <NoCL.h>
#include <Rand.h>
#include <FastZero.h>
#include <Pebbles/CSRs/CycleCount.h>

struct VecZero : Kernel {
  int size;
  char *b;

  void kernel() {
    for (int i = threadIdx.x; i < size; i += blockDim.x)
      b[i] = 0;
  }
};

void zeroByLoop(char *mem, uint32_t size) 
{
  for (int i = 0; i < size; i++) 
  {
    mem[i] = 0;
  }
}

int main()
{

  uint32_t N = 64 * 1;
  simt_aligned char a[N], b[N], c[N];
  puts("Start init\n");
  // Initialise inputs
  for (int i = 0; i < N; i++) {
    a[i] = 'a';
    b[i] = 'b';
    c[i] = 'c';
  }

  uint32_t cycleCount = pebblesCycleCountL(); // zero the current cycleCount
  puts("zero by loop - cycle count: "); puthex(cycleCount); putchar('\n');
  zeroByLoop(c, N);  
  cycleCount = pebblesCycleCountL();
  puts("zero by loop - cycle count: "); puthex(cycleCount); putchar('\n');
  printStat("Total DRAM Access: ", STAT_SIMT_TOTAL_DRAM_ACCESSES);

  pebblesCycleCountL();
  fastZero(a, N);
  cycleCount = pebblesCycleCountL();
  puts("fastZero - cycle count: "); puthex(cycleCount); putchar('\n');
  printStat("Total DRAM Access: ", STAT_SIMT_TOTAL_DRAM_ACCESSES);


  VecZero k;
  k.blockDim.x = SIMTWarps * SIMTLanes;
  k.size = N;
  k.b = b;

  pebblesCycleCountL();
  noclRunKernel(&k);
  cycleCount = pebblesCycleCountL();
  puts("zero by kernel - cycle count: "); puthex(cycleCount); putchar('\n');
  printStat("Total DRAM Access: ", STAT_SIMT_TOTAL_DRAM_ACCESSES);


  bool flag = 0;
  for (int i = 0; i < N; i++) 
  {
    if (a[i] || b[i] || c[i]) 
    {
      flag = 1;
      break;
    }
  }
  if (flag) puts("A method is not fully zeroing its memory\n");
  else puts("All methods successfully zero the corresponding memory\n");
  

  puts("FINISHED\n");
  return 0;
}
