#include <NoCL.h>
#include <Rand.h>
#include <FastZero.h>
#include <Pebbles/CSRs/CycleCount.h>

struct VecZero : Kernel {
  int size;
  uint32_t *b;

  void kernel() {
    for (int i = threadIdx.x; i < size; i += blockDim.x)
      b[i] = 0;
  }
};

void zeroByLoop(uint32_t *mem, uint32_t size) 
{
  for (int i = 0; i < size; i++) 
    mem[i] = 0;
}

int main()
{

  uint32_t N = 16 * 100 * 100;

  // Repeat the simulation for 10 times with increasing data size
  for (uint32_t i = 0; i < 10; i++)
  {

    // puts("Processing with data size: "); putchar(N*4); puts(" * 100 byte\n");
    printf("Prcessing with data size: %d * 100 byte\n", N * 4);
    // printStat("Total DRAM Access: ", STAT_SIMT_TOTAL_DRAM_ACCESSES);
    

    simt_aligned uint32_t a[N], b[N], c[N];
    puts("Start init\n");
    // Initialise inputs
    uint32_t seed = 3;
    for (int i = 0; i < N; i++) {
      a[i] = rand15(&seed);
      b[i] = rand15(&seed);
      c[i] = rand15(&seed);
    }

    uint64_t cycleCount1 = pebblesCycleCount(); 
    zeroByLoop(c, N);  
    uint64_t cycleCount2 = pebblesCycleCount();
    uint64_t cycles = cycleCount2 - cycleCount1;
    puts("zero by loop - cycle count: "); puthex(cycles >> 32);  puthex(cycles); putchar('\n');
    // printStat("Total DRAM Access: ", STAT_SIMT_TOTAL_DRAM_ACCESSES);


    VecZero k;
    k.blockDim.x = SIMTWarps * SIMTLanes;
    k.size = N;
    k.b = b;

    cycleCount1 = pebblesCycleCount();
    noclRunKernel(&k);
    cycleCount2 = pebblesCycleCount();
    cycles = cycleCount2 - cycleCount1;
    puts("zero by kernel - cycle count: "); puthex(cycles >> 32); puthex(cycles); putchar('\n');
    // printStat("Total DRAM Access: ", STAT_SIMT_TOTAL_DRAM_ACCESSES);

    cycleCount1 = pebblesCycleCount();
    fastZero(&a, N * sizeof(uint32_t));
    cycleCount2 = pebblesCycleCount();
    cycles = cycleCount2 - cycleCount1;
    puts("fast zero - cycle count: "); puthex(cycles >> 32); puthex(cycles); putchar('\n');
    // printStat("Total DRAM Access: ", STAT_SIMT_TOTAL_DRAM_ACCESSES);


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

    puts("\n\n");

    N *= 2;
  }
    
  

  puts("FINISHED\n");
  return 0;
}
