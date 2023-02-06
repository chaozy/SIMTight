#include <NoCL.h>
#include <Rand.h>
#include <FastZero.h>


int main()
{

  uint32_t N = 1024000;
    
	simt_aligned uint32_t a[N];
	puts("Start init\n");

	uint32_t seed = 3;
	for (int i = 0; i < N; i++) 
		a[i] = rand15(&seed);

	fastZero(&a, N * sizeof(uint32_t));

	int flag = 0;
	for (int i = 0; i < N; i++) 
		if ((flag = a[i])) 
			break;
	
	printf("Here: %x\n", flag);
	if (flag) puts("Fastzeroing is not fully zeroing its memory\n");
	else puts("Fastzeroing successfully zero the corresponding memory\n");

  return 0;
}
