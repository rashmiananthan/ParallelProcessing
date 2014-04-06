#include<omp.h>
#include<stdio.h>
#include<math.h>
#include <limits.h>

#define MAXCITIES 1296

int tsp(int thread_count, int samples, int o_samples, int cities, int length, float *posx, float *posy)
{
# pragma omp parallel num_threads(thread_count) default(none) shared(o_samples,samples,cities,length,posx,posy,thread_count)
{
unsigned int rndstate;
unsigned short tour[MAXCITIES + 1];
register unsigned short tmp;
register int iter, i,j,len, from, to, local_len;
register float dx, dy;
long my_rank;
  
  my_rank = omp_get_thread_num();

  local_len = INT_MAX;
  tour[cities] = 0;

  for (iter = o_samples+my_rank; iter <= samples; iter+=thread_count) {
 /* generate a random tour */
    rndstate = iter;
    for (i = 1; i < cities; i++) tour[i] = i;
    for (i = 1; i < cities; i++) {
      j = rand_r(&rndstate) % (cities - 1) + 1;
      tmp = tour[i];
      tour[i] = tour[j];
      tour[j] = tmp;
    }

 /* compute tour length */
    len = 0;
    from = 0;
    for (i = 1; i <= cities; i++) {
      to = tour[i];
      dx = posx[to] - posx[from];
      dy = posy[to] - posy[from];
      len += (int)(sqrtf(dx * dx + dy * dy) + 0.5f);
      from = to;
    }

  /* check if new shortest tour */
    if (local_len > len) {
      local_len = len;
    }
  }
# pragma omp critical
{
   if(length > local_len)
   {
      length = local_len;
   }
}
}
return length;
}
