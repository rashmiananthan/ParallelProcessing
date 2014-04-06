#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand_kernel.h>

#define THREADS 512
#define MAXCITIES 1296

extern "C" int tsp(int, int,int, int, int, float *, float *);

__global__ void TspKernel(int kCities, int kSamples, float *kPosx, float *kPosy, int *dlength)
{
  __shared__ int local_length;
  register int iter, i, j, len, from, to;
  register float dx, dy;
  register unsigned short tmp;
  unsigned short tour[MAXCITIES+1];
  curandState rndstate;
  iter = threadIdx.x + blockIdx.x * blockDim.x;
  tour[kCities] = 0;
  local_length = INT_MAX;
 
  if(iter==0)
  {
     *dlength = INT_MAX;
  }
  __syncthreads();

/* iterate number of sample times */
  if (iter < kSamples) {
  
/* generate a random tour */
    curand_init(iter, 0, 0, &rndstate);
    for (i = 1; i < kCities; i++) tour[i] = i;
    for (i = 1; i < kCities; i++) {
      j = curand(&rndstate) % (kCities - 1) + 1;
      tmp = tour[i];
      tour[i] = tour[j];
      tour[j] = tmp;
    }

 /* compute tour length */
    len = 0;
    from = 0;
    for (i = 1; i <= kCities; i++) {
      to = tour[i];
      dx = kPosx[to] - kPosx[from];
      dy = kPosy[to] - kPosy[from];
      len += (int)(sqrtf(dx * dx + dy * dy) + 0.5f);
      from = to;
    }

 /* check if new shortest tour */
     atomicMin(&local_length, len);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicMin(dlength, local_length);
 }
}

static int read_input(char *filename, float *posx, float *posy)
{
  register int cnt;
  int i1, cities;
  float i2, i3;
  register FILE *f;

 /* open input text file */
  f = fopen(filename, "r+t");
  if (f == NULL) {fprintf(stderr, "could not open file %s\n", filename); exit(-1);}

 /* read the number of cities from first line */
  cities = -1;
  fscanf(f, "%d\n", &cities);
  if ((cities < 1) || (cities >= MAXCITIES)) {fprintf(stderr, "cities out of range\n"); exit(-1);}

  /* read in the cities' coordinates */
  cnt = 0;
  while (fscanf(f, "%d %f %f\n", &i1, &i2, &i3)) {
            posx[cnt] = i2;
    posy[cnt] = i3;
    cnt++;
    if (cnt > cities) {fprintf(stderr, "input too long\n"); exit(-1);}
    if (cnt != i1) {fprintf(stderr, "input line mismatch\n"); exit(-1);}
  }
  if (cnt != cities) {fprintf(stderr, "wrong number of cities read\n"); exit(-1);}

  /* return the number of cities */
  fclose(f);
  return cities;
}

int main(int argc, char *argv[])
{
  register int blocks, samples, c_samples, o_samples, cities;
  float posx[MAXCITIES], posy[MAXCITIES], *dposx, *dposy;
  struct timeval start, end;
  int *dlength, length, o_length, final_length, thread_count;

  printf("TSP v1.0(CUDA)\n");

  /* check command line */
  if (argc != 4) {fprintf(stderr, "usage: %s input_file_name number_of_samples\n", argv[0]); exit(-1);}
  cities = read_input(argv[1], posx, posy);
  samples = atoi(argv[2]);
  if (samples < 1) {fprintf(stderr, "number of samples must be at least 1\n"); exit(-1);}
  printf("%d cities and %d samples (%s)\n", cities, samples, argv[1]);
  o_length = INT_MAX;
  
  thread_count = strtol(argv[3],NULL,10);
  c_samples = (int)ceil(samples/2);
  o_samples = (int)floor(samples/2);
  blocks = (c_samples + THREADS - 1) / THREADS;

  if (cudaSuccess != cudaMalloc((void **)&dlength, sizeof(int))) fprintf(stderr, "could not allocate array\n");
  if (cudaSuccess != cudaMalloc((void **)&dposx, (cities*sizeof(float)))) fprintf(stderr, "could not allocate array\n");
  if (cudaSuccess != cudaMalloc((void **)&dposy, (cities*sizeof(float)))) fprintf(stderr, "could not allocate array\n");

  /* start time */
  gettimeofday(&start, NULL);

  if (cudaSuccess != cudaMemcpy(dposx, posx, (cities*sizeof(float)), cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posx to device failed\n");
  if (cudaSuccess != cudaMemcpy(dposy, posy, (cities*sizeof(float)), cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posy to device failed\n");

  TspKernel<<<blocks, THREADS>>>(cities, c_samples, dposx, dposy, dlength);

  o_length = tsp(thread_count, samples, o_samples, cities, o_length, posx, posy);

 if (cudaSuccess != cudaMemcpy(&length, dlength, sizeof(1), cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of dlength from device failed\n");

 /* end time */
  gettimeofday(&end, NULL);
  printf("runtime: %.4lf s\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);

 /* output result */
 if(length < o_length) final_length = length;
 else final_length = o_length;
  
  printf("length of shortest found tour: %d\n\n", final_length);

 /* freeing memory */
  cudaFree(dlength);
  cudaFree(dposx);
  cudaFree(dposy);
  return 0;
}


