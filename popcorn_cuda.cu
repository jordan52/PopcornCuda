#include <stdio.h>
#include <sys/time.h>

// a kernel that will turn 1 to 42 (ascii for *)
__global__
void pop_kernel(int n, float *x)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) x[i] = 42.0*x[i];
}

int main() {
  // use the first cuda device you can find.
  cudaSetDevice(0);

  // let's pop about a million kernels, allocate memory
  // and set values to 1 (1 means the heat is on)
  int N = 1<<20;
  float *x, *d_x;
  x = (float*)malloc(N*sizeof(float));
  cudaMalloc(&d_x, N*sizeof(float)); 
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
  }

  // copy a million 1's to the GPU
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);

  // set up and start a timer
  float gpu_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // run the kernel on 1M elements
  pop_kernel<<<(N+255)/256, 256>>>(N, d_x);

  // stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time, start, stop);

  // get your popcorn off the GPU
  cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("Time to generate on GPU:  %3.1f ms \n", gpu_time);

  cudaFree(d_x);
  free(x);

  // Now let's try doing the same thing on the CPU

  // allocate fresh memory and set up timers
  x = (float*)malloc(N*sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
  }
  struct timeval t1, t2;
  gettimeofday(&t1, 0);
  
  // pop that corn on the CPU
  for (int i = 0; i < N; i++) {
    x[i] = 42*x[i];
  }

  gettimeofday(&t2, 0);
  double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

  printf("Time to generate on CPU:  %3.1f ms \n", time);
}
