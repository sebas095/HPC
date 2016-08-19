#include <cuda.h>
#include <stdio.h>
#include <time.h>
#define N 100

__global__
void addVectorGPU(int* a, int* b, int* c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

void addVectorCPU(int* a, int* b, int* c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

void printVector(int* a) {
  for (int i = 0; i < N; i++) {
    if (i) printf(" ");
    printf("%d", a[i]);
  }

  printf("\n");
}

int main() {
  clock_t start, end;
  double cpu_time_used, gpu_time_used;
  int *h_a, *h_b, *h_c, *h_result;
  int *d_a, *d_b, *d_c;
  const int size = sizeof(int) * N;

  h_a = (int*)malloc(size);
  h_b = (int*)malloc(size);
  h_c = (int*)malloc(size);
  h_result = (int*)malloc(size);

  // init
  for (int i = 0; i < N; i++) {
    h_a[i] = i + 1;
    h_b[i] = (i + 1) * 2;
    h_c[i] = 0;
  }

  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, size);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  start = clock();
  addVectorCPU(h_a, h_b, h_c);
  // printVector(h_c);
  end = clock();

  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido CPU = %lf s\n", cpu_time_used);

  dim3 threads_per_block(10, 1, 1);
  dim3 number_of_blocks((N / threads_per_block.x) + 1, 1, 1);

  start = clock();
  addVectorGPU<<< number_of_blocks, threads_per_block >>>(d_a, d_b, d_c);
  cudaMemcpy(h_result, d_c, size, cudaMemcpyDeviceToHost);
  end = clock();

  gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido GPU = %lf s\n", gpu_time_used);

  free(h_a); free(h_b);
  free(h_c); free(h_result);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
