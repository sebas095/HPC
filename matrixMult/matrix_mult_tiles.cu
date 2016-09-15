#include <cuda.h>
#include <cstdio>
#include <ctime>
#include <iostream>

#define TILE_WIDTH 32
#define H 100000
#define W 100000

using namespace std;

void foo(float* v) {
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      v[i * W + j] = 2;
    }
  }
}

void mult(float* A, float* B, float* C) {
  int aux = 0;
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      aux = 0;
      for (int k = 0; k < W; k++) aux += A[i * W + k] * B[k * W + j];
      C[i * W + j] = aux;
  }
  }
}

void mostrar(float* v) {
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      cout << v[i * W + j] << " ";
    }
    cout << endl;
  }
}

__global__ void multMat(float* d_A, float* d_B, float* d_C) {
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  int Pvalue = 0;
  for (size_t phase = 0; phase < W / TILE_WIDTH; phase++) {
    ds_A[ty][tx] = d_A[row * W + phase * TILE_WIDTH + tx];
    ds_B[ty][tx] = d_B[(phase * TILE_WIDTH + ty) * W + col];
    __syncthreads();

    for (int k = 0; k < W; k++) {
      Pvalue += ds_A[ty][k] * ds_B[k][tx];
    }

    __syncthreads();
  }
  d_C[col * W + row] = Pvalue;
}

int main() {
  float* A = new float[H * W];
  float* B = new float[H * W];
  float* C = new float[H * W];
  float* D = new float[H * W];

  foo(A);
  foo(B);

  // {
  //   clock_t start = clock();
  //
  //   mult(A, B, C);
  //
  //   clock_t end = clock();
  //   double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  //   printf("Tiempo invertido CPU = %lf s\n", cpu_time_used);
  // }

  float *d_A, *d_B, *d_D;
  float blockSize = TILE_WIDTH;
  dim3 dimBlock(blockSize, blockSize);
  dim3 dimGrid(ceil(W / float(blockSize)), ceil(H / float(blockSize)), 1);

  cudaMalloc((void**)&d_A, sizeof(float) * H * W);
  cudaMalloc((void**)&d_B, sizeof(float) * H * W);
  cudaMalloc((void**)&d_D, sizeof(float) * H * W);

  {
    clock_t start = clock();

    cudaMemcpy(d_A, A, sizeof(float) * H * W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * H * W, cudaMemcpyHostToDevice);

    multMat<<<dimGrid, dimBlock>>>(d_A, d_B, d_D);
    cudaMemcpy(D, d_D, sizeof(float) * H * W, cudaMemcpyDeviceToHost);

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    std::cout << "Tiempo invertido GPU = " << cpu_time_used << "s\n";
  }

  delete A;
  delete B;
  delete C;
  delete D;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);
}
