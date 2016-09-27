#include <cuda.h>
#include <ctime>
#include <iostream>

#define TILE_WIDTH 1
#define endl '\n'

__global__
void multKernelTiled(float *d_M, float *d_N, float *d_R, int width_M, int height, int width_N) {
  __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float Pvalue = 0;

  for (int p = 0; p < width_M / TILE_WIDTH; p++) {
    if (row < height and (p * TILE_WIDTH + tx) < width_M) {
      ds_M[ty][tx] = d_M[row * width_M + p * TILE_WIDTH + tx];
    } else {
      ds_M[ty][tx] = 0.0;
    }

    if ((p * TILE_WIDTH + ty) < width_M and col < width_N) {
      ds_N[ty][tx] = d_N[(p * TILE_WIDTH + ty) * width_N + col];
    } else {
      ds_N[ty][tx] = 0.0;
    }
    __syncthreads();

    if (row < height and col < width_N)
      for (int k = 0; k < TILE_WIDTH; k++) {
        Pvalue += ds_M[ty][k] * ds_N[k][tx];
      }
    __syncthreads();
  }

  if (row < height and col < width_N)
    d_R[row * width_N + col] = Pvalue;
}

__global__
void multKernel(float *d_M, float *d_N, float *d_R, int width_M, int height_M, int width_N) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height_M and j < width_N) {
    int Pvalue = 0;
    for (int k = 0; k < width_M; k++) {
      Pvalue += d_M[i * width_M + k] * d_N[k * width_N + j];
    }
    d_R[i * width_N + j] = Pvalue;
  }
}

void mult(float *A, float *B, float *C, int width_A, int height_A, int width_B) {
  int aux = 0;
  for (int i = 0; i < height_A; i++) {
    for (int j = 0; j < width_B; j++) {
      aux = 0;
      for (int k = 0; k < width_A; k++)
        aux += A[i * width_A + k] * B[k * width_B + j];
      C[i * width_B + j] = aux;
    }
  }
}

void initValues(float *m, int width, int height) {
  int values = 1;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      m[i * width + j] = values++;
    }
  }
}

int testValues(float *A, float *B, int width, int height) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (A[i * width + j] != B[i * width + j]) {
        std::cout << "Mal Calculo..." << endl;
        return false;
      }
    }
  }

  std::cout << "Buen Calculo..." << endl;
  return true;
}

void print(float *m, int width, int height) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (j) std::cout << " ";
      std::cout << m[i * width + j];
    }
    std::cout << endl;
  }
  std::cout << endl;
}

int main() {
  int height_A = 2;
  int width_A = 4;
  int width_B = 2;

  float *A = new float[height_A * width_A];
  float *B = new float[width_A * width_B];
  float *C = new float[width_A * width_B];
  float *D = new float[width_A * width_B];

  initValues(A, width_A, height_A);
  initValues(B, width_B, width_A);

  print(A, width_A, height_A);
  print(B, width_B, width_A);

  float *d_A, *d_B, *d_D;
  int blocksize = 1;

  dim3 dimBlock(blocksize, blocksize, 1);
  dim3 dimGrid(ceil(width_B / float(blocksize)), ceil(height_A / float(blocksize)), 1);

  cudaMalloc((void**)&d_A, sizeof(float) * height_A * width_A);
  cudaMalloc((void**)&d_B, sizeof(float) * width_A * width_B);
  cudaMalloc((void**)&d_D, sizeof(float) * height_A * width_B);
  std::cout << std::fixed;

  // Mult CPU
  {
    clock_t start = clock();
    mult(A, B, C, width_A, height_A, width_B);
    clock_t end = clock();
    double cpu_time_used = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Tiempo invertido CPU = " << cpu_time_used << "s\n";
    // print(C, width_B, height_A);

  }

  // Mult GPU without tiles
  // {
  //   clock_t start = clock();
  //
  //   cudaMemcpy(d_A, A, sizeof(float) * height_A * width_A, cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_B, B, sizeof(float) * width_A * width_B, cudaMemcpyHostToDevice);
  //
  //   multKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_D, width_A, height_A, width_B);
  //   cudaMemcpy(D, d_D, sizeof(float) * height_A * width_B, cudaMemcpyDeviceToHost);
  //
  //   clock_t end = clock();
  //   double cpu_time_used = double(end - start) / CLOCKS_PER_SEC;
  //   std::cout << "Tiempo invertido GPU = " << cpu_time_used << "s\n";
  //   testValues(C, D, width_B, height_A);
  //   print(D, width_B, height_A);
  // }

  // Mult GPU with tiles
  {
    clock_t start = clock();

    cudaMemcpy(d_A, A, sizeof(float) * height_A * width_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * width_A * width_B, cudaMemcpyHostToDevice);

    multKernelTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_D, width_A, height_A, width_B);
    cudaMemcpy(D, d_D, sizeof(float) * height_A * width_B, cudaMemcpyDeviceToHost);

    clock_t end = clock();
    double cpu_time_used = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Tiempo invertido GPU = " << cpu_time_used << "s\n";
    testValues(C, D, width_B, height_A);
    // print(D, width_B, height_A);
  }

  delete A;
  delete B;
  delete C;
  delete D;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);

  return 0;
}
