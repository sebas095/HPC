#include <bits/stdc++.h>
#include <cuda.h>

#define H 1000
#define W 1000

using namespace std;

void foo(float* v) {
  for(int i = 0; i < H; i++) {
    for(int j = 0; j < W; j++) {
      v[i * W + j] = 2;
    }
  }
}

void mult(float *A, float *B,float *C) {
  float aux = 0;
  for(int i = 0; i < H; i++) {
    for(int j = 0; j < W; j++) {
      aux = 0;
      for(int k=0; k < W; k++)
        aux += A[i * W + k] * B[k * W + j];
     C[i * W + j] = aux;
    }
  }
}

void mostrar(float *v) {
  for(int i=0; i<H; i++){
    for(int j = 0; j < W; j++) {
      cout << v[i * W + j] << " ";
    }
    cout << endl;
  }
}

__global__
void multMat(float *d_A, float *d_B, float *d_C ) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < H && j < W){
    int Pvalue = 0;
    for(int k = 0; k < W; k++) {
       Pvalue += d_A[i * W + k] * d_B[k * W + j];
    }
    d_C[i * W + j] = Pvalue;
  }
}

int main() {
  float* A = new float[H * W];
  float* B = new float[H * W];
  float* C = new float[H * W];
  float* D = new float[H * W];

  foo(A);
  foo(B);

  {
    clock_t start = clock();

    mult(A, B, C);

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tiempo invertido CPU = %lf s\n", cpu_time_used);
  }

  float *d_A, *d_B, *d_D;
  float blockSize = 32;
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
