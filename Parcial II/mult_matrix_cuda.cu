#include <cuda.h>
#include <cstdio>

__global__
void mult_mat(double *d_a, double *d_b, double *d_c, int height, int width_a, int width_b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width_b) {
    double p_result = 0;
    for (int k = 0; k < width_a; k++) {
      p_result += d_a[row * width_a + k] * d_b[k * width_b + col];
    }
    d_c[row * width_b + col] = p_result;
  }
}

void mult_mat_CUDA(double *h_a, double *h_b, double *h_c, int height, int width_a, int width_b) {
  double blocksize = 32;
  double *d_a, *d_b, *d_c;

  // Asign memory in the device
  cudaMalloc(&d_a, sizeof(double) * height * width_a);
  cudaMalloc(&d_b, sizeof(double) * width_a * width_b);
  cudaMalloc(&d_c, sizeof(double) * height * width_b);

  cudaMemcpy(d_a, h_a, height * width_a * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, width_a * width_b * sizeof(double), cudaMemcpyHostToDevice);

  dim3 dimBlock(blocksize, blocksize, 1);
  dim3 dimGrid(ceil(width_b / blocksize), ceil(height / blocksize), 1);

  mult_mat<<< dimGrid, dimBlock >>>(d_a, d_b, d_c, height, width_a, width_b);
  cudaMemcpy(h_c, d_c, height * width_b * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
