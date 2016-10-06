#include <ctime>
#include <cuda.h>
#include <iomanip>
#include <iostream>

using namespace std;

#define MASK_WIDTH 5
#define WIDTH 7

// Secuencial
void convolution_2D(double *m, double *mask, double *result) {
  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < WIDTH; j++) {
      double Pvalue = 0;

      int N_start_point = i - (MASK_WIDTH / 2);
      int M_start_point = j - (MASK_WIDTH / 2);

      for (int ii = 0; ii < MASK_WIDTH; ii++) {
        for (int jj = 0; jj < MASK_WIDTH; jj++) {
          if (N_start_point + ii >= 0 && N_start_point + ii < WIDTH &&
              M_start_point + jj >= 0 && M_start_point + jj < WIDTH) {
            Pvalue += m[WIDTH * (N_start_point + ii) + (M_start_point + jj)] * mask[MASK_WIDTH * ii + jj];
          }
        }
      }
      result[WIDTH * i + j] = Pvalue;
    }
  }
}

__global__
void convolution_2D_kernel(double *m, double *mask, double *result) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < WIDTH && col < WIDTH) {
    double Pvalue = 0;

    int N_start_point = row - (MASK_WIDTH / 2);
    int M_start_point = col - (MASK_WIDTH / 2);

    for (int i = 0; i < MASK_WIDTH; i++) {
      for (int j = 0; j < MASK_WIDTH; j++) {
        if (N_start_point + i >= 0 && N_start_point + i < WIDTH &&
            M_start_point + j >= 0 && M_start_point + j < WIDTH) {
          Pvalue += m[WIDTH * (N_start_point + i) + (M_start_point + j)] * mask[MASK_WIDTH * i + j];
        }
      }
    }
    result[WIDTH * row + col] = Pvalue;
  }
}

void printMatrix(double *v) {
  for (int i = 0; i < WIDTH; i++) {
    cout << "[";
    for (int j = 0; j < WIDTH; j++) {
      if (j) cout << ", ";
      cout << v[WIDTH * i + j];
    }
    cout << "]" << endl;
  }
  cout << endl;
}

void fillMatrix(double *v) {
  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < WIDTH; j++) {
      v[WIDTH * i + j] = j + 1;
    }
  }
}

int main() {
  // Host
  double h_mask[] = {1, 2, 3, 2, 1, \
                     2, 3, 4, 3, 2, \
                     3, 4, 5, 4, 3, \
                     2, 3, 4, 3, 2, \
                     1, 2, 3, 2, 1};

  double h_m[] = {1, 2, 3, 4, 5, 6, 7, \
                 2, 3, 4, 5, 6, 7, 8, \
                 3, 4, 5, 6, 7, 8, 9, \
                 4, 5, 6, 7, 8, 5, 6, \
                 5, 6, 7, 8, 5, 6, 7, \
                 6, 7, 8, 9, 0, 1, 2, \
                 7, 8, 9, 0, 1, 2, 3};

  double *h_result = new double[WIDTH * WIDTH];
  double *ans = new double[WIDTH * WIDTH];
  // fillMatrix(h_m);

  {
    clock_t start = clock();

    convolution_2D(h_m, h_mask, h_result);
    printMatrix(h_m);
    printMatrix(h_result);

    clock_t end = clock();
    double time_used = double(end - start) / CLOCKS_PER_SEC;
    cout << "Tiempo invertido CPU = " << setprecision(10) << time_used << "s" << endl << endl;
  }

  // Device
  double *d_mask, *d_m, *d_result;
  int blockSize = 4;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(WIDTH / float(blockSize)), ceil(WIDTH / float(blockSize)), 1);

  cudaMalloc(&d_mask, sizeof(double) * MASK_WIDTH * MASK_WIDTH);
  cudaMalloc(&d_m, sizeof(double) * WIDTH * WIDTH);
  cudaMalloc(&d_result, sizeof(double) * WIDTH * WIDTH);

  // Device
  {
    clock_t start = clock();

    cudaMemcpy(d_m, h_m, sizeof(double) * WIDTH * WIDTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, sizeof(double) * MASK_WIDTH * MASK_WIDTH, cudaMemcpyHostToDevice);

    convolution_2D_kernel<<< dimGrid, dimBlock >>>(d_m, d_mask, d_result);
    cudaMemcpy(ans, d_result, sizeof(double) * WIDTH * WIDTH, cudaMemcpyDeviceToHost);
    printMatrix(h_m);
    printMatrix(ans);

    clock_t end = clock();
    double time_used = double(end - start) / CLOCKS_PER_SEC;
    cout << "Tiempo invertido GPU = " << setprecision(10) << time_used << "s" << endl << endl;
  }

  return 0;
}
