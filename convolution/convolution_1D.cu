#include <ctime>
#include <cuda.h>
#include <iomanip>
#include <iostream>

using namespace std;

#define MASK_WIDTH 5
#define WIDTH 7

// Secuencial
void convolution_1D(double *v, double *mask, double *result) {
	for (int i = 0; i < WIDTH; i++) {
		double Pvalue = 0;
		int N_start_point = i - (MASK_WIDTH / 2);
		for (int j = 0; j < MASK_WIDTH; j++) {
			if (N_start_point + j >= 0 && N_start_point + j < WIDTH) {
				Pvalue += v[N_start_point + j] * mask[j];
			}
		}
		result[i] = Pvalue;
	}
}

__global__
void convolution_1D_kernel(double *v, double *mask, double *result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double Pvalue = 0;
	int N_start_point = idx - (MASK_WIDTH / 2);
	for (int j = 0; j < MASK_WIDTH; j++) {
		if (N_start_point + j >= 0 && N_start_point + j < WIDTH) {
			Pvalue += v[N_start_point + j] * mask[j];
		}
	}
	result[idx] = Pvalue;
}


void printVector(double *v) {
	cout << "[";
	for (int i = 0; i < WIDTH; i++) {
		if (i) cout << ", ";
		cout << setprecision(0) << v[i];
	}
	cout << "]" << endl;
}

void fillVector(double *v) {
	for (int i = 0; i < WIDTH; i++) {
		v[i] = i + 1;
	}
}

int main() {
  // Host variables
	double h_mask[] = {3, 4, 5, 4, 3};
	double *h_v = new double[WIDTH];
	double *h_result = new double[WIDTH];
	double *ans = new double[WIDTH];

	cout << fixed;
	fillVector(h_v);

	// Device
	{
		clock_t start = clock();

		convolution_1D(h_v, h_mask, h_result);
		printVector(h_v);
		printVector(h_result);

		clock_t end = clock();
		double time_used = double(end - start) / CLOCKS_PER_SEC;
		cout << "Tiempo invertido CPU = " << setprecision(10) << time_used << "s" << endl << endl;
	}

	// Device variables
	double *d_mask, *d_v, *d_result;
  int blockSize = 4;
	dim3 dimBlock(blockSize, 1, 1);
	dim3 dimGrid(ceil(WIDTH / float(blockSize)), 1, 1);

	cudaMalloc(&d_mask, sizeof(double) * MASK_WIDTH);
	cudaMalloc(&d_v, sizeof(double) * WIDTH);
	cudaMalloc(&d_result, sizeof(double) * WIDTH);

	// Host
	{
		clock_t start = clock();

		cudaMemcpy(d_v, h_v, sizeof(double) * WIDTH, cudaMemcpyHostToDevice);
		cudaMemcpy(d_mask, h_mask, sizeof(double) * MASK_WIDTH, cudaMemcpyHostToDevice);

		convolution_1D_kernel<<< dimGrid, dimBlock >>>(d_v, d_mask, d_result);
		cudaMemcpy(ans, d_result, sizeof(double) * WIDTH, cudaMemcpyDeviceToHost);
		printVector(h_v);
		printVector(ans);

		clock_t end = clock();
		double time_used = double(end - start) / CLOCKS_PER_SEC;
		cout << "Tiempo invertido GPU = " << setprecision(10) << time_used << "s" << endl;
	}

	delete h_v;
	delete h_result;
	delete ans;

	cudaFree(d_mask);
	cudaFree(d_v);
	cudaFree(d_result);

	return 0;
}
