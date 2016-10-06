#include <ctime>
#include <cuda.h>
#include <iomanip>
#include <iostream>

using namespace std;

#define MASK_WIDTH 5
#define WIDTH 7

// Secuencial
void convolution_2D(double *v, double *mask, double *result) {
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			double Pvalue = 0;

			int N_start_point = i - (MASK_WIDTH / 2);
			int M_start_point = j - (MASK_WIDTH / 2);

			for (int ii = 0; ii < MASK_WIDTH; ii++) {
				for (int jj = 0; jj < MASK_WIDTH; jj++) {
					if (N_start_point + ii >= 0 && N_start_point + ii < WIDTH &&
							M_start_point + jj >= 0 && M_start_point + jj < WIDTH) {
						Pvalue += v[WIDTH * (N_start_point + ii) + (M_start_point + jj)] * mask[MASK_WIDTH * ii + jj];
					}
				}
			}
			result[WIDTH * i + j] = Pvalue;
		}
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

	// fillMatrix(h_m);
	convolution_2D(h_m, h_mask, h_result);
	printMatrix(h_m);
	printMatrix(h_result);

	return 0;
}
