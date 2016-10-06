#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <cuda.h>

__global__ void multKernelTiled(const float* d_M, const float* d_N, float* d_R,
                                int width_M, int height_M, int width_N,
                                int tileWidth) {
    // Variable externa que almacena el tamaño del buffer asignado al inicial el
    // kernel
    extern __shared__ float buffer[];

    // Declaramos el tamaño de los bloques de memoria compartida
    float* ds_M = &buffer[0];
    float* ds_N = &buffer[tileWidth * tileWidth];

    // Referenciamos la ID del bloque y del hilo
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Obtenemos la dimension del Tile
    int row = by * tileWidth + ty;
    int col = bx * tileWidth + tx;

    // Variable para almacenar el valor final de la multiplicación de la
    // posición
    // [ty, tx]
    float Pvalue = 0;

    // Se recorre cada uno de los Tiles
    for (int p = 0; p < (tileWidth + width_M - 1) / tileWidth; p++) {
        // for (int p = 0; p < width_M / tileWidth; p++) {
        // Esta asigne los valores correspondientes de cada fila a la memoria
        // compartida teniendo en cuenta que si se sale del rango en valor es 0
        if (row < height_M && (p * tileWidth + tx) < width_M) {
            ds_M[ty * tileWidth + tx] = d_M[row * width_M + p * tileWidth + tx];
        } else {
            ds_M[ty * tileWidth + tx] = 0.0f;
        }

        // Esta asigne los valores correspondientes de cada columna a la memoria
        // compartida teniendo en cuenta que si se sale del rango en valor es 0
        if ((p * tileWidth + ty) < width_M && col < width_N) {
            ds_N[ty * tileWidth + tx] =
                d_N[(p * tileWidth + ty) * width_N + col];
        } else {
            ds_N[ty * tileWidth + tx] = 0.0f;
        }

        __syncthreads();  // Esperamos a que todos los hilos del bloque lleguen
                          // acá

        // Se hace el producto punto entre las dos memorias compartidas
        if (row < height_M && col < width_N) {
            for (int k = 0; k < tileWidth; k++) {
                Pvalue += ds_M[ty * tileWidth + k] * ds_N[k * tileWidth + tx];
            }
        }

        __syncthreads();  // Esperamos a que todos los hilos del bloque lleguen
                          // acá
    }

    // Se almacena el valor calculado en la posicion correspondiente de la
    // matriz
    // resultante
    if (row < height_M && col < width_N) {
        d_R[row * width_N + col] = Pvalue;
    }
}

void mult(const float* A, const float* B, float* C, int width_A, int height_A,
          int width_B) {
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

void initValues(float* m, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            m[i * width + j] = 1.0;
        }
    }
}

bool verifyValues(const float* A, const float* B, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (A[i * width + j] != B[i * width + j]) {
                return false;
            }
        }
    }
    return true;
}

void printMatrix(const float* m, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (j) std::cout << " ";
            std::cout << m[i * width + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int run(int height_A, int width_A, int width_B, bool do_print = false) {
    float* A = new float[height_A * width_A];
    float* B = new float[width_A * width_B];
    float* C = new float[width_A * width_B];
    float* D = new float[width_A * width_B];

    int tileWidth = 1;
    for (size_t i = 32; i >= 1; i--) {
        if (i * i <= height_A * width_B) {
            tileWidth = i;
            break;
        }
    };

    std::cout << "Tamaño del Tile: " << tileWidth << "\n";
    size_t memSize = (2 * tileWidth * tileWidth) * sizeof(float);

    initValues(A, width_A, height_A);
    initValues(B, width_B, width_A);

    if (do_print) printMatrix(A, width_A, height_A);
    if (do_print) printMatrix(B, width_B, width_A);

    float *d_A, *d_B, *d_D;
    int blocksize = tileWidth;

    dim3 dimBlock(blocksize, blocksize, 1);
    dim3 dimGrid(ceil(width_B / float(blocksize)),
                 ceil(height_A / float(blocksize)), 1);

    cudaMalloc(&d_A, sizeof(float) * height_A * width_A);
    cudaMalloc(&d_B, sizeof(float) * width_A * width_B);
    cudaMalloc(&d_D, sizeof(float) * height_A * width_B);

    std::cout << std::fixed;

    // Mult CPU
    double cpu_time_used;
    {
        clock_t start = clock();
        mult(A, B, C, width_A, height_A, width_B);
        clock_t end = clock();
        cpu_time_used = double(end - start) / CLOCKS_PER_SEC;
        std::cout << "Tiempo invertido CPU = " << cpu_time_used << "s\n";
        if (do_print) printMatrix(C, width_B, height_A);
    }

    // Mult GPU with tiles
    double gpu_time_used;
    {
        clock_t start = clock();

        cudaMemcpy(d_A, A, sizeof(float) * height_A * width_A,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, sizeof(float) * width_A * width_B,
                   cudaMemcpyHostToDevice);

        multKernelTiled<<<dimGrid, dimBlock, memSize>>>(
            d_A, d_B, d_D, width_A, height_A, width_B, tileWidth);
        cudaMemcpy(D, d_D, sizeof(float) * height_A * width_B,
                   cudaMemcpyDeviceToHost);

        clock_t end = clock();
        gpu_time_used = double(end - start) / CLOCKS_PER_SEC;
        std::cout << "Tiempo invertido GPU = " << gpu_time_used << "s\n";
        if (do_print) printMatrix(D, width_B, height_A);
    }

    std::cout << "La aceleracion total obtenida es de: "
              << (cpu_time_used / gpu_time_used) << "x" << std::endl;

    bool equal = verifyValues(C, D, width_B, height_A);

    if (equal) {
        std::cout << "Buen calculo!, las matrices son iguales 😄" << std::endl;
    } else {
        std::cout << "Mal calculo!, las matrices son diferentes 😱" << std::endl;
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

struct TestCase {
    TestCase(int a, int b, int c) : a(a), b(b), c(c) {}
    int a;
    int b;
    int c;
};

int main() {
    std::vector<TestCase> tests;
    tests.push_back(TestCase(2, 4, 2));
    tests.push_back(TestCase(30, 50, 80));
    tests.push_back(TestCase(99, 100, 80));
    tests.push_back(TestCase(350, 400, 700));
    tests.push_back(TestCase(500, 800, 900));
    tests.push_back(TestCase(1000, 1000, 1000));

    for (size_t i = 0; i < tests.size(); i++) {
        TestCase& t = tests[i];
        std::cout << "Probando Matrices de tamaño [" << t.a << ", " << t.b
                  << "] * [" << t.b << ", " << t.c << "]" << std::endl;
        for (size_t j = 0; j < 5; j++) {
            run(t.a, t.b, t.c);
            if (j < 5 - 1)
                std::cout
                    << "--------------------------------------------------\n";
        }
        std::cout << "==================================================\n";
    }
}
