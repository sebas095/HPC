#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
#include <cmath>

using namespace cv;
using namespace std;

const int MASK_WIDTH = 3;
const int MASK_HEIGHT = 3;

__global__
void sobelOperatorKernel(unsigned char *img, unsigned char *mask_x, unsigned char *mask_y,
                       unsigned char *grad_x, unsigned char *grad_y, int height, int width) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    double Px_value = 0;
    double Py_value = 0;

    int row_start_point = row - (MASK_HEIGHT / 2);
    int col_start_point = col - (MASK_WIDTH / 2);

    for (int i = 0; i < MASK_HEIGHT; i++) {
      for (int j = 0; j < MASK_WIDTH; j++) {
        int currRow = row_start_point + i;
        int currCol = col_start_point + j;
        if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width) {
          Px_value += img[width * currRow + currCol] * mask_x[MASK_WIDTH * i + j];
          Py_value += img[width * currRow + currCol] * mask_y[MASK_WIDTH * i + j];
        }
      }
    }

    grad_x[width * row + col] = Px_value;
    grad_y[width * row + col] = Py_value;
  }
}

__global__
void sobelFilterKernel(unsigned char *grad_x, unsigned char *grad_y, unsigned char *grad, int height, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    unsigned char currGrad_x = grad_x[width * row + col];
    unsigned char currGrad_y = grad_y[width * row + col];

    // currGrad_x = (currGrad_x > 255) ? 255 : currGrad_x;
    // currGrad_x = (currGrad_x < 0) ? 0 : currGrad_x;
    //
    // currGrad_y = (currGrad_y > 255) ? 255 : currGrad_y;
    // currGrad_y = (currGrad_y < 0) ? 0 : currGrad_y;

    double grad_value = sqrt(double(currGrad_x * currGrad_x) + double(currGrad_y * currGrad_y));
    // double grad_value = fabs((double)currGrad_x) + fabs((double)currGrad_y);
    // grad_value = (grad_value > 255) ? 255 : grad_value;
    // grad_value = (grad_value < 0) ? 0 : grad_value;
    grad[width * row + col] = (unsigned char)ceil(grad_value);
  }
}

void sobelFilter(Mat &image, Mat &image_gray, Mat &grad) {
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  GaussianBlur(image, image, Size(3, 3), 0, 0, BORDER_DEFAULT);
  cvtColor(image, image_gray, COLOR_RGB2GRAY);

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  Sobel(image_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);

  Sobel(image_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_y, abs_grad_y);

  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
}

int main() {
  cudaError_t error = cudaSuccess;
  clock_t startCPU, endCPU, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;
  unsigned char *d_grad_x, *d_grad_y, *h_grad, *d_grad;
  unsigned char *d_mask_x, *d_mask_y;
  unsigned char *d_dataRawImage, *h_dataRawImage;

  unsigned char h_mask_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  unsigned char h_mask_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  Mat image, image_gray, grad;
  image = imread("../img/gatos.jpg", CV_LOAD_IMAGE_COLOR);

  if (!image.data) {
    cerr << "No image data" << endl;
    return EXIT_FAILURE;
  }

  // CPU
  startCPU = clock();
  sobelFilter(image, image_gray, grad);
  endCPU = clock();

  Size s = image_gray.size();
  int width = s.width;
  int height = s.height;
  int size = sizeof(unsigned char) * width * height;
  int maskSize = sizeof(unsigned char) * MASK_WIDTH * MASK_HEIGHT;
  h_dataRawImage = new unsigned char[width * height];
  h_grad = new unsigned char[width * height];

  startGPU = clock();
  error = cudaMalloc(&d_mask_x, maskSize);
  if (error != cudaSuccess) {
    cerr << "Error reservando memoria para d_mask_x" << endl;
    return EXIT_FAILURE;
  }

  error = cudaMalloc(&d_mask_y, maskSize);
  if (error != cudaSuccess) {
    cerr << "Error reservando memoria para d_mask_y" << endl;
    return EXIT_FAILURE;
  }

  error = cudaMalloc(&d_grad_x, size);
  if (error != cudaSuccess) {
    cerr << "Error reservando memoria para d_grad_x" << endl;
    return EXIT_FAILURE;
  }

  error = cudaMalloc(&d_grad_y, size);
  if (error != cudaSuccess) {
    cerr << "Error reservando memoria para d_grad_y" << endl;
    return EXIT_FAILURE;
  }

  error = cudaMalloc(&d_grad, size);
  if (error != cudaSuccess) {
    cerr << "Error reservando memoria para d_grad" << endl;
    return EXIT_FAILURE;
  }

  error = cudaMalloc(&d_dataRawImage, size);
  if (error != cudaSuccess) {
    cerr << "Error reservando memoria para d_dataRawImage" << endl;
    return EXIT_FAILURE;
  }

  h_dataRawImage = image_gray.data;
  error = cudaMemcpy(d_dataRawImage, h_dataRawImage, size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cerr << "Error copiando los datos de h_dataRawImage a d_dataRawImage" << endl;
    return EXIT_FAILURE;
  }

  error = cudaMemcpy(d_mask_x, h_mask_x, maskSize, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cerr << "Error copiando los datos de h_mask_x a d_mask_x" << endl;
    return EXIT_FAILURE;
  }

  error = cudaMemcpy(d_mask_y, h_mask_y, maskSize, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cerr << "Error copiando los datos de h_mask_y a d_mask_y" << endl;
    return EXIT_FAILURE;
  }

  // GPU
  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width / float(blockSize)), ceil(height / float(blockSize)), 1);
  sobelOperatorKernel<<< dimGrid, dimBlock >>>(d_dataRawImage, d_mask_x, d_mask_y, d_grad_x, d_grad_y, height, width);
  cudaDeviceSynchronize();
  sobelFilterKernel<<< dimGrid, dimBlock >>>(d_grad_x, d_grad_y, d_grad, height, width);
  cudaMemcpy(h_grad, d_grad, size, cudaMemcpyDeviceToHost);
  endGPU = clock();

  Mat sobelImg;
  sobelImg.create(height, width, CV_8UC1);
  sobelImg.data = h_grad;

  // namedWindow("gatos.jpg", WINDOW_AUTOSIZE);
  // namedWindow("SobelFilterCPU", WINDOW_AUTOSIZE);
  namedWindow("SobelFilterGPU", WINDOW_AUTOSIZE);

  // imshow("gatos.jpg", image);
  // imshow("SobelFilterCPU", grad);
  imshow("SobelFilterGPU", sobelImg);

  waitKey(0);

  cout << fixed;
  cout.precision(10);

  gpu_time_used = double(endGPU - startGPU) / CLOCKS_PER_SEC;
  cout << "Tiempo Algoritmo Paralelo: " << gpu_time_used << "s." << endl;

  cpu_time_used = double(endCPU - startCPU) / CLOCKS_PER_SEC;
  cout << "Tiempo Algoritmo Secuencial: " << cpu_time_used << "s." << endl;
  cout << "La aceleracion obtenida es de: " << (cpu_time_used / gpu_time_used )<< "X" << endl;

  free(h_grad);

  cudaFree(d_dataRawImage);
  cudaFree(d_mask_x);
  cudaFree(d_mask_y);
  cudaFree(d_grad_x);
  cudaFree(d_grad_y);
  cudaFree(d_grad);

  return 0;
}
