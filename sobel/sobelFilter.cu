#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
#include <cmath>

using namespace cv;
using namespace std;

const int MASK_WIDTH = 3;
const int MASK_HEIGHT = 3;

// Cache memory
__constant__ char MASK_X[MASK_WIDTH * MASK_HEIGHT];
__constant__ char MASK_Y[MASK_WIDTH * MASK_HEIGHT];

__device__
unsigned char check(double Pvalue) {
  Pvalue = (Pvalue < 0) ? 0 : Pvalue;
  Pvalue = (Pvalue > 255) ? 255 : Pvalue;
  return (unsigned char)Pvalue;
}

__global__
void sobelFilterKernel(unsigned char *img, unsigned char *grad_x,unsigned char *grad_y,
                       unsigned char *grad, int height, int width) {

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
          Px_value += img[width * currRow + currCol] * MASK_X[MASK_WIDTH * i + j];
          Py_value += img[width * currRow + currCol] * MASK_Y[MASK_WIDTH * i + j];
        }
      }
    }

    Px_value = check(Px_value);
    Py_value = check(Py_value);

    grad_x[width * row + col] = (unsigned char)Px_value;
    grad_y[width * row + col] = (unsigned char)Py_value;
    grad[width * row + col] = (unsigned char)sqrtf((Px_value * Px_value) + (Py_value * Py_value));
  }
}

void sobelFilter(Mat &image, Mat &image_gray, Mat &grad) {
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  GaussianBlur(image, image, Size(3, 3), 0, 0, BORDER_DEFAULT);
  cvtColor(image, image_gray, CV_BGR2GRAY);

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  Sobel(image_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);

  Sobel(image_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_y, abs_grad_y);

  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
}

void cudaCheckError(cudaError error, string error_msg) {
	if(error != cudaSuccess){
    cerr << error_msg << endl;
    exit(-1);
  }
}

int main() {
  cudaError_t error = cudaSuccess;
  clock_t startCPU, endCPU, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;

  // Device varuables
  unsigned char *d_grad_x, *d_grad_y, *d_grad;
  unsigned char *d_dataRawImage, *h_dataRawImage;

  // Host varuables
  unsigned char *h_grad, *h_grad_x, *h_grad_y;
  char h_mask_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  char h_mask_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  // Reading image with OpenCV
  Mat image, image_gray, grad;
  image = imread("../img/gatos.jpg", CV_LOAD_IMAGE_COLOR);
  // image = imread("./inputs/img4.jpg", CV_LOAD_IMAGE_COLOR);

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
  int maskSize = sizeof(char) * MASK_WIDTH * MASK_HEIGHT;

  // allocating memory for the host variables
  h_dataRawImage = new unsigned char[width * height];
  h_grad = new unsigned char[width * height];
  h_grad_x = new unsigned char[width * height];
  h_grad_y = new unsigned char[width * height];

  startGPU = clock();
  // allocating memory for the device variables
  error = cudaMalloc(&d_grad_x, size);
  cudaCheckError(error, "Error reservando memoria para d_grad_x");

  error = cudaMalloc(&d_grad_y, size);
  cudaCheckError(error, "Error reservando memoria para d_grad_y");

  error = cudaMalloc(&d_grad, size);
  cudaCheckError(error, "Error reservando memoria para d_grad");

  error = cudaMalloc(&d_dataRawImage, size);
  cudaCheckError(error, "Error reservando memoria para d_dataRawImage");

  // Copying data from host to device
  h_dataRawImage = image_gray.data;
  error = cudaMemcpy(d_dataRawImage, h_dataRawImage, size, cudaMemcpyHostToDevice);
  cudaCheckError(error, "Error copiando los datos de h_dataRawImage a d_dataRawImage");

  error = cudaMemcpyToSymbol(MASK_X, h_mask_x, maskSize);
  cudaCheckError(error, "Error copiando los datos de h_dataRawImage a d_dataRawImage");

  error = cudaMemcpyToSymbol(MASK_Y, h_mask_y, maskSize);
  cudaCheckError(error, "Error copiando los datos de h_mask_y a d_mask_y");

  // GPU
  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  // Launching kernel function
  dim3 dimGrid(ceil(width / float(blockSize)), ceil(height / float(blockSize)), 1);
  sobelFilterKernel<<< dimGrid, dimBlock >>>(d_dataRawImage, d_grad_x, d_grad_y, d_grad, height, width);

  // Copying data from device to host
  cudaMemcpy(h_grad_x, d_grad_x, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grad_y, d_grad_y, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grad, d_grad, size, cudaMemcpyDeviceToHost);
  endGPU = clock();

  Mat sobelImg, sobelImg_x, sobelImg_y;

  // Generating output images
  sobelImg.create(height, width, CV_8UC1);
  sobelImg_x.create(height, width, CV_8UC1);
  sobelImg_y.create(height, width, CV_8UC1);

  sobelImg.data = h_grad;
  sobelImg_x.data = h_grad_x;
  sobelImg_y.data = h_grad_y;

  // imwrite("../img/outputs/gatos_cpu_SF.png", grad);
  // imwrite("../img/outputs/gatos_gpu_SF_x.png", sobelImg_x);
  // imwrite("../img/outputs/gatos_gpu_SF_y.png", sobelImg_y);
  // imwrite("../img/outputs/gatos_gpu_SF.png", sobelImg);
  // imwrite("./outputs/1112783873.png", sobelImg);

  namedWindow("gatos.jpg", WINDOW_AUTOSIZE);
  namedWindow("SobelFilterCPU", WINDOW_AUTOSIZE);
  namedWindow("SobelFilterGPU", WINDOW_AUTOSIZE);
  namedWindow("SobelFilterGPU_x", WINDOW_AUTOSIZE);
  namedWindow("SobelFilterGPU_y", WINDOW_AUTOSIZE);

  imshow("gatos.jpg", image);
  imshow("SobelFilterCPU", grad);
  imshow("SobelFilterGPU", sobelImg);
  imshow("SobelFilterGPU_x", sobelImg_x);
  imshow("SobelFilterGPU_y", sobelImg_y);

  waitKey(0);

  cout << fixed;
  cout.precision(10);

  gpu_time_used = double(endGPU - startGPU) / CLOCKS_PER_SEC;
  cout << "Tiempo Algoritmo Paralelo: " << gpu_time_used << "s." << endl;

  cpu_time_used = double(endCPU - startCPU) / CLOCKS_PER_SEC;
  cout << "Tiempo Algoritmo Secuencial: " << cpu_time_used << "s." << endl;
  cout << "La aceleracion obtenida es de: " << (cpu_time_used / gpu_time_used )<< "X" << endl;

  // Freeing up memory
  free(h_grad);
  free(h_grad_x);
  free(h_grad_y);

  cudaFree(d_dataRawImage);
  cudaFree(MASK_X);
  cudaFree(MASK_Y);
  cudaFree(d_grad_x);
  cudaFree(d_grad_y);
  cudaFree(d_grad);

  return 0;
}
