#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

#define RED 2
#define GREEN 1
#define BLUE 0
#define CHANNELS 3

using namespace cv;
using namespace std;

__global__
void imgGrayGPU(unsigned char *imageInput, unsigned char *imageOutput, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if ((col < width) and (row < height)) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * CHANNELS;

    unsigned char b = imageInput[rgbOffset + BLUE];
    unsigned char g = imageInput[rgbOffset + GREEN];
    unsigned char r = imageInput[rgbOffset + RED];

    imageOutput[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

void imgGrayCPU(unsigned char *imageInput, unsigned char *imageOutput, int width, int rows, int cols) {
  int grayOffset = 0, rgbOffset = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      grayOffset = i * width + j;
      rgbOffset = grayOffset * CHANNELS;

      unsigned char b = imageInput[rgbOffset + BLUE];
      unsigned char g = imageInput[rgbOffset + GREEN];
      unsigned char r = imageInput[rgbOffset + RED];

      imageOutput[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
  }
}

int main() {
  cudaError_t error = cudaSuccess;
  clock_t startCPU, endCPU, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;

  Mat image;
  image = imread("../img/gatos2.jpg", CV_LOAD_IMAGE_COLOR);

  if (!image.data) {
    cerr << "No image data" << endl;
    return EXIT_FAILURE;
  }

  unsigned char *grayImgCPU, *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput;
  Size s = image.size();

  int rows = image.rows;
  int cols = image.cols;
  int width = s.width;
  int height = s.height;
  int size = sizeof(unsigned char) * width * height * image.channels();
  int sizeGray = sizeof(unsigned char) * width * height;

  dataRawImage = (unsigned char*)malloc(size);
  grayImgCPU = (unsigned char*)malloc(sizeGray);
  h_imageOutput = (unsigned char*)malloc(sizeGray);

  error = cudaMalloc((void**)&d_dataRawImage, size);
  if (error != cudaSuccess) {
   cerr << "Error reservando memoria para d_dataRawImage" << endl;
   return EXIT_FAILURE;
  }

  dataRawImage = image.data;
  startGPU = clock();
  error = cudaMemcpy(d_dataRawImage, dataRawImage, size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cerr << "Error copiando los datos de dataRawImage a d_dataRawImage" << endl;
    return EXIT_FAILURE;
  }

  error = cudaMalloc((void**)&d_imageOutput, sizeGray);
  if (error != cudaSuccess) {
    cerr << "Error reservando memoria para d_imageOutput" << endl;
    return EXIT_FAILURE;
  }

  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width / float(blockSize)), ceil(height / float(blockSize)), 1);
  imgGrayGPU<<< dimGrid, dimBlock >>>(d_dataRawImage, d_imageOutput, width, height);
  cudaMemcpy(h_imageOutput, d_imageOutput, sizeGray, cudaMemcpyDeviceToHost);
  endGPU = clock();

  Mat grayImg;
  grayImg.create(height, width, CV_8UC1);
  grayImg.data = h_imageOutput;

  startCPU = clock();
  Mat grayImg2;
  imgGrayCPU(dataRawImage, grayImgCPU, width, rows, cols);
  grayImg2.create(height, width, CV_8UC1);
  grayImg2.data = grayImgCPU;
  endCPU = clock();

  imwrite("../img/Gray_Image_CPU.jpg", grayImg2);
  imwrite("../img/Gray_Image_GPU.jpg", grayImg);

  namedWindow("gatos.jpg", WINDOW_AUTOSIZE);
  namedWindow("grayImgGPU", WINDOW_AUTOSIZE);
  namedWindow("grayImgCPU", WINDOW_AUTOSIZE);

  imshow("gatos.jpg", image);
  imshow("grayImgGPU", grayImg);
  imshow("grayImgCPU", grayImg2);
  waitKey(0);

  cout << fixed;
  cout.precision(10);

  gpu_time_used = ((double)(endGPU - startGPU)) / CLOCKS_PER_SEC;
  cout << "Tiempo Algoritmo Paralelo: " << gpu_time_used << endl;

  cpu_time_used = ((double)(endCPU - startCPU)) / CLOCKS_PER_SEC;
  cout << "Tiempo Algoritmo Secuencual: " << cpu_time_used << endl;
  cout << "La aceleracion obtenida es de " << (cpu_time_used / gpu_time_used) << "X"<< endl;

  cudaFree(d_dataRawImage);
  cudaFree(d_imageOutput);

  free(grayImgCPU);
  free(dataRawImage);
  free(h_imageOutput);

  return 0;
}
