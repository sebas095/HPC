#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

#define BLUR_SIZE 100

using namespace cv;
using namespace std;

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if (Col < w and Row < h) {
    int pixVal = 0;
    int pixels = 0;
    // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
    for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
      for (int blurCol = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurCol) {
        int currRow = Row + blurRow;
        int currCol = Col + blurCol;
        // Verify we have a valid image pixel
        if (currRow > -1 and currRow < h and currCol > -1 and currCol < w) {
          pixVal += in[currRow * w + currCol];
          // Keep track of number of pixels in the accumulated total
          pixels++;
        }
      }
    }
    // Write our new pixel value out
    out[Row * w + Col] = (unsigned char)(pixVal / pixels);
  }
}

void blurHost(unsigned char *in, unsigned char *out, int w, int h) {
  for (int Row = 0; Row < h; Row++) {
    for (int Col = 0; Col < w; Col++) {
      int pixVal = 0;
      int pixels = 0;
      // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
      for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++) {
        for (int blurCol = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurCol++) {
          int currRow = Row + blurRow;
          int currCol = Col + blurCol;
          // Verify we have a valid image pixel
          if (currRow > -1 and currRow < h and currCol > -1 and currCol < w) {
            pixVal += in[currRow * w + currCol];
            // Keep track of number of pixels in the accumulated total
            pixels++;
          }
        }
      }
      // Write our new pixel value out
      out[Row * w + Col] = (unsigned char)(pixVal / pixels);
    }
  }
}

int main() {
  cudaError_t error = cudaSuccess;
  clock_t startCPU, endCPU, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;

  Mat image;
  image = imread("../img/city.png", CV_LOAD_IMAGE_COLOR);

  if (!image.data) {
    cerr << "No image data" << endl;
    return EXIT_FAILURE;
  }

  unsigned char *blurImgCPU, *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput;
  Size s = image.size();

  // int rows = image.rows;
  // int cols = image.cols;
  int width = s.width;
  int height = s.height;
  int size = sizeof(unsigned char) * width * height * image.channels();

  dataRawImage = (unsigned char*)malloc(size);
  blurImgCPU = (unsigned char*)malloc(size);
  h_imageOutput = (unsigned char*)malloc(size);

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

  error = cudaMalloc((void**)&d_imageOutput, size);
  if (error != cudaSuccess) {
    cerr << "Error reservando memoria para d_imageOutput" << endl;
    return EXIT_FAILURE;
  }

  // Paralelo
  // int blockSize = 32;
  // dim3 dimBlock(blockSize, blockSize, 1);
  // dim3 dimGrid(ceil(width / float(blockSize)), ceil(height / float(blockSize)), 1);
  // blurKernel<<< dimGrid, dimBlock >>>(d_dataRawImage, d_imageOutput, width, height);
  // cudaMemcpy(h_imageOutput, d_imageOutput, size, cudaMemcpyDeviceToHost);
  // endGPU = clock();
  //
  // Mat blurImg;
  // blurImg.create(height, width, CV_8UC1);
  // blurImg.data = h_imageOutput;

  // Secuencial
  // startCPU = clock();
  // Mat blurImg2;
  // blurHost(dataRawImage, blurImgCPU, width, height);
  // blurImg2.create(height, width, CV_8UC1);
  // blurImg2.data = blurImgCPU;
  // endCPU = clock();
  //
  // imwrite("../img/Blur_Image_CPU.jpg", blurImg2);
  // imwrite("../img/Blur_Image_GPU.jpg", blurImg);

  namedWindow("city.png", WINDOW_AUTOSIZE);
  // namedWindow("blurImgGPU", WINDOW_AUTOSIZE);
  // namedWindow("blurImgCPU", WINDOW_AUTOSIZE);

  imshow("city.png", image);
  // imshow("blurImgGPU", blurImg);
  // imshow("blurImgCPU", blurImg2);
  waitKey(0);

  // cout << fixed;
  // cout.precision(10);
  //
  // gpu_time_used = ((double)(endGPU - startGPU)) / CLOCKS_PER_SEC;
  // cout << "Tiempo Algoritmo Paralelo: " << gpu_time_used << endl;

  // cpu_time_used = ((double)(endCPU - startCPU)) / CLOCKS_PER_SEC;
  // cout << "Tiempo Algoritmo Secuencual: " << cpu_time_used << endl;
  // cout << "La aceleracion obtenida es de " << (cpu_time_used / gpu_time_used) << "X"<< endl;

  cudaFree(d_dataRawImage);
  cudaFree(d_imageOutput);

  free(blurImgCPU);
  free(h_imageOutput);

  return 0;
}
