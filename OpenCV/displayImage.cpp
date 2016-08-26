#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void display(Mat &image) {
  Size s = image.size();
  int rows = image.rows;
  int cols = image.cols;
  int width = s.width;
  cout << "Matrix Image:" << endl;
  cout << "[";

  for (int i = 0; i < rows; i++) {
    cout << "[ ";
    for (int j = 0; j < cols; j++) {
      if (j) cout << " ";
      cout << (int)image.data[i * width + j];
    }
    cout << " ]" << endl;
  }

  cout << "]" << endl;
}

int showImage(string &path, string windowName) {
  // Read image file.
  string root = "../img/";
  char *dir = (char*)root.c_str();
  char *img = (char*)path.c_str();

  strcat(dir, img);
  Mat image = imread(dir, CV_LOAD_IMAGE_COLOR);
  display(image);

  // Check for invalid input.
  if (!image.data) {
    cerr << "Could not open or image not found" << endl;
    return EXIT_FAILURE;
  }

  // Create a window for display.
  namedWindow(windowName.c_str(), WINDOW_AUTOSIZE);
  // Show our image inside it.
  imshow(windowName.c_str(), image);
  // Wait for a keystroke in the window.
  waitKey(0);
}

int main(int argc, char const *argv[]) {
  if (argc != 2) {
    cerr << "Usage: ./displayImage <Image_Path>" << endl;
    return EXIT_FAILURE;
  }

  string path(argv[1]);
  string windowName("Display Image");
  showImage(path, windowName);

  return 0;
}
