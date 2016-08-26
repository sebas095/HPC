#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void showImage(string &path, string &windowName) {
  // Read image file.
  Mat image = imread(path, CV_LOAD_IMAGE_COLOR);
  display(image);

  // Check for invalid input.
  if (!image.data) {
    cerr << "Could not open or image not found" << endl;
    return EXIT_FAILURE;
  }

  // Create a window for display.
  nameWindow(windowName, WINDOW_AUTOSIZE);
  // Show our image inside it.
  imshow(windowName, image);
  // Wait for a keystroke in the window.
  waitKey(0);
}

void display(Mat &image) {
  int rows = image.rows;
  int cols = image.cols;
  cout << "["

  for (int i = 0; i < rows; i++) {
    cout << "[ "
    for (int j = 0; j < cols; j++) {
      if (j) cout << " ";
      cout << image.data[i][j];
    }
    cout << " ]" << endl;
  }

  cout << "]" << endl;
}

int main(int argc, char const *argv[]) {
  if (argc != 2) {
    cerr << "Usage: ./displayImage <Image_Path>" << endl;
    return EXIT_FAILURE;
  }

  string path(argv[1]);
  showImage("../img/" + path, "Display Image");

  return 0;
}
