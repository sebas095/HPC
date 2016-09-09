# How to install OpenCV

[Step by step](http://www.linuxhispano.net/2012/11/05/instalar-opencv-2-4-2-ubuntu-12-04/)

For possible install error:

      sudo apt-get install qt5-default

Build and Execution

This part is very simple, It proceeds as any other project using CMake:

```
mkdir build
cd build/
cmake ..
make
```

Result

By now you should have an executable (called displayImage in this case).
You just have to run it given an image path as an argument, for example:

      ./blur
