CC = g++ -std=c++11 -O3

all: hello addVector

hello: hello.cpp
	$(CC) -o hello hello.cpp -fopenmp

hello: addVector.cpp
	$(CC) -o addVector addVector.cpp -fopenmp

clean:
	rm -rf hello addVector
