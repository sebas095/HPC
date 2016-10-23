#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>

using namespace std;

int main() {
  int nthreads, tid;
#pragma omp parallel private(nthreads, tid)
  {
    tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);

    if (tid == 0) {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
  }
  return 0;
}
