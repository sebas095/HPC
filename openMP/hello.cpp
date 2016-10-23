#include <cstdlib>
#include <iostream>
#include <omp.h>

using namespace std;

int main() {
  int nthreads, tid;
#pragma omp parallel private(nthreads, tid)
  {
    tid = omp_get_thread_num();
    cout << "Hello World from thread = " << tid << endl;

    if (tid == 0) {
      nthreads = omp_get_num_threads();
      cout << "Number of threads = " << nthreads << endl;
    }
  }
  return 0;
}
