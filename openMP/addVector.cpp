#include <cstdlib>
#include <iostream>
#include <omp.h>

#define CHUNKSIZE 10
#define N 100

using namespace std;

void init(float *v) {
  for (int i = 0; i < N; i++) {
    v[i] = rand() % N;
  }
}

int main() {
  int nthreads, tid, i, chunk;
  float a[N], b[N], c[N];

  // Initialization
  init(a);
  init(b);
  chunk = CHUNKSIZE;

#pragma omp parallel shared(a, b, c, nthreads, chunk) private(i, tid)
  {
    tid = omp_get_thread_num();
    if (tid == 0) {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }

    printf("Thread %d starting...\n", tid);

#pragma omp for schedule(dynamic, chunk)
    for (i = 0; i < N; i++) {
      c[i] = a[i] + b[i];
      printf("Thread %d:\n", tid);
      printf("a[%d]: %f + b[%d]: %f = c[%d]: %f\n\n", i, a[i], i, b[i], i,
             c[i]);
    }
  }
  return 0;
}
