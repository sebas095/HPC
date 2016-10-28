#include <cstdio>
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[]) {
  int rank, size;

  MPI::Init(argc, argv);             // starts MPI
  rank = MPI::COMM_WORLD.Get_rank(); // get current process id
  size = MPI::COMM_WORLD.Get_size(); // get number of processes
  cout << "Hello world from process " << rank << " of " << size << endl;
  MPI_Finalize();

  return 0;
}
