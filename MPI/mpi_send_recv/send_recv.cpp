#include <iostream>
#include <mpi.h>

using namespace std;

int main() {
  // Initialize the MPI environment
  MPI::Init();
  // Find out rank, size
  int world_rank = MPI::COMM_WORLD.Get_rank();
  int world_size = MPI::COMM_WORLD.Get_size();
  MPI::Status mpi_status;

  // Get the name of processor
  char processor_name[MPI::MAX_PROCESSOR_NAME];
  int name_len;
  MPI::Get_processor_name(processor_name, name_len);

  // We are assuming at least 2 processes for this task
  if (world_size < 2) {
    cerr << "World size must be greater than 1 for " << processor_name << endl;
    MPI::COMM_WORLD.Abort(1);
  }

  int number;
  if (world_rank == 0) {
    // If we are rank 0, set the number to -1 and send it to process 1
    number = -1;
    // Send(void* buf, int count, MPI::Datatype& datatype, int dest, int tag)
    MPI::COMM_WORLD.Send(&number, 1, MPI::INT, 1, 0);
  } else if (world_rank == 1) {
    // Recv(void* buf, int count, MPI::Datatype& datatype, int source, int tag,
    // MPI::Status* status)
    MPI::COMM_WORLD.Recv(&number, 1, MPI::INT, 0, 0, mpi_status);
    cout << "Process 1 received number " << number << " from process 0" << endl;
  }

  MPI::Finalize();
}
