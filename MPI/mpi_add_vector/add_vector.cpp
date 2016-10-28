#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>

using namespace std;

#define SIZE 1000000  // size of vectors A, B and C
#define MASTER 0      // taskid of first taskid
#define FROM_MASTER 1 // setting a message type
#define FROM_WORKER 2 // setting a message type

int main() {
  int taskid, numtask, numworkers, source, mtype, rc;
  int i, aversize, elements, extra, offset, dest;
  float a[SIZE]; // vector A to be added
  float b[SIZE]; // vector B to be added
  float c[SIZE]; // result vector c
  MPI::Status mpi_status;

  MPI::Init();
  taskid = MPI::COMM_WORLD.Get_rank();
  numtask = MPI::COMM_WORLD.Get_size();

  if (numtask < 2) {
    cerr << "Need at least two MPI tasks. Quitting..." << endl;
    MPI::COMM_WORLD.Abort(rc);
    exit(1);
  }

  numworkers = numtask - 1;

  //------------------------ master task --------------------------------------
  if (taskid == MASTER) {
    cout << "mpi_add_vector has started with " << numtask << " tasks." << endl;
    cout << "Initializing arrays..." << endl;
    for (i = 0; i < SIZE; i++) {
      a[i] = rand() % SIZE;
      b[i] = rand() % SIZE;
      // cout << "a[" << i << "] = " << a[i] << "   b[" << i << "] = " << b[i]
      //      << endl;
    }

    // Send vector data to the worker tasks
    aversize = SIZE / numworkers;
    extra = SIZE % numworkers;
    offset = 0;
    mtype = FROM_MASTER;

    for (dest = 1; dest <= numworkers; dest++) {
      elements = (dest <= extra) ? aversize + 1 : aversize;
      cout << "Sending " << elements << " elements to task " << dest
           << " offset = " << offset << endl;

      MPI::COMM_WORLD.Send(&offset, 1, MPI::INT, dest, mtype);
      MPI::COMM_WORLD.Send(&elements, 1, MPI::INT, dest, mtype);
      MPI::COMM_WORLD.Send(&a[offset], elements, MPI::FLOAT, dest, mtype);
      MPI::COMM_WORLD.Send(&b[offset], elements, MPI::FLOAT, dest, mtype);
      offset += elements;
    }

    // Receive results from worker tasks
    mtype = FROM_WORKER;
    for (i = 1; i <= numworkers; i++) {
      source = i;
      MPI::COMM_WORLD.Recv(&offset, 1, MPI::INT, source, mtype, mpi_status);
      MPI::COMM_WORLD.Recv(&elements, 1, MPI::INT, source, mtype, mpi_status);
      MPI::COMM_WORLD.Recv(&c[offset], elements, MPI::FLOAT, source, mtype,
                           mpi_status);
      cout << "Received results from task " << source << endl;
    }

    // Print results
    string star(55, '*');
    // cout << star << endl;
    // cout << "Result Vector:" << endl;
    // for (i = 0; i < SIZE; i++) {
    //   cout << fixed << setprecision(2) << c[i] << "    ";
    // }
    // cout << endl << star << endl;
    cout << "Done." << endl;
  }

  //------------------------ worker task --------------------------------------
  if (taskid > MASTER) {
    mtype = FROM_MASTER;
    MPI::COMM_WORLD.Recv(&offset, 1, MPI::INT, MASTER, mtype, mpi_status);
    MPI::COMM_WORLD.Recv(&elements, 1, MPI::INT, MASTER, mtype, mpi_status);
    MPI::COMM_WORLD.Recv(&a, elements, MPI::FLOAT, MASTER, mtype, mpi_status);
    MPI::COMM_WORLD.Recv(&b, elements, MPI::FLOAT, MASTER, mtype, mpi_status);

    for (i = 0; i < elements; i++) {
      c[i] = a[i] + b[i];
    }

    mtype = FROM_WORKER;
    MPI::COMM_WORLD.Send(&offset, 1, MPI::INT, MASTER, mtype);
    MPI::COMM_WORLD.Send(&elements, 1, MPI::INT, MASTER, mtype);
    MPI::COMM_WORLD.Send(&c, elements, MPI::FLOAT, MASTER, mtype);
  }

  MPI::Finalize();
}
