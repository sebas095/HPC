#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>

using namespace std;

#define NRA 62        // number of rows in matrix A
#define NCA 15        // number of columns in maatrix A
#define NCB 7         // number of columns in matrix B
#define MASTER 0      // taskid of first taskid
#define FROM_MASTER 1 // setting a message type
#define FROM_WORKER 2 // setting a message type

int main() {
  int numtask, taskid, numworkers, source, dest, mtype, rows;
  int averow, extra, offset, i, j, k, rc;
  float a[NRA][NCA]; // matrix A to be multiplied
  float b[NCA][NCB]; // matrix B to be multiplied
  float c[NRA][NCB]; // result matrix C
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
    cout << "mpi_mult_matrix has started with " << numtask << " tasks." << endl;
    cout << "Initializing arrays..." << endl;
    for (i = 0; i < NRA; i++) {
      for (j = 0; j < NCA; j++) {
        a[i][j] = i + j;
      }
    }

    for (i = 0; i < NCA; i++) {
      for (j = 0; j < NCB; j++) {
        b[i][j] = i * j;
      }
    }

    // Send matrix data to the worker tasks
    averow = NRA / numworkers;
    extra = NRA % numworkers;
    offset = 0;
    mtype = FROM_MASTER;

    for (dest = 1; dest <= numworkers; dest++) {
      rows = (dest <= extra) ? averow + 1 : averow;
      cout << "Sending " << rows << " rows to task " << dest
           << " offset = " << offset << endl;

      MPI::COMM_WORLD.Send(&offset, 1, MPI::INT, dest, mtype);
      MPI::COMM_WORLD.Send(&rows, 1, MPI::INT, dest, mtype);
      MPI::COMM_WORLD.Send(&a[offset][0], rows * NCA, MPI::FLOAT, dest, mtype);
      MPI::COMM_WORLD.Send(&b, NCA * NCB, MPI::FLOAT, dest, mtype);
      offset += rows;
    }

    // Receive results from worker tasks
    mtype = FROM_WORKER;
    for (i = 1; i <= numworkers; i++) {
      source = i;
      MPI::COMM_WORLD.Recv(&offset, 1, MPI::INT, source, mtype, mpi_status);
      MPI::COMM_WORLD.Recv(&rows, 1, MPI::INT, source, mtype, mpi_status);
      MPI::COMM_WORLD.Recv(&c[offset][0], rows * NCB, MPI::FLOAT, source, mtype,
                           mpi_status);
      cout << "Received results from task " << source << endl;
    }

    // Print results
    string star(55, '*');
    cout << star << endl;
    cout << "Result Matrix:" << endl;
    for (i = 0; i < NRA; i++) {
      cout << endl;
      for (j = 0; j < NCB; j++) {
        cout << fixed << setprecision(2) << c[i][j] << "   ";
      }
    }
    cout << endl << star << endl;
    cout << "Done." << endl;
  }

  //------------------------ worker task --------------------------------------
  if (taskid > MASTER) {
    mtype = FROM_MASTER;
    MPI::COMM_WORLD.Recv(&offset, 1, MPI::INT, MASTER, mtype, mpi_status);
    MPI::COMM_WORLD.Recv(&rows, 1, MPI::INT, MASTER, mtype, mpi_status);
    MPI::COMM_WORLD.Recv(&a, rows * NCA, MPI::FLOAT, MASTER, mtype, mpi_status);
    MPI::COMM_WORLD.Recv(&b, rows * NCB, MPI::FLOAT, MASTER, mtype, mpi_status);

    for (k = 0; k < NCB; k++) {
      for (i = 0; i < rows; i++) {
        c[i][k] = 0.0;
        for (j = 0; j < NCA; j++) {
          c[i][k] += a[i][k] * b[k][j];
        }
      }
    }

    mtype = FROM_WORKER;
    MPI::COMM_WORLD.Send(&offset, 1, MPI::INT, MASTER, mtype);
    MPI::COMM_WORLD.Send(&rows, 1, MPI::INT, MASTER, mtype);
    MPI::COMM_WORLD.Send(&c, rows * NCB, MPI::FLOAT, MASTER, mtype);
  }

  MPI::Finalize();
}
