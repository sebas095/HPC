#include <iostream>
#include <mpi.h>

using namespace std;

#define NRA 62        // number of rows in matrix A
#define NCA 15        // number of columns in matrix A
#define NCB 7         // number of columns in matrix B
#define MASTER 0      // taskid of first task
#define FROM_MASTER 1 // setting a message type
#define FROM_WORKER 2 // setting a message type

void init(double *mat, int h, int w) {
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      mat[i * w + j] = 2;
    }
  }
}

int main(int argc, char *argv[]) {
  int numtask, taskid, numworkers, source, dest, mtype, rows;
  int averow, extra, offset, i, j, k, rc;
  double *a = new double[NRA * NCA]; // matrix A to be multiĺied
  double *b = new double[NCA * NCB]; // matrix B to be multiplied
  double *c = new double[NRA * NCB]; // reesult matrix C
  MPI_Status mpi_status;

  MPI_Init(NULL, NULL);                    // starts MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);  // get current process id
  MPI_Comm_size(MPI_COMM_WORLD, &numtask); // get number of processes

  if (numtask < 2) {
    cerr << "Need at least two MPI tasks. Quitting..." << endl;
    MPI_Abort(MPI_COMM_WORLD, rc);
    return EXIT_FAILURE;
  }

  numworkers = numtask - 1;

  //---------------------------------- master task -----------------------------
  if (taskid == MASTER) {
    cout << "mpi_mult_matrix has started with " << numtask << " tasks." << endl;
    cout << "Initializing arrays..." << endl;
    init(a, NRA, NCA);
    init(b, NCA, NCB);

    // Send matrix data to the worker tasks
    averow = NRA / numworkers;
    extra = NRA % numworkers;
    offset = 0;
    mtype = FROM_MASTER;

    for (dest = 1; dest <= numworkers; dest++) {
      rows = (dest <= extra) ? averow + 1 : averow;
      cout << "Sending " << rows << " to task " << dest
           << " offset = " << offset << endl;
      MPI_Send(); // TODO
      MPI_Send(); // TODO
      MPI_Send(); // TODO
      MPI_Send(); // TODO
      offset += rows;
    }

    // Receive results from worker tasks
    mtype = FROM_WORKER;
    for (i = 1; i <= numworkers; i++) {
      source = i;
      MPI_Recv(); // TODO
      MPI_Recv(); // TODO
      MPI_Recv(); // TODO
      cout << "Received results from task " << source << endl;
    }

    // Print results
    cout << "******************************************************" << endl;
    for (i = 0; i < NRA; i++) {
      cout << endl;
      for (j = 0; j < NCB; j++) {
        cout << "   " << fixed << setprecision(2) << c[i * NCB + j];
      }
    }
    cout << endl
         << "******************************************************" << endl;
    cout << "Done." << endl;
  }

  //---------------------------------- worker task -----------------------------
  if (taskid > MASTER) {
    mtype = FROM_MASTER;
    MPI_Recv(); // TODO
    MPI_Recv(); // TODO
    MPI_Recv(); // TODO
    MPI_Recv(); // TODO

    for (k = 0; k < NCB; k++) {
      for (i = 0; i < rows; i++) {
        for (j = 0 : j < NCA; j++) {
          c[i * NCB + k] += a[i * NCB + k] * b[k * NCA + j];
        }
      }
    }

    mtype = FROM_WORKER;
    MPI_Send(); // TODO
    MPI_Send(); // TODO
    MPI_Send(); // TODO
    MPI_Send(); // TODO
  }

  MPI_Finalize();
}
