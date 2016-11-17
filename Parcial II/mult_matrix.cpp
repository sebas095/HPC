#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>

using namespace std;

#define NRA 3         // number of rows in matrix A
#define NCA 3         // number of columns in matrix A
#define NCB 2         // number of columns in matrix B
#define MASTER 0      // taskid of first task
#define FROM_MASTER 1 // setting a message type
#define FROM_WORKER 2 // setting a message type

void init(double *mat, int h, int w) {
  double n = 1;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      mat[i * w + j] = n++;
    }
  }
}

void print(double *mat, int h, int w) {
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      cout << mat[i * w + j] << "  ";
    }
    cout << endl;
  }
}

// Mult in MPI
void MPI_Multiply(double *a, double *b, double *c, int rows, int h, int w) {
  for (int j = 0; j < w; j++) {
    for (int i = 0; i < rows; i++) {
      for (int k = 0; k < h; k++) {
        c[i * w + j] += a[i * h + k] * b[k * w + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int numtask, taskid, numworkers, source, dest, mtype, rows;
  int averow, extra, offset, i, j, k, rc;
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
    double *a = new double[NRA * NCA]; // matrix A to be multiĺied
    double *b = new double[NCA * NCB]; // matrix B to be multiplied
    double *c = new double[NRA * NCB]; // reesult matrix C
    cout << "mpi_mult_matrix has started with " << numtask << " tasks." << endl;
    cout << "Initializing arrays..." << endl;
    init(a, NRA, NCA);
    init(b, NCA, NCB);
    // print(a, NRA, NCA);
    // print(b, NCA, NCB);

    // Send matrix data to the worker tasks
    averow = NRA / numworkers;
    extra = NRA % numworkers;
    offset = 0;
    mtype = FROM_MASTER;

    for (dest = 1; dest <= numworkers; dest++) {
      rows = (dest <= extra) ? averow + 1 : averow;
      cout << "Sending " << rows << " to task " << dest
           << " offset = " << offset << endl;
      MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&a[offset * NCA], rows * NCA, MPI_DOUBLE, dest, mtype,
               MPI_COMM_WORLD);
      MPI_Send(b, NCA * NCB, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
      offset += rows;
    }

    // Receive results from worker tasks
    mtype = FROM_WORKER;
    for (i = 1; i <= numworkers; i++) {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &mpi_status);
      MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &mpi_status);
      MPI_Recv(&c[offset * NCB], rows * NCB, MPI_DOUBLE, source, mtype,
               MPI_COMM_WORLD, &mpi_status);
      cout << "Received results from task " << source << endl;
    }

    // Print results
    cout << "******************************************************" << endl;
    for (i = 0; i < NRA; i++) {
      cout << endl;
      for (j = 0; j < NCB; j++) {
        cout << fixed << setprecision(2) << c[i * NCB + j] << "   ";
      }
    }
    cout << endl
         << "******************************************************" << endl;
    cout << "Done." << endl;
    free(a);
    free(b);
    free(c);
  }

  //---------------------------------- worker task -----------------------------
  if (taskid > MASTER) {
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &mpi_status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &mpi_status);

    double *a = new double[rows * NCA];
    double *b = new double[NCA * NCB];
    double *c = new double[rows * NCB];

    MPI_Recv(a, rows * NCA, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD,
             &mpi_status);
    MPI_Recv(b, NCA * NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD,
             &mpi_status);
    MPI_Multiply(a, b, c, rows, NCA, NCB);

    mtype = FROM_WORKER;
    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(c, rows * NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);

    free(a);
    free(b);
    free(c);
  }

  MPI_Finalize();
}
