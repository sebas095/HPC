#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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
  MPI_Status mpi_status;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtask);

  if (numtask < 2) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }

  numworkers = numtask - 1;

  //------------------------ master task --------------------------------------
  if (taskid == MASTER) {
    printf("mpi_mult_matrix has started with %d tasks.\n", numtask);
    printf("Initializing arrays...\n");
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
      printf("Sending %d rows to task %d offset = %d\n", rows, dest, offset);

      MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&a[offset][0], rows * NCA, MPI_FLOAT, dest, mtype,
               MPI_COMM_WORLD);
      MPI_Send(&b, NCA * NCB, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
      offset += rows;
    }

    // Receive results from worker tasks
    mtype = FROM_WORKER;
    for (i = 1; i <= numworkers; i++) {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &mpi_status);
      MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &mpi_status);
      MPI_Recv(&c[offset][0], rows * NCB, MPI_FLOAT, source, mtype,
               MPI_COMM_WORLD, &mpi_status);
      printf("Received results from task %d\n", source);
    }

    // Print results
    printf("******************************************************\n");
    printf("Result Matrix:\n");
    for (i = 0; i < NRA; i++) {
      printf("\n");
      for (j = 0; j < NCB; j++) {
        printf("%6.2f   ", c[i][j]);
      }
    }
    printf("\n******************************************************\n");
    printf("Done.\n");
  }

  //------------------------ worker task --------------------------------------
  if (taskid > MASTER) {
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &mpi_status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &mpi_status);
    MPI_Recv(&a, rows * NCA, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD,
             &mpi_status);
    MPI_Recv(&b, rows * NCB, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD,
             &mpi_status);

    for (k = 0; k < NCB; k++) {
      for (i = 0; i < rows; i++) {
        c[i][k] = 0.0;
        for (j = 0; j < NCA; j++) {
          c[i][k] += a[i][k] * b[k][j];
        }
      }
    }

    mtype = FROM_WORKER;
    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&c, rows * NCB, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
  }

  MPI_Finalize();
}
