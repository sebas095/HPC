#!/bin/bash
#SBATCH --nodes=1
export OMP_NUM_THREADS=16
./omp_hello -fopenmp
