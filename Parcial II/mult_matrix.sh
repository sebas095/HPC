#!/bin/bash

#SBATCH --job-name=mult_matrix
#SBATCH --output=res_mpi_mult_matrix.out
#SBATCH --ntasks=8
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun mult_matrix
