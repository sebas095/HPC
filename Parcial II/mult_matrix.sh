#!/bin/bash

#SBATCH --job-name=mult_matrix
#SBATCH --output=res_mpi_mult_matrix.out
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 8
#sbatch --gres=gpu:1

mpirun ./build/mult_matrix
