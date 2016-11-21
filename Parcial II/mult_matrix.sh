#!/bin/bash

#SBATCH --job-name=mult_matrix
#SBATCH --output=res_mpi_mult_matrix.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
mpirun ./build/mult_matrix
