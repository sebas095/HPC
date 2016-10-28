#!/bin/bash

#SBATCH --job-name=add_vector
#SBATCH --output=res_mpi_add_vector.out
#SBATCH --nodes=6
#SBATCH --ntasks=8
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun add_vector
