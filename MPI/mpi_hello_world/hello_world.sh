#!/bin/bash

#SBATCH --job-name=helloWorld
#SBATCH --output=res_mpi_helloWorld.out
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun helloWorld
