#!/bin/bash

#SBATCH --job-name=mpi_hello_world
#SBATCH --output=res_mpi_helloWorld.out
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun helloWorld
