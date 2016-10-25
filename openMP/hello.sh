#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=hello.out

export OMP_NUM_THREADS=10
./hello
