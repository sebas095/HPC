#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output matriz.out
export OMP_NUM_THREADS=10
./addMatrix -fopenmp
