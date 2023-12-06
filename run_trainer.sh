#!/bin/bash -l
#SBATCH --job-name="debug trainer"
#SBATCH --account="go41"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=debug
#SBATCH --constraint=gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load daint-gpu 
source $SCRATCH/lightning19/bin/activate

srun -ul python main.py
