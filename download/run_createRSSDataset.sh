#!/bin/bash -l
#SBATCH --job-name="download RSS"
#SBATCH --account="go41"
#SBATCH --time=0-35:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=long
#SBATCH --constraint=gpu


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module load daint-gpu 
source $SCRATCH/lightning/bin/activate

srun -ul python /scratch/snx3000/kschuurm/irradiance_estimation/download/CreateRSSDataset.py 2015 2022
