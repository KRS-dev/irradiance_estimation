# make sure to set `--job-name "interactive"`

# module load daint-gpu PyTorch h5py dask
source $SCRATCH/pytorch/bin/activate

srun --nodes=1 --ntasks-per-node=1 -C gpu --time=01:00:00 --account go41 --pty bash -i

# now run scripts normally