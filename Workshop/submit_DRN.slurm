#!/bin/bash -l
#SBATCH --job-name="rechit_gnn_train"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bapibasakfulia@gmail.com
#SBATCH -o logs/slurm-%j.out
#SBATCH --exclude=gpu024  # Avoid node where other job is running

echo "SLURM job started on $(hostname) at $(date)"

# ------------------------------
# Load system modules
# ------------------------------
module load iiser/apps/cuda/11.4
module load cdac/spack/0.17
source /home/apps/spack/share/spack/setup-env.sh
spack load python@3.8.2

# ------------------------------
# Activate main training environment
# ------------------------------
source /home/apps/iiser/pytorch-venv/bin/activate

# ------------------------------
# Extend PYTHONPATH to include dask-env site-packages
# ------------------------------
DASK_ENV_SITEPKG=~/venvs/dask-env/lib/python3.8/site-packages
export PYTHONPATH="$DASK_ENV_SITEPKG:$PYTHONPATH"

# ------------------------------
# Run training
# ------------------------------
mkdir -p logs
echo "Starting DRN_train.py at $(date)" >> logs/DRN_train.log
python DRN_train.py >> logs/DRN_train.log 2>&1
echo "Finished DRN_train.py at $(date)" >> logs/DRN_train.log

echo "SLURM job finished at $(date)"

