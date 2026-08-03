#!/bin/bash
# Slurm batch script for the full heatmap scan
# This script assumes you already uploaded the `pricing_marl` folder to the server.

#SBATCH --job-name=heatmap              # Job name shown by `squeue`
#SBATCH --output=heatmap_%j.out         # Stdout/stderr log file (%j = job ID)
#SBATCH --time=4-00:00:00               # Max wall time (HH:MM:SS)
#SBATCH --mem=32G                       # Total memory for the job
#SBATCH --cpus-per-task=24              # Number of CPU cores
#SBATCH --partition=bigmem              # Queue/partition name (change if needed)
#SBATCH --export=ALL                    # Export current env vars into the job

# 1) Start from a clean environment, then load Python
module purge
module load anaconda3/py3.9

# 2) Set your project directory on the server
# Example: if you uploaded to ~/pricing_marl, keep as-is.
PROJECT_DIR="$HOME/bigdata/pricing_marl"
cd "$PROJECT_DIR" || exit 1

# 3) Validate conda environment
# conda run: always execute inside the env
conda run -n pricing_marl python - <<'PY'
import sys, pyarrow, pandas
print("PYTHON:", sys.executable)
print("PYARROW:", pyarrow.__version__)
print("PANDAS:", pandas.__version__)
PY

# 4) Control thread over-subscription from BLAS/OpenMP libs
# This avoids each process using many threads and overloading the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 5) Tell the code how many parallel workers to use
# Our Python code reads PRICING_MARL_N_JOBS (highest priority) or SLURM_CPUS_PER_TASK.
export PRICING_MARL_N_JOBS="$SLURM_CPUS_PER_TASK"

# 6) Choose where results are written
# If your cluster has a fast scratch directory, use it here.
# Example: RESULTS_DIR="/scratch/$USER/pricing_marl_results"
RESULTS_DIR="$PROJECT_DIR/data/results"
export PRICING_MARL_RESULTS_DIR="$RESULTS_DIR"

# 7) Select the experiment set
# We only run the 4-strategy set for now.
# export PRICING_MARL_EXPERIMENT_SET="4strats"

# 8) Run the experiment
conda run -n pricing_marl python -u experiments/exp02_heatmap_scan.py
