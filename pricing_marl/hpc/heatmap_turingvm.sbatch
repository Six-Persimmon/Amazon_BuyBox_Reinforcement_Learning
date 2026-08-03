#!/bin/bash
# Slurm batch script for the full heatmap scan on turingvm.

#SBATCH --job-name=heatmap_turing
#SBATCH --output=heatmap_turing_%j.out
#SBATCH --time=4-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=48
#SBATCH --partition=bigmem
#SBATCH --nodelist=turingvm
#SBATCH --export=ALL

set -euo pipefail

module purge
module load anaconda3/py3.9

# Use submit dir to avoid login/compute path mapping mismatch.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
# The sbatch scripts live in pricing_marl/hpc/; if the job was submitted from
# inside hpc/, step up to the project root.
if [ "$(basename "$PROJECT_DIR")" = "hpc" ]; then
  PROJECT_DIR="$(dirname "$PROJECT_DIR")"
fi
echo "PROJECT_DIR=$PROJECT_DIR"
cd "$PROJECT_DIR" || exit 1

# Avoid stale conda directory overrides from caller environment.
unset CONDA_ENVS_PATH CONDA_PKGS_DIRS

# Resolve env python path robustly across possible mount points.
ENV_PY=""

# 1) Try to read pricing_marl env prefix from conda metadata.
ENV_PREFIX_FROM_CONDA="$(conda info --envs 2>/dev/null | awk '$1=="pricing_marl"{print $NF; exit}')"
if [ -n "${ENV_PREFIX_FROM_CONDA:-}" ] && [ -x "$ENV_PREFIX_FROM_CONDA/bin/python" ]; then
  ENV_PY="$ENV_PREFIX_FROM_CONDA/bin/python"
fi

# 2) Fallback candidates.
if [ -z "$ENV_PY" ]; then
  for candidate in \
    "/mnt/netapp_data/bighomes-active-netapp/$USER/.conda/envs/pricing_marl/bin/python" \
    "/mnt/netapp_data/bighomes-active-netapp/$USER/bigdata/.conda/envs/pricing_marl/bin/python" \
    "$HOME/.conda/envs/pricing_marl/bin/python" \
    "$PROJECT_DIR/.conda/envs/pricing_marl/bin/python"
  do
    if [ -x "$candidate" ]; then
      ENV_PY="$candidate"
      break
    fi
  done
fi

if [ -z "$ENV_PY" ]; then
  echo "ERROR: pricing_marl env python not found on turingvm."
  echo "HOME=$HOME"
  echo "pwd -P=$(pwd -P)"
  echo "conda info --envs:"
  conda info --envs || true
  echo "Checked:"
  echo "  /mnt/netapp_data/bighomes-active-netapp/$USER/.conda/envs/pricing_marl/bin/python"
  echo "  /mnt/netapp_data/bighomes-active-netapp/$USER/bigdata/.conda/envs/pricing_marl/bin/python"
  echo "  $HOME/.conda/envs/pricing_marl/bin/python"
  echo "  $PROJECT_DIR/.conda/envs/pricing_marl/bin/python"
  exit 1
fi

echo "ENV_PY=$ENV_PY"

"$ENV_PY" - <<'PY'
import sys, pyarrow, pandas
print("PYTHON:", sys.executable)
print("PYARROW:", pyarrow.__version__)
print("PANDAS:", pandas.__version__)
PY

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PRICING_MARL_N_JOBS="$SLURM_CPUS_PER_TASK"
export PRICING_MARL_RESULTS_DIR="$PROJECT_DIR/data/results"

# Optional safety: clear stale grid filters inherited from shell.
unset PRICING_MARL_FILTER_N PRICING_MARL_FILTER_MU PRICING_MARL_FILTER_K
unset PRICING_MARL_ROUNDS_PER_CONFIG PRICING_MARL_EXPERIMENT_SET

"$ENV_PY" -u experiments/exp02_heatmap_scan.py
