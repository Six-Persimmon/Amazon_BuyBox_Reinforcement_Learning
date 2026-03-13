#!/bin/bash
# Slurm batch script for the K=30 exp02 rerun on turingvm.

#SBATCH --job-name=heatmap_k30_qtable
#SBATCH --output=heatmap_k30_qtable_%j.out
#SBATCH --time=4-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=48
#SBATCH --partition=bigmem
#SBATCH --nodelist=turingvm
#SBATCH --export=ALL

set -euo pipefail

module purge
module load anaconda3/py3.9

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
echo "PROJECT_DIR=$PROJECT_DIR"
cd "$PROJECT_DIR" || exit 1

unset CONDA_ENVS_PATH CONDA_PKGS_DIRS

ENV_PY=""

ENV_PREFIX_FROM_CONDA="$(conda info --envs 2>/dev/null | awk '$1=="pricing_marl"{print $NF; exit}')"
if [ -n "${ENV_PREFIX_FROM_CONDA:-}" ] && [ -x "$ENV_PREFIX_FROM_CONDA/bin/python" ]; then
  ENV_PY="$ENV_PREFIX_FROM_CONDA/bin/python"
fi

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
export PRICING_MARL_RESULTS_DIR="$PROJECT_DIR/data/result_K30_qtable"
export PRICING_MARL_FILTER_K="30"
export PRICING_MARL_ROUNDS_PER_CONFIG="100"

unset PRICING_MARL_FILTER_N PRICING_MARL_FILTER_MU PRICING_MARL_EXPERIMENT_SET

mkdir -p "$PRICING_MARL_RESULTS_DIR"

"$ENV_PY" -u experiments/exp02_heatmap_scan.py
