#!/bin/bash
# Slurm batch script for exp04 fixed-K choice scan on the current SCRC Slurm cluster.

#SBATCH --job-name=exp04_fixk
#SBATCH --output=exp04_fixk_%j.out
#SBATCH --time=4-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --partition=def
#SBATCH --export=ALL

set -euo pipefail

module purge
module load anaconda3/py3.9

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
# The sbatch scripts live in pricing_marl/hpc/; if the job was submitted from
# inside hpc/, step up to the project root.
if [ "$(basename "$PROJECT_DIR")" = "hpc" ]; then
  PROJECT_DIR="$(dirname "$PROJECT_DIR")"
fi
echo "PROJECT_DIR=$PROJECT_DIR"
cd "$PROJECT_DIR" || exit 1

ENV_PY=""
ENV_PREFIX_FROM_CONDA="$(conda info --envs 2>/dev/null | awk '$1=="pricing_marl"{print $NF; exit}')"
if [ -n "${ENV_PREFIX_FROM_CONDA:-}" ] && [ -x "$ENV_PREFIX_FROM_CONDA/bin/python" ]; then
  ENV_PY="$ENV_PREFIX_FROM_CONDA/bin/python"
fi

if [ -z "$ENV_PY" ]; then
  for candidate in \
    "$HOME/.conda/envs/pricing_marl/bin/python" \
    "$HOME/bigdata/pricing_marl/.conda/envs/pricing_marl/bin/python" \
    "$PROJECT_DIR/.conda/envs/pricing_marl/bin/python"
  do
    if [ -x "$candidate" ]; then
      ENV_PY="$candidate"
      break
    fi
  done
fi

if [ -z "$ENV_PY" ]; then
  echo "ERROR: conda env 'pricing_marl' was listed, but no executable python was found."
  echo "conda info --envs:"
  conda info --envs || true
  echo "Checked:"
  echo "  $HOME/.conda/envs/pricing_marl/bin/python"
  echo "  $HOME/bigdata/pricing_marl/.conda/envs/pricing_marl/bin/python"
  echo "  $PROJECT_DIR/.conda/envs/pricing_marl/bin/python"
  exit 1
fi

echo "ENV_PY=$ENV_PY"

"$ENV_PY" - <<'PY'
import sys
import filelock
import joblib
import numpy
import pandas
import pyarrow
import scipy
import tqdm
import zstandard

print("PYTHON:", sys.executable)
print("FILELOCK:", filelock.__version__)
print("JOBLIB:", joblib.__version__)
print("NUMPY:", numpy.__version__)
print("PANDAS:", pandas.__version__)
print("PYARROW:", pyarrow.__version__)
print("SCIPY:", scipy.__version__)
print("TQDM:", tqdm.__version__)
print("ZSTANDARD:", zstandard.__version__)
PY

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Keep exp02/exp03 environment variables from affecting this run.
unset PRICING_MARL_FILTER_N PRICING_MARL_FILTER_MU PRICING_MARL_FILTER_K
unset PRICING_MARL_ROUNDS_PER_CONFIG PRICING_MARL_EXPERIMENT_SET PRICING_MARL_RESULTS_DIR
unset PRICING_MARL_EXP03_FILTER_N PRICING_MARL_EXP03_FILTER_MU PRICING_MARL_EXP03_FILTER_K
unset PRICING_MARL_EXP03_ROUNDS_PER_CONFIG PRICING_MARL_EXP03_EXPERIMENT_SET PRICING_MARL_EXP03_RESULTS_DIR

# Exp04 defaults. Override before sbatch for smoke tests or targeted reruns.
export PRICING_MARL_EXP04_N_JOBS="${PRICING_MARL_EXP04_N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
export PRICING_MARL_EXP04_RESULTS_DIR="${PRICING_MARL_EXP04_RESULTS_DIR:-$PROJECT_DIR/data/results_exp04}"
export PRICING_MARL_EXP04_BASE_K="${PRICING_MARL_EXP04_BASE_K:-10}"
export PRICING_MARL_EXP04_ROUNDS_PER_CONFIG="${PRICING_MARL_EXP04_ROUNDS_PER_CONFIG:-30}"
export PRICING_MARL_EXP04_MAX_EPISODES="${PRICING_MARL_EXP04_MAX_EPISODES:-2000000}"
export PRICING_MARL_EXP04_CONVERGE_PERIOD="${PRICING_MARL_EXP04_CONVERGE_PERIOD:-10000}"
export PRICING_MARL_EXP04_EVAL_H="${PRICING_MARL_EXP04_EVAL_H:-2000}"

: "${PRICING_MARL_EXP04_FILTER_N:=}"
: "${PRICING_MARL_EXP04_FILTER_MU:=}"
: "${PRICING_MARL_EXP04_FILTER_K_PROFILE:=}"
: "${PRICING_MARL_EXP04_EXPERIMENT_SET:=}"

if [ -n "$PRICING_MARL_EXP04_FILTER_N" ]; then
  export PRICING_MARL_EXP04_FILTER_N
else
  unset PRICING_MARL_EXP04_FILTER_N
fi

if [ -n "$PRICING_MARL_EXP04_FILTER_MU" ]; then
  export PRICING_MARL_EXP04_FILTER_MU
else
  unset PRICING_MARL_EXP04_FILTER_MU
fi

if [ -n "$PRICING_MARL_EXP04_FILTER_K_PROFILE" ]; then
  export PRICING_MARL_EXP04_FILTER_K_PROFILE
else
  unset PRICING_MARL_EXP04_FILTER_K_PROFILE
fi

if [ -n "$PRICING_MARL_EXP04_EXPERIMENT_SET" ]; then
  export PRICING_MARL_EXP04_EXPERIMENT_SET
else
  unset PRICING_MARL_EXP04_EXPERIMENT_SET
fi

mkdir -p "$PRICING_MARL_EXP04_RESULTS_DIR"

echo "EXP04 settings:"
echo "  CONDA_ENV=pricing_marl"
echo "  PRICING_MARL_EXP04_N_JOBS=$PRICING_MARL_EXP04_N_JOBS"
echo "  PRICING_MARL_EXP04_RESULTS_DIR=$PRICING_MARL_EXP04_RESULTS_DIR"
echo "  PRICING_MARL_EXP04_BASE_K=$PRICING_MARL_EXP04_BASE_K"
echo "  PRICING_MARL_EXP04_ROUNDS_PER_CONFIG=$PRICING_MARL_EXP04_ROUNDS_PER_CONFIG"
echo "  PRICING_MARL_EXP04_MAX_EPISODES=$PRICING_MARL_EXP04_MAX_EPISODES"
echo "  PRICING_MARL_EXP04_CONVERGE_PERIOD=$PRICING_MARL_EXP04_CONVERGE_PERIOD"
echo "  PRICING_MARL_EXP04_EVAL_H=$PRICING_MARL_EXP04_EVAL_H"
echo "  PRICING_MARL_EXP04_FILTER_N=${PRICING_MARL_EXP04_FILTER_N:-<unset>}"
echo "  PRICING_MARL_EXP04_FILTER_MU=${PRICING_MARL_EXP04_FILTER_MU:-<unset>}"
echo "  PRICING_MARL_EXP04_FILTER_K_PROFILE=${PRICING_MARL_EXP04_FILTER_K_PROFILE:-<unset>}"
echo "  PRICING_MARL_EXP04_EXPERIMENT_SET=${PRICING_MARL_EXP04_EXPERIMENT_SET:-<unset>}"

"$ENV_PY" -u experiments/exp04_fix_k_choice.py
