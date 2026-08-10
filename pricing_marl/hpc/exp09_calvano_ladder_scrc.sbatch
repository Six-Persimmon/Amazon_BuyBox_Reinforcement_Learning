#!/bin/bash
# Slurm batch script for exp09 (Calvano ladder) on the current SCRC Slurm cluster.
#
# Runs all five rungs: cal_full, cal_smin, cal_arule, cal_both, cal_both_k30
# x 10 mu values x 30 seeds = 1,500 runs. Estimated ~25 min on 32 cores.
#
# The optional Arm B of Extention_Plan.md 4.5 is the same job with
#   PRICING_MARL_EXP09_BETA=1e-6
#   PRICING_MARL_EXP09_RESULTS_DIR=$PROJECT_DIR/data/results_exp09_lowbeta
# exported before sbatch. It is NOT scheduled by default.

#SBATCH --job-name=exp09_calvano
#SBATCH --output=exp09_calvano_%j.out
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
for candidate in \
  "$HOME/.conda/envs/pricing_marl/bin/python" \
  "$HOME/bigdata/pricing_marl/.conda/envs/pricing_marl/bin/python" \
  "$PROJECT_DIR/.conda/envs/pricing_marl/bin/python" \
  "/home/sl9818/.conda/envs/pricing_marl/bin/python" \
  "/home/sl9818/bigdata/pricing_marl/.conda/envs/pricing_marl/bin/python" \
  "/mnt/netapp_data/bighomes-active-netapp/sl9818/pricing_marl/.conda/envs/pricing_marl/bin/python"
do
  if [ -x "$candidate" ]; then
    ENV_PY="$candidate"
    break
  fi
done

if [ -z "$ENV_PY" ]; then
  echo "ERROR: no executable python was found for conda env 'pricing_marl'."
  echo "Checked:"
  echo "  $HOME/.conda/envs/pricing_marl/bin/python"
  echo "  $HOME/bigdata/pricing_marl/.conda/envs/pricing_marl/bin/python"
  echo "  $PROJECT_DIR/.conda/envs/pricing_marl/bin/python"
  echo "  /home/sl9818/.conda/envs/pricing_marl/bin/python"
  echo "  /home/sl9818/bigdata/pricing_marl/.conda/envs/pricing_marl/bin/python"
  echo "  /mnt/netapp_data/bighomes-active-netapp/sl9818/pricing_marl/.conda/envs/pricing_marl/bin/python"
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

# Keep other experiment environment variables from affecting this run.
unset PRICING_MARL_FILTER_N PRICING_MARL_FILTER_MU PRICING_MARL_FILTER_K
unset PRICING_MARL_ROUNDS_PER_CONFIG PRICING_MARL_EXPERIMENT_SET PRICING_MARL_RESULTS_DIR
unset PRICING_MARL_EXP03_FILTER_N PRICING_MARL_EXP03_FILTER_MU PRICING_MARL_EXP03_FILTER_K
unset PRICING_MARL_EXP03_ROUNDS_PER_CONFIG PRICING_MARL_EXP03_EXPERIMENT_SET PRICING_MARL_EXP03_RESULTS_DIR
unset PRICING_MARL_EXP04_FILTER_N PRICING_MARL_EXP04_FILTER_MU PRICING_MARL_EXP04_FILTER_K_PROFILE
unset PRICING_MARL_EXP04_ROUNDS_PER_CONFIG PRICING_MARL_EXP04_EXPERIMENT_SET PRICING_MARL_EXP04_RESULTS_DIR
unset PRICING_MARL_EXP07_FILTER_N PRICING_MARL_EXP07_FILTER_MU PRICING_MARL_EXP07_RESULTS_DIR
unset PRICING_MARL_EXP08_FILTER_N PRICING_MARL_EXP08_FILTER_MU PRICING_MARL_EXP08_RESULTS_DIR

# Exp09 defaults. Override before sbatch for smoke tests or targeted reruns.
export PRICING_MARL_EXP09_N_JOBS="${PRICING_MARL_EXP09_N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
export PRICING_MARL_EXP09_RESULTS_DIR="${PRICING_MARL_EXP09_RESULTS_DIR:-$PROJECT_DIR/data/results_exp09}"
export PRICING_MARL_EXP09_ROUNDS_PER_CONFIG="${PRICING_MARL_EXP09_ROUNDS_PER_CONFIG:-30}"
export PRICING_MARL_EXP09_MAX_EPISODES="${PRICING_MARL_EXP09_MAX_EPISODES:-5000000}"
export PRICING_MARL_EXP09_CONVERGE_PERIOD="${PRICING_MARL_EXP09_CONVERGE_PERIOD:-100000}"
export PRICING_MARL_EXP09_EVAL_H="${PRICING_MARL_EXP09_EVAL_H:-10000}"
export PRICING_MARL_EXP09_BETA="${PRICING_MARL_EXP09_BETA:-1e-5}"
export PRICING_MARL_EXP09_TIE_BREAK="${PRICING_MARL_EXP09_TIE_BREAK:-random}"

: "${PRICING_MARL_EXP09_FILTER_N:=}"
: "${PRICING_MARL_EXP09_FILTER_MU:=}"
: "${PRICING_MARL_EXP09_FILTER_CELL:=}"

if [ -n "$PRICING_MARL_EXP09_FILTER_N" ]; then
  export PRICING_MARL_EXP09_FILTER_N
else
  unset PRICING_MARL_EXP09_FILTER_N
fi

if [ -n "$PRICING_MARL_EXP09_FILTER_MU" ]; then
  export PRICING_MARL_EXP09_FILTER_MU
else
  unset PRICING_MARL_EXP09_FILTER_MU
fi

if [ -n "$PRICING_MARL_EXP09_FILTER_CELL" ]; then
  export PRICING_MARL_EXP09_FILTER_CELL
else
  unset PRICING_MARL_EXP09_FILTER_CELL
fi

mkdir -p "$PRICING_MARL_EXP09_RESULTS_DIR"

echo "EXP09 settings:"
echo "  CONDA_ENV=pricing_marl"
echo "  PRICING_MARL_EXP09_N_JOBS=$PRICING_MARL_EXP09_N_JOBS"
echo "  PRICING_MARL_EXP09_RESULTS_DIR=$PRICING_MARL_EXP09_RESULTS_DIR"
echo "  PRICING_MARL_EXP09_ROUNDS_PER_CONFIG=$PRICING_MARL_EXP09_ROUNDS_PER_CONFIG"
echo "  PRICING_MARL_EXP09_MAX_EPISODES=$PRICING_MARL_EXP09_MAX_EPISODES"
echo "  PRICING_MARL_EXP09_CONVERGE_PERIOD=$PRICING_MARL_EXP09_CONVERGE_PERIOD"
echo "  PRICING_MARL_EXP09_EVAL_H=$PRICING_MARL_EXP09_EVAL_H"
echo "  PRICING_MARL_EXP09_BETA=$PRICING_MARL_EXP09_BETA"
echo "  PRICING_MARL_EXP09_TIE_BREAK=$PRICING_MARL_EXP09_TIE_BREAK"
echo "  PRICING_MARL_EXP09_FILTER_N=${PRICING_MARL_EXP09_FILTER_N:-<unset>}"
echo "  PRICING_MARL_EXP09_FILTER_MU=${PRICING_MARL_EXP09_FILTER_MU:-<unset>}"
echo "  PRICING_MARL_EXP09_FILTER_CELL=${PRICING_MARL_EXP09_FILTER_CELL:-<unset>}"

"$ENV_PY" -u experiments/exp09_calvano_ladder.py
