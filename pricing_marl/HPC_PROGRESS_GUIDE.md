# HPC Progress Guide (SCRC / Slurm)

[TOC]



This is a quick reference for logging in, checking job status, monitoring progress, and downloading results for `pricing_marl`.

**Login**
1. `ssh sl9818@rnd.scrc.nyu.edu`

**Go To Project**
1. `cd ~/bigdata/pricing_marl`

**Check Job Queue**
1. `squeue -u $USER`
2. If you need job details: `scontrol show job <jobid> | grep -E 'StdOut|WorkDir|State|StartTime'`

**Check If Job Is Actually Running**

1. `sstat -j <jobid>.batch --format=AveCPU,AveRSS,MaxRSS`

**Find Output Log**
1. `scontrol show job <jobid> | grep -E 'StdOut|WorkDir'`
2. Then view: `head -n 20 /path/from/StdOut` or `tail -f /path/from/StdOut`

**Monitor Progress (No Logs Needed)**
1. Count eval files: `find ~/bigdata/pricing_marl/data/results -type f -name "run_*.parquet" ! -name "run_*_qtable.parquet" | wc -l`
2. Count qtable files: `find ~/bigdata/pricing_marl/data/results -type f -name "run_*_qtable.parquet" | wc -l`
3. Recent eval files: `find ~/bigdata/pricing_marl/data/results -type f -name "run_*.parquet" ! -name "run_*_qtable.parquet" -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -n 10`
4. Auto-refresh eval count: `watch -n 30 "find ~/bigdata/pricing_marl/data/results -type f -name 'run_*.parquet' ! -name 'run_*_qtable.parquet' | wc -l"`

**Progress Report Script (recommended)**
1. Upload from your Mac: `rsync -av pricing_marl/progress_report.py sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/`
2. Run on server: `cd ~/bigdata/pricing_marl && python progress_report.py --rounds 100`
3. Optional: `python progress_report.py --rounds 100 --recent 20`
4. Optional (use true grid sizes): `python progress_report.py --rounds 100 --grid-file experiments/exp02_heatmap_scan.py`
5. Separately see 3 strat and 4 strat results: 
   1. `find ~/bigdata/pricing_marl/data/results/scan_4strats_* -type f -name "run_*.parquet" ! -name "run_*_qtable.parquet" | wc -l
      find ~/bigdata/pricing_marl/data/results/scan_3strats_* -type f -name "run_*.parquet" ! -name "run_*_qtable.parquet" | wc -l`
6. Print a summary"
   1. `cd ~/bigdata/pricing_marl
      python progress_report.py --rounds 100 --grid-file experiments/exp02_heatmap_scan.py`



**Confirm Completion**
1. `squeue -u $USER` should be empty for that job.
2. `sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode` should show `COMPLETED`.

**Cancel Jobs**
1. Cancel one job: `scancel <jobid>`
2. Cancel all your jobs: `scancel -u $USER`

**Restart / Continue**
1. Just re-submit: `sbatch ~/bigdata/pricing_marl/heatmap.sbatch`
2. The code skips one run only when both `run_<id>.parquet` and `run_<id>_qtable.parquet` already exist, so it resumes automatically.

**Download Results (from your Mac)**
1. `rsync -av sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/data/results/ /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/results/`

**Exit**
1. `exit`

## 2026-02-25 Grid-Floor 修复：服务器重跑操作清单（Step-by-step）

目标：修复低 `mu` 下 `price_grid[0] < c` 的问题后，仅重跑受影响参数；最后把服务器上的**完整新 results**重新下载到本地作为新数据集（不覆盖旧备份）。

### Step 0. 只同步关键代码（不上传本地大 results）
在本地机器执行：

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning

rsync -av pricing_marl/src/environment.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/src/environment.py

rsync -av pricing_marl/experiments/exp02_heatmap_scan.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/experiments/exp02_heatmap_scan.py

rsync -av pricing_marl/analysis/scan_grid_floor_cases.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/analysis/scan_grid_floor_cases.py
```

说明：这里不会同步 `pricing_marl/data/results`，所以不会把本地大数据传到服务器。

### Step 1. 登录并进入项目目录

```bash
ssh sl9818@rnd.scrc.nyu.edu
cd ~/bigdata/pricing_marl
```

### Step 2. 在服务器生成“受影响参数”清单

```bash
conda run -n pricing_marl python analysis/scan_grid_floor_cases.py
```

这会生成：
* `analysis/tables/grid_floor_scan_by_n_mu.csv`
* `analysis/tables/grid_floor_rerun_manifest.csv`

### Step 3. 删除受影响 cell 的旧 run 文件（必须）
如果不删，代码会跳过已存在的 `run_*.parquet`，导致不会真正重跑。

```bash
cd ~/bigdata/pricing_marl
conda run --no-capture-output -n pricing_marl python - <<'PY'
import csv
from pathlib import Path

results_root = Path("data/results")
scan_csv = Path("analysis/tables/grid_floor_scan_by_n_mu.csv")
k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
strategy_sets = ["3strats", "4strats"]

print("scan_csv exists:", scan_csv.exists())

affected = []
with scan_csv.open() as f:
    for row in csv.DictReader(f):
        if row["floor_applied"] == "True":
            affected.append((int(row["N"]), row["mu"]))

deleted = 0
for n, mu in affected:
    for k in k_values:
        for label in strategy_sets:
            run_dir = results_root / f"scan_{label}_mu{mu}_k{k}" / f"N_{n}"
            if not run_dir.exists():
                continue
            for fp in run_dir.glob("run_*.parquet"):
                fp.unlink()
                deleted += 1

print("affected cells:", len(affected))
print("deleted parquet files:", deleted)
PY
```

是否需要先 `conda activate`：不需要。上面使用了 `conda run -n pricing_marl ...`，会直接在目标环境里执行。

### Step 4. 提交仅受影响参数的重跑任务

```bash
cd ~/bigdata/pricing_marl

export PRICING_MARL_FILTER_N="2,3,5,7,10"
export PRICING_MARL_FILTER_MU="0.01,0.04,0.07,0.1"
export PRICING_MARL_FILTER_K="10,20,30,40,50,60,70,80,90,100"
unset PRICING_MARL_EXPERIMENT_SET

sbatch heatmap.sbatch
```

查看进度：

```bash
squeue -u $USER
tail -f heatmap_<jobid>.out
```

### Step 5. 本地先备份旧 results（你说的带日期后缀）
在本地机器执行（示例命名）：

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data
mv results results_2026_02_25
mkdir -p results
```

### Step 6. 把服务器完整新版 results 全量下载到本地新目录
在本地机器执行：

```bash
rsync -av \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/data/results/ \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/results/
```

这样你会同时拥有：
* `results_2026_02_25`（旧版备份）
* `results`（修复后新版完整数据）

## 2026-03-01 新实验重跑清单（含 Q-table 快照）

目标：重跑 `exp02` 新设定（`eval_H=2000`, `converge_period=100000`, 不含 `mu=0.01`），并为每个 run 同步产出：
* `run_<id>.parquet`（eval 数据）
* `run_<id>_qtable.parquet`（eval 开始前 Q-table 快照）

### Step 0. 本地先备份（建议）
在本地机器执行：

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data
mv results results_before_rerun_2026_03_01
mkdir -p results
```

说明：这样后续从服务器下载新结果时，不会覆盖你之前的数据。

### Step 1. 本地同步本次关键代码到服务器
在本地机器执行：

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning

rsync -av pricing_marl/experiments/exp02_heatmap_scan.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/experiments/exp02_heatmap_scan.py

rsync -av pricing_marl/src/simulation.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/src/simulation.py

rsync -av pricing_marl/src/runner.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/src/runner.py

rsync -av pricing_marl/progress_report.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/progress_report.py

rsync -av pricing_marl/HPC_PROGRESS_GUIDE.md \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/HPC_PROGRESS_GUIDE.md
```

### Step 2. 登录服务器并进入项目

```bash
ssh sl9818@rnd.scrc.nyu.edu
cd ~/bigdata/pricing_marl
```

### Step 3. 清理服务器旧结果（推荐做法：清空 `data/results`，保留 `data/lookup_tables`）
原因：本次重跑的变化在训练/评估流程（`eval_H`、`converge_period`、新增 qtable 输出），不影响 lookup 定义。  
因此可复用服务器上已有 `data/lookup_tables`，只清空 `data/results` 即可，最省时间。

执行前可选备份：

```bash
cd ~/bigdata/pricing_marl/data
mv results results_backup_before_2026_03_01
mkdir -p results
```

如果不需要服务器端备份，直接清空：

```bash
cd ~/bigdata/pricing_marl
rm -rf data/results/*
mkdir -p data/results
```

说明：此操作不会影响 `data/lookup_tables/*`。

### Step 4. 提交重跑任务
确保不使用旧筛选环境变量（让脚本按当前 exp02 全网格执行）：

```bash
cd ~/bigdata/pricing_marl
unset PRICING_MARL_FILTER_N
unset PRICING_MARL_FILTER_MU
unset PRICING_MARL_FILTER_K
unset PRICING_MARL_EXPERIMENT_SET

sbatch heatmap.sbatch
```

## 2026-03-12 K=30 + Q-table 重跑清单（恢复 10,000 convergence）

目标：只重跑 `K=30` 的全部参数组合，恢复旧的 `converge_period=10_000`，保留 `eval_H=2_000` 与 Q-table 输出，并把结果写入新的 `data/result_K30_qtable` 目录。

### Step 0. 本地同步这次需要更新的文件
在本地机器执行：

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning

rsync -av pricing_marl/experiments/exp02_heatmap_scan.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/experiments/exp02_heatmap_scan.py

rsync -av pricing_marl/src/simulation.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/src/simulation.py

rsync -av pricing_marl/src/runner.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/src/runner.py

rsync -av pricing_marl/progress_report.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/progress_report.py

rsync -av pricing_marl/heatmap_k30_qtable_turingvm.sbatch \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/heatmap_k30_qtable_turingvm.sbatch

rsync -av pricing_marl/README.md \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/README.md

rsync -av pricing_marl/HPC_PROGRESS_GUIDE.md \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/HPC_PROGRESS_GUIDE.md
```

### Step 1. 登录服务器并进入项目目录

```bash
ssh sl9818@rnd.scrc.nyu.edu
cd ~/bigdata/pricing_marl
```

### Step 2. 确认 K=30 rerun 目录为空
这次不要动 `data/lookup_tables`。
只清理新目录，避免和旧结果混在一起：

```bash
rm -rf data/result_K30_qtable
mkdir -p data/result_K30_qtable
```

### Step 3. 提交 K=30 专用任务

```bash
sbatch heatmap_k30_qtable_turingvm.sbatch
```

查看队列与日志：

```bash
squeue -u $USER
tail -f heatmap_k30_qtable_<jobid>.out
```

### Step 4. 用 progress_report.py 监控 K=30 项目
这次必须显式指定自定义 root，并把 `k-count` 设为 `1`：

```bash
cd ~/bigdata/pricing_marl
python progress_report.py \
  --root ~/bigdata/pricing_marl/data/result_K30_qtable \
  --rounds 100 \
  --grid-file experiments/exp02_heatmap_scan.py \
  --n-count 5 \
  --mu-count 10 \
  --k-count 1 \
  --recent 20
```

解释：
* `N=5` 是因为当前 `N_VALUES` 一共 5 个值
* `Mu=10` 是因为当前 `MU_VALUES` 一共 10 个值
* `K=1` 是因为这次只跑 `K=30` 这一种情况
* 因此 expected runs per strategy = `5 * 10 * 1 * 100 = 5000`

### Step 5. 下载新结果
在本地机器执行：

```bash
rsync -av --progress --partial \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/data/result_K30_qtable/ \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/result_K30_qtable/
```

说明：
* 这次请不要使用 `--info=progress2`。
* 你本机之前已经确认 `rsync` 版本过旧，不支持 `--info=progress2`，会直接报错。
* 因此这里统一使用：
  * `--progress`：显示每个文件的传输进度
  * `--partial`：如果下载中断，可以续传
* 这次新实验的数据在 `data/result_K30_qtable`，不是老的 `data/results`。
* 如果本地已经有同名目录，建议先改名备份，再下载新的结果。

如果下载过程中再次出现“传一阵后卡住”的情况，优先做法仍然是把服务器端目录先打成一个 `.tar`，再下载单个大文件。

服务器端打包示例：

```bash
cd ~/bigdata/pricing_marl/data
tar -cf result_K30_qtable_2026_03_12.tar result_K30_qtable
```

本地下载单个 tar 文件：

```bash
rsync -av --progress --partial \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/data/result_K30_qtable_2026_03_12.tar \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/
```

### Step 6. 本地验证
推荐先抽样查看：

* `analysis/debug_pkl_file_check.ipynb`

重点检查：
* `run_<id>_qtable.parquet` 是否包含 `init` 和 `final`
* 每个 seller 是否都有两张表
* `init` 与 `final` 是否不再完全相同

### Step 5. 监控任务（支持 eval + qtable 双进度）

```bash
squeue -u $USER
tail -f heatmap_<jobid>.out
```

推荐（新的 progress_report）：

```bash
cd ~/bigdata/pricing_marl
conda run -n pricing_marl python progress_report.py --rounds 100 --grid-file experiments/exp02_heatmap_scan.py --recent 20
```

只看最新 qtable 文件：

```bash
conda run -n pricing_marl python progress_report.py --rounds 100 --grid-file experiments/exp02_heatmap_scan.py --recent 20 --recent-kind qtable
```

快速 shell 检查：

```bash
find ~/bigdata/pricing_marl/data/results -type f -name "run_*.parquet" ! -name "run_*_qtable.parquet" | wc -l
find ~/bigdata/pricing_marl/data/results -type f -name "run_*_qtable.parquet" | wc -l
```

### Step 6. 确认完成
满足以下条件再下载：
1. `squeue -u $USER` 中该任务已结束。
2. `sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode` 显示 `COMPLETED`。
3. `progress_report.py` 中 `paired progress` 对两组策略均接近或达到 `100%`。

### Step 7. 下载服务器新结果到本地
在本地机器执行：

```bash
rsync -av \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/data/results/ \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/results/
```

如果你希望本地 `results/` 与服务器完全一致（删除本地多余文件），可用：

```bash
rsync -av --delete \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/data/results/ \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/results/
```

### Step 8. 本地验收
在本地机器执行：

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl
python progress_report.py --rounds 100 --grid-file experiments/exp02_heatmap_scan.py --recent 20
```

重点确认：
* `eval progress`、`qtable progress`、`paired progress` 都应接近或达到 `100%`
* 最近文件列表里能看到 `run_*_qtable.parquet`

## 2026-05-12 Exp03 endogenous-K scan on turing

目标：运行 `experiments/exp03_k_choice_scan.py`，让 seller 在训练中内生选择复合动作 `(pricing rule, K choice)`。结果写入新的 `data/results_exp03`，不覆盖 exp02 的 `data/results`。

本节使用：
* `exp03_k_choice_turingvm.sbatch`
* `exp03_progress_report.py`
* `experiments/exp03_k_choice_scan.py`
* `src_ext_K_action/`

### 当前实验参数

当前默认参数来自 `experiments/exp03_k_choice_scan.py` 和 `exp03_k_choice_turingvm.sbatch`：

* `N_VALUES = [3]`
* `MU_VALUES = [0.04, 0.07, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31]`
* `BASE_K = 10`
* `K_CHOICES = [10, 30, 60]`
* `ROUNDS_PER_CONFIG = 30`
* `MAX_EPISODES = 2_000_000`
* `CONVERGE_PERIOD = 10_000`
* `EVAL_H = 2_000`
* `beta = 1e-5`
* `save_training_data = False`
* strategy sets:
  * `3strats = [undercut, match, above]`
  * `4strats = [undercut, match, above, undercut+reset]`

输出目录结构：

```text
data/results_exp03/
  scan_kchoice_3strats_mu0.04_K10-30-60/N_3/run_0.parquet
  scan_kchoice_3strats_mu0.04_K10-30-60/N_3/run_0_qtable.parquet
  scan_kchoice_4strats_mu0.04_K10-30-60/N_3/run_0.parquet
  ...
```

默认总规模：

* 每个 strategy set: `1 N * 10 mu * 30 rounds = 300` runs
* 两个 strategy sets 合计: `600` runs
* 每个 run 产出 eval 和 qtable 两个 parquet 文件

### 参数在哪里改

最稳妥的方式是在提交任务前用环境变量改，不需要编辑 Python：

```bash
export PRICING_MARL_EXP03_FILTER_N="3"
export PRICING_MARL_EXP03_FILTER_MU="0.04,0.07"
export PRICING_MARL_EXP03_FILTER_K="10,30,60"
export PRICING_MARL_EXP03_EXPERIMENT_SET="4strats"
export PRICING_MARL_EXP03_ROUNDS_PER_CONFIG="10"
export PRICING_MARL_EXP03_MAX_EPISODES="200000"
export PRICING_MARL_EXP03_CONVERGE_PERIOD="10000"
export PRICING_MARL_EXP03_EVAL_H="2000"
```

常用调整：

* 只跑一个策略集：`PRICING_MARL_EXP03_EXPERIMENT_SET="3strats"` 或 `"4strats"`
* 只跑部分 `mu`：`PRICING_MARL_EXP03_FILTER_MU="0.04,0.1"`
* 只跑部分 K choices：`PRICING_MARL_EXP03_FILTER_K="10,30"`，要求每个 K 都是 `BASE_K` 的整数倍
* 小规模 smoke test：`PRICING_MARL_EXP03_ROUNDS_PER_CONFIG="2"`，再配合少量 `mu`
* 改输出目录：`PRICING_MARL_EXP03_RESULTS_DIR="$HOME/bigdata/pricing_marl/data/results_exp03_test"`

如果要永久修改默认网格，再编辑：

```text
pricing_marl/experiments/exp03_k_choice_scan.py
```

如果要改动作空间、学习率、折扣因子、经济参数或默认路径，再编辑：

```text
pricing_marl/src_ext_K_action/config.py
```

### Step 0. 本地同步 exp03 代码到服务器

在本地机器执行：

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning

rsync -av pricing_marl/experiments/exp03_k_choice_scan.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/experiments/exp03_k_choice_scan.py

rsync -av pricing_marl/src_ext_K_action/ \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/src_ext_K_action/

rsync -av pricing_marl/exp03_k_choice_turingvm.sbatch \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/exp03_k_choice_turingvm.sbatch

rsync -av pricing_marl/exp03_progress_report.py \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/exp03_progress_report.py

rsync -av pricing_marl/HPC_PROGRESS_GUIDE.md \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/HPC_PROGRESS_GUIDE.md
```

说明：这里不会上传 `data/results` 或 `data/results_exp03`，避免把本地大结果传到服务器。

### Step 1. 登录服务器并进入项目目录

```bash
ssh sl9818@rnd.scrc.nyu.edu
cd ~/bigdata/pricing_marl
```

### Step 2. 可选：先做一个小规模 smoke test

推荐先跑 `4strats`、一个 `mu`、2 个随机种子，确认环境和输出都正常。

```bash
cd ~/bigdata/pricing_marl

export PRICING_MARL_EXP03_FILTER_N="3"
export PRICING_MARL_EXP03_FILTER_MU="0.04"
export PRICING_MARL_EXP03_FILTER_K="10,30,60"
export PRICING_MARL_EXP03_EXPERIMENT_SET="4strats"
export PRICING_MARL_EXP03_ROUNDS_PER_CONFIG="2"
export PRICING_MARL_EXP03_RESULTS_DIR="$HOME/bigdata/pricing_marl/data/results_exp03_smoke"

sbatch exp03_k_choice_turingvm.sbatch
```

查看：

```bash
squeue -u $USER
tail -f exp03_kchoice_<jobid>.out
```

smoke test 完成后检查：

```bash
python exp03_progress_report.py \
  --root ~/bigdata/pricing_marl/data/results_exp03_smoke \
  --rounds 2 \
  --n-values 3 \
  --mu-values 0.04 \
  --k-values 10,30,60 \
  --experiment-set 4strats \
  --recent 20
```

### Step 3. 提交完整 exp03

完整运行前清掉 smoke test 的环境变量，避免继承筛选条件：

```bash
cd ~/bigdata/pricing_marl

unset PRICING_MARL_EXP03_FILTER_N
unset PRICING_MARL_EXP03_FILTER_MU
unset PRICING_MARL_EXP03_FILTER_K
unset PRICING_MARL_EXP03_EXPERIMENT_SET
unset PRICING_MARL_EXP03_ROUNDS_PER_CONFIG
unset PRICING_MARL_EXP03_RESULTS_DIR

sbatch exp03_k_choice_turingvm.sbatch
```

提交后记下返回的 job id。

### Step 4. 查看队列、日志和资源使用

```bash
squeue -u $USER
scontrol show job <jobid> | grep -E 'StdOut|WorkDir|State|StartTime'
tail -f exp03_kchoice_<jobid>.out
```

如果任务已经开始运行，可看 batch step 的 CPU 和内存：

```bash
sstat -j <jobid>.batch --format=AveCPU,AveRSS,MaxRSS
```

如果任务结束了：

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
```

`State` 应该是 `COMPLETED`，`ExitCode` 应该是 `0:0`。

### Step 5. 用 exp03_progress_report.py 看完成度

完整 exp03 默认检查：

```bash
cd ~/bigdata/pricing_marl
python exp03_progress_report.py \
  --rounds 30 \
  --grid-file experiments/exp03_k_choice_scan.py \
  --recent 20
```

只看最新 qtable：

```bash
python exp03_progress_report.py \
  --rounds 30 \
  --grid-file experiments/exp03_k_choice_scan.py \
  --recent 20 \
  --recent-kind qtable
```

快速 shell 计数：

```bash
find ~/bigdata/pricing_marl/data/results_exp03 -type f -name "run_*.parquet" ! -name "run_*_qtable.parquet" | wc -l
find ~/bigdata/pricing_marl/data/results_exp03 -type f -name "run_*_qtable.parquet" | wc -l
```

完整默认实验完成时，eval 文件数和 qtable 文件数都应接近或达到 `600`。更严格地看 `exp03_progress_report.py` 里的 `paired progress`，因为一个 run 只有 eval 和 qtable 都存在才算完整。

### Step 6. 继续或重跑

直接重新提交同一个 sbatch 即可：

```bash
cd ~/bigdata/pricing_marl
sbatch exp03_k_choice_turingvm.sbatch
```

`runner.py` 会跳过已经同时存在 `run_<id>.parquet` 和 `run_<id>_qtable.parquet` 的 run，因此可以断点续跑。

如果你改了训练逻辑、evaluation 逻辑或参数，并希望强制重跑，先备份或清空目标结果目录：

```bash
cd ~/bigdata/pricing_marl
mv data/results_exp03 data/results_exp03_backup_$(date +%Y_%m_%d)
mkdir -p data/results_exp03
```

不要删除 `data/lookup_tables`，除非你明确改了 lookup table 的定义、价格 grid、策略函数或经济参数。exp03 使用 `base_K` lookup table，缓存文件在 `data/lookup_tables`。

### Step 7. 下载结果到本地

在本地机器执行：

```bash
rsync -av --progress --partial \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/data/results_exp03/ \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/results_exp03/
```

如果文件太多导致传输不稳定，可以先在服务器打包：

```bash
ssh sl9818@rnd.scrc.nyu.edu
cd ~/bigdata/pricing_marl/data
tar -cf results_exp03_$(date +%Y_%m_%d).tar results_exp03
exit
```

然后在本地下载单个 tar：

```bash
rsync -av --progress --partial \
  sl9818@rnd.scrc.nyu.edu:~/bigdata/pricing_marl/data/results_exp03_*.tar \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/
```

### Step 8. 本地验收

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl
python exp03_progress_report.py \
  --root data/results_exp03 \
  --rounds 30 \
  --grid-file experiments/exp03_k_choice_scan.py \
  --recent 20
```

重点确认：

* `paired progress` 对 `3strats` 和 `4strats` 都接近或达到 `100%`
* 最近文件列表里同时能看到 `run_*.parquet` 和 `run_*_qtable.parquet`
* 目录名形如 `scan_kchoice_4strats_mu0.25_K10-30-60/N_3`
