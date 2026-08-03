# HPC Progress Guide (SCRC / Slurm)

[TOC]



This is a quick reference for logging in, checking job status, monitoring progress, and downloading results for `pricing_marl`.
The current SCRC login host is `login.scrc.nyu.edu`. Older dated sections below retain historical `rnd.scrc.nyu.edu` commands only to document past reruns.

**Login**
1. `ssh sl9818@login.scrc.nyu.edu`

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
1. Upload from your Mac: `rsync -av pricing_marl/progress_report.py sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/`
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
1. `rsync -av sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/data/results/ /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/results/`

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

## 2026-05-12 Exp03 endogenous-K scan on the new SCRC Slurm cluster

目标：只上传运行 exp03 所需的最小文件集合，在新的 `login.scrc.nyu.edu` 集群上完成 smoke test 和正式运行。旧集群上的结果已经本地归档，因此本节不做跨集群数据迁移，只负责新集群上的 exp03 执行。

本节使用：
* `exp03_k_choice_scrc.sbatch`
* `exp03_progress_report.py`
* `requirements.txt`
* `experiments/exp03_k_choice_scan.py`
* `src_ext_K_action/`

新集群实际检查结果：
* `~/bigdata/pricing_marl` 仍存在，说明你的项目存储在新入口下可直接访问。
* `module load anaconda3/py3.9` 仍可用。
* `conda info --envs` 仍能看到 `pricing_marl` 环境。
* `pricing_marl` 环境已通过 exp03 依赖导入检查。
* 新集群没有旧脚本依赖的 `bigmem` partition 和 `turingvm` 节点。
* 当前默认分区是 `def`，其中所有节点都至少有 32 CPU；因此 `exp03_k_choice_scrc.sbatch` 请求 `32` CPU，使任务可以被 `def` 分区的全部节点接收，而不是只等待唯一的 48-CPU 节点。
* exp03 使用 `exp03_k_choice_scrc.sbatch`，而不是旧的 `exp03_k_choice_turingvm.sbatch`。

### 当前实验参数

当前默认参数来自 `experiments/exp03_k_choice_scan.py` 和 `exp03_k_choice_scrc.sbatch`：

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

### Step 0. 登录新集群并确认目录

```bash
ssh sl9818@login.scrc.nyu.edu
```

第一次 SSH 连接时，系统会要求你确认 ED25519 host key；你已经完成这一步。登录后创建 exp03 需要的目录：

```bash
hostname
readlink -f ~/bigdata
ls -lah ~/bigdata/pricing_marl | head
exit
```

如果 `~/bigdata/pricing_marl` 仍存在，就不需要重新建项目目录。



### Step 1. 从本地只上传 exp03 必需文件

在本地机器执行：

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning

rsync -av pricing_marl/requirements.txt \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/requirements.txt

rsync -av pricing_marl/experiments/exp03_k_choice_scan.py \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/experiments/exp03_k_choice_scan.py

rsync -av pricing_marl/src_ext_K_action/ \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/src_ext_K_action/

rsync -av pricing_marl/exp03_k_choice_scrc.sbatch \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/exp03_k_choice_scrc.sbatch

rsync -av pricing_marl/exp03_progress_report.py \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/exp03_progress_report.py

rsync -av pricing_marl/HPC_PROGRESS_GUIDE.md \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/HPC_PROGRESS_GUIDE.md
```

这一步不会上传本地 results、notebooks、exp02 代码或 lookup table。exp03 需要的 lookup table 会在新集群第一次运行时自动生成到 `data/lookup_tables`。

### Step 2. 在新集群验证现有 conda 环境

```bash
ssh sl9818@login.scrc.nyu.edu
cd ~/bigdata/pricing_marl

module purge
module load anaconda3/py3.9
conda info --envs
```

然后确认 `pricing_marl` 能导入 exp03 所需依赖：

```bash
conda run -n pricing_marl python - <<'PY'
import sys
import filelock, joblib, numpy, pandas, pyarrow, scipy, tqdm, zstandard
print(sys.executable)
print("exp03 python environment OK")
PY
```

### Step 3. 先做一个小规模 smoke test

推荐先跑 `4strats`、一个 `mu`、2 个随机种子，确认环境和输出都正常。

```bash
cd ~/bigdata/pricing_marl

export PRICING_MARL_EXP03_FILTER_N="3"
export PRICING_MARL_EXP03_FILTER_MU="0.04"
export PRICING_MARL_EXP03_FILTER_K="10,30,60"
export PRICING_MARL_EXP03_EXPERIMENT_SET="4strats"
export PRICING_MARL_EXP03_ROUNDS_PER_CONFIG="2"
export PRICING_MARL_EXP03_RESULTS_DIR="$HOME/bigdata/pricing_marl/data/results_exp03_smoke"

sbatch exp03_k_choice_scrc.sbatch
```

如果 `sbatch` 立刻报 partition 或资源限制错误，先在 login node 上运行：

```bash
getSlurmExamples.sh
sinfo -s
```

然后以新集群示例和 `sinfo -s` 的实际 partition 为准，调整 `exp03_k_choice_scrc.sbatch` 里的 `#SBATCH --partition=def`、`--cpus-per-task` 或 `--time`。

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

### Step 4. 提交完整 exp03

完整运行前清掉 smoke test 的环境变量，避免继承筛选条件：

```bash
cd ~/bigdata/pricing_marl

unset PRICING_MARL_EXP03_FILTER_N
unset PRICING_MARL_EXP03_FILTER_MU
unset PRICING_MARL_EXP03_FILTER_K
unset PRICING_MARL_EXP03_EXPERIMENT_SET
unset PRICING_MARL_EXP03_ROUNDS_PER_CONFIG
unset PRICING_MARL_EXP03_RESULTS_DIR

sbatch exp03_k_choice_scrc.sbatch
```

提交后记下返回的 job id。

### Step 5. 查看队列、日志和资源使用

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

### Step 6. 用 exp03_progress_report.py 看完成度

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

### Step 7. 继续或重跑

直接重新提交同一个 sbatch 即可：

```bash
cd ~/bigdata/pricing_marl
sbatch exp03_k_choice_scrc.sbatch
```

`runner.py` 会跳过已经同时存在 `run_<id>.parquet` 和 `run_<id>_qtable.parquet` 的 run，因此可以断点续跑。

如果你改了训练逻辑、evaluation 逻辑或参数，并希望强制重跑，先备份或清空目标结果目录：

```bash
cd ~/bigdata/pricing_marl
mv data/results_exp03 data/results_exp03_backup_$(date +%Y_%m_%d)
mkdir -p data/results_exp03
```

不要删除 `data/lookup_tables`，除非你明确改了 lookup table 的定义、价格 grid、策略函数或经济参数。exp03 使用 `base_K` lookup table，缓存文件在 `data/lookup_tables`。

### Step 8. 下载结果到本地

在本地机器执行：

```bash
rsync -av --progress --partial \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/data/results_exp03/ \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/results_exp03/
```

如果文件太多导致传输不稳定，可以先在服务器打包：

```bash
ssh sl9818@login.scrc.nyu.edu
cd ~/bigdata/pricing_marl/data
tar -cf results_exp03_$(date +%Y_%m_%d).tar results_exp03
exit
```

然后在本地下载单个 tar：

```bash
rsync -av --progress --partial \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/data/results_exp03_*.tar \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/
```

### Step 9. 本地验收

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

## 2026-06-07 Exp05 K=1 scan on the new SCRC Slurm cluster

目标：运行固定 `K=1` 的 exp02-style heatmap scan。这个实验使用标准 `src/` fixed-K code path，不使用 exp03/exp04 的 endogenous-K extension。结果单独写入 `data/results_exp05`，避免和 exp02 的 `data/results` 混在一起。

本节使用：
* `experiments/exp05_k_1_scan.py`
* `exp05_k_1_scan_scrc.sbatch`
* `exp05_progress_report.py`
* `src/`

重要：`exp05_k_1_scan_scrc.sbatch` 不应该在 Slurm job 内用 `conda info --envs` 自动查找环境。之前 job `4231` 卡在只输出 `PROJECT_DIR=...` 的阶段，就是因为环境发现步骤可能挂住。当前脚本直接检查已知的 `pricing_marl` Python 路径：

```text
/home/sl9818/.conda/envs/pricing_marl/bin/python
/home/sl9818/bigdata/pricing_marl/.conda/envs/pricing_marl/bin/python
/mnt/netapp_data/bighomes-active-netapp/sl9818/pricing_marl/.conda/envs/pricing_marl/bin/python
```

正常启动后，log 应很快出现 `ENV_PY=...`、`PYTHON: ...`、`EXP05 settings:` 和 `Starting exp05 K=1 scan...`。

### 当前实验参数

当前默认参数来自 `experiments/exp05_k_1_scan.py` 和 `exp05_k_1_scan_scrc.sbatch`：

* `N_VALUES = [2, 3, 5, 7, 10]`
* `MU_VALUES = [0.04, 0.07, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31]`
* `K_VALUES = [1]`
* `ROUNDS_PER_CONFIG = 30`
* `MAX_EPISODES = 2_000_000`
* `CONVERGE_PERIOD = 10_000`
* `EVAL_H = 2_000`
* `beta = 1e-5`
* `save_training_data = False`
* strategy sets:
  * `3strats = [undercut, match, above]`
  * `4strats = [undercut, match, above, undercut+reset]`

`K=1` 的含义：每次 RL action 只持续 1 个 atomic price-update step。相比 exp02 里 `K>1` 的情况，这里没有多步 action commitment，也没有跨多个 inner steps 的 reward averaging；agent 选择策略后，价格更新一次，观察一次利润，并进入这一步之后的 lowest-price state。

输出目录结构：

```text
data/results_exp05/
  scan_k1_3strats_mu0.04_k1/N_2/run_0.parquet
  scan_k1_3strats_mu0.04_k1/N_2/run_0_qtable.parquet
  scan_k1_4strats_mu0.04_k1/N_2/run_0.parquet
  ...
```

默认总规模：

* 每个 strategy set: `5 N * 10 mu * 1 K * 30 rounds = 1500` runs
* 两个 strategy sets 合计: `3000` runs
* 每个 run 产出 eval 和 qtable 两个 parquet 文件

### 参数在哪里改

最稳妥的方式是在提交任务前用环境变量改，不需要编辑 Python：

```bash
export PRICING_MARL_EXP05_FILTER_N="2,3"
export PRICING_MARL_EXP05_FILTER_MU="0.04,0.07"
export PRICING_MARL_EXP05_EXPERIMENT_SET="3strats"
export PRICING_MARL_EXP05_ROUNDS_PER_CONFIG="10"
export PRICING_MARL_EXP05_MAX_EPISODES="200000"
export PRICING_MARL_EXP05_CONVERGE_PERIOD="10000"
export PRICING_MARL_EXP05_EVAL_H="2000"
export PRICING_MARL_EXP05_RESULTS_DIR="$HOME/bigdata/pricing_marl/data/results_exp05_test"
```

常用调整：

* 只跑一个策略集：`PRICING_MARL_EXP05_EXPERIMENT_SET="3strats"` 或 `"4strats"`
* 只跑部分 `N`：`PRICING_MARL_EXP05_FILTER_N="2,3"`
* 只跑部分 `mu`：`PRICING_MARL_EXP05_FILTER_MU="0.04,0.1"`
* 小规模 smoke test：`PRICING_MARL_EXP05_ROUNDS_PER_CONFIG="2"`，再配合一个 `N`、一个 `mu`、一个 strategy set
* 改输出目录：`PRICING_MARL_EXP05_RESULTS_DIR="$HOME/bigdata/pricing_marl/data/results_exp05_smoke"`

如果要永久修改默认网格，再编辑：

```text
pricing_marl/experiments/exp05_k_1_scan.py
```

如果要改动作空间、学习率、折扣因子、经济参数或默认路径，再编辑：

```text
pricing_marl/src/config.py
```

### Step 0. 登录新集群并确认目录

```bash
ssh sl9818@login.scrc.nyu.edu
cd ~/bigdata/pricing_marl
hostname
ls -lah | head
exit
```

如果 `~/bigdata/pricing_marl` 仍存在，就不需要重新建项目目录。

### Step 1. 从本地只上传 exp05 必需文件

在本地机器执行：

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning

rsync -av pricing_marl/experiments/exp05_k_1_scan.py \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/experiments/exp05_k_1_scan.py

rsync -av pricing_marl/exp05_k_1_scan_scrc.sbatch \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/exp05_k_1_scan_scrc.sbatch

rsync -av pricing_marl/exp05_progress_report.py \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/exp05_progress_report.py

rsync -av pricing_marl/src/ \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/src/

rsync -av pricing_marl/HPC_PROGRESS_GUIDE.md \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/HPC_PROGRESS_GUIDE.md
```

这一步不会上传本地 results、notebooks 或 exp03/exp04 extension code。exp05 需要的 lookup table 会在服务器第一次运行时自动生成到 `data/lookup_tables`。

### Step 2. 在新集群验证现有 conda 环境

```bash
ssh sl9818@login.scrc.nyu.edu
cd ~/bigdata/pricing_marl

module purge
module load anaconda3/py3.9
conda info --envs
```

然后确认 `pricing_marl` 能导入 exp05 所需依赖：

```bash
conda run -n pricing_marl python - <<'PY'
import sys
import filelock, joblib, numpy, pandas, pyarrow, scipy, tqdm, zstandard
print(sys.executable)
print("exp05 python environment OK")
PY
```

如果 `conda info --envs` 输出包含多个 `pricing_marl` 路径，这是正常的。exp05 sbatch 会优先使用可执行的 `$HOME/.conda/envs/pricing_marl/bin/python`，而不是在 job 内再次调用 `conda info --envs`。

### Step 3. 先做一个小规模 smoke test

推荐先跑 `3strats`、`N=2`、`mu=0.04`、2 个随机种子，确认环境和输出都正常。

```bash
cd ~/bigdata/pricing_marl

export PRICING_MARL_EXP05_FILTER_N="2"
export PRICING_MARL_EXP05_FILTER_MU="0.04"
export PRICING_MARL_EXP05_EXPERIMENT_SET="3strats"
export PRICING_MARL_EXP05_ROUNDS_PER_CONFIG="2"
export PRICING_MARL_EXP05_RESULTS_DIR="$HOME/bigdata/pricing_marl/data/results_exp05_smoke"

sbatch exp05_k_1_scan_scrc.sbatch
```

如果 `sbatch` 立刻报 partition 或资源限制错误，先在 login node 上运行：

```bash
getSlurmExamples.sh
sinfo -s
```

然后以新集群示例和 `sinfo -s` 的实际 partition 为准，调整 `exp05_k_1_scan_scrc.sbatch` 里的 `#SBATCH --partition=def`、`--cpus-per-task` 或 `--time`。

查看：

```bash
squeue -u $USER
tail -f exp05_k1_<jobid>.out
```

如果 log 长时间只停在：

```text
PROJECT_DIR=/mnt/netapp_data/bighomes-active-netapp/sl9818/pricing_marl
```

并且 `sstat` 显示 CPU 很低，例如：

```bash
sstat -j <jobid>.batch --format=AveCPU,AveRSS,MaxRSS
```

则说明 job 还没有进入 exp05 Python 脚本，通常是 sbatch 环境定位步骤卡住。处理方式：

```bash
scancel <jobid>
```

然后从本地重新上传最新版 sbatch：

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning
rsync -av pricing_marl/exp05_k_1_scan_scrc.sbatch \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/exp05_k_1_scan_scrc.sbatch
```

再重新提交 smoke test。新版 sbatch 应该很快打印 `ENV_PY=/home/sl9818/.conda/envs/pricing_marl/bin/python` 或另一个可执行的 `pricing_marl` Python 路径。

smoke test 完成后检查：

```bash
python exp05_progress_report.py \
  --root ~/bigdata/pricing_marl/data/results_exp05_smoke \
  --rounds 2 \
  --n-values 2 \
  --mu-values 0.04 \
  --experiment-set 3strats \
  --recent 20
```

重点确认能看到：

```text
scan_k1_3strats_mu0.04_k1/N_2/run_0.parquet
scan_k1_3strats_mu0.04_k1/N_2/run_0_qtable.parquet
```

### Step 4. 提交完整 exp05

完整运行前清掉 smoke test 的环境变量，避免继承筛选条件：

```bash
cd ~/bigdata/pricing_marl

unset PRICING_MARL_EXP05_FILTER_N
unset PRICING_MARL_EXP05_FILTER_MU
unset PRICING_MARL_EXP05_EXPERIMENT_SET
unset PRICING_MARL_EXP05_ROUNDS_PER_CONFIG
unset PRICING_MARL_EXP05_RESULTS_DIR

sbatch exp05_k_1_scan_scrc.sbatch
```

提交后记下返回的 job id。

### Step 5. 查看队列、日志和资源使用

```bash
squeue -u $USER
scontrol show job <jobid> | grep -E 'StdOut|WorkDir|State|StartTime'
tail -f exp05_k1_<jobid>.out
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

### Step 6. 用 exp05_progress_report.py 看完成度

完整 exp05 默认检查：

```bash
cd ~/bigdata/pricing_marl
python exp05_progress_report.py \
  --rounds 30 \
  --grid-file experiments/exp05_k_1_scan.py \
  --recent 20
```

只看最新 qtable：

```bash
python exp05_progress_report.py \
  --rounds 30 \
  --grid-file experiments/exp05_k_1_scan.py \
  --recent 20 \
  --recent-kind qtable
```

快速 shell 计数：

```bash
find ~/bigdata/pricing_marl/data/results_exp05 -type f -name "run_*.parquet" ! -name "run_*_qtable.parquet" | wc -l
find ~/bigdata/pricing_marl/data/results_exp05 -type f -name "run_*_qtable.parquet" | wc -l
```

完整默认实验完成时，eval 文件数和 qtable 文件数都应接近或达到 `3000`。更严格地看 `exp05_progress_report.py` 里的 `paired progress`，因为一个 run 只有 eval 和 qtable 都存在才算完整。

### Step 7. 继续或重跑

直接重新提交同一个 sbatch 即可：

```bash
cd ~/bigdata/pricing_marl
sbatch exp05_k_1_scan_scrc.sbatch
```

`runner.py` 会跳过已经同时存在 `run_<id>.parquet` 和 `run_<id>_qtable.parquet` 的 run，因此可以断点续跑。

如果你改了训练逻辑、evaluation 逻辑或参数，并希望强制重跑，先备份或清空目标结果目录：

```bash
cd ~/bigdata/pricing_marl
mv data/results_exp05 data/results_exp05_backup_$(date +%Y_%m_%d)
mkdir -p data/results_exp05
```

不要删除 `data/lookup_tables`，除非你明确改了 lookup table 的定义、价格 grid、策略函数、经济参数或 `K` 的含义。exp05 的 lookup table 会包含 `K=1`。

### Step 8. 下载结果到本地

在本地机器执行：

```bash
rsync -av --progress --partial \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/data/results_exp05/ \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/results_exp05/
```

如果文件太多导致传输不稳定，可以先在服务器打包：

```bash
ssh sl9818@login.scrc.nyu.edu
cd ~/bigdata/pricing_marl/data
tar -cf results_exp05_$(date +%Y_%m_%d).tar results_exp05
exit
```

然后在本地下载单个 tar：

```bash
rsync -av --progress --partial \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/data/results_exp05_*.tar \
  /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/
```

### Step 9. 本地验收

```bash
cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl
python exp05_progress_report.py \
  --root data/results_exp05 \
  --rounds 30 \
  --grid-file experiments/exp05_k_1_scan.py \
  --recent 20
```

重点确认：

* `paired progress` 对 `3strats` 和 `4strats` 都接近或达到 `100%`
* 最近文件列表里同时能看到 `run_*.parquet` 和 `run_*_qtable.parquet`
* 目录名形如 `scan_k1_4strats_mu0.25_k1/N_10`
