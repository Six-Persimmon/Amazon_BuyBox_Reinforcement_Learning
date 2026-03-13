# HPC Progress Guide (SCRC / Slurm)

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

如果下载过程中再次出现“传一阵后卡住”的情况，优先做法仍然是把服务器端目录先打成一个 `.tar`，再下载单个大文件。

### Step 6. 本地验证
推荐先抽样查看：

* `analysis/debug_pkl_file_check.ipynb`

重点检查：
* `run_<id>_qtable.parquet` 是否包含 `init` 和 `final`
* 每个 seller 是否都有两张表
* `init` 与 `final` 是否不再完全相同
```

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
