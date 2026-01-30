# Parquet 数据文件详细说明

## 概述

每个实验 run 的 parquet 文件包含了该次模拟在 **Evaluation Phase（评估阶段）** 的详细数据。数据以"原子时间步"（atomic time step）为单位记录，即每一个 K-period 内的每一个小步骤都会被记录。

## 数据结构

### 1. 元数据列

| 列名 | 类型 | 含义 |
|------|------|------|
| `run_id` | int32 | 当前 run 的编号（0-29） |
| `t_global` | int32 | 全局原子时间步，从 0 开始递增 |
| `episode` | int32 | 该时间步属于 Training Phase 的哪个 episode（收敛时的 episode 编号） |
| `step_in_k` | int32 | 在当前 K-period 内的步骤编号（0 到 K-1） |
| `converged` | bool | 该 run 是否在 Training Phase 收敛了 |
| `is_cycle` | bool | 是否处于循环状态（目前是 placeholder，值为 False） |

### 2. 聚合统计列

| 列名 | 类型 | 计算方式 |
|------|------|----------|
| **`price_min`** | float32 | **当前时间步所有卖家价格的最小值** |
| **`price_mean`** | float32 | **当前时间步所有卖家价格的平均值** |
| `delta` | float32 | 平均利润相对于 Nash-Monopoly 的归一化偏离度 |

#### `price_min` 和 `price_mean` 的计算

这两个变量是在 **每个原子时间步** 独立计算的：

```python
# 在 simulation.py 的 evaluation phase
for k in range(config.K):
    p_t = prices_seq[k]   # 当前时间步每个卖家的价格数组 [p0, p1, ..., pN]
    
    # 计算该时间步的统计量
    min_p = float(np.min(p_t))      # 所有卖家中的最低价
    mean_p = float(np.mean(p_t))    # 所有卖家价格的平均值
```

**关键点：**
- `price_min` = `min(p_0, p_1, ..., p_N)` 在时间 t
- `price_mean` = `(p_0 + p_1 + ... + p_N) / N` 在时间 t
- 这两个值都是**瞬时统计量**，不是跨时间的平均

#### `delta` 的计算

Delta 是衡量合谋程度的指标：

```python
# 预计算基准利润
pi_nash = 所有卖家在 Nash 均衡下的平均利润
pi_mon = 所有卖家在 Monopoly 均衡下的平均利润

# 当前时间步的 delta
mean_pi = np.mean(pi_t)  # 当前所有卖家的平均利润
delta = (mean_pi - pi_nash) / (pi_mon - pi_nash)
```

**解释：**
- delta = 0：接近 Nash 均衡（竞争）
- delta = 1：接近 Monopoly 均衡（完全合谋）
- delta > 1：超过 Monopoly（理论上不应该出现，但可能由于数值问题）

### 3. 个体卖家数据列

对于每个卖家 `i`（i = 0, 1, ..., N-1）：

| 列名 | 类型 | 含义 |
|------|------|------|
| `a_i` | int32 | 卖家 i 选择的策略 ID（在 active_strategies 中的索引） |
| `p_i` | float32 | 卖家 i 在该时间步的价格 |
| `pi_i` | float32 | 卖家 i 在该时间步的利润 |

**策略 ID 映射：**
- 0 = Undercut（下切）
- 1 = Match（匹配）
- 2 = Above（上浮）
- 3 = Under Reset（下切后重置）
- 4 = Match Reset（匹配后重置）

**注意：** 在同一个 K-period 内（即 `step_in_k` 从 0 到 K-1），所有时间步的 `a_i` 值是**相同的**，因为策略选择在 K-period 开始时决定，然后执行 K 步。

## 为什么每个 Parquet 的观测值数量不一致？

### 当前观察结果

你的数据显示所有文件都是 **9990 行**，这看起来是一致的！让我解释为什么：

```
eval_H = 10,000     # 期望的总原子时间步数
K = 30              # 每个 K-period 的步数

num_eval_episodes = eval_H // K = 10,000 // 30 = 333 个 outer episodes
实际记录的行数 = 333 × 30 = 9,990 行
```

### 为什么不是整整 10,000 行？

因为代码中使用了**整数除法**：

```python
# simulation.py, line 78
num_eval_episodes = max(1, config.eval_H // config.K)
```

- `10,000 // 30 = 333`（整数除法，舍弃余数）
- 因此实际运行了 333 个完整的 K-period
- `333 × 30 = 9,990` 步
- 最后的 10 步被舍弃了

### 如果观测值数量真的不一致会是什么原因？

理论上，如果你在不同的实验配置中看到不同的行数，原因包括：

1. **不同的 `eval_H` 或 `K` 参数**
   - 如果不同实验使用了不同的配置文件
   
2. **Training Phase 未收敛就达到 max_episodes**
   - 但这不会影响 Evaluation Phase 的长度
   
3. **代码版本差异**
   - 如果有些实验是用旧版本代码跑的

4. **实验中断或错误**
   - 如果某些 run 在保存数据时被中断

### 验证方法

对于任何 parquet 文件，其行数应该等于：

```python
rows = (eval_H // K) * K
```

例如：
- eval_H=10,000, K=30 → 9,990 行
- eval_H=20,000, K=50 → 20,000 行（整除）
- eval_H=15,000, K=40 → 15,000 行（整除）

## 数据生成流程

### Phase 1: Training（不保存数据）

1. 初始化 Q-learning agents
2. 运行最多 `max_episodes` 个 episodes
3. 每个 episode：
   - 选择 actions（ε-greedy）
   - 环境执行 K 步（通过 lookup table）
   - 更新 Q-tables
4. 检查收敛：如果策略连续 `converge_period` 个 episodes 不变，则收敛
5. **不记录任何数据**

### Phase 2: Evaluation（记录详细数据）

1. 冻结 agents（ε=0，greedy policy）
2. 运行 `num_eval_episodes = eval_H // K` 个 episodes
3. 每个 episode：
   - 选择 greedy actions（a_0, a_1, ..., a_N）
   - **展开执行 K 步**，每一步都记录：
     - 当前所有卖家的价格 `p_i`
     - 当前所有卖家的利润 `pi_i`
     - 全局统计量 `price_min`, `price_mean`, `delta`
4. 保存为 parquet 文件

Q：每个evaluation阶段的第一个state是怎么决定的？

1. **初始状态**（第29行）：
   - Training开始时：`state = int(np.random.randint(0, config.num_grids))`
   - 这是随机的
2. **Training阶段**（第45-68行）：
   - 每个episode后，state会更新为next_state
   - 训练结束时，state保持在最后一个训练episode的next_state
3. **Evaluation阶段**（第92-145行）：
   - **直接继承Training结束时的state**
   - 不重置，不随机化
   - 从训练收敛后的最终状态继续运行

Answer：

Evaluation的第一个数据点使用的是：

- **Training阶段最后一步的next_state**
- 这反映了训练收敛后的真实状态
- 不是随机的，而是训练过程的自然延续

## 使用示例

```python
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_parquet('run_0.parquet')

# 1. 分析价格收敛
print(f"最后100步的平均最低价: {df['price_min'].tail(100).mean():.3f}")
print(f"最后100步的价格标准差: {df['price_mean'].tail(100).std():.3f}")

# 2. 分析合谋程度
print(f"平均 delta: {df['delta'].mean():.3f}")
print(f"最后1000步的 delta: {df['delta'].tail(1000).mean():.3f}")

# 3. 分析策略使用
action_cols = [c for c in df.columns if c.startswith('a_')]
# 注意：K-period 内策略相同，所以要每 K 步采样一次
actions_per_episode = df[action_cols].iloc[::30]  # 每30步采样
print("策略使用频率:")
for col in action_cols:
    print(f"  {col}: {actions_per_episode[col].value_counts()}")

# 4. 识别 equilibrium
# 取最后几个 K-periods 的策略组合
last_actions = df[action_cols].iloc[-100:].iloc[::30]
equilibrium_signature = tuple(sorted(last_actions.iloc[0].values))
print(f"Equilibrium signature: {equilibrium_signature}")
```

## 常见问题

### Q1: 为什么 `step_in_k` 会重复？

因为每个 K-period 都会从 0 计数到 K-1。要区分不同的 K-period，应该看 `t_global` 或者 `t_global // K`。

### Q2: 为什么同一个 K-period 内 actions 都相同？

这是算法设计：agents 在 outer episode 开始时选择 action，然后环境模拟执行 K 步策略交互，最后才返回 averaged rewards 用于学习。

### Q3: `episode` 列的值为什么这么大？

这是 Training Phase 收敛时的 episode 编号，表示该 run 训练了多少个 episodes 才收敛。不同的 run 会在不同的 episode 收敛（或达到 max_episodes）。

### Q4: 如何判断是否收敛到了稳定的 equilibrium？

检查最后几百步的 `price_min` 和各个 `a_i` 的变化：
- 如果价格波动很小 → 可能是 static equilibrium
- 如果价格有周期性波动 → 可能是 cyclic equilibrium
- 可以用 viz_all_runs_grid.ipynb 可视化来判断

## 相关文件

- **数据生成**: `src/simulation.py` (run_simulation 函数)
- **环境逻辑**: `src/environment.py` (step 函数)
- **配置**: `src/config.py` (Config 类)
- **实验运行**: `src/runner.py` (run_experiment_batch 函数)
- **分析脚本**: `analysis/viz_heatmap_eq_diversity_k_mu_by_n.py`
