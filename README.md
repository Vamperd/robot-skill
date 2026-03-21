# robot-skill

这是一个分层机器人控制与协同调度实验仓库，当前主要包含三条主线：

- [无导航纯RL底层运动器](c:/Users/86136/Desktop/code/RL/robot-skill/无导航纯RL底层运动器/README.md)：训练底层连续运动 RL 控制器
- `导航结合RL运动/`：A* 全局导航 + PPO 底层避障的混合导航验证
- `协同调度/`：高层多机器人任务分配与协同调度

本次更新的重点是把 `协同调度` 从“演示型随机地图 + 写死任务序列”升级成“可训练高层调度器”的 v2 数据与环境体系。

## 本次更新了什么

### 1. 新版协同调度数据集 schema

核心文件：

- [scenario_generator.py](c:/Users/86136/Desktop/code/RL/robot-skill/协同调度/scenario_generator.py)
- [build_offline_maps.py](c:/Users/86136/Desktop/code/RL/robot-skill/协同调度/build_offline_maps.py)

现在生成的是 `offline_maps_v2` 风格的场景，不再为每个机器人提前写死 `task_sequence`。  
每个场景包含：

- `schema_version`
- `scenario_id`
- `split`
- `family`
- `obstacles`
- `robots`
- `tasks`
- `distance_matrix`
- `difficulty_meta`

也就是说，高层调度训练时拿到的是“机器人池 + 任务池 + 约束 + ETA”，而不是现成答案。

### 2. 场景分布从纯随机改成分族模板

当前固定 6 个 family：

- `open_balance`：开阔低冲突场景
- `role_mismatch`：最近机器人与任务角色需求不匹配
- `single_bottleneck`：单瓶颈拥堵
- `double_bottleneck`：双瓶颈交叉冲突
- `far_near_trap`：近任务诱惑与远任务先做的权衡
- `multi_sync_cluster`：多协同任务密集竞争

这样可以让训练集明确覆盖角色约束、拥堵、等待、冲突和协同资源竞争。

### 3. 可达性校验升级

旧版只粗略判断“路径是不是太短”，新版改成了 **A* 终点精确可达检查**：

- 路径最后一点必须真正到达目标
- 隐藏死区与伪可达场景会被直接拒绝
- 同时记录 `component_count`、`route_overlap_score` 等难度元数据

### 4. 新增事件驱动调度环境

核心文件：

- [scheduling_env.py](c:/Users/86136/Desktop/code/RL/robot-skill/协同调度/scheduling_env.py)

这个环境是为高层 RL 准备的，不直接做 Pygame 连续物理，而是做事件驱动调度：

- 决策时机：机器人空闲、任务完成、等待超时、阻塞超时
- 动作：给空闲机器人分配 `task_id` 或 `wait`
- 状态：
  - `robots`
  - `tasks`
  - `robot_task_eta`
  - `task_task_eta`
  - `action_mask`
- 奖励：
  - 任务完成奖励
  - 时间惩罚
  - 空转惩罚
  - 等待惩罚
  - 非法动作惩罚

### 5. 协同任务完成逻辑统一抽离

核心文件：

- [task_runtime.py](c:/Users/86136/Desktop/code/RL/robot-skill/协同调度/task_runtime.py)

现在统一采用以下规则：

- 单任务：进入服务区后累计进度，离开后暂停，不重置
- 协同任务：只有现场机器人集合满足 `required_roles` 时，才推进共享进度
- 错误角色不计入有效联盟
- 多余机器人不会阻塞任务完成

### 6. 现有极端导航评估脚本已接入统一任务逻辑

已更新：

- [eval_multi_agent_nav_extreme.py](c:/Users/86136/Desktop/code/RL/robot-skill/导航结合RL运动/eval_multi_agent_nav_extreme.py)

它现在不再使用旧版的“同名任务机器人都刷各自进度”的简化逻辑，而是复用统一 runtime。这样后续高层调度训练和最终导航联调时，任务推进语义是一致的。

### 7. 新增 3 个基线策略

核心文件：

- [baselines.py](c:/Users/86136/Desktop/code/RL/robot-skill/协同调度/baselines.py)

当前包含：

- `random`
- `nearest_eta_greedy`
- `role_aware_greedy`

它们可以直接用于 `val/test` 上做基础对比，检查高层调度环境是否合理。

### 8. 说明文档补充

我补充了：

- [improve.md](c:/Users/86136/Desktop/code/RL/robot-skill/improve.md)

如果你想快速看这次改造的摘要，也可以先读这个文件。

## 目录说明

```text
robot-skill/
├── README.md
├── improve.md
├── 无导航纯RL底层运动器/
│   └── README.md
├── 导航结合RL运动/
│   ├── a_star_planner.py
│   ├── local_rl_env.py
│   ├── eval_multi_agent_nav.py
│   └── eval_multi_agent_nav_extreme.py
└── 协同调度/
    ├── scenario_generator.py
    ├── build_offline_maps.py
    ├── view_offline_maps.py
    ├── task_runtime.py
    ├── scheduling_env.py
    └── baselines.py
```

## 具体应该如何使用

### 1. 生成新版协同调度数据集

正式生成：

```bash
python 协同调度/build_offline_maps.py --save-dir offline_maps_v2 --overwrite
```

说明：

- 默认会按 `train / val / test / stress` 生成
- 每个 split 下再按 6 个 family 分目录存放
- 会额外输出一个 `dataset_manifest.json`，记录数据集摘要

调试时建议先生成小样本：

```bash
python 协同调度/build_offline_maps.py --save-dir tmp_offline_maps_v2 --overwrite --limit-per-family 1
```

这样会每个 family 只生成 1 张图，方便快速检查流程。

### 2. 查看生成出的场景

随机查看 `train`：

```bash
python 协同调度/view_offline_maps.py --cache-dir offline_maps_v2 --split train
```

只查看某个 family：

```bash
python 协同调度/view_offline_maps.py --cache-dir offline_maps_v2 --split val --family single_bottleneck
```

如果只想打开一次预览窗口：

```bash
python 协同调度/view_offline_maps.py --cache-dir offline_maps_v2 --split val --family role_mismatch --single-shot
```

### 3. 跑基线评估

在验证集上跑 3 个基线：

```bash
python 协同调度/baselines.py --scenario-dir offline_maps_v2 --split val
```

调试时限制回合数：

```bash
python 协同调度/baselines.py --scenario-dir tmp_offline_maps_v2 --split val --max-episodes 2
```

输出指标包括：

- `success_rate`
- `mean_makespan`
- `mean_completion_rate`
- `mean_wait_time`
- `mean_idle_ratio`
- `mean_deadlock_events`

### 4. 在代码里直接使用场景生成与加载接口

场景接口在 [scenario_generator.py](c:/Users/86136/Desktop/code/RL/robot-skill/协同调度/scenario_generator.py) 里：

- `generate_scenario(...)`
- `load_random_scenario(...)`
- `load_scenarios(...)`
- `summarize_scenario(...)`
- `validate_scenario(...)`

适合：

- 训练前抽样检查
- 评估脚本直接读取数据
- 单场景分析

### 5. 在代码里直接使用高层调度环境

示例：

```python
from scheduling_env import SchedulingEnv

env = SchedulingEnv(scenario_dir="offline_maps_v2", split="train")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

观测结构：

- `obs["robots"]`
- `obs["tasks"]`
- `obs["robot_task_eta"]`
- `obs["task_task_eta"]`
- `obs["action_mask"]`

适合下一步直接接到高层 RL 训练脚本中。

### 6. 用统一任务逻辑做最终导航联调

如果你要验证“高层调度 + A* + 底层 PPO”的整套链路，可以使用：

```bash
python 导航结合RL运动/eval_multi_agent_nav_extreme.py
```

这个脚本现在已经接入统一的任务推进 runtime，协同任务会按照角色约束和共享进度来完成。

## 推荐工作流

建议按下面顺序推进：

1. 先生成一个小样本 `tmp_offline_maps_v2`
2. 用 `view_offline_maps.py` 肉眼检查 6 类 family 是否符合预期
3. 用 `baselines.py` 跑 `val`，确认指标和行为合理
4. 再把 `SchedulingEnv` 接到高层 RL 训练脚本
5. 高层策略训练完后，再接回 `eval_multi_agent_nav_extreme.py` 做联调

## 当前注意事项

- 这台环境里没有安装 `gymnasium`，所以 `SchedulingEnv` 里做了轻量回退，方便先跑基线和调试。
- 如果你接下来要用 Stable-Baselines3 训练高层 RL，建议先安装：

```bash
pip install gymnasium
```

- 旧的 `offline_maps` 还保留着，但现在更适合作为回归样本或可视化样本，不建议继续作为主训练集。
- `tmp_offline_maps_v2/` 是调试用的小样本目录，不是正式训练集。

## 相关文档

- 根目录说明：`README.md`
- 协同调度改造摘要：[improve.md](c:/Users/86136/Desktop/code/RL/robot-skill/improve.md)
- 底层运动器详细说明：[无导航纯RL底层运动器/README.md](c:/Users/86136/Desktop/code/RL/robot-skill/无导航纯RL底层运动器/README.md)
