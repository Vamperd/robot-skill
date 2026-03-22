# robot-skill

这是一个分层多机器人任务执行实验仓库，当前主线已经从“底层连续运动控制”扩展到“高层调度 + A* 导航 + 底层 PPO 控制”的完整链路。

目前仓库主要包含三部分：

- `无导航纯RL底层运动器/`：训练单机器人底层连续运动控制器
- `导航结合RL运动/`：A* 全局导航 + 局部 PPO 避障 + 多机器人联调
- `协同调度/`：高层任务分配、协同任务推进、顺序调度训练与评估

当前推荐主流程是：

1. 生成 `offline_maps_v2` 数据集
2. 用基线验证数据与环境
3. 用行为克隆预热高层 attention 调度器
4. 用自定义 masked PPO 微调高层调度器
5. 在 `test` / `stress` 上评估
6. 接入 A* 与底层 PPO，做可视化联调和 GIF 导出

## 本次更新了什么

### 1. v2 场景数据集

核心文件：

- `协同调度/scenario_generator.py`
- `协同调度/build_offline_maps.py`
- `协同调度/view_offline_maps.py`

现在主训练数据使用 `offline_maps_v2`，不再为每个机器人预写 `task_sequence`。  
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

训练时输入的是“机器人池 + 任务池 + 约束 + 路径预计耗时”，而不是固定答案。

### 2. 六类 family 场景

当前固定 6 个 family：

- `open_balance`
- `role_mismatch`
- `single_bottleneck`
- `double_bottleneck`
- `far_near_trap`
- `multi_sync_cluster`

默认会按 `train / val / test / stress` 四个 split 生成。

### 3. 统一任务推进逻辑

核心文件：

- `协同调度/task_runtime.py`

当前规则：

- 单任务进入服务区后累计进度，离开后暂停，不重置
- 协同任务只有满足 `required_roles` 时才推进共享进度
- 错误角色不计入有效联盟
- 多余机器人不会阻塞完成
- 任务链约束统一使用 `tasks[*].precedence`

### 4. 事件驱动高层环境

核心文件：

- `协同调度/scheduling_env.py`

这是联合动作版本的高层调度环境，适合做基线和回归验证。  
观测中主要包含：

- `robots`
- `tasks`
- `robot_task_eta`
- `task_task_eta`
- `action_mask`

### 5. HeteroMRTA 风格顺序调度环境

核心文件：

- `协同调度/sequential_scheduling_env.py`
- `协同调度/scheduler_utils.py`

这是当前高层训练真正使用的环境封装。  
它会把一个事件点上的“多个空闲机器人同时待分配”，改造成“逐机器人顺序决策”的形式，更适合 attention pointer policy。

顺序环境额外提供：

- `current_robot`
- `current_action_mask`
- `pending_assignment_mask`
- `remaining_role_deficit`
- `precedence_state`

### 6. 高层 attention 调度器与训练脚本

核心文件：

- `协同调度/attention_policy.py`
- `协同调度/scheduler_training_utils.py`
- `协同调度/train_scheduler_bc.py`
- `协同调度/train_scheduler_rl.py`
- `协同调度/evaluate_scheduler.py`
- `协同调度/render_scheduler_episode.py`

高层模型不是用 SB3 训练，而是自定义 PyTorch 训练器：

- 第一阶段：`role_aware_greedy` 专家轨迹行为克隆预热
- 第二阶段：自定义 masked PPO 风格微调

### 7. 低层模型统一适配与正式联调入口

核心文件：

- `导航结合RL运动/low_level_policy_adapter.py`
- `导航结合RL运动/scheduler_nav_runner.py`
- `导航结合RL运动/run_scheduled_nav.py`

当前正式联调入口已经变成 `run_scheduled_nav.py`。  
它会把：

- 高层 scheduler
- A* 全局路径
- 局部 waypoint
- 底层 PPO 模型
- 统一任务 runtime

串成完整的多机器人执行链路。

`low_level_policy_adapter.py` 会自动兼容两种底层模型输入：

- `22 x 4` 的局部导航模型
- `27 x 4` 的相对观测底层运动模型

### 8. 基线与参考脚本

基线文件：

- `协同调度/baselines.py`

当前可直接比较的基线：

- `random`
- `nearest_eta_greedy`
- `role_aware_greedy`

注意：

- `导航结合RL运动/eval_multi_agent_nav.py`
- `导航结合RL运动/eval_multi_agent_nav_extreme.py`

现在更适合作为历史参考脚本或回归脚本，不再是正式主入口。

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
│   ├── low_level_policy_adapter.py
│   ├── scheduler_nav_runner.py
│   └── run_scheduled_nav.py
└── 协同调度/
    ├── scenario_generator.py
    ├── build_offline_maps.py
    ├── view_offline_maps.py
    ├── task_runtime.py
    ├── scheduling_env.py
    ├── sequential_scheduling_env.py
    ├── attention_policy.py
    ├── scheduler_training_utils.py
    ├── train_scheduler_bc.py
    ├── train_scheduler_rl.py
    ├── evaluate_scheduler.py
    ├── render_scheduler_episode.py
    └── baselines.py
```

## 安装依赖

### 只跑高层调度数据、基线、BC、RL 训练

```bash
pip install numpy torch pygame imageio
```

### 跑完整导航联调

```bash
pip install numpy torch pygame imageio gymnasium stable-baselines3
```

如果你要继续使用旧的底层训练脚本，还需要参考：

- `无导航纯RL底层运动器/README.md`

## 完整训练流程

### 第 1 步：生成 v2 数据集

正式生成：

```bash
python 协同调度/build_offline_maps.py --save-dir offline_maps_v2 --overwrite
```

调试生成小样本：

```bash
python 协同调度/build_offline_maps.py --save-dir tmp_offline_maps_v2 --overwrite --limit-per-family 1
```

生成后目录会按下面方式组织：

- `offline_maps_v2/train/...`
- `offline_maps_v2/val/...`
- `offline_maps_v2/test/...`
- `offline_maps_v2/stress/...`

并自动生成：

- `offline_maps_v2/dataset_manifest.json`

### 第 2 步：查看和抽查场景

随机查看训练集：

```bash
python 协同调度/view_offline_maps.py --cache-dir offline_maps_v2 --split train
```

只查看某个 family：

```bash
python 协同调度/view_offline_maps.py --cache-dir offline_maps_v2 --split val --family single_bottleneck
```

只弹一次窗口：

```bash
python 协同调度/view_offline_maps.py --cache-dir offline_maps_v2 --split val --family role_mismatch --single-shot
```

### 第 3 步：先跑基线，确认环境合理

```bash
python 协同调度/baselines.py --scenario-dir offline_maps_v2 --split val
```

调试时可限制回合数：

```bash
python 协同调度/baselines.py --scenario-dir tmp_offline_maps_v2 --split val --max-episodes 2
```

基线输出指标包括：

- `success_rate`
- `mean_makespan`
- `mean_completion_rate`
- `mean_wait_time`
- `mean_idle_ratio`
- `mean_deadlock_events`

### 第 4 步：行为克隆预热高层 scheduler

这一步会使用 `role_aware_greedy` 在顺序调度环境上采专家轨迹，然后训练 attention scheduler。

标准命令：

```bash
python 协同调度/train_scheduler_bc.py --scenario-dir offline_maps_v2 --train-split train --val-split val
```

常用参数：

- `--epochs`
- `--batch-size`
- `--lr`
- `--limit-train-per-family`
- `--limit-val-per-family`
- `--save-dir`
- `--device`

默认保存目录：

- `协同调度/checkpoints_bc/`

关键输出文件：

- `协同调度/checkpoints_bc/latest_scheduler_bc.pt`
- `协同调度/checkpoints_bc/best_scheduler_bc.pt`

小样本烟测示例：

```bash
python 协同调度/train_scheduler_bc.py --scenario-dir offline_maps_v2 --train-split train --val-split val --epochs 1 --limit-train-per-family 1 --limit-val-per-family 1 --batch-size 32 --save-dir tmp_scheduler_bc_smoke --device cpu
```

### 第 5 步：masked PPO 微调高层 scheduler

这一步会从 BC checkpoint 或随机初始化出发，用自定义 actor-critic / PPO 风格更新继续优化高层调度器。

标准命令：

```bash
python 协同调度/train_scheduler_rl.py --scenario-dir offline_maps_v2 --init-model 协同调度/checkpoints_bc/best_scheduler_bc.pt
```

常用参数：

- `--total-updates`
- `--rollout-steps`
- `--ppo-epochs`
- `--batch-size`
- `--eval-every`
- `--limit-train-per-family`
- `--limit-val-per-family`
- `--save-dir`
- `--device`

默认保存目录：

- `协同调度/checkpoints_rl/`

关键输出文件：

- `协同调度/checkpoints_rl/latest_scheduler_rl.pt`
- `协同调度/checkpoints_rl/best_scheduler_rl.pt`

小样本烟测示例：

```bash
python 协同调度/train_scheduler_rl.py --scenario-dir offline_maps_v2 --train-split train --val-split val --total-updates 1 --rollout-steps 64 --ppo-epochs 1 --batch-size 16 --limit-train-per-family 1 --limit-val-per-family 1 --save-dir tmp_scheduler_rl_smoke --init-model tmp_scheduler_bc_smoke/best_scheduler_bc.pt --device cpu
```

### 第 6 步：在验证集或测试集评估模型

标准评估：

```bash
python 协同调度/evaluate_scheduler.py --model 协同调度/checkpoints_rl/best_scheduler_rl.pt --split test
```

与基线一起比较：

```bash
python 协同调度/evaluate_scheduler.py --model 协同调度/checkpoints_rl/best_scheduler_rl.pt --split test --include-baselines
```

保存 JSON：

```bash
python 协同调度/evaluate_scheduler.py --model 协同调度/checkpoints_rl/best_scheduler_rl.pt --split test --include-baselines --save-json scheduler_test_metrics.json
```

调试时限制回合数：

```bash
python 协同调度/evaluate_scheduler.py --model 协同调度/checkpoints_rl/best_scheduler_rl.pt --scenario-dir offline_maps_v2 --split val --max-episodes 4
```

### 第 7 步：可视化高层调度事件

只看高层事件，不运行底层连续物理：

使用启发式策略：

```bash
python 协同调度/render_scheduler_episode.py --split val --family open_balance --policy role_aware_greedy
```

使用训练好的模型：

```bash
python 协同调度/render_scheduler_episode.py --split val --family open_balance --policy model --model 协同调度/checkpoints_rl/best_scheduler_rl.pt
```

导出 GIF：

```bash
python 协同调度/render_scheduler_episode.py --split val --family multi_sync_cluster --policy model --model 协同调度/checkpoints_rl/best_scheduler_rl.pt --gif-name scheduler_episode.gif
```

### 第 8 步：接入 A* 和底层 PPO 做完整联调

这是当前正式的端到端联调入口。

使用训练好的高层模型：

```bash
python 导航结合RL运动/run_scheduled_nav.py --scenario-dir offline_maps_v2 --split stress --limit 1 --scheduler-model 协同调度/checkpoints_rl/best_scheduler_rl.pt --low-level-model 无导航纯RL底层运动器/results/generalization_eval_best_vel_punishment_狭窄距离，效果最好，可过U形弯/best_model.zip --render --gif-name demo.gif
```

如果暂时没有高层模型，可以先用启发式调度：

```bash
python 导航结合RL运动/run_scheduled_nav.py --scenario-dir offline_maps_v2 --split stress --limit 1 --scheduler-policy role_aware_greedy --low-level-model 导航结合RL运动/results/local_eval_best/best_model.zip --render
```

直接指定单个场景：

```bash
python 导航结合RL运动/run_scheduled_nav.py --scenario-file offline_maps_v2/stress/open_balance/scenario_0000.pkl --scheduler-model 协同调度/checkpoints_rl/best_scheduler_rl.pt --low-level-model 导航结合RL运动/results/local_eval_best/best_model.zip --render
```

常用参数：

- `--scenario-file`
- `--scenario-dir`
- `--split`
- `--family`
- `--limit`
- `--scheduler-model`
- `--scheduler-policy`
- `--low-level-model`
- `--render`
- `--gif-name`
- `--max-frames`

## 训练与评估时如何使用 split

- `train`：高层调度训练主数据
- `val`：训练过程中选模型、调参数、看是否过拟合
- `test`：模型定型后做正式结果报告
- `stress`：极端场景压力测试，不建议参与训练或调参

推荐顺序：

1. 在 `train` 上做 BC + RL 训练
2. 周期性在 `val` 上评估并保存最佳 checkpoint
3. 用 `test` 做正式对比结果
4. 用 `stress` 做泛化和极端场景联调回放

## 代码中如何直接使用接口

### 场景接口

在 `协同调度/scenario_generator.py` 中可直接使用：

- `generate_scenario(...)`
- `load_random_scenario(...)`
- `load_scenarios(...)`
- `summarize_scenario(...)`
- `validate_scenario(...)`

### 联合动作调度环境

```python
from scheduling_env import SchedulingEnv

env = SchedulingEnv(scenario_dir="offline_maps_v2", split="train")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### 顺序调度环境

```python
from sequential_scheduling_env import SequentialSchedulingEnv

env = SequentialSchedulingEnv(scenario_dir="offline_maps_v2", split="train")
obs, info = env.reset()
action = 0
obs, reward, terminated, truncated, info = env.step(action)
```

### 底层模型适配器

```python
from low_level_policy_adapter import LowLevelPolicyAdapter

adapter = LowLevelPolicyAdapter.from_model("导航结合RL运动/results/local_eval_best/best_model.zip")
action = adapter.predict_action(
    robot_state=robot_state,
    waypoint=waypoint,
    obstacles=obstacles,
    neighbors=neighbors,
)
```

## 推荐工作流

推荐按下面顺序推进：

1. 用 `build_offline_maps.py` 生成 `offline_maps_v2`
2. 用 `view_offline_maps.py` 抽查 family 分布与地图可视化
3. 用 `baselines.py` 在 `val` 上确认环境行为合理
4. 跑 `train_scheduler_bc.py` 做专家预热
5. 跑 `train_scheduler_rl.py` 做高层微调
6. 跑 `evaluate_scheduler.py` 在 `val/test` 对比基线
7. 跑 `render_scheduler_episode.py` 看高层任务链决策过程
8. 跑 `run_scheduled_nav.py` 做“高层调度 + A* + 底层 PPO”的完整联调
9. 在 `stress` 上导出 GIF 做最终展示

## 当前注意事项

- 高层 scheduler 训练是自定义 PyTorch 训练器，不是 SB3。
- 完整导航联调需要 `stable-baselines3` 和 `gymnasium`。
- 如果缺少 SB3，`low_level_policy_adapter.py` 会给出明确报错：

```text
LowLevelPolicyAdapter 需要 stable-baselines3。请先执行: pip install stable-baselines3 gymnasium
```

- `协同调度/checkpoints_bc/` 和 `协同调度/checkpoints_rl/` 默认会保存训练产物，当前已经写入 `.gitignore`。
- `tmp_offline_maps_v2/`、`tmp_scheduler_bc_smoke/`、`tmp_scheduler_rl_smoke/` 更适合调试，不是正式训练目录。
- 旧的 `offline_maps/` 更适合作为回归或展示样本，不建议继续作为主训练集。
- 当前环境里可能会出现 PyTorch 与 NumPy 2.x 的兼容警告；若后续训练不稳定，建议使用匹配版本的 `torch` 与 `numpy`。

## 相关文档

- 根目录改造规划摘要：`improve.md`
- 底层连续运动控制说明：`无导航纯RL底层运动器/README.md`
