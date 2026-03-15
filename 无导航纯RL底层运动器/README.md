# 无导航纯RL底层运动器

## 项目概述

这个目录实现了一个基于 PPO 的二维连续控制强化学习环境。智能体需要在连续平面中避障，并严格按照 `Task A -> Task B -> Task C` 的顺序完成任务点触发。

当前代码的核心目标不是传统全局导航，而是训练一个仅依赖局部观测的底层运动策略，使其在复杂障碍、狭窄通道、U 形弯和死角中仍能持续推进任务。

目前环境的关键特性包括：

- 16 束激光雷达观测
- `absolute` / `relative` 两种观测模式
- `VecFrameStack(n_stack=4)` 提供短时记忆
- 停滞检测与前方扇区阻塞检测
- 仅对接近当前目标的真实进展给势能奖励，卡死时追加惩罚
- 泛化训练中的布局随机化、翻转和课程学习

## 目录结构

```text
无导航纯RL底层运动器/
├── mutiple_train.py            # 基础环境定义 + 单场景训练脚本
├── generalization_train.py     # 泛化训练主脚本
├── finetune_train.py           # 在已有泛化模型上继续微调
├── transfer_eval.py            # 迁移评估，支持自动推断观测模式与 frame stack
├── record_agent.py             # 录制指定场景的 GIF 回放
├── plot_training_curves.py     # 绘制 reward / success 两列训练曲线
├── plot_comprehensive_curves.py # 绘制 reward / success / episode length 三列曲线
└── result/                     # 已保存的实验模型与结果目录
```

## 环境设计

### 连续动力学

- 地图大小：`800 x 600`
- 机器人半径：`15`
- 动作空间：二维连续速度 `Box([-1, -1], [1, 1])`
- 最大速度：`250.0`
- 积分步长：`dt = 1 / 60`
- 遇到边界或障碍物时按 X/Y 轴分别回退，并带有 `0.8` 的反弹系数

### DFA 任务状态机

环境内部用一个简单 DFA 跟踪任务进度：

```text
State 0: 等待 Task A
State 1: 已完成 Task A，等待 Task B
State 2: 已完成 Task A/B，等待 Task C
State 3: 全部完成（accepting）
```

只有按顺序触发任务点，状态才会推进。

### 观测空间

`mutiple_train.py` 中支持两种观测模式。

#### `absolute`（单帧 23 维）

- 归一化机器人位置：`x / width`, `y / height`
- 当前目标相对位移：`dx / width`, `dy / height`
- DFA 标量状态：`state / 3.0`
- 16 束归一化雷达距离
- `is_stagnant`
- `front_blocked`

#### `relative`（单帧 27 维）

- 当前目标方向单位向量：`dir_x`, `dir_y`
- 归一化目标距离
- 上一步碰撞标志：`last_collision`
- 剩余时间比例：`remaining_time`
- DFA one-hot（4 维）
- 16 束归一化雷达距离
- `is_stagnant`
- `front_blocked`

#### Frame stack 后的实际输入

训练脚本默认都使用 `VecFrameStack(n_stack=4)`：

- `absolute`: `23 x 4 = 92`
- `relative`: `27 x 4 = 108`

`transfer_eval.py` 和 `record_agent.py` 会从模型输入维度自动推断 `observation_mode` 与 `n_stack`，也可以通过 CLI 显式指定。

## 反卡死机制与奖励

### 停滞检测

- 位置历史保存在 `deque(maxlen=60)`
- 当窗口填满后，计算最近轨迹的 `x/y` 标准差之和
- 若标准差和小于 `5.0`，判定为停滞

### 前方扇区阻塞检测

- 根据动作方向或目标方向计算前方中心角
- 在 16 束雷达中取该方向附近 3 束
- 用扇区最小距离判断是否堵塞
- 当前阈值：`front_wall_threshold = 0.25`

### 速度方向危险惩罚

`step()` 中额外计算一个基于速度投影的危险惩罚：

- 仅在雷达距离小于 `0.15` 的方向上生效
- 如果当前速度正朝着障碍物方向推进，则追加惩罚
- 这一项用于减少“明知前方近墙还持续顶上去”的行为

### 当前默认奖励参数

`mutiple_train.py` 默认值如下：

| 奖励项 | 数值 | 作用 |
| --- | ---: | --- |
| `REWARD_GOAL` | `200.0` | 完成全部任务 |
| `REWARD_TRANSITION` | `50.0` | 完成中间任务 |
| `REWARD_COLLISION` | `-10.0` | 碰撞惩罚 |
| `REWARD_TIME_STEP` | `-0.1` | 每步时间惩罚 |
| `POTENTIAL_SCALE` | `1.0` | 仅对更接近目标的真实进展给奖励 |
| `REWARD_STAGNATION` | `-0.5` | 停滞或前方堵塞时惩罚 |

奖励逻辑的关键点：

- 只有当 `current_distance < closest_distance` 时才给势能奖励
- 一旦检测到停滞或前方堵塞，则该步势能奖励置零，并追加停滞惩罚
- 到达中间任务点给 `REWARD_TRANSITION`
- 完成最终任务点给 `REWARD_GOAL` 并结束回合

## 训练脚本

### 1. 基础单场景训练：`mutiple_train.py`

特点：

- 使用基础固定布局
- 默认观测模式为 `absolute`
- 单环境 `DummyVecEnv`
- 4 帧堆叠
- 训练 `150_000` 步
- 保存模型为 `ppo_ltlf_agent.zip`
- 训练完成后会自动打开 Pygame 窗口做回放测试

运行方式：

```bash
python mutiple_train.py
```

### 2. 泛化训练：`generalization_train.py`

特点：

- 环境类：`RandomizedLTLfGymEnv`
- 固定使用 `relative` 观测
- 从 `LAYOUT_LIBRARY` 随机采样基础模板
- 随机做水平/垂直翻转
- 对障碍物、起点、任务点施加课程式扰动
- 8 个并行环境 + `VecMonitor` + 4 帧堆叠
- 训练 `1_500_000` 步

当前 PPO 配置：

```python
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.03,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
    tensorboard_log="./ppo_ltlf_tensorboard_generalization_v2/",
)
```

此外还包含一个 `EntropyDecayCallback`，会把熵系数从 `0.03` 线性衰减到 `0.0001`。

输出目录：

- 最佳模型：`generalization_eval_best_vel_punishment/`
- 评估日志：`generalization_eval__vel_punishment_logs/`
- 中断或结束时的最新模型：`ppo_ltlf_agent_generalization_v2_relative_latest.zip`

运行方式：

```bash
python generalization_train.py
```

### 3. 微调训练：`finetune_train.py`

特点：

- 继续使用与泛化训练相同的随机化环境
- 默认从 `./generalization_eval_best_vel_punishment/best_model.zip` 加载基座模型
- 将学习率降到 `5e-5`
- `ent_coef=0.0005`
- `clip_range=0.1`
- 默认继续训练 `500_000` 步

输出目录：

- 最佳模型：`finetune_eval_best/`
- 评估日志：`finetune_eval_logs/`
- 最新模型：`ppo_ltlf_agent_finetuned_latest.zip`

运行方式：

```bash
python finetune_train.py
```

## 评估与回放

### 迁移评估：`transfer_eval.py`

脚本内置 7 个评估场景，包括：

- `baseline_train_layout`
- `shift_tasks_only`
- `shift_obstacles_only`
- `shift_both_layout`
- `hard_ood_layout`
- `hard_very_thin_corridor_layout`
- `ultimate_generalization_test`

评估输出内容包括：

- `success_rate`
- `avg_reward`
- `avg_final_dfa_state`
- 每个场景的 `success / steps / total_reward / final_dfa_state`

常用命令：

```bash
python transfer_eval.py
python transfer_eval.py --render-first
python transfer_eval.py --save-json transfer_eval_results.json
python transfer_eval.py --model-path path/to/model.zip --observation-mode relative --n-stack 4
```

### GIF 录制：`record_agent.py`

常用命令：

```bash
python record_agent.py
python record_agent.py --scenario hard_ood_layout --gif-name agent_trajectory.gif
python record_agent.py --model-path path/to/model.zip --observation-mode relative --n-stack 4
```

默认模型路径与评估脚本一致：`generalization_eval_best_vel_punishment/best_model.zip`。

## 训练曲线绘制

两个绘图脚本默认读取：

`./generalization_eval__vel_punishment_logs/evaluations.npz`

### `plot_training_curves.py`

- 绘制平均 reward
- 若 `successes` 存在，则绘制平均成功率
- 输出文件：`training_curves.png`

```bash
python plot_training_curves.py
```

### `plot_comprehensive_curves.py`

- 绘制平均 reward
- 绘制平均成功率
- 绘制平均 episode length
- 输出文件：`comprehensive_training_curves.png`

```bash
python plot_comprehensive_curves.py
```

## 依赖安装

```bash
pip install gymnasium stable-baselines3 numpy pygame imageio torch tensorboard matplotlib
```

## 轻量验证

这个仓库当前没有 pytest 测试集，最安全的快速验证方式是：

```bash
python -m py_compile *.py
python transfer_eval.py --help
python record_agent.py --help
```

如果你刚修改了训练逻辑，还可以按需运行：

```bash
python mutiple_train.py
python generalization_train.py
python finetune_train.py
```

## 使用建议

1. 优先使用 `generalization_train.py`，它更接近当前主线方案。
2. 训练和评估时尽量使用 `relative + n_stack=4`，这也是泛化脚本的默认配置。
3. 如果改动了观测结构，记得同步更新 `transfer_eval.py` 里的 `OBSERVATION_DIMS`。
4. 如果改动了奖励或环境状态，至少重新检查停滞检测、前方扇区检测、渲染和 DFA 推进是否仍一致。
