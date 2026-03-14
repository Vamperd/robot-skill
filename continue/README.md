# PPO LTLf 泛化训练项目说明文档

## 1. 项目概述

本项目实现了一个基于 **PPO (Proximal Policy Optimization)** 算法的强化学习智能体，用于在二维连续空间中完成 **LTLf (Linear Temporal Logic over finite traces)** 任务序列。具体任务要求机器人按照 `Task A → Task B → Task C` 的顺序依次到达三个目标点，同时需要避开障碍物。

项目核心特点是 **域随机化 (Domain Randomization)** 与 **课程学习 (Curriculum Learning)** 的结合，训练出的智能体具有很强的 **泛化能力**，能够在未见过的障碍物和目标点布局中成功完成任务。

---

## 2. 代码文件结构

```
.
├── mutiple_train.py           # 基础环境定义与单智能体训练入口
├── generalization_train.py   # 泛化训练主程序（推荐使用）
├── transfer_eval.py          # 迁移评估脚本
├── plot_transfer_results.py  # 生成HTML可视化报告
├── record_agent.py           # 录制智能体行为GIF动画
└── test/
    ├── base_env.py            # 基础物理环境测试
    ├── reward_base_env.py    # 奖励机制测试
    └── muti_reward_base_env.py # 多任务奖励测试
```

---

## 3. 核心组件详解

### 3.1 环境定义 (`mutiple_train.py`)

#### 3.1.1 LTLfGymEnv 类

这是核心环境类，继承自 `gymnasium.Env`，封装了以下功能：

**物理系统：**
- **连续状态空间**：机器人位置 `(x, y)` 使用浮点数表示
- **连续动作空间**：神经网络输出二维向量 `[-1, 1] × [-1, 1]`，映射为速度向量
- **最大速度**：250 像素/秒，时间步长 `dt = 1/60` 秒

**观测空间 (Observation Space)：**
项目支持两种观测模式：

1. **absolute 模式** (13维/21维)
   - 归一化机器人位置 `(x/width, y/height)`
   - 归一化目标相对位置 `(dx/width, dy/height)`
   - DFA状态 `state/3.0`
   - 16束激光雷达数据

2. **relative 模式** (17维/25维) - **推荐使用**
   - 目标方向单位向量 `(dir_x, dir_y)`
   - 归一化距离 `distance / sqrt(width² + height²)`
   - 碰撞标志 `last_collision`
   - 剩余时间比例 `remaining_time`
   - DFA状态 one-hot 编码 (4维)
   - 16束激光雷达数据

#### 3.1.2 DFA (Deterministic Finite Automaton)

使用确定性有限自动机维护任务进度：

```
State 0: 等待完成 Task A
State 1: Task A 已完成，等待 Task B  
State 2: Task A,B 已完成，等待 Task C
State 3 (Accepting): 全部完成
```

#### 3.1.3 奖励机制

| 奖励类型 | 数值 | 说明 |
|---------|------|------|
| `REWARD_GOAL` | +300.0 | 完成全部三个任务 |
| `REWARD_TRANSITION` | +80.0 | 完成单个中间任务 |
| `REWARD_COLLISION` | -1.5 | 发生碰撞 |
| `REWARD_TIME_STEP` | -0.04 | 每时间步的惩罚 |
| `POTENTIAL_SCALE` | 1.0 | 势能奖励系数 |

**势能奖励**：`(previous_distance - current_distance) * POTENTIAL_SCALE`，引导机器人朝目标方向移动。

#### 3.1.4 激光雷达 (Lidar)

- **16束射线**，均匀分布在 360° 范围内
- **最大探测距离**：220 像素
- **探测方式**：Raymarching，步长 5 像素
- **归一化输出**：距离 / 220.0 ∈ [0, 1]

---

### 3.2 泛化训练 (`generalization_train.py`)

#### 3.2.1 RandomizedLTLfGymEnv 类

继承自 `LTLfGymEnv`，通过以下方式实现域随机化：

**训练布局库 (TRAIN_LAYOUT_LIBRARY)：**

包含8种不同难度的障碍物布局，每种布局有不同权重：

| 布局名称 | 权重 | 特点 |
|---------|------|------|
| baseline | 0.10 | 基准简单布局 |
| center_corridor | 0.08 | 中心走廊 |
| hard_ood_seed | 0.18 | 高难度分布外 |
| narrow_upper_gate | 0.12 | 狭窄上门 |
| u_shaped_trap | 0.15 | U形陷阱 |
| long_detour | 0.14 | 长绕行 |
| double_passage | 0.13 | 双重通道 |
| offset_maze | 0.10 | 偏移迷宫 |

**随机化策略：**

1. **布局采样**：按权重从库中随机选择布局
2. **对称变换**：随机进行 X轴/Y轴 翻转 (50%概率)
3. **位置扰动**：基于课程学习的进度，逐步增加扰动幅度

#### 3.2.2 课程学习 (Curriculum Learning)

```python
scale = min(1.0, episode_count / curriculum_episodes)
```

- **前 8000 个 episode**：扰动幅度从 0 逐渐增加到最大值
- **8000 个 episode 后**：达到最大随机化程度

**扰动参数：**
- `obstacle_shift`: 180 像素
- `task_shift`: 220 像素  
- `start_shift`: 110 像素

#### 3.2.3 验证布局 (VALIDATION_LAYOUTS)

包含4种验证场景，用于评估泛化能力：

1. `val_baseline` - 基准布局
2. `val_shift_obstacles` - 仅移动障碍物
3. `val_shift_both` - 同时移动障碍物和目标
4. `val_hard_ood` - 高难度分布外场景

---

### 3.3 PPO 训练配置

```python
model = PPO(
    "MlpPolicy",           # 多层感知机策略网络
    vec_env,               # 向量化环境 (8个并行环境)
    learning_rate=1e-4,    # 学习率
    n_steps=2048,          # 每次收集的步数
    batch_size=256,        # 批次大小
    gamma=0.99,            # 折扣因子
    gae_lambda=0.95,       # GAE 参数
    ent_coef=0.02,         # 熵系数 (鼓励探索)
    use_sde=True,          # 状态依赖的探索
    sde_sample_freq=4,    # 探索噪声采样频率
    device="cuda",         # 使用GPU训练
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256, 256],   # 策略网络: 3层256神经元
            vf=[256, 256, 256]    # 价值网络: 3层256神经元
        )
    ),
    tensorboard_log="./ppo_ltlf_tensorboard_generalization_v3/"
)
```

---

## 4. 训练流程详解

### 4.1 一步一训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        PPO 训练主循环                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. 环境交互阶段                                                 │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ for each timestep in n_steps (2048):                    │  │
│    │   - 环境.reset() → 获取初始观测                         │  │
│    │   - model.predict(obs) → 获取动作                      │  │
│    │   - 环境.step(action) → 获取 (obs, reward, done, info)  │  │
│    │   - 存储 (obs, action, reward, value, log_prob)         │  │
│    └─────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│ 2. 优势估计与价值更新                                           │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │   GAE = Σ (γλ)^t * (reward + γ*V(s') - V(s))           │  │
│    │   使用 Generalized Advantage Estimation                │  │
│    └─────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│ 3. 策略优化阶段                                                 │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │   Loss = -min(r(θ)*A, clip(r(θ),1-ε,1+ε)*A)           │  │
│    │   + 0.5*ValueLoss + 0.01*EntropyLoss                   │  │
│    │   使用 Adam 优化器更新神经网络参数                       │  │
│    └─────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│ 4. 评估阶段 (EvalCallback)                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │   每 20,000 步：                                         │  │
│    │   - 在验证布局上运行 2*4=8 个 episode                   │  │
│    │   - 计算平均成功率、回报                                 │  │
│    │   - 保存 best_model.zip (如果性能提升)                   │  │
│    └─────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│ 5. 循环直到 total_timesteps (2,500,000)                         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 泛化训练的随机化流程

每个 episode 开始时 (`env.reset()`)：

```
┌─────────────────────────────────────────────────────────────────┐
│                    单个 Episode 的初始化流程                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. 课程进度计算                                                 │
│    scale = min(1.0, episode_count / 8000)                       │
│    (8000 episodes 前，扰动幅度逐渐增加)                          │
│                              ↓                                   │
│ 2. 布局模板采样                                                 │
│    - 按权重从8种布局中随机选择                                   │
│    - 随机决定是否 X轴翻转、Y轴翻转                              │
│                              ↓                                   │
│ 3. 障碍物位置扰动                                               │
│    - 对每个障碍物：                                             │
│      dx = random(-180, 181) * scale                             │
│      dy = random(-180, 181) * scale                             │
│    - 检验与其它障碍物是否重叠                                    │
│    - 确保不超出边界                                             │
│                              ↓                                   │
│ 4. 起点位置扰动                                                 │
│    - 在扰动范围内随机生成起点                                   │
│    - 确保不与障碍物碰撞                                         │
│                              ↓                                   │
│ 5. 任务点位置扰动                                               │
│    - 对每个任务点进行扰动                                       │
│    - 确保不与障碍物碰撞                                         │
│    - 确保与起点和其他任务点保持距离                             │
│                              ↓                                   │
│ 6. 返回初始观测，Episode 开始                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 训练启动命令

```bash
cd /home/vamper/robot\ skill/continue
python generalization_train.py
```

训练过程中：
- TensorBoard 日志保存在 `./ppo_ltlf_tensorboard_generalization_v3/`
- 最佳模型保存在 `./generalization_eval_best_v3/best_model.zip`
- 评估日志保存在 `./generalization_eval_logs_v3/evaluations.npz`

---

## 5. 评估与可视化

### 5.1 迁移评估

```bash
python transfer_eval.py \
    --model-path generalization_eval_best_v3/best_model.zip \
    --render-first
```

测试场景包括：
1. `baseline_train_layout` - 训练时见过的基准布局
2. `shift_tasks_only` - 仅移动任务点位置
3. `shift_obstacles_only` - 仅移动障碍物位置
4. `shift_both_layout` - 同时移动障碍物和任务点
5. `hard_ood_layout` - 高难度分布外布局

### 5.2 生成可视化报告

```bash
python plot_transfer_results.py \
    --input transfer_eval_results.json \
    --output transfer_eval_report.html
```

生成的 HTML 报告包含：
- 成功率、平均回报等指标卡片
- 每个场景的详细结果表格
- 回报和 DFA 进度的条形图可视化

### 5.3 录制 GIF 动画

```bash
python record_agent.py \
    --model-path generalization_eval_best_v3/best_model.zip \
    --scenario hard_ood_layout \
    --gif-name agent_trajectory.gif
```

---

## 6. 泛化原理分析

### 6.1 为什么能泛化？

1. **状态表示的泛化性**
   - 使用 **相对观测**（目标方向、距离、激光雷达）而非绝对坐标
   - 智能体学习的是"如何应对当前感知到的环境状态"而非"如何执行固定路径"

2. **域随机化的覆盖**
   - 8种不同布局 + 随机扰动覆盖了大部分可能的障碍物配置
   - 对称变换进一步增强了数据多样性

3. **课程学习的渐进性**
   - 从简单场景开始，逐步增加难度
   - 智能体先学习基本技能（避障、导航），再适应复杂变化

4. **激光雷达感知的通用性**
   - 16束激光雷达提供与具体障碍物形状无关的通用感知
   - 无论障碍物如何变化，激光雷达数据格式保持一致

### 6.2 关键设计决策

| 设计 | 作用 |
|------|------|
| relative 观测模式 | 使智能体不依赖绝对位置，适应环境变换 |
| 势能奖励 | 提供稠密奖励信号，加速学习 |
| 熵系数 0.02 | 保持适度探索，避免过早收敛 |
| 课程学习 | 从易到难，确保智能体学到正确策略 |
| 验证布局评估 | 监控泛化性能，及时保存最佳模型 |

---

## 7. 依赖库

```
gymnasium
stable-baselines3
numpy
pygame
imageio
torch (stable-baselines3 依赖)
tensorboard
```

安装命令：
```bash
pip install gymnasium stable-baselines3 numpy pygame imageio
```

---

## 8. 训练技巧与注意事项

1. **GPU 训练**：代码默认使用 CUDA，若无 GPU 可改为 `device="cpu"`
2. **并行环境**：8个并行环境可显著加速采样
3. **评估频率**：每 20,000 步评估一次，避免过于频繁影响训练速度
4. **Early Stopping**：可通过监控验证成功率来提前停止训练
5. **render_mode**：`render_mode="None"` 可大幅提升训练速度

---

## 9. 总结

本项目展示了如何结合 **域随机化**、**课程学习** 和 **PPO 算法** 训练具有强泛化能力的机器人导航智能体。关键在于：

1. 设计与具体坐标无关的状态表示
2. 在训练时充分随机化环境参数
3. 使用课程学习渐进增加难度
4. 通过验证布局持续监控泛化性能

这样训练出的智能体不仅能在训练分布内完成任务，还能很好地泛化到未见过的环境配置中。
