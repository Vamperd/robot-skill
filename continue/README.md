# PPO LTLf 泛化训练项目说明文档

## 1. 项目概述

本项目实现了一个基于 PPO（Proximal Policy Optimization）的二维连续空间导航智能体。机器人需要在避障的同时，按照 `Task A -> Task B -> Task C` 的顺序依次完成 LTLf 任务点触发。

当前版本重点解决两个问题：

- 机器人在 U 型弯或死角中容易震荡、停滞，不愿意后退脱困
- 稀疏雷达和无时序记忆会让策略难以识别“前方堵死但需要撤退”的状态

为此，代码已经升级为：

- **16 束激光雷达**，提高障碍边缘感知密度
- **VecFrameStack 4 帧堆叠**，使用 Stable-Baselines3 官方时序包装器提供短期记忆
- **停滞检测 + 前方扇区阻塞检测**，辅助策略识别卡死状态
- **势能截断 + 停滞惩罚**，避免在死角里被欧氏距离势能“锁死”

---

## 2. 代码结构

```text
.
├── mutiple_train.py           # 基础环境、单场景训练与可视化测试
├── generalization_train.py    # 泛化训练主程序（推荐使用）
├── transfer_eval.py           # 模型迁移评估脚本
├── plot_transfer_results.py   # 将评估结果生成 HTML 报告
├── record_agent.py            # 录制策略回放 GIF
└── test/
    ├── base_env.py            # 基础物理环境交互测试
    ├── reward_base_env.py     # 奖励塑形测试
    └── muti_reward_base_env.py # 多任务奖励测试
```

---

## 3. 环境设计

### 3.1 连续动力学

- 机器人状态是连续位置 `(x, y)`
- 动作为二维连续速度控制 `[-1, 1] x [-1, 1]`
- 最大速度 `robot_vmax = 250.0`
- 积分步长 `dt = 1 / 60`
- 撞墙或撞障碍后会回退到上一步位置，保持不可穿透

### 3.2 DFA 任务状态机

环境内部使用一个简单 DFA 表示任务进度：

```text
State 0: 等待完成 Task A
State 1: Task A 已完成，等待 Task B
State 2: Task A、B 已完成，等待 Task C
State 3: 全部完成（accepting）
```

只有按顺序触发任务点，状态才会推进。

### 3.3 观测空间

当前环境支持两种观测模式。

#### absolute 模式（单帧 23 维）

- 归一化机器人位置 `(x / width, y / height)`
- 归一化目标偏移 `(dx / width, dy / height)`
- DFA 标量状态 `state / 3.0`
- 16 束激光雷达
- `is_stagnant`：是否停滞
- `front_blocked`：前方扇区是否被近距离障碍阻塞

#### relative 模式（单帧 27 维，推荐）

- 当前目标方向单位向量 `(dir_x, dir_y)`
- 归一化目标距离
- 碰撞标志 `last_collision`
- 剩余时间比例 `remaining_time`
- DFA one-hot 编码（4 维）
- 16 束激光雷达
- `is_stagnant`
- `front_blocked`

### 3.4 Frame Stack 后的实际输入维度

训练代码默认使用 `VecFrameStack(n_stack=4)`，因此实际送入策略网络的是 4 帧堆叠观测：

- `absolute`: `23 x 4 = 92` 维
- `relative`: `27 x 4 = 108` 维

这意味着旧版模型（8 雷达、无停滞特征、无 frame stack）与当前环境**不兼容**，需要重新训练。

---

## 4. 反死锁机制

### 4.1 停滞检测

环境内部维护一个 `collections.deque(maxlen=60)`，记录过去约 1 秒的位置历史。

- 每一步把 `(self.rx, self.ry)` 写入滑动窗口
- 当窗口满 60 帧时，计算 X/Y 坐标标准差之和
- 如果标准差之和小于阈值 `5.0`，则判定为停滞震荡

### 4.2 前方扇区阻塞检测

相比“只看正前方一束雷达”，当前版本使用前进方向附近的一个小扇区：

- 先根据动作方向（step 中）或目标方向（obs 中）找到前方中心角
- 取该方向附近 3 束雷达
- 使用这个扇区里的**最小距离**作为前方拥堵判断依据

当前默认阈值：

- `front_wall_threshold = 0.12`

### 4.3 势能截断与停滞惩罚

原始势能奖励是：

```python
potential_reward = (self.prev_distance - current_distance) * self.POTENTIAL_SCALE
```

但在 U 型弯中，真正的脱困动作往往会暂时增加目标距离，因此当前版本改成：

- 如果 `is_stagnant == True`，或 `front_blocked == True`
  - 将 `potential_reward` 强制设为 `0`
  - 追加 `REWARD_STAGNATION = -0.5`
- 否则，正常计算欧氏距离势能奖励

这样可以打断“越靠近目标越有利”的局部贪心陷阱，鼓励策略主动倒车、横移或转向脱困。

---

## 5. 奖励设计

`mutiple_train.py` 中当前主要奖励参数为：

| 奖励项 | 当前值 | 说明 |
|---|---:|---|
| `REWARD_GOAL` | `200.0` | 完成全部任务 |
| `REWARD_TRANSITION` | `50.0` | 完成中间任务 |
| `REWARD_COLLISION` | `-0.2` | 碰撞惩罚 |
| `REWARD_TIME_STEP` | `-0.1` | 时间步惩罚 |
| `POTENTIAL_SCALE` | `2.0` | 势能奖励系数 |
| `REWARD_STAGNATION` | `-0.5` | 停滞/前堵惩罚 |

这些值是当前代码中的默认值，不一定是最终最优解，但适合作为目前训练基线。

---

## 6. 泛化训练

`generalization_train.py` 定义了 `RandomizedLTLfGymEnv`，在基础环境上加入了布局随机化与课程学习。

### 6.1 随机化内容

- 从 `LAYOUT_LIBRARY` 中采样基础障碍和任务模板
- 随机进行水平/垂直镜像
- 随课程进度逐步扩大：
  - 障碍物偏移
  - 起点偏移
  - 任务点偏移

### 6.2 当前训练配置

当前主训练入口使用：

- `DummyVecEnv`：8 个并行环境
- `VecMonitor`
- `VecFrameStack(n_stack=4)`
- `PPO("MlpPolicy", ...)`

核心超参数：

```python
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
    tensorboard_log="./ppo_ltlf_tensorboard_generalization_v2/",
)
```

训练输出：

- 最佳模型目录：`generalization_eval_best/`
- 评估日志目录：`generalization_eval_logs/`
- 最终模型文件：`ppo_ltlf_agent_generalization_v2_relative.zip`

---

## 7. 安装依赖

基础依赖：

```bash
pip install gymnasium stable-baselines3 numpy pygame imageio torch tensorboard
```

如果你还要运行 `sample/phase3_single_agent_rl.py`，还需要：

```bash
pip install spot
```

---

## 8. 如何训练

### 8.1 训练基础版环境

在 `continue/` 目录下运行：

```bash
python mutiple_train.py
```

这个脚本会：

- 创建 `DummyVecEnv + VecFrameStack(4)` 的训练环境
- 训练一个基础 PPO 模型
- 保存为 `ppo_ltlf_agent.zip`
- 然后启动一个带 Pygame 窗口的回放测试

### 8.2 训练泛化版环境（推荐）

```bash
python generalization_train.py
```

这个脚本会：

- 使用 8 个并行随机化环境训练
- 自动进行周期性评估
- 将最佳模型保存到 `generalization_eval_best/best_model.zip`
- 最终导出 `ppo_ltlf_agent_generalization_v2_relative.zip`

---

## 9. 如何评估新模型

`transfer_eval.py` 已更新为支持：

- 新版单帧观测维度（23/27）
- `VecFrameStack`
- 自动从模型输入维度推断 `observation_mode` 和 `n_stack`

### 9.1 自动推断配置评估

```bash
python transfer_eval.py \
    --model-path ppo_ltlf_agent_generalization_v2_relative.zip
```

### 9.2 显式指定 relative + 4-stack

```bash
python transfer_eval.py \
    --model-path ppo_ltlf_agent_generalization_v2_relative.zip \
    --observation-mode relative \
    --n-stack 4
```

### 9.3 渲染第一个测试场景

```bash
python transfer_eval.py \
    --model-path ppo_ltlf_agent_generalization_v2_relative.zip \
    --render-first
```

### 9.4 保存 JSON 结果

```bash
python transfer_eval.py \
    --model-path ppo_ltlf_agent_generalization_v2_relative.zip \
    --save-json transfer_eval_results.json
```

输出 JSON 里会额外记录：

- `observation_mode`
- `n_stack`
- 每个场景的 `success`
- `steps`
- `total_reward`
- `final_dfa_state`

---

## 10. 如何录制 GIF 回放

`record_agent.py` 也已经适配新的观测结构和 frame stack。

### 10.1 自动推断

```bash
python record_agent.py \
    --model-path ppo_ltlf_agent_generalization_v2_relative.zip \
    --scenario hard_ood_layout \
    --gif-name agent_trajectory.gif
```

### 10.2 显式指定

```bash
python record_agent.py \
    --model-path ppo_ltlf_agent_generalization_v2_relative.zip \
    --scenario hard_ood_layout \
    --observation-mode relative \
    --n-stack 4 \
    --gif-name agent_trajectory.gif
```

---

## 11. 生成 HTML 报告

先评估生成 JSON，再执行：

```bash
python plot_transfer_results.py \
    --input transfer_eval_results.json \
    --output transfer_eval_report.html
```

HTML 报告会展示：

- 成功率
- 平均回报
- 平均最终 DFA 状态
- 每个场景的回报条形图与 DFA 进度条

---

## 12. 常用测试脚本

这些不是 pytest 测试，而是交互式脚本：

```bash
python test/base_env.py
python test/reward_base_env.py
python test/muti_reward_base_env.py
```

如果你只是想快速检查语法：

```bash
python -m py_compile *.py test/*.py
```

---

## 13. 使用建议

1. **优先训练泛化版**：`generalization_train.py` 比基础单场景训练更接近最终目标
2. **重新训练而不是复用旧模型**：当前环境维度和输入结构已经变化
3. **优先使用 relative 模式**：它对布局变化更鲁棒
4. **观察停滞惩罚是否过强**：如果机器人过于激进，可适当减小 `REWARD_STAGNATION`
5. **检查前方扇区阈值**：若过于敏感，可调 `front_wall_threshold`
6. **关注 TensorBoard**：看 reward 曲线是否出现长时间停滞

---

## 14. 当前版本总结

当前框架已经从“单帧、低密度雷达、纯欧氏势能”的 PPO 导航环境，升级为：

- 16 雷达高密度感知
- 4 帧时序堆叠
- 显式停滞/前堵状态特征
- 势能截断与停滞惩罚联动
- 自动适配新模型维度的评估与 GIF 回放脚本

这套设计更适合训练能够在 U 型弯、死胡同和复杂障碍布局中主动脱困的连续控制导航策略。

1. rollout/ (环境采样指标，衡量机器人的外部表现)这里反映的是机器人在虚拟世界里玩得好不好：
ep_rew_mean: （同上）平均回合总回报。
ep_len_mean: Episode Length Mean（平均回合存活步数）。表示机器人平均走多少步回合就会结束。在你的任务中，如果它变短且伴随 ep_rew_mean 上升，说明机器人变聪明了，学会了走捷径、用更少的时间通关；如果它极短且回报是负数，说明出门就撞死。
2. time/ (时间与进度指标，衡量系统的物理运行状态)这里反映的是你的电脑跑得快不快：
fps: Frames Per Second（每秒采样帧数）。表示你的 CPU/GPU 每秒能让环境跑多少步。这是衡量环境底层逻辑优化好坏的核心指标。iterations: 当前 PPO 算法进行到了第几次数据收集与网络更新的循环迭代。
time_elapsed: 训练已经经过的真实物理时间（秒）。total_timesteps: 训练从开始到现在，机器人在环境里总共已经交互了多少个 Step。
3. train/ (神经网络更新指标，衡量模型的内部健康度)这里反映的是 Actor 和 Critic 网络在反向传播更新时的数学状态：value_loss: 价值网络损失（Critic 网络的误差）。反映了 Critic 预测“未来能拿多少分”的准确度。如果它在训练中途突然爆炸（变成极大的数字），通常说明奖励函数（Reward Shaping）给出的信号存在不平滑的跳变，导致网络梯度崩溃。
policy_gradient_loss: 策略梯度损失（Actor 网络的误差）。通常为负数，反映了机器人动作策略优化的方向和幅度。
entropy_loss: 熵损失。用来衡量机器人输出动作的“随机性”或“探索欲”。训练初期这个负值通常绝对值较大（鼓励多试错），随着训练进行会逐渐向 0 靠拢（策略逐渐确定，不再乱跑）。
approx_kl: 近似 KL 散度。衡量 PPO 在当前这次参数更新中，新脑子（新策略）和旧脑子（旧策略）的差异有多大。如果它经常过大（例如远超 0.05），说明模型单次更新“步子迈得太大”，容易导致策略震荡。
clip_fraction: 触发截断机制的比例。PPO 算法的核心特色就是为了保证训练稳定，不允许策略单次变化过大。这个值表示有多少比例的动作优化被 PPO 强制“踩了刹车”。
explained_variance: 解释方差（取值通常在 $-\infty$ 到 1 之间）。衡量 Critic 网络对实际得分的预测能力。越接近 1 越好（说明模型完全看透了环境），如果是负数或者在 0 附近，说明 Critic 目前对得分规律一头雾水（还在瞎猜）。