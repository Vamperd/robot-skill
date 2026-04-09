# 论文写作支撑文档：从 pointer-BC 到 hetero_ranker 主线

本文档用于沉淀当前仓库里高层协同调度方法的真实演化过程、关键正负结果、工程约束和论文可直接复用的结论。它不承担命令手册职责；具体可执行流程以 `README.md` 为准。

## 1. 问题设定与接入约束

当前系统真正必须冻结的不是内部模型结构，而是接入契约：
- 场景仍来自 `offline_maps_v2`
- 高层运行时输入仍是：
  - `agent_inputs`
  - `task_inputs`
  - `global_mask`
  - `current_agent_index`
- 高层输出仍是离散动作：
  - `0 = wait`
  - `1..N = task_id`
- 联调链路仍是：
  - 高层调度
  - A*
  - 底层 PPO
  - `service_ready / task_runtime`

因此，当前方法演化的核心思想是：
- 冻结外部接口
- 允许彻底重构内部训练目标和模型结构

这正是从 pointer-BC 转向 `hetero_ranker` 主线的根本前提。

## 2. 实际程序完整流程

### 2.1 数据生成

数据生成入口是：
- `协同调度/build_offline_maps.py`
- `协同调度/scenario_generator.py`

主数据集是 `offline_maps_v2`。每个场景提供：
- 机器人配置
- 任务集合
- precedence 约束
- 障碍物信息
- 距离矩阵
- family 标签

当前 family 包括：
- `open_balance`
- `role_mismatch`
- `single_bottleneck`
- `double_bottleneck`
- `far_near_trap`
- `multi_sync_cluster`
- `partial_coalition_trap`

### 2.2 高层环境与运行语义

高层主环境是：
- `协同调度/hetero_dispatch_env.py`

底层状态推进与任务完成逻辑继续复用：
- `协同调度/scheduling_env.py`
- `协同调度/task_runtime.py`
- `协同调度/coop_docking.py`

当前高层训练的核心语义是：
- 每次只给当前释放出来的机器人决策
- 决策目标不是构造一整条路径，而是比较“此刻最值得做哪个动作”
- `wait` 被当作动作 token，而不是独立控制头

### 2.3 `wait` 语义修复后的统一规则

当前训练、评估、联调都共享：
- `legal_action_mask_for_robot(...)`
- `constrain_wait_action_mask(...)`
- `fallback_legal_action_from_mask(...)`

统一规则是：
- 若机器人处于 `idle` 或 `waiting_idle`
- 且存在合法非 `wait` 动作
- 则 `wait` 会被 mask 掉

这项修复解决了早期“原地等待 / 任务跳变 / waiting_idle 反复续等”的工程漏洞，使后续所有实验至少站在一致的动作语义上。

### 2.4 训练、评估、联调分工

当前正式入口如下：
- `协同调度/train_scheduler_hetero_bc.py`
  - 教师诊断 / 历史 pointer-BC 对照 / 推荐 warm start
- `协同调度/train_scheduler_hetero_ranker.py`
  - 当前正式主线
- `协同调度/evaluate_scheduler.py`
  - 正式评估入口
- `导航结合RL运动/run_scheduled_nav.py`
  - 联调入口
- `导航结合RL运动/scheduler_nav_runner.py`
  - 运行时加载、动作桥接与 guard

## 3. 方法演化时间线

### 3.1 第一阶段：legacy PPO

早期主线是 attention + PPO，对应：
- `协同调度/train_scheduler_bc.py`
- `协同调度/train_scheduler_rl.py`

这条线的价值在于：
- 提供了早期可用正结果
- 证明高层调度可以通过学习方法优于简单启发式
- 留下后续 Hetero 线路的重要对照基线

### 3.2 第二阶段：improved / staged PPO

为了抑制 PPO 坍缩，仓库引入了：
- `waiting_sync`
- 奖励重构
- staged curriculum
- 一系列防坍缩工程

对应脚本：
- `协同调度/train_scheduler_hetero_ppo.py`

这条线的重要性主要在工程证据，而不在成为最终主线。它解释了为什么 actor-critic 在当前组合派单问题上很难稳定超过强初始化。

### 3.3 第三阶段：hetero_bc

`hetero_bc` 的初衷是：
- 用 Hetero 输入四件套替代旧版观测
- 用行为克隆快速获得更强的高层初始化

当前脚本：
- `协同调度/train_scheduler_hetero_bc.py`

它支持：
- 自动比较三类教师
- 输出 `teacher_val_reference.json`
- 输出 `teacher_agreement`
- 输出 `hard_teacher_agreement`
- 输出 `single_vs_sync_conflict_agreement`
- 输出 trap / family 指标

但实验结果表明它长期卡住。

### 3.4 第四阶段：hetero actor-only / hetero_ppo

随后仓库尝试把 Hetero actor 主干继续用于：
- `hetero_actor_only`
- `hetero_ppo`

其中：
- `hetero_ppo` 是 actor-critic 路线
- `hetero_actor_only` 是 self-critical + group baseline 路线

两者都保留了工程价值，但在当前阶段都没有证明自己能稳定超过强离线模型，因此不再作为默认主线。

### 3.5 第五阶段：hetero_ranker

`hetero_ranker` 的引入，代表当前项目的正式转向：
- 保持接入接口不变
- 不再把问题建模为 pointer 选点
- 而是改为 masked action ranking

当前脚本：
- `协同调度/train_scheduler_hetero_ranker.py`

当前推荐仍然先接一次 `hetero_bc` 做教师诊断与 encoder warm start，但这不是结构性的硬前置；`hetero_ranker` 也可以从零开始训练。

## 4. 为什么 hetero_bc 卡住

### 4.1 教师并不弱

从 `teacher_val_reference.json` 可见，当前自动选中的教师仍然是：
- `upfront_wait_aware_greedy`

其参考水平大约达到：
- `success_rate = 1.000`
- `mean_makespan = 796.08`
- `mean_wait_time = 47.01`
- `mean_avoidable_wait_time = 2.68`
- `mean_direct_sync_misassignment_rate = 0.00899`

因此，当前 BC 卡住不是因为“教师太弱没有上限”。

### 4.2 hard-state 学不动才是核心瓶颈

近期 `hetero_bc` 长期停在：
- `teacher_agreement ≈ 0.881`
- `hard_teacher_agreement ≈ 0.860`
- `single_vs_sync_conflict_agreement ≈ 0.283`
- `makespan ≈ 864.63`
- `wait ≈ 83.49`
- `avoidable_wait ≈ 9.20`

这说明：
- 总体成功率虽高
- 但关键 hard-state 尤其 `single vs sync` 冲突状态没有学会教师边界

### 4.3 根因是目标与模型头不匹配

当前问题的难点不是“合法动作太多”，而是：
- 合法动作通常不多
- 少量 hard-state 的边界极其重要

pointer-BC 只学“老师最后选了谁”，却不学：
- 第二名是谁
- `single` 比 `sync` 好多少
- hardest negative 是哪个动作

这正是 BC 长期平台化的根因。

## 5. 为什么 hetero_ranker 有效

### 5.1 结构更贴近任务本质

`hetero_ranker` 复用了 Hetero 编码器，但把决策头改成了 per-action scorer：
- 当前机器人 embedding
- 当前候选任务 embedding
- 全局 task context
- 全局 agent context
- 当前机器人与候选任务的交互项
- 当前机器人原始输入
- 原始 `task_inputs`

这些特征拼接后进入 `action_scorer`，对每个动作单独打分，再结合 `global_mask` 输出 logits。

它本质上是在做 masked action ranking，而不是 pointer 式 top-1 复制。

### 5.2 CE 阶段已经显著优于 BC

近期一次 `hetero_ranker` 训练中，`CE` 阶段最佳 checkpoint 达到：
- `teacher_agreement = 0.993`
- `hard_teacher_agreement = 0.992`
- `single_vs_sync_conflict_agreement = 0.955`
- `success = 1.000`
- `makespan = 794.43`
- `wait = 48.73`
- `avoidable_wait = 3.39`
- `trap_avoidable_wait = 0.96`

与当前 BC 相比，这已经是显著跨代提升。

### 5.3 成功点在 CE，而不是后半程强推

当前最重要的经验结论是：
- `hetero_ranker` 的成功已经证明，问题主要在求解器建模，而不是接口或数据集
- 当前稳定的正结果主要来自 `CE` 阶段
- 因此当前默认训练策略已经收缩为 `CE-only`

## 6. 为什么 rank / DAgger 会退化

尽管 `hetero_ranker` 的 `CE` 阶段已经很好，但后续 `rank / DAgger` 并不总是稳定。

近期退化表现大致包括：
- `teacher_agreement` 下滑到 `0.94~0.95`
- `single_vs_sync_conflict_agreement` 下滑到 `0.55~0.60`
- `makespan` 回升到 `806~833`
- `avoidable_wait` 回升到 `4.2~7.1`
- `trap_avoidable_wait` 回升到 `3.25~5.50`

当前较可信的原因包括：
1. pairwise 正样本如果偏离 chosen teacher，会发生目标错位
2. pairwise ranking 对 hardest negative 的推动过强
3. DAgger 若记录过量，会扰乱 CE 阶段已经稳定的全局结构
4. late-stage 数据刷新容易偏向模型当前错误分布

因此当前默认策略已经固定为：
- `base_epochs = 4`
- `rank_epochs = 0`
- `dagger_mode = off`
- `ce_mix = 0.5`
- `dagger_max_records_per_family = 512`

也就是说，当前正式主线默认先停在 `CE`，而不是默认继续推进 `rank / DAgger`。

## 7. 为什么 CE 成功后仍然需要部署 guard

当前剩余问题已经从“能不能学到教师”转移到“部署时如何处理 hard-state 不确定性”。

原因是：
- `best_scheduler_ranker.pt` 是训练期综合最优，不一定等于部署最安全
- 联调是 deterministic argmax，缺少训练阶段的统计平均效应
- hard-state 尤其 `single vs sync` 冲突状态，少量低 margin 错误就会明显放大 `avoidable_wait` 和 trap 指标

因此当前部署策略改为：
- 训练端区分：
  - `best_scheduler_ranker.pt`
  - `best_scheduler_ranker_deploy.pt`
- 评估端明确输出：
  - `deploy_ready`
  - `deployment_failures`
  - `deployment_recommendation`
- 联调端对 `hetero_ranker` 增加 conservative guard：
  - hard-state 且 low-margin 时
  - fallback 到 `upfront_wait_aware` 运行时启发式动作

当前结论不是“ranker 训练还没成功”，而是：
- CE 阶段已经成功
- 剩余问题主要是部署闭环与 hard-state 保守性

## 8. 当前正式结论

当前最合理的论文结论是：
1. 对当前多机器人协同调度任务，真正需要冻结的是接入契约，而不是内部模型结构。
2. pointer-BC 虽然能提供一个稳定但偏弱的初始化，但在 hard-state 上长期学不动。
3. `hetero_ranker` 把问题重写成 masked action ranking 之后，CE 阶段已经显著优于 pointer-BC，并接近教师上限。
4. 当前真正需要继续研究的不是“如何让 BC 再多涨一点”，而是“如何让 ranker 的后半程训练稳定，以及如何在部署时处理 hard-state 不确定性”。
5. 因此，当前正式推荐主线不是 `hetero_bc + hetero_actor_only`，而是：

`hetero_bc(推荐 warm start / 教师诊断) -> hetero_ranker -> evaluate -> nav`

## 9. 当前各线在论文中的定位

- `legacy PPO`
  - 历史主线与早期正结果
- `improved / staged PPO`
  - 防坍缩工程与中间阶段证据
- `hetero_bc`
  - 负结果：说明 pointer-BC 的局限
- `hetero_actor_only / hetero_ppo`
  - 阶段性负结果：说明弱初始化下在线微调难以直接突破
- `hetero_ranker`
  - 当前正式主线
  - 当前最重要的正结果

## 10. 后续开放问题

当前最值得继续研究的方向是：
1. `ranker -> pointer` 蒸馏桥接
2. 更稳的 hard-only DAgger
3. 直接针对 ranker 本身的后续在线微调
4. 更系统的部署 guard 校准与 hard-state 风险估计

## 11. 可直接复用的论文表述

以下表述当前可以直接作为论文草稿的基础结论：
- “在保持高层输入输出接口不变的前提下，我们将高层调度从 pointer-style imitation 重构为 masked action ranking。”
- “实验表明，单纯的 pointer-BC 在 hard state，尤其是 `single vs sync` 冲突状态上长期无法逼近教师上限。”
- “与之相比，`hetero_ranker` 的 teacher-choice CE 阶段即可显著提升 `teacher_agreement`、`hard_teacher_agreement` 与 trap 指标，并基本追平教师基线。”
- “当前剩余问题已从训练能否学到教师，转移到部署时如何处理 hard-state 不确定性；因此正式部署优先使用 deploy-ready checkpoint，并在必要时启用 conservative guard。”

## 12. 材料引用位置

当前论文写作时可优先引用：
- `协同调度/checkpoints_hetero_bc/teacher_val_reference.json`
- `协同调度/checkpoints_hetero_ranker/teacher_val_reference.json`
- `协同调度/checkpoints_hetero_ranker/deployment_report.json`
- `README.md`
- 各类结果对比 JSON

这些材料连同 BC / PPO / actor-only 的负结果，都应保留为方法演化叙事的一部分，不应删除。
