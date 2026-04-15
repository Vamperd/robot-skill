# robot-skill

这是一个分层多机器人任务执行仓库。系统由三层组成：

- 高层协同调度：决定当前机器人执行哪个 `task_id`，或输出 `wait`
- 中层全局规划：A* 负责全局可行路径
- 底层连续运动：PPO 控制局部避障与跟踪

当前正式推荐主线已经切换为：

`offline_maps_v2 -> hetero_bc(教师诊断/推荐 warm start) -> hetero_ranker -> evaluate -> run_scheduled_nav`

## 1. 当前推荐结论

- 正式推荐主线是 `hetero_ranker`
- `hetero_bc` 是推荐前置，用于教师诊断和 encoder warm start，但不是结构性硬依赖
- `hetero_ranker` 正式部署使用 `best_scheduler_ranker.pt`，不是 `latest_scheduler_ranker.pt`
- `hetero_actor_only / hetero_ppo` 继续保留，但当前定位是实验性后续微调 / 论文对照

## 2. 仓库结构

- `协同调度/`
  高层调度、数据集、训练、评估、渲染
- `导航结合RL运动/`
  高层调度 + A* + 底层 PPO 联调入口
- `无导航纯RL底层运动器/`
  底层连续运动 PPO 训练与评估
- `paper.md`
  方法演化、失败分析、论文叙事材料
- `AGENTS.md`
  面向后续 agent 的稳定工程上下文

## 3. 使用前说明

### 3.1 推荐环境

默认假设在 conda 的 `(pytorch)` 环境中运行。系统 Python 下偶尔出现的 `torch + numpy 2.x` 警告，不应被直接视为代码主问题。

### 3.2 固定接入契约

当前高层接口已经冻结，不要擅自改动：

- 输入四件套：
  - `agent_inputs`
  - `task_inputs`
  - `global_mask`
  - `current_agent_index`
- 输出动作协议：
  - `0 = wait`
  - `1..N = task_id`
- 联调链路固定为：
  - 高层派单
  - A*
  - 底层 PPO
  - `service_ready / task_runtime`

### 3.3 `wait` 的统一语义

训练、评估、联调统一使用同一套 `wait` 约束：

- 当机器人处于 `idle` 或 `waiting_idle`
- 且存在合法非 `wait` 动作时
- `wait` 会被统一 mask 掉

这样可以避免训练和联调之间的动作语义不一致。

## 4. 推荐主线流程

### 4.1 生成或更新 `offline_maps_v2`

正式生成：

```bash
python 协同调度/build_offline_maps.py --save-dir offline_maps_v2 --overwrite
```

小规模调试：

```bash
python 协同调度/build_offline_maps.py --save-dir tmp_offline_maps_v2 --overwrite --limit-per-family 1
```

可选浏览：

```bash
python 协同调度/view_offline_maps.py --cache-dir offline_maps_v2 --split val --family partial_coalition_trap
```

### 4.2 [推荐前置] 运行 `hetero_bc`

用途：

- 对当前教师做正式比较与诊断
- 生成 `teacher_val_reference.json`
- 产出一个推荐的 Hetero encoder warm start

命令：

```bash
python 协同调度/train_scheduler_hetero_bc.py --scenario-dir offline_maps_v2
```

当前会自动比较三类教师：

- `upfront_wait_aware_greedy`
- `rollout_upfront_teacher`
- `hybrid_upfront_teacher`

训练结束后重点看：

- `协同调度/checkpoints_hetero_bc/teacher_val_reference.json`
- `协同调度/checkpoints_hetero_bc/best_scheduler_bc.pt`

说明：

- `hetero_bc` 现在主要用于教师诊断和历史对照
- 它不是当前默认部署模型
- 但仍然是 `hetero_ranker` 的推荐 warm start

### 4.3 [正式主线] 训练 `hetero_ranker`

推荐用 BC warm start：

```bash
python 协同调度/train_scheduler_hetero_ranker.py --scenario-dir offline_maps_v2 --init-model 协同调度/checkpoints_hetero_bc/best_scheduler_bc.pt
```

从零开始训练：

```bash
python 协同调度/train_scheduler_hetero_ranker.py --scenario-dir offline_maps_v2 --init-model ""
```

当前默认训练策略：

- `CE` 是主训练阶段
- `rank` 是轻量边界修正阶段
- `DAgger` 默认关闭
- 若要显式开启，只支持：
  - `--dagger-mode hard_only`

关键默认超参数：

- `base_epochs = 4`
- `rank_epochs = 4`
- `dagger_mode = off`
- `batch_size = 128`
- `pairwise_batch_size = 64`
- `lr = 3e-4`
- `ce_mix = 0.1`
- `dagger_max_records_per_family = 512`

训练完成后重点看：

- `协同调度/checkpoints_hetero_ranker/teacher_val_reference.json`
- `协同调度/checkpoints_hetero_ranker/best_scheduler_ranker.pt`
- `协同调度/checkpoints_hetero_ranker/latest_scheduler_ranker.pt`

部署规则：

- `best_scheduler_ranker.pt`：正式评估与联调使用
- `latest_scheduler_ranker.pt`：仅表示最后一次训练快照

### 4.4 [推荐] 正式评估

```bash
python 协同调度/evaluate_scheduler.py --model 协同调度/checkpoints_hetero_ranker/best_scheduler_ranker.pt --policy-type hetero_ranker --split test --include-baselines
```

可选 family / trap 评估：

```bash
python 协同调度/evaluate_scheduler.py --model 协同调度/checkpoints_hetero_ranker/best_scheduler_ranker.pt --policy-type hetero_ranker --split val --family-breakdown --trap-eval
```

纯教师评估：

```bash
python 协同调度/evaluate_scheduler.py --expert-policy upfront_wait_aware_greedy --policy-type hetero_ranker --split val --family-breakdown --trap-eval --save-json upfront_teacher_val.json
python 协同调度/evaluate_scheduler.py --expert-policy rollout_upfront_teacher --policy-type hetero_ranker --split val --family-breakdown --trap-eval --save-json rollout_teacher_val.json
python 协同调度/evaluate_scheduler.py --expert-policy hybrid_upfront_teacher --policy-type hetero_ranker --split val --family-breakdown --trap-eval --save-json hybrid_teacher_val.json
```

### 4.5 [推荐] 联调到 A* + 底层 PPO

```bash
python 导航结合RL运动/run_scheduled_nav.py --scenario-dir offline_maps_v2 --split stress --limit 1 --scheduler-model 协同调度/checkpoints_hetero_ranker_ce_only/best_scheduler_ranker.pt --scheduler-policy-type hetero_ranker --low-level-model 无导航纯RL底层运动器/results/generalization_eval_best_vel_punishment_狭窄距离，效果最好，可过U形弯/best_model.zip --render --gif-name demo.gif
```

`run_scheduled_nav.py` 与 `scheduler_nav_runner.py` 现在已原生支持：

- `legacy`
- `hetero_ppo`
- `hetero_actor_only`
- `hetero_ranker`

### 4.6 [实验] 墙壁-only 障碍的 A* + PPO 联调

这条实验线用于排除机器人间互避对联调结果的干扰，只验证：

- 高层调度仍按原路线派单
- A* 仍只对墙壁/静态障碍规划
- 底层 PPO 仍负责跟踪 waypoint
- 机器人之间不互避、不碰撞、可重叠穿过

它是独立于正式联调路径的平行实验，不会改动现有 `导航结合RL运动/run_scheduled_nav.py` 的使用方式。

实验命令：

```bash
python 实验/无互避A星PPO联调/run_scheduled_nav_no_robot_avoid.py --scenario-dir offline_maps_v2 --split stress --limit 1 --scheduler-model 协同调度/checkpoints_hetero_ranker_ce_only/best_scheduler_ranker.pt --scheduler-policy-type hetero_ranker --low-level-model 无导航纯RL底层运动器/results/generalization_eval_best_vel_punishment_狭窄距离，效果最好，可过U形弯/best_model.zip --render --gif-name demo_no_robot_avoid.gif
```

提醒：

- 这条实验线仅用于验证与排障，不替代正式联调结论
- 正式联调仍以 `导航结合RL运动/run_scheduled_nav.py` 为准

## 5. 关键输出文件怎么读

### 5.1 `teacher_val_reference.json`

它记录当前教师候选在验证集上的正式参考结果，包括：

- 自动选择了哪个教师
- 每个教师的综合分数
- 整体指标
- trap 子集指标
- family breakdown

如果 BC 或 ranker 训练效果异常，先看这个文件，不要只看训练日志。

### 5.2 `best` 和 `latest` 的区别

- `best_scheduler_ranker.pt`
  - 基于验证指标保存
  - 当前默认部署模型
- `latest_scheduler_ranker.pt`
  - 最后一个 epoch 的快照
  - 可能比 `best` 更差

当前经验结论是：

- `hetero_ranker` 的最佳模型通常来自 `CE` 或轻量 `rank` 阶段
- 后续 `rank / DAgger` 可能把好模型拉坏

## 6. 实验性后续微调

### 6.1 [实验] `hetero_actor_only <- hetero_ranker`

时间敏感、优先看趋势的推荐配置：

```bash
python 协同调度/train_scheduler_hetero_actor_only.py --profile quick_trend --scenario-dir offline_maps_v2 --init-model 协同调度/checkpoints_hetero_ranker/best_scheduler_ranker.pt --allow-weak-init
```

当前 `quick_trend` 预设：

- `total_updates = 96`
- `scenarios_per_update = 24`
- `rollouts_per_scenario = 4`
- `eval_every = 8`
- `full_eval_every = 16`
- `agreement_eval_every = 16`
- `min_updates_before_stop = 32`
- `early_stop_patience = 4`

当前经验耗时：

- 默认长训配置：约 `2.5 分钟 / update`
- `quick_trend` 目标：约 `0.8~1.2 分钟 / update`

默认长训配置仍可用：

```bash
python 协同调度/train_scheduler_hetero_actor_only.py --scenario-dir offline_maps_v2 --init-model 协同调度/checkpoints_hetero_ranker/best_scheduler_ranker.pt --allow-weak-init
```

说明：

- 当前只推荐作为实验性 warm start
- 默认配置是高成本长训
- `quick_trend` 是当前推荐的时间敏感实验配置
- 它只会复用 Hetero 编码器
- 不会复用 ranker 的 `action_scorer`
- 常规评估现在拆成 quick eval 和 full eval，只有 full eval 会更新 `best_scheduler_actor_only.pt`

### 6.2 [实验 / 对照] `hetero_ppo <- hetero_ranker`

```bash
python 协同调度/train_scheduler_hetero_ppo.py --scenario-dir offline_maps_v2 --init-model 协同调度/checkpoints_hetero_ranker/best_scheduler_ranker.pt
```

说明：

- 当前只保留为实验性后续微调 / 论文对照
- 不是默认推荐主线

## 7. 各条训练线的当前定位

- `hetero_ranker`
  - 当前正式主线
- `hetero_bc`
  - 教师诊断 / 历史 pointer-BC 对照
- `hetero_actor_only`
  - 实验性后续微调
- `hetero_ppo`
  - 实验性后续微调 / 论文对照
- `legacy PPO`
  - 历史链路与早期对照

## 8. 当前最重要的工程结论

- 当前最强的默认部署模型是 `hetero_ranker`
- `hetero_bc` 卡住的根因不是教师弱，而是 pointer-BC 的目标与模型头不适合当前任务
- `hetero_ranker` 的成功点在于 masked action ranking 更贴近 `single vs sync` 的边界比较
- 当前真正需要继续稳定的是 ranker 的后半程训练，而不是接口或联调链路

## 9. [ʵ��] ��ͳ�滮����

��ǰ�Ѿ�����������ͨ�����нӿڵ��õĴ�ͳ�滮���ߣ�

- `auction_mrta`
  - ������ͳ�������
  - ��ֱ�ӽ���ȫ��ģ����������
- `milp_scheduler_small`
  - С��ģ��ȷ�Ż�����
  - ������ѡ `PuLP + CBC`
  - �������� `small / medium` �Ӽ�����ʽ����

���� `auction_mrta`��

```bash
python Эͬ����/evaluate_scheduler.py --expert-policy auction_mrta --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --save-json �ɹ�/json/auction_test_eval.json
```

���� `milp_scheduler_small`��

```bash
python Эͬ����/evaluate_scheduler.py --expert-policy milp_scheduler_small --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --max-episodes 32 --save-json �ɹ�/json/milp_small_eval.json
```

˵����

- `auction_mrta` �ʺ���Ϊ���������� hard / trap ���Ĵ�ͳ�滮����
- `milp_scheduler_small` �����Զ����� `--include-baselines`
- `milp_scheduler_small` ��ȱ�� `PuLP + CBC`����������ʱֱ�ӱ���������
