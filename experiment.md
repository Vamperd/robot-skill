# experiment.md

本文件是当前仓库的论文实验设计与执行手册。它只回答三件事：

- 为了证明当前方法可行、有效、可解释，需要做哪些实验
- 每组实验的比较对象、指标、命令模板、结论标准是什么
- 哪些实验属于主文必须项，哪些适合放附录或系统诊断章节

文档分工固定如下：

- `README.md`：怎么跑
- `paper.md`：方法演化、失败分析、研究结论
- `experiment.md`：实验设计、实验矩阵、指标口径、出图出表规范

## 1. 研究目标与论文对标

当前系统要证明的是：在异构多机器人、`single/sync` 协同任务、precedence 约束和任务运行时推进共同作用下，高层调度方法既能获得高质量分配结果，也能接入实际联调链路。

参考论文的方式是“借鉴实验轴”，不是“完全照搬原问题设定”。

### 1.1 对标 `73-RAL2025-HetMRTA.pdf`

重点借鉴它的实验轴：

- 解质量
- waiting time
- success rate
- computation time
- scalability

这篇论文与当前仓库更接近，因为都强调：

- 异构机器人
- 协同任务
- 等待时间与协同代价
- 学习式或近学习式高层任务分配

### 1.2 对标《时序逻辑约束下的多机器人任务分配与协同优化》

重点借鉴它的实验组织方式：

- 可行性案例
- 求解效率
- 可扩展性
- 等待时间优化机制
- 小规模精确规划对照

### 1.3 统一表述边界

当前论文不直接宣称“完整 LTLf 规划系统”，统一表述为：

- precedence + `single/sync` 协同约束下的高层任务分配与执行

为了证明当前方法不仅优于已有启发式，而且相对传统规划方法也具有竞争力，实验将增加两条面向当前仓库接口的传统规划基线：

- `sas`
  - 全规模可跑的顺序式传统规划主基线
- `ctas_d`
  - Linux + Gurobi 条件下的小到中规模联合优化强基线

## 2. 当前正式主线与实验边界

当前正式推荐主线：

`offline_maps_v2 -> hetero_bc(教师诊断 / 推荐 warm start) -> hetero_ranker -> evaluate -> nav`

当前方法定位：

- `upfront_wait_aware_greedy`
  - 当前最强纯启发式教师候选
- `rollout_upfront_teacher`
  - 教师候选 / 对照
- `hybrid_upfront_teacher`
  - 教师候选 / 对照
- `hetero_bc`
  - 教师诊断、历史 pointer-BC 对照、推荐 warm start
- `hetero_ranker (CE-only)`
  - 当前正式主线
- `hetero_actor_only`
  - 实验性后续微调
- `hetero_ppo`
  - 实验性后续微调 / 论文对照
- `legacy`
  - 历史学习基线

默认实验设置：

- 主结果与主消融：`5` 个随机 seed
- 联调：固定 checkpoint + `10` 个代表性 `stress` 场景
- 主方法默认使用：
  - `hetero_ranker`
  - `CE-only`
  - `best_scheduler_ranker.pt` 或 `best_scheduler_ranker_deploy.pt`
- `hetero_actor_only / hetero_ppo` 只作为 exploratory

统一指标口径：

- `success_rate`
- `mean_makespan`
- `mean_wait_time`
- `mean_avoidable_wait_time`
- `mean_direct_sync_misassignment_rate`
- `teacher_agreement`
- `hard_teacher_agreement`
- `single_vs_sync_conflict_agreement`
- `trap_success`
- `trap_avoidable_wait`
- `trap_sync_misassign`

## 3. 传统规划基线的引入原则

传统规划方法的对比必须分层，不能把当前仓库内可复现的经典方法和新增的外部传统规划器混成一张表。

### 3.1 A 层：当前仓库内可复现基线

- `nearest_eta_greedy`
- `role_aware_greedy`
- `wait_aware_role_greedy`
- `upfront_wait_aware_greedy`
- `rollout_upfront_teacher`
- `hybrid_upfront_teacher`
- `hetero_bc`
- `legacy`

### 3.2 B 层：当前仓库内新增的传统规划基线

- `sas`
  - 全规模可跑，进入主文主表、hard/trap 表、规模效率表
- `ctas_d`
  - 依赖 Linux + Gurobi，进入小规模 / 中规模精确对比表

### 3.3 C 层：辅助与回归基线

- `auction_mrta`
  - 轻量代价竞价式基线，用于 sanity check、附录和开发对照
- `milp_scheduler_small`
  - `PuLP + CBC` 的小规模精确规划器，用于回归和开发阶段的求解校验

固定说明：

- `random` 只保留为 sanity baseline，不进入主文主表
- 当前论文主文只承诺两条传统规划主对照：
  - `sas`
  - `ctas_d`
- 当前不承诺复现：
  - `TACO`
  - `full-LTL / product automaton` 的 SMT / 自动机规划器
- 当前实现的 `sas / ctas_d` 都是 **repo-aligned 近似实现**：
  - 保持当前仓库动作协议 `0=wait, 1..N=task_id`
  - 直接接入现有评估与联调接口
  - 不声称是参考论文的逐字忠实复现

## 4. 传统规划基线的当前实现状态

当前仓库中已经落地的传统规划策略层位于：

- `实验/传统规划基线/planner_baselines.py`
- `实验/传统规划基线/solver_backend_gurobi.py`
- `实验/传统规划基线/auction_mrta.py`
- `实验/传统规划基线/milp_scheduler_small.py`
- `实验/传统规划基线/sas.py`
- `实验/传统规划基线/ctas_d.py`

当前定位如下：

- `sas`
  - 全规模传统规划主基线
  - 默认应进入主结果、hard/trap、规模效率、联调对照
- `ctas_d`
  - Gurobi 驱动的联合优化强基线
  - 默认只在 small/medium bucket 或显式实验中运行
  - **不自动加入** `--include-baselines`
- `auction_mrta`
  - 轻量传统分配 sanity / 附录基线
- `milp_scheduler_small`
  - `PuLP + CBC` 的开发期回归精确基线

## 5. 实验一：高层调度主结果对比

### 5.1 目标

证明 `hetero_ranker` 在当前任务设定下优于旧学习路线，并达到教师级别的高层调度质量；同时证明它相对当前可复现的传统规划主基线具有竞争力。

### 5.2 比较对象

`A. 仓库内可复现基线`

- `nearest_eta_greedy`
- `role_aware_greedy`
- `wait_aware_role_greedy`
- `upfront_wait_aware_greedy`
- `rollout_upfront_teacher`
- `hybrid_upfront_teacher`
- `hetero_bc`
- `legacy`

`B. 传统规划主基线`

- `sas`

`C. 主方法`

- `hetero_ranker (CE-only)`

### 5.3 数据与运行设置

- 数据集：`offline_maps_v2`
- split：`test`
- 推理方式：deterministic
- 学习方法至少跑 `5` 个 seed

### 5.4 核心指标

- `success_rate`
- `mean_makespan`
- `mean_wait_time`
- `mean_avoidable_wait_time`
- `mean_direct_sync_misassignment_rate`

### 5.5 命令模板

评估 `hetero_ranker`：

```bash
python 协同调度/evaluate_scheduler.py --model 协同调度/checkpoints_hetero_ranker/best_scheduler_ranker.pt --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --include-baselines --save-json 成果/json/ranker_test_eval.json
```

单独评估最强教师：

```bash
python 协同调度/evaluate_scheduler.py --expert-policy upfront_wait_aware_greedy --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --save-json 成果/json/upfront_teacher_test_eval.json
```

单独评估 `sas`：

```bash
python 协同调度/evaluate_scheduler.py --expert-policy sas --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --save-json 成果/json/sas_test_eval.json
```

### 5.6 结论标准

- `hetero_ranker` 必须明显优于 `hetero_bc`
- 相比最强启发式教师，至少在 `makespan / avoidable_wait / sync_misassign` 中有 `2` 项持平或更优
- 相比 `sas`，至少在 `mean_makespan` 和 `mean_avoidable_wait_time` 中体现稳定优势或显著持平
- 如果某个教师或传统方法在单项指标上更强，要保留该负结果，不做弱化

## 6. 实验二：hard-state 与 trap 子集验证

### 6.1 目标

证明当前方法不是只在 easy state 有效，而是真的改进了协同等待与同步冲突边界；同时检验传统规划主基线在 hard case 下的行为质量。

### 6.2 固定内容

- `trap_subset = partial_coalition_trap`
- `family_breakdown`
- hard-state agreement 三指标

### 6.3 比较对象

- `hetero_ranker`
- `hetero_bc`
- `upfront_wait_aware_greedy`
- `hybrid_upfront_teacher`
- `sas`

### 6.4 核心指标

- `trap_success`
- `trap_avoidable_wait`
- `trap_sync_misassign`
- family 的 `success / avoidable_wait / sync_misassign`
- `teacher_agreement`
- `hard_teacher_agreement`
- `single_vs_sync_conflict_agreement`

### 6.5 命令模板

```bash
python 协同调度/evaluate_scheduler.py --model 协同调度/checkpoints_hetero_ranker/best_scheduler_ranker.pt --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --family-breakdown --trap-eval --save-json 成果/json/ranker_test_family_eval.json
python 协同调度/evaluate_scheduler.py --expert-policy sas --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --family-breakdown --trap-eval --save-json 成果/json/sas_family_eval.json
```

### 6.6 结论标准

- `hetero_ranker` 在 `partial_coalition_trap` 和关键 hard family 上必须显著优于 `hetero_bc`
- `single_vs_sync_conflict_agreement` 作为主文指标，不放附录
- `sas` 若在总体上接近但在 trap 上明显变差，应明确写为传统方法对协同等待边界不够敏感

## 7. 实验三：机制消融与训练路线对比

### 7.1 目标

证明当前正结果来自正确建模与训练路线，而不是偶然训练出来的单个 checkpoint。

### 7.2 固定消融项

- `hetero_bc` vs `hetero_ranker`
- `hetero_ranker scratch` vs `hetero_ranker + BC warm start`
- `hetero_ranker CE-only` vs `hetero_ranker CE + rank`
- `hetero_ranker CE-only` vs `hetero_actor_only <- ranker`
- `hetero_ranker CE-only` vs `hetero_ppo <- ranker`

约束：

- 前三项进入主文
- 后两项写成 exploratory，放附录或系统章节

### 7.3 核心指标

- `success_rate`
- `mean_makespan`
- `mean_avoidable_wait_time`
- `trap_avoidable_wait`
- `single_vs_sync_conflict_agreement`

### 7.4 结论标准

- 当前稳定主线是 `teacher -> BC warm start -> ranker CE-only`
- 当前后续 RL 微调尚未稳定带来进一步收益

## 8. 实验四：规模泛化与计算效率

### 8.1 目标

证明当前方法不仅在当前数据分布上有效，而且在规模增长和困难 split 上具有平滑退化与可部署推理开销。

### 8.2 bucket 设计

默认按以下 bucket 组织：

- 小规模
- 中规模
- 大规模 / 困难场景

优先使用 `offline_maps_v2` 内部按机器人数、任务数、family 做 bucket；若覆盖不足，再补额外 scale 数据集。

### 8.3 4A：全规模效率

比较对象：

- `hetero_ranker`
- `upfront_wait_aware_greedy`
- `sas`

核心指标：

- `success_rate`
- `mean_makespan`
- `mean_avoidable_wait_time`
- 每 100 个 scenario 的总评估 wall-clock
- 单次高层决策 latency

固定写法：

- `sas` 代表传统可扩展规划器
- 这一小节只讨论全规模可运行方法

### 8.4 4B：精确规划效率

比较对象：

- `ctas_d`

运行范围：

- 只在 `small / medium` bucket 上跑

额外记录：

- 求解 wall-clock
- 预算截断率
- 规模增长趋势

固定写法：

- `ctas_d` 代表传统联合优化器
- 这一小节不与全规模主表混写

## 9. 实验四-B：小规模精确优化对比

### 9.1 目标

单独回答“在可解规模下，学习方法离传统联合优化上界还有多远”。

### 9.2 比较对象

- `hetero_ranker`
- `upfront_wait_aware_greedy`
- `auction_mrta`
- `sas`
- `ctas_d`
- `milp_scheduler_small`

### 9.3 运行设定

- 只跑 `small / medium` bucket
- `ctas_d` 默认在 Linux + Gurobi 主机上运行
- 记录每个场景的求解 wall-clock
- 记录预算截断率、超时率和可解率

### 9.4 核心指标

- `success_rate`
- `mean_makespan`
- `mean_avoidable_wait_time`
- 求解 wall-clock
- 超时率 / 预算截断率 / 可解率

### 9.5 结论标准

- 小规模下，`ctas_d` 提供传统联合优化上界或近上界参考
- `hetero_ranker` 若在质量接近的同时推理显著更快，应作为主文重要结论
- `milp_scheduler_small` 主要用作开发期交叉校验，不作为主文核心传统基线

## 10. 实验五：A* + PPO 联调可行性与瓶颈定位

### 10.1 正式联调

比较：

- `upfront_wait_aware_greedy + A* + PPO`
- `hetero_ranker + A* + PPO`
- `sas + A* + PPO`

固定场景：

- `stress` 中至少 `10` 个代表性场景
- 必须覆盖：
  - `partial_coalition_trap`
  - `double_bottleneck`
  - `multi_sync_cluster`
  - 至少 1 个窄通道障碍场景

固定指标：

- episode success / truncation
- task completion count
- final makespan
- stuck / deadlock rate
- 关键 GIF 回放

写作约束：

- `ctas_d` 只做 smoke 或 subset 联调，不作为全规模联调主结论方法

### 10.2 无互避实验线

入口：

- `实验/无互避A星PPO联调/run_scheduled_nav_no_robot_avoid.py`

固定比较：

- 正式联调：机器人互避开启
- 诊断联调：墙壁-only obstacle，机器人可重叠穿过

固定解释逻辑：

- 若正式联调失败、无互避实验线成功，则瓶颈主要在局部互避 / 低层运动
- 若两者都失败，再回头怀疑高层调度本身

写作约束：

- 这一部分放附录或系统章节
- 它是诊断路径，不替代正式联调，也不替代高层主结果表

## 11. 图表与出文规范

主文必须有的表：

- 表 1：主结果对比（含 `sas`）
- 表 2：hard / trap 结果（含 `sas`）
- 表 3：机制消融
- 表 4：规模与效率实验
- 表 5：小规模精确优化对比（含 `ctas_d`）

主文必须有的图：

- `hetero_ranker vs hetero_bc` 关键指标柱状图
- family breakdown 图
- 规模 / 延迟曲线图
- 典型场景轨迹图

附录必须有：

- 一个小规模可行性案例
- 一个 trap case 对照
- 一个正式联调 vs 无互避实验线 GIF 或关键帧对照

## 12. 实验结果落盘规范

建议统一将结构化结果保存到：

- `成果/json/`

建议命名方式：

- `ranker_test_eval.json`
- `ranker_test_family_eval.json`
- `sas_test_eval.json`
- `sas_family_eval.json`
- `ctasd_small_eval.json`
- `ctasd_small_family_eval.json`
- `auction_test_eval.json`
- `milp_small_eval.json`
- `nav_ranker_case_01.json`
- `nav_no_robot_avoid_case_01.json`

建议每个主表都保留：

- 原始 JSON
- 出图脚本
- 生成图表的导出数据

## 13. 执行顺序

推荐严格按下面顺序推进，避免被联调问题拖住主结果：

1. 先做高层主结果
2. 再做 hard / trap
3. 再做机制消融
4. 再做规模与效率
5. 再做小规模精确优化对比
6. 最后做联调和无互避诊断

## 14. 验收标准

本次实验设计是否完成，按以下标准检查：

- 主结果、hard / trap、消融、规模效率、小规模精确优化、联调、无互避诊断这 `7` 类实验都有明确定义
- 文档里引用的脚本都是真实存在的当前入口：
  - `协同调度/evaluate_scheduler.py`
  - `协同调度/train_scheduler_hetero_ranker.py`
  - `协同调度/train_scheduler_hetero_bc.py`
  - `导航结合RL运动/run_scheduled_nav.py`
  - `实验/无互避A星PPO联调/run_scheduled_nav_no_robot_avoid.py`
- 文档明确写出“应该与传统规划方法对比”
- 传统规划主方法不再泛指，而是固定成两条：
  - `sas`
  - `ctas_d`
- 明确区分：
  - 当前仓库内可复现基线
  - 当前仓库内新增的传统规划主基线
  - 辅助与回归基线
- 主文全规模表和小规模联合优化表分开，不把 `ctas_d` 强行塞进全规模主表
- 文档口径与当前 `README.md / paper.md / AGENTS.md` 一致：
  - 正式主线仍是 `hetero_ranker`
  - 无互避实验线只是诊断路径，不是正式结论路径

## 15. 传统规划基线的具体使用方法

### 15.1 `sas` 的使用方法

定位：

- 全规模可跑的顺序式传统规划主基线
- 适合进入主结果表、hard / trap 表、规模效率表、联调对照
- 若 Linux 主机上有 Gurobi，可在局部 repair 中利用它；没有 Gurobi 也能运行纯顺序启发式版本

#### 15.1.1 主结果评估

```bash
python 协同调度/evaluate_scheduler.py --expert-policy sas --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --save-json 成果/json/sas_test_eval.json
```

#### 15.1.2 hard / trap 评估

```bash
python 协同调度/evaluate_scheduler.py --expert-policy sas --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --family-breakdown --trap-eval --save-json 成果/json/sas_family_eval.json
```

#### 15.1.3 正式联调

```bash
python 导航结合RL运动/run_scheduled_nav.py --scenario-dir offline_maps_v2 --split stress --limit 1 --scheduler-policy sas --low-level-model 无导航纯RL底层运动器/results/generalization_eval_best_vel_punishment_狭窄距离，效果最好，可过U形弯/best_model.zip --render --gif-name sas_nav_demo.gif
```

#### 15.1.4 无互避实验联调

```bash
python 实验/无互避A星PPO联调/run_scheduled_nav_no_robot_avoid.py --scenario-dir offline_maps_v2 --split stress --limit 1 --scheduler-policy sas --low-level-model 无导航纯RL底层运动器/results/generalization_eval_best_vel_punishment_狭窄距离，效果最好，可过U形弯/best_model.zip --render --gif-name sas_nav_no_robot_avoid.gif
```

建议提取指标：

- `success_rate`
- `mean_makespan`
- `mean_wait_time`
- `mean_avoidable_wait_time`
- `planner_diagnostics.mean_round_eval_time_ms`
- `planner_diagnostics.mean_robot_decision_time_ms`
- `planner_diagnostics.mean_local_repair_time_ms`
- `planner_diagnostics.local_timeout_rate`

### 15.2 `ctas_d` 的使用方法

定位：

- Linux + Gurobi 条件下的小到中规模联合优化强基线
- 主要进入“小规模精确优化对比表”
- 不默认加入 `--include-baselines`

运行前说明：

- 需要 Linux 主机可调用 `gurobipy`
- 若当前环境没有 Gurobi，选中该策略时会直接报清晰错误
- 默认不建议在 `stress` 全量场景上批跑
- 默认只建议配合：
  - `small / medium` bucket
  - 或显式限制 `--max-episodes`

#### 15.2.1 小规模精确评估

```bash
python 协同调度/evaluate_scheduler.py --expert-policy ctas_d --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --max-episodes 32 --save-json 成果/json/ctasd_small_eval.json
```

#### 15.2.2 小规模 hard / trap 补充评估

```bash
python 协同调度/evaluate_scheduler.py --expert-policy ctas_d --policy-type hetero_ranker --scenario-dir offline_maps_v2 --split test --max-episodes 32 --family-breakdown --trap-eval --save-json 成果/json/ctasd_small_family_eval.json
```

#### 15.2.3 联调 smoke / subset 评估

```bash
python 导航结合RL运动/run_scheduled_nav.py --scenario-dir offline_maps_v2 --split stress --limit 1 --scheduler-policy ctas_d --low-level-model 无导航纯RL底层运动器/results/generalization_eval_best_vel_punishment_狭窄距离，效果最好，可过U形弯/best_model.zip --render --gif-name ctasd_nav_demo.gif
```

建议提取指标：

- `success_rate`
- `mean_makespan`
- `mean_avoidable_wait_time`
- `planner_diagnostics.mean_search_time_ms`
- `planner_diagnostics.oversize_fallback_rate`
- `planner_diagnostics.budget_truncation_rate`
- `planner_diagnostics.mean_primitive_count`
- `planner_diagnostics.mean_sync_primitive_count`

### 15.3 `auction_mrta` 与 `milp_scheduler_small` 的使用口径

这两条方法保留，但角色降级：

- `auction_mrta`
  - 轻量传统分配 sanity / 附录对照
- `milp_scheduler_small`
  - `PuLP + CBC` 的开发期回归精确基线

推荐用途：

- 算法开发与回归测试
- 小规模 sanity 对照
- 附录中补一张“轻量传统 / 小规模精确 / Gurobi 联合优化”的内部对照表
