from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from attention_policy import (
    AttentionSchedulerPolicy,
    load_scheduler_checkpoint,
    masked_logits,
    obs_to_torch,
    save_scheduler_checkpoint,
)
from scheduler_training_utils import (
    clone_obs,
    collect_expert_samples,
    evaluate_policy_on_scenarios,
)
from scheduler_utils import curriculum_families, load_split_scenarios
from sequential_scheduling_env import SequentialSchedulingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用受约束的 masked PPO 微调高层 attention 调度器。")
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--total-updates", type=int, default=120)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--bc-anchor-coef-start", type=float, default=0.5)
    parser.add_argument("--bc-anchor-coef-end", type=float, default=0.05)
    parser.add_argument("--bc-anchor-decay-ratio", type=float, default=0.4)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--success-drop-threshold", type=float, default=0.02)
    parser.add_argument(
        "--trap-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否额外在 partial_coalition_trap 子集上做单独验证。",
    )
    parser.add_argument("--limit-train-per-family", type=int, default=None)
    parser.add_argument("--limit-val-per-family", type=int, default=None)
    parser.add_argument("--save-dir", default="协同调度/checkpoints_improve_rl")
    parser.add_argument("--init-model", default="协同调度/checkpoints_improve_bc/best_scheduler_bc.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def stack_obs_list(obs_list: Sequence[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = obs_list[0].keys()
    return {
        key: torch.as_tensor(np.stack([obs[key] for obs in obs_list], axis=0), dtype=torch.float32, device=device)
        for key in keys
    }


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    for index in reversed(range(len(rewards))):
        if dones[index]:
            next_non_terminal = 0.0
            next_value = 0.0
        elif index == len(rewards) - 1:
            next_non_terminal = 1.0
            next_value = last_value
        else:
            next_non_terminal = 1.0
            next_value = values[index + 1]
        delta = rewards[index] + gamma * next_value * next_non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[index] = last_advantage
    returns = advantages + values
    return advantages, returns


def metric_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["success_rate"] * 1500.0
        - metrics["mean_makespan"]
        - 0.4 * metrics["mean_wait_time"]
        - 0.8 * metrics["mean_avoidable_wait_time"]
    )


def bc_anchor_coef_for_update(
    update: int,
    total_updates: int,
    start: float,
    end: float,
    decay_ratio: float,
) -> float:
    if total_updates <= 0:
        return end
    progress = update / max(total_updates, 1)
    if progress >= decay_ratio:
        return end
    ratio = progress / max(decay_ratio, 1e-6)
    return start + ratio * (end - start)


def sample_expert_batch(
    samples: Sequence[tuple[Dict[str, np.ndarray], int]],
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor] | None:
    if not samples:
        return None
    replace = len(samples) < batch_size
    indices = rng.choice(len(samples), size=batch_size, replace=replace)
    obs_list = [samples[int(index)][0] for index in indices]
    action_array = np.asarray([samples[int(index)][1] for index in indices], dtype=np.int64)
    return stack_obs_list(obs_list, device=device), torch.as_tensor(action_array, dtype=torch.long, device=device)


def checkpoint_metadata(
    base_metadata: Dict,
    update: int,
    families: Sequence[str],
    val_metrics: Dict[str, float],
    trap_val_metrics: Dict[str, float] | None,
) -> Dict:
    metadata = {
        **base_metadata,
        "stage": "ppo_finetune",
        "update": update,
        "families": list(families),
        "val_metrics": val_metrics,
    }
    if trap_val_metrics is not None:
        metadata["trap_val_metrics"] = trap_val_metrics
    return metadata


def log_eval(
    update: int,
    families: Sequence[str],
    policy_losses: List[float],
    value_losses: List[float],
    entropies: List[float],
    bc_losses: List[float],
    bc_anchor_coef: float,
    val_metrics: Dict[str, float],
    trap_val_metrics: Dict[str, float] | None,
) -> None:
    line = (
        f"[RL][update {update:03d}] "
        f"families={','.join(families)} "
        f"bc_anchor={bc_anchor_coef:.3f} "
        f"policy={np.mean(policy_losses):.4f} "
        f"value={np.mean(value_losses):.4f} "
        f"bc={np.mean(bc_losses) if bc_losses else 0.0:.4f} "
        f"entropy={np.mean(entropies):.4f} "
        f"success={val_metrics['success_rate']:.3f} "
        f"makespan={val_metrics['mean_makespan']:.2f} "
        f"wait={val_metrics['mean_wait_time']:.2f} "
        f"avoidable_wait={val_metrics['mean_avoidable_wait_time']:.2f} "
        f"reassign_rate={val_metrics['mean_productive_reassign_rate']:.3f}"
    )
    if trap_val_metrics is not None:
        line += (
            f" trap_success={trap_val_metrics['success_rate']:.3f}"
            f" trap_avoidable_wait={trap_val_metrics['mean_avoidable_wait_time']:.2f}"
            f" trap_delay={trap_val_metrics['mean_coalition_activation_delay']:.2f}"
        )
    print(line)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(0)

    family_order = [
        "open_balance",
        "role_mismatch",
        "single_bottleneck",
        "double_bottleneck",
        "far_near_trap",
        "multi_sync_cluster",
        "partial_coalition_trap",
    ]
    train_by_family = {
        family: load_split_scenarios(
            scenario_dir=args.scenario_dir,
            split=args.train_split,
            families=[family],
            limit_per_family=args.limit_train_per_family,
        )
        for family in family_order
    }
    val_scenarios = load_split_scenarios(
        scenario_dir=args.scenario_dir,
        split=args.val_split,
        limit_per_family=args.limit_val_per_family,
    )
    trap_val_scenarios = []
    if args.trap_eval:
        trap_val_scenarios = load_split_scenarios(
            scenario_dir=args.scenario_dir,
            split=args.val_split,
            families=["partial_coalition_trap"],
            limit_per_family=args.limit_val_per_family,
        )
    if not any(train_by_family.values()):
        raise ValueError("训练场景为空，请先生成 offline_maps_v2 数据集。")
    if not val_scenarios:
        raise ValueError("验证场景为空，请检查 val split。")

    expert_samples_by_family = {}
    for family, scenarios in train_by_family.items():
        if not scenarios:
            continue
        expert_samples_by_family[family] = collect_expert_samples(
            scenarios,
            expert_policy="wait_aware_role_greedy",
        )

    if args.init_model:
        model, checkpoint = load_scheduler_checkpoint(args.init_model, device=device)
        print(f"加载初始模型: {args.init_model}")
        metadata = checkpoint.get("metadata", {})
    else:
        model = AttentionSchedulerPolicy().to(device)
        metadata = {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    env: SequentialSchedulingEnv | None = None
    obs = None

    init_families = curriculum_families(0.0)
    init_val_metrics = evaluate_policy_on_scenarios(val_scenarios, model, device=device, deterministic=True)
    init_trap_metrics = (
        evaluate_policy_on_scenarios(trap_val_scenarios, model, device=device, deterministic=True)
        if trap_val_scenarios
        else None
    )
    init_success_rate = init_val_metrics["success_rate"]
    best_score = metric_score(init_val_metrics)
    no_improve_evals = 0

    init_metadata = checkpoint_metadata(
        metadata,
        update=0,
        families=init_families,
        val_metrics=init_val_metrics,
        trap_val_metrics=init_trap_metrics,
    )
    save_scheduler_checkpoint(save_dir / "latest_scheduler_rl.pt", model, optimizer=optimizer, metadata=init_metadata)
    save_scheduler_checkpoint(save_dir / "best_scheduler_rl.pt", model, optimizer=optimizer, metadata=init_metadata)
    log_eval(
        update=0,
        families=init_families,
        policy_losses=[0.0],
        value_losses=[0.0],
        entropies=[0.0],
        bc_losses=[0.0],
        bc_anchor_coef=args.bc_anchor_coef_start,
        val_metrics=init_val_metrics,
        trap_val_metrics=init_trap_metrics,
    )

    for update in range(1, args.total_updates + 1):
        progress = (update - 1) / max(args.total_updates - 1, 1)
        active_families = curriculum_families(progress)
        active_scenarios = [scenario for family in active_families for scenario in train_by_family.get(family, [])]
        if not active_scenarios:
            raise ValueError(f"课程阶段 {active_families} 没有可用场景。")

        if env is None or set(active_families) != set(curriculum_families((update - 2) / max(args.total_updates - 1, 1))):
            env = SequentialSchedulingEnv(scenarios=active_scenarios)
            obs, _ = env.reset()
        elif env is not None:
            env.base_env.scenarios = list(active_scenarios)

        active_expert_samples = [
            sample
            for family in active_families
            for sample in expert_samples_by_family.get(family, [])
        ]

        rollout_obs: List[Dict[str, np.ndarray]] = []
        rollout_actions: List[int] = []
        rollout_log_probs: List[float] = []
        rollout_rewards: List[float] = []
        rollout_dones: List[bool] = []
        rollout_values: List[float] = []

        model.eval()
        for _ in range(args.rollout_steps):
            obs_tensors = obs_to_torch(obs, device=device)
            with torch.no_grad():
                action, log_prob, value = model.act(obs_tensors, deterministic=False)

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))

            rollout_obs.append(clone_obs(obs))
            rollout_actions.append(int(action.item()))
            rollout_log_probs.append(float(log_prob.item()))
            rollout_rewards.append(float(reward))
            rollout_dones.append(bool(terminated or truncated))
            rollout_values.append(float(value.item()))

            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

        with torch.no_grad():
            last_value = 0.0
            if not rollout_dones[-1]:
                _, _, bootstrap_value = model.act(obs_to_torch(obs, device=device), deterministic=True)
                last_value = float(bootstrap_value.item())

        rewards = np.asarray(rollout_rewards, dtype=np.float32)
        values = np.asarray(rollout_values, dtype=np.float32)
        dones = np.asarray(rollout_dones, dtype=np.bool_)
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)

        index = np.arange(len(rollout_actions))
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        bc_losses: List[float] = []
        bc_anchor_coef = bc_anchor_coef_for_update(
            update=update,
            total_updates=args.total_updates,
            start=args.bc_anchor_coef_start,
            end=args.bc_anchor_coef_end,
            decay_ratio=args.bc_anchor_decay_ratio,
        )

        model.train()
        for _ in range(args.ppo_epochs):
            rng.shuffle(index)
            for start in range(0, len(index), args.batch_size):
                batch_index = index[start:start + args.batch_size]
                batch_obs = stack_obs_list([rollout_obs[i] for i in batch_index], device=device)
                batch_actions = torch.as_tensor(
                    np.asarray([rollout_actions[i] for i in batch_index]),
                    dtype=torch.long,
                    device=device,
                )
                batch_old_log_prob = torch.as_tensor(
                    np.asarray([rollout_log_probs[i] for i in batch_index]),
                    dtype=torch.float32,
                    device=device,
                )
                batch_advantages = torch.as_tensor(advantages[batch_index], dtype=torch.float32, device=device)
                batch_returns = torch.as_tensor(returns[batch_index], dtype=torch.float32, device=device)

                logits, values_pred = model(batch_obs)
                masked = masked_logits(logits, batch_obs["current_action_mask"])
                dist = Categorical(logits=masked)
                log_prob = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_prob - batch_old_log_prob)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values_pred, batch_returns)

                bc_loss = torch.zeros((), dtype=torch.float32, device=device)
                if active_expert_samples and bc_anchor_coef > 0.0:
                    expert_batch = sample_expert_batch(
                        active_expert_samples,
                        batch_size=len(batch_index),
                        rng=rng,
                        device=device,
                    )
                    if expert_batch is not None:
                        expert_obs, expert_actions = expert_batch
                        expert_logits, _ = model(expert_obs)
                        expert_masked = masked_logits(expert_logits, expert_obs["current_action_mask"])
                        bc_loss = F.cross_entropy(expert_masked, expert_actions)

                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy + bc_anchor_coef * bc_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                bc_losses.append(float(bc_loss.item()))

        if update % args.eval_every == 0 or update == 1 or update == args.total_updates:
            val_metrics = evaluate_policy_on_scenarios(val_scenarios, model, device=device, deterministic=True)
            trap_val_metrics = (
                evaluate_policy_on_scenarios(trap_val_scenarios, model, device=device, deterministic=True)
                if trap_val_scenarios
                else None
            )
            current_score = metric_score(val_metrics)
            metadata_now = checkpoint_metadata(
                metadata,
                update=update,
                families=active_families,
                val_metrics=val_metrics,
                trap_val_metrics=trap_val_metrics,
            )
            save_scheduler_checkpoint(
                save_dir / "latest_scheduler_rl.pt",
                model,
                optimizer=optimizer,
                metadata=metadata_now,
            )

            if val_metrics["success_rate"] >= init_success_rate - args.success_drop_threshold and current_score > best_score:
                best_score = current_score
                no_improve_evals = 0
                save_scheduler_checkpoint(
                    save_dir / "best_scheduler_rl.pt",
                    model,
                    optimizer=optimizer,
                    metadata=metadata_now,
                )
            else:
                no_improve_evals += 1

            log_eval(
                update=update,
                families=active_families,
                policy_losses=policy_losses,
                value_losses=value_losses,
                entropies=entropies,
                bc_losses=bc_losses,
                bc_anchor_coef=bc_anchor_coef,
                val_metrics=val_metrics,
                trap_val_metrics=trap_val_metrics,
            )

            if (
                update >= 12
                and no_improve_evals >= args.early_stop_patience
                and val_metrics["success_rate"] < init_success_rate - 0.04
            ):
                print(
                    f"[RL] 提前停止: 连续 {no_improve_evals} 次评估未优于 best，"
                    f"且 success={val_metrics['success_rate']:.3f} 低于 init={init_success_rate:.3f} 超过 0.04。"
                )
                break
        else:
            print(
                f"[RL][update {update:03d}] "
                f"families={','.join(active_families)} "
                f"bc_anchor={bc_anchor_coef:.3f} "
                f"policy={np.mean(policy_losses):.4f} "
                f"value={np.mean(value_losses):.4f} "
                f"bc={np.mean(bc_losses) if bc_losses else 0.0:.4f} "
                f"entropy={np.mean(entropies):.4f}"
            )


if __name__ == "__main__":
    main()
