from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from hetero_attention_policy import (
    HeteroAttentionSchedulerPolicy,
    load_hetero_scheduler_checkpoint,
    obs_to_torch,
    save_hetero_scheduler_checkpoint,
)
from hetero_dispatch_env import HeteroDispatchEnv
from hetero_training_utils import clone_obs, collect_expert_samples, evaluate_policy_on_scenarios
from scheduler_utils import load_split_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Hetero mainline scheduler with PPO.")
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--total-updates", type=int, default=120)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None, help="Legacy override for both actor/critic lrs.")
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--clip-range", type=float, default=0.05)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=1e-4)
    parser.add_argument("--warmup-ent-coef", type=float, default=0.0)
    parser.add_argument("--ent-switch-update", type=int, default=21)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--limit-train-per-family", type=int, default=None)
    parser.add_argument("--limit-val-per-family", type=int, default=None)
    parser.add_argument("--save-dir", default="协同调度/checkpoints_hetero_rl")
    parser.add_argument("--init-model", default="协同调度/checkpoints_hetero_bc/best_scheduler_bc.pt")
    parser.add_argument("--value-warmup-updates", type=int, default=4)
    parser.add_argument("--bc-anchor-coef-start", type=float, default=0.25)
    parser.add_argument("--bc-anchor-coef-end", type=float, default=0.03)
    parser.add_argument("--bc-anchor-decay-ratio", type=float, default=0.35)
    parser.add_argument("--kl-guard-threshold", type=float, default=0.03)
    parser.add_argument("--expert-episodes-per-family", type=int, default=12)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-updates-before-stop", type=int, default=30)
    parser.add_argument("--success-drop-threshold", type=float, default=0.05)
    parser.add_argument("--severe-regression-success-drop", type=float, default=0.10)
    parser.add_argument("--severe-regression-idle-wait-rate", type=float, default=0.25)
    parser.add_argument("--rollback-limit", type=int, default=2)
    parser.add_argument(
        "--trap-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate partial_coalition_trap subset separately.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def hetero_curriculum_families(progress: float) -> list[str]:
    if progress < 0.20:
        return ["open_balance", "role_mismatch"]
    if progress < 0.40:
        return ["open_balance", "role_mismatch", "single_bottleneck"]
    if progress < 0.60:
        return ["open_balance", "role_mismatch", "single_bottleneck", "far_near_trap"]
    if progress < 0.80:
        return [
            "open_balance",
            "role_mismatch",
            "single_bottleneck",
            "far_near_trap",
            "partial_coalition_trap",
        ]
    return [
        "open_balance",
        "role_mismatch",
        "single_bottleneck",
        "double_bottleneck",
        "far_near_trap",
        "multi_sync_cluster",
        "partial_coalition_trap",
    ]


def build_optimizer(
    model: HeteroAttentionSchedulerPolicy,
    actor_lr: float,
    critic_lr: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": list(model.actor.parameters()), "lr": actor_lr},
            {"params": list(model.value_head.parameters()), "lr": critic_lr},
        ]
    )


def set_actor_requires_grad(model: HeteroAttentionSchedulerPolicy, requires_grad: bool) -> None:
    for param in model.actor.parameters():
        param.requires_grad = requires_grad


def stack_obs_list(obs_list: Sequence[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    batch = {
        key: torch.as_tensor(np.stack([obs[key] for obs in obs_list], axis=0), dtype=torch.float32, device=device)
        for key in obs_list[0].keys()
    }
    batch["current_agent_index"] = batch["current_agent_index"].long()
    return batch


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
        - 50.0 * metrics["mean_direct_sync_misassignment_rate"]
        - 100.0 * metrics["mean_idle_wait_rate"]
        - 20.0 * metrics["mean_wait_action_rate"]
        - 10.0 * metrics["mean_dispatch_gap_penalty"]
    )


def checkpoint_metadata(
    base_metadata: Dict,
    update: int,
    families: Sequence[str],
    val_metrics: Dict[str, float],
    trap_val_metrics: Dict[str, float] | None,
    extra: Dict | None = None,
) -> Dict:
    metadata = {
        **base_metadata,
        "policy_type": "hetero_ppo",
        "stage": "ppo_finetune",
        "update": update,
        "families": list(families),
        "val_metrics": val_metrics,
    }
    if trap_val_metrics is not None:
        metadata["trap_val_metrics"] = trap_val_metrics
    if extra:
        metadata.update(extra)
    return metadata


def log_eval(
    update: int,
    families: Sequence[str],
    policy_losses: List[float],
    value_losses: List[float],
    entropies: List[float],
    val_metrics: Dict[str, float],
    trap_val_metrics: Dict[str, float] | None,
    bc_kl: float,
) -> None:
    line = (
        f"[HPPO][update {update:03d}] "
        f"families={','.join(families)} "
        f"policy={np.mean(policy_losses):.4f} "
        f"value={np.mean(value_losses):.4f} "
        f"entropy={np.mean(entropies):.4f} "
        f"success={val_metrics['success_rate']:.3f} "
        f"makespan={val_metrics['mean_makespan']:.2f} "
        f"wait={val_metrics['mean_wait_time']:.2f} "
        f"avoidable_wait={val_metrics['mean_avoidable_wait_time']:.2f} "
        f"sync_misassign={val_metrics['mean_direct_sync_misassignment_rate']:.3f} "
        f"wait_rate={val_metrics['mean_wait_action_rate']:.3f} "
        f"idle_wait={val_metrics['mean_idle_wait_rate']:.3f} "
        f"dispatch_gap={val_metrics['mean_dispatch_gap_penalty']:.3f} "
        f"bc_kl={bc_kl:.4f}"
    )
    if trap_val_metrics is not None:
        line += (
            f" trap_success={trap_val_metrics['success_rate']:.3f}"
            f" trap_avoidable_wait={trap_val_metrics['mean_avoidable_wait_time']:.2f}"
            f" trap_sync_misassign={trap_val_metrics['mean_direct_sync_misassignment_rate']:.3f}"
        )
    print(line)


def collect_expert_pools(
    train_by_family: Dict[str, Sequence[Dict]],
    max_episodes_per_family: int,
) -> Dict[str, List[tuple[Dict[str, np.ndarray], int]]]:
    pools: Dict[str, List[tuple[Dict[str, np.ndarray], int]]] = {}
    for family, scenarios in train_by_family.items():
        if not scenarios:
            continue
        pools[family] = collect_expert_samples(
            scenarios,
            expert_policy="upfront_wait_aware_greedy",
            max_episodes=max_episodes_per_family,
            seed=hash(family) % 10000,
        )
    return pools


def sample_anchor_batch(
    expert_pools: Dict[str, List[tuple[Dict[str, np.ndarray], int]]],
    families: Sequence[str],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[Dict[str, np.ndarray], torch.Tensor] | None:
    active_pools = [expert_pools[family] for family in families if expert_pools.get(family)]
    if not active_pools:
        return None

    sampled = []
    for _ in range(batch_size):
        pool = active_pools[int(rng.integers(len(active_pools)))]
        sampled.append(pool[int(rng.integers(len(pool)))])

    obs_batch = {
        key: np.stack([sample[0][key] for sample in sampled], axis=0)
        for key in sampled[0][0].keys()
    }
    actions = np.asarray([sample[1] for sample in sampled], dtype=np.int64)
    return obs_batch, torch.as_tensor(actions, dtype=torch.long)


def bc_anchor_coef(
    update: int,
    total_updates: int,
    coef_start: float,
    coef_end: float,
    decay_ratio: float,
) -> float:
    if total_updates <= 1:
        return coef_end
    decay_updates = max(int(total_updates * decay_ratio), 1)
    progress = float(np.clip(update / decay_updates, 0.0, 1.0))
    return float(coef_start + (coef_end - coef_start) * progress)


def compute_bc_kl(
    model: HeteroAttentionSchedulerPolicy,
    reference_model: HeteroAttentionSchedulerPolicy,
    obs_batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad():
        _, ref_logps, _ = reference_model(obs_batch)
        ref_probs = ref_logps.exp()
    _, cur_logps, _ = model(obs_batch)
    return torch.sum(ref_probs * (ref_logps - cur_logps), dim=-1).mean()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(0)

    if args.lr is not None:
        actor_lr = args.lr
        critic_lr = args.lr
    else:
        actor_lr = args.actor_lr
        critic_lr = args.critic_lr
    current_actor_lr = actor_lr
    current_critic_lr = critic_lr
    ent_coef_scale = 1.0
    rollback_count = 0

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
        raise ValueError("Training scenarios are empty. Generate offline_maps_v2 first.")
    if not val_scenarios:
        raise ValueError("Validation scenarios are empty.")

    expert_pools = collect_expert_pools(train_by_family, max_episodes_per_family=args.expert_episodes_per_family)

    if args.init_model:
        model, checkpoint = load_hetero_scheduler_checkpoint(args.init_model, device=device)
        print(f"Load initial model: {args.init_model}")
        metadata = checkpoint.get("metadata", {})
    else:
        bootstrap_scenarios = next(iter([value for value in train_by_family.values() if value]))
        bootstrap_env = HeteroDispatchEnv(scenarios=bootstrap_scenarios)
        bootstrap_obs, _ = bootstrap_env.reset()
        model = HeteroAttentionSchedulerPolicy(
            agent_input_dim=int(bootstrap_obs["agent_inputs"].shape[-1]),
            task_input_dim=int(bootstrap_obs["task_inputs"].shape[-1]),
        ).to(device)
        metadata = {}

    reference_model = copy.deepcopy(model).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    optimizer = build_optimizer(model, actor_lr=current_actor_lr, critic_lr=current_critic_lr)
    env: HeteroDispatchEnv | None = None
    obs = None

    init_families = hetero_curriculum_families(0.0)
    init_val_metrics = evaluate_policy_on_scenarios(val_scenarios, model, device=device, deterministic=True)
    init_trap_metrics = (
        evaluate_policy_on_scenarios(trap_val_scenarios, model, device=device, deterministic=True)
        if trap_val_scenarios
        else None
    )
    init_success_rate = init_val_metrics["success_rate"]
    best_score = metric_score(init_val_metrics)
    best_model_state = copy.deepcopy(model.state_dict())
    best_metadata = checkpoint_metadata(
        metadata,
        update=0,
        families=init_families,
        val_metrics=init_val_metrics,
        trap_val_metrics=init_trap_metrics,
        extra={"rollback_count": rollback_count},
    )
    no_improve_evals = 0

    save_hetero_scheduler_checkpoint(save_dir / "latest_scheduler_rl.pt", model, optimizer=optimizer, metadata=best_metadata)
    save_hetero_scheduler_checkpoint(save_dir / "best_scheduler_rl.pt", model, optimizer=optimizer, metadata=best_metadata)
    log_eval(0, init_families, [0.0], [0.0], [0.0], init_val_metrics, init_trap_metrics, bc_kl=0.0)

    for update in range(1, args.total_updates + 1):
        progress = (update - 1) / max(args.total_updates - 1, 1)
        active_families = hetero_curriculum_families(progress)
        active_scenarios = [scenario for family in active_families for scenario in train_by_family.get(family, [])]
        if not active_scenarios:
            raise ValueError(f"No active scenarios for families: {active_families}")

        prev_progress = (update - 2) / max(args.total_updates - 1, 1)
        prev_families = hetero_curriculum_families(prev_progress) if update > 1 else []
        if env is None or set(active_families) != set(prev_families):
            env = HeteroDispatchEnv(scenarios=active_scenarios)
            obs, _ = env.reset()
        else:
            env.base_env.scenarios = list(active_scenarios)

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
        advantages, returns = compute_gae(rewards, values, dones, last_value, args.gamma, args.gae_lambda)
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)

        actor_trainable = update > args.value_warmup_updates
        set_actor_requires_grad(model, actor_trainable)
        current_ent_coef = args.ent_coef * ent_coef_scale if update >= args.ent_switch_update else args.warmup_ent_coef * ent_coef_scale
        current_bc_anchor_coef = (
            bc_anchor_coef(
                update=update,
                total_updates=args.total_updates,
                coef_start=args.bc_anchor_coef_start,
                coef_end=args.bc_anchor_coef_end,
                decay_ratio=args.bc_anchor_decay_ratio,
            )
            if actor_trainable
            else 0.0
        )

        indices = np.arange(len(rollout_actions))
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        bc_kls: List[float] = []
        kl_guard_triggered = False

        model.train()
        for _ in range(args.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start:start + args.batch_size]
                obs_batch = stack_obs_list([rollout_obs[i] for i in batch_indices], device=device)
                actions = torch.as_tensor([rollout_actions[i] for i in batch_indices], dtype=torch.long, device=device)
                old_log_probs = torch.as_tensor(
                    [rollout_log_probs[i] for i in batch_indices],
                    dtype=torch.float32,
                    device=device,
                )
                batch_advantages = torch.as_tensor(advantages[batch_indices], dtype=torch.float32, device=device)
                batch_returns = torch.as_tensor(returns[batch_indices], dtype=torch.float32, device=device)

                batch_bc_kl = compute_bc_kl(model, reference_model, obs_batch)
                bc_kls.append(float(batch_bc_kl.item()))
                if actor_trainable and batch_bc_kl.item() > args.kl_guard_threshold:
                    kl_guard_triggered = True
                    break

                log_prob, entropy, value = model.evaluate_actions(obs_batch, actions)
                policy_loss = torch.zeros((), dtype=torch.float32, device=device)
                entropy_bonus = torch.zeros((), dtype=torch.float32, device=device)
                bc_loss = torch.zeros((), dtype=torch.float32, device=device)

                if actor_trainable:
                    ratio = torch.exp(log_prob - old_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                    policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                    entropy_bonus = entropy.mean()

                    anchor_batch = sample_anchor_batch(expert_pools, active_families, args.batch_size, rng)
                    if anchor_batch is not None:
                        anchor_obs_np, anchor_actions = anchor_batch
                        anchor_obs = {
                            key: torch.as_tensor(value, dtype=torch.float32, device=device)
                            for key, value in anchor_obs_np.items()
                        }
                        anchor_obs["current_agent_index"] = anchor_obs["current_agent_index"].long()
                        _, anchor_logps, _ = model(anchor_obs)
                        bc_loss = F.nll_loss(anchor_logps, anchor_actions.to(device))

                value_loss = 0.5 * (batch_returns - value).pow(2).mean()
                loss = (
                    policy_loss
                    + args.vf_coef * value_loss
                    - current_ent_coef * entropy_bonus
                    + current_bc_anchor_coef * bc_loss
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_bonus.item()))

            if kl_guard_triggered:
                break

        set_actor_requires_grad(model, True)

        val_metrics = None
        trap_val_metrics = None
        should_eval = update % args.eval_every == 0 or update == args.total_updates
        if should_eval:
            val_metrics = evaluate_policy_on_scenarios(val_scenarios, model, device=device, deterministic=True)
            if trap_val_scenarios:
                trap_val_metrics = evaluate_policy_on_scenarios(
                    trap_val_scenarios,
                    model,
                    device=device,
                    deterministic=True,
                )

            candidate_score = metric_score(val_metrics)
            allow_best = val_metrics["success_rate"] >= init_success_rate - 0.02
            if allow_best and candidate_score > best_score:
                best_score = candidate_score
                no_improve_evals = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_metadata = checkpoint_metadata(
                    metadata,
                    update=update,
                    families=active_families,
                    val_metrics=val_metrics,
                    trap_val_metrics=trap_val_metrics,
                    extra={"rollback_count": rollback_count, "bc_kl": float(np.mean(bc_kls)) if bc_kls else 0.0},
                )
                save_hetero_scheduler_checkpoint(
                    save_dir / "best_scheduler_rl.pt",
                    model,
                    optimizer=optimizer,
                    metadata=best_metadata,
                )
            else:
                no_improve_evals += 1

            severe_regression = (
                val_metrics["success_rate"] < init_success_rate - args.severe_regression_success_drop
                or val_metrics["mean_idle_wait_rate"] > args.severe_regression_idle_wait_rate
            )
            rolled_back = False
            if severe_regression:
                model.load_state_dict(best_model_state)
                rollback_count += 1
                current_actor_lr *= 0.5
                ent_coef_scale *= 0.5
                optimizer = build_optimizer(model, actor_lr=current_actor_lr, critic_lr=current_critic_lr)
                no_improve_evals = 0
                rolled_back = True

            latest_metadata = checkpoint_metadata(
                metadata,
                update=update,
                families=active_families,
                val_metrics=val_metrics,
                trap_val_metrics=trap_val_metrics,
                extra={
                    "rollback_count": rollback_count,
                    "rolled_back": rolled_back,
                    "kl_guard_triggered": kl_guard_triggered,
                    "bc_kl": float(np.mean(bc_kls)) if bc_kls else 0.0,
                    "actor_lr": current_actor_lr,
                    "critic_lr": current_critic_lr,
                },
            )
            save_hetero_scheduler_checkpoint(
                save_dir / "latest_scheduler_rl.pt",
                model,
                optimizer=optimizer,
                metadata=latest_metadata,
            )
            log_eval(
                update,
                active_families,
                policy_losses or [0.0],
                value_losses or [0.0],
                entropies or [0.0],
                val_metrics,
                trap_val_metrics,
                bc_kl=float(np.mean(bc_kls)) if bc_kls else 0.0,
            )

            if (
                update >= args.min_updates_before_stop
                and no_improve_evals >= args.early_stop_patience
                and val_metrics["success_rate"] < init_success_rate - args.success_drop_threshold
                and rollback_count >= args.rollback_limit
            ):
                print(
                    f"[HPPO] early stop: no improvement for {no_improve_evals} evals, "
                    f"success below update0 threshold, rollback_count={rollback_count}."
                )
                break
        else:
            print(
                f"[HPPO][update {update:03d}] "
                f"families={','.join(active_families)} "
                f"policy={np.mean(policy_losses or [0.0]):.4f} "
                f"value={np.mean(value_losses or [0.0]):.4f} "
                f"entropy={np.mean(entropies or [0.0]):.4f} "
                f"bc_kl={np.mean(bc_kls or [0.0]):.4f}"
            )


if __name__ == "__main__":
    main()
