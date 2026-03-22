from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from attention_policy import AttentionSchedulerPolicy, load_scheduler_checkpoint, masked_logits, obs_to_torch, save_scheduler_checkpoint
from scheduler_training_utils import clone_obs, evaluate_policy_on_scenarios
from scheduler_utils import curriculum_families, load_split_scenarios
from sequential_scheduling_env import SequentialSchedulingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用自定义 masked PPO 微调高层 attention 调度器。")
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--total-updates", type=int, default=120)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--limit-train-per-family", type=int, default=None)
    parser.add_argument("--limit-val-per-family", type=int, default=None)
    parser.add_argument("--save-dir", default="协同调度/checkpoints_rl")
    parser.add_argument("--init-model", default="协同调度/checkpoints_bc/best_scheduler_bc.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def stack_obs_list(obs_list, device: torch.device):
    keys = obs_list[0].keys()
    return {
        key: torch.as_tensor(np.stack([obs[key] for obs in obs_list], axis=0), dtype=torch.float32, device=device)
        for key in keys
    }


def compute_gae(rewards, values, dones, last_value, gamma: float, gae_lambda: float):
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


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_by_family = {
        family: load_split_scenarios(
            scenario_dir=args.scenario_dir,
            split=args.train_split,
            families=[family],
            limit_per_family=args.limit_train_per_family,
        )
        for family in [
            "open_balance",
            "role_mismatch",
            "single_bottleneck",
            "double_bottleneck",
            "far_near_trap",
            "multi_sync_cluster",
        ]
    }
    val_scenarios = load_split_scenarios(
        scenario_dir=args.scenario_dir,
        split=args.val_split,
        limit_per_family=args.limit_val_per_family,
    )
    if not any(train_by_family.values()):
        raise ValueError("训练场景为空，请先生成 offline_maps_v2。")
    if not val_scenarios:
        raise ValueError("验证场景为空，请检查 val split。")

    if args.init_model:
        model, checkpoint = load_scheduler_checkpoint(args.init_model, device=device)
        print(f"加载初始模型: {args.init_model}")
        metadata = checkpoint.get("metadata", {})
    else:
        model = AttentionSchedulerPolicy().to(device)
        metadata = {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_score = float("-inf")
    env: SequentialSchedulingEnv | None = None
    obs = None
    rng = np.random.default_rng(0)

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

        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

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
        policy_losses = []
        value_losses = []
        entropies = []

        model.train()
        for _ in range(args.ppo_epochs):
            rng.shuffle(index)
            for start in range(0, len(index), args.batch_size):
                batch_index = index[start:start + args.batch_size]
                batch_obs = stack_obs_list([rollout_obs[i] for i in batch_index], device=device)
                batch_actions = torch.as_tensor(np.asarray([rollout_actions[i] for i in batch_index]), dtype=torch.long, device=device)
                batch_old_log_prob = torch.as_tensor(np.asarray([rollout_log_probs[i] for i in batch_index]), dtype=torch.float32, device=device)
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
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        if update % args.eval_every == 0 or update == 1 or update == args.total_updates:
            val_metrics = evaluate_policy_on_scenarios(val_scenarios, model, device=device, deterministic=True)
            score = (
                val_metrics["success_rate"] * 1000.0
                - val_metrics["mean_makespan"]
                - 0.5 * val_metrics["mean_wait_time"]
            )

            latest_path = save_dir / "latest_scheduler_rl.pt"
            save_scheduler_checkpoint(
                latest_path,
                model,
                optimizer=optimizer,
                metadata={
                    **metadata,
                    "stage": "ppo_finetune",
                    "update": update,
                    "families": active_families,
                    "val_metrics": val_metrics,
                },
            )
            if score > best_score:
                best_score = score
                best_path = save_dir / "best_scheduler_rl.pt"
                save_scheduler_checkpoint(
                    best_path,
                    model,
                    optimizer=optimizer,
                    metadata={
                        **metadata,
                        "stage": "ppo_finetune",
                        "update": update,
                        "families": active_families,
                        "val_metrics": val_metrics,
                    },
                )

            print(
                f"[RL][update {update:03d}] "
                f"families={','.join(active_families)} "
                f"policy={np.mean(policy_losses):.4f} "
                f"value={np.mean(value_losses):.4f} "
                f"entropy={np.mean(entropies):.4f} "
                f"success={val_metrics['success_rate']:.3f} "
                f"makespan={val_metrics['mean_makespan']:.2f} "
                f"wait={val_metrics['mean_wait_time']:.2f}"
            )
        else:
            print(
                f"[RL][update {update:03d}] "
                f"families={','.join(active_families)} "
                f"policy={np.mean(policy_losses):.4f} "
                f"value={np.mean(value_losses):.4f} "
                f"entropy={np.mean(entropies):.4f}"
            )


if __name__ == "__main__":
    main()
