from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from hetero_attention_policy import (
    HeteroActorOnlyPolicy,
    load_hetero_actor_only_checkpoint,
    obs_to_torch,
    save_hetero_actor_only_checkpoint,
)
from hetero_dispatch_env import HeteroDispatchEnv
from hetero_training_utils import (
    clone_obs,
    collect_expert_sample_records,
    collect_expert_samples,
    evaluate_policy_on_scenarios,
)
from scheduler_utils import load_split_scenarios


FULL_FIDELITY_DEFAULTS = {
    "total_updates": 160,
    "scenarios_per_update": 48,
    "rollouts_per_scenario": 8,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "sample_temperature": 1.25,
    "small_margin": 5.0,
    "eval_every": 4,
    "expert_episodes_per_family": 12,
    "bc_anchor_updates": 20,
    "bc_anchor_coef": 0.02,
    "anchor_batch_size": 128,
    "early_stop_patience": 16,
    "min_updates_before_stop": 60,
    "quick_val_max_episodes": None,
    "quick_trap_max_episodes": None,
    "full_eval_every": 4,
    "agreement_eval_every": 4,
    "full_eval_on_improve": True,
}

QUICK_TREND_PROFILE = {
    "total_updates": 96,
    "scenarios_per_update": 24,
    "rollouts_per_scenario": 4,
    "eval_every": 8,
    "min_updates_before_stop": 32,
    "early_stop_patience": 4,
    "lr": 2e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "sample_temperature": 1.25,
    "small_margin": 5.0,
    "bc_anchor_updates": 8,
    "bc_anchor_coef": 0.01,
    "anchor_batch_size": 64,
    "expert_episodes_per_family": 6,
    "trap_eval": True,
    "allow_weak_init": True,
    "quick_val_max_episodes": 56,
    "quick_trap_max_episodes": 8,
    "full_eval_every": 16,
    "agreement_eval_every": 16,
    "full_eval_on_improve": True,
}

PROFILE_BASELINE_DEFAULTS = {
    **FULL_FIDELITY_DEFAULTS,
    "trap_eval": True,
    "allow_weak_init": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Hetero scheduler with self-critical actor-only RL.")
    parser.add_argument("--profile", choices=["full_fidelity", "quick_trend"], default="full_fidelity")
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--total-updates", type=int, default=FULL_FIDELITY_DEFAULTS["total_updates"])
    parser.add_argument("--scenarios-per-update", type=int, default=FULL_FIDELITY_DEFAULTS["scenarios_per_update"])
    parser.add_argument("--rollouts-per-scenario", type=int, default=FULL_FIDELITY_DEFAULTS["rollouts_per_scenario"])
    parser.add_argument("--lr", type=float, default=FULL_FIDELITY_DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=FULL_FIDELITY_DEFAULTS["weight_decay"])
    parser.add_argument("--max-grad-norm", type=float, default=FULL_FIDELITY_DEFAULTS["max_grad_norm"])
    parser.add_argument("--sample-temperature", type=float, default=FULL_FIDELITY_DEFAULTS["sample_temperature"])
    parser.add_argument("--small-margin", type=float, default=FULL_FIDELITY_DEFAULTS["small_margin"])
    parser.add_argument("--eval-every", type=int, default=FULL_FIDELITY_DEFAULTS["eval_every"])
    parser.add_argument("--limit-train-per-family", type=int, default=None)
    parser.add_argument("--limit-val-per-family", type=int, default=None)
    parser.add_argument("--save-dir", default="协同调度/checkpoints_hetero_actor_only")
    parser.add_argument("--init-model", default="协同调度/checkpoints_hetero_bc/best_scheduler_bc.pt")
    parser.add_argument("--expert-episodes-per-family", type=int, default=FULL_FIDELITY_DEFAULTS["expert_episodes_per_family"])
    parser.add_argument(
        "--anchor-expert-policy",
        default="hybrid_upfront_teacher",
        choices=["upfront_wait_aware_greedy", "rollout_upfront_teacher", "hybrid_upfront_teacher"],
    )
    parser.add_argument("--teacher-rollout-depth", type=int, default=2)
    parser.add_argument("--bc-anchor-updates", type=int, default=FULL_FIDELITY_DEFAULTS["bc_anchor_updates"])
    parser.add_argument("--bc-anchor-coef", type=float, default=FULL_FIDELITY_DEFAULTS["bc_anchor_coef"])
    parser.add_argument("--anchor-batch-size", type=int, default=FULL_FIDELITY_DEFAULTS["anchor_batch_size"])
    parser.add_argument("--early-stop-patience", type=int, default=FULL_FIDELITY_DEFAULTS["early_stop_patience"])
    parser.add_argument("--min-updates-before-stop", type=int, default=FULL_FIDELITY_DEFAULTS["min_updates_before_stop"])
    parser.add_argument("--quick-val-max-episodes", type=int, default=FULL_FIDELITY_DEFAULTS["quick_val_max_episodes"])
    parser.add_argument("--quick-trap-max-episodes", type=int, default=FULL_FIDELITY_DEFAULTS["quick_trap_max_episodes"])
    parser.add_argument("--full-eval-every", type=int, default=FULL_FIDELITY_DEFAULTS["full_eval_every"])
    parser.add_argument("--agreement-eval-every", type=int, default=FULL_FIDELITY_DEFAULTS["agreement_eval_every"])
    parser.add_argument(
        "--full-eval-on-improve",
        action=argparse.BooleanOptionalAction,
        default=FULL_FIDELITY_DEFAULTS["full_eval_on_improve"],
    )
    parser.add_argument("--allow-weak-init", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--trap-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate partial_coalition_trap subset separately.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return apply_profile_defaults(args)


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.profile != "quick_trend":
        return args
    for key, value in QUICK_TREND_PROFILE.items():
        if getattr(args, key) == PROFILE_BASELINE_DEFAULTS.get(key):
            setattr(args, key, value)
    return args


def actor_only_curriculum(progress: float) -> list[str]:
    if progress < (1.0 / 6.0):
        return ["open_balance", "role_mismatch", "single_bottleneck"]
    if progress < (2.0 / 6.0):
        return ["open_balance", "role_mismatch", "single_bottleneck", "far_near_trap"]
    if progress < (3.0 / 6.0):
        return ["open_balance", "role_mismatch", "single_bottleneck", "far_near_trap", "partial_coalition_trap"]
    if progress < (4.0 / 6.0):
        return [
            "open_balance",
            "role_mismatch",
            "single_bottleneck",
            "far_near_trap",
            "partial_coalition_trap",
            "double_bottleneck",
        ]
    if progress < (5.0 / 6.0):
        return [
            "open_balance",
            "role_mismatch",
            "single_bottleneck",
            "far_near_trap",
            "partial_coalition_trap",
            "double_bottleneck",
            "multi_sync_cluster",
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


def oversampled_family_pool(families: Sequence[str]) -> list[str]:
    weighted: list[str] = []
    for family in families:
        weight = 2 if family in {"far_near_trap", "partial_coalition_trap", "double_bottleneck", "multi_sync_cluster"} else 1
        weighted.extend([family] * weight)
    return weighted


def sample_scenarios_for_update(
    train_by_family: Dict[str, Sequence[Dict]],
    families: Sequence[str],
    count: int,
    rng: np.random.Generator,
) -> list[Dict]:
    weighted_families = oversampled_family_pool(families)
    sampled: list[Dict] = []
    for _ in range(count):
        family = weighted_families[int(rng.integers(len(weighted_families)))]
        family_scenarios = train_by_family.get(family) or []
        if not family_scenarios:
            continue
        sampled.append(family_scenarios[int(rng.integers(len(family_scenarios)))])
    return sampled


def collect_expert_pools(
    train_by_family: Dict[str, Sequence[Dict]],
    max_episodes_per_family: int,
    expert_policy: str,
    teacher_rollout_depth: int,
) -> Dict[str, List[tuple[Dict[str, np.ndarray], int]]]:
    pools: Dict[str, List[tuple[Dict[str, np.ndarray], int]]] = {}
    for family, scenarios in train_by_family.items():
        if not scenarios:
            continue
        pools[family] = collect_expert_samples(
            scenarios,
            expert_policy=expert_policy,
            max_episodes=max_episodes_per_family,
            seed=hash(family) % 10000,
            teacher_rollout_depth=teacher_rollout_depth,
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


def metric_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["success_rate"] * 1500.0
        - metrics["mean_makespan"]
        - 0.4 * metrics["mean_wait_time"]
        - 0.8 * metrics["mean_avoidable_wait_time"]
        - 80.0 * metrics["mean_direct_sync_misassignment_rate"]
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
        "policy_type": "hetero_actor_only",
        "stage": "actor_only_finetune",
        "update": update,
        "families": list(families),
        "val_metrics": val_metrics,
    }
    if trap_val_metrics is not None:
        metadata["trap_val_metrics"] = trap_val_metrics
    if extra:
        metadata.update(extra)
    return metadata


def episode_return(final_info: Dict[str, float], step_shaping_total: float, done: bool, truncated: bool) -> float:
    success = 1.0 if done and not truncated else 0.0
    completion_rate = float(final_info.get("completion_rate", 0.0))
    makespan = float(final_info.get("makespan", 0.0))
    wait_time = float(final_info.get("average_wait_time", 0.0))
    avoidable_wait = float(final_info.get("average_avoidable_wait_time", 0.0))
    misassign_rate = float(final_info.get("direct_sync_misassignment_rate", 0.0))
    activation_delay = float(final_info.get("coalition_activation_delay", 0.0))
    deadlock_events = float(final_info.get("deadlock_events", 0.0))
    return (
        2000.0 * success
        + 500.0 * completion_rate
        - 1.0 * makespan
        - 0.4 * wait_time
        - 0.8 * avoidable_wait
        - 80.0 * misassign_rate
        - 0.05 * activation_delay
        - 100.0 * deadlock_events
        + float(step_shaping_total)
    )


def rollout_once(
    env: HeteroDispatchEnv,
    scenario: Dict,
    policy: HeteroActorOnlyPolicy,
    device: torch.device,
    *,
    deterministic: bool,
    sample_temperature: float,
) -> Dict:
    obs, _ = env.reset(options={"scenario": scenario})
    done = False
    truncated = False
    final_info: Dict = {}
    obs_sequence: list[Dict[str, np.ndarray]] = []
    action_sequence: list[int] = []
    step_shaping_total = 0.0
    policy.eval()

    while not (done or truncated):
        obs_sequence.append(clone_obs(obs))
        obs_tensor = obs_to_torch(obs, device=device)
        with torch.inference_mode():
            _, logps = policy(obs_tensor)
            if deterministic:
                action = torch.argmax(logps, dim=-1)
            else:
                dist = Categorical(logits=logps / max(sample_temperature, 1e-6))
                action = dist.sample()
        action_int = int(action.item())
        action_sequence.append(action_int)
        obs, _, done, truncated, final_info = env.step(action_int)
        step_shaping_total += float(final_info.get("step_shaping_reward", 0.0))

    total_return = episode_return(final_info, step_shaping_total, done, truncated)
    return {
        "observations": obs_sequence,
        "actions": action_sequence,
        "return": total_return,
        "final_info": dict(final_info),
        "step_shaping_total": step_shaping_total,
    }


def log_prob_sum_for_rollout(
    model: HeteroActorOnlyPolicy,
    rollout: Dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = rollout["observations"]
    if not observations:
        zero = torch.zeros((), device=device)
        return zero, zero
    obs_batch_np = {
        key: np.stack([obs[key] for obs in observations], axis=0)
        for key in observations[0].keys()
    }
    obs_batch = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in obs_batch_np.items()
    }
    obs_batch["current_agent_index"] = obs_batch["current_agent_index"].long()
    actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
    log_prob, entropy = model.evaluate_actions(obs_batch, actions)
    return log_prob.sum(), entropy.mean()


def batched_log_prob_sums_for_rollouts(
    model: HeteroActorOnlyPolicy,
    rollouts: Sequence[Dict],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if not rollouts:
        return []

    non_empty_rollouts: list[Dict] = []
    lengths: list[int] = []
    obs_by_key: Dict[str, list[np.ndarray]] | None = None
    all_actions: list[int] = []
    results: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(rollouts)

    for rollout_index, rollout in enumerate(rollouts):
        observations = rollout["observations"]
        if not observations:
            zero = torch.zeros((), device=device)
            results[rollout_index] = (zero, zero)
            continue
        non_empty_rollouts.append(rollout)
        lengths.append(len(observations))
        if obs_by_key is None:
            obs_by_key = {key: [] for key in observations[0].keys()}
        for obs in observations:
            for key in obs_by_key.keys():
                obs_by_key[key].append(obs[key])
        all_actions.extend(int(action) for action in rollout["actions"])

    if not non_empty_rollouts:
        return [(torch.zeros((), device=device), torch.zeros((), device=device)) for _ in rollouts]

    assert obs_by_key is not None
    obs_batch_np = {
        key: np.stack(values, axis=0)
        for key, values in obs_by_key.items()
    }
    obs_batch = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in obs_batch_np.items()
    }
    obs_batch["current_agent_index"] = obs_batch["current_agent_index"].long()
    actions = torch.as_tensor(all_actions, dtype=torch.long, device=device)
    log_prob, entropy = model.evaluate_actions(obs_batch, actions)
    log_prob_splits = torch.split(log_prob, lengths)
    entropy_splits = torch.split(entropy, lengths)

    non_empty_index = 0
    for rollout_index, rollout in enumerate(rollouts):
        if results[rollout_index] is not None:
            continue
        results[rollout_index] = (
            log_prob_splits[non_empty_index].sum(),
            entropy_splits[non_empty_index].mean(),
        )
        non_empty_index += 1

    return [item for item in results if item is not None]


def _action_disagreement(sample_rollout: Dict, greedy_rollout: Dict) -> float:
    sample_actions = sample_rollout["actions"]
    greedy_actions = greedy_rollout["actions"]
    if not sample_actions or not greedy_actions:
        return 0.0
    count = min(len(sample_actions), len(greedy_actions))
    mismatches = sum(int(sample_actions[index] != greedy_actions[index]) for index in range(count))
    mismatches += abs(len(sample_actions) - len(greedy_actions))
    return mismatches / max(len(sample_actions), len(greedy_actions), 1)


def summarize_rollouts(
    sampled_rollouts: Sequence[Dict],
    greedy_rollouts: Sequence[Dict],
    advantages: Sequence[float],
) -> Dict[str, float]:
    if not sampled_rollouts:
        return {
            "mean_return": 0.0,
            "mean_sample_return": 0.0,
            "mean_greedy_return": 0.0,
            "mean_greedy_gap": 0.0,
            "mean_abs_advantage": 0.0,
            "mean_sampled_vs_greedy_action_disagreement": 0.0,
            "mean_wait_action_rate": 0.0,
            "mean_idle_wait_rate": 0.0,
            "mean_waiting_idle_wait_rate": 0.0,
            "mean_stalled_wait_rate": 0.0,
            "mean_wait_flip_rate": 0.0,
            "mean_dispatch_gap_penalty": 0.0,
            "mean_direct_sync_misassignment_rate": 0.0,
        }
    return {
        "mean_return": float(np.mean([item["return"] for item in sampled_rollouts])),
        "mean_sample_return": float(np.mean([item["return"] for item in sampled_rollouts])),
        "mean_greedy_return": float(np.mean([item["return"] for item in greedy_rollouts])) if greedy_rollouts else 0.0,
        "mean_greedy_gap": float(
            np.mean([sample["return"] - greedy["return"] for sample, greedy in zip(sampled_rollouts, greedy_rollouts)])
        ) if greedy_rollouts else 0.0,
        "mean_abs_advantage": float(np.mean(np.abs(np.asarray(advantages, dtype=np.float32)))) if advantages else 0.0,
        "mean_sampled_vs_greedy_action_disagreement": float(
            np.mean([_action_disagreement(sample, greedy) for sample, greedy in zip(sampled_rollouts, greedy_rollouts)])
        ) if greedy_rollouts else 0.0,
        "mean_wait_action_rate": float(np.mean([item["final_info"].get("wait_action_rate", 0.0) for item in sampled_rollouts])),
        "mean_idle_wait_rate": float(np.mean([item["final_info"].get("idle_wait_rate", 0.0) for item in sampled_rollouts])),
        "mean_waiting_idle_wait_rate": float(
            np.mean([item["final_info"].get("waiting_idle_wait_rate", 0.0) for item in sampled_rollouts])
        ),
        "mean_stalled_wait_rate": float(np.mean([item["final_info"].get("stalled_wait_rate", 0.0) for item in sampled_rollouts])),
        "mean_wait_flip_rate": float(np.mean([item["final_info"].get("wait_flip_rate", 0.0) for item in sampled_rollouts])),
        "mean_dispatch_gap_penalty": float(
            np.mean([item["final_info"].get("dispatch_gap_penalty", 0.0) for item in sampled_rollouts])
        ),
        "mean_direct_sync_misassignment_rate": float(
            np.mean([item["final_info"].get("direct_sync_misassignment_rate", 0.0) for item in sampled_rollouts])
        ),
    }


def _agreement_on_records(
    model: HeteroActorOnlyPolicy,
    records: Sequence[Dict[str, object]],
    device: torch.device,
    batch_size: int,
) -> float:
    if not records:
        return 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = list(records[start:start + batch_size])
            obs_batch_np = {
                key: np.stack([record["obs"][key] for record in batch], axis=0)
                for key in batch[0]["obs"].keys()
            }
            obs_batch = {
                key: torch.as_tensor(value, dtype=torch.float32, device=device)
                for key, value in obs_batch_np.items()
            }
            obs_batch["current_agent_index"] = obs_batch["current_agent_index"].long()
            probs, _ = model(obs_batch)
            predictions = torch.argmax(probs, dim=-1)
            actions = torch.as_tensor([int(record["action"]) for record in batch], dtype=torch.long, device=device)
            correct += int((predictions == actions).sum().item())
            total += len(batch)
    model.train()
    return correct / max(total, 1)


def _temperature_for_update(update: int, args: argparse.Namespace) -> float:
    if update <= 20:
        return max(args.sample_temperature, 1.5)
    if update <= 60:
        return args.sample_temperature
    return min(args.sample_temperature, 1.0)


def _evaluate_metrics(
    *,
    model: HeteroActorOnlyPolicy,
    val_scenarios: Sequence[Dict],
    trap_val_scenarios: Sequence[Dict],
    device: torch.device,
    val_max_episodes: int | None,
    trap_max_episodes: int | None,
) -> tuple[Dict[str, float], Dict[str, float] | None]:
    val_metrics = evaluate_policy_on_scenarios(
        scenarios=val_scenarios,
        policy=model,
        device=device,
        max_episodes=val_max_episodes,
        deterministic=True,
    )
    trap_val_metrics = None
    if trap_val_scenarios:
        trap_val_metrics = evaluate_policy_on_scenarios(
            scenarios=trap_val_scenarios,
            policy=model,
            device=device,
            max_episodes=trap_max_episodes,
            deterministic=True,
        )
    return val_metrics, trap_val_metrics


def _evaluate_teacher_agreements(
    *,
    model: HeteroActorOnlyPolicy,
    val_teacher_records: Sequence[Dict[str, object]],
    hard_val_teacher_records: Sequence[Dict[str, object]],
    conflict_val_teacher_records: Sequence[Dict[str, object]],
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, float]:
    hard_teacher_agreement = _agreement_on_records(
        model,
        hard_val_teacher_records,
        device=device,
        batch_size=batch_size,
    )
    teacher_agreement = _agreement_on_records(
        model,
        val_teacher_records,
        device=device,
        batch_size=batch_size,
    )
    conflict_teacher_agreement = _agreement_on_records(
        model,
        conflict_val_teacher_records,
        device=device,
        batch_size=batch_size,
    )
    return teacher_agreement, hard_teacher_agreement, conflict_teacher_agreement


def _should_run_full_eval(update: int, args: argparse.Namespace, quick_improved: bool) -> bool:
    if update == args.total_updates:
        return True
    if args.full_eval_on_improve and quick_improved:
        return True
    return args.full_eval_every > 0 and update % args.full_eval_every == 0


def _should_run_agreement_eval(
    update: int,
    args: argparse.Namespace,
    *,
    quick_improved: bool,
    run_full_eval: bool,
) -> bool:
    if not run_full_eval:
        return False
    if update == args.total_updates:
        return True
    if args.full_eval_on_improve and quick_improved:
        return True
    return args.agreement_eval_every > 0 and update % args.agreement_eval_every == 0


def _format_optional_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "skip"


def log_eval(
    update: int,
    families: Sequence[str],
    train_summary: Dict[str, float],
    policy_loss: float,
    anchor_loss: float,
    val_metrics: Dict[str, float],
    trap_val_metrics: Dict[str, float] | None,
    teacher_agreement: float | None,
    hard_teacher_agreement: float | None,
    conflict_teacher_agreement: float | None,
    eval_label: str,
) -> None:
    line = (
        f"[HAO][update {update:03d}][{eval_label}] "
        f"families={','.join(families)} "
        f"return={train_summary['mean_return']:.2f} "
        f"sample_return={train_summary['mean_sample_return']:.2f} "
        f"greedy_return={train_summary['mean_greedy_return']:.2f} "
        f"greedy_gap={train_summary['mean_greedy_gap']:.2f} "
        f"abs_adv={train_summary['mean_abs_advantage']:.2f} "
        f"action_disagree={train_summary['mean_sampled_vs_greedy_action_disagreement']:.3f} "
        f"policy={policy_loss:.4f} "
        f"anchor={anchor_loss:.4f} "
        f"teacher_agreement={_format_optional_metric(teacher_agreement)} "
        f"hard_teacher_agreement={_format_optional_metric(hard_teacher_agreement)} "
        f"single_vs_sync_conflict_agreement={_format_optional_metric(conflict_teacher_agreement)} "
        f"success={val_metrics['success_rate']:.3f} "
        f"makespan={val_metrics['mean_makespan']:.2f} "
        f"wait={val_metrics['mean_wait_time']:.2f} "
        f"avoidable_wait={val_metrics['mean_avoidable_wait_time']:.2f} "
        f"sync_misassign={val_metrics['mean_direct_sync_misassignment_rate']:.3f} "
        f"wait_rate={val_metrics['mean_wait_action_rate']:.3f} "
        f"idle_wait={val_metrics['mean_idle_wait_rate']:.3f} "
        f"waiting_idle_wait={val_metrics['mean_waiting_idle_wait_rate']:.3f} "
        f"stalled_wait={val_metrics['mean_stalled_wait_rate']:.3f} "
        f"wait_flip={val_metrics['mean_wait_flip_rate']:.3f} "
        f"dispatch_gap={val_metrics['mean_dispatch_gap_penalty']:.3f}"
    )
    if trap_val_metrics is not None:
        line += (
            f" trap_success={trap_val_metrics['success_rate']:.3f}"
            f" trap_avoidable_wait={trap_val_metrics['mean_avoidable_wait_time']:.2f}"
            f" trap_sync_misassign={trap_val_metrics['mean_direct_sync_misassignment_rate']:.3f}"
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
    trap_val_scenarios: list[Dict] = []
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

    expert_pools = collect_expert_pools(
        train_by_family,
        max_episodes_per_family=args.expert_episodes_per_family,
        expert_policy=args.anchor_expert_policy,
        teacher_rollout_depth=args.teacher_rollout_depth,
    )
    val_teacher_records = collect_expert_sample_records(
        val_scenarios,
        expert_policy=args.anchor_expert_policy,
        teacher_rollout_depth=args.teacher_rollout_depth,
    )
    hard_val_teacher_records = [record for record in val_teacher_records if bool(record["hard_state"])]
    conflict_val_teacher_records = [record for record in val_teacher_records if bool(record["single_sync_conflict"])]

    if args.init_model:
        model, checkpoint = load_hetero_actor_only_checkpoint(args.init_model, device=device)
        print(f"加载初始模型: {args.init_model}")
        metadata = checkpoint.get("metadata", {})
    else:
        bootstrap_scenarios = next(iter([value for value in train_by_family.values() if value]))
        bootstrap_env = HeteroDispatchEnv(scenarios=bootstrap_scenarios)
        bootstrap_obs, _ = bootstrap_env.reset()
        model = HeteroActorOnlyPolicy(
            agent_input_dim=int(bootstrap_obs["agent_inputs"].shape[-1]),
            task_input_dim=int(bootstrap_obs["task_inputs"].shape[-1]),
        ).to(device)
        metadata = {}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_env = HeteroDispatchEnv(scenario_dir=args.scenario_dir, split=args.train_split)

    initial_families = actor_only_curriculum(0.0)
    initial_val_metrics, initial_trap_metrics = _evaluate_metrics(
        model=model,
        val_scenarios=val_scenarios,
        trap_val_scenarios=trap_val_scenarios,
        device=device,
        val_max_episodes=None,
        trap_max_episodes=None,
    )
    (
        initial_teacher_agreement,
        initial_hard_teacher_agreement,
        initial_conflict_teacher_agreement,
    ) = _evaluate_teacher_agreements(
        model=model,
        val_teacher_records=val_teacher_records,
        hard_val_teacher_records=hard_val_teacher_records,
        conflict_val_teacher_records=conflict_val_teacher_records,
        device=device,
        batch_size=args.anchor_batch_size,
    )
    bc_ready = (
        initial_teacher_agreement >= 0.93
        and initial_hard_teacher_agreement >= 0.90
        and initial_conflict_teacher_agreement >= 0.60
        and initial_val_metrics["mean_avoidable_wait_time"] <= 4.0
        and (initial_trap_metrics is None or initial_trap_metrics["mean_avoidable_wait_time"] <= 1.5)
    )
    if not bc_ready and not args.allow_weak_init:
        raise ValueError(
            "Current BC initialization does not meet actor-only entry thresholds."
            f" teacher_agreement={initial_teacher_agreement:.3f},"
            f" hard_teacher_agreement={initial_hard_teacher_agreement:.3f},"
            f" single_vs_sync_conflict_agreement={initial_conflict_teacher_agreement:.3f},"
            f" avoidable_wait={initial_val_metrics['mean_avoidable_wait_time']:.2f},"
            f" trap_avoidable_wait={(initial_trap_metrics['mean_avoidable_wait_time'] if initial_trap_metrics else 0.0):.2f}."
            " Strengthen BC first, or pass --allow-weak-init for exploratory runs."
        )

    best_score = metric_score(initial_val_metrics)
    best_quick_score = best_score
    best_metrics = dict(initial_val_metrics)
    best_trap_metrics = dict(initial_trap_metrics) if initial_trap_metrics is not None else None
    best_teacher_agreement = initial_teacher_agreement
    best_hard_teacher_agreement = initial_hard_teacher_agreement
    best_conflict_teacher_agreement = initial_conflict_teacher_agreement
    no_improve_evals = 0
    initial_metadata = checkpoint_metadata(
        metadata,
        update=0,
        families=initial_families,
        val_metrics=initial_val_metrics,
        trap_val_metrics=initial_trap_metrics,
        extra={
            "train_mode": "actor_only",
            "profile": args.profile,
            "anchor_expert_policy": args.anchor_expert_policy,
            "teacher_rollout_depth": args.teacher_rollout_depth,
            "teacher_agreement": initial_teacher_agreement,
            "hard_teacher_agreement": initial_hard_teacher_agreement,
            "single_vs_sync_conflict_agreement": initial_conflict_teacher_agreement,
            "bc_ready_for_actor_only": bc_ready,
            "quick_val_max_episodes": args.quick_val_max_episodes,
            "quick_trap_max_episodes": args.quick_trap_max_episodes,
            "full_eval_every": args.full_eval_every,
            "agreement_eval_every": args.agreement_eval_every,
            "full_eval_on_improve": args.full_eval_on_improve,
        },
    )
    save_hetero_actor_only_checkpoint(save_dir / "latest_scheduler_actor_only.pt", model, optimizer=optimizer, metadata=initial_metadata)
    save_hetero_actor_only_checkpoint(save_dir / "best_scheduler_actor_only.pt", model, optimizer=optimizer, metadata=initial_metadata)
    log_eval(
        update=0,
        families=initial_families,
        train_summary={
            "mean_return": 0.0,
            "mean_sample_return": 0.0,
            "mean_greedy_return": 0.0,
            "mean_greedy_gap": 0.0,
            "mean_abs_advantage": 0.0,
            "mean_sampled_vs_greedy_action_disagreement": 0.0,
            "mean_wait_action_rate": initial_val_metrics["mean_wait_action_rate"],
            "mean_idle_wait_rate": initial_val_metrics["mean_idle_wait_rate"],
            "mean_waiting_idle_wait_rate": initial_val_metrics["mean_waiting_idle_wait_rate"],
            "mean_stalled_wait_rate": initial_val_metrics["mean_stalled_wait_rate"],
            "mean_wait_flip_rate": initial_val_metrics["mean_wait_flip_rate"],
            "mean_dispatch_gap_penalty": initial_val_metrics["mean_dispatch_gap_penalty"],
            "mean_direct_sync_misassignment_rate": initial_val_metrics["mean_direct_sync_misassignment_rate"],
        },
        policy_loss=0.0,
        anchor_loss=0.0,
        val_metrics=initial_val_metrics,
        trap_val_metrics=initial_trap_metrics,
        teacher_agreement=initial_teacher_agreement,
        hard_teacher_agreement=initial_hard_teacher_agreement,
        conflict_teacher_agreement=initial_conflict_teacher_agreement,
        eval_label="full",
    )

    for update in range(1, args.total_updates + 1):
        progress = float(update - 1) / max(args.total_updates, 1)
        active_families = actor_only_curriculum(progress)
        sampled_scenarios = sample_scenarios_for_update(
            train_by_family=train_by_family,
            families=active_families,
            count=args.scenarios_per_update,
            rng=rng,
        )
        if not sampled_scenarios:
            raise ValueError("No training scenarios sampled for the active curriculum families.")

        sampled_rollouts: list[Dict] = []
        greedy_rollouts: list[Dict] = []
        advantages: list[float] = []
        current_temperature = _temperature_for_update(update, args)
        model.eval()
        for scenario in sampled_scenarios:
            greedy_rollout = rollout_once(
                train_env,
                scenario=scenario,
                policy=model,
                device=device,
                deterministic=True,
                sample_temperature=current_temperature,
            )
            group: list[Dict] = []
            for _ in range(args.rollouts_per_scenario):
                rollout = rollout_once(
                    train_env,
                    scenario=scenario,
                    policy=model,
                    device=device,
                    deterministic=False,
                    sample_temperature=current_temperature,
                )
                group.append(rollout)
                sampled_rollouts.append(rollout)
                greedy_rollouts.append(greedy_rollout)

            group_returns = np.asarray([item["return"] for item in group], dtype=np.float32)
            group_mean = float(np.mean(group_returns))
            greedy_return = float(greedy_rollout["return"])

            for rollout, total_return in zip(group, group_returns):
                greedy_gap = float(total_return - greedy_return)
                if abs(greedy_gap) < args.small_margin:
                    advantage = float(total_return - group_mean)
                else:
                    advantage = greedy_gap
                advantages.append(advantage)

        if not sampled_rollouts:
            raise RuntimeError("No rollout loss terms were collected.")

        model.train()
        rollout_log_prob_summaries = batched_log_prob_sums_for_rollouts(model, sampled_rollouts, device=device)
        policy_terms = [
            -log_prob_sum * float(advantage)
            for (log_prob_sum, _), advantage in zip(rollout_log_prob_summaries, advantages)
        ]

        policy_loss = torch.stack(policy_terms).mean()
        anchor_loss_value = 0.0
        total_loss = policy_loss
        if update <= args.bc_anchor_updates and args.bc_anchor_coef > 0.0:
            anchor_batch = sample_anchor_batch(
                expert_pools=expert_pools,
                families=active_families,
                batch_size=args.anchor_batch_size,
                rng=rng,
            )
            if anchor_batch is not None:
                obs_batch_np, expert_actions = anchor_batch
                obs_batch = {
                    key: torch.as_tensor(value, dtype=torch.float32, device=device)
                    for key, value in obs_batch_np.items()
                }
                obs_batch["current_agent_index"] = obs_batch["current_agent_index"].long()
                expert_actions = expert_actions.to(device=device)
                _, logps = model(obs_batch)
                anchor_loss = F.nll_loss(logps, expert_actions)
                total_loss = total_loss + args.bc_anchor_coef * anchor_loss
                anchor_loss_value = float(anchor_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()

        train_summary = summarize_rollouts(sampled_rollouts, greedy_rollouts, advantages)
        latest_metadata = checkpoint_metadata(
            metadata,
            update=update,
            families=active_families,
            val_metrics=initial_val_metrics,
            trap_val_metrics=initial_trap_metrics,
            extra={
                "train_mode": "actor_only",
                "profile": args.profile,
                "anchor_expert_policy": args.anchor_expert_policy,
                "teacher_rollout_depth": args.teacher_rollout_depth,
                "train_policy_loss": float(policy_loss.item()),
                "train_anchor_loss": anchor_loss_value,
                "train_mean_return": train_summary["mean_return"],
                "train_mean_sample_return": train_summary["mean_sample_return"],
                "train_mean_greedy_return": train_summary["mean_greedy_return"],
                "train_mean_greedy_gap": train_summary["mean_greedy_gap"],
                "train_mean_abs_advantage": train_summary["mean_abs_advantage"],
                "sample_temperature": current_temperature,
                "quick_val_max_episodes": args.quick_val_max_episodes,
                "quick_trap_max_episodes": args.quick_trap_max_episodes,
                "full_eval_every": args.full_eval_every,
                "agreement_eval_every": args.agreement_eval_every,
                "full_eval_on_improve": args.full_eval_on_improve,
            },
        )
        save_hetero_actor_only_checkpoint(
            save_dir / "latest_scheduler_actor_only.pt",
            model,
            optimizer=optimizer,
            metadata=latest_metadata,
        )

        if update % args.eval_every != 0 and update != args.total_updates:
            continue

        quick_val_metrics, quick_trap_metrics = _evaluate_metrics(
            model=model,
            val_scenarios=val_scenarios,
            trap_val_scenarios=trap_val_scenarios,
            device=device,
            val_max_episodes=args.quick_val_max_episodes,
            trap_max_episodes=args.quick_trap_max_episodes,
        )
        quick_score = metric_score(quick_val_metrics)
        quick_improved = quick_score > best_quick_score + 1e-6
        if quick_improved:
            best_quick_score = quick_score
            no_improve_evals = 0
        else:
            no_improve_evals += 1

        run_full_eval = _should_run_full_eval(update, args, quick_improved)
        run_agreement_eval = _should_run_agreement_eval(
            update,
            args,
            quick_improved=quick_improved,
            run_full_eval=run_full_eval,
        )

        logged_val_metrics = quick_val_metrics
        logged_trap_metrics = quick_trap_metrics
        teacher_agreement: float | None = None
        hard_teacher_agreement: float | None = None
        conflict_teacher_agreement: float | None = None
        eval_label = "quick"

        if run_full_eval:
            logged_val_metrics, logged_trap_metrics = _evaluate_metrics(
                model=model,
                val_scenarios=val_scenarios,
                trap_val_scenarios=trap_val_scenarios,
                device=device,
                val_max_episodes=None,
                trap_max_episodes=None,
            )
            eval_label = "full"
            if run_agreement_eval:
                (
                    teacher_agreement,
                    hard_teacher_agreement,
                    conflict_teacher_agreement,
                ) = _evaluate_teacher_agreements(
                    model=model,
                    val_teacher_records=val_teacher_records,
                    hard_val_teacher_records=hard_val_teacher_records,
                    conflict_val_teacher_records=conflict_val_teacher_records,
                    device=device,
                    batch_size=args.anchor_batch_size,
                )

                score = metric_score(logged_val_metrics)
                gate_ok = logged_val_metrics["success_rate"] >= initial_val_metrics["success_rate"] - 0.01
                hard_improved = False
                if logged_trap_metrics is not None and best_trap_metrics is not None:
                    hard_improved = hard_improved or (
                        logged_trap_metrics["mean_avoidable_wait_time"] < best_trap_metrics["mean_avoidable_wait_time"] - 1e-6
                    )
                hard_improved = hard_improved or (
                    logged_val_metrics["mean_direct_sync_misassignment_rate"]
                    < best_metrics["mean_direct_sync_misassignment_rate"] - 1e-6
                )
                hard_improved = hard_improved or (teacher_agreement > best_teacher_agreement + 1e-6)
                hard_improved = hard_improved or (hard_teacher_agreement > best_hard_teacher_agreement + 1e-6)
                hard_improved = hard_improved or (conflict_teacher_agreement > best_conflict_teacher_agreement + 1e-6)

                eval_metadata = checkpoint_metadata(
                    metadata,
                    update=update,
                    families=active_families,
                    val_metrics=logged_val_metrics,
                    trap_val_metrics=logged_trap_metrics,
                    extra={
                        "train_mode": "actor_only",
                        "profile": args.profile,
                        "anchor_expert_policy": args.anchor_expert_policy,
                        "teacher_rollout_depth": args.teacher_rollout_depth,
                        "train_policy_loss": float(policy_loss.item()),
                        "train_anchor_loss": anchor_loss_value,
                        "train_mean_return": train_summary["mean_return"],
                        "train_mean_sample_return": train_summary["mean_sample_return"],
                        "train_mean_greedy_return": train_summary["mean_greedy_return"],
                        "train_mean_greedy_gap": train_summary["mean_greedy_gap"],
                        "train_mean_abs_advantage": train_summary["mean_abs_advantage"],
                        "teacher_agreement": teacher_agreement,
                        "hard_teacher_agreement": hard_teacher_agreement,
                        "single_vs_sync_conflict_agreement": conflict_teacher_agreement,
                        "sample_temperature": current_temperature,
                        "quick_val_max_episodes": args.quick_val_max_episodes,
                        "quick_trap_max_episodes": args.quick_trap_max_episodes,
                        "full_eval_every": args.full_eval_every,
                        "agreement_eval_every": args.agreement_eval_every,
                        "full_eval_on_improve": args.full_eval_on_improve,
                    },
                )
                save_hetero_actor_only_checkpoint(
                    save_dir / "latest_scheduler_actor_only.pt",
                    model,
                    optimizer=optimizer,
                    metadata=eval_metadata,
                )
                if gate_ok and hard_improved and score > best_score:
                    best_score = score
                    best_metrics = dict(logged_val_metrics)
                    best_trap_metrics = dict(logged_trap_metrics) if logged_trap_metrics is not None else None
                    best_teacher_agreement = teacher_agreement
                    best_hard_teacher_agreement = hard_teacher_agreement
                    best_conflict_teacher_agreement = conflict_teacher_agreement
                    save_hetero_actor_only_checkpoint(
                        save_dir / "best_scheduler_actor_only.pt",
                        model,
                        optimizer=optimizer,
                        metadata=eval_metadata,
                    )

        log_eval(
            update=update,
            families=active_families,
            train_summary=train_summary,
            policy_loss=float(policy_loss.item()),
            anchor_loss=anchor_loss_value,
            val_metrics=logged_val_metrics,
            trap_val_metrics=logged_trap_metrics,
            teacher_agreement=teacher_agreement,
            hard_teacher_agreement=hard_teacher_agreement,
            conflict_teacher_agreement=conflict_teacher_agreement,
            eval_label=eval_label,
        )

        if update >= args.min_updates_before_stop and no_improve_evals >= args.early_stop_patience:
            print(f"[HAO] 提前停止: 连续 {no_improve_evals} 次评估未提升。")
            break


if __name__ == "__main__":
    main()
