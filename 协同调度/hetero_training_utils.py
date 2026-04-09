from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from hetero_attention_policy import HeteroActorOnlyPolicy, HeteroAttentionSchedulerPolicy, HeteroRankerPolicy, obs_to_torch
from hetero_dispatch_env import HeteroDispatchEnv, TASK_IDX


HeteroPolicy = HeteroAttentionSchedulerPolicy | HeteroActorOnlyPolicy | HeteroRankerPolicy
HARD_SAMPLE_FAMILIES = {
    "role_mismatch",
    "single_bottleneck",
    "double_bottleneck",
    "multi_sync_cluster",
    "partial_coalition_trap",
}


def clone_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in obs.items()}


def _is_action_legal(obs: Dict[str, np.ndarray], action: int) -> bool:
    if action < 0 or action >= int(obs["global_mask"].shape[0]):
        return False
    return bool(obs["global_mask"][action] < 0.5)


def _fallback_legal_action(obs: Dict[str, np.ndarray]) -> int:
    legal_actions = [int(action) for action in np.flatnonzero(obs["global_mask"] < 0.5)]
    non_wait_actions = [action for action in legal_actions if action != 0]
    if non_wait_actions:
        return non_wait_actions[0]
    if legal_actions:
        return legal_actions[0]
    return 0


def legal_non_wait_actions(obs: Dict[str, np.ndarray]) -> list[int]:
    return [int(action) for action in np.flatnonzero(obs["global_mask"] < 0.5) if int(action) != 0]


def legal_action_count(obs: Dict[str, np.ndarray]) -> int:
    return int(len(legal_non_wait_actions(obs)))


def single_sync_conflict(obs: Dict[str, np.ndarray]) -> bool:
    actions = legal_non_wait_actions(obs)
    if not actions:
        return False
    task_inputs = obs["task_inputs"]
    has_single = False
    has_sync = False
    for action in actions:
        row = task_inputs[action]
        if row[TASK_IDX["is_single"]] > 0.5:
            has_single = True
        if row[TASK_IDX["is_sync"]] > 0.5:
            has_sync = True
        if has_single and has_sync:
            return True
    return False


def is_hard_sample(family: str, obs: Dict[str, np.ndarray]) -> bool:
    if family in HARD_SAMPLE_FAMILIES:
        return True
    if legal_action_count(obs) > 2:
        return True
    if single_sync_conflict(obs):
        return True
    return False


def sample_metadata(family: str, obs: Dict[str, np.ndarray], action: int) -> Dict[str, object]:
    return {
        "family": family,
        "obs": clone_obs(obs),
        "action": int(action),
        "legal_action_count": legal_action_count(obs),
        "single_sync_conflict": single_sync_conflict(obs),
        "hard_state": is_hard_sample(family, obs),
    }


def ranker_sample_metadata(
    family: str,
    obs: Dict[str, np.ndarray],
    action: int,
    candidate_scores: Dict[int, float],
) -> Dict[str, object]:
    record = sample_metadata(family, obs, action)
    ordered_candidates = sorted(
        ((int(candidate), float(score)) for candidate, score in candidate_scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    record["candidate_actions"] = [candidate for candidate, _ in ordered_candidates]
    record["candidate_scores"] = [score for _, score in ordered_candidates]
    return record


def policy_action(
    env: HeteroDispatchEnv,
    obs: Dict[str, np.ndarray],
    policy: str | HeteroPolicy,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    teacher_rollout_depth: int = 2,
) -> int:
    if isinstance(policy, str):
        if policy == "upfront_wait_aware_greedy":
            return int(env.upfront_wait_aware_action())
        if policy == "rollout_upfront_teacher":
            return int(env.rollout_upfront_teacher_action(rollout_depth=teacher_rollout_depth))
        if policy == "hybrid_upfront_teacher":
            return int(env.hybrid_upfront_teacher_action(rollout_depth=teacher_rollout_depth))
        if policy == "random":
            valid = np.flatnonzero(obs["global_mask"] < 0.5)
            return int(rng.choice(valid)) if len(valid) else 0
        raise ValueError(f"未知策略: {policy}")

    obs_tensors = obs_to_torch(obs, device=device)
    policy.eval()
    with torch.no_grad():
        action, _, _ = policy.act(obs_tensors, deterministic=deterministic)
    return int(action.item())


def _episode_metrics(final_info: Dict, env: HeteroDispatchEnv, done: bool, truncated: bool) -> Dict[str, float]:
    return {
        "success": float(done and not truncated),
        "makespan": float(final_info.get("makespan", env.base_env.time)),
        "completion_rate": float(final_info.get("completion_rate", 0.0)),
        "average_wait_time": float(final_info.get("average_wait_time", 0.0)),
        "average_avoidable_wait_time": float(final_info.get("average_avoidable_wait_time", 0.0)),
        "idle_ratio": float(final_info.get("idle_ratio", 0.0)),
        "deadlock_events": float(final_info.get("deadlock_events", 0.0)),
        "coalition_activation_delay": float(final_info.get("coalition_activation_delay", 0.0)),
        "direct_sync_misassignment_rate": float(final_info.get("direct_sync_misassignment_rate", 0.0)),
        "wait_action_rate": float(final_info.get("wait_action_rate", 0.0)),
        "idle_wait_rate": float(final_info.get("idle_wait_rate", 0.0)),
        "waiting_idle_wait_rate": float(final_info.get("waiting_idle_wait_rate", 0.0)),
        "stalled_wait_rate": float(final_info.get("stalled_wait_rate", 0.0)),
        "wait_flip_rate": float(final_info.get("wait_flip_rate", 0.0)),
        "dispatch_gap_penalty": float(final_info.get("dispatch_gap_penalty", 0.0)),
    }


def evaluate_policy_on_scenarios(
    scenarios: Sequence[Dict],
    policy: str | HeteroPolicy,
    device: torch.device | str = "cpu",
    max_episodes: int | None = None,
    deterministic: bool = True,
    seed: int = 0,
    teacher_rollout_depth: int = 2,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    env = HeteroDispatchEnv(scenarios=scenarios)
    episode_count = min(len(scenarios), max_episodes) if max_episodes is not None else len(scenarios)
    metrics = []

    for episode_index in range(episode_count):
        obs, _ = env.reset(options={"scenario": scenarios[episode_index]})
        done = False
        truncated = False
        final_info = {}

        while not (done or truncated):
            action = policy_action(
                env,
                obs,
                policy=policy,
                rng=rng,
                device=device,
                deterministic=deterministic,
                teacher_rollout_depth=teacher_rollout_depth,
            )
            obs, _, done, truncated, final_info = env.step(action)

        metrics.append(_episode_metrics(final_info, env, done, truncated))

    return {
        "episodes": float(episode_count),
        "success_rate": mean(item["success"] for item in metrics),
        "mean_makespan": mean(item["makespan"] for item in metrics),
        "mean_completion_rate": mean(item["completion_rate"] for item in metrics),
        "mean_wait_time": mean(item["average_wait_time"] for item in metrics),
        "mean_avoidable_wait_time": mean(item["average_avoidable_wait_time"] for item in metrics),
        "mean_idle_ratio": mean(item["idle_ratio"] for item in metrics),
        "mean_deadlock_events": mean(item["deadlock_events"] for item in metrics),
        "mean_coalition_activation_delay": mean(item["coalition_activation_delay"] for item in metrics),
        "mean_direct_sync_misassignment_rate": mean(item["direct_sync_misassignment_rate"] for item in metrics),
        "mean_wait_action_rate": mean(item["wait_action_rate"] for item in metrics),
        "mean_idle_wait_rate": mean(item["idle_wait_rate"] for item in metrics),
        "mean_waiting_idle_wait_rate": mean(item["waiting_idle_wait_rate"] for item in metrics),
        "mean_stalled_wait_rate": mean(item["stalled_wait_rate"] for item in metrics),
        "mean_wait_flip_rate": mean(item["wait_flip_rate"] for item in metrics),
        "mean_dispatch_gap_penalty": mean(item["dispatch_gap_penalty"] for item in metrics),
    }


def evaluate_policy_family_breakdown(
    scenarios: Sequence[Dict],
    policy: str | HeteroPolicy,
    device: torch.device | str = "cpu",
    max_episodes: int | None = None,
    deterministic: bool = True,
    seed: int = 0,
    teacher_rollout_depth: int = 2,
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for scenario in scenarios:
        grouped[str(scenario.get("family", "unknown"))].append(scenario)

    breakdown: Dict[str, Dict[str, float]] = {}
    for family, family_scenarios in grouped.items():
        breakdown[family] = evaluate_policy_on_scenarios(
            scenarios=family_scenarios,
            policy=policy,
            device=device,
            max_episodes=max_episodes,
            deterministic=deterministic,
            seed=seed,
            teacher_rollout_depth=teacher_rollout_depth,
        )
    return breakdown


def collect_expert_samples(
    scenarios: Sequence[Dict],
    expert_policy: str = "upfront_wait_aware_greedy",
    max_episodes: int | None = None,
    seed: int = 0,
    teacher_rollout_depth: int = 2,
) -> List[tuple[Dict[str, np.ndarray], int]]:
    rng = np.random.default_rng(seed)
    env = HeteroDispatchEnv(scenarios=scenarios)
    episode_count = min(len(scenarios), max_episodes) if max_episodes is not None else len(scenarios)
    samples: List[tuple[Dict[str, np.ndarray], int]] = []
    invalid_action_count = 0
    corrected_wait_count = 0

    for episode_index in range(episode_count):
        obs, _ = env.reset(options={"scenario": scenarios[episode_index]})
        done = False
        truncated = False
        while not (done or truncated):
            action = policy_action(
                env,
                obs,
                policy=expert_policy,
                rng=rng,
                deterministic=True,
                teacher_rollout_depth=teacher_rollout_depth,
            )
            if not _is_action_legal(obs, action):
                invalid_action_count += 1
                if action == 0:
                    corrected_wait_count += 1
                action = _fallback_legal_action(obs)
            samples.append((clone_obs(obs), int(action)))
            obs, _, done, truncated, _ = env.step(action)
    if invalid_action_count > 0:
        print(
            "[Hetero expert] corrected "
            f"{invalid_action_count} invalid labels "
            f"(wait->task: {corrected_wait_count})"
        )
    return samples


def collect_expert_sample_records(
    scenarios: Sequence[Dict],
    expert_policy: str = "upfront_wait_aware_greedy",
    max_episodes: int | None = None,
    seed: int = 0,
    teacher_rollout_depth: int = 2,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    env = HeteroDispatchEnv(scenarios=scenarios)
    episode_count = min(len(scenarios), max_episodes) if max_episodes is not None else len(scenarios)
    records: List[Dict[str, object]] = []
    invalid_action_count = 0

    for episode_index in range(episode_count):
        scenario = scenarios[episode_index]
        family = str(scenario.get("family", "unknown"))
        obs, _ = env.reset(options={"scenario": scenario})
        done = False
        truncated = False
        while not (done or truncated):
            action = policy_action(
                env,
                obs,
                policy=expert_policy,
                rng=rng,
                deterministic=True,
                teacher_rollout_depth=teacher_rollout_depth,
            )
            if not _is_action_legal(obs, action):
                invalid_action_count += 1
                action = _fallback_legal_action(obs)
            records.append(sample_metadata(family, obs, int(action)))
            obs, _, done, truncated, _ = env.step(action)
    if invalid_action_count > 0:
        print(f"[Hetero expert] corrected {invalid_action_count} invalid labels in record collection")
    return records


def collect_ranker_sample_records(
    scenarios: Sequence[Dict],
    expert_policy: str = "hybrid_upfront_teacher",
    max_episodes: int | None = None,
    seed: int = 0,
    teacher_rollout_depth: int = 2,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    env = HeteroDispatchEnv(scenarios=scenarios)
    episode_count = min(len(scenarios), max_episodes) if max_episodes is not None else len(scenarios)
    records: List[Dict[str, object]] = []
    invalid_action_count = 0

    for episode_index in range(episode_count):
        scenario = scenarios[episode_index]
        family = str(scenario.get("family", "unknown"))
        obs, _ = env.reset(options={"scenario": scenario})
        done = False
        truncated = False
        while not (done or truncated):
            action = policy_action(
                env,
                obs,
                policy=expert_policy,
                rng=rng,
                deterministic=True,
                teacher_rollout_depth=teacher_rollout_depth,
            )
            if not _is_action_legal(obs, action):
                invalid_action_count += 1
                action = _fallback_legal_action(obs)
            candidate_scores = env.teacher_candidate_scores(rollout_depth=teacher_rollout_depth, top_k=2)
            if int(action) not in candidate_scores:
                candidate_scores[int(action)] = float(np.max(list(candidate_scores.values())) if candidate_scores else 0.0)
            records.append(ranker_sample_metadata(family, obs, int(action), candidate_scores))
            obs, _, done, truncated, _ = env.step(action)
    if invalid_action_count > 0:
        print(f"[Hetero ranker teacher] corrected {invalid_action_count} invalid labels in record collection")
    return records


def mean_legal_action_count(samples: Sequence[tuple[Dict[str, np.ndarray], int]]) -> float:
    if not samples:
        return 0.0
    return float(np.mean([float(len(legal_non_wait_actions(obs))) for obs, _ in samples]))


class HeteroSupervisedDataset(Dataset):
    def __init__(self, samples: Sequence[tuple[Dict[str, np.ndarray], int]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Dict[str, np.ndarray], int]:
        return self.samples[index]


def collate_supervised_batch(batch: Iterable[tuple[Dict[str, np.ndarray], int]]):
    batch_list = list(batch)
    obs_keys = batch_list[0][0].keys()
    obs_batch = {
        key: np.stack([item[0][key] for item in batch_list], axis=0)
        for key in obs_keys
    }
    actions = np.asarray([item[1] for item in batch_list], dtype=np.int64)
    return obs_batch, actions
