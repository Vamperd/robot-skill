from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from attention_policy import AttentionSchedulerPolicy, obs_to_torch
from sequential_scheduling_env import SequentialSchedulingEnv


def clone_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in obs.items()}


def policy_action(
    env: SequentialSchedulingEnv,
    obs: Dict[str, np.ndarray],
    policy: str | AttentionSchedulerPolicy,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
) -> int:
    if isinstance(policy, str):
        if policy == "role_aware_greedy":
            return int(env.role_aware_action())
        if policy == "wait_aware_role_greedy":
            return int(env.wait_aware_action())
        if policy == "random":
            valid = np.flatnonzero(obs["current_action_mask"] > 0.0)
            return int(rng.choice(valid)) if len(valid) else 0
        raise ValueError(f"未知策略: {policy}")

    obs_tensors = obs_to_torch(obs, device=device)
    policy.eval()
    with torch.no_grad():
        action, _, _ = policy.act(obs_tensors, deterministic=deterministic)
    return int(action.item())


def evaluate_policy_on_scenarios(
    scenarios: Sequence[Dict],
    policy: str | AttentionSchedulerPolicy,
    device: torch.device | str = "cpu",
    max_episodes: int | None = None,
    deterministic: bool = True,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    env = SequentialSchedulingEnv(scenarios=scenarios)
    episode_count = min(len(scenarios), max_episodes) if max_episodes is not None else len(scenarios)
    metrics = []

    for episode_index in range(episode_count):
        obs, _ = env.reset(options={"scenario": scenarios[episode_index]})
        done = False
        truncated = False
        final_info = {}

        while not (done or truncated):
            action = policy_action(env, obs, policy=policy, rng=rng, device=device, deterministic=deterministic)
            obs, _, done, truncated, final_info = env.step(action)

        metrics.append(
            {
                "success": float(done and not truncated),
                "makespan": float(final_info.get("makespan", env.base_env.time)),
                "completion_rate": float(final_info.get("completion_rate", 0.0)),
                "average_wait_time": float(final_info.get("average_wait_time", 0.0)),
                "average_avoidable_wait_time": float(final_info.get("average_avoidable_wait_time", 0.0)),
                "idle_ratio": float(final_info.get("idle_ratio", 0.0)),
                "deadlock_events": float(final_info.get("deadlock_events", 0.0)),
                "waiting_sync_reassign_count": float(final_info.get("waiting_sync_reassign_count", 0.0)),
                "productive_reassign_rate": float(final_info.get("productive_reassign_rate", 0.0)),
                "coalition_activation_delay": float(final_info.get("coalition_activation_delay", 0.0)),
            }
        )

    return {
        "episodes": float(episode_count),
        "success_rate": mean(item["success"] for item in metrics),
        "mean_makespan": mean(item["makespan"] for item in metrics),
        "mean_completion_rate": mean(item["completion_rate"] for item in metrics),
        "mean_wait_time": mean(item["average_wait_time"] for item in metrics),
        "mean_avoidable_wait_time": mean(item["average_avoidable_wait_time"] for item in metrics),
        "mean_idle_ratio": mean(item["idle_ratio"] for item in metrics),
        "mean_deadlock_events": mean(item["deadlock_events"] for item in metrics),
        "mean_waiting_sync_reassign_count": mean(item["waiting_sync_reassign_count"] for item in metrics),
        "mean_productive_reassign_rate": mean(item["productive_reassign_rate"] for item in metrics),
        "mean_coalition_activation_delay": mean(item["coalition_activation_delay"] for item in metrics),
    }


def collect_expert_samples(
    scenarios: Sequence[Dict],
    expert_policy: str = "wait_aware_role_greedy",
    max_episodes: int | None = None,
    seed: int = 0,
) -> List[tuple[Dict[str, np.ndarray], int]]:
    rng = np.random.default_rng(seed)
    env = SequentialSchedulingEnv(scenarios=scenarios)
    episode_count = min(len(scenarios), max_episodes) if max_episodes is not None else len(scenarios)
    samples: List[tuple[Dict[str, np.ndarray], int]] = []

    for episode_index in range(episode_count):
        obs, _ = env.reset(options={"scenario": scenarios[episode_index]})
        done = False
        truncated = False
        while not (done or truncated):
            action = policy_action(env, obs, policy=expert_policy, rng=rng, deterministic=True)
            samples.append((clone_obs(obs), int(action)))
            obs, _, done, truncated, _ = env.step(action)

    return samples


class SchedulerSupervisedDataset(Dataset):
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
