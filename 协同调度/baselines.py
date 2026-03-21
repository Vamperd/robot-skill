from __future__ import annotations

import argparse
from statistics import mean
from typing import Callable, Dict

import numpy as np

from scheduling_env import SchedulingEnv


PolicyFn = Callable[[SchedulingEnv, Dict[str, np.ndarray], np.random.Generator], np.ndarray]


def _empty_action(env: SchedulingEnv) -> np.ndarray:
    return np.zeros(env.max_robots, dtype=np.int64)


def random_policy(env: SchedulingEnv, obs: Dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    action = _empty_action(env)
    mask = obs["action_mask"]
    for slot, robot_id in enumerate(env.robot_order):
        if env.robot_states[robot_id]["status"] != "idle":
            continue
        valid = np.flatnonzero(mask[slot])
        action[slot] = int(rng.choice(valid)) if len(valid) else 0
    return action


def nearest_eta_greedy_policy(env: SchedulingEnv, obs: Dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    action = _empty_action(env)
    taken_single = set()
    eta = obs["robot_task_eta"]

    for slot, robot_id in enumerate(env.robot_order):
        if env.robot_states[robot_id]["status"] != "idle":
            continue
        candidates = []
        for task_slot, task_id in enumerate(env.task_order, start=1):
            if not obs["action_mask"][slot, task_slot]:
                continue
            if env.task_specs[task_id]["kind"] == "single" and task_id in taken_single:
                continue
            candidates.append((eta[slot, task_slot - 1], task_slot, task_id))

        if not candidates:
            action[slot] = 0
            continue

        _, task_slot, task_id = min(candidates, key=lambda item: item[0])
        action[slot] = task_slot
        if env.task_specs[task_id]["kind"] == "single":
            taken_single.add(task_id)
    return action


def role_aware_greedy_policy(env: SchedulingEnv, obs: Dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    action = _empty_action(env)
    taken_single = set()

    for slot, robot_id in enumerate(env.robot_order):
        robot = env.robot_states[robot_id]
        if robot["status"] != "idle":
            continue

        ranked = []
        for task_slot, task_id in enumerate(env.task_order, start=1):
            if not obs["action_mask"][slot, task_slot]:
                continue
            task = env.task_specs[task_id]
            if task["kind"] == "single" and task_id in taken_single:
                continue

            eta = float(obs["robot_task_eta"][slot, task_slot - 1])
            role_bonus = 0.0
            if robot["role"] in task.get("required_roles", {}):
                role_bonus = 0.25
            priority_bonus = 0.08 * float(task.get("priority", 1.0))
            sync_bonus = 0.05 if task["kind"] == "sync" else 0.0
            precedence_penalty = 0.15 * len(task.get("precedence", []))
            score = eta + precedence_penalty - role_bonus - priority_bonus - sync_bonus
            ranked.append((score, eta, task_slot, task_id))

        if not ranked:
            action[slot] = 0
            continue

        _, _, task_slot, task_id = min(ranked, key=lambda item: (item[0], item[1]))
        action[slot] = task_slot
        if env.task_specs[task_id]["kind"] == "single":
            taken_single.add(task_id)
    return action


POLICIES: Dict[str, PolicyFn] = {
    "random": random_policy,
    "nearest_eta_greedy": nearest_eta_greedy_policy,
    "role_aware_greedy": role_aware_greedy_policy,
}


def evaluate_policy(
    env: SchedulingEnv,
    policy_name: str,
    max_episodes: int | None = None,
    seed: int = 0,
) -> Dict[str, float]:
    if policy_name not in POLICIES:
        raise ValueError(f"未知基线策略: {policy_name}")

    policy = POLICIES[policy_name]
    rng = np.random.default_rng(seed)
    episode_count = min(len(env.scenarios), max_episodes) if max_episodes is not None else len(env.scenarios)
    metrics = []

    for episode_index in range(episode_count):
        obs, _ = env.reset(options={"scenario_index": episode_index})
        done = False
        truncated = False
        final_info = {}

        while not (done or truncated):
            action = policy(env, obs, rng)
            obs, _, done, truncated, final_info = env.step(action)

        metrics.append(
            {
                "success": float(done and not truncated),
                "makespan": float(final_info.get("makespan", env.time)),
                "completion_rate": float(final_info.get("completion_rate", 0.0)),
                "average_wait_time": float(final_info.get("average_wait_time", 0.0)),
                "idle_ratio": float(final_info.get("idle_ratio", 0.0)),
                "deadlock_events": float(final_info.get("deadlock_events", 0.0)),
            }
        )

    return {
        "episodes": float(episode_count),
        "success_rate": mean(item["success"] for item in metrics),
        "mean_makespan": mean(item["makespan"] for item in metrics),
        "mean_completion_rate": mean(item["completion_rate"] for item in metrics),
        "mean_wait_time": mean(item["average_wait_time"] for item in metrics),
        "mean_idle_ratio": mean(item["idle_ratio"] for item in metrics),
        "mean_deadlock_events": mean(item["deadlock_events"] for item in metrics),
    }


def evaluate_all(
    scenario_dir: str = "offline_maps_v2",
    split: str = "val",
    family: str | None = None,
    max_episodes: int | None = None,
) -> Dict[str, Dict[str, float]]:
    results = {}
    for name in POLICIES:
        env = SchedulingEnv(scenario_dir=scenario_dir, split=split, family=family)
        results[name] = evaluate_policy(env, name, max_episodes=max_episodes, seed=0)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估协同调度基线策略。")
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--split", default="val")
    parser.add_argument("--family", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_all(
        scenario_dir=args.scenario_dir,
        split=args.split,
        family=args.family,
        max_episodes=args.max_episodes,
    )
    for name, result in results.items():
        print(f"[{name}] {result}")
