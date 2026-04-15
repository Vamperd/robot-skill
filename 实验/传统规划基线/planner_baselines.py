from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
SCHED_DIR = REPO_ROOT / "协同调度"
for path in (REPO_ROOT, SCHED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scheduler_utils import (  # noqa: E402
    constrain_wait_action_mask,
    legal_action_mask_for_robot,
    pending_task_assignments,
    task_id_from_action,
)
from task_runtime import count_roles, role_deficit, service_rate  # noqa: E402


PLANNER_POLICY_NAMES = {"auction_mrta", "milp_scheduler_small", "sas", "ctas_d"}
ACTIVE_DISPATCH_STATUSES = {"idle", "waiting_idle"}
EPS = 1e-6


class PlannerBaselinePolicy:
    name = "planner_baseline"

    def __init__(self) -> None:
        self._episode_count = 0

    def reset_episode(self, scenario: Dict) -> None:
        self._episode_count += 1

    def select_action(
        self,
        env_like,
        obs: Dict[str, np.ndarray] | None,
        current_robot_index: int,
        legal_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        raise NotImplementedError

    def get_diagnostics(self) -> Dict[str, float]:
        return {"episodes_evaluated": float(self._episode_count)}


def is_planner_policy_name(name: str | None) -> bool:
    return bool(name in PLANNER_POLICY_NAMES)


def load_planner_policy(name: str) -> PlannerBaselinePolicy:
    if name == "auction_mrta":
        from auction_mrta import AuctionMRTA

        return AuctionMRTA()
    if name == "milp_scheduler_small":
        from milp_scheduler_small import MilpSchedulerSmall

        return MilpSchedulerSmall()
    if name == "sas":
        from sas import SAS

        return SAS()
    if name == "ctas_d":
        from ctas_d import CTASD

        return CTASD()
    raise ValueError(f"Unknown planner baseline: {name}")


def get_base_env(env_like):
    return getattr(env_like, "base_env", env_like)


def get_robot_order(env_like) -> list[str]:
    return list(getattr(env_like, "robot_order"))


def get_task_order(env_like) -> list[str]:
    return list(getattr(env_like, "task_order"))


def get_robot_states(env_like) -> Dict[str, Dict]:
    return getattr(get_base_env(env_like), "robot_states")


def get_task_states(env_like) -> Dict[str, Dict]:
    return getattr(get_base_env(env_like), "task_states")


def get_task_specs(env_like) -> Dict[str, Dict]:
    return getattr(get_base_env(env_like), "task_specs")


def get_pending_actions(env_like) -> np.ndarray:
    pending = getattr(env_like, "pending_actions", None)
    if pending is None:
        return np.zeros(len(get_robot_order(env_like)), dtype=np.int64)
    return np.asarray(pending, dtype=np.int64)


def get_scenario_id(env_like) -> str:
    base_env = get_base_env(env_like)
    scenario = getattr(base_env, "current_scenario", None) or getattr(env_like, "scenario", None) or {}
    return str(scenario.get("scenario_id", "unknown"))


def get_round_key(env_like) -> tuple[str, int]:
    return (get_scenario_id(env_like), int(getattr(env_like, "event_index", 0)))


def current_robot_id(env_like, current_robot_index: int) -> str:
    return get_robot_order(env_like)[int(current_robot_index)]


def action_from_task_id(task_id: str | None, env_like) -> int:
    if task_id is None:
        return 0
    return int(get_task_order(env_like).index(task_id) + 1)


def legal_mask_for_robot(env_like, robot_id: str, pending_actions: np.ndarray | None = None) -> np.ndarray:
    if pending_actions is None:
        pending_actions = get_pending_actions(env_like)

    if hasattr(env_like, "_legal_action_mask") and hasattr(env_like, "base_env"):
        return np.asarray(env_like._legal_action_mask(robot_id), dtype=np.float32)

    mask = legal_action_mask_for_robot(
        robot_id=robot_id,
        robot_order=get_robot_order(env_like),
        task_order=get_task_order(env_like),
        robot_states=get_robot_states(env_like),
        task_states=get_task_states(env_like),
        task_specs=get_task_specs(env_like),
        pending_actions=pending_actions,
        max_tasks=len(get_task_order(env_like)),
    ).astype(np.float32)
    wait_timeout = float(getattr(env_like, "wait_timeout", getattr(get_base_env(env_like), "wait_timeout", 60.0)))
    del wait_timeout
    return constrain_wait_action_mask(mask, get_robot_states(env_like)[robot_id])


def predict_task_outcome(env_like, robot_id: str, task_id: str, pending_actions: np.ndarray | None = None) -> tuple[float, float, float]:
    if pending_actions is None:
        pending_actions = get_pending_actions(env_like)

    if hasattr(env_like, "base_env"):
        return env_like._predict_task_outcome(robot_id, task_id)
    return env_like._predict_task_outcome(robot_id, task_id, pending_actions)


def robot_available_eta(env_like, robot_id: str, task_id: str) -> float:
    base_env = get_base_env(env_like)
    return float(base_env._robot_available_eta_for_task(robot_id, task_id))


def task_unlock_value(env_like, task_id: str) -> float:
    base_env = get_base_env(env_like)
    return float(base_env._task_unlock_value(task_id))


def existing_task_ready_eta(env_like, task_id: str, pending_actions: np.ndarray | None = None) -> float:
    if pending_actions is None:
        pending_actions = get_pending_actions(env_like)
    task_state = get_task_states(env_like)[task_id]
    if np.isfinite(float(task_state.get("coalition_ready_eta", float("inf")))):
        return float(task_state["coalition_ready_eta"])
    if hasattr(env_like, "_estimate_coalition_ready_eta"):
        return float(env_like._estimate_coalition_ready_eta(task_id, pending_actions))
    base_env = get_base_env(env_like)
    if hasattr(base_env, "_estimate_coalition_ready_eta"):
        return float(base_env._estimate_coalition_ready_eta(task_id))
    return 0.0


def active_robot_ids(env_like) -> list[str]:
    robot_states = get_robot_states(env_like)
    return [robot_id for robot_id in get_robot_order(env_like) if robot_states[robot_id].get("status") in ACTIVE_DISPATCH_STATUSES]


def legal_non_wait_count(env_like, robot_id: str, pending_actions: np.ndarray | None = None) -> int:
    mask = np.asarray(legal_mask_for_robot(env_like, robot_id, pending_actions), dtype=np.float32)
    return int(np.count_nonzero(mask[1:] > 0.0))


def ordered_active_robot_ids(env_like, pending_actions: np.ndarray | None = None) -> list[str]:
    robot_states = get_robot_states(env_like)
    robot_ids = active_robot_ids(env_like)
    return sorted(
        robot_ids,
        key=lambda robot_id: (
            0 if robot_states[robot_id].get("status") == "waiting_idle" else 1,
            legal_non_wait_count(env_like, robot_id, pending_actions),
            robot_id,
        ),
    )


def legal_task_ids_for_robot(env_like, robot_id: str, pending_actions: np.ndarray | None = None) -> list[str]:
    mask = legal_mask_for_robot(env_like, robot_id, pending_actions)
    task_order = get_task_order(env_like)
    return [
        task_id_from_action(int(action), task_order)
        for action in np.flatnonzero(mask > 0.0)
        if int(action) != 0 and task_id_from_action(int(action), task_order) is not None
    ]


def candidate_task_ids(env_like, robot_ids: Sequence[str], pending_actions: np.ndarray | None = None) -> list[str]:
    task_ids = set()
    for robot_id in robot_ids:
        task_ids.update(legal_task_ids_for_robot(env_like, robot_id, pending_actions))
    ordered = [task_id for task_id in get_task_order(env_like) if task_id in task_ids]
    return ordered


def enumerate_sync_coalitions(
    env_like,
    task_id: str,
    robot_ids: Sequence[str],
    pending_actions: np.ndarray | None = None,
    *,
    max_role_candidates: int = 3,
    max_primitives: int = 24,
) -> list[Dict[str, object]]:
    if pending_actions is None:
        pending_actions = get_pending_actions(env_like)
    task = get_task_specs(env_like)[task_id]
    if task.get("kind") != "sync":
        return []

    deficit = role_deficit_for_task(env_like, task_id, pending_actions)
    required_slots = int(sum(deficit.values()))
    if required_slots <= 0:
        return []

    robot_states = get_robot_states(env_like)
    role_candidates: list[list[str]] = []
    for role, need in sorted(deficit.items()):
        if int(need) <= 0:
            continue
        candidates: list[tuple[float, str]] = []
        for robot_id in robot_ids:
            robot = robot_states[robot_id]
            if robot.get("role") != role:
                continue
            if legal_mask_for_robot(env_like, robot_id, pending_actions)[action_from_task_id(task_id, env_like)] <= 0.0:
                continue
            eta = robot_available_eta(env_like, robot_id, task_id)
            if np.isfinite(eta):
                candidates.append((float(eta), robot_id))
        candidates.sort(key=lambda item: (item[0], item[1]))
        filtered = [robot_id for _, robot_id in candidates[:max_role_candidates]]
        for _ in range(int(need)):
            role_candidates.append(filtered)
    if not role_candidates:
        return []

    pool = sorted({robot_id for candidates in role_candidates for robot_id in candidates})
    if len(pool) < required_slots:
        return []

    primitive_candidates: list[tuple[float, Dict[str, object]]] = []
    seen_coalitions: set[tuple[str, ...]] = set()
    from itertools import combinations

    for coalition in combinations(pool, required_slots):
        coalition_key = tuple(sorted(str(robot_id) for robot_id in coalition))
        if coalition_key in seen_coalitions:
            continue
        seen_coalitions.add(coalition_key)
        coalition_info = coalition_info_for_task(
            env_like,
            task_id,
            forced_robot_id=coalition_key[0],
            pending_actions=pending_actions,
            selected_new_robot_ids=coalition_key,
        )
        if coalition_info is None:
            continue
        primitive = {
            "task_id": task_id,
            "robot_ids": coalition_key,
            "coalition_eta": float(coalition_info["coalition_eta"]),
            "coalition_wait": float(coalition_info["coalition_wait"]),
            "coalition_size": float(coalition_info["coalition_size"]),
            "predicted_finish": float(coalition_info["predicted_finish"]),
        }
        primitive_candidates.append(
            (
                float(coalition_info["coalition_eta"]) + 0.5 * float(coalition_info["coalition_wait"]),
                primitive,
            )
        )

    primitive_candidates.sort(key=lambda item: (item[0], item[1]["task_id"], item[1]["robot_ids"]))
    return [primitive for _, primitive in primitive_candidates[:max_primitives]]


def role_deficit_for_task(env_like, task_id: str, pending_actions: np.ndarray | None = None) -> Dict[str, int]:
    if pending_actions is None:
        pending_actions = get_pending_actions(env_like)
    task_state = get_task_states(env_like)[task_id]
    task = get_task_specs(env_like)[task_id]
    robot_states = get_robot_states(env_like)
    robot_ids = set(task_state.get("assigned_robot_ids", set())) | set(task_state.get("onsite_robot_ids", set()))
    robot_ids |= pending_task_assignments(get_robot_order(env_like), get_task_order(env_like), pending_actions).get(task_id, set())
    return role_deficit(task.get("required_roles", {}), count_roles(robot_ids, robot_states))


def coalition_info_for_task(
    env_like,
    task_id: str,
    forced_robot_id: str,
    pending_actions: np.ndarray | None = None,
    selected_new_robot_ids: Sequence[str] | None = None,
) -> Dict[str, float] | None:
    if pending_actions is None:
        pending_actions = get_pending_actions(env_like)

    task_specs = get_task_specs(env_like)
    task_states = get_task_states(env_like)
    robot_states = get_robot_states(env_like)
    task = task_specs[task_id]
    if task.get("kind") != "sync":
        return None

    deficit = role_deficit_for_task(env_like, task_id, pending_actions)
    required_slots = sum(int(value) for value in deficit.values())
    forced_role = robot_states[forced_robot_id]["role"]
    if deficit.get(forced_role, 0) <= 0:
        return None

    assignments = pending_task_assignments(get_robot_order(env_like), get_task_order(env_like), pending_actions)
    other_tasks = {
        robot_id
        for other_task_id, robot_ids in assignments.items()
        if other_task_id != task_id
        for robot_id in robot_ids
    }

    existing_robot_ids = set(task_states[task_id].get("assigned_robot_ids", set())) | set(task_states[task_id].get("onsite_robot_ids", set()))
    existing_ready = existing_task_ready_eta(env_like, task_id, pending_actions)

    if selected_new_robot_ids is None:
        selected_ids: list[str] = []
        for role, need in deficit.items():
            if need <= 0:
                continue
            if role == forced_role:
                selected_ids.append(forced_robot_id)
                need -= 1
            if need <= 0:
                continue
            candidates: list[tuple[float, str]] = []
            for candidate_id, candidate in robot_states.items():
                if candidate_id == forced_robot_id or candidate_id in selected_ids:
                    continue
                if candidate_id in other_tasks:
                    continue
                if candidate["role"] != role:
                    continue
                if action_from_task_id(task_id, env_like) not in np.flatnonzero(legal_mask_for_robot(env_like, candidate_id, pending_actions) > 0.0):
                    continue
                eta = robot_available_eta(env_like, candidate_id, task_id)
                if np.isfinite(eta):
                    candidates.append((eta, candidate_id))
            candidates.sort(key=lambda item: (item[0], item[1]))
            if len(candidates) < need:
                return None
            selected_ids.extend(candidate_id for _, candidate_id in candidates[:need])
    else:
        selected_ids = list(dict.fromkeys(str(robot_id) for robot_id in selected_new_robot_ids))
        if forced_robot_id not in selected_ids:
            selected_ids.append(forced_robot_id)
        if len(selected_ids) != required_slots:
            return None
        selected_roles = count_roles(selected_ids, robot_states)
        if any(int(selected_roles.get(role, 0)) != int(deficit.get(role, 0)) for role in deficit):
            return None

    ready_etas = []
    for robot_id in selected_ids:
        if robot_id in other_tasks:
            return None
        eta = robot_available_eta(env_like, robot_id, task_id)
        if not np.isfinite(eta):
            return None
        ready_etas.append(float(eta))
    coalition_eta = max([float(existing_ready)] + ready_etas) if ready_etas or np.isfinite(existing_ready) else float("inf")
    coalition_wait = sum(max(0.0, coalition_eta - eta) for eta in ready_etas)
    all_participants = sorted(existing_robot_ids | set(selected_ids))
    rate = service_rate(
        all_participants,
        {
            robot_id: {"service_multiplier": robot_states[robot_id]["service_multiplier"]}
            for robot_id in all_participants
        },
    )
    if rate <= EPS:
        return None
    service_duration = float(task["service_time"]) / max(rate, EPS)
    predicted_finish = coalition_eta + service_duration
    return {
        "coalition_eta": float(coalition_eta),
        "coalition_wait": float(coalition_wait),
        "coalition_size": float(len(selected_ids)),
        "service_duration": float(service_duration),
        "predicted_finish": float(predicted_finish),
        "selected_robot_ids": list(selected_ids),
    }


def single_task_runtime(env_like, robot_id: str, task_id: str) -> float:
    task = get_task_specs(env_like)[task_id]
    robot = get_robot_states(env_like)[robot_id]
    return float(task["service_time"]) / max(float(robot["service_multiplier"]), EPS)


def mean_ms(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def now_ms() -> float:
    return time.perf_counter() * 1000.0
