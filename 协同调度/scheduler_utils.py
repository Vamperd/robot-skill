from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from scenario_generator import FAMILY_NAMES, HEIGHT, ROBOT_ROLES, WIDTH, load_scenarios
from task_runtime import count_roles, role_deficit


ROLE_NAMES = tuple(ROBOT_ROLES.keys())
ROLE_TO_INDEX = {role: index for index, role in enumerate(ROLE_NAMES)}
ROLE_DIM = len(ROLE_NAMES)
MAX_ROLE_COUNT = 3.0
ROBOT_FEATURE_DIM = 12 + ROLE_DIM
TASK_FEATURE_DIM = 12 + ROLE_DIM
EPS = 1e-6


def task_id_from_action(action_value: int, task_order: Sequence[str]) -> Optional[str]:
    if action_value <= 0 or action_value > len(task_order):
        return None
    return task_order[action_value - 1]


def action_from_task_id(task_id: Optional[str], task_order: Sequence[str]) -> int:
    if task_id is None:
        return 0
    try:
        return int(task_order.index(task_id) + 1)
    except ValueError:
        return 0


def role_vector(role: str) -> np.ndarray:
    vector = np.zeros(ROLE_DIM, dtype=np.float32)
    if role in ROLE_TO_INDEX:
        vector[ROLE_TO_INDEX[role]] = 1.0
    return vector


def required_roles_vector(required_roles: Dict[str, int]) -> np.ndarray:
    vector = np.zeros(ROLE_DIM, dtype=np.float32)
    for role, count in required_roles.items():
        if role in ROLE_TO_INDEX:
            vector[ROLE_TO_INDEX[role]] = min(float(count) / MAX_ROLE_COUNT, 1.0)
    return vector


def _completed_task_ids(task_states: Dict[str, Dict]) -> set[str]:
    return {task_id for task_id, state in task_states.items() if state.get("completed", False)}


def precedence_state_vector(
    task_order: Sequence[str],
    task_specs: Dict[str, Dict],
    task_states: Dict[str, Dict],
    max_tasks: int,
) -> np.ndarray:
    done = _completed_task_ids(task_states)
    vector = np.zeros(max_tasks, dtype=np.float32)
    for slot, task_id in enumerate(task_order):
        if slot >= max_tasks:
            break
        task = task_specs[task_id]
        unlocked = all(parent in done for parent in task.get("precedence", []))
        vector[slot] = float(unlocked)
    return vector


def pending_assignment_mask(
    pending_actions: Sequence[int],
    max_robots: int,
    max_tasks: int,
) -> np.ndarray:
    mask = np.zeros((max_robots, max_tasks + 1), dtype=np.float32)
    for slot, action_value in enumerate(pending_actions[:max_robots]):
        if 0 <= int(action_value) <= max_tasks:
            mask[slot, int(action_value)] = 1.0
    return mask


def pending_task_assignments(
    robot_order: Sequence[str],
    task_order: Sequence[str],
    pending_actions: Sequence[int],
) -> Dict[str, set[str]]:
    assignments: Dict[str, set[str]] = {}
    for slot, robot_id in enumerate(robot_order):
        task_id = task_id_from_action(int(pending_actions[slot]), task_order)
        if task_id is None:
            continue
        assignments.setdefault(task_id, set()).add(robot_id)
    return assignments


def remaining_role_deficit_matrix(
    task_order: Sequence[str],
    task_specs: Dict[str, Dict],
    task_states: Dict[str, Dict],
    robot_states: Dict[str, Dict],
    pending_actions: Sequence[int],
    robot_order: Sequence[str],
    max_tasks: int,
) -> np.ndarray:
    deficits = np.zeros((max_tasks, ROLE_DIM), dtype=np.float32)
    pending_assignments = pending_task_assignments(robot_order, task_order, pending_actions)

    for task_slot, task_id in enumerate(task_order):
        if task_slot >= max_tasks:
            break
        task = task_specs[task_id]
        required_roles = task.get("required_roles", {})
        if not required_roles:
            continue

        task_state = task_states[task_id]
        robot_ids = set(task_state.get("assigned_robot_ids", set())) | set(task_state.get("onsite_robot_ids", set()))
        robot_ids |= pending_assignments.get(task_id, set())
        current_roles = count_roles(robot_ids, robot_states)
        deficit = role_deficit(required_roles, current_roles)
        deficits[task_slot] = required_roles_vector(deficit)
    return deficits


def _precedence_satisfied(task_id: str, task_specs: Dict[str, Dict], task_states: Dict[str, Dict]) -> bool:
    done = _completed_task_ids(task_states)
    return all(parent in done for parent in task_specs[task_id].get("precedence", []))


def legal_action_mask_for_robot(
    robot_id: str,
    robot_order: Sequence[str],
    task_order: Sequence[str],
    robot_states: Dict[str, Dict],
    task_states: Dict[str, Dict],
    task_specs: Dict[str, Dict],
    pending_actions: Sequence[int],
    max_tasks: int,
) -> np.ndarray:
    mask = np.zeros(max_tasks + 1, dtype=np.float32)
    mask[0] = 1.0

    robot = robot_states[robot_id]
    if robot.get("status") not in {"idle", "waiting_idle"}:
        return mask

    pending_assignments = pending_task_assignments(robot_order, task_order, pending_actions)
    reserved_single = {
        task_id
        for task_id, assignees in pending_assignments.items()
        if task_specs[task_id].get("kind") == "single" and assignees
    }

    for task_slot, task_id in enumerate(task_order, start=1):
        if task_slot > max_tasks:
            break
        task_state = task_states[task_id]
        task = task_specs[task_id]

        if task_state.get("completed", False) or not _precedence_satisfied(task_id, task_specs, task_states):
            continue

        if task.get("kind") == "single":
            occupied = bool(task_state.get("assigned_robot_ids")) or bool(task_state.get("onsite_robot_ids"))
            occupied = occupied or (task_id in reserved_single and robot_id not in pending_assignments.get(task_id, set()))
            if occupied:
                continue
            required_roles = task.get("required_roles", {})
            if required_roles and robot.get("role") not in required_roles:
                continue
            mask[task_slot] = 1.0
            continue

        required_roles = task.get("required_roles", {})
        if not required_roles:
            mask[task_slot] = 1.0
            continue

        robot_ids = set(task_state.get("assigned_robot_ids", set())) | set(task_state.get("onsite_robot_ids", set()))
        robot_ids |= pending_assignments.get(task_id, set())
        if robot_id in robot_ids:
            continue
        current_roles = count_roles(robot_ids, robot_states)
        deficit = role_deficit(required_roles, current_roles)
        if deficit.get(robot.get("role"), 0) > 0:
            mask[task_slot] = 1.0

    return mask


def role_aware_greedy_action_for_robot(
    robot_id: str,
    robot_order: Sequence[str],
    task_order: Sequence[str],
    robot_states: Dict[str, Dict],
    task_states: Dict[str, Dict],
    task_specs: Dict[str, Dict],
    robot_task_eta_row: np.ndarray,
    pending_actions: Sequence[int],
    max_tasks: int,
) -> int:
    action_mask = legal_action_mask_for_robot(
        robot_id=robot_id,
        robot_order=robot_order,
        task_order=task_order,
        robot_states=robot_states,
        task_states=task_states,
        task_specs=task_specs,
        pending_actions=pending_actions,
        max_tasks=max_tasks,
    )
    robot = robot_states[robot_id]
    ranked = []

    pending_assignments = pending_task_assignments(robot_order, task_order, pending_actions)

    for task_slot, task_id in enumerate(task_order, start=1):
        if task_slot > max_tasks or action_mask[task_slot] <= 0.0:
            continue
        task = task_specs[task_id]
        eta = float(robot_task_eta_row[task_slot - 1])
        role_bonus = 0.25 if robot.get("role") in task.get("required_roles", {}) else 0.0
        priority_bonus = 0.08 * float(task.get("priority", 1.0))
        sync_bonus = 0.05 if task.get("kind") == "sync" else 0.0
        precedence_penalty = 0.15 * len(task.get("precedence", []))

        deficit_bonus = 0.0
        if task.get("kind") == "sync":
            robot_ids = set(task_states[task_id].get("assigned_robot_ids", set())) | set(task_states[task_id].get("onsite_robot_ids", set()))
            robot_ids |= pending_assignments.get(task_id, set())
            current_roles = count_roles(robot_ids, robot_states)
            deficit = role_deficit(task.get("required_roles", {}), current_roles)
            deficit_bonus = 0.05 * float(deficit.get(robot.get("role"), 0))

        score = eta + precedence_penalty - role_bonus - priority_bonus - sync_bonus - deficit_bonus
        ranked.append((score, eta, task_slot))

    if not ranked:
        return 0
    return int(min(ranked, key=lambda item: (item[0], item[1]))[2])


def build_scheduler_observation(
    *,
    robot_order: Sequence[str],
    task_order: Sequence[str],
    robot_states: Dict[str, Dict],
    task_states: Dict[str, Dict],
    task_specs: Dict[str, Dict],
    robot_task_eta: np.ndarray,
    task_task_eta: np.ndarray,
    current_slot: Optional[int],
    pending_actions: Sequence[int],
    max_robots: int,
    max_tasks: int,
    max_time: float,
    wait_timeout: float,
) -> Dict[str, np.ndarray]:
    robots_obs = np.zeros((max_robots, ROBOT_FEATURE_DIM), dtype=np.float32)
    tasks_obs = np.zeros((max_tasks, TASK_FEATURE_DIM), dtype=np.float32)
    current_robot = np.zeros(max_robots, dtype=np.float32)

    for slot, robot_id in enumerate(robot_order):
        if slot >= max_robots:
            break
        robot = robot_states[robot_id]
        x, y = robot.get("position", (0.0, 0.0))
        status = robot.get("status", "idle")
        status_features = np.array(
            [
                float(status == "idle"),
                float(status == "travel"),
                float(status == "onsite"),
                float(status == "waiting_idle"),
            ],
            dtype=np.float32,
        )
        base = np.array(
            [
                1.0,
                *status_features,
                x / max(float(WIDTH), 1.0),
                y / max(float(HEIGHT), 1.0),
                min(float(robot.get("speed_multiplier", 1.0)) / 1.5, 1.0),
                min(float(robot.get("service_multiplier", 1.0)) / 2.5, 1.0),
                min(float(robot.get("eta_remaining", 0.0)) / max(max_time, EPS), 1.0),
                min(float(robot.get("wait_elapsed", 0.0)) / max(wait_timeout, EPS), 1.0),
                min(float(robot.get("blocked_count", 0.0)) / 5.0, 1.0),
            ],
            dtype=np.float32,
        )
        robots_obs[slot] = np.concatenate([base, role_vector(str(robot.get("role", "")))])

    precedence_state = precedence_state_vector(task_order, task_specs, task_states, max_tasks)
    remaining_role_deficit = remaining_role_deficit_matrix(
        task_order=task_order,
        task_specs=task_specs,
        task_states=task_states,
        robot_states=robot_states,
        pending_actions=pending_actions,
        robot_order=robot_order,
        max_tasks=max_tasks,
    )

    pending_assignments = pending_task_assignments(robot_order, task_order, pending_actions)
    for slot, task_id in enumerate(task_order):
        if slot >= max_tasks:
            break
        task_state = task_states[task_id]
        task = task_specs[task_id]
        progress = float(task_state.get("progress", 0.0))
        service_time = max(float(task.get("service_time", 1.0)), EPS)
        assigned_count = len(set(task_state.get("assigned_robot_ids", set())) | pending_assignments.get(task_id, set()))
        onsite_count = len(task_state.get("onsite_robot_ids", set()))
        contributor_count = len(task_state.get("contributors", set()))
        required_roles = task.get("required_roles", {})
        base = np.array(
            [
                1.0,
                float(task_state.get("completed", False)),
                float(task.get("kind") == "sync"),
                float(task.get("pos", (0.0, 0.0))[0]) / max(float(WIDTH), 1.0),
                float(task.get("pos", (0.0, 0.0))[1]) / max(float(HEIGHT), 1.0),
                min(progress / service_time, 1.0),
                min(float(task.get("priority", 1.0)) / 2.0, 1.0),
                min(float(sum(required_roles.values())) / MAX_ROLE_COUNT, 1.0),
                precedence_state[slot],
                min(float(assigned_count) / MAX_ROLE_COUNT, 1.0),
                min(float(onsite_count) / MAX_ROLE_COUNT, 1.0),
                min(float(contributor_count) / MAX_ROLE_COUNT, 1.0),
            ],
            dtype=np.float32,
        )
        tasks_obs[slot] = np.concatenate([base, required_roles_vector(required_roles)])

    if current_slot is not None and 0 <= current_slot < max_robots:
        current_robot[current_slot] = 1.0
        current_robot_id = robot_order[current_slot]
        current_action_mask = legal_action_mask_for_robot(
            robot_id=current_robot_id,
            robot_order=robot_order,
            task_order=task_order,
            robot_states=robot_states,
            task_states=task_states,
            task_specs=task_specs,
            pending_actions=pending_actions,
            max_tasks=max_tasks,
        )
    else:
        current_action_mask = np.zeros(max_tasks + 1, dtype=np.float32)
        current_action_mask[0] = 1.0

    return {
        "current_robot": current_robot,
        "robots": robots_obs,
        "tasks": tasks_obs,
        "robot_task_eta": robot_task_eta.astype(np.float32),
        "task_task_eta": task_task_eta.astype(np.float32),
        "current_action_mask": current_action_mask.astype(np.float32),
        "pending_assignment_mask": pending_assignment_mask(pending_actions, max_robots=max_robots, max_tasks=max_tasks),
        "remaining_role_deficit": remaining_role_deficit,
        "precedence_state": precedence_state,
    }


def curriculum_families(progress: float) -> List[str]:
    if progress < 0.25:
        return ["open_balance", "role_mismatch"]
    if progress < 0.50:
        return ["open_balance", "role_mismatch", "single_bottleneck"]
    if progress < 0.75:
        return ["open_balance", "role_mismatch", "single_bottleneck", "double_bottleneck", "far_near_trap"]
    return list(
        [
            "open_balance",
            "role_mismatch",
            "single_bottleneck",
            "double_bottleneck",
            "far_near_trap",
            "multi_sync_cluster",
        ]
    )


def load_split_scenarios(
    scenario_dir: str | Path,
    split: str,
    families: Optional[Iterable[str]] = None,
    limit_per_family: Optional[int] = None,
) -> List[Dict]:
    if families:
        family_list = list(families)
    elif limit_per_family is not None:
        family_list = list(FAMILY_NAMES)
    else:
        family_list = [None]
    scenarios: List[Dict] = []
    for family in family_list:
        scenarios.extend(load_scenarios(cache_dir=scenario_dir, split=split, family=family, limit=limit_per_family))
    return scenarios
