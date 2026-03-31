from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


def count_roles(robot_ids: Iterable[str], snapshots: Dict[str, Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for robot_id in robot_ids:
        role = snapshots[robot_id]["role"]
        counts[role] = counts.get(role, 0) + 1
    return counts


def role_deficit(required_roles: Dict[str, int], current_roles: Dict[str, int]) -> Dict[str, int]:
    deficit: Dict[str, int] = {}
    for role, need in required_roles.items():
        remaining = need - current_roles.get(role, 0)
        if remaining > 0:
            deficit[role] = remaining
    return deficit


def roles_satisfied(required_roles: Dict[str, int], current_roles: Dict[str, int]) -> bool:
    return not role_deficit(required_roles, current_roles)


def select_contributing_robot_ids(
    task_spec: Dict,
    robot_ids: Sequence[str],
    snapshots: Dict[str, Dict],
) -> List[str]:
    required_roles = task_spec.get("required_roles", {})
    if not robot_ids:
        return []

    if task_spec.get("kind") == "single":
        if not required_roles:
            return [min(robot_ids, key=lambda robot_id: snapshots[robot_id]["service_multiplier"])]
        role = next(iter(required_roles))
        eligible = [
            robot_id
            for robot_id in robot_ids
            if snapshots[robot_id]["role"] == role
        ]
        if len(eligible) < required_roles[role]:
            return []
        return sorted(eligible, key=lambda robot_id: snapshots[robot_id]["service_multiplier"])[: required_roles[role]]

    if not required_roles:
        return sorted(robot_ids, key=lambda robot_id: snapshots[robot_id]["service_multiplier"])

    selected: List[str] = []
    for role, need in required_roles.items():
        eligible = [
            robot_id
            for robot_id in robot_ids
            if snapshots[robot_id]["role"] == role
        ]
        if len(eligible) < need:
            return []
        eligible = sorted(eligible, key=lambda robot_id: snapshots[robot_id]["service_multiplier"])
        selected.extend(eligible[:need])
    return selected


def service_rate(robot_ids: Sequence[str], snapshots: Dict[str, Dict]) -> float:
    return sum(1.0 / max(0.05, snapshots[robot_id]["service_multiplier"]) for robot_id in robot_ids)


@dataclass
class TaskRuntimeEvent:
    completed_tasks: List[str]
    advance_robot_ids: List[str]
    robot_waiting: Dict[str, bool]
    robot_wait_times: Dict[str, float]
    robot_task_progress: Dict[str, float]


class ContinuousTaskRuntime:
    def __init__(self, tasks: Sequence[Dict], service_radius: float = 40.0):
        self.service_radius = service_radius
        self.tasks = {task["id"]: task for task in tasks}
        self.task_states = {
            task["id"]: {
                "progress": 0.0,
                "completed": False,
                "present_robot_ids": set(),
                "contributors": set(),
            }
            for task in tasks
        }
        self.robot_wait_times: Dict[str, float] = {}

    def update(self, robot_snapshots: Sequence[Dict], dt: float = 1.0) -> TaskRuntimeEvent:
        snapshot_map = {snapshot["id"]: snapshot for snapshot in robot_snapshots}
        completed_tasks: List[str] = []
        advance_robot_ids: List[str] = []
        robot_waiting = {snapshot["id"]: False for snapshot in robot_snapshots}
        robot_task_progress = {snapshot["id"]: 0.0 for snapshot in robot_snapshots}
        current_wait_times = {snapshot["id"]: 0.0 for snapshot in robot_snapshots}

        for task_id, task_spec in self.tasks.items():
            state = self.task_states[task_id]
            if state["completed"]:
                continue

            present_robot_ids = [
                snapshot["id"]
                for snapshot in robot_snapshots
                if snapshot.get("assigned_task") == task_id
                and (
                    bool(snapshot["service_ready"])
                    if "service_ready" in snapshot
                    else snapshot.get("distance_to_task", float("inf")) <= self.service_radius
                )
            ]
            state["present_robot_ids"] = set(present_robot_ids)

            contributors = select_contributing_robot_ids(task_spec, present_robot_ids, snapshot_map)
            state["contributors"] = set(contributors)

            if contributors:
                state["progress"] += dt * service_rate(contributors, snapshot_map)

            for robot_id in present_robot_ids:
                robot_task_progress[robot_id] = state["progress"]
                is_waiting = robot_id not in state["contributors"]
                robot_waiting[robot_id] = is_waiting
                if is_waiting:
                    current_wait_times[robot_id] = self.robot_wait_times.get(robot_id, 0.0) + dt

            if state["progress"] >= task_spec["service_time"]:
                state["completed"] = True
                completed_tasks.append(task_id)
                advance_robot_ids.extend(
                    snapshot["id"]
                    for snapshot in robot_snapshots
                    if snapshot.get("assigned_task") == task_id
                )

        self.robot_wait_times = current_wait_times
        return TaskRuntimeEvent(
            completed_tasks=completed_tasks,
            advance_robot_ids=sorted(set(advance_robot_ids)),
            robot_waiting=robot_waiting,
            robot_wait_times=dict(self.robot_wait_times),
            robot_task_progress=robot_task_progress,
        )
