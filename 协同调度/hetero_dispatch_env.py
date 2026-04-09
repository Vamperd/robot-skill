from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except ImportError:  # pragma: no cover
        class _FallbackEnv:
            def reset(self, seed: int | None = None, options: Optional[dict] = None):
                self.np_random = np.random.default_rng(seed)
                return None

        class _FallbackBox:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _FallbackDiscrete:
            def __init__(self, n):
                self.n = n

        class _FallbackDict:
            def __init__(self, spaces_dict):
                self.spaces = spaces_dict

        class _FallbackGym:
            Env = _FallbackEnv

        class _FallbackSpaces:
            Box = _FallbackBox
            Dict = _FallbackDict
            Discrete = _FallbackDiscrete

        gym = _FallbackGym()
        spaces = _FallbackSpaces()

from scenario_generator import HEIGHT, WIDTH
from scheduling_env import EPS, MAX_ROBOTS, MAX_TASKS, SchedulingEnv
from scheduler_utils import (
    ROLE_DIM,
    constrain_wait_action_mask,
    fallback_legal_action_from_mask,
    legal_action_mask_for_robot,
    pending_task_assignments,
    precedence_state_vector,
    required_roles_vector,
    remaining_role_deficit_matrix,
    role_vector,
    task_id_from_action,
)
from task_runtime import service_rate


AGENT_FEATURE_DIM = 11 + ROLE_DIM
TASK_FEATURE_DIM = 15 + ROLE_DIM * 2

TASK_IDX = {
    "presence": 0,
    "remaining_role_deficit_start": 1,
    "required_roles_start": 1 + ROLE_DIM,
    "service_time": 1 + ROLE_DIM * 2,
    "eta": 2 + ROLE_DIM * 2,
    "predicted_start": 3 + ROLE_DIM * 2,
    "predicted_finish": 4 + ROLE_DIM * 2,
    "priority": 5 + ROLE_DIM * 2,
    "unlock_value": 6 + ROLE_DIM * 2,
    "precedence_satisfied": 7 + ROLE_DIM * 2,
    "contributors_count": 8 + ROLE_DIM * 2,
    "waiting_count": 9 + ROLE_DIM * 2,
    "is_single": 10 + ROLE_DIM * 2,
    "is_sync": 11 + ROLE_DIM * 2,
    "is_wait_node": 12 + ROLE_DIM * 2,
    "dx": 13 + ROLE_DIM * 2,
    "dy": 14 + ROLE_DIM * 2,
}

TRAINABLE_STATUSES = {"idle", "waiting_idle"}
HARD_TEACHER_FAMILIES = {
    "role_mismatch",
    "single_bottleneck",
    "double_bottleneck",
    "multi_sync_cluster",
    "partial_coalition_trap",
}


def build_hetero_scheduler_observation(
    *,
    robot_order: Sequence[str],
    task_order: Sequence[str],
    robot_states: Dict[str, Dict],
    task_states: Dict[str, Dict],
    task_specs: Dict[str, Dict],
    current_slot: Optional[int],
    legal_mask: np.ndarray,
    remaining_role_deficits: np.ndarray,
    precedence_state: np.ndarray,
    robot_release_eta_fn: Callable[[str], float],
    predict_task_outcome_fn: Callable[[str, str], tuple[float, float, float]],
    task_waiting_count_fn: Callable[[str, Dict], int],
    task_unlock_value_fn: Callable[[str], float],
    max_time: float,
    max_robots: Optional[int] = None,
    num_task_tokens: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if max_robots is None:
        max_robots = len(robot_order)
    if num_task_tokens is None:
        num_task_tokens = len(task_order) + 1

    agent_inputs = np.zeros((max_robots, AGENT_FEATURE_DIM), dtype=np.float32)
    task_inputs = np.zeros((num_task_tokens, TASK_FEATURE_DIM), dtype=np.float32)
    current_agent_index = np.zeros((1,), dtype=np.int64)

    wait_row = np.zeros(TASK_FEATURE_DIM, dtype=np.float32)
    wait_row[TASK_IDX["presence"]] = 1.0
    wait_row[TASK_IDX["is_wait_node"]] = 1.0
    task_inputs[0] = wait_row

    if current_slot is None or current_slot >= len(robot_order):
        return {
            "agent_inputs": agent_inputs,
            "task_inputs": task_inputs,
            "global_mask": (1.0 - legal_mask).astype(np.float32),
            "current_agent_index": current_agent_index,
        }

    current_agent_index[0] = current_slot
    current_robot_id = robot_order[current_slot]
    current_robot = robot_states[current_robot_id]
    current_x, current_y = current_robot["position"]

    for slot, robot_id in enumerate(robot_order):
        if slot >= max_robots:
            break
        robot = robot_states[robot_id]
        x, y = robot["position"]
        status = robot["status"]
        busy_flag = float(status in {"travel", "onsite", "waiting_sync"})
        row = np.array(
            [
                1.0,
                min(float(robot["speed_multiplier"]) / 1.5, 1.0),
                min(float(robot["service_multiplier"]) / 2.5, 1.0),
                min(robot_release_eta_fn(robot_id) / max(max_time, EPS), 1.0),
                min(float(robot.get("wait_elapsed", 0.0)) / max(float(robot.get("wait_timeout", 0.0)) or 1.0, EPS), 1.0),
                np.clip((float(x) - float(current_x)) / max(float(WIDTH), 1.0) + 0.5, 0.0, 1.0),
                np.clip((float(y) - float(current_y)) / max(float(HEIGHT), 1.0) + 0.5, 0.0, 1.0),
                float(robot.get("assigned_task") is not None),
                busy_flag,
                float(status == "waiting_sync"),
                float(status == "onsite"),
            ],
            dtype=np.float32,
        )
        agent_inputs[slot] = np.concatenate([row, role_vector(str(robot["role"]))])

    for task_slot, task_id in enumerate(task_order, start=1):
        if task_slot >= num_task_tokens:
            break
        task = task_specs[task_id]
        task_state = task_states[task_id]
        x, y = task["pos"]
        eta, predicted_start, predicted_finish = predict_task_outcome_fn(current_robot_id, task_id)
        required_roles = task.get("required_roles", {})
        required_vec = required_roles_vector(required_roles)
        row = np.zeros(TASK_FEATURE_DIM, dtype=np.float32)
        row[TASK_IDX["presence"]] = 1.0
        row[TASK_IDX["remaining_role_deficit_start"]:TASK_IDX["remaining_role_deficit_start"] + ROLE_DIM] = remaining_role_deficits[task_slot - 1]
        row[TASK_IDX["required_roles_start"]:TASK_IDX["required_roles_start"] + ROLE_DIM] = required_vec
        row[TASK_IDX["service_time"]] = min(float(task["service_time"]) / max(max_time, EPS), 1.0)
        row[TASK_IDX["eta"]] = min(eta / max(max_time, EPS), 1.0) if np.isfinite(eta) else 1.0
        row[TASK_IDX["predicted_start"]] = min(predicted_start / max(max_time, EPS), 1.0) if np.isfinite(predicted_start) else 1.0
        row[TASK_IDX["predicted_finish"]] = min(predicted_finish / max(max_time, EPS), 1.0) if np.isfinite(predicted_finish) else 1.0
        row[TASK_IDX["priority"]] = min(float(task.get("priority", 1.0)) / 2.0, 1.0)
        row[TASK_IDX["unlock_value"]] = min(float(task_unlock_value_fn(task_id)) / 4.0, 1.0)
        row[TASK_IDX["precedence_satisfied"]] = precedence_state[task_slot - 1]
        row[TASK_IDX["contributors_count"]] = min(len(task_state.get("contributors", set())) / 3.0, 1.0)
        row[TASK_IDX["waiting_count"]] = min(float(task_waiting_count_fn(task_id, task_state)) / 3.0, 1.0)
        row[TASK_IDX["is_single"]] = float(task["kind"] == "single")
        row[TASK_IDX["is_sync"]] = float(task["kind"] == "sync")
        row[TASK_IDX["dx"]] = np.clip((float(x) - float(current_x)) / max(float(WIDTH), 1.0) + 0.5, 0.0, 1.0)
        row[TASK_IDX["dy"]] = np.clip((float(y) - float(current_y)) / max(float(HEIGHT), 1.0) + 0.5, 0.0, 1.0)
        task_inputs[task_slot] = row

    return {
        "agent_inputs": agent_inputs,
        "task_inputs": task_inputs,
        "global_mask": (1.0 - legal_mask).astype(np.float32),
        "current_agent_index": current_agent_index,
    }


class HeteroDispatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: Optional[SchedulingEnv] = None,
        scenarios: Optional[Sequence[Dict]] = None,
        scenario_dir: str = "offline_maps_v2",
        split: str = "train",
        family: Optional[str] = None,
        max_robots: int = MAX_ROBOTS,
        max_tasks: int = MAX_TASKS,
        max_time: float = 2500.0,
        wait_timeout: float = 60.0,
        mask_idle_wait_when_legal: bool = True,
        completion_reward: float = 12.0,
        progress_reward_scale: float = 6.0,
        coalition_activation_bonus: float = 0.5,
        episode_success_bonus: float = 40.0,
        unfinished_task_penalty: float = 12.0,
        locked_task_penalty: float = 5.0,
        precedence_unlock_bonus: float = 3.0,
        time_penalty: float = 0.02,
        idle_penalty: float = 0.01,
        wait_penalty: float = 0.01,
        avoidable_wait_penalty: float = 0.02,
        reassign_progress_bonus_scale: float = 0.0,
        reassign_completion_bonus: float = 0.0,
        bad_reassign_penalty: float = 0.0,
        direct_sync_gap_margin: float = 20.0,
        direct_sync_gap_penalty_scale: float = 120.0,
        direct_sync_gap_penalty_max: float = 2.0,
        productive_single_margin: float = 20.0,
        productive_single_bonus_scale: float = 160.0,
        productive_single_bonus_max: float = 1.0,
        unlock_shaping_scale: float = 0.15,
    ):
        super().__init__()
        self.base_env = base_env or SchedulingEnv(
            scenarios=scenarios,
            scenario_dir=scenario_dir,
            split=split,
            family=family,
            max_robots=max_robots,
            max_tasks=max_tasks,
            max_time=max_time,
            wait_timeout=wait_timeout,
            completion_reward=completion_reward,
            progress_reward_scale=progress_reward_scale,
            coalition_activation_bonus=coalition_activation_bonus,
            episode_success_bonus=episode_success_bonus,
            unfinished_task_penalty=unfinished_task_penalty,
            locked_task_penalty=locked_task_penalty,
            precedence_unlock_bonus=precedence_unlock_bonus,
            time_penalty=time_penalty,
            idle_penalty=idle_penalty,
            wait_penalty=wait_penalty,
            avoidable_wait_penalty=avoidable_wait_penalty,
            reassign_progress_bonus_scale=reassign_progress_bonus_scale,
            reassign_completion_bonus=reassign_completion_bonus,
            bad_reassign_penalty=bad_reassign_penalty,
        )
        self.max_robots = self.base_env.max_robots
        self.max_tasks = self.base_env.max_tasks
        self.num_task_tokens = self.max_tasks + 1
        self.mask_idle_wait_when_legal = mask_idle_wait_when_legal
        self.direct_sync_gap_margin = direct_sync_gap_margin
        self.direct_sync_gap_penalty_scale = direct_sync_gap_penalty_scale
        self.direct_sync_gap_penalty_max = direct_sync_gap_penalty_max
        self.productive_single_margin = productive_single_margin
        self.productive_single_bonus_scale = productive_single_bonus_scale
        self.productive_single_bonus_max = productive_single_bonus_max
        self.unlock_shaping_scale = unlock_shaping_scale
        self.action_space = spaces.Discrete(self.num_task_tokens)
        self.observation_space = spaces.Dict(
            {
                "agent_inputs": spaces.Box(0.0, 1.0, shape=(self.max_robots, AGENT_FEATURE_DIM), dtype=np.float32),
                "task_inputs": spaces.Box(0.0, 1.0, shape=(self.num_task_tokens, TASK_FEATURE_DIM), dtype=np.float32),
                "global_mask": spaces.Box(0.0, 1.0, shape=(self.num_task_tokens,), dtype=np.float32),
                "current_agent_index": spaces.Box(0.0, float(self.max_robots), shape=(1,), dtype=np.float32),
            }
        )
        self.pending_actions = np.zeros(self.max_robots, dtype=np.int64)
        self.event_slots: list[int] = []
        self.event_cursor = 0
        self.event_index = 0
        self.last_info: Dict = {}
        self.terminated = False
        self.truncated = False
        self.metrics = {
            "direct_sync_assignments": 0,
            "direct_sync_misassignments": 0,
            "decision_count": 0,
            "wait_action_count": 0,
            "idle_legal_decision_count": 0,
            "idle_wait_action_count": 0,
            "waiting_idle_legal_decision_count": 0,
            "waiting_idle_wait_action_count": 0,
            "stalled_wait_count": 0,
            "wait_flip_count": 0,
            "dispatch_gap_penalty_total": 0.0,
        }
        self.last_step_shaping_reward = 0.0
        self.last_step_dispatch_gap_penalty = 0.0

    @property
    def robot_order(self) -> list[str]:
        return self.base_env.robot_order

    @property
    def task_order(self) -> list[str]:
        return self.base_env.task_order

    def _current_slot(self) -> Optional[int]:
        if self.event_cursor >= len(self.event_slots):
            return None
        return self.event_slots[self.event_cursor]

    def get_current_robot_id(self) -> Optional[str]:
        slot = self._current_slot()
        if slot is None:
            return None
        return self.robot_order[slot]

    def _collect_event_slots(self) -> list[int]:
        slots: list[int] = []
        for slot, robot_id in enumerate(self.robot_order):
            robot = self.base_env.robot_states[robot_id]
            if robot.get("status") in TRAINABLE_STATUSES:
                slots.append(slot)
        return slots

    def _reset_metrics(self) -> None:
        self.metrics = {
            "direct_sync_assignments": 0,
            "direct_sync_misassignments": 0,
            "decision_count": 0,
            "wait_action_count": 0,
            "idle_legal_decision_count": 0,
            "idle_wait_action_count": 0,
            "waiting_idle_legal_decision_count": 0,
            "waiting_idle_wait_action_count": 0,
            "stalled_wait_count": 0,
            "wait_flip_count": 0,
            "dispatch_gap_penalty_total": 0.0,
        }
        self.last_step_shaping_reward = 0.0
        self.last_step_dispatch_gap_penalty = 0.0

    def _robot_release_eta(self, robot_id: str) -> float:
        robot = self.base_env.robot_states[robot_id]
        status = robot["status"]
        if status in {"idle", "waiting_idle"}:
            return 0.0
        if status == "travel":
            return float(robot.get("eta_remaining", 0.0))
        if status == "waiting_sync":
            return max(
                float(robot.get("current_task_ready_eta", self.base_env.wait_timeout)),
                self.base_env.wait_timeout - float(robot.get("wait_elapsed", 0.0)),
            )
        if status == "onsite" and robot.get("assigned_task") is not None:
            return float(self.base_env._service_eta_for_task(robot["assigned_task"]))
        return 0.0

    @staticmethod
    def _task_waiting_count(task_id: str, task_state: Dict) -> int:
        del task_id
        return int(len(task_state.get("waiting_sync_robot_ids", set())))

    def _auto_advance_until_trainable(self) -> float:
        reward = 0.0
        while True:
            self.event_slots = self._collect_event_slots()
            if self.event_slots or self.base_env.terminated or self.base_env.truncated:
                break
            wait_action = np.zeros(self.base_env.max_robots, dtype=np.int64)
            _, step_reward, terminated, truncated, _ = self.base_env.step(wait_action)
            reward += float(step_reward)
            self.terminated = terminated
            self.truncated = truncated
            if terminated or truncated:
                break
        self.event_cursor = 0
        return reward

    def _legal_action_mask(self, robot_id: Optional[str]) -> np.ndarray:
        if robot_id is None:
            mask = np.zeros(self.num_task_tokens, dtype=np.float32)
            mask[0] = 1.0
            return mask
        mask = legal_action_mask_for_robot(
            robot_id=robot_id,
            robot_order=self.robot_order,
            task_order=self.task_order,
            robot_states=self.base_env.robot_states,
            task_states=self.base_env.task_states,
            task_specs=self.base_env.task_specs,
            pending_actions=self.pending_actions,
            max_tasks=self.max_tasks,
        ).astype(np.float32)
        if self.mask_idle_wait_when_legal:
            mask = constrain_wait_action_mask(mask, self.base_env.robot_states[robot_id])
        return mask

    def _record_wait_pattern_metrics(
        self,
        robot_id: str,
        *,
        robot_status: str,
        has_legal_task: bool,
        action: int,
    ) -> None:
        robot = self.base_env.robot_states[robot_id]
        action_is_wait = int(action) == 0
        if robot_status == "idle" and has_legal_task:
            self.metrics["idle_legal_decision_count"] += 1
            if action_is_wait:
                self.metrics["idle_wait_action_count"] += 1
        if robot_status == "waiting_idle" and has_legal_task:
            self.metrics["waiting_idle_legal_decision_count"] += 1
            legal_streak = int(robot.get("waiting_idle_legal_streak", 0)) + 1
            robot["waiting_idle_legal_streak"] = legal_streak
            if action_is_wait:
                self.metrics["waiting_idle_wait_action_count"] += 1
                if legal_streak >= 2:
                    self.metrics["stalled_wait_count"] += 1
            else:
                robot["waiting_idle_legal_streak"] = 0
        else:
            robot["waiting_idle_legal_streak"] = 0

        previous_wait = robot.get("last_dispatch_was_wait")
        if has_legal_task and previous_wait is not None and bool(previous_wait) != action_is_wait:
            self.metrics["wait_flip_count"] += 1
        if has_legal_task:
            robot["last_dispatch_was_wait"] = action_is_wait

    def _pending_other_assignments(self, task_id: str) -> tuple[set[str], set[str]]:
        assignments = pending_task_assignments(self.robot_order, self.task_order, self.pending_actions)
        same_task = set(assignments.get(task_id, set()))
        other_tasks = {
            robot_id
            for other_task_id, robot_ids in assignments.items()
            if other_task_id != task_id
            for robot_id in robot_ids
        }
        return same_task, other_tasks

    def _predict_task_outcome(self, robot_id: str, task_id: str) -> tuple[float, float, float]:
        task = self.base_env.task_specs[task_id]
        robot = self.base_env.robot_states[robot_id]
        direct_eta = float(self.base_env._robot_available_eta_for_task(robot_id, task_id))
        if task["kind"] == "single":
            finish_eta = direct_eta + float(task["service_time"]) / max(float(robot["service_multiplier"]), EPS)
            return direct_eta, direct_eta, finish_eta

        same_task_pending, other_pending = self._pending_other_assignments(task_id)
        required_roles = task.get("required_roles", {})
        if not required_roles:
            finish_eta = direct_eta + float(task["service_time"]) / max(float(robot["service_multiplier"]), EPS)
            return direct_eta, direct_eta, finish_eta

        selected_ids: list[str] = []
        ready_etas: list[float] = []

        for role, need in required_roles.items():
            role_candidates: list[tuple[float, str]] = []
            for candidate_id, candidate in self.base_env.robot_states.items():
                if candidate["role"] != role:
                    continue
                if candidate_id in other_pending:
                    continue
                if candidate_id == robot_id or candidate_id in same_task_pending:
                    eta = float(self.base_env._robot_available_eta_for_task(candidate_id, task_id))
                else:
                    eta = float(self.base_env._robot_available_eta_for_task(candidate_id, task_id))
                if np.isfinite(eta):
                    role_candidates.append((eta, candidate_id))
            role_candidates.sort(key=lambda item: item[0])

            chosen = []
            used = set(selected_ids)
            for eta, candidate_id in role_candidates:
                if candidate_id in used:
                    continue
                chosen.append((eta, candidate_id))
                used.add(candidate_id)
                if len(chosen) >= need:
                    break
            if len(chosen) < need:
                return direct_eta, float("inf"), float("inf")
            ready_etas.extend(eta for eta, _ in chosen)
            selected_ids.extend(candidate_id for _, candidate_id in chosen)

        predicted_start = max(ready_etas, default=direct_eta)
        rate = service_rate(
            selected_ids,
            {
                candidate_id: {
                    "service_multiplier": self.base_env.robot_states[candidate_id]["service_multiplier"],
                }
                for candidate_id in selected_ids
            },
        )
        predicted_finish = predicted_start + float(task["service_time"]) / max(rate, EPS)
        return direct_eta, predicted_start, predicted_finish

    def _agent_inputs(self, current_robot_id: str) -> np.ndarray:
        current_robot = self.base_env.robot_states[current_robot_id]
        current_x, current_y = current_robot["position"]
        agent_inputs = np.zeros((self.max_robots, AGENT_FEATURE_DIM), dtype=np.float32)

        for slot, robot_id in enumerate(self.robot_order):
            if slot >= self.max_robots:
                break
            robot = self.base_env.robot_states[robot_id]
            x, y = robot["position"]
            status = robot["status"]
            if status in {"idle", "waiting_idle"}:
                time_to_release = 0.0
            elif status == "travel":
                time_to_release = float(robot.get("eta_remaining", 0.0))
            elif status == "waiting_sync":
                time_to_release = max(
                    float(robot.get("current_task_ready_eta", self.base_env.wait_timeout)),
                    self.base_env.wait_timeout - float(robot.get("wait_elapsed", 0.0)),
                )
            elif status == "onsite" and robot.get("assigned_task") is not None:
                time_to_release = float(self.base_env._service_eta_for_task(robot["assigned_task"]))
            else:
                time_to_release = 0.0
            busy_flag = float(status in {"travel", "onsite", "waiting_sync"})
            row = np.array(
                [
                    1.0,
                    min(float(robot["speed_multiplier"]) / 1.5, 1.0),
                    min(float(robot["service_multiplier"]) / 2.5, 1.0),
                    min(time_to_release / max(self.base_env.max_time, EPS), 1.0),
                    min(float(robot.get("wait_elapsed", 0.0)) / max(self.base_env.wait_timeout, EPS), 1.0),
                    np.clip((float(x) - float(current_x)) / max(float(WIDTH), 1.0) + 0.5, 0.0, 1.0),
                    np.clip((float(y) - float(current_y)) / max(float(HEIGHT), 1.0) + 0.5, 0.0, 1.0),
                    float(robot.get("assigned_task") is not None),
                    busy_flag,
                    float(status == "waiting_sync"),
                    float(status == "onsite"),
                ],
                dtype=np.float32,
            )
            agent_inputs[slot] = np.concatenate([row, role_vector(str(robot["role"]))])
        return agent_inputs

    def _task_inputs(self, current_robot_id: str) -> np.ndarray:
        current_robot = self.base_env.robot_states[current_robot_id]
        current_x, current_y = current_robot["position"]
        task_inputs = np.zeros((self.num_task_tokens, TASK_FEATURE_DIM), dtype=np.float32)

        deficits = remaining_role_deficit_matrix(
            task_order=self.task_order,
            task_specs=self.base_env.task_specs,
            task_states=self.base_env.task_states,
            robot_states=self.base_env.robot_states,
            pending_actions=self.pending_actions,
            robot_order=self.robot_order,
            max_tasks=self.max_tasks,
        )
        precedence_state = precedence_state_vector(
            task_order=self.task_order,
            task_specs=self.base_env.task_specs,
            task_states=self.base_env.task_states,
            max_tasks=self.max_tasks,
        )

        wait_row = np.zeros(TASK_FEATURE_DIM, dtype=np.float32)
        wait_row[TASK_IDX["presence"]] = 1.0
        wait_row[TASK_IDX["is_wait_node"]] = 1.0
        task_inputs[0] = wait_row

        for task_slot, task_id in enumerate(self.task_order, start=1):
            if task_slot >= self.num_task_tokens:
                break
            task = self.base_env.task_specs[task_id]
            task_state = self.base_env.task_states[task_id]
            x, y = task["pos"]
            eta, predicted_start, predicted_finish = self._predict_task_outcome(current_robot_id, task_id)
            required_roles = task.get("required_roles", {})
            required_vec = required_roles_vector(required_roles)
            deficit_vec = deficits[task_slot - 1]
            row = np.zeros(TASK_FEATURE_DIM, dtype=np.float32)
            row[TASK_IDX["presence"]] = 1.0
            row[TASK_IDX["remaining_role_deficit_start"]:TASK_IDX["remaining_role_deficit_start"] + ROLE_DIM] = deficit_vec
            row[TASK_IDX["required_roles_start"]:TASK_IDX["required_roles_start"] + ROLE_DIM] = required_vec
            row[TASK_IDX["service_time"]] = min(float(task["service_time"]) / max(self.base_env.max_time, EPS), 1.0)
            row[TASK_IDX["eta"]] = min(eta / max(self.base_env.max_time, EPS), 1.0) if np.isfinite(eta) else 1.0
            row[TASK_IDX["predicted_start"]] = (
                min(predicted_start / max(self.base_env.max_time, EPS), 1.0) if np.isfinite(predicted_start) else 1.0
            )
            row[TASK_IDX["predicted_finish"]] = (
                min(predicted_finish / max(self.base_env.max_time, EPS), 1.0) if np.isfinite(predicted_finish) else 1.0
            )
            row[TASK_IDX["priority"]] = min(float(task.get("priority", 1.0)) / 2.0, 1.0)
            row[TASK_IDX["unlock_value"]] = min(float(self.base_env._task_unlock_value(task_id)) / 4.0, 1.0)
            row[TASK_IDX["precedence_satisfied"]] = precedence_state[task_slot - 1]
            row[TASK_IDX["contributors_count"]] = min(len(task_state["contributors"]) / 3.0, 1.0)
            row[TASK_IDX["waiting_count"]] = min(len(task_state["waiting_sync_robot_ids"]) / 3.0, 1.0)
            row[TASK_IDX["is_single"]] = float(task["kind"] == "single")
            row[TASK_IDX["is_sync"]] = float(task["kind"] == "sync")
            row[TASK_IDX["dx"]] = np.clip((float(x) - float(current_x)) / max(float(WIDTH), 1.0) + 0.5, 0.0, 1.0)
            row[TASK_IDX["dy"]] = np.clip((float(y) - float(current_y)) / max(float(HEIGHT), 1.0) + 0.5, 0.0, 1.0)
            task_inputs[task_slot] = row
        return task_inputs

    def _get_obs(self) -> Dict[str, np.ndarray]:
        current_slot = self._current_slot()
        current_robot_id = self.robot_order[current_slot] if current_slot is not None else None
        legal_mask = self._legal_action_mask(current_robot_id) if current_robot_id is not None else np.zeros(self.num_task_tokens, dtype=np.float32)
        if current_slot is None:
            legal_mask[0] = 1.0
        deficits = remaining_role_deficit_matrix(
            task_order=self.task_order,
            task_specs=self.base_env.task_specs,
            task_states=self.base_env.task_states,
            robot_states=self.base_env.robot_states,
            pending_actions=self.pending_actions,
            robot_order=self.robot_order,
            max_tasks=self.max_tasks,
        )
        precedence_state = precedence_state_vector(
            task_order=self.task_order,
            task_specs=self.base_env.task_specs,
            task_states=self.base_env.task_states,
            max_tasks=self.max_tasks,
        )
        return build_hetero_scheduler_observation(
            robot_order=self.robot_order,
            task_order=self.task_order,
            robot_states=self.base_env.robot_states,
            task_states=self.base_env.task_states,
            task_specs=self.base_env.task_specs,
            current_slot=current_slot,
            legal_mask=legal_mask,
            remaining_role_deficits=deficits,
            precedence_state=precedence_state,
            robot_release_eta_fn=self._robot_release_eta,
            predict_task_outcome_fn=self._predict_task_outcome,
            task_waiting_count_fn=self._task_waiting_count,
            task_unlock_value_fn=self.base_env._task_unlock_value,
            max_time=self.base_env.max_time,
            max_robots=self.max_robots,
            num_task_tokens=self.num_task_tokens,
        )

    def _sync_misassignment_for(self, robot_id: str, chosen_task_id: str) -> bool:
        chosen_task = self.base_env.task_specs[chosen_task_id]
        if chosen_task["kind"] != "sync":
            return False

        _, predicted_start, _ = self._predict_task_outcome(robot_id, chosen_task_id)
        best_single_finish = float("inf")
        for task_id in self.task_order:
            task = self.base_env.task_specs[task_id]
            if task["kind"] != "single":
                continue
            if not self.base_env._is_action_legal(robot_id, task_id):
                continue
            _, _, predicted_finish = self._predict_task_outcome(robot_id, task_id)
            best_single_finish = min(best_single_finish, predicted_finish)
        return np.isfinite(best_single_finish) and predicted_start > best_single_finish + self.direct_sync_gap_margin

    def _comparison_summary(self, robot_id: str) -> Dict[str, float]:
        legal_mask = self._legal_action_mask(robot_id)
        best_single_finish = float("inf")
        best_sync_start = float("inf")
        best_sync_finish = float("inf")

        for action in np.flatnonzero(legal_mask > 0.0):
            if action == 0:
                continue
            task_id = task_id_from_action(int(action), self.task_order)
            if task_id is None:
                continue
            task = self.base_env.task_specs[task_id]
            _, predicted_start, predicted_finish = self._predict_task_outcome(robot_id, task_id)
            if not np.isfinite(predicted_finish):
                continue
            if task["kind"] == "single":
                best_single_finish = min(best_single_finish, predicted_finish)
            else:
                best_sync_start = min(best_sync_start, predicted_start)
                best_sync_finish = min(best_sync_finish, predicted_finish)

        return {
            "best_single_finish": best_single_finish,
            "best_sync_start": best_sync_start,
            "best_sync_finish": best_sync_finish,
        }

    def _has_single_sync_conflict(self, robot_id: str, legal_mask: np.ndarray | None = None) -> bool:
        if legal_mask is None:
            legal_mask = self._legal_action_mask(robot_id)
        has_single = False
        has_sync = False
        for action in np.flatnonzero(legal_mask > 0.0):
            if int(action) == 0:
                continue
            task_id = task_id_from_action(int(action), self.task_order)
            if task_id is None:
                continue
            task = self.base_env.task_specs[task_id]
            if task["kind"] == "single":
                has_single = True
            elif task["kind"] == "sync":
                has_sync = True
            if has_single and has_sync:
                return True
        return False

    def _is_hard_teacher_state(self, robot_id: str, legal_mask: np.ndarray | None = None) -> bool:
        if legal_mask is None:
            legal_mask = self._legal_action_mask(robot_id)
        family = str((self.base_env.current_scenario or {}).get("family", ""))
        legal_non_wait = int(np.count_nonzero(legal_mask[1:] > 0.0))
        if family in HARD_TEACHER_FAMILIES:
            return True
        if legal_non_wait > 2:
            return True
        if self._has_single_sync_conflict(robot_id, legal_mask):
            return True
        return False

    def _upfront_action_shaping(self, robot_id: str, chosen_task_id: str) -> tuple[float, float]:
        task = self.base_env.task_specs[chosen_task_id]
        _, predicted_start, predicted_finish = self._predict_task_outcome(robot_id, chosen_task_id)
        if not np.isfinite(predicted_finish):
            return 0.0, 0.0

        reward = self.unlock_shaping_scale * min(float(self.base_env._task_unlock_value(chosen_task_id)), 3.0)
        dispatch_gap_penalty = 0.0
        comparison = self._comparison_summary(robot_id)

        if task["kind"] == "sync" and np.isfinite(comparison["best_single_finish"]):
            gap = predicted_start - comparison["best_single_finish"] - self.direct_sync_gap_margin
            if gap > 0.0:
                penalty = min(self.direct_sync_gap_penalty_max, gap / self.direct_sync_gap_penalty_scale)
                reward -= penalty
                dispatch_gap_penalty = penalty

        if task["kind"] == "single" and np.isfinite(comparison["best_sync_start"]):
            lead = comparison["best_sync_start"] - predicted_finish - self.productive_single_margin
            if lead > 0.0:
                reward += min(self.productive_single_bonus_max, lead / self.productive_single_bonus_scale)

        return reward, dispatch_gap_penalty

    def upfront_wait_aware_action(self) -> int:
        current_robot_id = self.get_current_robot_id()
        if current_robot_id is None:
            return 0

        legal_mask = self._legal_action_mask(current_robot_id)
        best_action = 0
        best_score = float("inf")
        fallback_action = 0
        fallback_key: tuple[float, float, int] | None = None
        has_legal_task = False

        for action in np.flatnonzero(legal_mask > 0.0):
            if action == 0:
                continue
            has_legal_task = True
            task_id = task_id_from_action(int(action), self.task_order)
            if task_id is None:
                continue
            task = self.base_env.task_specs[task_id]
            eta, predicted_start, predicted_finish = self._predict_task_outcome(current_robot_id, task_id)
            fallback_eta = eta if np.isfinite(eta) else float("inf")
            fallback_start = predicted_start if np.isfinite(predicted_start) else float("inf")
            candidate_key = (fallback_eta, fallback_start, int(action))
            if fallback_key is None or candidate_key < fallback_key:
                fallback_key = candidate_key
                fallback_action = int(action)
            if not np.isfinite(predicted_finish):
                continue
            waiting_gap = max(0.0, predicted_start - eta)
            unlock_bonus = 20.0 * float(self.base_env._task_unlock_value(task_id))
            priority_bonus = 45.0 * float(task.get("priority", 1.0))
            single_bias = 10.0 if task["kind"] == "single" else 0.0
            score = predicted_finish + 0.8 * waiting_gap - unlock_bonus - priority_bonus - single_bias
            if score < best_score - 1e-6:
                best_score = score
                best_action = int(action)

        if best_action != 0:
            return best_action
        if has_legal_task:
            return fallback_action
        return 0

    def _build_info(self, committed: bool) -> Dict:
        info = dict(self.base_env._build_info())
        current_slot = self._current_slot()
        rate = 0.0
        if self.metrics["direct_sync_assignments"] > 0:
            rate = self.metrics["direct_sync_misassignments"] / self.metrics["direct_sync_assignments"]
        wait_action_rate = 0.0
        if self.metrics["decision_count"] > 0:
            wait_action_rate = self.metrics["wait_action_count"] / self.metrics["decision_count"]
        idle_wait_rate = 0.0
        if self.metrics["idle_legal_decision_count"] > 0:
            idle_wait_rate = self.metrics["idle_wait_action_count"] / self.metrics["idle_legal_decision_count"]
        waiting_idle_wait_rate = 0.0
        if self.metrics["waiting_idle_legal_decision_count"] > 0:
            waiting_idle_wait_rate = (
                self.metrics["waiting_idle_wait_action_count"] / self.metrics["waiting_idle_legal_decision_count"]
            )
        stalled_wait_rate = 0.0
        if self.metrics["waiting_idle_legal_decision_count"] > 0:
            stalled_wait_rate = self.metrics["stalled_wait_count"] / self.metrics["waiting_idle_legal_decision_count"]
        wait_flip_rate = 0.0
        if self.metrics["decision_count"] > 0:
            wait_flip_rate = self.metrics["wait_flip_count"] / self.metrics["decision_count"]
        dispatch_gap_penalty = 0.0
        if self.metrics["decision_count"] > 0:
            dispatch_gap_penalty = self.metrics["dispatch_gap_penalty_total"] / self.metrics["decision_count"]
        info.update(
            {
                "event_committed": committed,
                "event_index": self.event_index,
                "current_robot_slot": current_slot,
                "current_robot_id": self.robot_order[current_slot] if current_slot is not None else None,
                "direct_sync_misassignment_rate": rate,
                "wait_action_rate": wait_action_rate,
                "idle_wait_rate": idle_wait_rate,
                "waiting_idle_wait_rate": waiting_idle_wait_rate,
                "stalled_wait_rate": stalled_wait_rate,
                "wait_flip_rate": wait_flip_rate,
                "dispatch_gap_penalty": dispatch_gap_penalty,
            }
        )
        return info

    def reset(self, seed: int | None = None, options: Optional[dict] = None):
        self.base_env.reset(seed=seed, options=options)
        for robot in self.base_env.robot_states.values():
            robot["last_dispatch_was_wait"] = None
            robot["waiting_idle_legal_streak"] = 0
        self.pending_actions = np.zeros(self.max_robots, dtype=np.int64)
        self.event_index = 0
        self.terminated = False
        self.truncated = False
        self._reset_metrics()
        self._auto_advance_until_trainable()
        self.last_info = self._build_info(committed=False)
        return self._get_obs(), self.last_info

    def get_state_snapshot(self) -> Dict:
        return {
            "base_env": {
                "current_scenario": deepcopy(self.base_env.current_scenario),
                "robot_order": list(self.base_env.robot_order),
                "task_order": list(self.base_env.task_order),
                "robot_specs": deepcopy(self.base_env.robot_specs),
                "task_specs": deepcopy(self.base_env.task_specs),
                "robot_states": deepcopy(self.base_env.robot_states),
                "task_states": deepcopy(self.base_env.task_states),
                "metrics": deepcopy(self.base_env.metrics),
                "pending_reassign_evals": deepcopy(self.base_env.pending_reassign_evals),
                "time": float(self.base_env.time),
                "terminated": bool(self.base_env.terminated),
                "truncated": bool(self.base_env.truncated),
            },
            "pending_actions": np.array(self.pending_actions, copy=True),
            "event_slots": list(self.event_slots),
            "event_cursor": int(self.event_cursor),
            "event_index": int(self.event_index),
            "last_info": deepcopy(self.last_info),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "metrics": deepcopy(self.metrics),
            "last_step_shaping_reward": float(self.last_step_shaping_reward),
            "last_step_dispatch_gap_penalty": float(self.last_step_dispatch_gap_penalty),
        }

    def set_state_snapshot(self, snapshot: Dict) -> None:
        base_env_snapshot = snapshot["base_env"]
        self.base_env.current_scenario = deepcopy(base_env_snapshot["current_scenario"])
        self.base_env.robot_order = list(base_env_snapshot["robot_order"])
        self.base_env.task_order = list(base_env_snapshot["task_order"])
        self.base_env.robot_specs = deepcopy(base_env_snapshot["robot_specs"])
        self.base_env.task_specs = deepcopy(base_env_snapshot["task_specs"])
        self.base_env.robot_states = deepcopy(base_env_snapshot["robot_states"])
        self.base_env.task_states = deepcopy(base_env_snapshot["task_states"])
        self.base_env.metrics = deepcopy(base_env_snapshot["metrics"])
        self.base_env.pending_reassign_evals = deepcopy(base_env_snapshot["pending_reassign_evals"])
        self.base_env.time = float(base_env_snapshot["time"])
        self.base_env.terminated = bool(base_env_snapshot["terminated"])
        self.base_env.truncated = bool(base_env_snapshot["truncated"])

        self.pending_actions = np.array(snapshot["pending_actions"], copy=True)
        self.event_slots = list(snapshot["event_slots"])
        self.event_cursor = int(snapshot["event_cursor"])
        self.event_index = int(snapshot["event_index"])
        self.last_info = deepcopy(snapshot["last_info"])
        self.terminated = bool(snapshot["terminated"])
        self.truncated = bool(snapshot["truncated"])
        self.metrics = deepcopy(snapshot["metrics"])
        self.last_step_shaping_reward = float(snapshot["last_step_shaping_reward"])
        self.last_step_dispatch_gap_penalty = float(snapshot["last_step_dispatch_gap_penalty"])

    @staticmethod
    def _teacher_short_horizon_score(final_info: Dict, done: bool, truncated: bool, shaping_total: float) -> float:
        success = 1.0 if done and not truncated else 0.0
        completion_rate = float(final_info.get("completion_rate", 0.0))
        makespan = float(final_info.get("makespan", 0.0))
        wait_time = float(final_info.get("average_wait_time", 0.0))
        avoidable_wait = float(final_info.get("average_avoidable_wait_time", 0.0))
        misassign_rate = float(final_info.get("direct_sync_misassignment_rate", 0.0))
        activation_delay = float(final_info.get("coalition_activation_delay", 0.0))
        deadlock_events = float(final_info.get("deadlock_events", 0.0))
        return (
            1000.0 * success
            + 400.0 * completion_rate
            - 1.0 * makespan
            - 0.4 * wait_time
            - 0.8 * avoidable_wait
            - 80.0 * misassign_rate
            - 0.05 * activation_delay
            - 100.0 * deadlock_events
            + float(shaping_total)
        )

    def rollout_upfront_teacher_action(self, rollout_depth: int = 2) -> int:
        current_robot_id = self.get_current_robot_id()
        if current_robot_id is None:
            return 0

        legal_mask = self._legal_action_mask(current_robot_id)
        legal_actions = [int(action) for action in np.flatnonzero(legal_mask > 0.0)]
        if not legal_actions:
            return 0

        best_action = legal_actions[0]
        best_score = float("-inf")
        root_snapshot = self.get_state_snapshot()

        for action in legal_actions:
            score = self._teacher_action_score(int(action), root_snapshot, rollout_depth=rollout_depth)
            if score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and action < best_action):
                best_score = score
                best_action = int(action)

        self.set_state_snapshot(root_snapshot)
        return best_action

    def _heuristic_action_ranking(
        self,
        robot_id: str,
        legal_mask: np.ndarray,
    ) -> list[int]:
        ranked: list[tuple[float, int]] = []
        for action in np.flatnonzero(legal_mask > 0.0):
            action = int(action)
            if action == 0:
                continue
            task_id = task_id_from_action(action, self.task_order)
            if task_id is None:
                continue
            task = self.base_env.task_specs[task_id]
            eta, predicted_start, predicted_finish = self._predict_task_outcome(robot_id, task_id)
            if not np.isfinite(predicted_finish):
                fallback_finish = float("inf")
            else:
                fallback_finish = predicted_finish
            waiting_gap = max(0.0, predicted_start - eta) if np.isfinite(predicted_start) and np.isfinite(eta) else float("inf")
            unlock_bonus = 20.0 * float(self.base_env._task_unlock_value(task_id))
            priority_bonus = 45.0 * float(task.get("priority", 1.0))
            single_bias = 10.0 if task["kind"] == "single" else 0.0
            score = fallback_finish + 0.8 * waiting_gap - unlock_bonus - priority_bonus - single_bias
            ranked.append((score, action))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [action for _, action in ranked]

    def _best_action_by_kind(
        self,
        robot_id: str,
        legal_mask: np.ndarray,
        kind: str,
    ) -> int | None:
        best_action: int | None = None
        best_key: tuple[float, float, int] | None = None
        for action in np.flatnonzero(legal_mask > 0.0):
            action = int(action)
            if action == 0:
                continue
            task_id = task_id_from_action(action, self.task_order)
            if task_id is None:
                continue
            task = self.base_env.task_specs[task_id]
            if task["kind"] != kind:
                continue
            eta, predicted_start, predicted_finish = self._predict_task_outcome(robot_id, task_id)
            candidate_key = (
                predicted_finish if np.isfinite(predicted_finish) else float("inf"),
                predicted_start if np.isfinite(predicted_start) else (eta if np.isfinite(eta) else float("inf")),
                action,
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_action = action
        return best_action

    def teacher_candidate_actions(self, rollout_depth: int = 2, top_k: int = 2) -> list[int]:
        del rollout_depth
        current_robot_id = self.get_current_robot_id()
        if current_robot_id is None:
            return [0]
        legal_mask = self._legal_action_mask(current_robot_id)
        legal_actions = [int(action) for action in np.flatnonzero(legal_mask > 0.0)]
        if not legal_actions:
            return [0]
        if not self._is_hard_teacher_state(current_robot_id, legal_mask):
            return [int(self.upfront_wait_aware_action())]

        candidates: list[int] = []
        candidates.append(int(self.upfront_wait_aware_action()))
        best_single = self._best_action_by_kind(current_robot_id, legal_mask, "single")
        if best_single is not None:
            candidates.append(int(best_single))
        best_sync = self._best_action_by_kind(current_robot_id, legal_mask, "sync")
        if best_sync is not None:
            candidates.append(int(best_sync))
        heuristic_ranked = self._heuristic_action_ranking(current_robot_id, legal_mask)
        candidates.extend(int(action) for action in heuristic_ranked[:max(top_k, 0)])

        deduped: list[int] = []
        seen: set[int] = set()
        for action in candidates:
            if action in seen:
                continue
            if action < 0 or action >= legal_mask.shape[0] or legal_mask[action] <= 0.0:
                continue
            seen.add(action)
            deduped.append(int(action))
        if not deduped:
            fallback = fallback_legal_action_from_mask(legal_mask)
            return [int(fallback)]
        return deduped

    def teacher_candidate_scores(self, rollout_depth: int = 2, top_k: int = 2) -> Dict[int, float]:
        actions = self.teacher_candidate_actions(rollout_depth=rollout_depth, top_k=top_k)
        root_snapshot = self.get_state_snapshot()
        scores = {
            int(action): self._teacher_action_score(int(action), root_snapshot, rollout_depth=rollout_depth)
            for action in actions
        }
        self.set_state_snapshot(root_snapshot)
        return scores

    def hybrid_upfront_teacher_action(self, rollout_depth: int = 2) -> int:
        current_robot_id = self.get_current_robot_id()
        if current_robot_id is None:
            return 0
        legal_mask = self._legal_action_mask(current_robot_id)
        upfront_action = self.upfront_wait_aware_action()
        if not self._is_hard_teacher_state(current_robot_id, legal_mask):
            return upfront_action
        candidate_scores = self.teacher_candidate_scores(rollout_depth=rollout_depth, top_k=2)
        best_action = int(upfront_action)
        best_score = float(candidate_scores.get(int(upfront_action), float("-inf")))
        for action, score in candidate_scores.items():
            if score > best_score + 1e-6:
                best_action = int(action)
                best_score = float(score)
            elif abs(score - best_score) <= 1e-6:
                current_is_wait = int(best_action) == 0
                challenger_is_wait = int(action) == 0
                if current_is_wait and not challenger_is_wait:
                    best_action = int(action)
                elif current_is_wait == challenger_is_wait and int(action) == int(upfront_action):
                    best_action = int(action)
        return int(best_action)

    def _teacher_action_score(self, action: int, root_snapshot: Dict, rollout_depth: int) -> float:
        self.set_state_snapshot(root_snapshot)
        shaping_total = 0.0
        _, _, done, truncated, info = self.step(int(action))
        shaping_total += float(info.get("step_shaping_reward", 0.0))

        for _ in range(max(rollout_depth - 1, 0)):
            if done or truncated:
                break
            greedy_action = self.upfront_wait_aware_action()
            _, _, done, truncated, info = self.step(int(greedy_action))
            shaping_total += float(info.get("step_shaping_reward", 0.0))

        return self._teacher_short_horizon_score(info, done, truncated, shaping_total)

    def step(self, action: int):
        if self.terminated or self.truncated:
            raise RuntimeError("Episode has ended. Call reset() before step().")

        current_slot = self._current_slot()
        if current_slot is None:
            reward = self._auto_advance_until_trainable()
            self.last_info = self._build_info(committed=False)
            return self._get_obs(), reward, self.terminated, self.truncated, self.last_info

        robot_id = self.robot_order[current_slot]
        legal_mask = self._legal_action_mask(robot_id)
        robot_status = self.base_env.robot_states[robot_id].get("status")
        has_legal_task = bool(np.any(legal_mask[1:] > 0.0))
        if int(action) < 0 or int(action) >= len(legal_mask) or legal_mask[int(action)] <= 0.0:
            action = fallback_legal_action_from_mask(legal_mask)

        reward = 0.0
        shaping_reward = 0.0
        dispatch_gap_penalty = 0.0
        self.metrics["decision_count"] += 1
        if int(action) == 0:
            self.metrics["wait_action_count"] += 1
        self._record_wait_pattern_metrics(
            robot_id,
            robot_status=robot_status,
            has_legal_task=has_legal_task,
            action=int(action),
        )

        chosen_task_id = task_id_from_action(int(action), self.task_order)
        if chosen_task_id is not None and self.base_env.task_specs[chosen_task_id]["kind"] == "sync":
            self.metrics["direct_sync_assignments"] += 1
            if self._sync_misassignment_for(robot_id, chosen_task_id):
                self.metrics["direct_sync_misassignments"] += 1
        if chosen_task_id is not None:
            shaping_reward, dispatch_gap_penalty = self._upfront_action_shaping(robot_id, chosen_task_id)
            reward += shaping_reward
            self.metrics["dispatch_gap_penalty_total"] += dispatch_gap_penalty
        self.last_step_shaping_reward = float(shaping_reward)
        self.last_step_dispatch_gap_penalty = float(dispatch_gap_penalty)

        self.pending_actions[current_slot] = int(action)
        self.event_cursor += 1

        if self.event_cursor < len(self.event_slots):
            self.last_info = self._build_info(committed=False)
            self.last_info["step_shaping_reward"] = float(shaping_reward)
            self.last_info["step_dispatch_gap_penalty"] = float(dispatch_gap_penalty)
            return self._get_obs(), reward, False, False, self.last_info

        joint_action = np.zeros(self.base_env.max_robots, dtype=np.int64)
        for slot in self.event_slots:
            joint_action[slot] = self.pending_actions[slot]
        self.pending_actions.fill(0)

        _, base_reward, terminated, truncated, info = self.base_env.step(joint_action)
        reward += float(base_reward)
        self.terminated = terminated
        self.truncated = truncated
        if not (terminated or truncated):
            reward += self._auto_advance_until_trainable()
        self.event_index += 1
        self.last_info = self._build_info(committed=True)
        self.last_info.update(info)
        self.last_info["step_shaping_reward"] = float(shaping_reward)
        self.last_info["step_dispatch_gap_penalty"] = float(dispatch_gap_penalty)
        return self._get_obs(), float(reward), self.terminated, self.truncated, self.last_info
