from __future__ import annotations

from typing import Dict, Optional, Sequence

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

        class _FallbackMultiBinary:
            def __init__(self, shape):
                self.shape = shape

        class _FallbackDict:
            def __init__(self, spaces_dict):
                self.spaces = spaces_dict

        class _FallbackGym:
            Env = _FallbackEnv

        class _FallbackSpaces:
            Box = _FallbackBox
            Dict = _FallbackDict
            Discrete = _FallbackDiscrete
            MultiBinary = _FallbackMultiBinary

        gym = _FallbackGym()
        spaces = _FallbackSpaces()

from scheduling_env import MAX_ROBOTS, MAX_TASKS, SchedulingEnv
from scheduler_utils import (
    ROBOT_FEATURE_DIM,
    ROLE_DIM,
    TASK_FEATURE_DIM,
    build_scheduler_observation,
    legal_action_mask_for_robot,
    role_aware_greedy_action_for_robot,
    wait_aware_role_greedy_action_for_robot,
    task_id_from_action,
)


EPS = 1e-6


class SequentialSchedulingEnv(gym.Env):
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
        completion_reward: float = 25.0,
        progress_reward_scale: float = 25.0,
        coalition_activation_bonus: float = 5.0,
        illegal_action_penalty: float = 5.0,
        idle_penalty: float = 0.03,
        wait_penalty: float = 0.08,
        time_penalty: float = 0.05,
        timeout_penalty: float = 8.0,
        deadlock_penalty: float = 20.0,
        deadlock_limit: int = 8,
        avoidable_wait_penalty: float = 0.20,
        reassign_progress_bonus_scale: float = 4.0,
        reassign_completion_bonus: float = 2.0,
        bad_reassign_penalty: float = 3.0,
        reassign_margin: float = 20.0,
        reassign_success_window_min: float = 150.0,
        reassign_success_window_factor: float = 2.0,
        reassign_success_window_max: float = 360.0,
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
            illegal_action_penalty=illegal_action_penalty,
            idle_penalty=idle_penalty,
            wait_penalty=wait_penalty,
            time_penalty=time_penalty,
            timeout_penalty=timeout_penalty,
            deadlock_penalty=deadlock_penalty,
            deadlock_limit=deadlock_limit,
            avoidable_wait_penalty=avoidable_wait_penalty,
            reassign_progress_bonus_scale=reassign_progress_bonus_scale,
            reassign_completion_bonus=reassign_completion_bonus,
            bad_reassign_penalty=bad_reassign_penalty,
            reassign_margin=reassign_margin,
            reassign_success_window_min=reassign_success_window_min,
            reassign_success_window_factor=reassign_success_window_factor,
            reassign_success_window_max=reassign_success_window_max,
        )
        self.max_robots = self.base_env.max_robots
        self.max_tasks = self.base_env.max_tasks
        self.action_space = spaces.Discrete(self.max_tasks + 1)
        self.observation_space = spaces.Dict(
            {
                "current_robot": spaces.Box(0.0, 1.0, shape=(self.max_robots,), dtype=np.float32),
                "robots": spaces.Box(0.0, 1.0, shape=(self.max_robots, ROBOT_FEATURE_DIM), dtype=np.float32),
                "tasks": spaces.Box(0.0, 1.0, shape=(self.max_tasks, TASK_FEATURE_DIM), dtype=np.float32),
                "robot_task_eta": spaces.Box(0.0, 1.0, shape=(self.max_robots, self.max_tasks), dtype=np.float32),
                "task_task_eta": spaces.Box(0.0, 1.0, shape=(self.max_tasks, self.max_tasks), dtype=np.float32),
                "current_action_mask": spaces.Box(0.0, 1.0, shape=(self.max_tasks + 1,), dtype=np.float32),
                "pending_assignment_mask": spaces.MultiBinary((self.max_robots, self.max_tasks + 1)),
                "remaining_role_deficit": spaces.Box(0.0, 1.0, shape=(self.max_tasks, ROLE_DIM), dtype=np.float32),
                "precedence_state": spaces.Box(0.0, 1.0, shape=(self.max_tasks,), dtype=np.float32),
            }
        )
        self.pending_actions = np.zeros(self.max_robots, dtype=np.int64)
        self.event_slots: list[int] = []
        self.event_cursor = 0
        self.event_index = 0
        self.terminated = False
        self.truncated = False
        self.last_info: Dict = {}

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

    def _collect_event_slots(self) -> list[int]:
        slots: list[int] = []
        for slot, robot_id in enumerate(self.robot_order):
            robot = self.base_env.robot_states[robot_id]
            if robot.get("status") in {"idle", "waiting_idle", "waiting_sync"}:
                slots.append(slot)
        return slots

    def _normalized_robot_task_eta(self) -> np.ndarray:
        eta = np.zeros((self.max_robots, self.max_tasks), dtype=np.float32)
        matrix = self.base_env.current_scenario["distance_matrix"]

        for robot_slot, robot_id in enumerate(self.robot_order):
            if robot_slot >= self.max_robots:
                break
            robot = self.base_env.robot_states[robot_id]
            speed = max(0.05, float(robot.get("speed_multiplier", 1.0)))
            status = robot.get("status")
            current_task = robot.get("assigned_task")

            for task_slot, task_id in enumerate(self.task_order):
                if task_slot >= self.max_tasks:
                    break
                if status == "travel" and current_task:
                    if task_id == current_task:
                        raw_eta = float(robot.get("eta_remaining", 0.0))
                    else:
                        raw_eta = float(robot.get("eta_remaining", 0.0))
                        raw_eta += float(matrix["task_to_task"][current_task][task_id]["base_eta"]) / speed
                elif robot.get("location_type") == "task" and robot.get("location_id") is not None:
                    raw_eta = float(matrix["task_to_task"][robot["location_id"]][task_id]["base_eta"]) / speed
                else:
                    raw_eta = float(matrix["robot_to_task"][robot_id][task_id]["eta"])
                eta[robot_slot, task_slot] = min(raw_eta / max(self.base_env.max_time, EPS), 1.0)
        return eta

    def _normalized_task_task_eta(self) -> np.ndarray:
        eta = np.zeros((self.max_tasks, self.max_tasks), dtype=np.float32)
        matrix = self.base_env.current_scenario["distance_matrix"]["task_to_task"]
        for src_slot, src_task_id in enumerate(self.task_order):
            if src_slot >= self.max_tasks:
                break
            for dst_slot, dst_task_id in enumerate(self.task_order):
                if dst_slot >= self.max_tasks:
                    break
                raw_eta = float(matrix[src_task_id][dst_task_id]["base_eta"])
                eta[src_slot, dst_slot] = min(raw_eta / max(self.base_env.max_time, EPS), 1.0)
        return eta

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return build_scheduler_observation(
            robot_order=self.robot_order,
            task_order=self.task_order,
            robot_states=self.base_env.robot_states,
            task_states=self.base_env.task_states,
            task_specs=self.base_env.task_specs,
            robot_task_eta=self._normalized_robot_task_eta(),
            task_task_eta=self._normalized_task_task_eta(),
            current_slot=self._current_slot(),
            pending_actions=self.pending_actions,
            max_robots=self.max_robots,
            max_tasks=self.max_tasks,
            max_time=self.base_env.max_time,
            wait_timeout=self.base_env.wait_timeout,
        )

    def _build_info(self, committed: bool) -> Dict:
        info = dict(self.base_env._build_info())
        current_slot = self._current_slot()
        info.update(
            {
                "event_committed": committed,
                "event_index": self.event_index,
                "pending_actions": self.pending_actions.copy(),
                "current_robot_slot": current_slot,
                "current_robot_id": self.robot_order[current_slot] if current_slot is not None else None,
                "current_action_mask": self._get_obs()["current_action_mask"],
            }
        )
        return info

    def reset(self, seed: int | None = None, options: Optional[dict] = None):
        self.base_env.reset(seed=seed, options=options)
        self.pending_actions = np.zeros(self.max_robots, dtype=np.int64)
        self.event_slots = self._collect_event_slots()
        self.event_cursor = 0
        self.event_index = 0
        self.terminated = False
        self.truncated = False
        self.last_info = self._build_info(committed=False)
        return self._get_obs(), self.last_info

    def get_current_robot_id(self) -> Optional[str]:
        slot = self._current_slot()
        if slot is None:
            return None
        return self.robot_order[slot]

    def get_current_action_mask(self) -> np.ndarray:
        robot_id = self.get_current_robot_id()
        if robot_id is None:
            mask = np.zeros(self.max_tasks + 1, dtype=np.float32)
            mask[0] = 1.0
            return mask
        return legal_action_mask_for_robot(
            robot_id=robot_id,
            robot_order=self.robot_order,
            task_order=self.task_order,
            robot_states=self.base_env.robot_states,
            task_states=self.base_env.task_states,
            task_specs=self.base_env.task_specs,
            pending_actions=self.pending_actions,
            max_tasks=self.max_tasks,
        )

    def role_aware_action(self) -> int:
        slot = self._current_slot()
        if slot is None:
            return 0
        robot_id = self.robot_order[slot]
        return role_aware_greedy_action_for_robot(
            robot_id=robot_id,
            robot_order=self.robot_order,
            task_order=self.task_order,
            robot_states=self.base_env.robot_states,
            task_states=self.base_env.task_states,
            task_specs=self.base_env.task_specs,
            robot_task_eta_row=self._normalized_robot_task_eta()[slot],
            pending_actions=self.pending_actions,
            max_tasks=self.max_tasks,
        )

    def wait_aware_action(self) -> int:
        slot = self._current_slot()
        if slot is None:
            return 0
        robot_id = self.robot_order[slot]
        return wait_aware_role_greedy_action_for_robot(
            robot_id=robot_id,
            robot_order=self.robot_order,
            task_order=self.task_order,
            robot_states=self.base_env.robot_states,
            task_states=self.base_env.task_states,
            task_specs=self.base_env.task_specs,
            robot_task_eta_row=self._normalized_robot_task_eta()[slot],
            pending_actions=self.pending_actions,
            max_tasks=self.max_tasks,
        )

    def pending_task_assignments(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for slot, robot_id in enumerate(self.robot_order):
            task_id = task_id_from_action(int(self.pending_actions[slot]), self.task_order)
            if task_id is not None:
                result[robot_id] = task_id
        return result

    def step(self, action: int):
        if self.terminated or self.truncated:
            raise RuntimeError("当前回合已经结束，请先 reset。")

        current_slot = self._current_slot()
        if current_slot is None:
            raise RuntimeError("当前没有可决策的机器人。")

        mask = self.get_current_action_mask()
        reward = 0.0
        action = int(action)

        if action < 0 or action >= self.max_tasks + 1 or mask[action] <= 0.0:
            reward -= self.base_env.illegal_action_penalty
            self.base_env.metrics["illegal_actions"] += 1
            action = 0

        self.pending_actions[current_slot] = action
        self.event_cursor += 1
        committed = False

        if self.event_cursor >= len(self.event_slots):
            _, base_reward, self.terminated, self.truncated, _ = self.base_env.step(self.pending_actions.copy())
            reward += float(base_reward)
            committed = True
            self.event_index += 1
            self.pending_actions[:] = 0
            if not (self.terminated or self.truncated):
                self.event_slots = self._collect_event_slots()
                self.event_cursor = 0
            else:
                self.event_slots = []
                self.event_cursor = 0

        self.last_info = self._build_info(committed=committed)
        return self._get_obs(), float(reward), self.terminated, self.truncated, self.last_info
