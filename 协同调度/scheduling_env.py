from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback for lighter local setups
    try:
        import gym
        from gym import spaces
    except ImportError:  # pragma: no cover - fallback when neither gymnasium nor gym is installed
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

        class _FallbackMultiDiscrete:
            def __init__(self, nvec):
                self.nvec = np.asarray(nvec)

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
            MultiDiscrete = _FallbackMultiDiscrete
            MultiBinary = _FallbackMultiBinary
            Dict = _FallbackDict

        gym = _FallbackGym()
        spaces = _FallbackSpaces()

from scenario_generator import BASE_TRAVEL_SPEED, load_scenarios
from task_runtime import count_roles, role_deficit, select_contributing_robot_ids, service_rate


MAX_ROBOTS = 6
MAX_TASKS = 10
EPS = 1e-6
DISPATCHABLE_STATUSES = {"idle", "waiting_idle", "waiting_sync"}


class SchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: Optional[Sequence[Dict]] = None,
        scenario_dir: str = "offline_maps_v2",
        split: str = "train",
        family: Optional[str] = None,
        max_robots: int = MAX_ROBOTS,
        max_tasks: int = MAX_TASKS,
        max_time: float = 2500.0,
        wait_timeout: float = 60.0,
        completion_reward: float = 12.0,
        progress_reward_scale: float = 8.0,
        coalition_activation_bonus: float = 1.5,
        episode_success_bonus: float = 30.0,
        unfinished_task_penalty: float = 10.0,
        locked_task_penalty: float = 4.0,
        precedence_unlock_bonus: float = 4.0,
        illegal_action_penalty: float = 5.0,
        idle_penalty: float = 0.01,
        wait_penalty: float = 0.02,
        time_penalty: float = 0.02,
        timeout_penalty: float = 8.0,
        deadlock_penalty: float = 20.0,
        deadlock_limit: int = 8,
        avoidable_wait_penalty: float = 0.10,
        reassign_progress_bonus_scale: float = 4.0,
        reassign_completion_bonus: float = 2.0,
        bad_reassign_penalty: float = 1.0,
        reassign_margin: float = 20.0,
        reassign_success_window_min: float = 150.0,
        reassign_success_window_factor: float = 2.0,
        reassign_success_window_max: float = 360.0,
    ):
        super().__init__()
        self.scenarios = list(scenarios) if scenarios is not None else load_scenarios(scenario_dir, split=split, family=family)
        if not self.scenarios:
            raise ValueError("SchedulingEnv 没有可用场景，请先生成 offline_maps_v2 数据集。")

        self.max_robots = max_robots
        self.max_tasks = max_tasks
        self.max_time = max_time
        self.wait_timeout = wait_timeout
        self.completion_reward = completion_reward
        self.progress_reward_scale = progress_reward_scale
        self.coalition_activation_bonus = coalition_activation_bonus
        self.episode_success_bonus = episode_success_bonus
        self.unfinished_task_penalty = unfinished_task_penalty
        self.locked_task_penalty = locked_task_penalty
        self.precedence_unlock_bonus = precedence_unlock_bonus
        self.illegal_action_penalty = illegal_action_penalty
        self.idle_penalty = idle_penalty
        self.wait_penalty = wait_penalty
        self.time_penalty = time_penalty
        self.timeout_penalty = timeout_penalty
        self.deadlock_penalty = deadlock_penalty
        self.deadlock_limit = deadlock_limit
        self.avoidable_wait_penalty = avoidable_wait_penalty
        self.reassign_progress_bonus_scale = reassign_progress_bonus_scale
        self.reassign_completion_bonus = reassign_completion_bonus
        self.bad_reassign_penalty = bad_reassign_penalty
        self.reassign_margin = reassign_margin
        self.reassign_success_window_min = reassign_success_window_min
        self.reassign_success_window_factor = reassign_success_window_factor
        self.reassign_success_window_max = reassign_success_window_max

        self.action_space = spaces.MultiDiscrete(np.full(self.max_robots, self.max_tasks + 1, dtype=np.int64))
        self.observation_space = spaces.Dict(
            {
                "robots": spaces.Box(0.0, 1.0, shape=(self.max_robots, 11), dtype=np.float32),
                "tasks": spaces.Box(0.0, 1.0, shape=(self.max_tasks, 13), dtype=np.float32),
                "robot_task_eta": spaces.Box(0.0, 1.0, shape=(self.max_robots, self.max_tasks), dtype=np.float32),
                "task_task_eta": spaces.Box(0.0, 1.0, shape=(self.max_tasks, self.max_tasks), dtype=np.float32),
                "action_mask": spaces.MultiBinary((self.max_robots, self.max_tasks + 1)),
            }
        )

        self.current_scenario: Optional[Dict] = None
        self.robot_order: List[str] = []
        self.task_order: List[str] = []
        self.robot_specs: Dict[str, Dict] = {}
        self.task_specs: Dict[str, Dict] = {}
        self.robot_states: Dict[str, Dict] = {}
        self.task_states: Dict[str, Dict] = {}
        self.metrics: Dict[str, float] = {}
        self.pending_reassign_evals: Dict[str, Dict] = {}
        self.time = 0.0
        self.terminated = False
        self.truncated = False

    def reset(self, seed: int | None = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        if "scenario" in options:
            scenario = options["scenario"]
        elif "scenario_index" in options:
            scenario = self.scenarios[int(options["scenario_index"]) % len(self.scenarios)]
        else:
            index = int(self.np_random.integers(0, len(self.scenarios)))
            scenario = self.scenarios[index]

        if len(scenario["robots"]) > self.max_robots or len(scenario["tasks"]) > self.max_tasks:
            raise ValueError("场景规模超过 SchedulingEnv 的固定上限。")

        self.current_scenario = scenario
        self.robot_order = [robot["id"] for robot in scenario["robots"]]
        self.task_order = [task["id"] for task in scenario["tasks"]]
        self.robot_specs = {robot["id"]: robot for robot in scenario["robots"]}
        self.task_specs = {task["id"]: task for task in scenario["tasks"]}
        self.robot_states = {
            robot["id"]: {
                "id": robot["id"],
                "role": robot["role"],
                "speed_multiplier": robot["speed_multiplier"],
                "service_multiplier": robot["service_multiplier"],
                "position": tuple(robot["start_pos"]),
                "location_type": "start",
                "location_id": None,
                "status": "idle",
                "assigned_task": None,
                "eta_remaining": 0.0,
                "wait_elapsed": 0.0,
                "idle_time": 0.0,
                "wait_time": 0.0,
                "travel_time": 0.0,
                "busy_time": 0.0,
                "blocked_count": 0,
                "best_alternative_eta": self.max_time,
                "current_task_ready_eta": 0.0,
                "has_legal_alternative": False,
                "has_better_alternative": False,
            }
            for robot in scenario["robots"]
        }
        self.task_states = {
            task["id"]: {
                "spec": task,
                "progress": 0.0,
                "completed": False,
                "assigned_robot_ids": set(),
                "onsite_robot_ids": set(),
                "contributors": set(),
                "waiting_sync_robot_ids": set(),
                "avoidable_waiting_robot_ids": set(),
                "coalition_ready_eta": 0.0,
                "activation_started_at": None,
                "activation_delay": 0.0,
                "completed_at": None,
            }
            for task in scenario["tasks"]
        }
        self.metrics = {
            "completed_tasks": 0,
            "illegal_actions": 0,
            "timeout_events": 0,
            "deadlock_events": 0,
            "avoidable_wait_time": 0.0,
            "waiting_sync_reassign_count": 0,
            "productive_reassign_count": 0,
            "bad_reassign_count": 0,
            "coalition_activation_delay_total": 0.0,
            "coalition_activation_events": 0,
            "successful_episode": 0,
        }
        self.pending_reassign_evals = {}
        self.time = 0.0
        self.terminated = False
        self.truncated = False
        self._refresh_derived_state()
        return self._get_obs(), self._build_info()

    def _task_from_action(self, action_value: int) -> Optional[str]:
        if action_value <= 0 or action_value > len(self.task_order):
            return None
        return self.task_order[action_value - 1]

    def _estimate_eta(self, robot_id: str, task_id: str) -> float:
        robot = self.robot_states[robot_id]
        matrix = self.current_scenario["distance_matrix"]
        speed_multiplier = max(0.05, float(robot["speed_multiplier"]))

        if robot["location_type"] == "task" and robot["location_id"] is not None:
            base_eta = matrix["task_to_task"][robot["location_id"]][task_id]["base_eta"]
            return float(base_eta) / speed_multiplier

        return float(matrix["robot_to_task"][robot_id][task_id]["eta"])

    def _all_tasks_done(self) -> bool:
        return all(state["completed"] for state in self.task_states.values())

    def _completed_task_ids(self) -> set[str]:
        return {task_id for task_id, state in self.task_states.items() if state["completed"]}

    def _precedence_satisfied(self, task_id: str) -> bool:
        task = self.task_specs[task_id]
        done = self._completed_task_ids()
        return all(parent in done for parent in task.get("precedence", []))

    def _task_progress_remaining(self, task_id: str) -> float:
        task_state = self.task_states[task_id]
        task = task_state["spec"]
        return max(0.0, float(task["service_time"]) - float(task_state["progress"]))

    def _contributors_for_task(self, task_id: str) -> List[str]:
        task_state = self.task_states[task_id]
        onsite_ids = sorted(task_state["onsite_robot_ids"])
        snapshots = {
            robot_id: {
                "role": self.robot_states[robot_id]["role"],
                "service_multiplier": self.robot_states[robot_id]["service_multiplier"],
            }
            for robot_id in onsite_ids
        }
        return select_contributing_robot_ids(task_state["spec"], onsite_ids, snapshots)

    def _service_eta_for_task(self, task_id: str) -> float:
        task_state = self.task_states[task_id]
        if task_state["completed"]:
            return 0.0
        contributors = self._contributors_for_task(task_id)
        if not contributors:
            return float("inf")
        rate = service_rate(
            contributors,
            {
                robot_id: {"service_multiplier": self.robot_states[robot_id]["service_multiplier"]}
                for robot_id in contributors
            },
        )
        if rate <= EPS:
            return float("inf")
        return self._task_progress_remaining(task_id) / rate

    def _robot_available_eta_for_task(self, robot_id: str, task_id: str) -> float:
        robot = self.robot_states[robot_id]
        status = robot.get("status")
        current_task = robot.get("assigned_task")
        speed = max(0.05, float(robot.get("speed_multiplier", 1.0)))
        matrix = self.current_scenario["distance_matrix"]

        if status in {"idle", "waiting_idle"}:
            return self._estimate_eta(robot_id, task_id)

        if status == "waiting_sync":
            if current_task == task_id:
                return 0.0
            if robot.get("location_type") == "task" and robot.get("location_id") is not None:
                return float(matrix["task_to_task"][robot["location_id"]][task_id]["base_eta"]) / speed
            return self._estimate_eta(robot_id, task_id)

        if status == "travel":
            if current_task == task_id:
                return float(robot.get("eta_remaining", 0.0))
            travel_left = float(robot.get("eta_remaining", 0.0))
            if current_task is None:
                return travel_left + self._estimate_eta(robot_id, task_id)
            transfer = float(matrix["task_to_task"][current_task][task_id]["base_eta"]) / speed
            return travel_left + transfer

        if status == "onsite":
            if current_task == task_id:
                return 0.0
            if current_task is None:
                return self._estimate_eta(robot_id, task_id)
            finish_eta = self._service_eta_for_task(current_task)
            if not np.isfinite(finish_eta):
                return float("inf")
            transfer = float(matrix["task_to_task"][current_task][task_id]["base_eta"]) / speed
            return finish_eta + transfer

        return self._estimate_eta(robot_id, task_id)

    def _estimate_coalition_ready_eta(self, task_id: str) -> float:
        task_state = self.task_states[task_id]
        task = task_state["spec"]
        if task_state["completed"]:
            return 0.0

        required_roles = task.get("required_roles", {})
        if not required_roles:
            return 0.0

        ready_etas: List[float] = []
        for role, need in required_roles.items():
            role_candidates = [
                self._robot_available_eta_for_task(robot_id, task_id)
                for robot_id, robot in self.robot_states.items()
                if robot["role"] == role
            ]
            role_candidates = sorted(eta for eta in role_candidates if np.isfinite(eta))
            if len(role_candidates) < need:
                return float("inf")
            ready_etas.extend(role_candidates[:need])
        return max(ready_etas, default=0.0)

    def _best_legal_alternative_eta(self, robot_id: str) -> tuple[float, Optional[str]]:
        robot = self.robot_states[robot_id]
        best_eta = float("inf")
        best_task_id: Optional[str] = None

        for task_id in self.task_order:
            if task_id == robot.get("assigned_task"):
                continue
            if not self._is_action_legal(robot_id, task_id):
                continue
            eta = self._robot_available_eta_for_task(robot_id, task_id)
            if eta + EPS < best_eta:
                best_eta = eta
                best_task_id = task_id

        return best_eta, best_task_id

    def _task_unlock_value(self, task_id: str) -> int:
        return sum(task_id in task.get("precedence", []) for task in self.task_specs.values())

    def _refresh_derived_state(self) -> None:
        for task_id, task_state in self.task_states.items():
            task_state["contributors"] = set(self._contributors_for_task(task_id))
            task_state["waiting_sync_robot_ids"].clear()
            task_state["avoidable_waiting_robot_ids"].clear()
            task_state["coalition_ready_eta"] = 0.0

        for robot in self.robot_states.values():
            robot["best_alternative_eta"] = self.max_time
            robot["current_task_ready_eta"] = 0.0
            robot["has_legal_alternative"] = False
            robot["has_better_alternative"] = False

        for task_id, task_state in self.task_states.items():
            task = task_state["spec"]
            contributors = set(task_state["contributors"])
            onsite_ids = set(task_state["onsite_robot_ids"])

            if task_state["completed"]:
                continue

            if task["kind"] == "sync":
                waiting_ids = onsite_ids - contributors
                task_state["waiting_sync_robot_ids"] = set(waiting_ids)

                if waiting_ids and not contributors and task_state["activation_started_at"] is None:
                    task_state["activation_started_at"] = self.time
                if contributors and task_state["activation_started_at"] is not None and task_state["activation_delay"] <= EPS:
                    delay = max(0.0, self.time - float(task_state["activation_started_at"]))
                    task_state["activation_delay"] = delay
                    self.metrics["coalition_activation_delay_total"] += delay
                    self.metrics["coalition_activation_events"] += 1

                task_state["coalition_ready_eta"] = self._estimate_coalition_ready_eta(task_id)

                for robot_id in onsite_ids:
                    robot = self.robot_states[robot_id]
                    robot["status"] = "onsite" if robot_id in contributors else "waiting_sync"
            else:
                task_state["coalition_ready_eta"] = 0.0
                for robot_id in onsite_ids:
                    self.robot_states[robot_id]["status"] = "onsite"

        for robot_id, robot in self.robot_states.items():
            if robot["status"] not in DISPATCHABLE_STATUSES:
                continue

            best_eta, _ = self._best_legal_alternative_eta(robot_id)
            robot["best_alternative_eta"] = best_eta if np.isfinite(best_eta) else self.max_time
            robot["has_legal_alternative"] = bool(np.isfinite(best_eta))

            assigned_task = robot.get("assigned_task")
            if robot["status"] == "waiting_sync" and assigned_task is not None:
                ready_eta = float(self.task_states[assigned_task]["coalition_ready_eta"])
                robot["current_task_ready_eta"] = ready_eta if np.isfinite(ready_eta) else self.max_time
                _, best_task_id = self._best_legal_alternative_eta(robot_id)
                better = False
                if robot["has_legal_alternative"] and best_task_id is not None:
                    alt_task = self.task_specs[best_task_id]
                    current_task_spec = self.task_specs[assigned_task]
                    alt_value = float(alt_task.get("priority", 1.0)) + 0.4 * self._task_unlock_value(best_task_id)
                    current_value = float(current_task_spec.get("priority", 1.0)) + 0.4 * self._task_unlock_value(assigned_task)
                    task_started = bool(self.task_states[assigned_task]["contributors"])
                    better = (
                        not task_started
                        and ready_eta > best_eta + self.reassign_margin
                        and alt_value + 1e-6 >= current_value
                    )
                robot["has_better_alternative"] = bool(better)
                if better:
                    self.task_states[assigned_task]["avoidable_waiting_robot_ids"].add(robot_id)

    def _is_action_legal(self, robot_id: str, task_id: str, reserved_single: Optional[set[str]] = None) -> bool:
        reserved_single = reserved_single or set()
        task_state = self.task_states[task_id]
        task = task_state["spec"]
        robot = self.robot_states[robot_id]

        if task_state["completed"] or not self._precedence_satisfied(task_id):
            return False

        if task["kind"] == "single":
            if task_id in reserved_single or task_state["assigned_robot_ids"] or task_state["onsite_robot_ids"]:
                return False
            required_roles = task.get("required_roles", {})
            if required_roles and robot["role"] not in required_roles:
                return False
            return True

        required_roles = task.get("required_roles", {})
        if not required_roles:
            return True
        assigned_roles = count_roles(task_state["assigned_robot_ids"], self.robot_states)
        deficit = role_deficit(required_roles, assigned_roles)
        return deficit.get(robot["role"], 0) > 0

    def get_action_mask(self) -> np.ndarray:
        self._refresh_derived_state()
        mask = np.zeros((self.max_robots, self.max_tasks + 1), dtype=np.int8)
        for slot in range(self.max_robots):
            mask[slot, 0] = 1
            if slot >= len(self.robot_order):
                continue
            robot_id = self.robot_order[slot]
            robot = self.robot_states[robot_id]
            if robot["status"] not in DISPATCHABLE_STATUSES:
                continue
            for task_index, task_id in enumerate(self.task_order, start=1):
                if self._is_action_legal(robot_id, task_id):
                    mask[slot, task_index] = 1
        return mask

    def _assign_robot(self, robot_id: str, task_id: str) -> None:
        robot = self.robot_states[robot_id]
        task_state = self.task_states[task_id]
        robot["status"] = "travel"
        robot["assigned_task"] = task_id
        robot["eta_remaining"] = self._estimate_eta(robot_id, task_id)
        robot["wait_elapsed"] = 0.0
        task_state["assigned_robot_ids"].add(robot_id)

    def _release_robot(self, robot_id: str, task_id: Optional[str] = None) -> None:
        robot = self.robot_states[robot_id]
        if task_id:
            task_state = self.task_states[task_id]
            task_state["assigned_robot_ids"].discard(robot_id)
            task_state["onsite_robot_ids"].discard(robot_id)
            task_state["contributors"].discard(robot_id)
            task_state["waiting_sync_robot_ids"].discard(robot_id)
            task_state["avoidable_waiting_robot_ids"].discard(robot_id)
            robot["position"] = tuple(task_state["spec"]["pos"])
            robot["location_type"] = "task"
            robot["location_id"] = task_id

        robot["status"] = "idle"
        robot["assigned_task"] = None
        robot["eta_remaining"] = 0.0
        robot["wait_elapsed"] = 0.0

    def _release_waiting_idle(self) -> None:
        for robot in self.robot_states.values():
            if robot["status"] == "waiting_idle":
                robot["status"] = "idle"
                robot["wait_elapsed"] = 0.0

    def _handle_successful_reassigns(self) -> float:
        reward = 0.0
        for robot_id, record in list(self.pending_reassign_evals.items()):
            target_task = record["to_task"]
            task_state = self.task_states[target_task]
            progress_gain = max(0.0, float(task_state["progress"]) - float(record["target_progress_at_reassign"]))
            normalized_gain = progress_gain / max(EPS, float(task_state["spec"]["service_time"]))
            robot_contributed = bool(robot_id in record["contributors_seen"] or robot_id in task_state["contributors"])
            if robot_id in task_state["contributors"]:
                record["contributors_seen"].add(robot_id)
                robot_contributed = True
            if robot_contributed and normalized_gain > 0.05:
                reward += self.reassign_progress_bonus_scale * normalized_gain
                if task_state["completed"] and not record["target_completed_at_reassign"]:
                    reward += self.reassign_completion_bonus
                self.metrics["productive_reassign_count"] += 1
                del self.pending_reassign_evals[robot_id]
                continue
            if self.time + EPS >= record["expires_at"]:
                reward -= self.bad_reassign_penalty
                self.metrics["bad_reassign_count"] += 1
                del self.pending_reassign_evals[robot_id]
        return reward

    def _compute_reassign_success_window(self, eta_to_target: float) -> float:
        if not np.isfinite(eta_to_target):
            return self.reassign_success_window_max
        scaled = float(eta_to_target) * self.reassign_success_window_factor
        return float(
            np.clip(
                scaled,
                self.reassign_success_window_min,
                self.reassign_success_window_max,
            )
        )

    def _advance_by(self, delta: float) -> float:
        reward = -delta * self.time_penalty
        contributors_by_task: Dict[str, List[str]] = {}

        self._refresh_derived_state()
        for task_id, task_state in self.task_states.items():
            if task_state["completed"]:
                continue
            contributors_by_task[task_id] = sorted(task_state["contributors"])

        for robot_id, robot in self.robot_states.items():
            status = robot["status"]
            if status == "travel":
                robot["eta_remaining"] = max(0.0, float(robot["eta_remaining"]) - delta)
                robot["travel_time"] += delta
            elif status == "waiting_idle":
                robot["idle_time"] += delta
                robot["wait_elapsed"] += delta
                reward -= delta * self.idle_penalty
            elif status == "waiting_sync":
                robot["busy_time"] += delta
                robot["wait_time"] += delta
                robot["wait_elapsed"] += delta
                reward -= delta * self.wait_penalty
                if robot.get("has_better_alternative", False):
                    reward -= delta * self.avoidable_wait_penalty
                    self.metrics["avoidable_wait_time"] += delta
            elif status == "onsite":
                robot["busy_time"] += delta
                task_id = robot["assigned_task"]
                contributors = contributors_by_task.get(task_id, [])
                if robot_id not in contributors:
                    robot["wait_time"] += delta
                    robot["wait_elapsed"] += delta
                    reward -= delta * self.wait_penalty
                else:
                    robot["wait_elapsed"] = 0.0

        for task_id, contributors in contributors_by_task.items():
            if not contributors:
                continue
            rate = service_rate(
                contributors,
                {
                    robot_id: {"service_multiplier": self.robot_states[robot_id]["service_multiplier"]}
                    for robot_id in contributors
                },
            )
            task_state = self.task_states[task_id]
            task = task_state["spec"]
            delta_progress = min(self._task_progress_remaining(task_id), delta * rate)
            task_state["progress"] += delta_progress
            reward += (
                self.progress_reward_scale
                * float(task.get("priority", 1.0))
                * (delta_progress / max(EPS, float(task["service_time"])))
            )

        self.time += delta
        reward += self._handle_successful_reassigns()
        return reward

    def _process_task_completions(self) -> tuple[list[str], float]:
        completed_tasks: list[str] = []
        reward = 0.0
        for task_id, task_state in self.task_states.items():
            if task_state["completed"]:
                continue
            task = task_state["spec"]
            if task_state["progress"] + EPS < task["service_time"]:
                continue

            unlocked_count = sum(
                not self.task_states[child_task_id]["completed"] and task_id in self.task_specs[child_task_id].get("precedence", [])
                for child_task_id in self.task_order
            )
            task_state["completed"] = True
            task_state["completed_at"] = self.time
            self.metrics["completed_tasks"] += 1
            reward += self.completion_reward * float(task.get("priority", 1.0))
            if unlocked_count > 0:
                reward += min(8.0, self.precedence_unlock_bonus * float(unlocked_count))
            completed_tasks.append(task_id)

            assigned_robot_ids = list(task_state["assigned_robot_ids"] | task_state["onsite_robot_ids"])
            for robot_id in assigned_robot_ids:
                self._release_robot(robot_id, task_id)

            task_state["contributors"].clear()
            task_state["waiting_sync_robot_ids"].clear()
            task_state["avoidable_waiting_robot_ids"].clear()
        return completed_tasks, reward

    def _final_episode_reward(self) -> float:
        if self._all_tasks_done():
            self.metrics["successful_episode"] = 1
            return self.episode_success_bonus

        remaining_tasks = [task_id for task_id, task_state in self.task_states.items() if not task_state["completed"]]
        if not remaining_tasks:
            return 0.0

        reward = -self.unfinished_task_penalty * float(len(remaining_tasks))
        locked_count = sum(1 for task_id in remaining_tasks if self._task_unlock_value(task_id) > 0)
        reward -= self.locked_task_penalty * float(locked_count)
        return reward

    def _process_arrivals(self) -> tuple[list[str], float]:
        arrivals: list[str] = []
        reward = 0.0
        activation_before = {
            task_id: float(task_state.get("activation_delay", 0.0))
            for task_id, task_state in self.task_states.items()
        }
        for robot_id, robot in self.robot_states.items():
            if robot["status"] != "travel" or robot["eta_remaining"] > EPS:
                continue
            task_id = robot["assigned_task"]
            task_state = self.task_states[task_id]
            if task_state["completed"]:
                self._release_robot(robot_id, task_id)
                arrivals.append(robot_id)
                continue

            robot["status"] = "onsite"
            robot["position"] = tuple(task_state["spec"]["pos"])
            robot["location_type"] = "task"
            robot["location_id"] = task_id
            robot["wait_elapsed"] = 0.0
            task_state["onsite_robot_ids"].add(robot_id)
            arrivals.append(robot_id)

        if arrivals:
            self._refresh_derived_state()
            for task_id, task_state in self.task_states.items():
                if (
                    task_state["spec"]["kind"] == "sync"
                    and activation_before.get(task_id, 0.0) <= EPS
                    and float(task_state.get("activation_delay", 0.0)) > EPS
                ):
                    reward += self.coalition_activation_bonus
        return arrivals, reward

    def _process_wait_timeouts(self) -> tuple[list[str], float]:
        timed_out: list[str] = []
        penalty = 0.0

        for robot_id, robot in self.robot_states.items():
            if robot["status"] == "waiting_idle" and robot["wait_elapsed"] + EPS >= self.wait_timeout:
                timed_out.append(robot_id)
                robot["status"] = "idle"
                robot["wait_elapsed"] = 0.0

        for robot_id, robot in self.robot_states.items():
            if robot["status"] != "waiting_sync" or robot["wait_elapsed"] + EPS < self.wait_timeout:
                continue
            task_id = robot["assigned_task"]
            self.metrics["timeout_events"] += 1
            robot["blocked_count"] += 1
            penalty -= self.timeout_penalty
            timed_out.append(robot_id)
            self._release_robot(robot_id, task_id)
            self.pending_reassign_evals.pop(robot_id, None)

        if timed_out:
            self._refresh_derived_state()
        return timed_out, penalty

    def _advance_until_decision(self) -> float:
        reward = 0.0

        while True:
            if self._all_tasks_done():
                self._release_waiting_idle()
                self._refresh_derived_state()
                return reward

            candidates: list[float] = []
            for robot in self.robot_states.values():
                if robot["status"] == "travel" and robot["eta_remaining"] > EPS:
                    candidates.append(float(robot["eta_remaining"]))
                elif robot["status"] in {"waiting_idle", "waiting_sync"}:
                    candidates.append(max(EPS, self.wait_timeout - float(robot["wait_elapsed"])))

            for task_id, task_state in self.task_states.items():
                if task_state["completed"]:
                    continue
                contributors = sorted(task_state["contributors"])
                if contributors:
                    current_rate = service_rate(
                        contributors,
                        {
                            robot_id: {"service_multiplier": self.robot_states[robot_id]["service_multiplier"]}
                            for robot_id in contributors
                        },
                    )
                    if current_rate > EPS:
                        remaining = self._task_progress_remaining(task_id)
                        candidates.append(max(EPS, remaining / current_rate))

            for record in self.pending_reassign_evals.values():
                candidates.append(max(EPS, float(record["expires_at"]) - self.time))

            if not candidates:
                self.metrics["deadlock_events"] += 1
                self._release_waiting_idle()
                self._refresh_derived_state()
                return reward - self.deadlock_penalty

            delta = min(candidates)
            reward += self._advance_by(delta)

            completed, completion_reward = self._process_task_completions()
            reward += completion_reward
            arrivals, activation_reward = self._process_arrivals()
            reward += activation_reward
            timed_out, timeout_penalty = self._process_wait_timeouts()
            reward += timeout_penalty
            self._refresh_derived_state()

            has_decision_robot = any(robot["status"] in {"idle", "waiting_sync"} for robot in self.robot_states.values())
            if completed or timed_out or (arrivals and has_decision_robot):
                self._release_waiting_idle()
                self._refresh_derived_state()
                return reward

            if any(robot["status"] == "idle" for robot in self.robot_states.values()):
                self._release_waiting_idle()
                self._refresh_derived_state()
                return reward

    def _build_info(self) -> Dict:
        total_robot_time = max(EPS, self.time * max(1, len(self.robot_order)))
        total_idle = sum(float(robot["idle_time"]) for robot in self.robot_states.values())
        total_wait = sum(float(robot["wait_time"]) for robot in self.robot_states.values())
        avg_activation_delay = 0.0
        if self.metrics["coalition_activation_events"] > 0:
            avg_activation_delay = self.metrics["coalition_activation_delay_total"] / self.metrics["coalition_activation_events"]
        productive_rate = 0.0
        if self.metrics["waiting_sync_reassign_count"] > 0:
            productive_rate = self.metrics["productive_reassign_count"] / self.metrics["waiting_sync_reassign_count"]
        return {
            "scenario_id": self.current_scenario.get("scenario_id", "unknown") if self.current_scenario else "unknown",
            "completion_rate": self.metrics["completed_tasks"] / max(1, len(self.task_order)),
            "makespan": self.time,
            "average_wait_time": total_wait / max(1, len(self.robot_order)),
            "average_idle_time": total_idle / max(1, len(self.robot_order)),
            "average_avoidable_wait_time": self.metrics["avoidable_wait_time"] / max(1, len(self.robot_order)),
            "idle_ratio": total_idle / total_robot_time,
            "deadlock_events": self.metrics["deadlock_events"],
            "timeout_events": self.metrics["timeout_events"],
            "illegal_actions": self.metrics["illegal_actions"],
            "waiting_sync_reassign_count": self.metrics["waiting_sync_reassign_count"],
            "productive_reassign_rate": productive_rate,
            "coalition_activation_delay": avg_activation_delay,
            "action_mask": self.get_action_mask(),
        }

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robots_obs = np.zeros((self.max_robots, 11), dtype=np.float32)
        tasks_obs = np.zeros((self.max_tasks, 13), dtype=np.float32)
        robot_task_eta = np.zeros((self.max_robots, self.max_tasks), dtype=np.float32)
        task_task_eta = np.zeros((self.max_tasks, self.max_tasks), dtype=np.float32)

        for slot, robot_id in enumerate(self.robot_order):
            robot = self.robot_states[robot_id]
            x, y = robot["position"]
            robots_obs[slot] = np.array(
                [
                    1.0,
                    float(robot["status"] == "idle"),
                    float(robot["status"] == "travel"),
                    float(robot["status"] == "onsite"),
                    float(robot["status"] == "waiting_idle"),
                    float(robot["status"] == "waiting_sync"),
                    x / 800.0,
                    y / 600.0,
                    min(float(robot["best_alternative_eta"]) / self.max_time, 1.0),
                    min(float(robot["current_task_ready_eta"]) / self.max_time, 1.0),
                    float(robot["has_better_alternative"]),
                ],
                dtype=np.float32,
            )

            for task_slot, task_id in enumerate(self.task_order):
                task_state = self.task_states[task_id]
                if task_state["completed"]:
                    continue
                robot_task_eta[slot, task_slot] = min(self._robot_available_eta_for_task(robot_id, task_id) / self.max_time, 1.0)

        for slot, task_id in enumerate(self.task_order):
            task_state = self.task_states[task_id]
            task = task_state["spec"]
            precedence_remaining = sum(not self.task_states[parent]["completed"] for parent in task.get("precedence", []))
            tasks_obs[slot] = np.array(
                [
                    1.0,
                    float(task_state["completed"]),
                    float(task["kind"] == "sync"),
                    task["pos"][0] / 800.0,
                    task["pos"][1] / 600.0,
                    min(float(task_state["progress"]) / max(EPS, float(task["service_time"])), 1.0),
                    min(float(task.get("priority", 1.0)) / 2.0, 1.0),
                    min(sum(task.get("required_roles", {}).values()) / 3.0, 1.0),
                    min(precedence_remaining / max(1, len(task.get("precedence", []))), 1.0),
                    min(len(task_state["waiting_sync_robot_ids"]) / 3.0, 1.0),
                    min(len(task_state["avoidable_waiting_robot_ids"]) / 3.0, 1.0),
                    min(float(task_state.get("coalition_ready_eta", 0.0)) / self.max_time, 1.0),
                    min(float(task_state.get("activation_delay", 0.0)) / self.max_time, 1.0),
                ],
                dtype=np.float32,
            )
            for other_slot, other_task_id in enumerate(self.task_order):
                task_task_eta[slot, other_slot] = min(
                    self.current_scenario["distance_matrix"]["task_to_task"][task_id][other_task_id]["base_eta"] / self.max_time,
                    1.0,
                )

        return {
            "robots": robots_obs,
            "tasks": tasks_obs,
            "robot_task_eta": robot_task_eta,
            "task_task_eta": task_task_eta,
            "action_mask": self.get_action_mask(),
        }

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("当前回合已经结束，请先 reset。")

        self._refresh_derived_state()
        action = np.asarray(action, dtype=np.int64).reshape(self.max_robots)
        reward = 0.0
        reserved_single: set[str] = set()

        for slot, robot_id in enumerate(self.robot_order):
            robot = self.robot_states[robot_id]
            if robot["status"] not in DISPATCHABLE_STATUSES:
                continue

            task_id = self._task_from_action(int(action[slot]))
            if task_id is None:
                if robot["status"] == "idle":
                    robot["status"] = "waiting_idle"
                    robot["wait_elapsed"] = 0.0
                continue

            if not self._is_action_legal(robot_id, task_id, reserved_single):
                reward -= self.illegal_action_penalty
                self.metrics["illegal_actions"] += 1
                if robot["status"] == "idle":
                    robot["status"] = "waiting_idle"
                    robot["wait_elapsed"] = 0.0
                continue

            if robot["status"] == "waiting_sync":
                old_task_id = robot["assigned_task"]
                success_window = self._compute_reassign_success_window(self._robot_available_eta_for_task(robot_id, task_id))
                self.metrics["waiting_sync_reassign_count"] += 1
                target_task_state = self.task_states[task_id]
                self._release_robot(robot_id, old_task_id)
                self.pending_reassign_evals[robot_id] = {
                    "from_task": old_task_id,
                    "to_task": task_id,
                    "start_time": self.time,
                    "target_progress_at_reassign": float(target_task_state["progress"]),
                    "target_completed_at_reassign": bool(target_task_state["completed"]),
                    "contributors_seen": set(),
                    "expires_at": self.time + success_window,
                }

            self._assign_robot(robot_id, task_id)
            if self.task_states[task_id]["spec"]["kind"] == "single":
                reserved_single.add(task_id)

        self._refresh_derived_state()
        reward += self._advance_until_decision()

        self.terminated = self._all_tasks_done()
        self.truncated = self.time >= self.max_time or self.metrics["deadlock_events"] >= self.deadlock_limit
        if self.terminated or self.truncated:
            reward += self._final_episode_reward()
        info = self._build_info()
        return self._get_obs(), float(reward), self.terminated, self.truncated, info
