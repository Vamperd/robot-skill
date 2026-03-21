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
        completion_reward: float = 100.0,
        illegal_action_penalty: float = 5.0,
        idle_penalty: float = 0.03,
        wait_penalty: float = 0.06,
        time_penalty: float = 0.02,
        timeout_penalty: float = 8.0,
        deadlock_penalty: float = 20.0,
        deadlock_limit: int = 8,
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
        self.illegal_action_penalty = illegal_action_penalty
        self.idle_penalty = idle_penalty
        self.wait_penalty = wait_penalty
        self.time_penalty = time_penalty
        self.timeout_penalty = timeout_penalty
        self.deadlock_penalty = deadlock_penalty
        self.deadlock_limit = deadlock_limit

        self.action_space = spaces.MultiDiscrete(np.full(self.max_robots, self.max_tasks + 1, dtype=np.int64))
        self.observation_space = spaces.Dict(
            {
                "robots": spaces.Box(0.0, 1.0, shape=(self.max_robots, 9), dtype=np.float32),
                "tasks": spaces.Box(0.0, 1.0, shape=(self.max_tasks, 9), dtype=np.float32),
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
                "completed_at": None,
            }
            for task in scenario["tasks"]
        }
        self.metrics = {
            "completed_tasks": 0,
            "illegal_actions": 0,
            "timeout_events": 0,
            "deadlock_events": 0,
        }
        self.time = 0.0
        self.terminated = False
        self.truncated = False

        return self._get_obs(), self._build_info()

    def _task_from_action(self, action_value: int) -> Optional[str]:
        if action_value <= 0 or action_value > len(self.task_order):
            return None
        return self.task_order[action_value - 1]


    def _estimate_eta(self, robot_id: str, task_id: str) -> float:
        robot = self.robot_states[robot_id]
        matrix = self.current_scenario["distance_matrix"]
        speed_multiplier = max(0.05, robot["speed_multiplier"])

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
        mask = np.zeros((self.max_robots, self.max_tasks + 1), dtype=np.int8)
        for slot in range(self.max_robots):
            mask[slot, 0] = 1
            if slot >= len(self.robot_order):
                continue
            robot_id = self.robot_order[slot]
            robot = self.robot_states[robot_id]
            if robot["status"] not in {"idle", "waiting_idle"}:
                continue
            for task_index, task_id in enumerate(self.task_order, start=1):
                if self._is_action_legal(robot_id, task_id):
                    mask[slot, task_index] = 1
        return mask


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


    def _advance_by(self, delta: float) -> float:
        reward = -delta * self.time_penalty
        contributors_by_task = {}

        for task_id, task_state in self.task_states.items():
            if task_state["completed"]:
                continue
            contributors = self._contributors_for_task(task_id)
            task_state["contributors"] = set(contributors)
            contributors_by_task[task_id] = contributors

        for robot_id, robot in self.robot_states.items():
            status = robot["status"]
            if status == "travel":
                robot["eta_remaining"] = max(0.0, robot["eta_remaining"] - delta)
                robot["travel_time"] += delta
            elif status == "waiting_idle":
                robot["idle_time"] += delta
                robot["wait_elapsed"] += delta
                reward -= delta * self.idle_penalty
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
            self.task_states[task_id]["progress"] += delta * rate

        self.time += delta
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

            task_state["completed"] = True
            task_state["completed_at"] = self.time
            self.metrics["completed_tasks"] += 1
            reward += self.completion_reward * float(task.get("priority", 1.0))
            completed_tasks.append(task_id)

            assigned_robot_ids = list(task_state["assigned_robot_ids"] | task_state["onsite_robot_ids"])
            for robot_id in assigned_robot_ids:
                self._release_robot(robot_id, task_id)

            task_state["contributors"].clear()
        return completed_tasks, reward


    def _process_arrivals(self) -> list[str]:
        arrivals: list[str] = []
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
        return arrivals


    def _process_wait_timeouts(self) -> tuple[list[str], float]:
        timed_out: list[str] = []
        penalty = 0.0

        for robot_id, robot in self.robot_states.items():
            if robot["status"] == "waiting_idle" and robot["wait_elapsed"] + EPS >= self.wait_timeout:
                timed_out.append(robot_id)
                robot["status"] = "idle"
                robot["wait_elapsed"] = 0.0

        for robot_id, robot in self.robot_states.items():
            if robot["status"] != "onsite" or robot["wait_elapsed"] + EPS < self.wait_timeout:
                continue
            task_id = robot["assigned_task"]
            task_state = self.task_states[task_id]
            if robot_id in task_state["contributors"]:
                continue
            self.metrics["timeout_events"] += 1
            robot["blocked_count"] += 1
            penalty -= self.timeout_penalty
            timed_out.append(robot_id)
            self._release_robot(robot_id, task_id)

        return timed_out, penalty


    def _advance_until_decision(self) -> float:
        reward = 0.0

        while True:
            if self._all_tasks_done():
                self._release_waiting_idle()
                return reward

            candidates: list[float] = []
            for robot in self.robot_states.values():
                if robot["status"] == "travel" and robot["eta_remaining"] > EPS:
                    candidates.append(robot["eta_remaining"])
                elif robot["status"] == "waiting_idle":
                    candidates.append(max(EPS, self.wait_timeout - robot["wait_elapsed"]))

            for task_id, task_state in self.task_states.items():
                if task_state["completed"]:
                    continue
                contributors = self._contributors_for_task(task_id)
                if contributors:
                    current_rate = service_rate(
                        contributors,
                        {
                            robot_id: {"service_multiplier": self.robot_states[robot_id]["service_multiplier"]}
                            for robot_id in contributors
                        },
                    )
                    if current_rate > EPS:
                        remaining = max(0.0, task_state["spec"]["service_time"] - task_state["progress"])
                        candidates.append(max(EPS, remaining / current_rate))

            for robot_id, robot in self.robot_states.items():
                if robot["status"] != "onsite":
                    continue
                task_id = robot["assigned_task"]
                contributors = self._contributors_for_task(task_id)
                if robot_id not in contributors:
                    candidates.append(max(EPS, self.wait_timeout - robot["wait_elapsed"]))

            if not candidates:
                self.metrics["deadlock_events"] += 1
                self._release_waiting_idle()
                return reward - self.deadlock_penalty

            delta = min(candidates)
            reward += self._advance_by(delta)

            completed, completion_reward = self._process_task_completions()
            reward += completion_reward
            self._process_arrivals()
            timed_out, timeout_penalty = self._process_wait_timeouts()
            reward += timeout_penalty

            if completed or timed_out:
                self._release_waiting_idle()
                return reward

            if any(robot["status"] == "idle" for robot in self.robot_states.values()):
                self._release_waiting_idle()
                return reward


    def _build_info(self) -> Dict:
        total_robot_time = max(EPS, self.time * max(1, len(self.robot_order)))
        total_idle = sum(robot["idle_time"] for robot in self.robot_states.values())
        total_wait = sum(robot["wait_time"] for robot in self.robot_states.values())
        return {
            "scenario_id": self.current_scenario.get("scenario_id", "unknown") if self.current_scenario else "unknown",
            "completion_rate": self.metrics["completed_tasks"] / max(1, len(self.task_order)),
            "makespan": self.time,
            "average_wait_time": total_wait / max(1, len(self.robot_order)),
            "average_idle_time": total_idle / max(1, len(self.robot_order)),
            "idle_ratio": total_idle / total_robot_time,
            "deadlock_events": self.metrics["deadlock_events"],
            "timeout_events": self.metrics["timeout_events"],
            "illegal_actions": self.metrics["illegal_actions"],
            "action_mask": self.get_action_mask(),
        }


    def _get_obs(self) -> Dict[str, np.ndarray]:
        robots_obs = np.zeros((self.max_robots, 9), dtype=np.float32)
        tasks_obs = np.zeros((self.max_tasks, 9), dtype=np.float32)
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
                    x / 800.0,
                    y / 600.0,
                    min(robot["speed_multiplier"] / 1.5, 1.0),
                    min(robot["service_multiplier"] / 2.5, 1.0),
                    min(robot["wait_elapsed"] / max(EPS, self.wait_timeout), 1.0),
                ],
                dtype=np.float32,
            )

            for task_slot, task_id in enumerate(self.task_order):
                task_state = self.task_states[task_id]
                if task_state["completed"]:
                    continue
                robot_task_eta[slot, task_slot] = min(self._estimate_eta(robot_id, task_id) / self.max_time, 1.0)

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
                    min(task_state["progress"] / max(EPS, task["service_time"]), 1.0),
                    min(float(task.get("priority", 1.0)) / 2.0, 1.0),
                    min(sum(task.get("required_roles", {}).values()) / 3.0, 1.0),
                    min(precedence_remaining / max(1, len(task.get("precedence", []))), 1.0),
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
            raise RuntimeError("当前回合已结束，请先 reset。")

        action = np.asarray(action, dtype=np.int64).reshape(self.max_robots)
        reward = 0.0
        reserved_single: set[str] = set()

        for slot, robot_id in enumerate(self.robot_order):
            robot = self.robot_states[robot_id]
            if robot["status"] != "idle":
                continue

            task_id = self._task_from_action(int(action[slot]))
            if task_id is None:
                robot["status"] = "waiting_idle"
                robot["wait_elapsed"] = 0.0
                continue

            if not self._is_action_legal(robot_id, task_id, reserved_single):
                reward -= self.illegal_action_penalty
                self.metrics["illegal_actions"] += 1
                robot["status"] = "waiting_idle"
                robot["wait_elapsed"] = 0.0
                continue

            self._assign_robot(robot_id, task_id)
            if self.task_states[task_id]["spec"]["kind"] == "single":
                reserved_single.add(task_id)

        reward += self._advance_until_decision()

        self.terminated = self._all_tasks_done()
        self.truncated = self.time >= self.max_time or self.metrics["deadlock_events"] >= self.deadlock_limit
        info = self._build_info()
        return self._get_obs(), float(reward), self.terminated, self.truncated, info
