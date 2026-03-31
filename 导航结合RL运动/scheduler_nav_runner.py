from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from a_star_planner import AStarPlanner, get_lookahead_waypoint
from low_level_policy_adapter import LowLevelPolicyAdapter


CURRENT_DIR = Path(__file__).resolve().parent
SCHED_DIR = CURRENT_DIR.parent / "协同调度"
if str(SCHED_DIR) not in sys.path:
    sys.path.insert(0, str(SCHED_DIR))

from scheduler_utils import (  # noqa: E402
    build_scheduler_observation,
    legal_action_mask_for_robot,
    role_aware_greedy_action_for_robot,
)
from coop_docking import (  # noqa: E402
    DEFAULT_DOCKING_RADIUS,
    DEFAULT_PLANNER_MARGIN,
    DEFAULT_PLANNER_RESOLUTION,
    DEFAULT_SLOT_CAPTURE_RADIUS,
    generate_docking_slots,
)
from task_runtime import ContinuousTaskRuntime  # noqa: E402


WHITE = (240, 240, 240)
BLACK = (30, 30, 30)
GREEN = (80, 210, 120)
YELLOW = (255, 215, 0)
BLUE = (80, 150, 255)
RED = (255, 100, 100)
GRAY = (150, 150, 150)
ORANGE = (255, 160, 80)
LIGHT_GRAY = (200, 200, 200)
WIDTH = 800
HEIGHT = 600
MOTION_DT = 1.0 / 60.0
SIM_DT = 1.0
DEFAULT_SERVICE_RADIUS = 40.0
BASE_TRAVEL_SPEED = 4.0
EPS = 1e-6
DEFAULT_SYNC_DOCKING_RADIUS = DEFAULT_DOCKING_RADIUS
DEFAULT_SLOT_CAPTURE_RADIUS_PIXELS = DEFAULT_SLOT_CAPTURE_RADIUS
DEFAULT_FINAL_SNAP_RADIUS = 18.0


class SchedulerNavRunner:
    def __init__(
        self,
        scheduler_policy: str | object = "role_aware_greedy",
        low_level_adapter: LowLevelPolicyAdapter | None = None,
        wait_timeout: float = 60.0,
        max_frames: int = 2500,
        service_radius: float = DEFAULT_SERVICE_RADIUS,
        sync_docking_radius: float = DEFAULT_SYNC_DOCKING_RADIUS,
        slot_capture_radius: float = DEFAULT_SLOT_CAPTURE_RADIUS_PIXELS,
        final_snap_radius: float = DEFAULT_FINAL_SNAP_RADIUS,
        render: bool = False,
        gif_path: str | None = None,
    ):
        self.scheduler_policy = scheduler_policy
        self.low_level_adapter = low_level_adapter
        self.wait_timeout = wait_timeout
        self.max_frames = max_frames
        self.service_radius = service_radius
        self.sync_docking_radius = sync_docking_radius
        self.slot_capture_radius = slot_capture_radius
        self.final_snap_radius = final_snap_radius
        self.render = render
        self.gif_path = gif_path
        self.rng = np.random.default_rng(0)
        self.frames = []

        self.scenario: dict | None = None
        self.robot_order: list[str] = []
        self.task_order: list[str] = []
        self.robot_specs: Dict[str, Dict] = {}
        self.task_specs: Dict[str, Dict] = {}
        self.robot_states: Dict[str, Dict] = {}
        self.task_states: Dict[str, Dict] = {}
        self.runtime: ContinuousTaskRuntime | None = None
        self.time = 0.0
        self.frame_index = 0
        self.event_index = 0
        self.metrics: Dict[str, float] = {}
        self.completed_order: list[str] = []

    @classmethod
    def load_scheduler(cls, model_path: str | None, device: torch.device | str = "cpu"):
        if model_path is None:
            return "role_aware_greedy"
        from attention_policy import load_scheduler_checkpoint

        model, _ = load_scheduler_checkpoint(model_path, device=device)
        return model

    def _init_state(self, scenario: Dict) -> None:
        self.scenario = scenario
        self.robot_order = [robot["id"] for robot in scenario["robots"]]
        self.task_order = [task["id"] for task in scenario["tasks"]]
        self.robot_specs = {robot["id"]: robot for robot in scenario["robots"]}
        self.task_specs = {task["id"]: task for task in scenario["tasks"]}
        self.runtime = ContinuousTaskRuntime(tasks=scenario["tasks"], service_radius=self.service_radius)
        self.time = 0.0
        self.frame_index = 0
        self.event_index = 0
        self.frames = []
        self.completed_order = []
        self.metrics = {
            "completed_tasks": 0,
            "timeout_events": 0,
            "deadlock_events": 0,
            "illegal_actions": 0,
        }

        self.robot_states = {}
        for robot in scenario["robots"]:
            self.robot_states[robot["id"]] = {
                "id": robot["id"],
                "role": robot["role"],
                "speed_multiplier": float(robot["speed_multiplier"]),
                "service_multiplier": float(robot["service_multiplier"]),
                "position": tuple(robot["start_pos"]),
                "x": float(robot["start_pos"][0]),
                "y": float(robot["start_pos"][1]),
                "radius": 15.0,
                "status": "idle",
                "assigned_task": None,
                "eta_remaining": 0.0,
                "wait_elapsed": 0.0,
                "idle_time": 0.0,
                "wait_time": 0.0,
                "travel_time": 0.0,
                "busy_time": 0.0,
                "blocked_count": 0,
                "location_type": "start",
                "location_id": None,
                "assigned_slot_index": None,
                "assigned_slot_pos": None,
                "global_path": [],
                "lookahead_wp": tuple(robot["start_pos"]),
                "last_collision": 0.0,
                "position_history": deque(maxlen=60),
                "task_progress": 0.0,
                "is_finished": False,
                "frames_since_replan": 999,
            }
            self.robot_states[robot["id"]]["position_history"].append(tuple(robot["start_pos"]))

        self.task_states = {
            task["id"]: {
                "spec": task,
                "progress": 0.0,
                "completed": False,
                "assigned_robot_ids": set(),
                "onsite_robot_ids": set(),
                "contributors": set(),
                "completed_at": None,
                "slot_positions": self._build_task_slots(task),
                "slot_assignments": {},
            }
            for task in scenario["tasks"]
        }
        for task_id, task_state in self.task_states.items():
            task = task_state["spec"]
            if task.get("kind") != "sync":
                continue
            required_slots = max(1, sum(task.get("required_roles", {}).values()))
            if len(task_state["slot_positions"]) < required_slots:
                raise ValueError(f"协同任务 {task_id} 缺少足够停靠位，无法联调。")
        if self.low_level_adapter is not None:
            self.low_level_adapter.reset(self.robot_order)

    def _build_task_slots(self, task: Dict) -> List[Tuple[float, float]]:
        if task.get("kind") != "sync":
            return []
        slot_count = max(1, sum(task.get("required_roles", {}).values()))
        return generate_docking_slots(
            task_pos=task["pos"],
            slot_count=slot_count,
            obstacles=self.scenario["obstacles"],
            width=WIDTH,
            height=HEIGHT,
            robot_radius=15.0,
            docking_radius=self.sync_docking_radius,
            planner_margin=DEFAULT_PLANNER_MARGIN,
            planner_resolution=DEFAULT_PLANNER_RESOLUTION,
        )

    def _robot_target_pos(self, robot_id: str) -> Tuple[float, float]:
        robot = self.robot_states[robot_id]
        task_id = robot.get("assigned_task")
        if task_id is None:
            return (robot["x"], robot["y"])
        if self.task_specs[task_id].get("kind") == "sync" and robot.get("assigned_slot_pos") is not None:
            return tuple(robot["assigned_slot_pos"])
        return tuple(self.task_specs[task_id]["pos"])

    def _distance_to_target(self, robot_id: str) -> float:
        robot = self.robot_states[robot_id]
        target_x, target_y = self._robot_target_pos(robot_id)
        return math.hypot(robot["x"] - target_x, robot["y"] - target_y)

    def _is_service_ready(self, robot_id: str) -> bool:
        robot = self.robot_states[robot_id]
        task_id = robot.get("assigned_task")
        if task_id is None:
            return False
        if self.task_specs[task_id].get("kind") == "sync":
            return self._distance_to_target(robot_id) <= self.slot_capture_radius
        return self._distance_to_task(robot_id, task_id) <= self.service_radius

    def _assign_slot(self, robot_id: str, task_id: str) -> None:
        task = self.task_specs[task_id]
        robot = self.robot_states[robot_id]
        if task.get("kind") != "sync":
            robot["assigned_slot_index"] = None
            robot["assigned_slot_pos"] = None
            return

        task_state = self.task_states[task_id]
        slot_positions = task_state["slot_positions"]
        if not slot_positions:
            raise ValueError(f"协同任务 {task_id} 没有可用停靠位。")

        current_index = robot.get("assigned_slot_index")
        assignments = task_state["slot_assignments"]
        if current_index is not None and assignments.get(current_index) == robot_id:
            robot["assigned_slot_pos"] = slot_positions[current_index]
            return

        free_indices = [
            slot_index
            for slot_index in range(len(slot_positions))
            if assignments.get(slot_index) in {None, robot_id}
        ]
        if not free_indices:
            raise ValueError(f"协同任务 {task_id} 的停靠位已全部占用。")

        robot_pos = (robot["x"], robot["y"])
        best_index = min(
            free_indices,
            key=lambda slot_index: math.hypot(
                robot_pos[0] - slot_positions[slot_index][0],
                robot_pos[1] - slot_positions[slot_index][1],
            ),
        )
        assignments[best_index] = robot_id
        robot["assigned_slot_index"] = best_index
        robot["assigned_slot_pos"] = slot_positions[best_index]

    def _release_slot(self, robot_id: str, task_id: Optional[str]) -> None:
        robot = self.robot_states[robot_id]
        if task_id is None:
            robot["assigned_slot_index"] = None
            robot["assigned_slot_pos"] = None
            return

        task_state = self.task_states.get(task_id)
        slot_index = robot.get("assigned_slot_index")
        if task_state is not None and slot_index is not None:
            if task_state["slot_assignments"].get(slot_index) == robot_id:
                del task_state["slot_assignments"][slot_index]

        robot["assigned_slot_index"] = None
        robot["assigned_slot_pos"] = None

    def _neighbors(self, robot_id: str) -> list[Dict]:
        neighbors = []
        for other_id, other in self.robot_states.items():
            if other_id == robot_id:
                continue
            if self._should_relax_same_task_collision(robot_id, other_id) or self._should_ignore_neighbor(robot_id, other_id):
                continue
            neighbors.append({"id": other_id, "x": other["x"], "y": other["y"], "radius": other["radius"]})
        return neighbors

    def _distance_to_task(self, robot_id: str, task_id: str) -> float:
        robot = self.robot_states[robot_id]
        task = self.task_specs[task_id]
        return math.hypot(robot["x"] - task["pos"][0], robot["y"] - task["pos"][1])

    def _task_zone_contains(self, task_id: str, x: float, y: float, extra: float = 0.0) -> bool:
        task = self.task_specs[task_id]
        return math.hypot(x - task["pos"][0], y - task["pos"][1]) <= self.service_radius + extra

    def _should_relax_same_task_collision(
        self,
        robot_id: str,
        other_id: str,
        candidate_x: float | None = None,
        candidate_y: float | None = None,
    ) -> bool:
        robot = self.robot_states[robot_id]
        other = self.robot_states[other_id]
        task_id = robot.get("assigned_task")
        if task_id is None or task_id != other.get("assigned_task"):
            return False
        if self.task_specs[task_id].get("kind") != "sync":
            return False

        cx = robot["x"] if candidate_x is None else candidate_x
        cy = robot["y"] if candidate_y is None else candidate_y
        return self._task_zone_contains(task_id, cx, cy, extra=robot["radius"]) and self._task_zone_contains(
            task_id,
            other["x"],
            other["y"],
            extra=other["radius"],
        )

    def _should_ignore_neighbor(self, robot_id: str, other_id: str) -> bool:
        robot = self.robot_states[robot_id]
        other = self.robot_states[other_id]
        task_id = robot.get("assigned_task")
        if task_id is None or task_id != other.get("assigned_task"):
            return False
        if self.task_specs[task_id].get("kind") != "sync":
            return False

        robot_near_task = self._task_zone_contains(task_id, robot["x"], robot["y"], extra=robot["radius"])
        other_near_task = self._task_zone_contains(task_id, other["x"], other["y"], extra=other["radius"])
        return robot_near_task or other_near_task or other.get("status") == "onsite"

    def _use_task_zone_guidance(self, robot_id: str) -> bool:
        robot = self.robot_states[robot_id]
        task_id = robot.get("assigned_task")
        if task_id is None:
            return False
        if self.task_specs[task_id].get("kind") != "sync":
            return False
        return self._task_zone_contains(task_id, robot["x"], robot["y"], extra=robot["radius"])

    def _try_snap_to_target(self, robot_id: str) -> bool:
        robot = self.robot_states[robot_id]
        task_id = robot.get("assigned_task")
        if task_id is None:
            return False

        snap_radius = self.final_snap_radius
        if self.task_specs[task_id].get("kind") == "sync":
            snap_radius = max(self.final_snap_radius, self.slot_capture_radius + 4.0)

        if self._distance_to_target(robot_id) > snap_radius:
            return False

        target_x, target_y = self._robot_target_pos(robot_id)
        robot["x"] = float(target_x)
        robot["y"] = float(target_y)
        robot["position"] = (robot["x"], robot["y"])
        robot["position_history"].append((robot["x"], robot["y"]))
        robot["last_collision"] = 0.0
        robot["lookahead_wp"] = (robot["x"], robot["y"])
        robot["status"] = "onsite"
        robot["eta_remaining"] = 0.0
        self.task_states[task_id]["onsite_robot_ids"].add(robot_id)
        return True

    def _path_remaining_length(self, robot: Dict) -> float:
        path = robot.get("global_path") or []
        if not path:
            return 0.0
        points = [(robot["x"], robot["y"])] + list(path)
        return sum(
            math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
            for index in range(1, len(points))
        )

    def _estimate_eta(self, robot_id: str, task_id: str) -> float:
        matrix = self.scenario["distance_matrix"]
        robot = self.robot_states[robot_id]
        speed_multiplier = max(0.05, robot["speed_multiplier"])

        if robot["status"] == "travel" and robot["assigned_task"]:
            if task_id == robot["assigned_task"]:
                return float(robot["eta_remaining"])
            return float(robot["eta_remaining"]) + float(matrix["task_to_task"][robot["assigned_task"]][task_id]["base_eta"]) / speed_multiplier

        if robot["location_type"] == "task" and robot["location_id"] is not None:
            return float(matrix["task_to_task"][robot["location_id"]][task_id]["base_eta"]) / speed_multiplier

        return float(matrix["robot_to_task"][robot_id][task_id]["eta"])

    def _robot_task_eta_matrix(self) -> np.ndarray:
        eta = np.zeros((len(self.robot_order), len(self.task_order)), dtype=np.float32)
        for robot_slot, robot_id in enumerate(self.robot_order):
            for task_slot, task_id in enumerate(self.task_order):
                eta[robot_slot, task_slot] = min(self._estimate_eta(robot_id, task_id) / max(self.max_frames, 1.0), 1.0)
        return eta

    def _task_task_eta_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.task_order), len(self.task_order)), dtype=np.float32)
        for src_slot, src_task in enumerate(self.task_order):
            for dst_slot, dst_task in enumerate(self.task_order):
                base_eta = self.scenario["distance_matrix"]["task_to_task"][src_task][dst_task]["base_eta"]
                matrix[src_slot, dst_slot] = min(float(base_eta) / max(self.max_frames, 1.0), 1.0)
        return matrix

    def _build_scheduler_obs(self, current_slot: int, pending_actions: np.ndarray) -> Dict[str, np.ndarray]:
        return build_scheduler_observation(
            robot_order=self.robot_order,
            task_order=self.task_order,
            robot_states=self.robot_states,
            task_states=self.task_states,
            task_specs=self.task_specs,
            robot_task_eta=self._robot_task_eta_matrix(),
            task_task_eta=self._task_task_eta_matrix(),
            current_slot=current_slot,
            pending_actions=pending_actions,
            max_robots=len(self.robot_order),
            max_tasks=len(self.task_order),
            max_time=float(self.max_frames),
            wait_timeout=self.wait_timeout,
        )

    def _scheduler_action(self, obs: Dict[str, np.ndarray], current_slot: int, pending_actions: np.ndarray) -> int:
        robot_id = self.robot_order[current_slot]
        if isinstance(self.scheduler_policy, str):
            if self.scheduler_policy == "random":
                valid = np.flatnonzero(obs["current_action_mask"] > 0.0)
                return int(self.rng.choice(valid)) if len(valid) else 0
            return role_aware_greedy_action_for_robot(
                robot_id=robot_id,
                robot_order=self.robot_order,
                task_order=self.task_order,
                robot_states=self.robot_states,
                task_states=self.task_states,
                task_specs=self.task_specs,
                robot_task_eta_row=self._robot_task_eta_matrix()[current_slot],
                pending_actions=pending_actions,
                max_tasks=len(self.task_order),
            )

        from attention_policy import obs_to_torch

        obs_tensors = obs_to_torch(obs, device="cpu")
        self.scheduler_policy.eval()
        import torch

        with torch.no_grad():
            action, _, _ = self.scheduler_policy.act(obs_tensors, deterministic=True)
        return int(action.item())

    def _assign_robot(self, robot_id: str, task_id: str) -> None:
        robot = self.robot_states[robot_id]
        robot["status"] = "travel"
        robot["assigned_task"] = task_id
        self._assign_slot(robot_id, task_id)
        robot["wait_elapsed"] = 0.0
        robot["frames_since_replan"] = 999
        robot["lookahead_wp"] = self._robot_target_pos(robot_id)
        self.task_states[task_id]["assigned_robot_ids"].add(robot_id)

        if self._is_service_ready(robot_id):
            robot["status"] = "onsite"
            self.task_states[task_id]["onsite_robot_ids"].add(robot_id)
        else:
            self.task_states[task_id]["onsite_robot_ids"].discard(robot_id)

        robot["eta_remaining"] = self._estimate_eta(robot_id, task_id)

    def _release_robot(self, robot_id: str, task_id: Optional[str] = None) -> None:
        robot = self.robot_states[robot_id]
        if task_id:
            self.task_states[task_id]["assigned_robot_ids"].discard(robot_id)
            self.task_states[task_id]["onsite_robot_ids"].discard(robot_id)
            self.task_states[task_id]["contributors"].discard(robot_id)
            robot["location_type"] = "task"
            robot["location_id"] = task_id
            self._release_slot(robot_id, task_id)
        else:
            self._release_slot(robot_id, None)

        robot["status"] = "idle"
        robot["assigned_task"] = None
        robot["eta_remaining"] = 0.0
        robot["wait_elapsed"] = 0.0
        robot["global_path"] = []
        robot["lookahead_wp"] = (robot["x"], robot["y"])
        robot["task_progress"] = 0.0

    def _dispatch_idle_robots(self) -> None:
        idle_slots = [
            slot
            for slot, robot_id in enumerate(self.robot_order)
            if self.robot_states[robot_id]["status"] in {"idle", "waiting_idle"}
        ]
        if not idle_slots:
            return

        pending_actions = np.zeros(len(self.robot_order), dtype=np.int64)
        for slot in idle_slots:
            obs = self._build_scheduler_obs(slot, pending_actions)
            action = self._scheduler_action(obs, slot, pending_actions)
            if action < 0 or action > len(self.task_order):
                action = 0
            if obs["current_action_mask"][action] <= 0.0:
                self.metrics["illegal_actions"] += 1
                action = 0
            pending_actions[slot] = action

        for slot in idle_slots:
            robot_id = self.robot_order[slot]
            task_id = self.task_order[pending_actions[slot] - 1] if pending_actions[slot] > 0 else None
            if task_id is None:
                self.robot_states[robot_id]["status"] = "waiting_idle"
                self.robot_states[robot_id]["wait_elapsed"] = 0.0
                continue
            self._assign_robot(robot_id, task_id)

        self.event_index += 1

    def _check_collision(self, x: float, y: float, robot_id: str) -> bool:
        for ox, oy, ow, oh in self.scenario["obstacles"]:
            closest_x = max(ox, min(x, ox + ow))
            closest_y = max(oy, min(y, oy + oh))
            if math.hypot(x - closest_x, y - closest_y) < self.robot_states[robot_id]["radius"]:
                return True
        for other_id, other in self.robot_states.items():
            if other_id == robot_id:
                continue
            if self._should_relax_same_task_collision(robot_id, other_id, candidate_x=x, candidate_y=y):
                continue
            if math.hypot(x - other["x"], y - other["y"]) < self.robot_states[robot_id]["radius"] + other["radius"]:
                return True
        return False

    def _apply_physics(self, robot_id: str, action: np.ndarray) -> None:
        robot = self.robot_states[robot_id]
        alpha = 0.4
        last_action = np.asarray(robot.get("last_action", np.zeros(2, dtype=np.float32)), dtype=np.float32)
        smoothed = alpha * np.asarray(action, dtype=np.float32) + (1.0 - alpha) * last_action
        robot["last_action"] = smoothed

        if np.linalg.norm(smoothed) < 0.15:
            smoothed = np.zeros(2, dtype=np.float32)
            robot["last_action"] = smoothed

        vx = float(smoothed[0]) * 250.0
        vy = float(smoothed[1]) * 250.0
        old_x, old_y = robot["x"], robot["y"]
        robot["last_collision"] = 0.0

        next_x = old_x + vx * MOTION_DT
        if next_x < robot["radius"] or next_x > WIDTH - robot["radius"] or self._check_collision(next_x, old_y, robot_id):
            robot["last_collision"] = 1.0
            next_x = old_x

        next_y = old_y + vy * MOTION_DT
        if next_y < robot["radius"] or next_y > HEIGHT - robot["radius"] or self._check_collision(next_x, next_y, robot_id):
            robot["last_collision"] = 1.0
            next_y = old_y

        robot["x"] = next_x
        robot["y"] = next_y
        robot["position"] = (next_x, next_y)
        robot["position_history"].append((next_x, next_y))

    def _step_motion(self) -> None:
        obstacle_rects = [tuple(rect) for rect in self.scenario["obstacles"]]
        for robot_id, robot in self.robot_states.items():
            status = robot["status"]
            if status == "travel" and robot["assigned_task"]:
                if self._try_snap_to_target(robot_id):
                    continue
                target_pos = self._robot_target_pos(robot_id)
                if robot["frames_since_replan"] >= 5:
                    planner = AStarPlanner(width=WIDTH, height=HEIGHT, resolution=10, robot_radius=int(robot["radius"]), margin=5)
                    robot["global_path"] = planner.plan((robot["x"], robot["y"]), target_pos, obstacle_rects)
                    robot["frames_since_replan"] = 0
                else:
                    robot["frames_since_replan"] += 1
                lookahead = 40.0 if robot["last_collision"] else 60.0
                robot["lookahead_wp"] = get_lookahead_waypoint((robot["x"], robot["y"]), robot["global_path"], lookahead_dist=lookahead)
                if self._use_task_zone_guidance(robot_id):
                    dx = target_pos[0] - robot["x"]
                    dy = target_pos[1] - robot["y"]
                    norm = max(math.hypot(dx, dy), EPS)
                    action = np.asarray([dx / norm, dy / norm], dtype=np.float32)
                elif self.low_level_adapter is None:
                    dx = robot["lookahead_wp"][0] - robot["x"]
                    dy = robot["lookahead_wp"][1] - robot["y"]
                    norm = max(math.hypot(dx, dy), EPS)
                    action = np.asarray([dx / norm, dy / norm], dtype=np.float32)
                else:
                    action = self.low_level_adapter.predict_action(
                        robot_state=robot,
                        waypoint=robot["lookahead_wp"],
                        obstacles=obstacle_rects,
                        neighbors=self._neighbors(robot_id),
                    )
                self._apply_physics(robot_id, action)
                robot["travel_time"] += SIM_DT
                if self._try_snap_to_target(robot_id):
                    continue
                remaining_length = self._path_remaining_length(robot)
                robot["eta_remaining"] = remaining_length / max(BASE_TRAVEL_SPEED * robot["speed_multiplier"], EPS)
                if self._is_service_ready(robot_id):
                    robot["status"] = "onsite"
                    robot["eta_remaining"] = 0.0
                    self.task_states[robot["assigned_task"]]["onsite_robot_ids"].add(robot_id)
            elif status == "onsite":
                robot["busy_time"] += SIM_DT
                robot["eta_remaining"] = 0.0
            elif status == "waiting_idle":
                robot["idle_time"] += SIM_DT
                robot["wait_elapsed"] += SIM_DT

    def _sync_runtime(self) -> tuple[set[str], set[str]]:
        snapshots = []
        for robot_id, robot in self.robot_states.items():
            assigned_task = robot["assigned_task"]
            distance_to_task = float("inf")
            service_ready = False
            if assigned_task is not None:
                distance_to_task = self._distance_to_task(robot_id, assigned_task)
                service_ready = self._is_service_ready(robot_id)
                if service_ready:
                    self.task_states[assigned_task]["onsite_robot_ids"].add(robot_id)
                else:
                    self.task_states[assigned_task]["onsite_robot_ids"].discard(robot_id)

            snapshots.append(
                {
                    "id": robot_id,
                    "role": robot["role"],
                    "service_multiplier": robot["service_multiplier"],
                    "assigned_task": assigned_task,
                    "distance_to_task": distance_to_task,
                    "service_ready": service_ready,
                }
            )

        event = self.runtime.update(snapshots, dt=SIM_DT)
        completed_tasks = set(event.completed_tasks)
        released_robots = set()

        for task_id, runtime_state in self.runtime.task_states.items():
            self.task_states[task_id]["progress"] = runtime_state["progress"]
            self.task_states[task_id]["completed"] = runtime_state["completed"]
            self.task_states[task_id]["contributors"] = set(runtime_state["contributors"])
            if runtime_state["completed"] and self.task_states[task_id]["completed_at"] is None:
                self.task_states[task_id]["completed_at"] = self.time
                self.metrics["completed_tasks"] += 1
                self.completed_order.append(task_id)

        for robot_id, is_waiting in event.robot_waiting.items():
            robot = self.robot_states[robot_id]
            robot["task_progress"] = event.robot_task_progress.get(robot_id, 0.0)
            if robot["assigned_task"] is None:
                continue
            if is_waiting:
                robot["wait_time"] += SIM_DT
                robot["wait_elapsed"] += SIM_DT
                if robot["wait_elapsed"] >= self.wait_timeout and robot_id not in self.task_states[robot["assigned_task"]]["contributors"]:
                    task_id = robot["assigned_task"]
                    self.metrics["timeout_events"] += 1
                    robot["blocked_count"] += 1
                    released_robots.add(robot_id)
                    self._release_robot(robot_id, task_id)
            else:
                robot["wait_elapsed"] = 0.0

        for task_id in completed_tasks:
            assigned = list(self.task_states[task_id]["assigned_robot_ids"] | self.task_states[task_id]["onsite_robot_ids"])
            for robot_id in assigned:
                released_robots.add(robot_id)
                self._release_robot(robot_id, task_id)

        return completed_tasks, released_robots

    def _waiting_idle_timeouts(self) -> set[str]:
        released = set()
        for robot_id, robot in self.robot_states.items():
            if robot["status"] == "waiting_idle" and robot["wait_elapsed"] >= self.wait_timeout:
                robot["status"] = "idle"
                robot["wait_elapsed"] = 0.0
                released.add(robot_id)
        return released

    def _all_tasks_done(self) -> bool:
        return all(state["completed"] for state in self.task_states.values())

    def _idle_ratio(self) -> float:
        total_time = max(self.time * max(len(self.robot_order), 1), 1.0)
        return sum(robot["idle_time"] for robot in self.robot_states.values()) / total_time

    def _info(self, success: bool, truncated: bool) -> Dict:
        return {
            "scenario_id": self.scenario["scenario_id"],
            "success": success,
            "truncated": truncated,
            "completion_rate": self.metrics["completed_tasks"] / max(len(self.task_order), 1),
            "makespan": self.time,
            "average_wait_time": sum(robot["wait_time"] for robot in self.robot_states.values()) / max(len(self.robot_order), 1),
            "average_idle_time": sum(robot["idle_time"] for robot in self.robot_states.values()) / max(len(self.robot_order), 1),
            "idle_ratio": self._idle_ratio(),
            "deadlock_events": self.metrics["deadlock_events"],
            "timeout_events": self.metrics["timeout_events"],
            "illegal_actions": self.metrics["illegal_actions"],
            "completed_order": list(self.completed_order),
            "event_index": self.event_index,
            "frame_index": self.frame_index,
        }

    def _draw_scene(self, screen, font, small_font) -> None:
        screen.fill(WHITE)
        for rect in self.scenario["obstacles"]:
            pygame.draw.rect(screen, BLACK, rect)

        for task in self.scenario["tasks"]:
            task_id = task["id"]
            state = self.task_states[task_id]
            if state["completed"]:
                color = GREEN
            elif not all(self.task_states[parent]["completed"] for parent in task.get("precedence", [])):
                color = GRAY
            elif state["contributors"]:
                color = BLUE
            elif state["assigned_robot_ids"] or state["onsite_robot_ids"]:
                color = ORANGE
            else:
                color = YELLOW

            px, py = int(task["pos"][0]), int(task["pos"][1])
            if task.get("kind") == "sync":
                pygame.draw.circle(screen, LIGHT_GRAY, (px, py), int(self.service_radius), 1)
            pygame.draw.circle(screen, color, (px, py), 18)
            pygame.draw.circle(screen, BLACK, (px, py), 18, 2)
            label = font.render(task_id.split()[-1], True, (255, 255, 255))
            screen.blit(label, label.get_rect(center=(px, py)))

            progress = state["progress"] / max(float(task["service_time"]), 1.0)
            pygame.draw.rect(screen, BLACK, (px - 22, py + 22, 44, 6), 1)
            pygame.draw.rect(screen, GREEN, (px - 21, py + 23, int(42 * min(progress, 1.0)), 4))

            for parent in task.get("precedence", []):
                parent_task = self.task_specs[parent]
                pygame.draw.line(screen, (90, 90, 90), parent_task["pos"], task["pos"], 2)

            for slot_index, slot_pos in enumerate(state.get("slot_positions", [])):
                assigned_robot = state.get("slot_assignments", {}).get(slot_index)
                if assigned_robot is None:
                    slot_color = LIGHT_GRAY
                elif assigned_robot in state.get("onsite_robot_ids", set()):
                    slot_color = BLUE
                else:
                    slot_color = ORANGE
                slot_center = (int(slot_pos[0]), int(slot_pos[1]))
                pygame.draw.circle(screen, slot_color, slot_center, 7)
                pygame.draw.circle(screen, BLACK, slot_center, 7, 1)

        for robot_id in self.robot_order:
            robot = self.robot_states[robot_id]
            color = tuple(self.robot_specs[robot_id]["color"])
            pos = (int(robot["x"]), int(robot["y"]))
            pygame.draw.circle(screen, color, pos, int(robot["radius"]))
            if robot["status"] == "waiting_idle":
                pygame.draw.circle(screen, YELLOW, pos, int(robot["radius"]) + 5, 3)
            elif robot["last_collision"] > 0.0:
                pygame.draw.circle(screen, RED, pos, int(robot["radius"]) + 4, 3)
            else:
                pygame.draw.circle(screen, BLACK, pos, int(robot["radius"]), 2)

            if robot.get("global_path"):
                points = [(int(p[0]), int(p[1])) for p in robot["global_path"]]
                if len(points) > 1:
                    pygame.draw.lines(screen, color, False, points, 2)
            if robot.get("lookahead_wp"):
                pygame.draw.circle(screen, color, (int(robot["lookahead_wp"][0]), int(robot["lookahead_wp"][1])), 6)
            if robot.get("assigned_slot_pos") is not None and robot.get("assigned_task") is not None:
                slot_pos = robot["assigned_slot_pos"]
                pygame.draw.line(screen, color, pos, (int(slot_pos[0]), int(slot_pos[1])), 1)

            status = robot["status"]
            target = robot.get("assigned_task")
            info = f"{robot_id} {status}"
            if target:
                info += f"->{target.split()[-1]}"
            if robot.get("assigned_slot_index") is not None:
                info += f" s{robot['assigned_slot_index']}"
            if robot["wait_elapsed"] > 0.0:
                info += f" w={robot['wait_elapsed']:.0f}"
            text = small_font.render(info, True, BLACK)
            screen.blit(text, (pos[0] + 18, pos[1] - 10))

        title = font.render(
            f"{self.scenario['scenario_id']} event={self.event_index} frame={self.frame_index} done={self.metrics['completed_tasks']}/{len(self.task_order)}",
            True,
            BLACK,
        )
        screen.blit(title, (12, 12))
        line2 = font.render(
            f"time={self.time:.1f} idle_ratio={self._idle_ratio():.3f} wait={sum(r['wait_time'] for r in self.robot_states.values()) / max(len(self.robot_order), 1):.1f}",
            True,
            BLACK,
        )
        screen.blit(line2, (12, 36))

    def _advance_until_event(self, screen=None, font=None, small_font=None) -> tuple[bool, bool]:
        frames_since_progress = 0
        while True:
            if self._all_tasks_done():
                return True, False
            if self.frame_index >= self.max_frames:
                return False, True

            self.frame_index += 1
            self.time += SIM_DT
            self._step_motion()
            completed_tasks, released_robots = self._sync_runtime()
            released_robots |= self._waiting_idle_timeouts()

            if completed_tasks or released_robots:
                return False, False

            if any(robot["status"] == "idle" for robot in self.robot_states.values()):
                return False, False

            frames_since_progress += 1
            if frames_since_progress >= 600:
                self.metrics["deadlock_events"] += 1
                return False, True

            if self.render and screen is not None:
                self._draw_scene(screen, font, small_font)
                pygame.display.flip()
                if self.gif_path:
                    frame = pygame.surfarray.array3d(screen)
                    self.frames.append(np.transpose(frame, (1, 0, 2)))

    def run_episode(self, scenario: Dict, render: bool = True, gif_path: str | None = None) -> Dict:
        self.render = render
        self.gif_path = gif_path
        self._init_state(scenario)

        screen = None
        font = None
        small_font = None
        clock = None
        if render:
            pygame.init()
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Scheduled Multi-Robot Navigation")
            font = pygame.font.SysFont("arial", 18, bold=True)
            small_font = pygame.font.SysFont("arial", 13)
            clock = pygame.time.Clock()

        running = True
        success = False
        truncated = False
        while running:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        truncated = True

            self._dispatch_idle_robots()
            success, truncated = self._advance_until_event(screen=screen, font=font, small_font=small_font)

            if render and screen is not None:
                self._draw_scene(screen, font, small_font)
                pygame.display.flip()
                if self.gif_path:
                    frame = pygame.surfarray.array3d(screen)
                    self.frames.append(np.transpose(frame, (1, 0, 2)))
                clock.tick(30)

            if success or truncated:
                running = False

        if render:
            pygame.quit()
        if self.gif_path and self.frames:
            import imageio

            imageio.mimsave(self.gif_path, self.frames, duration=1 / 20)
        return self._info(success=success, truncated=truncated)
