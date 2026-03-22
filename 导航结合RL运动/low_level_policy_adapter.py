from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np


class FrameStacker:
    def __init__(self, frame_dim: int, n_stack: int):
        self.frame_dim = frame_dim
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(frame)
        return self.get()

    def append(self, frame: np.ndarray) -> np.ndarray:
        if not self.frames:
            return self.reset(frame)
        self.frames.append(frame)
        return self.get()

    def get(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(self.frame_dim * self.n_stack, dtype=np.float32)
        return np.concatenate(list(self.frames), axis=0)


class LowLevelPolicyAdapter:
    def __init__(self, model, base_mode: str, n_stack: int):
        self.model = model
        self.base_mode = base_mode
        self.n_stack = n_stack
        self.frame_dim = 22 if base_mode == "local22" else 27
        self.max_lidar_range = 150.0
        self.front_wall_threshold = 0.25
        self.stackers: Dict[str, FrameStacker] = {}

    @classmethod
    def from_model(cls, model_path: str | Path) -> "LowLevelPolicyAdapter":
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "LowLevelPolicyAdapter 需要 stable-baselines3。请先执行: "
                "pip install stable-baselines3 gymnasium"
            ) from exc

        model = PPO.load(str(model_path), device="cpu")
        obs_shape = getattr(model.observation_space, "shape", None)
        if obs_shape is None:
            raise ValueError("底层模型缺少 observation_space.shape，无法推断观测模式。")
        obs_dim = int(obs_shape[0])

        matches = []
        for base_dim, mode in [(22, "local22"), (27, "relative27")]:
            if obs_dim % base_dim == 0:
                matches.append((mode, obs_dim // base_dim))
        if len(matches) != 1:
            raise ValueError(f"无法根据底层模型输入维度 {obs_dim} 唯一确定接口类型。")
        base_mode, n_stack = matches[0]
        return cls(model=model, base_mode=base_mode, n_stack=int(n_stack))

    def reset(self, robot_ids: Iterable[str] | None = None) -> None:
        if robot_ids is None:
            self.stackers.clear()
            return
        for robot_id in robot_ids:
            self.stackers.pop(robot_id, None)

    def _get_stacker(self, robot_id: str) -> FrameStacker:
        if robot_id not in self.stackers:
            self.stackers[robot_id] = FrameStacker(frame_dim=self.frame_dim, n_stack=self.n_stack)
        return self.stackers[robot_id]

    def _is_stagnant(self, position_history: Sequence[tuple[float, float]]) -> bool:
        if len(position_history) < 60:
            return False
        positions = np.asarray(position_history, dtype=np.float32)
        std_sum = float(np.std(positions[:, 0]) + np.std(positions[:, 1]))
        if std_sum < 5.0:
            return True
        displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        return displacement < 30.0

    def _front_sector_blocked(self, lidar_distances: Sequence[float], vx: float, vy: float) -> bool:
        if abs(vx) <= 1e-6 and abs(vy) <= 1e-6:
            return False
        move_angle = math.atan2(vy, vx)
        num_rays = len(lidar_distances)
        angle_step = 2 * math.pi / num_rays
        center_idx = int(round(move_angle / angle_step)) % num_rays
        front_sector = [lidar_distances[(center_idx + offset) % num_rays] for offset in (-1, 0, 1)]
        return min(front_sector) < self.front_wall_threshold

    def _get_lidar(self, robot_state: Dict, obstacles: Sequence[tuple[float, float, float, float]], neighbors: Sequence[Dict]) -> list[float]:
        num_rays = 16
        lidar_distances = []
        rx = float(robot_state["x"])
        ry = float(robot_state["y"])
        robot_radius = float(robot_state.get("radius", 15.0))

        for ray_index in range(num_rays):
            angle = ray_index * (2 * math.pi / num_rays)
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
            distance = self.max_lidar_range

            for step in range(1, int(self.max_lidar_range), 5):
                test_x = rx + ray_dx * step
                test_y = ry + ray_dy * step
                if test_x < 0 or test_x > 800 or test_y < 0 or test_y > 600:
                    distance = step
                    break

                hit = False
                for ox, oy, ow, oh in obstacles:
                    closest_x = max(ox, min(test_x, ox + ow))
                    closest_y = max(oy, min(test_y, oy + oh))
                    if math.hypot(test_x - closest_x, test_y - closest_y) < robot_radius:
                        hit = True
                        break
                if hit:
                    distance = step
                    break

                for neighbor in neighbors:
                    if neighbor["id"] == robot_state["id"]:
                        continue
                    if math.hypot(test_x - neighbor["x"], test_y - neighbor["y"]) < neighbor.get("radius", 15.0):
                        hit = True
                        break
                if hit:
                    distance = step
                    break

            lidar_distances.append(distance / self.max_lidar_range)

        return lidar_distances

    def _build_frame(
        self,
        robot_state: Dict,
        waypoint: tuple[float, float] | None,
        obstacles: Sequence[tuple[float, float, float, float]],
        neighbors: Sequence[Dict],
    ) -> np.ndarray:
        if waypoint is None:
            if self.base_mode == "local22":
                return np.zeros(22, dtype=np.float32)
            return np.zeros(27, dtype=np.float32)

        rx = float(robot_state["x"])
        ry = float(robot_state["y"])
        tx, ty = waypoint
        dx = tx - rx
        dy = ty - ry
        distance = math.hypot(dx, dy)
        if distance > 1e-6:
            dir_x = dx / distance
            dir_y = dy / distance
        else:
            dir_x = 0.0
            dir_y = 0.0
        distance_norm = min(distance / self.max_lidar_range, 1.0)

        lidar = self._get_lidar(robot_state, obstacles, neighbors)
        is_stagnant = float(self._is_stagnant(robot_state.get("position_history", [])))
        front_blocked = float(self._front_sector_blocked(lidar, dir_x, dir_y))
        last_collision = float(robot_state.get("last_collision", 0.0))

        if self.base_mode == "local22":
            frame = [dir_x, dir_y, distance_norm] + lidar + [is_stagnant, front_blocked, last_collision]
            return np.asarray(frame, dtype=np.float32)

        remaining_time = 0.5
        dfa_one_hot = [1.0, 0.0, 0.0, 0.0]
        frame = [dir_x, dir_y, distance_norm, last_collision, remaining_time] + dfa_one_hot + lidar + [is_stagnant, front_blocked]
        return np.asarray(frame, dtype=np.float32)

    def predict_action(
        self,
        robot_state: Dict,
        waypoint: tuple[float, float] | None,
        obstacles: Sequence[tuple[float, float, float, float]],
        neighbors: Sequence[Dict],
    ) -> np.ndarray:
        frame = self._build_frame(robot_state, waypoint, obstacles, neighbors)
        stacker = self._get_stacker(str(robot_state["id"]))
        stacked = stacker.append(frame)
        action, _ = self.model.predict(np.asarray([stacked], dtype=np.float32), deterministic=True)
        return np.asarray(action[0], dtype=np.float32)
