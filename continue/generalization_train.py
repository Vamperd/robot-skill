import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

from mutiple_train import HEIGHT, WIDTH, LTLfGymEnv, Obstacle, TaskPoint


BASE_OBSTACLES = [
    (200, 150, 50, 300),
    (450, 250, 200, 50),
    (500, 400, 50, 200),
]

BASE_TASKS = {
    "Task A": (100, 500),
    "Task B": (400, 100),
    "Task C": (700, 500),
}

BASE_START = (100.0, 100.0)

TRAIN_LAYOUT_LIBRARY = [
    {
        "name": "baseline",
        "obstacles": BASE_OBSTACLES,
        "tasks": BASE_TASKS,
        "start": BASE_START,
        "weight": 0.10,
    },
    {
        "name": "center_corridor",
        "obstacles": [(180, 170, 60, 260), (440, 230, 180, 60), (540, 390, 60, 180)],
        "tasks": {"Task A": (140, 500), "Task B": (470, 120), "Task C": (680, 470)},
        "start": (110.0, 90.0),
        "weight": 0.08,
    },
    {
        "name": "hard_ood_seed",
        "obstacles": [(150, 120, 80, 320), (320, 260, 240, 60), (620, 120, 50, 260)],
        "tasks": {"Task A": (120, 520), "Task B": (520, 90), "Task C": (720, 520)},
        "start": (85.0, 85.0),
        "weight": 0.18,
    },
    {
        "name": "narrow_upper_gate",
        "obstacles": [(140, 80, 60, 360), (260, 250, 220, 60), (560, 180, 60, 300)],
        "tasks": {"Task A": (90, 520), "Task B": (430, 90), "Task C": (720, 460)},
        "start": (80.0, 80.0),
        "weight": 0.12,
    },
    {
        "name": "u_shaped_trap",
        "obstacles": [(240, 120, 50, 280), (240, 350, 220, 50), (410, 120, 50, 280)],
        "tasks": {"Task A": (120, 500), "Task B": (350, 170), "Task C": (700, 500)},
        "start": (90.0, 90.0),
        "weight": 0.15,
    },
    {
        "name": "long_detour",
        "obstacles": [(180, 200, 420, 60), (180, 200, 60, 250), (540, 60, 60, 260)],
        "tasks": {"Task A": (120, 500), "Task B": (650, 110), "Task C": (700, 520)},
        "start": (100.0, 80.0),
        "weight": 0.14,
    },
    {
        "name": "double_passage",
        "obstacles": [(180, 130, 70, 340), (360, 0, 70, 260), (360, 340, 70, 260), (560, 140, 70, 320)],
        "tasks": {"Task A": (110, 520), "Task B": (500, 80), "Task C": (720, 520)},
        "start": (90.0, 90.0),
        "weight": 0.13,
    },
    {
        "name": "offset_maze",
        "obstacles": [(120, 150, 90, 260), (300, 80, 240, 70), (300, 320, 240, 70), (620, 180, 50, 260)],
        "tasks": {"Task A": (130, 500), "Task B": (550, 120), "Task C": (740, 470)},
        "start": (85.0, 95.0),
        "weight": 0.10,
    },
]

VALIDATION_LAYOUTS = [
    {
        "name": "val_baseline",
        "obstacles": BASE_OBSTACLES,
        "tasks": BASE_TASKS,
        "start": BASE_START,
    },
    {
        "name": "val_shift_obstacles",
        "obstacles": [(240, 120, 50, 280), (420, 300, 220, 50), (540, 380, 50, 180)],
        "tasks": BASE_TASKS,
        "start": BASE_START,
    },
    {
        "name": "val_shift_both",
        "obstacles": [(180, 180, 60, 260), (420, 220, 180, 60), (560, 360, 60, 200)],
        "tasks": {"Task A": (160, 480), "Task B": (500, 120), "Task C": (680, 420)},
        "start": (120.0, 80.0),
    },
    {
        "name": "val_hard_ood",
        "obstacles": [(150, 120, 80, 320), (320, 260, 260, 60), (620, 100, 50, 280)],
        "tasks": {"Task A": (120, 520), "Task B": (520, 80), "Task C": (720, 520)},
        "start": (80.0, 80.0),
    },
]


def circle_hits_rect(cx: float, cy: float, radius: float, rect: Tuple[float, float, float, float]) -> bool:
    x, y, w, h = rect
    closest_x = max(x, min(cx, x + w))
    closest_y = max(y, min(cy, y + h))
    return math.hypot(cx - closest_x, cy - closest_y) < radius


def rects_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], margin: float = 20.0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + margin < bx
        or bx + bw + margin < ax
        or ay + ah + margin < by
        or by + bh + margin < ay
    )


class RandomizedLTLfGymEnv(LTLfGymEnv):
    def __init__(
        self,
        render_mode: str = "None",
        seed: int = 0,
        curriculum_episodes: int = 8000,
        obstacle_shift: int = 180,
        task_shift: int = 220,
        start_shift: int = 110,
    ):
        super().__init__(render_mode=render_mode, observation_mode="relative")
        self.rng = np.random.default_rng(seed)
        self.episode_count = 0
        self.curriculum_episodes = curriculum_episodes
        self.obstacle_shift = obstacle_shift
        self.task_shift = task_shift
        self.start_shift = start_shift

    def _sample_layout_template(self):
        weights = np.array([layout["weight"] for layout in TRAIN_LAYOUT_LIBRARY], dtype=np.float64)
        weights = weights / weights.sum()
        index = int(self.rng.choice(len(TRAIN_LAYOUT_LIBRARY), p=weights))
        return TRAIN_LAYOUT_LIBRARY[index]

    def _maybe_reflect_point(self, point: Tuple[float, float], flip_x: bool, flip_y: bool):
        x, y = point
        if flip_x:
            x = WIDTH - x
        if flip_y:
            y = HEIGHT - y
        return (x, y)

    def _maybe_reflect_rect(self, rect: Tuple[int, int, int, int], flip_x: bool, flip_y: bool):
        x, y, w, h = rect
        if flip_x:
            x = WIDTH - x - w
        if flip_y:
            y = HEIGHT - y - h
        return (x, y, w, h)

    def _curriculum_scale(self) -> float:
        return min(1.0, self.episode_count / max(1, self.curriculum_episodes))

    def _sample_obstacles(self, base_obstacles: Sequence[Tuple[int, int, int, int]], scale: float) -> List[Tuple[int, int, int, int]]:
        sampled = []
        for base_rect in base_obstacles:
            bx, by, bw, bh = base_rect
            rect = base_rect
            for _ in range(80):
                dx = int(self.rng.integers(-self.obstacle_shift, self.obstacle_shift + 1) * scale)
                dy = int(self.rng.integers(-self.obstacle_shift, self.obstacle_shift + 1) * scale)
                x = int(np.clip(bx + dx, 40, WIDTH - bw - 40))
                y = int(np.clip(by + dy, 40, HEIGHT - bh - 40))
                candidate = (x, y, bw, bh)
                if any(rects_overlap(candidate, other) for other in sampled):
                    continue
                rect = candidate
                break
            sampled.append(rect)
        return sampled

    def _sample_start(self, base_start: Tuple[float, float], obstacles: Sequence[Tuple[int, int, int, int]], scale: float) -> Tuple[float, float]:
        for _ in range(100):
            dx = float(self.rng.integers(-self.start_shift, self.start_shift + 1) * scale)
            dy = float(self.rng.integers(-self.start_shift, self.start_shift + 1) * scale)
            sx = float(np.clip(base_start[0] + dx, 30, WIDTH - 30))
            sy = float(np.clip(base_start[1] + dy, 30, HEIGHT - 30))
            if any(circle_hits_rect(sx, sy, self.robot_radius + 10, rect) for rect in obstacles):
                continue
            return (sx, sy)
        return base_start

    def _sample_tasks(
        self,
        base_tasks: Dict[str, Tuple[int, int]],
        obstacles: Sequence[Tuple[int, int, int, int]],
        start_pos: Tuple[float, float],
        scale: float,
    ) -> Dict[str, Tuple[int, int]]:
        tasks: Dict[str, Tuple[int, int]] = {}
        for name, (bx, by) in base_tasks.items():
            point = (bx, by)
            for _ in range(120):
                dx = int(self.rng.integers(-self.task_shift, self.task_shift + 1) * scale)
                dy = int(self.rng.integers(-self.task_shift, self.task_shift + 1) * scale)
                x = int(np.clip(bx + dx, 35, WIDTH - 35))
                y = int(np.clip(by + dy, 35, HEIGHT - 35))

                if any(circle_hits_rect(x, y, 35, rect) for rect in obstacles):
                    continue
                if math.hypot(x - start_pos[0], y - start_pos[1]) < 120:
                    continue
                if any(math.hypot(x - tx, y - ty) < 120 for tx, ty in tasks.values()):
                    continue

                point = (x, y)
                break
            tasks[name] = point
        return tasks

    def reset(self, seed=None, options=None):
        self.episode_count += 1
        scale = self._curriculum_scale()
        layout = self._sample_layout_template()
        flip_x = bool(self.rng.integers(0, 2))
        flip_y = bool(self.rng.integers(0, 2))

        reflected_obstacles = [self._maybe_reflect_rect(rect, flip_x, flip_y) for rect in layout["obstacles"]]
        reflected_tasks = {name: self._maybe_reflect_point(point, flip_x, flip_y) for name, point in layout["tasks"].items()}
        reflected_start = self._maybe_reflect_point(layout["start"], flip_x, flip_y)

        obstacle_specs = self._sample_obstacles(reflected_obstacles, scale)
        start_pos = self._sample_start(reflected_start, obstacle_specs, scale)
        task_specs = self._sample_tasks(reflected_tasks, obstacle_specs, start_pos, scale)

        self.obstacles = [Obstacle(x, y, w, h) for x, y, w, h in obstacle_specs]
        self.tasks = {name: TaskPoint(x, y, name) for name, (x, y) in task_specs.items()}
        self.start_pos = start_pos
        return super().reset(seed=seed, options=options)


class ValidationLTLfGymEnv(LTLfGymEnv):
    def __init__(self, render_mode="None", observation_mode="relative"):
        first_layout = VALIDATION_LAYOUTS[0]
        super().__init__(
            render_mode=render_mode,
            obstacles=first_layout["obstacles"],
            tasks=first_layout["tasks"],
            start_pos=first_layout["start"],
            observation_mode=observation_mode,
        )
        self.layout_index = -1

    def reset(self, seed=None, options=None):
        self.layout_index = (self.layout_index + 1) % len(VALIDATION_LAYOUTS)
        layout = VALIDATION_LAYOUTS[self.layout_index]
        self.obstacles = [Obstacle(x, y, w, h) for x, y, w, h in layout["obstacles"]]
        self.tasks = {name: TaskPoint(x, y, name) for name, (x, y) in layout["tasks"].items()}
        self.start_pos = layout["start"]
        return super().reset(seed=seed, options=options)


def make_env(rank: int, base_seed: int = 42):
    def _init():
        return RandomizedLTLfGymEnv(render_mode="None", seed=base_seed + rank)

    return _init


def make_eval_env():
    def _init():
        return ValidationLTLfGymEnv(render_mode="None", observation_mode="relative")

    return _init


def main():
    num_envs = 8
    total_timesteps = 2_500_000

    vec_env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    vec_env = VecMonitor(vec_env)
    eval_env = DummyVecEnv([make_eval_env()])
    eval_env = VecMonitor(eval_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        use_sde=True,
        sde_sample_freq=4,
        device="cuda",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])),
        tensorboard_log="./ppo_ltlf_tensorboard_generalization_v3/",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./generalization_eval_best_v3/",
        log_path="./generalization_eval_logs_v3/",
        eval_freq=20_000,
        n_eval_episodes=len(VALIDATION_LAYOUTS) * 2,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
    model.save("ppo_ltlf_agent_generalization_v3_relative")
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()