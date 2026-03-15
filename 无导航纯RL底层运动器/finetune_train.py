import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
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

LAYOUT_LIBRARY = [
    {
        "obstacles": BASE_OBSTACLES,
        "tasks": BASE_TASKS,
        "start": BASE_START,
    },
    {
        "obstacles": [(180, 170, 60, 260), (440, 230, 180, 60), (540, 390, 60, 180)],
        "tasks": {"Task A": (140, 500), "Task B": (470, 120), "Task C": (680, 470)},
        "start": (110.0, 90.0),
    },
    {
        "obstacles": [(150, 120, 80, 320), (320, 260, 240, 60), (620, 120, 50, 260)],
        "tasks": {"Task A": (120, 520), "Task B": (520, 90), "Task C": (720, 520)},
        "start": (85.0, 85.0),
    },
    {
        # 创造一个横向的极窄缝隙通道
        "obstacles": [(200, 0, 400, 250), (200, 350, 400, 250)], 
        "tasks": {"Task A": (100, 300), "Task B": (700, 300), "Task C": (100, 100)},
        "start": (100.0, 300.0),
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
        curriculum_episodes: int = 3000,
        obstacle_shift: int = 140,
        task_shift: int = 180,
        start_shift: int = 80,
    ):
        super().__init__(render_mode=render_mode, observation_mode="relative")
        self.rng = np.random.default_rng(seed)
        self.episode_count = 0
        self.curriculum_episodes = curriculum_episodes
        self.obstacle_shift = obstacle_shift
        self.task_shift = task_shift
        self.start_shift = start_shift

    def _sample_layout_template(self):
        return LAYOUT_LIBRARY[int(self.rng.integers(0, len(LAYOUT_LIBRARY)))]

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
        base_tasks: Dict[str, Tuple[float, float]],
        obstacles: Sequence[Tuple[int, int, int, int]],
        start_pos: Tuple[float, float],
        scale: float,
    ) -> Dict[str, Tuple[float, float]]:
        tasks: Dict[str, Tuple[float, float]] = {}
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


def make_env(rank: int, base_seed: int = 42):
    def _init():
        env = RandomizedLTLfGymEnv(render_mode="None", seed=base_seed + rank)
        return Monitor(env)

    return _init


def make_eval_env():
    def _init():
        env = LTLfGymEnv(render_mode="None", observation_mode="relative")
        return Monitor(env)

    return _init


def main():
    num_envs = 8
    total_timesteps = 500_000

    vec_env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    eval_env = DummyVecEnv([make_eval_env()])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # 指定你想要微调的基座模型路径（请根据实际情况修改）
    model_path = "./generalization_eval_best_vel_punishment/best_model.zip" 
    
    print(f"正在加载预训练模型: {model_path}")
    
    # 为微调定制超参数字典
    custom_objects = {
        "learning_rate": 5e-5,  # 极低的学习率，防止灾难性遗忘
        "ent_coef": 0.0005,     # 极低的熵，抑制胡乱探索，保持双手平稳
        "clip_range": 0.1,      # 更保守的策略更新截断
    }
    
    # 加载模型，并将新的向量化环境和微调超参数注入
    model = PPO.load(
        model_path, 
        env=vec_env, 
        custom_objects=custom_objects,
        tensorboard_log="./ppo_ltlf_tensorboard_finetune/"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./finetune_eval_best/",
        log_path="./finetune_eval_logs/",
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )

    # 将学习过程放入 try-except 块中
    try:
        print("开始微调训练... (按 Ctrl+C 可安全中断并保存当前进度)")
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断信号
        print("\n[警告] 训练被手动中断！正在保存当前最新模型...")
    except Exception as e:
        # 捕获其他未知报错
        print(f"\n[错误] 训练发生异常崩溃: {e}")
        print("正在尝试抢救保存当前最新模型...")
    finally:
        # 无论正常结束、被中断还是报错，这里的代码都一定会执行！
        save_path = "ppo_ltlf_agent_finetuned_latest"
        model.save(save_path)
        print(f"最新微调模型已安全保存至: {save_path}.zip")
        
        # 必须显式关闭环境，释放内存和线程
        vec_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
