import gymnasium as gym
from collections import deque
from gymnasium import spaces
import numpy as np
import pygame
import math
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# --- 环境基础设定 ---
WIDTH, HEIGHT = 800, 600
WHITE = (240, 240, 240)
BLUE = (50, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
DARK_GRAY = (50, 50, 50)
BLACK = (30, 30, 30)
YELLOW = (255, 200, 0)

class Obstacle:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
    def draw(self, surface):
        pygame.draw.rect(surface, BLACK, self.rect)
    def is_colliding_with_circle(self, cx, cy, radius):
        closest_x = max(self.rect.left, min(cx, self.rect.right))
        closest_y = max(self.rect.top, min(cy, self.rect.bottom))
        return math.hypot(cx - closest_x, cy - closest_y) < radius

class TaskPoint:
    def __init__(self, x, y, name):
        self.x, self.y = float(x), float(y)
        self.radius = 25
        self.name = name

class LocalDFA:
    def __init__(self):
        self.state = 0
        self.accepting_state = 3

    def step(self, triggered_task_name):
        if self.state == 0 and triggered_task_name == "Task A":
            self.state = 1
            return True
        elif self.state == 1 and triggered_task_name == "Task B":
            self.state = 2
            return True
        elif self.state == 2 and triggered_task_name == "Task C":
            self.state = 3
            return True
        return False
        
    def get_current_target_name(self):
        if self.state == 0: return "Task A"
        if self.state == 1: return "Task B"
        if self.state == 2: return "Task C"
        return None

# --- 核心：Gymnasium 封装 ---
class LTLfGymEnv(gym.Env):
    """标准的 Gymnasium 环境封装"""
    metadata = {"render_modes": ["human", "None"], "render_fps": 60}

    def __init__(self, render_mode="None", obstacles=None, tasks=None, start_pos=(100.0, 100.0), max_steps=2000, observation_mode="absolute"):
        super().__init__()
        self.render_mode = render_mode
        self.robot_radius = 15
        self.robot_vmax = 250.0 
        self.dt = 1.0 / 60.0    
        self.max_steps = max_steps
        self.current_step = 0
        self.start_pos = (float(start_pos[0]), float(start_pos[1]))
        self.observation_mode = observation_mode
        self.last_collision = 0.0
        self.closest_distance = 0.0
        
        # 1. Action Space: 神经网络输出 [-1.0, 1.0] 之间的连续二维向量
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = self._build_observation_space()

        obstacle_specs = obstacles or [
            (200, 150, 50, 300),
            (450, 250, 200, 50),
            (500, 400, 50, 200),
        ]
        task_specs = tasks or {
            "Task A": (100, 500),
            "Task B": (400, 100),
            "Task C": (700, 500),
        }
        self.obstacles = [Obstacle(x, y, w, h) for x, y, w, h in obstacle_specs]
        self.tasks = {name: TaskPoint(x, y, name) for name, (x, y) in task_specs.items()}

        # 奖励设置
        self.REWARD_GOAL = 200.0        # 提高最终奖励
        self.REWARD_TRANSITION = 50.0   # 提高阶段奖励
        self.REWARD_COLLISION = -1.0    # (关键) 大幅降低碰撞惩罚，允许试错
        self.REWARD_TIME_STEP = -0.1    # (关键) 加大时间惩罚，逼迫它动起来
        self.POTENTIAL_SCALE = 2.0      # (关键) 加大势能引力   
        self.REWARD_STAGNATION = -0.5
        self.position_history = deque(maxlen=60)
        self.stagnation_std_threshold = 5.0
        self.front_wall_threshold = 0.25

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("PPO Agent Testing")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 20, bold=True)

    def _build_observation_space(self):
        if self.observation_mode == "relative":
            low = np.array([-1.0, -1.0, 0.0, 0.0, 0.0] + [0.0] * 4 + [0.0] * 16 + [0.0, 0.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * 4 + [1.0] * 16 + [1.0, 1.0], dtype=np.float32)
        else:
            low = np.array([0.0, 0.0, -1.0, -1.0, 0.0] + [0.0] * 16 + [0.0, 0.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * 16 + [1.0, 1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _is_stagnant(self) -> bool:
        maxlen = self.position_history.maxlen
        if maxlen is None:
            maxlen = 60
        if len(self.position_history) < maxlen:
            return False

        positions = np.array(self.position_history, dtype=np.float32)
        std_sum = float(np.std(positions[:, 0]) + np.std(positions[:, 1]))
        return std_sum < self.stagnation_std_threshold

    def _front_sector_min_distance(self, lidar_distances, vx: float, vy: float) -> float:
        if abs(vx) <= 1e-6 and abs(vy) <= 1e-6:
            return 1.0

        move_angle = math.atan2(vy, vx)
        num_rays = len(lidar_distances)
        angle_step = 2 * math.pi / num_rays
        center_idx = int(round(move_angle / angle_step)) % num_rays
        sector_half_width = 1
        front_sector = [lidar_distances[(center_idx + offset) % num_rays] for offset in range(-sector_half_width, sector_half_width + 1)]
        return float(min(front_sector))

    def _front_sector_blocked(self, lidar_distances, vx: float, vy: float) -> bool:
        return self._front_sector_min_distance(lidar_distances, vx, vy) < self.front_wall_threshold

    def _get_dfa_one_hot(self):
        dfa_one_hot = [0.0] * 4
        dfa_one_hot[self.dfa.state] = 1.0
        return dfa_one_hot

    def _get_normalized_obs(self):
        norm_x = self.rx / WIDTH
        norm_y = self.ry / HEIGHT
        norm_dfa = self.dfa.state / 3.0 
        
        target_name = self.dfa.get_current_target_name()
        if target_name:
            target = self.tasks[target_name]
            dx = (target.x - self.rx) / WIDTH
            dy = (target.y - self.ry) / HEIGHT
            front_vx = target.x - self.rx
            front_vy = target.y - self.ry
        else:
            dx, dy = 0.0, 0.0 
            front_vx, front_vy = 0.0, 0.0
            
        base_obs = [norm_x, norm_y, dx, dy, norm_dfa]
        lidar_obs = self._get_lidar_data()
        is_stagnant = float(self._is_stagnant())
        front_blocked = float(self._front_sector_blocked(lidar_obs, front_vx, front_vy))
        
        # 将基础列表和雷达列表拼接后转换为 numpy 数组
        return np.array(base_obs + lidar_obs + [is_stagnant, front_blocked], dtype=np.float32)

    def _get_relative_obs(self):
        target_name = self.dfa.get_current_target_name()
        if target_name:
            target = self.tasks[target_name]
            dx = target.x - self.rx
            dy = target.y - self.ry
            distance = math.hypot(dx, dy)
            if distance > 1e-6:
                dir_x = dx / distance
                dir_y = dy / distance
            else:
                dir_x, dir_y = 0.0, 0.0
            distance_norm = min(distance / math.hypot(WIDTH, HEIGHT), 1.0)
        else:
            dir_x, dir_y, distance_norm = 0.0, 0.0, 0.0

        remaining_time = 1.0 - (self.current_step / self.max_steps)
        lidar_obs = self._get_lidar_data()
        base_obs = [dir_x, dir_y, distance_norm, self.last_collision, remaining_time]
        is_stagnant = float(self._is_stagnant())
        front_blocked = float(self._front_sector_blocked(lidar_obs, dir_x, dir_y))
        return np.array(base_obs + self._get_dfa_one_hot() + lidar_obs + [is_stagnant, front_blocked], dtype=np.float32)

    def _get_obs(self):
        if self.observation_mode == "relative":
            return self._get_relative_obs()
        return self._get_normalized_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rx, self.ry = self.start_pos
        self.dfa = LocalDFA()
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.last_collision = 0.0
        self.position_history.clear()
        self.position_history.append((self.rx, self.ry))
        
        self._update_potential_anchor()
        return self._get_obs(), {}

    def _update_potential_anchor(self):
        target_name = self.dfa.get_current_target_name()
        if target_name:
            target = self.tasks[target_name]
            self.closest_distance = math.hypot(self.rx - target.x, self.ry - target.y)

    def step(self, action):
        self.current_step += 1
        vx = action[0] * self.robot_vmax
        vy = action[1] * self.robot_vmax
        
        step_reward = self.REWARD_TIME_STEP
        
        # 1. 记录初始安全位置
        old_x, old_y = self.rx, self.ry
        collided = False
        BOUNCE_COEF = 0.8  # 物理反弹系数（0.0为死锁停滞，1.0为完全弹性反弹）

        # 2. 仅更新 X 轴并检测碰撞
        self.rx += vx * self.dt
        if self.rx < self.robot_radius or self.rx > WIDTH - self.robot_radius:
            # 触发边界反弹：退回到安全位置，并向反方向弹开
            self.rx = old_x - vx * self.dt * BOUNCE_COEF
            collided = True
        else:
            for obs in self.obstacles:
                if obs.is_colliding_with_circle(self.rx, self.ry, self.robot_radius):
                    # 触发障碍物反弹
                    self.rx = old_x - vx * self.dt * BOUNCE_COEF
                    collided = True
                    break

        # 3. 仅更新 Y 轴并检测碰撞
        self.ry += vy * self.dt
        if self.ry < self.robot_radius or self.ry > HEIGHT - self.robot_radius:
            self.ry = old_y - vy * self.dt * BOUNCE_COEF
            collided = True
        else:
            for obs in self.obstacles:
                if obs.is_colliding_with_circle(self.rx, self.ry, self.robot_radius):
                    self.ry = old_y - vy * self.dt * BOUNCE_COEF
                    collided = True
                    break

        self.position_history.append((self.rx, self.ry))
        lidar_distances = self._get_lidar_data()
        is_stagnant = self._is_stagnant()
        front_min_lidar = self._front_sector_min_distance(lidar_distances, vx, vy)
        front_blocked = self._front_sector_blocked(lidar_distances, vx, vy)
        
        # 计算基于速度向量投影的动态惩罚
        DANGER_ZONE = 0.15
        VELOCITY_PENALTY_SCALE = 0.005
        danger_penalty = 0.0
        num_rays = len(lidar_distances)
        for i, dist in enumerate(lidar_distances):
            if dist < DANGER_ZONE:
                angle = i * (2 * math.pi / num_rays)
                ray_dx = math.cos(angle)
                ray_dy = math.sin(angle)
                v_proj = vx * ray_dx + vy * ray_dy
                if v_proj > 0:
                    penalty = (DANGER_ZONE - dist) * v_proj * VELOCITY_PENALTY_SCALE
                    danger_penalty += penalty
                    
        step_reward -= danger_penalty

        self.last_collision = float(collided)
        if collided:
            step_reward += self.REWARD_COLLISION

        terminated = False
        truncated = False

        target_name = self.dfa.get_current_target_name()
        if target_name:
            target = self.tasks[target_name]
            current_distance = math.hypot(self.rx - target.x, self.ry - target.y)

            if is_stagnant or front_blocked:
                potential_reward = 0.0
                step_reward += self.REWARD_STAGNATION
            else:
                if current_distance < self.closest_distance:
                    potential_reward = (self.closest_distance - current_distance) * self.POTENTIAL_SCALE
                    self.closest_distance = current_distance
                else:
                    potential_reward = 0.0

            step_reward += potential_reward

            if current_distance <= (self.robot_radius + target.radius):
                if self.dfa.step(target_name):
                    if self.dfa.state == self.dfa.accepting_state:
                        step_reward += self.REWARD_GOAL
                        terminated = True
                    else:
                        step_reward += self.REWARD_TRANSITION
                        self._update_potential_anchor()

        if self.current_step >= self.max_steps:
            truncated = True

        self.cumulative_reward += step_reward
        info = {
            "dfa_state": int(self.dfa.state),
            "is_success": bool(terminated and self.dfa.state == self.dfa.accepting_state),
            "is_stagnant": bool(is_stagnant),
            "front_blocked": bool(front_blocked),
            "front_min_lidar": float(front_min_lidar),
        }
        return self._get_obs(), step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human": return

        self.screen.fill(WHITE)
        for obs in self.obstacles:
            obs.draw(self.screen)

        active_target = self.dfa.get_current_target_name()
        for name, task in self.tasks.items():
            color = GREEN if ["Task A", "Task B", "Task C"].index(name) < self.dfa.state else RED
            pygame.draw.circle(self.screen, color, (int(task.x), int(task.y)), task.radius)
            if name == active_target:
                pygame.draw.circle(self.screen, YELLOW, (int(task.x), int(task.y)), task.radius + 5, 3)
            text = self.font.render(name, True, WHITE)
            self.screen.blit(text, text.get_rect(center=(int(task.x), int(task.y))))

        # =========================================
        # 【新增代码】：绘制激光雷达 (Lidar) 射线
        # =========================================
        if hasattr(self, 'last_lidar_distances'):
            num_rays = 16
            max_range = 150.0
            for i in range(num_rays):
                angle = i * (2 * math.pi / num_rays)
                # 将归一化的数据恢复为实际物理像素长度
                dist = self.last_lidar_distances[i] * max_range 
                
                # 计算射线末端坐标
                end_x = self.rx + math.cos(angle) * dist
                end_y = self.ry + math.sin(angle) * dist
                
                # 视觉反馈逻辑：如果距离小于最大量程，说明撞墙了，画红色；如果前方畅通，画浅灰色
                if dist < max_range:
                    line_color = (255, 80, 80)   # 红色警报
                    point_color = (255, 0, 0)
                else:
                    line_color = (200, 200, 200) # 安全灰色
                    point_color = (150, 150, 150)
                
                # 画出从机器人中心到探测点的线段
                pygame.draw.line(self.screen, line_color, (int(self.rx), int(self.ry)), (int(end_x), int(end_y)), 2)
                # 在探测的末端画一个反馈小圆点
                pygame.draw.circle(self.screen, point_color, (int(end_x), int(end_y)), 3)
        # =========================================

        pygame.draw.circle(self.screen, BLUE, (int(self.rx), int(self.ry)), self.robot_radius)
        pygame.draw.circle(self.screen, WHITE, (int(self.rx), int(self.ry)), 3)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _get_lidar_data(self):
        num_rays = 16
        max_range = 150.0  # 雷达最大探测距离（像素）
        lidar_distances = []
        
        for i in range(num_rays):
            angle = i * (2 * math.pi / num_rays)
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
            
            distance = max_range
            # 步进探测 (Raymarching)，每次步进 5 个像素
            for step in range(1, int(max_range), 5):
                test_x = self.rx + ray_dx * step
                test_y = self.ry + ray_dy * step
                
                # 1. 检查是否撞到屏幕边界
                if test_x < 0 or test_x > WIDTH or test_y < 0 or test_y > HEIGHT:
                    distance = step
                    break
                    
                # 2. 检查是否撞到障碍物
                hit = False
                for obs in self.obstacles:
                    # Pygame 的 collidepoint 函数可以极速判断点是否在矩形内
                    if obs.rect.collidepoint(test_x, test_y):
                        hit = True
                        break
                
                if hit:
                    distance = step
                    break
                    
            # 将测量的距离归一化到 [0, 1]，喂给神经网络更稳定
            lidar_distances.append(distance / max_range)
        self.last_lidar_distances = lidar_distances
        return lidar_distances

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

# ==========================================
# 训练与测试主逻辑
# ==========================================
if __name__ == "__main__":
    print("=== 开始训练 PPO 智能体 (关闭画面以全速运行) ===")
    # 训练时使用 render_mode="None" 可以极大提升采样速度
    train_env = DummyVecEnv([lambda: LTLfGymEnv(render_mode="None")])
    train_env = VecFrameStack(train_env, n_stack=4)
    
    # 实例化 PPO 模型 (MlpPolicy 适用于一维向量输入)
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4, tensorboard_log="./ppo_ltlf_tensorboard/")
    
    # 开始训练！由于有极好的势能奖励，10万步左右就能看出明显效果
    # 想要更完美可以改成 300_000 甚至 500_000
    model.learn(total_timesteps=150_000)
    
    # 保存模型
    model.save("ppo_ltlf_agent")
    train_env.close()

    print("\n=== 训练完成！开启 Pygame 画面进行测试 ===")
    test_env = DummyVecEnv([lambda: LTLfGymEnv(render_mode="human")])
    test_env = VecFrameStack(test_env, n_stack=4)
    model = PPO.load("ppo_ltlf_agent")

    obs = test_env.reset()
    for _ in range(2000): # 跑一局测试
        # 处理窗口退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                test_env.close()
                sys.exit()
                
        # 让训练好的模型根据 Observation 预测动作 (deterministic=True 表现更稳定)
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        
        if done[0]:
            print("回合结束！准备重置...")
            pygame.time.wait(1000) # 停顿一秒让你看清胜利画面
            obs = test_env.reset()

    test_env.close()
