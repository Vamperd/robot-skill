import gymnasium as gym
from collections import deque
from gymnasium import spaces
import numpy as np
import pygame
import math
import random

# --- 环境基础设定 ---
WIDTH, HEIGHT = 800, 600
WHITE = (240, 240, 240)
BLUE = (50, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
BLACK = (30, 30, 30)

class Obstacle:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
    def draw(self, surface):
        pygame.draw.rect(surface, BLACK, self.rect)
    def is_colliding_with_circle(self, cx, cy, radius):
        closest_x = max(self.rect.left, min(cx, self.rect.right))
        closest_y = max(self.rect.top, min(cy, self.rect.bottom))
        return math.hypot(cx - closest_x, cy - closest_y) < radius

class LocalRLEnv(gym.Env):
    """
    纯物理底层 RL 运动器环境 (A* + PPO 混合导航系统局部规划模块)
    不包含任何 DFA、多任务序列或上层逻辑。仅执行 point-to-point 移动并避障。
    """
    metadata = {"render_modes": ["human", "None"], "render_fps": 60}

    def __init__(self, render_mode="None", max_steps=500):
        super().__init__()
        self.render_mode = render_mode
        self.robot_radius = 15
        self.robot_vmax = 250.0 
        self.dt = 1.0 / 60.0    
        self.max_steps = max_steps
        self.current_step = 0
        self.last_collision = 0.0
        self.closest_distance = 0.0
        
        # 1. Action Space: [-1.0, 1.0] 的 2D 连续向量
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 2. Observation Space: 22 维 (3引力 + 16雷达 + 3反馈)
        low = np.array([-1.0, -1.0, 0.0] + [0.0] * 16 + [0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0] + [1.0] * 16 + [1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.obstacles = []
        self.target_waypoint = (0.0, 0.0)

        # 奖励设置
        self.REWARD_GOAL = 200.0        
        self.REWARD_COLLISION = -10.0   
        self.REWARD_TIME_STEP = -0.1    
        self.POTENTIAL_SCALE = 1.0      
        self.REWARD_STAGNATION = -0.5
        
        self.position_history = deque(maxlen=60)
        self.stagnation_std_threshold = 5.0
        self.front_wall_threshold = 0.25
        self.max_lidar_range = 150.0    # 归一化用的距离基准也是 150.0

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Local RL Planner")
            self.clock = pygame.time.Clock()

    def _is_stagnant(self) -> bool:
        maxlen = self.position_history.maxlen if self.position_history.maxlen else 60
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

    def _get_obs(self):
        # 1. 引力（3维）
        tx, ty = self.target_waypoint
        dx = tx - self.rx
        dy = ty - self.ry
        distance = math.hypot(dx, dy)
        
        if distance > 1e-6:
            dir_x = dx / distance
            dir_y = dy / distance
        else:
            dir_x, dir_y = 0.0, 0.0
            
        distance_norm = min(distance / self.max_lidar_range, 1.0)
        
        # 2. 雷达（16维）
        lidar_obs = self._get_lidar_data()
        
        # 3. 反馈（3维）
        is_stagnant = float(self._is_stagnant())
        front_blocked = float(self._front_sector_blocked(lidar_obs, dir_x, dir_y))
        last_collision = float(self.last_collision)
        
        obs_list = [dir_x, dir_y, distance_norm] + lidar_obs + [is_stagnant, front_blocked, last_collision]
        return np.array(obs_list, dtype=np.float32)

    def _get_lidar_data(self):
        num_rays = 16
        lidar_distances = []
        for i in range(num_rays):
            angle = i * (2 * math.pi / num_rays)
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
            
            distance = self.max_lidar_range
            for step in range(1, int(self.max_lidar_range), 5):
                test_x = self.rx + ray_dx * step
                test_y = self.ry + ray_dy * step
                
                if test_x < 0 or test_x > WIDTH or test_y < 0 or test_y > HEIGHT:
                    distance = step
                    break
                    
                hit = False
                for obs in self.obstacles:
                    if obs.rect.collidepoint(test_x, test_y):
                        hit = True
                        break
                if hit:
                    distance = step
                    break
            
            normalized_dist = distance / self.max_lidar_range
            lidar_distances.append(normalized_dist)
            
        self.last_lidar_distances = lidar_distances  # 供渲染使用
        return lidar_distances

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # 随机分配起点坐标保证不越界
        margin = 50
        self.rx = np.random.uniform(margin, WIDTH - margin)
        self.ry = np.random.uniform(margin, HEIGHT - margin)
        
        # 随机分配目标点，与小车距离保持在一定范围内(50~200)
        while True:
            tx = np.random.uniform(margin, WIDTH - margin)
            ty = np.random.uniform(margin, HEIGHT - margin)
            dist = math.hypot(tx - self.rx, ty - self.ry)
            if 50 <= dist <= 200:
                self.target_waypoint = (tx, ty)
                break
                
        # 随机生成3~4个矩形障碍物，并确保它不压住机器人或目标点
        self.obstacles = []
        num_obstacles = random.randint(3, 4)
        for _ in range(num_obstacles):
            for _ in range(50): # 尝试多次放置
                w = random.uniform(50, 200)
                h = random.uniform(50, 200)
                x = random.uniform(0, WIDTH - w)
                y = random.uniform(0, HEIGHT - h)
                obs_rect = pygame.Rect(x, y, w, h)
                
                # 检查是否离起点或终点太近
                dist_to_start = math.hypot(obs_rect.centerx - self.rx, obs_rect.centery - self.ry)
                dist_to_end = math.hypot(obs_rect.centerx - tx, obs_rect.centery - ty)
                
                if dist_to_start > (max(w, h)/2 + self.robot_radius + 20) and \
                   dist_to_end > (max(w, h)/2 + 20):
                    self.obstacles.append(Obstacle(x, y, w, h))
                    break

        self.current_step = 0
        self.last_collision = 0.0
        self.position_history.clear()
        self.position_history.append((self.rx, self.ry))
        
        self.closest_distance = math.hypot(self.rx - tx, self.ry - ty)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        vx = action[0] * self.robot_vmax
        vy = action[1] * self.robot_vmax
        
        step_reward = self.REWARD_TIME_STEP
        
        old_x, old_y = self.rx, self.ry
        collided = False
        BOUNCE_COEF = 0.8 

        # X 轴
        self.rx += vx * self.dt
        if self.rx < self.robot_radius or self.rx > WIDTH - self.robot_radius:
            self.rx = old_x - vx * self.dt * BOUNCE_COEF
            collided = True
        else:
            for obs in self.obstacles:
                if obs.is_colliding_with_circle(self.rx, self.ry, self.robot_radius):
                    self.rx = old_x - vx * self.dt * BOUNCE_COEF
                    collided = True
                    break

        # Y 轴
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
        front_blocked = self._front_sector_blocked(lidar_distances, vx, vy)
        
        # 雷达危险斥力惩罚
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
                    penalty = ((DANGER_ZONE - dist) / DANGER_ZONE) ** 2 * v_proj * VELOCITY_PENALTY_SCALE
                    danger_penalty += penalty
                    
        step_reward -= danger_penalty
        
        self.last_collision = float(collided)
        if collided:
            step_reward += self.REWARD_COLLISION

        terminated = False
        truncated = False

        tx, ty = self.target_waypoint
        current_distance = math.hypot(self.rx - tx, self.ry - ty)

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

        # 到达判断 (15像素)
        if current_distance <= 15.0:
            step_reward += self.REWARD_GOAL
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True

        info = {
            "is_success": terminated,
            "is_stagnant": bool(is_stagnant),
            "front_blocked": bool(front_blocked),
        }
        return self._get_obs(), float(step_reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human": return

        self.screen.fill(WHITE)
        for obs in self.obstacles:
            obs.draw(self.screen)

        # 绘制目标点 (红色小圈)
        tx, ty = self.target_waypoint
        pygame.draw.circle(self.screen, RED, (int(tx), int(ty)), 15)
        # 用空心圈标记15范围的成功区
        pygame.draw.circle(self.screen, (255, 100, 100), (int(tx), int(ty)), 15, 2)

        # 绘制激光雷达射线
        if hasattr(self, 'last_lidar_distances'):
            num_rays = 16
            for i in range(num_rays):
                angle = i * (2 * math.pi / num_rays)
                dist = self.last_lidar_distances[i] * self.max_lidar_range
                
                end_x = self.rx + math.cos(angle) * dist
                end_y = self.ry + math.sin(angle) * dist
                
                if dist < self.max_lidar_range:
                    line_color = (255, 80, 80)
                    point_color = (255, 0, 0)
                else:
                    line_color = (200, 200, 200)
                    point_color = (150, 150, 150)
                
                pygame.draw.line(self.screen, line_color, (int(self.rx), int(self.ry)), (int(end_x), int(end_y)), 2)
                pygame.draw.circle(self.screen, point_color, (int(end_x), int(end_y)), 3)

        # 绘制机器人
        pygame.draw.circle(self.screen, BLUE, (int(self.rx), int(self.ry)), self.robot_radius)
        pygame.draw.circle(self.screen, WHITE, (int(self.rx), int(self.ry)), 3)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
