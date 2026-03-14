import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import sys
from stable_baselines3 import PPO

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
        self.num_lidar_rays = 16
        self.lidar_range = 220.0
        
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
        self.REWARD_GOAL = 300.0        # 提高最终奖励，鼓励完整任务闭环
        self.REWARD_TRANSITION = 80.0   # 提高阶段奖励，稳定通过中间目标
        self.REWARD_COLLISION = -1.5    # 显著提高碰撞代价，鼓励绕障而非硬顶
        self.REWARD_TIME_STEP = -0.04   # 适度时间惩罚，避免过度追求局部最短路
        self.POTENTIAL_SCALE = 1.0      # 降低势能牵引，减少被障碍物诱导卡死

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("PPO Agent Testing")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 20, bold=True)

    def _build_observation_space(self):
        if self.observation_mode == "relative":
            low = np.array([-1.0, -1.0, 0.0, 0.0, 0.0] + [0.0] * 4 + [0.0] * self.num_lidar_rays, dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * 4 + [1.0] * self.num_lidar_rays, dtype=np.float32)
        else:
            low = np.array([0.0, 0.0, -1.0, -1.0, 0.0] + [0.0] * self.num_lidar_rays, dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * self.num_lidar_rays, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

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
        else:
            dx, dy = 0.0, 0.0 
            
        base_obs = [norm_x, norm_y, dx, dy, norm_dfa]
        lidar_obs = self._get_lidar_data()
        
        # 将基础列表和雷达列表拼接后转换为 numpy 数组
        return np.array(base_obs + lidar_obs, dtype=np.float32)

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
        return np.array(base_obs + self._get_dfa_one_hot() + lidar_obs, dtype=np.float32)

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
        
        self._update_potential_anchor()
        return self._get_obs(), {}

    def _update_potential_anchor(self):
        target_name = self.dfa.get_current_target_name()
        if target_name:
            target = self.tasks[target_name]
            self.prev_distance = math.hypot(self.rx - target.x, self.ry - target.y)

    def step(self, action):
        self.current_step += 1
        vx = action[0] * self.robot_vmax
        vy = action[1] * self.robot_vmax
        
        step_reward = self.REWARD_TIME_STEP
        old_x, old_y = self.rx, self.ry

        self.rx += vx * self.dt
        self.ry += vy * self.dt

        collided = False
        if self.rx < self.robot_radius or self.rx > WIDTH - self.robot_radius:
            self.rx, collided = old_x, True
        if self.ry < self.robot_radius or self.ry > HEIGHT - self.robot_radius:
            self.ry, collided = old_y, True

        for obs in self.obstacles:
            if obs.is_colliding_with_circle(self.rx, self.ry, self.robot_radius):
                self.rx, self.ry = old_x, old_y
                collided = True
                break
        
        self.last_collision = float(collided)
        if collided: step_reward += self.REWARD_COLLISION

        terminated = False
        truncated = False

        target_name = self.dfa.get_current_target_name()
        if target_name:
            target = self.tasks[target_name]
            current_distance = math.hypot(self.rx - target.x, self.ry - target.y)
            
            # 势能计算保持不变，因为是相对差值
            potential_reward = (self.prev_distance - current_distance) * self.POTENTIAL_SCALE
            step_reward += potential_reward
            self.prev_distance = current_distance

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
        return self._get_obs(), step_reward, terminated, truncated, {}

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
            for i in range(self.num_lidar_rays):
                angle = i * (2 * math.pi / self.num_lidar_rays)
                # 将归一化的数据恢复为实际物理像素长度
                dist = self.last_lidar_distances[i] * self.lidar_range 
                
                # 计算射线末端坐标
                end_x = self.rx + math.cos(angle) * dist
                end_y = self.ry + math.sin(angle) * dist
                
                # 视觉反馈逻辑：如果距离小于最大量程，说明撞墙了，画红色；如果前方畅通，画浅灰色
                if dist < self.lidar_range:
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
        lidar_distances = []
        
        for i in range(self.num_lidar_rays):
            angle = i * (2 * math.pi / self.num_lidar_rays)
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
            
            distance = self.lidar_range
            # 步进探测 (Raymarching)，每次步进 5 个像素
            for step in range(1, int(self.lidar_range), 5):
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
            lidar_distances.append(distance / self.lidar_range)
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
    train_env = LTLfGymEnv(render_mode="None")
    
    # 实例化 PPO 模型 (MlpPolicy 适用于一维向量输入)
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4, tensorboard_log="./ppo_ltlf_tensorboard/")
    
    # 开始训练！由于有极好的势能奖励，10万步左右就能看出明显效果
    # 想要更完美可以改成 300_000 甚至 500_000
    model.learn(total_timesteps=150_000)
    
    # 保存模型
    model.save("ppo_ltlf_agent")
    train_env.close()

    print("\n=== 训练完成！开启 Pygame 画面进行测试 ===")
    test_env = LTLfGymEnv(render_mode="human")
    model = PPO.load("ppo_ltlf_agent")

    obs, _ = test_env.reset()
    for _ in range(2000): # 跑一局测试
        # 处理窗口退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                test_env.close()
                sys.exit()
                
        # 让训练好的模型根据 Observation 预测动作 (deterministic=True 表现更稳定)
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        
        if terminated or truncated:
            print("回合结束！准备重置...")
            pygame.time.wait(1000) # 停顿一秒让你看清胜利画面
            obs, _ = test_env.reset()

    test_env.close()