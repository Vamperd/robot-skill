import os
import sys
import pygame
import math
import numpy as np
import imageio
from collections import deque
from stable_baselines3 import PPO

from a_star_planner import AStarPlanner, get_lookahead_waypoint

# --- UI 颜色与设定 ---
WIDTH, HEIGHT = 800, 600
WHITE = (240, 240, 240)
BLUE = (50, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
YELLOW = (255, 200, 0)
BLACK = (30, 30, 30)
LIGHT_BLUE = (150, 200, 255)
PINK = (255, 105, 180)

# ====== 录制配置 ======
RECORD_GIF = True
GIF_SAVE_PATH = "mixed_nav_hard_result.gif"
# ======================

# 定义预设任务序列
TASK_SEQUENCE = ["Task A", "Task B", "Task C"]

# 终极泛化深渊地图
HARD_OBSTACLES = [
    (250, 0, 40, 480),
    (450, 120, 40, 480),
    (290, 280, 110, 40),
    (0, 400, 150, 40),
    (150, 400, 40, 140),
    (580, 360, 220, 40),
    (580, 480, 220, 40),
    (760, 400, 40, 80)
]

HARD_TASKS = {
    "Task A": (680, 60), 
    "Task B": (80, 500), 
    "Task C": (680, 420)
}

HARD_START = (60.0, 60.0)


class FrameStacker:
    """手动实现 4 帧堆叠，以匹配 VecFrameStack 的输入格式"""
    def __init__(self, shape_1d=22, n_stack=4):
        self.n_stack = n_stack
        self.shape_1d = shape_1d
        self.frames = deque(maxlen=n_stack)
        
    def reset(self, obs):
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self.get_stacked_obs()
        
    def append(self, obs):
        self.frames.append(obs)
        return self.get_stacked_obs()
        
    def get_stacked_obs(self):
        stacked = np.concatenate(self.frames)
        return stacked


class MixedNavEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hard Mixed Navigation (A* + PPO)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20, bold=True)
        
        # 物理与约束
        self.robot_radius = 15
        self.robot_vmax = 250.0
        self.dt = 1.0 / 60.0
        
        # 初始状态
        self.rx, self.ry = HARD_START
        self.obstacles = [pygame.Rect(x, y, w, h) for x, y, w, h in HARD_OBSTACLES]
        
        # A* 规划器 (加入5像素膨胀裕度)
        self.planner = AStarPlanner(width=WIDTH, height=HEIGHT, resolution=10, robot_radius=self.robot_radius, margin=10)
        
        # 任务进度管理
        self.task_idx = 0
        
        # 历史记录用于滞留检测
        self.pos_history = deque(maxlen=60)
        self.last_collision = False
        self.is_stuck = False  # 新增卡住状态记录
        
        # 记录雷达数据用于渲染
        self.last_lidar_dists = []

    def get_current_task_name(self):
        if self.task_idx < len(TASK_SEQUENCE):
            return TASK_SEQUENCE[self.task_idx]
        return None

    def _get_lidar_data(self):
        num_rays = 16
        max_range = 150.0
        lidar_distances = []
        for i in range(num_rays):
            angle = i * (2 * math.pi / num_rays)
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
            
            distance = max_range
            for step in range(1, int(max_range), 5):
                test_x = self.rx + ray_dx * step
                test_y = self.ry + ray_dy * step
                
                if test_x < 0 or test_x > WIDTH or test_y < 0 or test_y > HEIGHT:
                    distance = step
                    break
                    
                if any(obs.collidepoint(test_x, test_y) for obs in self.obstacles):
                    distance = step
                    break
            
            lidar_distances.append(distance / max_range)
        self.last_lidar_dists = lidar_distances
        return lidar_distances

    def _is_stagnant(self):
        if len(self.pos_history) < 60:
            return False
        positions = np.array(self.pos_history, dtype=np.float32)
        
        # 1. 紧凑方差检测：排查是否死锁在一个极小点微震
        std_sum = float(np.std(positions[:, 0]) + np.std(positions[:, 1]))
        if std_sum < 3.0:
            return True
            
        # 2. 净位移检测（解决原地转圈/反复横跳）：
        # 60帧约1秒，正常全速(250像素/秒)走直线位移应为250。如果首尾净位移极小，说明在打转。
        oldest = positions[0]
        newest = positions[-1]
        displacement = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1])
        if displacement < 10.0:
            return True
            
        return False

    def _front_sector_blocked(self, lidar_dists, vx, vy):
        if abs(vx) <= 1e-6 and abs(vy) <= 1e-6:
            return False
        move_angle = math.atan2(vy, vx)
        num_rays = len(lidar_dists)
        angle_step = 2 * math.pi / num_rays
        center_idx = int(round(move_angle / angle_step)) % num_rays
        front_sector = [lidar_dists[(center_idx + offset) % num_rays] for offset in [-1, 0, 1]]
        return min(front_sector) < 0.25

    def get_rl_obs(self, target_waypoint):
        """生成底层 RL 所需的 22 维基底向量"""
        tx, ty = target_waypoint
        dx = tx - self.rx
        dy = ty - self.ry
        distance = math.hypot(dx, dy)
        
        if distance > 1e-6:
            dir_x, dir_y = dx / distance, dy / distance
        else:
            dir_x, dir_y = 0.0, 0.0
            
        distance_norm = min(distance / 150.0, 1.0)
        lidar_obs = self._get_lidar_data()
        
        is_stagnant = float(self._is_stagnant())
        front_blocked = float(self._front_sector_blocked(lidar_obs, dir_x, dir_y))
        last_collision = float(self.last_collision)
        
        obs_1d = [dir_x, dir_y, distance_norm] + lidar_obs + [is_stagnant, front_blocked, last_collision]
        return np.array(obs_1d, dtype=np.float32)

    def step_physics(self, action):
        """应用 action 更新坐标并处理碰撞"""
        vx = action[0] * self.robot_vmax
        vy = action[1] * self.robot_vmax
        
        old_x, old_y = self.rx, self.ry
        self.last_collision = False
        BOUNCE = 0.8
        
        # X轴更新
        self.rx += vx * self.dt
        if self.rx < self.robot_radius or self.rx > WIDTH - self.robot_radius:
            self.rx = old_x - vx * self.dt * BOUNCE
            self.last_collision = True
        else:
            for obs in self.obstacles:
                closest_x = max(obs.left, min(self.rx, obs.right))
                closest_y = max(obs.top, min(self.ry, obs.bottom))
                if math.hypot(self.rx - closest_x, self.ry - closest_y) < self.robot_radius:
                    self.rx = old_x - vx * self.dt * BOUNCE
                    self.last_collision = True
                    break

        # Y轴更新
        self.ry += vy * self.dt
        if self.ry < self.robot_radius or self.ry > HEIGHT - self.robot_radius:
            self.ry = old_y - vy * self.dt * BOUNCE
            self.last_collision = True
        else:
            for obs in self.obstacles:
                closest_x = max(obs.left, min(self.rx, obs.right))
                closest_y = max(obs.top, min(self.ry, obs.bottom))
                if math.hypot(self.rx - closest_x, self.ry - closest_y) < self.robot_radius:
                    self.ry = old_y - vy * self.dt * BOUNCE
                    self.last_collision = True
                    break
                    
        self.pos_history.append((self.rx, self.ry))
        self.is_stuck = self._is_stagnant()  # 每步更新卡死状态

    def render(self, global_path, lookahead_wp):
        self.screen.fill(WHITE)
        
        # 画障碍物
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, BLACK, obs)
            
        # 画任务点
        active_task = self.get_current_task_name()
        completed_tasks = TASK_SEQUENCE[:self.task_idx]
        
        for name, pt in HARD_TASKS.items():
            if name not in TASK_SEQUENCE:
                continue
                
            color = RED 
            if name in completed_tasks:
                color = GREEN
            elif name == active_task:
                color = YELLOW
                
            px, py = int(pt[0]), int(pt[1])
            pygame.draw.circle(self.screen, color, (px, py), 25)
            # 黄色高亮光环
            if name == active_task:
                pygame.draw.circle(self.screen, (255, 140, 0), (px, py), 28, 3)
                
            text = self.font.render(name, True, BLACK if color == YELLOW else WHITE)
            self.screen.blit(text, text.get_rect(center=(px, py)))

        # 画 A* 路径
        if global_path and len(global_path) > 1:
            points = [(int(p[0]), int(p[1])) for p in global_path]
            pygame.draw.lines(self.screen, LIGHT_BLUE, False, points, 4)
            
        # 画稀疏前瞻引导点 (Waypoint)
        if lookahead_wp:
            pygame.draw.circle(self.screen, PINK, (int(lookahead_wp[0]), int(lookahead_wp[1])), 10)
            pygame.draw.circle(self.screen, (200, 0, 0), (int(lookahead_wp[0]), int(lookahead_wp[1])), 10, 2)

        # 画小车及雷达
        if hasattr(self, 'last_lidar_dists') and self.last_lidar_dists:
            for i, dist_norm in enumerate(self.last_lidar_dists):
                angle = i * (2 * math.pi / 16)
                dist = dist_norm * 150.0
                end_x = self.rx + math.cos(angle) * dist
                end_y = self.ry + math.sin(angle) * dist
                
                line_color = (255, 80, 80) if dist_norm < 1.0 else (200, 200, 200)
                pt_color = (255, 0, 0) if dist_norm < 1.0 else (150, 150, 150)
                
                pygame.draw.line(self.screen, line_color, (int(self.rx), int(self.ry)), (int(end_x), int(end_y)), 2)
                pygame.draw.circle(self.screen, pt_color, (int(end_x), int(end_y)), 3)

        # 画小车本体
        pygame.draw.circle(self.screen, BLUE, (int(self.rx), int(self.ry)), self.robot_radius)
        
        # 可视化：小车是否卡住
        if getattr(self, 'is_stuck', False):
            # 绘制红色的警示光环
            pygame.draw.circle(self.screen, RED, (int(self.rx), int(self.ry)), self.robot_radius + 5, 3)
            # 在左上角显示危险文本提示
            stuck_text = self.font.render("STUCK: REDUCING LOOKAHEAD!", True, RED)
            self.screen.blit(stuck_text, (20, 20))
        else:
            pygame.draw.circle(self.screen, WHITE, (int(self.rx), int(self.ry)), 3)
        
        pygame.display.flip()
        self.clock.tick(60)


def main():
    model_path = "导航结合RL运动/results/local_eval_best/best_model.zip"
    if not os.path.exists(model_path):
        print(f"找不到模型文件 {model_path}！请确保在运行此评估前已经训练出了模型。")
        model_exists = False
    else:
        model = PPO.load(model_path)
        model_exists = True
        print(f"已加载稀疏感知测试模型: {model_path}")

    env = MixedNavEnv()
    stacker = FrameStacker(shape_1d=22, n_stack=4)
    
    global_path = []
    lookahead_wp = (env.rx, env.ry)
    steps_since_replan = 0
    
    running = True
    
    # 初始化起点的第一个状态组
    dummy_obs_1d = np.zeros(22, dtype=np.float32)
    stacked_obs = stacker.reset(dummy_obs_1d)

    frames = []  # 存储录制帧

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        active_task_name = env.get_current_task_name()
        if not active_task_name:
            print("=== 所有任务圆满完成！终极迷宫挑战成功！ ===")
            pygame.time.delay(3000)
            break
            
        target_pos = HARD_TASKS[active_task_name]
        
        # 判断是否到达当前大目标
        dist_to_target = math.hypot(env.rx - target_pos[0], env.ry - target_pos[1])
        if dist_to_target < 25 + env.robot_radius: # 15 像素容差加半径
            print(f"到达 {active_task_name}！即将前往下一个。")
            env.task_idx += 1
            steps_since_replan = 999  # 强制立即刷新路径
            continue

        # 触发 A* 全局重规划 
        if steps_since_replan > 5:
            global_path = env.planner.plan((env.rx, env.ry), target_pos, HARD_OBSTACLES)
            steps_since_replan = 0
        steps_since_replan += 1
        
        # 动态调整 A* 引导：平时拉长前瞻距离逼迫微操，一旦发现自己卡住，就降低前瞻距离进行手把手引导！
        current_lookahead = 25.0 if getattr(env, 'is_stuck', False) else 50.0
        lookahead_wp = get_lookahead_waypoint((env.rx, env.ry), global_path, lookahead_dist=current_lookahead)

        # 构建底层特征输入并预测
        obs_1d = env.get_rl_obs(lookahead_wp)
        stacked_obs = stacker.append(obs_1d)
        
        if model_exists:
            action, _ = model.predict(np.expand_dims(stacked_obs, axis=0), deterministic=True)
            action = action[0]
        else:
            # 没有模型时，强制保持静止仅展示 A* 探路
            action = [0.0, 0.0]

        env.step_physics(action)
        env.render(global_path, lookahead_wp)

        if RECORD_GIF:
            # 获取屏幕像素，Pygame 的格式是 (Width, Height, RGB)
            frame = pygame.surfarray.array3d(env.screen)
            # 转置为 imageio 需要的 (Height, Width, RGB)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)

    pygame.quit()
    
    if RECORD_GIF and frames:
        print(f"\n正在生成 GIF 录像并保存至 {GIF_SAVE_PATH}，请稍候...")
        # 适当抽帧以减小体积，保存为30fps
        imageio.mimsave(GIF_SAVE_PATH, frames[::2], fps=30)
        print(f"GIF 录像保存完毕！存储路径: {os.path.abspath(GIF_SAVE_PATH)}")

    sys.exit()

if __name__ == "__main__":
    main()
