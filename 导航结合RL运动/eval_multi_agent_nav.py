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
PURPLE = (180, 50, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
YELLOW = (255, 200, 0)
BLACK = (30, 30, 30)
LIGHT_BLUE = (150, 200, 255)
LIGHT_PURPLE = (220, 150, 255)
PINK = (255, 105, 180)

# ====== 录制配置 ======
RECORD_GIF = True
GIF_SAVE_PATH = "multi_agent_nav_result.gif"

# 终极泛化深渊地图
HARD_OBSTACLES = [
    pygame.Rect(250, 0, 40, 480),
    pygame.Rect(450, 120, 40, 480),
    pygame.Rect(290, 280, 110, 40),
    pygame.Rect(0, 400, 150, 40),
    pygame.Rect(150, 400, 40, 140),
    pygame.Rect(580, 360, 220, 40),
    pygame.Rect(580, 480, 220, 40),
    pygame.Rect(760, 400, 40, 80)
]

HARD_TASKS = {
    "Task A": (680, 60), 
    "Task B": (80, 500), 
    "Task C": (680, 420),
    "Task D": (60, 60)
}

class FrameStacker:
    """手动实现 4 帧堆叠"""
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
        return np.concatenate(self.frames)

class Robot:
    def __init__(self, name, start_pos, color, path_color, task_sequence):
        self.name = name
        self.rx, self.ry = start_pos
        self.color = color
        self.path_color = path_color
        self.robot_radius = 15
        self.robot_vmax = 250.0
        self.dt = 1.0 / 60.0
        
        self.task_sequence = task_sequence
        self.task_idx = 0
        
        self.planner = AStarPlanner(width=WIDTH, height=HEIGHT, resolution=10, robot_radius=self.robot_radius, margin=5)
        self.stacker = FrameStacker(shape_1d=27, n_stack=4)
        
        # 初始第一帧占位
        self.stacker.reset(np.zeros(27, dtype=np.float32))
        
        self.global_path = []
        self.lookahead_wp = (self.rx, self.ry)
        self.steps_since_replan = 999 
        
        self.pos_history = deque(maxlen=60)
        self.last_collision = False
        self.is_stuck = False
        self.last_lidar_dists = []

    def get_current_task_pos(self):
        if self.task_idx < len(self.task_sequence):
            task_name = self.task_sequence[self.task_idx]
            return HARD_TASKS.get(task_name)
        return None

    def _is_stagnant(self):
        if len(self.pos_history) < 60:
            return False
        positions = np.array(self.pos_history, dtype=np.float32)
        
        std_sum = float(np.std(positions[:, 0]) + np.std(positions[:, 1]))
        if std_sum < 5.0:
            return True
            
        oldest = positions[0]
        newest = positions[-1]
        displacement = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1])
        if displacement < 30.0:
            return True
        return False

    def _get_lidar_data(self, other_robots):
        """包含多机实体雷达检测的 raymarching"""
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
                
                # 墙壁边界
                if test_x < 0 or test_x > WIDTH or test_y < 0 or test_y > HEIGHT:
                    distance = step
                    break
                
                # 静态障碍物检测
                hit = False
                for obs in HARD_OBSTACLES:
                    if obs.collidepoint(test_x, test_y):
                        hit = True
                        break
                if hit:
                    distance = step
                    break
                    
                # 【多机感知核心】动态其他机器人避障检测
                for other in other_robots:
                    dist_to_other = math.hypot(test_x - other.rx, test_y - other.ry)
                    if dist_to_other < other.robot_radius:
                        hit = True
                        break
                if hit:
                    distance = step
                    break
                    
            lidar_distances.append(distance / max_range)
            
        self.last_lidar_dists = lidar_distances
        return lidar_distances

    def _front_sector_blocked(self, lidar_dists, vx, vy):
        if abs(vx) <= 1e-6 and abs(vy) <= 1e-6:
            return False
        move_angle = math.atan2(vy, vx)
        num_rays = len(lidar_dists)
        angle_step = 2 * math.pi / num_rays
        center_idx = int(round(move_angle / angle_step)) % num_rays
        front_sector = [lidar_dists[(center_idx + offset) % num_rays] for offset in [-1, 0, 1]]
        return min(front_sector) < 0.25

    def get_rl_obs_and_stack(self, other_robots):
        if not self.global_path:
            return self.stacker.get_stacked_obs()
            
        tx, ty = self.lookahead_wp
        dx = tx - self.rx
        dy = ty - self.ry
        distance = math.hypot(dx, dy)
        
        if distance > 1e-6:
            dir_x, dir_y = dx / distance, dy / distance
        else:
            dir_x, dir_y = 0.0, 0.0
            
        distance_norm = min(distance / 150.0, 1.0)
        lidar_obs = self._get_lidar_data(other_robots)
        
        is_stagnant = float(self._is_stagnant())
        front_blocked = float(self._front_sector_blocked(lidar_obs, dir_x, dir_y))
        last_collision = float(self.last_collision)
        
        # 伪造老模型需要的缺失维度
        remaining_time = 0.5  # 伪装一直处于中期
        dfa_one_hot = [1.0, 0.0, 0.0, 0.0]  # 伪装处于初始任务状态
        
        # 严格按照老模型 27 维的观测顺序拼接：
        # 基础(5维) + DFA(4维) + 雷达(16维) + 停滞与遮挡(2维)
        base_obs = [dir_x, dir_y, distance_norm, last_collision, remaining_time]
        obs_1d = base_obs + dfa_one_hot + lidar_obs + [is_stagnant, front_blocked]
        
        return self.stacker.append(np.array(obs_1d, dtype=np.float32))

    def step_physics(self, action, other_robots):
        vx = action[0] * self.robot_vmax
        vy = action[1] * self.robot_vmax
        
        old_x, old_y = self.rx, self.ry
        self.last_collision = False
        BOUNCE = 0.8
        
        # 定义一个通用的碰撞检测闭包
        def check_collisions(rx, ry):
            # 查墙壁静态矩形
            for obs in HARD_OBSTACLES:
                closest_x = max(obs.left, min(rx, obs.right))
                closest_y = max(obs.top, min(ry, obs.bottom))
                if math.hypot(rx - closest_x, ry - closest_y) < self.robot_radius:
                    return True
            # 查其他机器人圆碰圆
            for other in other_robots:
                if math.hypot(rx - other.rx, ry - other.ry) < self.robot_radius + other.robot_radius:
                    return True
            return False

        self.rx += vx * self.dt
        if self.rx < self.robot_radius or self.rx > WIDTH - self.robot_radius or check_collisions(self.rx, self.ry):
            self.rx = old_x - vx * self.dt * BOUNCE
            self.last_collision = True

        self.ry += vy * self.dt
        if self.ry < self.robot_radius or self.ry > HEIGHT - self.robot_radius or check_collisions(self.rx, self.ry):
            self.ry = old_y - vy * self.dt * BOUNCE
            self.last_collision = True
                    
        self.pos_history.append((self.rx, self.ry))
        self.is_stuck = self._is_stagnant()

def render_scene(screen, font, robots):
    screen.fill(WHITE)
    
    # 1. 障碍物
    for obs in HARD_OBSTACLES:
        pygame.draw.rect(screen, BLACK, obs)
        
    # 2. 渲染所有存在的任务点 (统一种颜色，或者标识出是谁的任务)
    for name, pt in HARD_TASKS.items():
        px, py = int(pt[0]), int(pt[1])
        pygame.draw.circle(screen, GREEN, (px, py), 20)
        pygame.draw.circle(screen, (0, 150, 0), (px, py), 20, 2)
        text = font.render(name[-1], True, WHITE) # 简化只画字母 A, B, C...
        screen.blit(text, text.get_rect(center=(px, py)))

    # 3. 渲染每个机器人的专属路径、雷达和本体
    for robot in robots:
        # A* 专属路径
        if robot.global_path and len(robot.global_path) > 1:
            points = [(int(p[0]), int(p[1])) for p in robot.global_path]
            pygame.draw.lines(screen, robot.path_color, False, points, 3)
            
        # 前瞻寻路引导点
        pygame.draw.circle(screen, robot.path_color, (int(robot.lookahead_wp[0]), int(robot.lookahead_wp[1])), 8)

        # 激光雷达射线
        if robot.last_lidar_dists:
            for i, dist_norm in enumerate(robot.last_lidar_dists):
                angle = i * (2 * math.pi / 16)
                dist = dist_norm * 150.0
                end_x = robot.rx + math.cos(angle) * dist
                end_y = robot.ry + math.sin(angle) * dist
                
                # 危险距离标红警告，安全则灰掉
                line_color = (255, 80, 80) if dist_norm < 1.0 else (200, 200, 200)
                pt_color = (255, 0, 0) if dist_norm < 1.0 else (150, 150, 150)
                
                pygame.draw.line(screen, line_color, (int(robot.rx), int(robot.ry)), (int(end_x), int(end_y)), 1)
                pygame.draw.circle(screen, pt_color, (int(end_x), int(end_y)), 2)

        # 小车本体
        pygame.draw.circle(screen, robot.color, (int(robot.rx), int(robot.ry)), robot.robot_radius)
        if robot.is_stuck:
            pygame.draw.circle(screen, RED, (int(robot.rx), int(robot.ry)), robot.robot_radius + 4, 3)
        else:
            pygame.draw.circle(screen, WHITE, (int(robot.rx), int(robot.ry)), 2)

def main():
    model_path = "无导航纯RL底层运动器/results/generalization_eval_best_vel_punishment_狭窄距离，效果最好，可过U形弯/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "ppo_local_planner.zip"
    
    if not os.path.exists(model_path):
        print(f"找不到共享使用的避障模型！请先训练。")
        model_exists = False
    else:
        model = PPO.load(model_path)
        model_exists = True
        print(f"成功加载底层共享神经网络策略 (MARL): {model_path}")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Multi-Agent Avoidance (Shared PPO + Individual A*)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18, bold=True)

    # 实例化两台智能跑车，各自有自己的起止任务队列
    robots = [
        Robot("BlueBot", HARD_TASKS["Task D"], BLUE, LIGHT_BLUE, ["Task C", "Task A"]),
        Robot("PurpleBot", HARD_TASKS["Task C"], PURPLE, LIGHT_PURPLE, ["Task A", "Task D"])
    ]

    running = True
    frames = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        all_finished = True
        
        # 1. 第一阶段：所有机器人收集环境信息，进行高层规划，准备神经网络观测数据
        obs_batch = []
        for i, robot in enumerate(robots):
            target_pos = robot.get_current_task_pos()
            if not target_pos:
                # 已经完成所有任务的车发呆
                obs_batch.append(robot.stacker.get_stacked_obs())
                continue
            
            all_finished = False
            
            dist_to_target = math.hypot(robot.rx - target_pos[0], robot.ry - target_pos[1])
            if dist_to_target < 25 + robot.robot_radius:
                print(f"[{robot.name}] 到达了当前目标！前往下一个任务...")
                robot.task_idx += 1
                robot.steps_since_replan = 999
                
                # 重新验证是否还有下一个任务
                target_pos = robot.get_current_task_pos()
                if not target_pos:
                    obs_batch.append(robot.stacker.get_stacked_obs())
                    continue

            robot.steps_since_replan += 1
            if robot.steps_since_replan > 15:
                robot.global_path = robot.planner.plan((robot.rx, robot.ry), target_pos, [(obs.x, obs.y, obs.w, obs.h) for obs in HARD_OBSTACLES])
                robot.steps_since_replan = 0
            
            # 多机防卡死引力拉扯
            current_lookahead = 40.0 if robot.is_stuck else 60.0
            robot.lookahead_wp = get_lookahead_waypoint((robot.rx, robot.ry), robot.global_path, lookahead_dist=current_lookahead)
            
            # 互相告知对方的存在，产生交叉的雷达遮挡数据并入栈
            other_robots = [r for j, r in enumerate(robots) if j != i]
            stacked_obs = robot.get_rl_obs_and_stack(other_robots)
            obs_batch.append(stacked_obs)

        if all_finished:
            print("\n=== MARL 跑图任务全员完成！===")
            pygame.time.delay(2000)
            break

        # 2. 第二阶段：共享模型批处理。把多个观测揉成 Batch 一次性推理。
        if model_exists and not all_finished:
            obs_array = np.array(obs_batch)
            actions, _ = model.predict(obs_array, deterministic=True)
        else:
            actions = np.zeros((len(robots), 2))

        # 3. 第三阶段：所有机器人各自应用推演出的物理动作
        for i, robot in enumerate(robots):
            if robot.get_current_task_pos(): # 还在工作的话，处理走位和全碰撞
                other_robots = [r for j, r in enumerate(robots) if j != i]
                robot.step_physics(actions[i], other_robots)

        # 4. 渲染发光
        render_scene(screen, font, robots)
        pygame.display.flip()
        clock.tick(60)

        if RECORD_GIF:
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)

    pygame.quit()
    
    if RECORD_GIF and frames:
        print(f"\n正在生成多机 GIF 录像并保存至 {GIF_SAVE_PATH}...")
        imageio.mimsave(GIF_SAVE_PATH, frames[::2], fps=30)
        print(f"GIF 录像保存完毕！存储路径: {os.path.abspath(GIF_SAVE_PATH)}")

    sys.exit()

if __name__ == "__main__":
    main()
