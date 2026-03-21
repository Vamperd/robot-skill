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
GIF_SAVE_PATH = "multi_agent_extreme_result.gif"

# 终极地图障碍物 (EXTREME_OBSTACLES)
EXTREME_OBSTACLES = [
    # 外围框架
    pygame.Rect(0, 0, 800, 20), pygame.Rect(0, 580, 800, 20),
    pygame.Rect(0, 0, 20, 600), pygame.Rect(780, 0, 20, 600),
    # 迷宫内部：狭窄的中央 H 型交叉通道
    pygame.Rect(200, 150, 400, 40), # 上横墙
    pygame.Rect(200, 400, 400, 40), # 下横墙
    pygame.Rect(380, 190, 40, 210), # 中央极窄竖向通道
    # 四个死胡同挡板
    pygame.Rect(100, 250, 100, 40),
    pygame.Rect(600, 250, 100, 40),
]

EXTREME_TASKS = {
    "Task A": (80, 80),   # 左上房间
    "Task B": (720, 80),  # 右上房间
    "Task C": (80, 520),  # 左下房间
    "Task D": (720, 520), # 右下房间
    "Task E": (400, 80),  # 顶部正中
    "Task F": (400, 520),  # 底部正中
    "Sync 1": (200, 330),  # H型通道左侧空地
    "Sync 2": (600, 330)   # H型通道右侧空地
}

# 定义需要多车同时到达才能解锁的任务
COOP_TASKS = {"Sync 1", "Sync 2"}


def get_work_frames(robot_name, task_name):
    # 基础任务耗时 (以帧为单位，60帧 = 1秒)
    base_times = {
        "Task A": 180, "Task B": 120, "Task C": 240,
        "Task D": 90,  "Task E": 300, "Task F": 150,
        "Sync 1": 180, "Sync 2": 180
    }
    # 机器人的工作效率倍率 (差异化)
    # OrangeBot 干活极快(0.5倍时间)，CyanBot 干活极慢(2.5倍时间)
    multipliers = {
        "BlueBot": 1.0, "PurpleBot": 1.5,
        "OrangeBot": 0.5, "CyanBot": 2.5
    }
    return int(base_times.get(task_name, 120) * multipliers.get(robot_name, 1.0))

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
        self.is_waiting = False
        self.wait_frames = 0  # 记录等待队友的帧数
        self.task_progress = 0  # 当前任务的完成进度（帧）
        self.is_finished = False  # 标记是否完成所有终极任务
        
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)

    def get_current_task_pos(self):
        if self.task_idx < len(self.task_sequence):
            task_name = self.task_sequence[self.task_idx]
            return EXTREME_TASKS.get(task_name)
        return None

    def get_current_task_name(self):
        if self.task_idx < len(self.task_sequence):
            return self.task_sequence[self.task_idx]
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
                for obs in EXTREME_OBSTACLES:
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
        
        # 【引力消除死区】：进入目标半径后抹除引力，防止原地画圈抽搐
        if distance > 15.0:
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
        # 1. EMA 低通滤波
        alpha = 0.4
        smoothed_action = alpha * np.array(action) + (1.0 - alpha) * self.last_action
        self.last_action = smoothed_action

        # 2. 智能驻车制动 (Smart Parking Brake)
        # 获取当前是否处于干活或等待状态
        is_parking = (getattr(self, 'task_progress', 0) > 0) or getattr(self, 'is_waiting', False) or getattr(self, 'is_finished', False)
        # 获取周围最危险的雷达距离 (0~1.0)
        min_lidar = min(self.last_lidar_dists) if getattr(self, 'last_lidar_dists', []) else 1.0

        # 如果神经网络输出力极小，或者（小车正在驻车 且 周围没有贴脸的队友(雷达>0.35)）
        if np.linalg.norm(smoothed_action) < 0.15 or (is_parking and min_lidar > 0.35):
            smoothed_action = np.array([0.0, 0.0])
            self.last_action = smoothed_action  # 彻底清空残余动量，像拉了手刹一样稳

        vx = smoothed_action[0] * self.robot_vmax
        vy = smoothed_action[1] * self.robot_vmax
        
        old_x, old_y = self.rx, self.ry
        self.last_collision = False
        BOUNCE = 0.8
        
        # 定义一个通用的碰撞检测闭包
        def check_collisions(rx, ry):
            # 查墙壁静态矩形
            for obs in EXTREME_OBSTACLES:
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
    for obs in EXTREME_OBSTACLES:
        pygame.draw.rect(screen, BLACK, obs)
        
    # 2. 渲染所有存在的任务点 (统一种颜色，或者标识出是谁的任务)
    for name, pt in EXTREME_TASKS.items():
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
        if getattr(robot, 'is_waiting', False):
            # 画一个刺眼的黄色外圈
            pygame.draw.circle(screen, YELLOW, (int(robot.rx), int(robot.ry)), robot.robot_radius + 6, 4)
            # 在小车旁边显示等待了多少秒
            wait_sec = robot.wait_frames // 60
            wait_text = font.render(f"Waiting: {wait_sec}s", True, RED)
            screen.blit(wait_text, (int(robot.rx) - 30, int(robot.ry) - 40))
        elif getattr(robot, 'is_stuck', False):
            pygame.draw.circle(screen, RED, (int(robot.rx), int(robot.ry)), robot.robot_radius + 4, 3)
        else:
            pygame.draw.circle(screen, WHITE, (int(robot.rx), int(robot.ry)), 2)

        # 绘制干活进度条
        task_name = robot.get_current_task_name()
        if task_name and not getattr(robot, 'is_waiting', False):
            req_frames = get_work_frames(robot.name, task_name)
            if 0 < robot.task_progress < req_frames:
                pct = robot.task_progress / req_frames
                bar_w = 40
                bar_h = 6
                bar_x = int(robot.rx) - bar_w // 2
                bar_y = int(robot.ry) - robot.robot_radius - 15
                # 黑框
                pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_w, bar_h), 1)
                # 绿色填充
                pygame.draw.rect(screen, GREEN, (bar_x + 1, bar_y + 1, int((bar_w - 2) * pct), bar_h - 2))

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
    pygame.display.set_caption("Extreme MARL Nav - Traffic Deadlock Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18, bold=True)

    robots = [
        # BlueBot (左上起): 很快就能到 Sync 1，然后开启漫长的挂机等待。
        Robot("BlueBot", EXTREME_TASKS["Task A"], (50, 150, 255), (150, 200, 255),
              ["Task C", "Sync 1", "Task D", "Task E"]),

        # PurpleBot (右上起): 需要跑遍大半个地图，做3个前置任务，最后才去 Sync 1 救 BlueBot。
        Robot("PurpleBot", EXTREME_TASKS["Task B"], (180, 50, 255), (220, 150, 255),
              ["Task E", "Task F", "Task C", "Sync 1", "Task A"]),

        # OrangeBot (右下起): 开局直接去 Sync 2 挂机，它将成为全场的“毒瘤路障”。
        Robot("OrangeBot", EXTREME_TASKS["Task D"], (255, 150, 50), (255, 200, 150),
              ["Sync 2", "Task B", "Task A"]),

        # CyanBot (左下起): 疯狂全图游走，沿途极大可能撞飞挂机的 OrangeBot，最后才去 Sync 2。
        Robot("CyanBot", EXTREME_TASKS["Task C"], (50, 255, 255), (150, 255, 255),
              ["Task F", "Task E", "Task B", "Sync 2", "Task D"])
    ]
    
    # 【必须保留错峰】：
    robots[0].steps_since_replan = 15
    robots[1].steps_since_replan = 11
    robots[2].steps_since_replan = 7
    robots[3].steps_since_replan = 3

    running = True
    frames = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 阶段 1：扫描并累加工作进度 ---
        completed_robots = set()
        for robot in robots:
            target_pos = robot.get_current_task_pos()
            if not target_pos:
                continue

            req_frames = get_work_frames(robot.name, robot.get_current_task_name())
            dist = math.hypot(robot.rx - target_pos[0], robot.ry - target_pos[1])

            # 只要小车停在目标点范围内，就开始“干活”
            if dist < 25 + robot.robot_radius:
                robot.task_progress += 1

            # 如果进度达标，标记为“已完成本职工作”
            if robot.task_progress >= req_frames:
                completed_robots.add(robot)

        # --- 阶段 2：处理解锁与等待逻辑（协同锁）---
        tasks_to_advance = set()
        for robot in robots:
            task_name = robot.get_current_task_name()
            if not task_name:
                continue

            robot.is_waiting = False

            if task_name not in COOP_TASKS:
                # 单机任务：自己干完就直接走
                if robot in completed_robots:
                    tasks_to_advance.add(robot)
                    robot.wait_frames = 0
            else:
                # 协同任务：要求分配到该任务的所有队友都干完活
                assigned_robots = [r for r in robots if r.get_current_task_name() == task_name]
                if all(r in completed_robots for r in assigned_robots):
                    if robot in completed_robots:
                        tasks_to_advance.add(robot)
                else:
                    # 如果自己干完了，但队友还没来或者还没干完活，进入挂机等待！
                    if robot in completed_robots:
                        robot.is_waiting = True
                        robot.wait_frames += 1

        # --- 阶段 3：执行解锁并清零进度 ---
        for robot in tasks_to_advance:
            task_name = robot.get_current_task_name()
            robot.task_idx += 1
            robot.steps_since_replan = 999
            robot.is_waiting = False
            robot.wait_frames = 0  # 解锁后清零
            robot.task_progress = 0  # 清零！准备做下一个任务
            prefix = "【协同同步】" if task_name in COOP_TASKS else "【单机完成】"
            print(f"[{robot.name}] {prefix} {task_name} 达成！前往下一站。")
                
        all_finished = True
        
        # --- 阶段 4：生成底层 RL 观测 ---
        obs_batch = []
        for i, robot in enumerate(robots):
            target_pos = robot.get_current_task_pos()
            if not target_pos:
                # 【修复核心1】：任务完成的车辆不再变成死砖头！
                # 把它当前脚下的位置设为引力点，更新雷达，保持 RL 大脑活跃
                robot.is_finished = True
                robot.lookahead_wp = (robot.rx, robot.ry)
                other_robots = [r for j, r in enumerate(robots) if j != i]
                stacked_obs = robot.get_rl_obs_and_stack(other_robots)
                obs_batch.append(stacked_obs)
                continue
            
            all_finished = False
            robot.is_finished = False

            robot.steps_since_replan += 1
            if robot.steps_since_replan > 5:
                robot.global_path = robot.planner.plan((robot.rx, robot.ry), target_pos, [(obs.x, obs.y, obs.w, obs.h) for obs in EXTREME_OBSTACLES])
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
            # 【修复核心2】：无论是否有任务，所有车辆都必须执行物理步进，维持弹性避让能力！
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
