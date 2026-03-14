import pygame
import math
import sys

# 初始化
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LTLf RL - Reward Shaping Env")
clock = pygame.time.Clock()

# 颜色定义
WHITE = (240, 240, 240)
BLUE = (50, 150, 255)     # 机器人
RED = (255, 100, 100)     # 任务点
GREEN = (100, 255, 100)   # 达成状态
DARK_GRAY = (50, 50, 50)
BLACK = (30, 30, 30)      # 障碍物

class Obstacle:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surface):
        pygame.draw.rect(surface, BLACK, self.rect)

    def is_colliding_with_circle(self, cx, cy, radius):
        """AABB 矩形与圆形的碰撞检测"""
        # 找到矩形上距离圆心最近的点
        closest_x = max(self.rect.left, min(cx, self.rect.right))
        closest_y = max(self.rect.top, min(cy, self.rect.bottom))
        
        # 计算最近点与圆心的距离
        distance = math.hypot(cx - closest_x, cy - closest_y)
        return distance < radius

class ContinuousEnv:
    def __init__(self):
        self.robot_radius = 15
        self.robot_vmax = 250.0
        
        # 初始化实体
        self.reset()
        
        # 定义障碍物
        self.obstacles = [
            Obstacle(300, 150, 50, 300),
            Obstacle(500, 100, 200, 50),
            Obstacle(100, 400, 250, 50),
            Obstacle(600, 350, 50, 200)
        ]

        # 奖励机制参数
        self.REWARD_GOAL = 100.0       # 达到目标的巨大奖励
        self.REWARD_COLLISION = -2.0   # 每次碰撞的惩罚
        self.REWARD_TIME_STEP = -0.05  # 时间流逝惩罚（鼓励速战速决）
        self.POTENTIAL_SCALE = 0.5     # 势能奖励的缩放系数

    def reset(self):
        self.rx, self.ry = 100.0, 100.0
        self.tx, self.ty = 700.0, 500.0
        self.task_radius = 25
        self.task_completed = False
        
        self.cumulative_reward = 0.0
        self.step_reward = 0.0
        self.prev_distance = math.hypot(self.rx - self.tx, self.ry - self.ty)

    def step(self, action_vx, action_vy, dt):
        """
        核心的 MDP 状态转移与奖励计算函数
        """
        if self.task_completed:
            return 0.0 # 任务完成后不再给奖励

        self.step_reward = 0.0
        
        # 1. 施加时间惩罚
        self.step_reward += self.REWARD_TIME_STEP

        # 保存上一步位置用于碰撞回退
        old_x, old_y = self.rx, self.ry

        # 积分更新坐标
        self.rx += action_vx * dt
        self.ry += action_vy * dt

        # 2. 边界碰撞检测
        collided = False
        if self.rx < self.robot_radius or self.rx > WIDTH - self.robot_radius:
            self.rx = old_x
            collided = True
        if self.ry < self.robot_radius or self.ry > HEIGHT - self.robot_radius:
            self.ry = old_y
            collided = True

        # 3. 障碍物碰撞检测
        for obs in self.obstacles:
            if obs.is_colliding_with_circle(self.rx, self.ry, self.robot_radius):
                self.rx, self.ry = old_x, old_y  # 物理回退，无法穿透
                collided = True
                break
        
        if collided:
            self.step_reward += self.REWARD_COLLISION

        # 4. 势能奖励计算 (Potential-based Reward)
        current_distance = math.hypot(self.rx - self.tx, self.ry - self.ty)
        # 核心公式：势能差 = 昨天的距离 - 今天的距离。靠近为正，远离为负。
        potential_reward = (self.prev_distance - current_distance) * self.POTENTIAL_SCALE
        self.step_reward += potential_reward
        self.prev_distance = current_distance

        # 5. 目标达成判定
        if current_distance <= (self.robot_radius + self.task_radius):
            self.task_completed = True
            self.step_reward += self.REWARD_GOAL

        self.cumulative_reward += self.step_reward
        return self.step_reward

    def render(self, screen, font):
        screen.fill(WHITE)
        
        # 画障碍物
        for obs in self.obstacles:
            obs.draw(screen)

        # 画任务点
        task_color = GREEN if self.task_completed else RED
        pygame.draw.circle(screen, task_color, (int(self.tx), int(self.ty)), self.task_radius)
        task_text = font.render("Target", True, WHITE)
        screen.blit(task_text, task_text.get_rect(center=(int(self.tx), int(self.ty))))

        # 画机器人
        robot_color = GREEN if self.task_completed else BLUE
        pygame.draw.circle(screen, robot_color, (int(self.rx), int(self.ry)), self.robot_radius)
        pygame.draw.circle(screen, WHITE, (int(self.rx), int(self.ry)), 3)

        # 渲染奖励仪表盘
        info_lines = [
            f"Step Reward: {self.step_reward:+.2f}",
            f"Cumulative Reward: {self.cumulative_reward:+.2f}",
            f"Distance to Target: {self.prev_distance:.1f}",
            "Status: " + ("Success!" if self.task_completed else "Running...")
        ]
        
        for i, text in enumerate(info_lines):
            # 颜色逻辑：正奖励绿色，负奖励红色
            color = DARK_GRAY
            if i < 2:
                value = self.step_reward if i == 0 else self.cumulative_reward
                color = (20, 180, 20) if value > 0 else (220, 20, 20) if value < 0 else DARK_GRAY
            
            surf = font.render(text, True, color)
            screen.blit(surf, (10, 10 + i * 25))

def main():
    env = ContinuousEnv()
    font = pygame.font.SysFont("arial", 20, bold=True)
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 按 R 键重置环境
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

        # 键盘指令映射到动作空间
        keys = pygame.key.get_pressed()
        vx, vy = 0.0, 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:  vx = -env.robot_vmax
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: vx =  env.robot_vmax
        if keys[pygame.K_UP] or keys[pygame.K_w]:    vy = -env.robot_vmax
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:  vy =  env.robot_vmax

        # 如果对角线移动，进行速度归一化，防止斜向超速
        if vx != 0 and vy != 0:
            vx *= 0.7071
            vy *= 0.7071

        # 环境单步更新并获取奖励
        env.step(vx, vy, dt)

        # 渲染
        env.render(screen, font)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()