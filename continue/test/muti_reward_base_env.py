import pygame
import math
import sys

# 初始化
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LTLf RL - Product MDP & Local DFA Env")
clock = pygame.time.Clock()

# 颜色定义
WHITE = (240, 240, 240)
BLUE = (50, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
DARK_GRAY = (50, 50, 50)
BLACK = (30, 30, 30)
YELLOW = (255, 200, 0) # 用于高亮当前 DFA 激活的目标

class Obstacle:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surface):
        pygame.draw.rect(surface, BLACK, self.rect)

    def is_colliding_with_circle(self, cx, cy, radius):
        closest_x = max(self.rect.left, min(cx, self.rect.right))
        closest_y = max(self.rect.top, min(cy, self.rect.bottom))
        distance = math.hypot(cx - closest_x, cy - closest_y)
        return distance < radius

class TaskPoint:
    def __init__(self, x, y, name):
        self.x, self.y = float(x), float(y)
        self.radius = 25
        self.name = name

class LocalDFA:
    """
    硬编码的本地确定性有限自动机
    对应的 LTLf 逻辑: 最终到达A，然后再到达B，然后再到达C
    """
    def __init__(self):
        self.state = 0
        self.accepting_state = 3

    def step(self, triggered_task_name):
        # 状态转移逻辑
        if self.state == 0 and triggered_task_name == "Task A":
            self.state = 1
            return True # 发生跃迁
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

class ProductMDPEnv:
    def __init__(self):
        self.robot_radius = 15
        self.robot_vmax = 250.0
        
        self.obstacles = [
            Obstacle(200, 150, 50, 300),
            Obstacle(450, 250, 200, 50),
            Obstacle(500, 400, 50, 200)
        ]
        
        self.tasks = {
            "Task A": TaskPoint(100, 500, "Task A"),
            "Task B": TaskPoint(400, 100, "Task B"),
            "Task C": TaskPoint(700, 500, "Task C")
        }

        # 奖励机制
        self.REWARD_GOAL = 100.0      # 达成最终状态
        self.REWARD_TRANSITION = 20.0 # 达成中间 DFA 状态
        self.REWARD_COLLISION = -2.0  
        self.REWARD_TIME_STEP = -0.05 
        self.POTENTIAL_SCALE = 0.5    

        self.reset()

    def reset(self):
        self.rx, self.ry = 100.0, 100.0
        self.dfa = LocalDFA()
        
        self.cumulative_reward = 0.0
        self.step_reward = 0.0
        self._update_potential_anchor() # 重置势能锚点

    def _update_potential_anchor(self):
        """更新当前势能场的中心坐标为 DFA 激活的下一个目标"""
        target_name = self.dfa.get_current_target_name()
        if target_name:
            target = self.tasks[target_name]
            self.prev_distance = math.hypot(self.rx - target.x, self.ry - target.y)

    def get_observation(self):
        """
        未来喂给 PPO 神经网络的 Observation
        结合了连续物理状态和离散逻辑状态 (Product MDP)
        """
        # 在实际训练中，你可以把 dfa.state 转换成 One-hot 编码
        return [self.rx, self.ry, self.dfa.state]

    def step(self, action_vx, action_vy, dt):
        if self.dfa.state == self.dfa.accepting_state:
            return 0.0 

        self.step_reward = self.REWARD_TIME_STEP
        old_x, old_y = self.rx, self.ry

        self.rx += action_vx * dt
        self.ry += action_vy * dt

        # 碰撞检测
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
        
        if collided:
            self.step_reward += self.REWARD_COLLISION

        # --- 核心：Product MDP 的状态转移与动态势能 ---
        target_name = self.dfa.get_current_target_name()
        if target_name:
            target = self.tasks[target_name]
            current_distance = math.hypot(self.rx - target.x, self.ry - target.y)
            
            # 动态势能引导
            potential_reward = (self.prev_distance - current_distance) * self.POTENTIAL_SCALE
            self.step_reward += potential_reward
            self.prev_distance = current_distance

            # DFA 跃迁判定 (Atomic Proposition 满足)
            if current_distance <= (self.robot_radius + target.radius):
                # 触发状态转移
                if self.dfa.step(target_name):
                    if self.dfa.state == self.dfa.accepting_state:
                        self.step_reward += self.REWARD_GOAL
                    else:
                        self.step_reward += self.REWARD_TRANSITION
                        # 【重要】DFA 状态变了，目标变了，必须重置势能锚点防止奖励跳变！
                        self._update_potential_anchor()

        self.cumulative_reward += self.step_reward
        return self.step_reward

    def render(self, screen, font):
        screen.fill(WHITE)
        
        for obs in self.obstacles:
            obs.draw(screen)

        active_target = self.dfa.get_current_target_name()
        
        for name, task in self.tasks.items():
            # 已完成的任务变绿，未完成变红
            color = RED
            # 如果当前任务的序号小于自动机状态，说明已完成
            task_idx = ["Task A", "Task B", "Task C"].index(name)
            if task_idx < self.dfa.state:
                color = GREEN
                
            pygame.draw.circle(screen, color, (int(task.x), int(task.y)), task.radius)
            
            # 高亮当前 DFA 激活的任务
            if name == active_target:
                pygame.draw.circle(screen, YELLOW, (int(task.x), int(task.y)), task.radius + 5, 3)

            text = font.render(name, True, WHITE)
            screen.blit(text, text.get_rect(center=(int(task.x), int(task.y))))

        pygame.draw.circle(screen, BLUE, (int(self.rx), int(self.ry)), self.robot_radius)
        pygame.draw.circle(screen, WHITE, (int(self.rx), int(self.ry)), 3)

        info_lines = [
            f"DFA State: {self.dfa.state} (Target: {active_target})",
            f"Step Reward: {self.step_reward:+.2f}",
            f"Cumulative Reward: {self.cumulative_reward:+.2f}",
            "Status: " + ("Success!" if self.dfa.state == self.dfa.accepting_state else "Running...")
        ]
        
        for i, text in enumerate(info_lines):
            color = DARK_GRAY
            if i in [1, 2]:
                val = self.step_reward if i == 1 else self.cumulative_reward
                color = (20, 180, 20) if val > 0 else (220, 20, 20) if val < 0 else DARK_GRAY
            if i == 0: color = (0, 0, 255) # 强调 DFA 状态
            
            surf = font.render(text, True, color)
            screen.blit(surf, (10, 10 + i * 25))

def main():
    env = ProductMDPEnv()
    font = pygame.font.SysFont("arial", 20, bold=True)
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

        keys = pygame.key.get_pressed()
        vx, vy = 0.0, 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:  vx = -env.robot_vmax
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: vx =  env.robot_vmax
        if keys[pygame.K_UP] or keys[pygame.K_w]:    vy = -env.robot_vmax
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:  vy =  env.robot_vmax

        if vx != 0 and vy != 0:
            vx *= 0.7071
            vy *= 0.7071

        env.step(vx, vy, dt)
        env.render(screen, font)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()