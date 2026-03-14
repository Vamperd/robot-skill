import pygame
import math
import sys

# 1. 环境全局参数初始化
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LTLf Multi-Agent RL - Continuous Point Mass Env")
clock = pygame.time.Clock()

# 定义颜色
WHITE = (240, 240, 240)
BLUE = (50, 150, 255)   # 机器人颜色
RED = (255, 100, 100)   # 任务点颜色
GREEN = (100, 255, 100) # 完成任务时的颜色
DARK_GRAY = (50, 50, 50)

class PointRobot:
    def __init__(self, x, y):
        # 核心：使用浮点数来表示连续状态空间
        self.x = float(x)
        self.y = float(y)
        self.radius = 15
        self.v_max = 250.0  # 机器人的最大速度 (像素/秒)
        self.color = BLUE

    def step(self, action_vx, action_vy, dt):
        """
        物理引擎核心：积分更新坐标
        在未来的 RL 环境中，action_vx 和 action_vy 就是神经网络输出的连续动作
        """
        self.x += action_vx * dt
        self.y += action_vy * dt

        # 连续空间的边界碰撞检测（限制在屏幕内）
        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        # 画一个指示方向的小点（可选，方便视觉确认）
        pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), 3)

class TaskPoint:
    def __init__(self, x, y, name="Task A"):
        self.x = float(x)
        self.y = float(y)
        self.radius = 25
        self.color = RED
        self.name = name
        self.is_completed = False

    def draw(self, surface):
        current_color = GREEN if self.is_completed else self.color
        pygame.draw.circle(surface, current_color, (int(self.x), int(self.y)), self.radius)
        
        # 渲染文字
        font = pygame.font.SysFont("arial", 14, bold=True)
        text = font.render(self.name, True, WHITE)
        text_rect = text.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(text, text_rect)

def check_trigger(robot, task):
    """
    检查连续空间下的触发条件：欧氏距离小于两者半径之和
    这是未来构建 DFA 状态跃迁 (Atomic Proposition) 的核心判定逻辑
    """
    distance = math.hypot(robot.x - task.x, robot.y - task.y)
    return distance <= (robot.radius + task.radius)

def main():
    # 实例化机器人与任务点
    robot = PointRobot(100, 100)
    task_A = TaskPoint(600, 400, "Task A")
    
    running = True
    font = pygame.font.SysFont("arial", 20)

    while running:
        # 获取每次循环经过的时间 dt (秒)，保证物理运动与帧率解耦 (纯连续概念)
        dt = clock.tick(60) / 1000.0 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2. 键盘事件映射到连续动作空间 [vx, vy]
        keys = pygame.key.get_pressed()
        vx, vy = 0.0, 0.0
        
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            vx = -robot.v_max
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            vx = robot.v_max
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            vy = -robot.v_max
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            vy = robot.v_max

        # 3. 执行环境 Step
        robot.step(vx, vy, dt)

        # 4. 逻辑判定（DFA 触发预演）
        if not task_A.is_completed and check_trigger(robot, task_A):
            task_A.is_completed = True
            robot.color = GREEN # 机器人变色作为视觉反馈

        # 5. 渲染画面
        screen.fill(WHITE)
        task_A.draw(screen)
        robot.draw(screen)

        # 渲染状态信息
        status_text = f"Pos: ({robot.x:.1f}, {robot.y:.1f}) | Task A: {'Done' if task_A.is_completed else 'Pending'}"
        text_surface = font.render(status_text, True, DARK_GRAY)
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()