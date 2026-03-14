import pygame
import sys

# ==========================================
# 阶段一：赛博朋克风 2D 多智能体环境基线
# ==========================================

# --- 基础配置 ---
GRID_SIZE = 10      # 10x10 的网格
CELL_SIZE = 60      # 每个网格的像素大小
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 30

# --- 极客/赛博朋克配色卡 ---
BG_COLOR = (12, 12, 18)           # 深邃黑底
GRID_COLOR = (0, 70, 70)          # 幽暗青色网格线
OBSTACLE_COLOR = (45, 45, 55)     # 障碍物暗灰色
R0_COLOR = (0, 255, 255)          # 机器人 0: 霓虹青 (Cyan)
R1_COLOR = (255, 0, 255)          # 机器人 1: 霓虹粉 (Magenta)
CT_COLOR = (255, 255, 0)          # 协同任务 (ct): 发光亮黄
IT_COLOR = (0, 255, 0)            # 个体任务 (ts): 霓虹亮绿

class CyberGridEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Phase 1: MARL Synergistic Environment")
        self.clock = pygame.time.Clock()
        
        # 物理拓扑：0=可行走, 1=障碍物
        self.map_data = [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        
        # 异构多智能体定义
        self.robots = {
            0: {'pos': [0, 0], 'color': R0_COLOR, 'name': 'Robot 0'},
            1: {'pos': [9, 9], 'color': R1_COLOR, 'name': 'Robot 1'}
        }
                       
        # 任务定义 (坐标 [x, y])
        self.c_tasks = {'ct_1': [4, 4], 'ct_2': [8, 1]} # 协同任务
        self.i_tasks = {'ts_1': [2, 2], 'ts_2': [7, 7]} # 个体任务
        
        # 状态拦截器（防止终端疯狂刷屏）
        self.r0_waiting_for = set()
        self.r1_waiting_for = set()
        self.completed_c_tasks = set()
        
    def draw_grid(self):
        self.screen.fill(BG_COLOR)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.map_data[y][x] == 1:
                    pygame.draw.rect(self.screen, OBSTACLE_COLOR, rect)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

    def draw_entities(self):
        # 绘制个体任务 (绿色方块)
        for task_name, pos in self.i_tasks.items():
            rect = pygame.Rect(pos[0] * CELL_SIZE + 15, pos[1] * CELL_SIZE + 15, CELL_SIZE - 30, CELL_SIZE - 30)
            pygame.draw.rect(self.screen, IT_COLOR, rect)

        # 绘制协同任务 (黄色圆点，若完成则消失)
        for task_name, pos in self.c_tasks.items():
            if task_name not in self.completed_c_tasks:
                center = (pos[0] * CELL_SIZE + CELL_SIZE // 2, pos[1] * CELL_SIZE + CELL_SIZE // 2)
                pygame.draw.circle(self.screen, CT_COLOR, center, CELL_SIZE // 4)
            
        # 绘制机器人 (霓虹方块)
        for r_id, bot in self.robots.items():
            x, y = bot['pos']
            rect = pygame.Rect(x * CELL_SIZE + 8, y * CELL_SIZE + 8, CELL_SIZE - 16, CELL_SIZE - 16)
            pygame.draw.rect(self.screen, bot['color'], rect, border_radius=6)

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            new_pos_0 = list(self.robots[0]['pos'])
            new_pos_1 = list(self.robots[1]['pos'])
            
            # Robot 0: WASD
            if event.key == pygame.K_w: new_pos_0[1] -= 1
            elif event.key == pygame.K_s: new_pos_0[1] += 1
            elif event.key == pygame.K_a: new_pos_0[0] -= 1
            elif event.key == pygame.K_d: new_pos_0[0] += 1
            
            # Robot 1: 方向键
            if event.key == pygame.K_UP: new_pos_1[1] -= 1
            elif event.key == pygame.K_DOWN: new_pos_1[1] += 1
            elif event.key == pygame.K_LEFT: new_pos_1[0] -= 1
            elif event.key == pygame.K_RIGHT: new_pos_1[0] += 1

            self._move_robot(0, new_pos_0)
            self._move_robot(1, new_pos_1)

    def _move_robot(self, robot_idx, new_pos):
        x, y = new_pos
        # 物理限制：不能越界，不能撞墙
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and self.map_data[y][x] == 0:
            # 只有坐标发生变化才进行判定
            if self.robots[robot_idx]['pos'] != new_pos:
                self.robots[robot_idx]['pos'] = new_pos
                self.check_tasks()

    def check_tasks(self):
        r0_pos = self.robots[0]['pos']
        r1_pos = self.robots[1]['pos']
        
        # 监听协同任务触发机制
        for ct_name, ct_pos in self.c_tasks.items():
            if ct_name in self.completed_c_tasks:
                continue
            
            r0_at_ct = (r0_pos == ct_pos)
            r1_at_ct = (r1_pos == ct_pos)
            
            # 完美闭环打印逻辑：只有一台机器人到达时
            if r0_at_ct and not r1_at_ct:
                if ct_name not in self.r0_waiting_for:
                    print(f">>> [LOGIC] Robot 0 reached {ct_name}, waiting for Robot 1...")
                    self.r0_waiting_for.add(ct_name)
                    
            elif r1_at_ct and not r0_at_ct:
                if ct_name not in self.r1_waiting_for:
                    print(f">>> [LOGIC] Robot 1 reached {ct_name}, waiting for Robot 0...")
                    self.r1_waiting_for.add(ct_name)
            
            # 当两台机器人同时抵达
            elif r0_at_ct and r1_at_ct:
                print(f"!!! [SYNERGY] Target {ct_name} COMPLETE !!!")
                self.completed_c_tasks.add(ct_name)
                # 清除等待状态
                self.r0_waiting_for.discard(ct_name)
                self.r1_waiting_for.discard(ct_name)
                
            # 如果机器人离开了任务点，重置它的等待状态
            if not r0_at_ct and ct_name in self.r0_waiting_for:
                self.r0_waiting_for.discard(ct_name)
                print(f"--- [LOGIC] Robot 0 left {ct_name} before completion.")
            if not r1_at_ct and ct_name in self.r1_waiting_for:
                self.r1_waiting_for.discard(ct_name)
                print(f"--- [LOGIC] Robot 1 left {ct_name} before completion.")

    def run(self):
        running = True
        print("\n--- Phase 1: Environment Active ---")
        print("Controls: R0 (Cyan) -> W/A/S/D | R1 (Magenta) -> Arrows")
        print("Goal: Move both robots to the Yellow dots (ct_1, ct_2).")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_input(event)
            
            self.draw_grid()
            self.draw_entities()
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    env = CyberGridEnv()
    env.run()