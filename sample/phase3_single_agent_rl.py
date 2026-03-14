import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import spot
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# ==========================================
# 阶段二继承：极简 Spot DFA 解析器
# ==========================================
class DFAExtractor:
    def __init__(self, ltlf_formula):
        f = spot.from_ltlf(ltlf_formula)
        aut = f.translate('det', 'sbacc')
        self.dfa = spot.to_finite(aut)
        self.bdict = self.dfa.get_dict()
        
    def get_graph_structure(self):
        dfa_struct = {}
        init_state = self.dfa.get_init_state_number()
        for s in range(self.dfa.num_states()):
            edges = []
            is_accepting = False
            for t in self.dfa.out(s):
                cond = spot.bdd_format_formula(self.bdict, t.cond)
                if cond == '1': cond = 'True'
                if cond == '0': cond = 'False'
                
                # 将 Spot 的逻辑符号转换为 Python 语法，方便直接 eval()
                py_cond = cond.replace('!', ' not ').replace('&', ' and ').replace('|', ' or ')
                edges.append({'dst_state': t.dst, 'condition': py_cond})
                if t.acc: is_accepting = True
            dfa_struct[s] = {'edges': edges, 'is_accepting': is_accepting}
        return init_state, dfa_struct

# ==========================================
# 阶段三：Product MDP 乘积强化学习环境
# ==========================================
class ProductMDPEnv(gym.Env):
    """
    单智能体乘积环境：物理状态 + DFA 状态
    任务 (LTLf): 必须先走到 ts_1，然后再走到 ts_2 -> F(ts_1 & F(ts_2))
    """
    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, render_mode=None):
        super().__init__()
        self.grid_size = 10
        self.render_mode = render_mode
        
        # 1. 物理环境定义
        self.map_data = [
            [0,0,0,1,0,0,0,0,0,0],
            [0,0,0,1,0,0,1,1,0,0],
            [0,0,0,0,0,0,0,1,0,0],
            [0,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,0,0,0,1,1,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
        ]
        self.start_pos = [0, 0]
        self.ts_1_pos = [2, 2]  # 个体任务 1
        self.ts_2_pos = [8, 8]  # 个体任务 2
        
        # 2. 逻辑大脑初始化
        self.formula = "F(ts_1 & F(ts_2))"
        self.dfa_parser = DFAExtractor(self.formula)
        self.init_dfa_state, self.dfa_graph = self.dfa_parser.get_graph_structure()
        
        # 3. 强化学习空间定义
        # Action: 0:上, 1:下, 2:左, 3:右, 4:停
        self.action_space = spaces.Discrete(5)
        # Observation: [robot_x, robot_y, dfa_state_id]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([self.grid_size-1, self.grid_size-1, len(self.dfa_graph)-1]), 
            dtype=np.float32
        )
        
        # Pygame 渲染配置
        self.cell_size = 50
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = list(self.start_pos)
        self.current_dfa_state = self.init_dfa_state
        self.step_count = 0
        
        obs = np.array([self.robot_pos[0], self.robot_pos[1], self.current_dfa_state], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        
        # 1. 物理动作执行
        new_pos = list(self.robot_pos)
        if action == 0: new_pos[1] -= 1   # UP
        elif action == 1: new_pos[1] += 1 # DOWN
        elif action == 2: new_pos[0] -= 1 # LEFT
        elif action == 3: new_pos[0] += 1 # RIGHT
        
        # 碰撞检测
        x, y = new_pos
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.map_data[y][x] == 0:
            self.robot_pos = new_pos
            
        # 2. 提取当前物理坐标蕴含的“原子命题” (Propositions)
        props = {
            'ts_1': self.robot_pos == self.ts_1_pos,
            'ts_2': self.robot_pos == self.ts_2_pos
        }
        
        # 3. 驱动 DFA 状态机发生转移
        next_dfa_state = self.current_dfa_state
        edges = self.dfa_graph[self.current_dfa_state]['edges']
        for edge in edges:
            cond_str = edge['condition']
            # 安全的动态执行字符串逻辑
            # 将字典中的原子命题变量注入到当前作用域
            if eval(cond_str, {}, props):
                next_dfa_state = edge['dst_state']
                break
                
        # 4. 奖励函数设计 (上帝视角 Reward Shaping)
        reward = -0.1  # 基础时间惩罚，逼迫走最短路径
        terminated = False
        
        # 判断 DFA 是否发生推进
        if next_dfa_state != self.current_dfa_state:
            self.current_dfa_state = next_dfa_state
            
            if self.dfa_graph[self.current_dfa_state]['is_accepting']:
                # 触发了 DFA 的接收状态 -> 任务彻底通关！
                reward += 200.0
                terminated = True
                print(f"[RL] GOAL REACHED! Steps: {self.step_count}")
            else:
                # 触发了中间节点 (如完成了 ts_1)，给中间势能奖励
                reward += 50.0
                print(f"[RL] Sub-task complete! DFA advanced to State {self.current_dfa_state}")
                
        # 截断机制防止死循环
        truncated = False
        if self.step_count > 200:
            truncated = True

        obs = np.array([self.robot_pos[0], self.robot_pos[1], self.current_dfa_state], dtype=np.float32)
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
            pygame.display.set_caption("Phase 3: RL PPO Agent")
            self.clock = pygame.time.Clock()

        if self.window is not None:
            self.window.fill((12, 12, 18))
            # 画网格与障碍物
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    rect = pygame.Rect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
                    if self.map_data[y][x] == 1:
                        pygame.draw.rect(self.window, (45, 45, 55), rect)
                    pygame.draw.rect(self.window, (0, 70, 70), rect, 1)
            
            # 画任务点
            pygame.draw.rect(self.window, (0, 255, 0), (self.ts_1_pos[0]*self.cell_size+10, self.ts_1_pos[1]*self.cell_size+10, 30, 30))
            pygame.draw.rect(self.window, (0, 255, 255), (self.ts_2_pos[0]*self.cell_size+10, self.ts_2_pos[1]*self.cell_size+10, 30, 30))
            
            # 画机器人
            pygame.draw.circle(self.window, (255, 0, 255), (self.robot_pos[0]*self.cell_size+25, self.robot_pos[1]*self.cell_size+25), 15)
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
# --- 1. 环境验证 ---
    env = ProductMDPEnv()
    check_env(env, warn=True)
    
    # --- 2. PPO 模型训练 ---
    print("\n[SYSTEM] 开始使用 PPO 进行降维打击训练...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
    # 训练 50000 步
    model.learn(total_timesteps=50000)
    
    # ==========================================
    # 核心新增：将训练好的大脑保存为 zip 压缩包
    # ==========================================
    model.save("ppo_phase3_agent")
    print("\n[SYSTEM] 模型已成功保存为 ppo_phase3_agent.zip！")
    
    env.close()