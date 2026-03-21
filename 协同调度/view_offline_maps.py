import os
import pickle
import random
import pygame
import sys

# --- UI 颜色设定 ---
WHITE = (240, 240, 240)
BLACK = (30, 30, 30)
GREEN = (100, 255, 100)
ORANGE = (255, 165, 0)
DARK_GRAY = (80, 80, 80)

def load_random_scenario(cache_dir="offline_maps"):
    """从本地缓存目录中随机加载一张 .pkl 地图"""
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"找不到地图缓存目录: {cache_dir}，请先运行 build_offline_maps.py")
    
    map_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    if not map_files:
        raise ValueError("缓存目录中没有找到任何 .pkl 地图文件！")
        
    chosen_file = random.choice(map_files)
    file_path = os.path.join(cache_dir, chosen_file)
    
    with open(file_path, 'rb') as f:
        scenario_config = pickle.load(f)
        
    print(f"\n✅ 成功加载: {chosen_file}")
    return scenario_config

def draw_scenario(screen, font, small_font, scenario):
    """将剧本数据渲染到屏幕上"""
    screen.fill(WHITE)
    
    # 1. 绘制障碍物
    for obs in scenario["obstacles"]:
        pygame.draw.rect(screen, BLACK, obs)
        
    # 2. 绘制单机任务点 (绿色)
    if "tasks" in scenario:
        for name, pos in scenario["tasks"].items():
            px, py = int(pos[0]), int(pos[1])
            pygame.draw.circle(screen, GREEN, (px, py), 20)
            pygame.draw.circle(screen, (0, 150, 0), (px, py), 20, 2)
            text = font.render(name.split()[-1], True, WHITE)
            screen.blit(text, text.get_rect(center=(px, py)))

    # 3. 绘制协同任务点与“构型锁” (橙色)
    if "coop_tasks_dict" in scenario:
        for name, pos in scenario["coop_tasks_dict"].items():
            px, py = int(pos[0]), int(pos[1])
            pygame.draw.circle(screen, ORANGE, (px, py), 22)
            pygame.draw.circle(screen, (200, 100, 0), (px, py), 22, 3)
            
            # 画任务编号
            text = font.render(name.split()[-1], True, WHITE)
            screen.blit(text, text.get_rect(center=(px, py)))
            
            # 【核心新增】：在下方画出这个任务需要的队伍构型 (如 H:1 L:1)
            reqs = scenario.get("coop_tasks_req", {}).get(name, {})
            if reqs:
                # 提取首字母作为简称
                req_str = " ".join([f"{role[0]}:{count}" for role, count in reqs.items()])
                req_text = small_font.render(req_str, True, DARK_GRAY)
                screen.blit(req_text, (px - 25, py + 25))

    # 4. 绘制异构机器人车队
    for robot in scenario["robots"]:
        pos = robot["start_pos"]
        color = robot["color"]
        px, py = int(pos[0]), int(pos[1])
        
        # 本体
        pygame.draw.circle(screen, color, (px, py), 15)
        pygame.draw.circle(screen, BLACK, (px, py), 15, 2)
        
        # 【核心新增】：在车顶画出角色的首字母简称 (H, L, S, M, C)
        role = robot.get("role", "?")
        role_initial = role[0] if role != "?" else "?"
        role_text = font.render(role_initial, True, WHITE)
        screen.blit(role_text, role_text.get_rect(center=(px, py)))
        
        # 画白色车头指示（装饰）
        pygame.draw.circle(screen, WHITE, (px + 10, py), 3)

def main():
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Heterogeneous MRS Map Viewer (Press SPACE)")
    
    font = pygame.font.SysFont("arial", 18, bold=True)
    small_font = pygame.font.SysFont("arial", 14, bold=True) # 专门用于画小字的字体
    clock = pygame.time.Clock()

    try:
        current_scenario = load_random_scenario()
    except Exception as e:
        print(f"加载失败: {e}")
        pygame.quit()
        sys.exit()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_scenario = load_random_scenario()

        draw_scenario(screen, font, small_font, current_scenario)
        
        tip_text = font.render("Press SPACE to load next map", True, (100, 100, 100))
        screen.blit(tip_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()