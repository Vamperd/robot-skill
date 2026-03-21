import math
import importlib.util
import os
import pickle
import random
import sys
from typing import Dict, List, Tuple

import pygame

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASTAR_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "导航结合RL运动"))
_ASTAR_FILE = os.path.join(_ASTAR_DIR, "a_star_planner.py")

if not os.path.exists(_ASTAR_FILE):
    raise FileNotFoundError(f"未找到 A* 模块文件: {_ASTAR_FILE}")

_ASTAR_SPEC = importlib.util.spec_from_file_location("a_star_planner", _ASTAR_FILE)
if _ASTAR_SPEC is None or _ASTAR_SPEC.loader is None:
    raise ImportError("无法加载 a_star_planner 模块规格。")

_ASTAR_MODULE = importlib.util.module_from_spec(_ASTAR_SPEC)
sys.modules["a_star_planner"] = _ASTAR_MODULE
_ASTAR_SPEC.loader.exec_module(_ASTAR_MODULE)
AStarPlanner = _ASTAR_MODULE.AStarPlanner

WIDTH = 800
HEIGHT = 600
ROBOT_RADIUS = 15

BG_COLOR = (245, 245, 245)
OBSTACLE_COLOR = (35, 35, 35)
SINGLE_TASK_COLOR = (80, 210, 120)
SYNC_TASK_COLOR = (255, 200, 0)
TEXT_COLOR = (25, 25, 25)

ROBOT_ROLES = {
    "Heavy": {"multiplier": 2.5, "color": (255, 100, 100)},      # 重型搬运车：极慢，红色
    "Light": {"multiplier": 0.5, "color": (100, 255, 100)},      # 轻型分拣车：极快，绿色
    "Scanner": {"multiplier": 1.0, "color": (100, 100, 255)},    # 巡检扫描车：标准，蓝色
    "Maintainer": {"multiplier": 1.5, "color": (255, 200, 50)},  # 维修保障车：偏慢，黄色
    "Commander": {"multiplier": 1.2, "color": (200, 100, 255)},  # 指挥协调车：稍慢，紫色
}


def _circle_hits_rect(cx: float, cy: float, radius: float, rect: pygame.Rect) -> bool:
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top, min(cy, rect.bottom))
    return math.hypot(cx - closest_x, cy - closest_y) < radius


def is_valid_point(x: float, y: float, obstacles: List[pygame.Rect], safe_margin: float) -> bool:
    """检测点位是否合法：在地图内，且与所有障碍物保持安全距离。"""
    if x < safe_margin or x > WIDTH - safe_margin:
        return False
    if y < safe_margin or y > HEIGHT - safe_margin:
        return False

    for obs in obstacles:
        if _circle_hits_rect(x, y, safe_margin, obs):
            return False

    return True


def _rects_too_close(a: pygame.Rect, b: pygame.Rect, margin: int = 18) -> bool:
    expanded = a.inflate(margin * 2, margin * 2)
    return expanded.colliderect(b)


def _sample_internal_obstacles(rng: random.Random) -> List[pygame.Rect]:
    obstacles: List[pygame.Rect] = [
        # 外墙
        pygame.Rect(0, 0, WIDTH, 20),
        pygame.Rect(0, HEIGHT - 20, WIDTH, 20),
        pygame.Rect(0, 0, 20, HEIGHT),
        pygame.Rect(WIDTH - 20, 0, 20, HEIGHT),
    ]

    internal_target = rng.randint(5, 10)
    internal_area_limit = int(WIDTH * HEIGHT * 0.28)
    internal_area = 0

    internal_rects: List[pygame.Rect] = []
    attempts = 0
    while len(internal_rects) < internal_target and attempts < 800:
        attempts += 1

        w = rng.randint(40, 200)
        h = rng.randint(40, 200)
        x = rng.randint(30, WIDTH - w - 30)
        y = rng.randint(30, HEIGHT - h - 30)

        candidate = pygame.Rect(x, y, w, h)

        if any(_rects_too_close(candidate, r, margin=24) for r in internal_rects):
            continue

        if internal_area + w * h > internal_area_limit:
            continue

        # 保留若干主干通道
        if 310 < candidate.centerx < 490 and 220 < candidate.centery < 380:
            continue

        internal_rects.append(candidate)
        internal_area += w * h

    obstacles.extend(internal_rects)
    return obstacles


def _sample_point_set(
    rng: random.Random,
    count: int,
    obstacles: List[pygame.Rect],
    safe_margin: float,
    min_pair_dist: float,
) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []

    for _ in range(count):
        placed = False
        for _attempt in range(500):
            x = rng.randint(40, WIDTH - 40)
            y = rng.randint(40, HEIGHT - 40)

            if not is_valid_point(x, y, obstacles, safe_margin):
                continue

            if any(math.hypot(x - px, y - py) < min_pair_dist for px, py in points):
                continue

            points.append((x, y))
            placed = True
            break

        if not placed:
            raise RuntimeError("无法生成合法点位，请重试或放宽约束。")

    return points


def _build_task_dict(
    rng: random.Random,
    obstacles: List[pygame.Rect],
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]], Dict[str, int]]:
    num_single = rng.randint(4, 8)
    num_sync = rng.randint(1, 3)

    total_tasks = num_single + num_sync
    points = _sample_point_set(
        rng=rng,
        count=total_tasks,
        obstacles=obstacles,
        safe_margin=ROBOT_RADIUS + 20,
        min_pair_dist=70.0,
    )

    tasks: Dict[str, Tuple[int, int]] = {}
    coop_tasks: Dict[str, Tuple[int, int]] = {}
    base_times: Dict[str, int] = {}

    single_names = [f"Task {chr(ord('A') + i)}" for i in range(num_single)]
    sync_names = [f"Sync {i + 1}" for i in range(num_sync)]

    for i, pt in enumerate(points[:num_single]):
        name = single_names[i]
        tasks[name] = pt
        base_times[name] = rng.randint(60, 300)

    for i, pt in enumerate(points[num_single:]):
        name = sync_names[i]
        coop_tasks[name] = pt
        base_times[name] = rng.randint(60, 300)

    return tasks, coop_tasks, base_times


def _sample_robot_starts(
    rng: random.Random,
    robot_count: int,
    obstacles: List[pygame.Rect],
) -> List[Tuple[int, int]]:
    starts: List[Tuple[int, int]] = []

    for _ in range(robot_count):
        ok = False
        for _attempt in range(600):
            x = rng.randint(40, WIDTH - 40)
            y = rng.randint(40, HEIGHT - 40)

            if not is_valid_point(x, y, obstacles, ROBOT_RADIUS + 12):
                continue

            if any(math.hypot(x - sx, y - sy) < (ROBOT_RADIUS * 2 + 12) for sx, sy in starts):
                continue

            starts.append((x, y))
            ok = True
            break

        if not ok:
            raise RuntimeError("无法生成互不重叠的机器人起点，请重试。")

    return starts


def _ensure_sync_assigned_at_least_two(
    rng: random.Random,
    sync_tasks: List[str],
    robot_cfgs: List[Dict],
) -> None:
    for sync in sync_tasks:
        assigned = [idx for idx, cfg in enumerate(robot_cfgs) if sync in cfg["task_sequence"]]

        while len(assigned) < 2:
            candidates = [i for i in range(len(robot_cfgs)) if i not in assigned]
            if not candidates:
                break

            idx = rng.choice(candidates)
            seq = robot_cfgs[idx]["task_sequence"]

            if len(seq) < 6:
                insert_pos = rng.randint(0, len(seq))
                seq.insert(insert_pos, sync)
            else:
                # 优先替换非协同任务
                non_sync_positions = [i for i, t in enumerate(seq) if not t.startswith("Sync")]
                rep_pos = rng.choice(non_sync_positions) if non_sync_positions else rng.randint(0, len(seq) - 1)
                seq[rep_pos] = sync

            assigned = [i for i, cfg in enumerate(robot_cfgs) if sync in cfg["task_sequence"]]


def validate_scenario(scenario_config: Dict) -> bool:
    """使用 A* 验证场景连通性。若任意机器人无法到达其任务链中的任一任务，则判定为无效场景。"""
    planner = AStarPlanner(width=WIDTH, height=HEIGHT, resolution=10, robot_radius=ROBOT_RADIUS, margin=5)
    obstacles = []
    for obs in scenario_config["obstacles"]:
        if isinstance(obs, pygame.Rect):
            obstacles.append((obs.x, obs.y, obs.w, obs.h))
        else:
            obstacles.append((int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3])))

    for robot_cfg in scenario_config["robots"]:
        start_pos = robot_cfg["start_pos"]

        for task_name in robot_cfg["task_sequence"]:
            task_pos = scenario_config["tasks"].get(task_name)
            if not task_pos:
                task_pos = scenario_config.get("coop_tasks_dict", {}).get(task_name)

            if task_pos:
                path = planner.plan(start_pos, task_pos, obstacles)

                # A* 路径过短且起终点不重合，通常表示被障碍完全堵死
                if len(path) <= 1 and math.hypot(start_pos[0] - task_pos[0], start_pos[1] - task_pos[1]) > 20.0:
                    return False

                start_pos = task_pos

    return True


def get_fallback_scenario() -> Dict:
    """兜底场景：空旷地图 + 少量任务，确保可通行。"""
    obstacles = [
        pygame.Rect(0, 0, WIDTH, 20),
        pygame.Rect(0, HEIGHT - 20, WIDTH, 20),
        pygame.Rect(0, 0, 20, HEIGHT),
        pygame.Rect(WIDTH - 20, 0, 20, HEIGHT),
    ]

    tasks = {
        "Task A": (120, 120),
        "Task B": (680, 120),
        "Task C": (120, 480),
        "Task D": (680, 480),
    }
    coop_tasks_dict = {"Sync 1": (400, 300)}
    coop_tasks = ["Sync 1"]
    base_times = {name: 120 for name in tasks}
    base_times["Sync 1"] = 120

    robots = [
        {
            "name": "Robot_1",
            "role": "Heavy",
            "start_pos": (120, 300),
            "color": ROBOT_ROLES["Heavy"]["color"],
            "multiplier": ROBOT_ROLES["Heavy"]["multiplier"],
            "task_sequence": ["Task A", "Sync 1", "Task D"],
        },
        {
            "name": "Robot_2",
            "role": "Light",
            "start_pos": (680, 300),
            "color": ROBOT_ROLES["Light"]["color"],
            "multiplier": ROBOT_ROLES["Light"]["multiplier"],
            "task_sequence": ["Task B", "Sync 1", "Task C"],
        },
    ]

    return {
        "obstacles": obstacles,
        "tasks": tasks,
        "coop_tasks_dict": coop_tasks_dict,
        "coop_tasks": coop_tasks,
        "coop_tasks_req": {"Sync 1": {"Heavy": 1, "Light": 1}},
        "base_times": base_times,
        "robots": robots,
    }


def _dehydrate_scenario(scenario_config: Dict) -> Dict:
    """将 pygame.Rect 转为 (x, y, w, h) 元组，便于序列化缓存。"""
    dehydrated = dict(scenario_config)
    dehydrated_obstacles = []

    for obs in scenario_config.get("obstacles", []):
        if isinstance(obs, pygame.Rect):
            dehydrated_obstacles.append((int(obs.x), int(obs.y), int(obs.w), int(obs.h)))
        else:
            dehydrated_obstacles.append((int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3])))

    dehydrated["obstacles"] = dehydrated_obstacles
    return dehydrated


def draw_scenario(
    scenario_config: Dict,
    hold_ms: int = 1800,
    window_title: str = "Random Scenario Preview",
) -> None:
    """使用 pygame 绘制当前场景。hold_ms>0 时自动关闭预览，=0 时等待手动关闭。"""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(window_title)
    font = pygame.font.SysFont("arial", 16, bold=True)
    small_font = pygame.font.SysFont("arial", 13)
    clock = pygame.time.Clock()

    obstacles: List[pygame.Rect] = [obs if isinstance(obs, pygame.Rect) else pygame.Rect(*obs) for obs in scenario_config["obstacles"]]
    tasks: Dict[str, Tuple[int, int]] = scenario_config["tasks"]
    coop_tasks_dict: Dict[str, Tuple[int, int]] = scenario_config.get("coop_tasks_dict", {})
    base_times: Dict[str, int] = scenario_config["base_times"]
    robots: List[Dict] = scenario_config["robots"]

    start_tick = pygame.time.get_ticks()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if hold_ms > 0 and pygame.time.get_ticks() - start_tick >= hold_ms:
            running = False

        screen.fill(BG_COLOR)

        # 障碍物
        for obs in obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, obs)

        # 任务点
        for task_name, (tx, ty) in tasks.items():
            pygame.draw.circle(screen, SINGLE_TASK_COLOR, (int(tx), int(ty)), 14)
            pygame.draw.circle(screen, OBSTACLE_COLOR, (int(tx), int(ty)), 14, 2)

            label = small_font.render(f"{task_name} ({base_times.get(task_name, 0)})", True, TEXT_COLOR)
            screen.blit(label, (int(tx) + 12, int(ty) - 8))

        # 协同任务点
        for task_name, (tx, ty) in coop_tasks_dict.items():
            pygame.draw.circle(screen, SYNC_TASK_COLOR, (int(tx), int(ty)), 14)
            pygame.draw.circle(screen, OBSTACLE_COLOR, (int(tx), int(ty)), 14, 2)

            req = scenario_config.get("coop_tasks_req", {}).get(task_name, {})
            req_text = "+".join([f"{k}:{v}" for k, v in req.items()]) if req else "N/A"
            label = small_font.render(f"{task_name} ({base_times.get(task_name, 0)}) [{req_text}]", True, TEXT_COLOR)
            screen.blit(label, (int(tx) + 12, int(ty) - 8))

        # 机器人
        for robot in robots:
            rx, ry = robot["start_pos"]
            color = robot["color"]
            name = robot["name"]
            multiplier = robot["multiplier"]

            pygame.draw.circle(screen, color, (int(rx), int(ry)), ROBOT_RADIUS)
            pygame.draw.circle(screen, OBSTACLE_COLOR, (int(rx), int(ry)), ROBOT_RADIUS, 2)

            role = robot.get("role", "Unknown")
            info = small_font.render(f"{name} [{role}] x{multiplier}", True, TEXT_COLOR)
            screen.blit(info, (int(rx) + 14, int(ry) - 8))

        tip = font.render("Sync=Yellow  Task=Green  Obstacle=Black", True, TEXT_COLOR)
        screen.blit(tip, (24, 24))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def generate_random_scenario(
    seed: int | None = None,
    visualize: bool = False,
    visualize_hold_ms: int = 5000,
) -> Dict:
    """生成一个合法的多智能体异构任务测试场景（带 A* 连通性拒绝采样）。"""
    max_retries = 100
    master_rng = random.Random(seed)

    for attempt in range(max_retries):
        # 每次尝试使用独立随机种子，既保证随机性，也保证可复现
        attempt_seed = master_rng.randint(0, 2**31 - 1)
        rng = random.Random(attempt_seed)

        obstacles = _sample_internal_obstacles(rng)
        tasks, coop_tasks_dict, base_times = _build_task_dict(rng, obstacles)

        # 决定出场机器人数量，从 2~5 提高到 3~6 辆，增加环境复杂度
        robot_count = rng.randint(3, 6)
        starts = _sample_robot_starts(rng, robot_count, obstacles)

        single_task_names = list(tasks.keys())
        robot_cfgs: List[Dict] = []

        # 生成机器人（构型化）
        robot_roles_list = list(ROBOT_ROLES.keys())
        for i in range(robot_count):
            role = rng.choice(robot_roles_list)
            robot_cfgs.append(
                {
                    "name": f"Robot_{i + 1}",
                    "role": role,
                    "start_pos": starts[i],
                    "color": ROBOT_ROLES[role]["color"],
                    "multiplier": ROBOT_ROLES[role]["multiplier"],
                    "task_sequence": [],
                }
            )

        # 生成协同任务构型锁并精准分配
        coop_tasks_req: Dict[str, Dict[str, int]] = {}
        for task_name in coop_tasks_dict.keys():
            # 从当前场上已有的机器人中，随机挑选 2 或 3 个不同的机器人来组成联盟
            if len(robot_cfgs) >= 2:
                # 如果场上车多于3辆，有 30% 概率生成需要 3 辆车共同完成的超级协同任务
                coalition_size = 3 if (len(robot_cfgs) >= 3 and rng.random() > 0.7) else 2
                assigned_bots = rng.sample(robot_cfgs, coalition_size)
                req: Dict[str, int] = {}
                for b in assigned_bots:
                    req[b["role"]] = req.get(b["role"], 0) + 1
                    b["task_sequence"].append(task_name)
                coop_tasks_req[task_name] = req

        for i in range(robot_count):
            seq_len = rng.randint(3, 6)
            seq = robot_cfgs[i]["task_sequence"]
            need = max(0, seq_len - len(seq))

            if need > 0:
                if need <= len(single_task_names):
                    seq.extend(rng.sample(single_task_names, need))
                else:
                    seq.extend([rng.choice(single_task_names) for _ in range(need)])

            robot_cfgs[i]["task_sequence"] = seq[:6]

        scenario_config = {
            "obstacles": obstacles,
            "tasks": tasks,
            "coop_tasks_dict": coop_tasks_dict,
            "coop_tasks": list(coop_tasks_dict.keys()),
            "coop_tasks_req": coop_tasks_req,
            "base_times": base_times,
            "robots": robot_cfgs,
        }

        # 连通性验证
        if validate_scenario(scenario_config):
            print(f"成功生成有效连通的场景！(尝试次数: {attempt + 1})")
            scenario_config = _dehydrate_scenario(scenario_config)
            if visualize:
                draw_scenario(scenario_config, hold_ms=visualize_hold_ms)
            return scenario_config

    # 如果运气极差，100次都没生成出来，返回保底安全地图
    print("警告：无法生成有效场景，返回保底空旷地图。")
    fallback = _dehydrate_scenario(get_fallback_scenario())
    if visualize:
        draw_scenario(fallback, hold_ms=visualize_hold_ms, window_title="Fallback Scenario Preview")
    return fallback


def load_random_scenario(cache_dir="offline_maps"):
    """训练时调用：从预生成的地图库中随机抽选一个加载"""
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"找不到地图缓存目录: {cache_dir}")

    map_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    if not map_files:
        raise ValueError("缓存目录中没有找到 .pkl 地图文件！")

    chosen_file = random.choice(map_files)
    file_path = os.path.join(cache_dir, chosen_file)

    with open(file_path, 'rb') as f:
        scenario_config = pickle.load(f)

    return scenario_config


if __name__ == "__main__":
    VISUALIZE_EACH_SCENARIO = True
    cfg = generate_random_scenario(visualize=VISUALIZE_EACH_SCENARIO)
    print("生成场景成功:")
    print(f"障碍物数量: {len(cfg['obstacles'])}")
    print(f"任务数量: {len(cfg['tasks'])}, 协同任务: {cfg['coop_tasks']}")
    print(f"机器人数量: {len(cfg['robots'])}")
