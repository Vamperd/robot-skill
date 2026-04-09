from __future__ import annotations

import importlib.util
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pygame

try:
    from coop_docking import (
        DEFAULT_DOCKING_RADIUS,
        DEFAULT_PLANNER_MARGIN,
        DEFAULT_PLANNER_RESOLUTION,
        generate_docking_slots,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .coop_docking import (
        DEFAULT_DOCKING_RADIUS,
        DEFAULT_PLANNER_MARGIN,
        DEFAULT_PLANNER_RESOLUTION,
        generate_docking_slots,
    )


RectLike = Tuple[int, int, int, int]
PointLike = Tuple[int, int]


_CURRENT_DIR = Path(__file__).resolve().parent
_NAV_DIR = _CURRENT_DIR.parent / "导航结合RL运动"
_ASTAR_FILE = _NAV_DIR / "a_star_planner.py"

if not _ASTAR_FILE.exists():
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
BOUNDARY_THICKNESS = 20
ROBOT_RADIUS = 15
BASE_TRAVEL_SPEED = 4.0
SCHEMA_VERSION = "v2"

BG_COLOR = (245, 245, 245)
OBSTACLE_COLOR = (35, 35, 35)
SINGLE_TASK_COLOR = (80, 210, 120)
SYNC_TASK_COLOR = (255, 200, 0)
TEXT_COLOR = (25, 25, 25)

FAMILY_NAMES = (
    "open_balance",
    "role_mismatch",
    "single_bottleneck",
    "double_bottleneck",
    "far_near_trap",
    "multi_sync_cluster",
    "partial_coalition_trap",
)

DEFAULT_SPLIT_COUNTS = {
    "train": {**{family: 200 for family in FAMILY_NAMES}, "partial_coalition_trap": 120},
    "val": {**{family: 40 for family in FAMILY_NAMES}, "partial_coalition_trap": 24},
    "test": {**{family: 40 for family in FAMILY_NAMES}, "partial_coalition_trap": 24},
    "stress": {**{family: 10 for family in FAMILY_NAMES}, "partial_coalition_trap": 10},
}

ROBOT_ROLES = {
    "Heavy": {
        "speed_multiplier": 0.80,
        "service_multiplier": 2.50,
        "color": (255, 100, 100),
    },
    "Light": {
        "speed_multiplier": 1.35,
        "service_multiplier": 0.55,
        "color": (100, 255, 100),
    },
    "Scanner": {
        "speed_multiplier": 1.15,
        "service_multiplier": 1.00,
        "color": (100, 100, 255),
    },
    "Maintainer": {
        "speed_multiplier": 0.95,
        "service_multiplier": 1.45,
        "color": (255, 200, 50),
    },
    "Commander": {
        "speed_multiplier": 1.00,
        "service_multiplier": 1.20,
        "color": (200, 100, 255),
    },
}


@dataclass(frozen=True)
class FamilySpec:
    robot_range: Tuple[int, int]
    task_range: Tuple[int, int]
    sync_range: Tuple[int, int]
    single_role_ratio: float
    precedence_mode: str


@dataclass
class LayoutBlueprint:
    obstacles: List[RectLike]
    robot_zones: List[RectLike]
    single_task_zones: List[RectLike]
    sync_task_zones: List[RectLike]
    chokepoint_zones: List[RectLike]


FAMILY_SPECS: Dict[str, FamilySpec] = {
    "open_balance": FamilySpec((3, 4), (5, 6), (1, 1), 0.05, "none"),
    "role_mismatch": FamilySpec((4, 5), (6, 7), (1, 2), 0.65, "none"),
    "single_bottleneck": FamilySpec((3, 5), (6, 8), (1, 2), 0.10, "none"),
    "double_bottleneck": FamilySpec((4, 6), (7, 9), (2, 3), 0.10, "none"),
    "far_near_trap": FamilySpec((3, 5), (5, 7), (1, 2), 0.20, "far_near"),
    "multi_sync_cluster": FamilySpec((4, 6), (7, 10), (2, 3), 0.25, "light"),
    "partial_coalition_trap": FamilySpec((4, 5), (6, 8), (2, 2), 0.30, "none"),
}


def _rect(x: int, y: int, w: int, h: int) -> RectLike:
    return (int(x), int(y), int(w), int(h))


def _point_in_rect(point: PointLike, rect: RectLike) -> bool:
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def _jitter_rect(
    rect: RectLike,
    rng: random.Random,
    dx: int = 0,
    dy: int = 0,
    dw: int = 0,
    dh: int = 0,
) -> RectLike:
    x, y, w, h = rect
    new_w = max(30, w + rng.randint(-dw, dw) if dw else w)
    new_h = max(30, h + rng.randint(-dh, dh) if dh else h)
    new_x = x + (rng.randint(-dx, dx) if dx else 0)
    new_y = y + (rng.randint(-dy, dy) if dy else 0)
    new_x = max(BOUNDARY_THICKNESS, min(new_x, WIDTH - BOUNDARY_THICKNESS - new_w))
    new_y = max(BOUNDARY_THICKNESS, min(new_y, HEIGHT - BOUNDARY_THICKNESS - new_h))
    return _rect(new_x, new_y, new_w, new_h)


def _border_obstacles() -> List[RectLike]:
    return [
        _rect(0, 0, WIDTH, BOUNDARY_THICKNESS),
        _rect(0, HEIGHT - BOUNDARY_THICKNESS, WIDTH, BOUNDARY_THICKNESS),
        _rect(0, 0, BOUNDARY_THICKNESS, HEIGHT),
        _rect(WIDTH - BOUNDARY_THICKNESS, 0, BOUNDARY_THICKNESS, HEIGHT),
    ]


def _circle_hits_rect(cx: float, cy: float, radius: float, rect: RectLike) -> bool:
    rx, ry, rw, rh = rect
    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))
    return math.hypot(cx - closest_x, cy - closest_y) < radius


def is_valid_point(x: float, y: float, obstacles: Sequence[RectLike], safe_margin: float) -> bool:
    if x < safe_margin or x > WIDTH - safe_margin:
        return False
    if y < safe_margin or y > HEIGHT - safe_margin:
        return False
    return not any(_circle_hits_rect(x, y, safe_margin, rect) for rect in obstacles)


def _sample_point_from_zones(
    rng: random.Random,
    zones: Sequence[RectLike],
    obstacles: Sequence[RectLike],
    safe_margin: float,
    occupied: Sequence[PointLike],
    min_pair_dist: float,
) -> PointLike:
    for _ in range(600):
        zone = rng.choice(zones)
        x = rng.randint(zone[0] + 10, zone[0] + zone[2] - 10)
        y = rng.randint(zone[1] + 10, zone[1] + zone[3] - 10)

        if not is_valid_point(x, y, obstacles, safe_margin):
            continue
        if any(math.hypot(x - px, y - py) < min_pair_dist for px, py in occupied):
            continue
        return (x, y)

    raise RuntimeError("无法在指定区域采样合法点位，请重试。")


def _path_length(path: Sequence[PointLike]) -> float:
    return sum(
        math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        for i in range(1, len(path))
    )


def _path_reaches_goal(path: Sequence[PointLike], goal: PointLike) -> bool:
    if not path:
        return False
    return math.isclose(path[-1][0], goal[0], abs_tol=1e-6) and math.isclose(path[-1][1], goal[1], abs_tol=1e-6)


def _path_cells(path: Sequence[PointLike], resolution: int) -> set[Tuple[int, int]]:
    return {
        (int(point[0] // resolution), int(point[1] // resolution))
        for point in path
    }


def _count_path_chokepoints(path: Sequence[PointLike], chokepoint_zones: Sequence[RectLike]) -> int:
    touched = set()
    for idx, zone in enumerate(chokepoint_zones):
        if any(_point_in_rect((int(px), int(py)), zone) for px, py in path):
            touched.add(idx)
    return len(touched)


def _count_free_components(planner: AStarPlanner) -> int:
    if planner.grid_blocked is None:
        return 0

    visited = set()
    motions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = 0

    for c in range(planner.cols):
        for r in range(planner.rows):
            if planner.grid_blocked[c][r] or (c, r) in visited:
                continue
            components += 1
            stack = [(c, r)]
            visited.add((c, r))
            while stack:
                cc, rr = stack.pop()
                for dc, dr in motions:
                    nc, nr = cc + dc, rr + dr
                    if 0 <= nc < planner.cols and 0 <= nr < planner.rows:
                        if planner.grid_blocked[nc][nr] or (nc, nr) in visited:
                            continue
                        visited.add((nc, nr))
                        stack.append((nc, nr))
    return components


def _compute_route_overlap_score(route_cells: Sequence[set[Tuple[int, int]]]) -> float:
    if len(route_cells) < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(len(route_cells)):
        for j in range(i + 1, len(route_cells)):
            denom = max(1, min(len(route_cells[i]), len(route_cells[j])))
            total += len(route_cells[i] & route_cells[j]) / denom
            count += 1
    return round(total / max(1, count), 4)


def _sample_robot_roles(rng: random.Random, robot_count: int, family: str) -> List[str]:
    base_roles = list(ROBOT_ROLES.keys())
    rng.shuffle(base_roles)
    roles = base_roles[: min(robot_count, len(base_roles))]
    while len(roles) < robot_count:
        if family == "role_mismatch":
            roles.append(rng.choice(["Heavy", "Light", "Maintainer", "Scanner", "Commander"]))
        else:
            roles.append(rng.choice(base_roles))
    rng.shuffle(roles)
    return roles


def _sample_counts(spec: FamilySpec, rng: random.Random, stress: bool) -> Tuple[int, int]:
    if stress:
        robot_count = spec.robot_range[1]
        total_tasks = spec.task_range[1]
    else:
        robot_count = rng.randint(*spec.robot_range)
        total_tasks = rng.randint(*spec.task_range)
    return robot_count, total_tasks


def _sample_sync_count(spec: FamilySpec, total_tasks: int, rng: random.Random, stress: bool) -> int:
    sync_count = spec.sync_range[1] if stress else rng.randint(*spec.sync_range)
    return min(sync_count, total_tasks - 2)


def _build_open_balance_layout(rng: random.Random, stress: bool) -> LayoutBlueprint:
    obstacles = _border_obstacles() + [
        _jitter_rect(_rect(250, 120, 60, 180), rng, dx=20, dy=20, dw=10, dh=15),
        _jitter_rect(_rect(500, 300, 60, 180), rng, dx=20, dy=20, dw=10, dh=15),
        _jitter_rect(_rect(320, 440, 160, 40), rng, dx=20, dy=10, dw=20, dh=5),
    ]
    return LayoutBlueprint(
        obstacles=obstacles,
        robot_zones=[
            _rect(50, 50, 170, 130),
            _rect(580, 50, 170, 130),
            _rect(50, 410, 170, 130),
            _rect(580, 410, 170, 130),
            _rect(315, 500, 170, 50),
        ],
        single_task_zones=[
            _rect(60, 60, 180, 150),
            _rect(560, 60, 180, 150),
            _rect(60, 360, 180, 150),
            _rect(560, 360, 180, 150),
            _rect(300, 60, 180, 120),
            _rect(300, 320, 180, 200),
        ],
        sync_task_zones=[
            _rect(305, 200, 190, 80),
            _rect(290, 295, 220, 80),
        ],
        chokepoint_zones=[],
    )


def _build_role_mismatch_layout(rng: random.Random, stress: bool) -> LayoutBlueprint:
    obstacles = _border_obstacles() + [
        _jitter_rect(_rect(180, 160, 80, 240), rng, dx=10, dy=20, dw=10, dh=15),
        _jitter_rect(_rect(540, 160, 80, 240), rng, dx=10, dy=20, dw=10, dh=15),
        _jitter_rect(_rect(320, 70, 160, 50), rng, dx=15, dy=10, dw=20, dh=5),
        _jitter_rect(_rect(320, 480, 160, 50), rng, dx=15, dy=10, dw=20, dh=5),
    ]
    return LayoutBlueprint(
        obstacles=obstacles,
        robot_zones=[
            _rect(50, 60, 160, 120),
            _rect(50, 420, 160, 120),
            _rect(600, 60, 150, 120),
            _rect(600, 420, 150, 120),
            _rect(325, 240, 150, 120),
        ],
        single_task_zones=[
            _rect(80, 210, 120, 170),
            _rect(600, 210, 120, 170),
            _rect(290, 140, 220, 80),
            _rect(290, 370, 220, 80),
            _rect(250, 250, 300, 120),
        ],
        sync_task_zones=[
            _rect(280, 225, 240, 60),
            _rect(280, 315, 240, 60),
        ],
        chokepoint_zones=[],
    )


def _build_single_bottleneck_layout(rng: random.Random, stress: bool) -> LayoutBlueprint:
    top_h = 205 if stress else 230
    bottom_y = 375 if stress else 350
    obstacles = _border_obstacles() + [
        _jitter_rect(_rect(370, 20, 40, top_h), rng, dx=5, dy=0, dw=0, dh=10),
        _jitter_rect(_rect(370, bottom_y, 40, 580 - bottom_y), rng, dx=5, dy=0, dw=0, dh=10),
        _jitter_rect(_rect(160, 250, 110, 40), rng, dx=20, dy=15, dw=10, dh=5),
        _jitter_rect(_rect(530, 310, 110, 40), rng, dx=20, dy=15, dw=10, dh=5),
    ]
    return LayoutBlueprint(
        obstacles=obstacles,
        robot_zones=[
            _rect(60, 60, 220, 170),
            _rect(60, 360, 220, 170),
            _rect(520, 60, 220, 170),
            _rect(520, 360, 220, 170),
        ],
        single_task_zones=[
            _rect(70, 70, 220, 150),
            _rect(70, 380, 220, 130),
            _rect(520, 70, 210, 150),
            _rect(520, 380, 210, 130),
            _rect(300, 240, 90, 120),
            _rect(410, 240, 90, 120),
        ],
        sync_task_zones=[
            _rect(295, 250, 90, 100),
            _rect(415, 250, 90, 100),
        ],
        chokepoint_zones=[_rect(355, 245, 70, 110)],
    )


def _build_double_bottleneck_layout(rng: random.Random, stress: bool) -> LayoutBlueprint:
    top_segment = 200 if stress else 225
    bottom_segment_y = 385 if stress else 360
    obstacles = _border_obstacles() + [
        _jitter_rect(_rect(255, 20, 40, top_segment), rng, dx=5, dy=0, dw=0, dh=10),
        _jitter_rect(_rect(255, bottom_segment_y, 40, 580 - bottom_segment_y), rng, dx=5, dy=0, dw=0, dh=10),
        _jitter_rect(_rect(505, 20, 40, top_segment), rng, dx=5, dy=0, dw=0, dh=10),
        _jitter_rect(_rect(505, bottom_segment_y, 40, 580 - bottom_segment_y), rng, dx=5, dy=0, dw=0, dh=10),
        _jitter_rect(_rect(320, 270, 160, 40), rng, dx=20, dy=10, dw=10, dh=5),
    ]
    return LayoutBlueprint(
        obstacles=obstacles,
        robot_zones=[
            _rect(60, 60, 160, 150),
            _rect(580, 60, 160, 150),
            _rect(60, 390, 160, 150),
            _rect(580, 390, 160, 150),
            _rect(330, 70, 140, 120),
            _rect(330, 410, 140, 120),
        ],
        single_task_zones=[
            _rect(60, 70, 170, 150),
            _rect(570, 70, 170, 150),
            _rect(60, 380, 170, 140),
            _rect(570, 380, 170, 140),
            _rect(315, 60, 170, 140),
            _rect(315, 380, 170, 140),
        ],
        sync_task_zones=[
            _rect(225, 250, 90, 100),
            _rect(485, 250, 90, 100),
            _rect(340, 220, 120, 150),
        ],
        chokepoint_zones=[_rect(240, 240, 70, 110), _rect(490, 240, 70, 110)],
    )


def _build_far_near_trap_layout(rng: random.Random, stress: bool) -> LayoutBlueprint:
    obstacles = _border_obstacles() + [
        _jitter_rect(_rect(280, 110, 40, 360), rng, dx=10, dy=20, dw=0, dh=15),
        _jitter_rect(_rect(480, 20, 40, 360), rng, dx=10, dy=20, dw=0, dh=15),
        _jitter_rect(_rect(320, 260, 160, 40), rng, dx=10, dy=15, dw=15, dh=5),
    ]
    return LayoutBlueprint(
        obstacles=obstacles,
        robot_zones=[
            _rect(80, 240, 130, 120),
            _rect(590, 240, 130, 120),
            _rect(340, 470, 120, 70),
            _rect(340, 60, 120, 70),
        ],
        single_task_zones=[
            _rect(70, 70, 170, 120),
            _rect(560, 70, 170, 120),
            _rect(70, 390, 170, 120),
            _rect(560, 390, 170, 120),
            _rect(320, 80, 160, 100),
            _rect(320, 390, 160, 100),
        ],
        sync_task_zones=[
            _rect(310, 205, 180, 70),
            _rect(310, 315, 180, 70),
        ],
        chokepoint_zones=[_rect(265, 235, 70, 90), _rect(465, 235, 70, 90)],
    )


def _build_multi_sync_cluster_layout(rng: random.Random, stress: bool) -> LayoutBlueprint:
    obstacles = _border_obstacles() + [
        _jitter_rect(_rect(250, 160, 120, 40), rng, dx=15, dy=10, dw=10, dh=5),
        _jitter_rect(_rect(430, 160, 120, 40), rng, dx=15, dy=10, dw=10, dh=5),
        _jitter_rect(_rect(250, 400, 120, 40), rng, dx=15, dy=10, dw=10, dh=5),
        _jitter_rect(_rect(430, 400, 120, 40), rng, dx=15, dy=10, dw=10, dh=5),
        _jitter_rect(_rect(340, 220, 40, 120), rng, dx=10, dy=15, dw=0, dh=10),
        _jitter_rect(_rect(420, 260, 40, 120), rng, dx=10, dy=15, dw=0, dh=10),
    ]
    return LayoutBlueprint(
        obstacles=obstacles,
        robot_zones=[
            _rect(60, 70, 160, 120),
            _rect(580, 70, 160, 120),
            _rect(60, 420, 160, 120),
            _rect(580, 420, 160, 120),
            _rect(300, 70, 200, 90),
            _rect(300, 430, 200, 90),
        ],
        single_task_zones=[
            _rect(70, 90, 180, 120),
            _rect(550, 90, 180, 120),
            _rect(70, 390, 180, 120),
            _rect(550, 390, 180, 120),
            _rect(300, 70, 200, 90),
            _rect(300, 430, 200, 90),
        ],
        sync_task_zones=[
            _rect(300, 200, 200, 70),
            _rect(300, 280, 200, 70),
            _rect(300, 360, 200, 60),
        ],
        chokepoint_zones=[_rect(320, 210, 60, 140), _rect(400, 240, 60, 140)],
    )


def _build_partial_coalition_trap_layout(rng: random.Random, stress: bool) -> LayoutBlueprint:
    obstacles = _border_obstacles() + [
        _jitter_rect(_rect(300, 120, 40, 320), rng, dx=10, dy=15, dw=0, dh=15),
        _jitter_rect(_rect(460, 80, 40, 320), rng, dx=10, dy=15, dw=0, dh=15),
        _jitter_rect(_rect(340, 260, 140, 40), rng, dx=15, dy=10, dw=10, dh=5),
    ]
    return LayoutBlueprint(
        obstacles=obstacles,
        robot_zones=[
            _rect(80, 230, 110, 120),
            _rect(610, 230, 110, 120),
            _rect(340, 65, 120, 90),
            _rect(340, 445, 120, 90),
            _rect(90, 70, 130, 90),
        ],
        single_task_zones=[
            _rect(90, 170, 140, 120),
            _rect(570, 170, 140, 120),
            _rect(90, 320, 140, 120),
            _rect(570, 320, 140, 120),
            _rect(330, 170, 140, 90),
            _rect(330, 330, 140, 90),
        ],
        sync_task_zones=[
            _rect(350, 240, 120, 80),
            _rect(350, 95, 120, 80),
            _rect(350, 405, 120, 80),
        ],
        chokepoint_zones=[_rect(285, 220, 70, 120), _rect(445, 220, 70, 120)],
    )


LAYOUT_BUILDERS = {
    "open_balance": _build_open_balance_layout,
    "role_mismatch": _build_role_mismatch_layout,
    "single_bottleneck": _build_single_bottleneck_layout,
    "double_bottleneck": _build_double_bottleneck_layout,
    "far_near_trap": _build_far_near_trap_layout,
    "multi_sync_cluster": _build_multi_sync_cluster_layout,
    "partial_coalition_trap": _build_partial_coalition_trap_layout,
}


def _sample_robots(
    rng: random.Random,
    family: str,
    blueprint: LayoutBlueprint,
    robot_count: int,
) -> List[Dict]:
    roles = _sample_robot_roles(rng, robot_count, family)
    robots: List[Dict] = []
    occupied: List[PointLike] = []

    for idx, role in enumerate(roles):
        start = _sample_point_from_zones(
            rng,
            blueprint.robot_zones,
            blueprint.obstacles,
            ROBOT_RADIUS + 12,
            occupied,
            min_pair_dist=ROBOT_RADIUS * 2 + 20,
        )
        occupied.append(start)
        role_cfg = ROBOT_ROLES[role]
        robots.append(
            {
                "id": f"Robot_{idx + 1}",
                "role": role,
                "start_pos": start,
                "speed_multiplier": role_cfg["speed_multiplier"],
                "service_multiplier": role_cfg["service_multiplier"],
                "color": role_cfg["color"],
            }
        )
    return robots


def _select_role_by_distance(
    robots: Sequence[Dict],
    task_pos: PointLike,
    prefer_far: bool,
) -> Optional[str]:
    by_role: Dict[str, float] = {}
    for robot in robots:
        dist = math.hypot(robot["start_pos"][0] - task_pos[0], robot["start_pos"][1] - task_pos[1])
        role = robot["role"]
        best = by_role.get(role)
        if best is None or dist < best:
            by_role[role] = dist
    if not by_role:
        return None
    items = sorted(by_role.items(), key=lambda item: item[1], reverse=prefer_far)
    return items[0][0]


def _choose_single_required_roles(
    rng: random.Random,
    family: str,
    robots: Sequence[Dict],
    task_pos: PointLike,
    ratio: float,
) -> Dict[str, int]:
    if rng.random() > ratio:
        return {}
    if family == "role_mismatch":
        role = _select_role_by_distance(robots, task_pos, prefer_far=True)
    else:
        role = _select_role_by_distance(robots, task_pos, prefer_far=False)
    return {role: 1} if role else {}


def _choose_sync_required_roles(
    rng: random.Random,
    family: str,
    robots: Sequence[Dict],
    task_pos: PointLike,
    stress: bool,
) -> Dict[str, int]:
    candidates = sorted(
        robots,
        key=lambda robot: math.hypot(robot["start_pos"][0] - task_pos[0], robot["start_pos"][1] - task_pos[1]),
        reverse=(family == "role_mismatch"),
    )
    coalition_size = 3 if stress or (len(robots) >= 5 and rng.random() > 0.65) else 2
    selected = candidates[: max(2, min(coalition_size, len(candidates)))]
    req: Dict[str, int] = {}
    for robot in selected:
        req[robot["role"]] = req.get(robot["role"], 0) + 1
    return req


def _sample_service_time(
    rng: random.Random,
    family: str,
    kind: str,
    required_roles: Dict[str, int],
    stress: bool,
) -> int:
    base = rng.randint(80, 180)
    if kind == "sync":
        base += 40 * max(1, sum(required_roles.values()) - 1)
    if required_roles and kind == "single":
        base += 20
    if family in {"single_bottleneck", "double_bottleneck", "multi_sync_cluster"}:
        base += 25
    if stress:
        base += 30
    return int(base)


def _apply_precedence(
    tasks: List[Dict],
    robots: Sequence[Dict],
    family: str,
    rng: random.Random,
    stress: bool,
) -> None:
    if family == "far_near_trap" and len(tasks) >= 3:
        centroid_x = sum(robot["start_pos"][0] for robot in robots) / len(robots)
        centroid_y = sum(robot["start_pos"][1] for robot in robots) / len(robots)
        ordered = sorted(
            tasks,
            key=lambda task: math.hypot(task["pos"][0] - centroid_x, task["pos"][1] - centroid_y),
        )
        far_task = ordered[-1]
        near_candidates = ordered[: 2 if stress else 1]
        far_task["priority"] = 2.0
        for task in near_candidates:
            task["precedence"] = [far_task["id"]]
            task["priority"] = max(task["priority"], 1.4)
    elif family == "multi_sync_cluster" and stress:
        sync_tasks = [task for task in tasks if task["kind"] == "sync"]
        if len(sync_tasks) >= 2:
            sync_tasks[1]["precedence"] = [sync_tasks[0]["id"]]
            sync_tasks[0]["priority"] = 1.9
            sync_tasks[1]["priority"] = 1.8
    elif family == "multi_sync_cluster" and rng.random() > 0.65:
        sync_tasks = [task for task in tasks if task["kind"] == "sync"]
        single_tasks = [task for task in tasks if task["kind"] == "single"]
        if sync_tasks and single_tasks:
            single_tasks[0]["precedence"] = [sync_tasks[0]["id"]]


def _apply_partial_coalition_trap(
    tasks: List[Dict],
    robots: Sequence[Dict],
) -> None:
    sync_tasks = [task for task in tasks if task["kind"] == "sync"]
    single_tasks = [task for task in tasks if task["kind"] == "single"]
    if len(sync_tasks) < 2 or len(robots) < 3:
        return

    main_sync = min(sync_tasks, key=lambda task: abs(task["pos"][1] - 300))
    side_syncs = [task for task in sync_tasks if task["id"] != main_sync["id"]]
    support_sync = side_syncs[0]

    ordered_robots = sorted(
        robots,
        key=lambda robot: math.hypot(robot["start_pos"][0] - main_sync["pos"][0], robot["start_pos"][1] - main_sync["pos"][1]),
    )
    near_robots = ordered_robots[:2]
    far_robot = ordered_robots[-1]

    main_required: Dict[str, int] = {}
    for robot in near_robots + [far_robot]:
        main_required[robot["role"]] = main_required.get(robot["role"], 0) + 1
    main_sync["required_roles"] = main_required
    main_sync["priority"] = 1.9

    support_required: Dict[str, int] = {far_robot["role"]: 1}
    if len(ordered_robots) >= 4:
        support_required[ordered_robots[-2]["role"]] = support_required.get(ordered_robots[-2]["role"], 0) + 1
    support_sync["required_roles"] = support_required
    support_sync["priority"] = 1.7

    nearby_singles = sorted(
        single_tasks,
        key=lambda task: math.hypot(task["pos"][0] - main_sync["pos"][0], task["pos"][1] - main_sync["pos"][1]),
    )
    for single_task, robot in zip(nearby_singles[:2], near_robots):
        single_task["required_roles"] = {robot["role"]: 1}
        single_task["priority"] = max(float(single_task["priority"]), 1.25)


def _build_tasks(
    rng: random.Random,
    family: str,
    robots: Sequence[Dict],
    blueprint: LayoutBlueprint,
    total_tasks: int,
    sync_count: int,
    stress: bool,
) -> List[Dict]:
    single_count = total_tasks - sync_count
    occupied: List[PointLike] = [tuple(robot["start_pos"]) for robot in robots]
    tasks: List[Dict] = []
    spec = FAMILY_SPECS[family]

    for idx in range(single_count):
        pos = _sample_point_from_zones(
            rng,
            blueprint.single_task_zones,
            blueprint.obstacles,
            ROBOT_RADIUS + 18,
            occupied,
            min_pair_dist=70.0,
        )
        occupied.append(pos)
        required_roles = _choose_single_required_roles(
            rng,
            family,
            robots,
            pos,
            spec.single_role_ratio + (0.10 if stress else 0.0),
        )
        tasks.append(
            {
                "id": f"Task {chr(ord('A') + idx)}",
                "kind": "single",
                "pos": pos,
                "service_time": _sample_service_time(rng, family, "single", required_roles, stress),
                "required_roles": required_roles,
                "precedence": [],
                "priority": 1.0,
            }
        )

    for idx in range(sync_count):
        pos = _sample_point_from_zones(
            rng,
            blueprint.sync_task_zones,
            blueprint.obstacles,
            ROBOT_RADIUS + int(DEFAULT_DOCKING_RADIUS) + int(DEFAULT_PLANNER_MARGIN),
            occupied,
            min_pair_dist=65.0,
        )
        occupied.append(pos)
        required_roles = _choose_sync_required_roles(rng, family, robots, pos, stress)
        tasks.append(
            {
                "id": f"Sync {idx + 1}",
                "kind": "sync",
                "pos": pos,
                "service_time": _sample_service_time(rng, family, "sync", required_roles, stress),
                "required_roles": required_roles,
                "precedence": [],
                "priority": 1.5 + 0.2 * len(required_roles),
            }
        )

    _apply_precedence(tasks, robots, family, rng, stress)
    if family == "partial_coalition_trap":
        _apply_partial_coalition_trap(tasks, robots)
    return tasks


def _validate_sync_docking_slots(
    tasks: Sequence[Dict],
    obstacles: Sequence[RectLike],
) -> bool:
    for task in tasks:
        if task.get("kind") != "sync":
            continue
        slot_count = max(1, sum(task.get("required_roles", {}).values()))
        slots = generate_docking_slots(
            task_pos=task["pos"],
            slot_count=slot_count,
            obstacles=obstacles,
            width=WIDTH,
            height=HEIGHT,
            robot_radius=ROBOT_RADIUS,
            docking_radius=DEFAULT_DOCKING_RADIUS,
            planner_margin=DEFAULT_PLANNER_MARGIN,
            planner_resolution=DEFAULT_PLANNER_RESOLUTION,
        )
        if len(slots) < slot_count:
            return False
    return True


def _build_distance_matrix(
    robots: Sequence[Dict],
    tasks: Sequence[Dict],
    obstacles: Sequence[RectLike],
    chokepoint_zones: Sequence[RectLike],
) -> Optional[Tuple[Dict, Dict, Dict]]:
    planner = AStarPlanner(
        width=WIDTH,
        height=HEIGHT,
        resolution=10,
        robot_radius=ROBOT_RADIUS,
        margin=5,
    )
    robot_to_task: Dict[str, Dict[str, Dict[str, float]]] = {}
    task_to_task: Dict[str, Dict[str, Dict[str, float]]] = {}
    route_cells: List[set[Tuple[int, int]]] = []
    total_route_chokepoints = 0

    for robot in robots:
        robot_to_task[robot["id"]] = {}
        for task in tasks:
            path = planner.plan(robot["start_pos"], task["pos"], list(obstacles))
            if not _path_reaches_goal(path, task["pos"]):
                return None
            path_length = _path_length(path)
            chokepoints = _count_path_chokepoints(path, chokepoint_zones)
            route_cells.append(_path_cells(path, planner.resolution))
            total_route_chokepoints += chokepoints
            robot_to_task[robot["id"]][task["id"]] = {
                "path_length": round(path_length, 2),
                "eta": round(path_length / max(0.05, BASE_TRAVEL_SPEED * robot["speed_multiplier"]), 2),
                "chokepoints": chokepoints,
            }

    for source in tasks:
        task_to_task[source["id"]] = {}
        for target in tasks:
            if source["id"] == target["id"]:
                task_to_task[source["id"]][target["id"]] = {
                    "path_length": 0.0,
                    "base_eta": 0.0,
                    "chokepoints": 0,
                }
                continue

            path = planner.plan(source["pos"], target["pos"], list(obstacles))
            if not _path_reaches_goal(path, target["pos"]):
                return None
            path_length = _path_length(path)
            chokepoints = _count_path_chokepoints(path, chokepoint_zones)
            route_cells.append(_path_cells(path, planner.resolution))
            total_route_chokepoints += chokepoints
            task_to_task[source["id"]][target["id"]] = {
                "path_length": round(path_length, 2),
                "base_eta": round(path_length / BASE_TRAVEL_SPEED, 2),
                "chokepoints": chokepoints,
            }

    meta = {
        "chokepoint_count": len(chokepoint_zones),
        "route_overlap_score": _compute_route_overlap_score(route_cells),
        "component_count": _count_free_components(planner),
        "route_chokepoint_load": total_route_chokepoints,
    }
    return robot_to_task, task_to_task, meta


def validate_scenario(scenario_config: Dict) -> bool:
    tasks = scenario_config.get("tasks", [])
    robots = scenario_config.get("robots", [])
    if not tasks or not robots:
        return False
    if not _validate_sync_docking_slots(tasks, scenario_config.get("obstacles", [])):
        return False

    result = _build_distance_matrix(
        robots=robots,
        tasks=tasks,
        obstacles=scenario_config.get("obstacles", []),
        chokepoint_zones=scenario_config.get("difficulty_meta", {}).get("chokepoint_zones", []),
    )
    if result is None:
        return False
    _, _, meta = result
    return meta["component_count"] == 1


def _legacy_fallback_scenario() -> Dict:
    return {
        "obstacles": _border_obstacles(),
        "tasks": {
            "Task A": (120, 120),
            "Task B": (680, 120),
            "Task C": (120, 480),
            "Task D": (680, 480),
        },
        "coop_tasks_dict": {"Sync 1": (400, 300)},
        "coop_tasks": ["Sync 1"],
        "coop_tasks_req": {"Sync 1": {"Heavy": 1, "Light": 1}},
        "base_times": {
            "Task A": 120,
            "Task B": 120,
            "Task C": 120,
            "Task D": 120,
            "Sync 1": 180,
        },
        "robots": [
            {
                "name": "Robot_1",
                "role": "Heavy",
                "start_pos": (120, 300),
                "color": ROBOT_ROLES["Heavy"]["color"],
                "multiplier": 2.5,
                "task_sequence": ["Task A", "Sync 1", "Task D"],
            },
            {
                "name": "Robot_2",
                "role": "Light",
                "start_pos": (680, 300),
                "color": ROBOT_ROLES["Light"]["color"],
                "multiplier": 0.5,
                "task_sequence": ["Task B", "Sync 1", "Task C"],
            },
        ],
    }


def generate_scenario(
    seed: int,
    family: str,
    split: str = "train",
    scenario_id: Optional[str] = None,
    max_retries: int = 120,
    visualize: bool = False,
    visualize_hold_ms: int = 3000,
) -> Dict:
    if family not in FAMILY_SPECS:
        raise ValueError(f"未知 family: {family}。可选值: {', '.join(FAMILY_NAMES)}")

    spec = FAMILY_SPECS[family]
    stress = split == "stress"
    master_rng = random.Random(seed)

    for _ in range(max_retries):
        try:
            attempt_seed = master_rng.randint(0, 2**31 - 1)
            rng = random.Random(attempt_seed)
            blueprint = LAYOUT_BUILDERS[family](rng, stress)

            robot_count, total_tasks = _sample_counts(spec, rng, stress)
            sync_count = _sample_sync_count(spec, total_tasks, rng, stress)
            robots = _sample_robots(rng, family, blueprint, robot_count)
            tasks = _build_tasks(rng, family, robots, blueprint, total_tasks, sync_count, stress)
        except RuntimeError:
            continue

        if not _validate_sync_docking_slots(tasks, blueprint.obstacles):
            continue

        matrix_result = _build_distance_matrix(robots, tasks, blueprint.obstacles, blueprint.chokepoint_zones)
        if matrix_result is None:
            continue
        robot_to_task, task_to_task, difficulty_meta = matrix_result
        if difficulty_meta["component_count"] != 1:
            continue

        scenario = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario_id or f"{split}_{family}_{seed}",
            "split": split,
            "family": family,
            "width": WIDTH,
            "height": HEIGHT,
            "obstacles": list(blueprint.obstacles),
            "robots": robots,
            "tasks": tasks,
            "distance_matrix": {
                "robot_to_task": robot_to_task,
                "task_to_task": task_to_task,
            },
            "difficulty_meta": {
                **difficulty_meta,
                "chokepoint_zones": list(blueprint.chokepoint_zones),
            },
            "seed": seed,
            "attempt_seed": attempt_seed,
        }

        if visualize:
            draw_scenario(scenario, hold_ms=visualize_hold_ms)
        return scenario

    raise RuntimeError(f"无法为 family={family}, split={split}, seed={seed} 生成合法场景。")


def _task_label(task: Dict) -> str:
    req = task.get("required_roles", {})
    req_text = " ".join(f"{role[0]}:{count}" for role, count in req.items())
    prefix = "Sync" if task["kind"] == "sync" else "Task"
    suffix = f" [{req_text}]" if req_text else ""
    precedence = f" <- {','.join(task['precedence'])}" if task.get("precedence") else ""
    return f"{prefix} {task['id'].split()[-1]} ({task['service_time']}){suffix}{precedence}"


def draw_scenario(
    scenario_config: Dict,
    hold_ms: int = 1800,
    window_title: str = "Scenario Preview",
) -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(window_title)
    font = pygame.font.SysFont("arial", 16, bold=True)
    small_font = pygame.font.SysFont("arial", 13)
    clock = pygame.time.Clock()

    tasks = scenario_config.get("tasks", [])
    robots = scenario_config.get("robots", [])
    obstacles = [pygame.Rect(*rect) for rect in scenario_config.get("obstacles", [])]
    start_tick = pygame.time.get_ticks()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if hold_ms > 0 and pygame.time.get_ticks() - start_tick >= hold_ms:
            running = False

        screen.fill(BG_COLOR)
        for obs in obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, obs)

        for task in tasks:
            tx, ty = task["pos"]
            color = SYNC_TASK_COLOR if task["kind"] == "sync" else SINGLE_TASK_COLOR
            pygame.draw.circle(screen, color, (int(tx), int(ty)), 14)
            pygame.draw.circle(screen, OBSTACLE_COLOR, (int(tx), int(ty)), 14, 2)
            label = small_font.render(_task_label(task), True, TEXT_COLOR)
            screen.blit(label, (int(tx) + 12, int(ty) - 8))

        for robot in robots:
            rx, ry = robot["start_pos"]
            color = tuple(robot["color"])
            pygame.draw.circle(screen, color, (int(rx), int(ry)), ROBOT_RADIUS)
            pygame.draw.circle(screen, OBSTACLE_COLOR, (int(rx), int(ry)), ROBOT_RADIUS, 2)
            info = small_font.render(
                f"{robot['id']} [{robot['role']}] v{robot['speed_multiplier']:.2f} s{robot['service_multiplier']:.2f}",
                True,
                TEXT_COLOR,
            )
            screen.blit(info, (int(rx) + 14, int(ry) - 8))

        title = font.render(
            f"{scenario_config.get('split', '?')} / {scenario_config.get('family', '?')} / {scenario_config.get('scenario_id', '?')}",
            True,
            TEXT_COLOR,
        )
        screen.blit(title, (24, 22))

        overlap = scenario_config.get("difficulty_meta", {}).get("route_overlap_score", 0.0)
        meta_text = font.render(f"Overlap={overlap:.3f}", True, TEXT_COLOR)
        screen.blit(meta_text, (24, 46))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _load_legacy_random_scenario(cache_dir: Path) -> Dict:
    map_files = list(cache_dir.glob("*.pkl"))
    if not map_files:
        return _legacy_fallback_scenario()
    chosen = random.choice(map_files)
    with chosen.open("rb") as handle:
        return pickle.load(handle)


def _resolve_cache_path(cache_dir: str | Path) -> Path:
    cache_path = Path(cache_dir)
    if cache_path.exists():
        return cache_path
    fallback = (_CURRENT_DIR / cache_path).resolve()
    return fallback if fallback.exists() else cache_path


def load_random_scenario(
    cache_dir: str | Path = "offline_maps_v2",
    split: str = "train",
    family: Optional[str] = None,
) -> Dict:
    cache_path = _resolve_cache_path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"找不到地图目录: {cache_path}")

    target_dir = cache_path / split
    if family:
        target_dir = target_dir / family

    if not target_dir.exists():
        if cache_path.name == "offline_maps" and split == "train":
            return _load_legacy_random_scenario(cache_path)
        raise FileNotFoundError(f"找不到 split/family 目录: {target_dir}")

    candidates = list(target_dir.rglob("*.pkl"))
    if not candidates:
        raise ValueError(f"目录中没有找到 .pkl 场景文件: {target_dir}")

    chosen = random.choice(candidates)
    with chosen.open("rb") as handle:
        return pickle.load(handle)


def load_scenarios(
    cache_dir: str | Path,
    split: Optional[str] = None,
    family: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    cache_path = _resolve_cache_path(cache_dir)
    if split:
        cache_path = cache_path / split
    if family:
        cache_path = cache_path / family

    files = sorted(cache_path.rglob("*.pkl"))
    if limit is not None:
        files = files[:limit]

    scenarios = []
    for file_path in files:
        with file_path.open("rb") as handle:
            scenarios.append(pickle.load(handle))
    return scenarios


def summarize_scenario(scenario_config: Dict) -> Dict[str, int | float | str]:
    tasks = scenario_config.get("tasks", [])
    sync_count = sum(task["kind"] == "sync" for task in tasks)
    return {
        "scenario_id": scenario_config.get("scenario_id", "unknown"),
        "family": scenario_config.get("family", "unknown"),
        "split": scenario_config.get("split", "unknown"),
        "robot_count": len(scenario_config.get("robots", [])),
        "task_count": len(tasks),
        "sync_task_count": sync_count,
        "route_overlap_score": scenario_config.get("difficulty_meta", {}).get("route_overlap_score", 0.0),
        "component_count": scenario_config.get("difficulty_meta", {}).get("component_count", 0),
    }


if __name__ == "__main__":
    demo = generate_scenario(seed=42, family="multi_sync_cluster", split="train", visualize=True, visualize_hold_ms=5000)
    print("生成场景成功：")
    print(summarize_scenario(demo))
