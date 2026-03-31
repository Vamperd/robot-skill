from __future__ import annotations

import math
from itertools import combinations
from typing import List, Sequence, Tuple


RectLike = Tuple[int, int, int, int]
PointLike = Tuple[float, float]

DEFAULT_DOCKING_RADIUS = 18.0
DEFAULT_SLOT_CAPTURE_RADIUS = 8.0
DEFAULT_SLOT_ANGLE_STEP_DEGREES = 15.0
DEFAULT_PLANNER_MARGIN = 5.0
DEFAULT_PLANNER_RESOLUTION = 10

EPS = 1e-6


def rect_distance(point: PointLike, rect: RectLike) -> float:
    px, py = point
    rx, ry, rw, rh = rect
    closest_x = max(rx, min(px, rx + rw))
    closest_y = max(ry, min(py, ry + rh))
    return math.hypot(px - closest_x, py - closest_y)


def circle_hits_rect(point: PointLike, radius: float, rect: RectLike) -> bool:
    return rect_distance(point, rect) < radius


def point_in_bounds(
    point: PointLike,
    width: int,
    height: int,
    clearance: float,
) -> bool:
    px, py = point
    return clearance <= px <= width - clearance and clearance <= py <= height - clearance


def planner_goal_is_free(
    point: PointLike,
    obstacles: Sequence[RectLike],
    width: int,
    height: int,
    robot_radius: float,
    margin: float = DEFAULT_PLANNER_MARGIN,
    resolution: int = DEFAULT_PLANNER_RESOLUTION,
) -> bool:
    if not point_in_bounds(point, width, height, robot_radius):
        return False

    cols = int(width / resolution)
    rows = int(height / resolution)
    col = int(min(max(point[0] / resolution, 0), cols - 1))
    row = int(min(max(point[1] / resolution, 0), rows - 1))
    cell_center = (
        col * resolution + resolution / 2.0,
        row * resolution + resolution / 2.0,
    )
    safe_dist = robot_radius + margin
    return all(rect_distance(cell_center, rect) >= safe_dist for rect in obstacles)


def _pairwise_min_distance(points: Sequence[PointLike]) -> float:
    if len(points) < 2:
        return float("inf")
    return min(
        math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    )


def generate_docking_slots(
    task_pos: PointLike,
    slot_count: int,
    obstacles: Sequence[RectLike],
    width: int,
    height: int,
    robot_radius: float,
    docking_radius: float = DEFAULT_DOCKING_RADIUS,
    angle_step_degrees: float = DEFAULT_SLOT_ANGLE_STEP_DEGREES,
    planner_margin: float = DEFAULT_PLANNER_MARGIN,
    planner_resolution: int = DEFAULT_PLANNER_RESOLUTION,
) -> List[PointLike]:
    if slot_count <= 0:
        return []

    candidates: List[Tuple[float, float, float]] = []
    angle = 0.0
    while angle < 360.0 - EPS:
        angle_rad = math.radians(angle)
        point = (
            task_pos[0] + docking_radius * math.cos(angle_rad),
            task_pos[1] + docking_radius * math.sin(angle_rad),
        )
        if (
            point_in_bounds(point, width, height, robot_radius)
            and not any(circle_hits_rect(point, robot_radius, rect) for rect in obstacles)
            and planner_goal_is_free(
                point,
                obstacles,
                width=width,
                height=height,
                robot_radius=robot_radius,
                margin=planner_margin,
                resolution=planner_resolution,
            )
        ):
            candidates.append((round(point[0], 3), round(point[1], 3), angle))
        angle += angle_step_degrees

    if len(candidates) < slot_count:
        return []

    best_combo: Tuple[Tuple[float, float, float], ...] | None = None
    best_score = float("-inf")
    required_spacing = 2.0 * robot_radius

    for combo in combinations(candidates, slot_count):
        combo_points = [(item[0], item[1]) for item in combo]
        min_dist = _pairwise_min_distance(combo_points)
        if min_dist + EPS < required_spacing:
            continue
        if min_dist > best_score + EPS:
            best_score = min_dist
            best_combo = combo

    if best_combo is None:
        return []

    ordered = sorted(best_combo, key=lambda item: item[2])
    return [(item[0], item[1]) for item in ordered]
