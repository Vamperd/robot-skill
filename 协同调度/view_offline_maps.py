import argparse
import sys
from pathlib import Path

import pygame

from scenario_generator import (
    BG_COLOR,
    OBSTACLE_COLOR,
    ROBOT_RADIUS,
    SINGLE_TASK_COLOR,
    SYNC_TASK_COLOR,
    TEXT_COLOR,
    draw_scenario as preview_scenario,
    load_random_scenario,
)


WHITE = (240, 240, 240)


def _collect_candidates(cache_dir: Path, split: str, family: str | None) -> list[Path]:
    target = cache_dir / split
    if family:
        target = target / family
    return sorted(target.rglob("*.pkl"))


def _task_req_text(task: dict) -> str:
    req = task.get("required_roles", {})
    if not req:
        return ""
    return " ".join(f"{role[0]}:{count}" for role, count in req.items())


def draw_scene(screen, font, small_font, scenario: dict) -> None:
    screen.fill(BG_COLOR)

    for rect in scenario.get("obstacles", []):
        pygame.draw.rect(screen, OBSTACLE_COLOR, rect)

    for task in scenario.get("tasks", []):
        px, py = int(task["pos"][0]), int(task["pos"][1])
        color = SYNC_TASK_COLOR if task["kind"] == "sync" else SINGLE_TASK_COLOR
        pygame.draw.circle(screen, color, (px, py), 18)
        pygame.draw.circle(screen, OBSTACLE_COLOR, (px, py), 18, 2)

        label = task["id"].split()[-1]
        text = font.render(label, True, WHITE)
        screen.blit(text, text.get_rect(center=(px, py)))

        req_text = _task_req_text(task)
        precedence = ",".join(task.get("precedence", []))
        footer = f"{task['kind']} t={task['service_time']}"
        if req_text:
            footer += f" [{req_text}]"
        if precedence:
            footer += f" <- {precedence}"
        info = small_font.render(footer, True, TEXT_COLOR)
        screen.blit(info, (px - 30, py + 24))

    for robot in scenario.get("robots", []):
        px, py = int(robot["start_pos"][0]), int(robot["start_pos"][1])
        color = tuple(robot["color"])
        pygame.draw.circle(screen, color, (px, py), ROBOT_RADIUS)
        pygame.draw.circle(screen, OBSTACLE_COLOR, (px, py), ROBOT_RADIUS, 2)

        role_initial = robot.get("role", "?")[0]
        role_text = font.render(role_initial, True, WHITE)
        screen.blit(role_text, role_text.get_rect(center=(px, py)))

        detail = small_font.render(
            f"{robot['id']} v{robot['speed_multiplier']:.2f} s{robot['service_multiplier']:.2f}",
            True,
            TEXT_COLOR,
        )
        screen.blit(detail, (px + 18, py - 10))

    top_line = font.render(
        f"{scenario.get('split', '?')} | {scenario.get('family', '?')} | {scenario.get('scenario_id', '?')}",
        True,
        TEXT_COLOR,
    )
    screen.blit(top_line, (14, 12))

    meta = scenario.get("difficulty_meta", {})
    bottom = font.render(
        f"overlap={meta.get('route_overlap_score', 0.0):.3f}  chokepoints={meta.get('chokepoint_count', 0)}  components={meta.get('component_count', 0)}",
        True,
        TEXT_COLOR,
    )
    screen.blit(bottom, (14, 36))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看协同调度 v2 数据集场景。")
    parser.add_argument("--cache-dir", default="offline_maps_v2", help="数据目录，默认 offline_maps_v2")
    parser.add_argument("--split", default="train", help="要查看的 split，默认 train")
    parser.add_argument("--family", default=None, help="可选，限定某个 family")
    parser.add_argument("--single-shot", action="store_true", help="只调用一次预览窗口，不进入循环浏览")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    if args.single_shot:
        scenario = load_random_scenario(cache_dir=cache_dir, split=args.split, family=args.family)
        preview_scenario(scenario, hold_ms=0, window_title="Offline Map Preview")
        return

    candidates = _collect_candidates(cache_dir, args.split, args.family)
    if not candidates:
        print(f"没有找到场景文件: {cache_dir / args.split}")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Offline Map Viewer (SPACE to refresh)")
    font = pygame.font.SysFont("arial", 18, bold=True)
    small_font = pygame.font.SysFont("arial", 13)
    clock = pygame.time.Clock()

    try:
        current = load_random_scenario(cache_dir=cache_dir, split=args.split, family=args.family)
    except Exception as exc:
        print(f"加载失败: {exc}")
        pygame.quit()
        sys.exit(1)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                current = load_random_scenario(cache_dir=cache_dir, split=args.split, family=args.family)

        draw_scene(screen, font, small_font, current)
        tip = font.render("SPACE: next map", True, (100, 100, 100))
        screen.blit(tip, (14, 565))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
