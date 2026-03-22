from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pygame
import torch

from attention_policy import load_scheduler_checkpoint
from scenario_generator import BG_COLOR, OBSTACLE_COLOR, TEXT_COLOR, load_random_scenario, load_scenarios
from scheduler_training_utils import policy_action
from sequential_scheduling_env import SequentialSchedulingEnv


LOCKED_COLOR = (160, 160, 160)
AVAILABLE_COLOR = (255, 215, 0)
ACTIVE_COLOR = (80, 150, 255)
COMPLETE_COLOR = (80, 210, 120)
WAITING_COLOR = (255, 140, 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化高层调度事件，不运行底层连续物理。")
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--split", default="val")
    parser.add_argument("--family", default=None)
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--policy", default="role_aware_greedy", choices=["role_aware_greedy", "random", "model"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--decision-delay-ms", type=int, default=350)
    parser.add_argument("--gif-name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_scenario(args: argparse.Namespace) -> dict:
    if args.scenario_file:
        with Path(args.scenario_file).open("rb") as handle:
            return pickle.load(handle)
    scenarios = load_scenarios(args.scenario_dir, split=args.split, family=args.family)
    if not scenarios:
        return load_random_scenario(cache_dir=args.scenario_dir, split=args.split, family=args.family)
    return scenarios[args.scenario_index % len(scenarios)]


def draw_arrow(screen, start, end, color):
    pygame.draw.line(screen, color, start, end, 2)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    left = (end[0] - 10 * ux + 5 * uy, end[1] - 10 * uy - 5 * ux)
    right = (end[0] - 10 * ux - 5 * uy, end[1] - 10 * uy + 5 * ux)
    pygame.draw.polygon(screen, color, [end, left, right])


def draw_scene(screen, font, small_font, env: SequentialSchedulingEnv, obs: dict) -> None:
    screen.fill(BG_COLOR)
    scenario = env.base_env.current_scenario
    for rect in scenario.get("obstacles", []):
        pygame.draw.rect(screen, OBSTACLE_COLOR, rect)

    pending = env.pending_task_assignments()
    current_robot_id = env.get_current_robot_id()

    task_by_id = {task["id"]: task for task in scenario["tasks"]}
    for task in scenario["tasks"]:
        task_id = task["id"]
        state = env.base_env.task_states[task_id]
        pos = (int(task["pos"][0]), int(task["pos"][1]))
        if state["completed"]:
            color = COMPLETE_COLOR
        elif not env.base_env._precedence_satisfied(task_id):
            color = LOCKED_COLOR
        elif state["contributors"]:
            color = ACTIVE_COLOR
        elif state["assigned_robot_ids"] or state["onsite_robot_ids"]:
            color = WAITING_COLOR
        else:
            color = AVAILABLE_COLOR

        pygame.draw.circle(screen, color, pos, 18)
        pygame.draw.circle(screen, OBSTACLE_COLOR, pos, 18, 2)
        label = font.render(task_id.split()[-1], True, (255, 255, 255))
        screen.blit(label, label.get_rect(center=pos))

        progress = state["progress"] / max(float(task["service_time"]), 1.0)
        bar_x, bar_y = pos[0] - 22, pos[1] + 22
        pygame.draw.rect(screen, OBSTACLE_COLOR, (bar_x, bar_y, 44, 6), 1)
        pygame.draw.rect(screen, COMPLETE_COLOR, (bar_x + 1, bar_y + 1, int(42 * min(progress, 1.0)), 4))

        req = " ".join(f"{role[0]}:{count}" for role, count in task.get("required_roles", {}).items())
        info = small_font.render(f"{task['kind']} {req}", True, TEXT_COLOR)
        screen.blit(info, (pos[0] - 25, pos[1] - 36))

    for task in scenario["tasks"]:
        for parent in task.get("precedence", []):
            parent_task = task_by_id[parent]
            draw_arrow(
                screen,
                (int(parent_task["pos"][0]), int(parent_task["pos"][1])),
                (int(task["pos"][0]), int(task["pos"][1])),
                (90, 90, 90),
            )

    for robot in scenario["robots"]:
        robot_id = robot["id"]
        state = env.base_env.robot_states[robot_id]
        pos = (int(state["position"][0]), int(state["position"][1]))
        color = tuple(robot["color"])
        pygame.draw.circle(screen, color, pos, 15)
        border = (255, 255, 255) if robot_id == current_robot_id else OBSTACLE_COLOR
        pygame.draw.circle(screen, border, pos, 15, 3)
        label = font.render(robot["role"][0], True, (255, 255, 255))
        screen.blit(label, label.get_rect(center=pos))

        assigned = state.get("assigned_task")
        pending_task = pending.get(robot_id)
        target_task = assigned or pending_task
        if target_task is not None:
            target_pos = task_by_id[target_task]["pos"]
            draw_arrow(screen, pos, (int(target_pos[0]), int(target_pos[1])), color)

        status_text = f"{robot_id} {state['status']}"
        if target_task is not None:
            status_text += f" -> {target_task}"
        if state.get("wait_elapsed", 0.0) > 0.0:
            status_text += f" w={state['wait_elapsed']:.1f}"
        info = small_font.render(status_text, True, TEXT_COLOR)
        screen.blit(info, (pos[0] + 18, pos[1] - 12))

    title = font.render(
        f"{scenario['scenario_id']} | event={env.event_index} | completion={env.base_env.metrics['completed_tasks']}/{len(env.task_order)}",
        True,
        TEXT_COLOR,
    )
    screen.blit(title, (12, 12))
    line2 = font.render(
        f"time={env.base_env.time:.1f} deadlock={env.base_env.metrics['deadlock_events']} timeout={env.base_env.metrics['timeout_events']}",
        True,
        TEXT_COLOR,
    )
    screen.blit(line2, (12, 36))
    line3 = font.render(
        f"current_robot={current_robot_id or '-'} action_mask={int(np.sum(obs['current_action_mask']))} pending={len(pending)}",
        True,
        TEXT_COLOR,
    )
    screen.blit(line3, (12, 60))


def main() -> None:
    args = parse_args()
    scenario = load_scenario(args)
    env = SequentialSchedulingEnv(scenarios=[scenario])
    obs, _ = env.reset(options={"scenario": scenario})

    device = torch.device(args.device)
    model = None
    if args.policy == "model":
        if not args.model:
            raise ValueError("选择 --policy model 时必须提供 --model。")
        model, _ = load_scheduler_checkpoint(args.model, device=device)

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Scheduler Episode Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18, bold=True)
    small_font = pygame.font.SysFont("arial", 13)
    rng = np.random.default_rng(0)
    last_step_tick = pygame.time.get_ticks()
    running = True
    frames = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now = pygame.time.get_ticks()
        if not (env.terminated or env.truncated) and now - last_step_tick >= args.decision_delay_ms:
            selected_policy = model if model is not None else args.policy
            action = policy_action(env, obs, selected_policy, rng=rng, device=device, deterministic=True)
            obs, _, _, _, _ = env.step(action)
            last_step_tick = now

        draw_scene(screen, font, small_font, env, obs)
        pygame.display.flip()

        if args.gif_name:
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)

        if env.terminated or env.truncated:
            pygame.time.delay(1200)
            running = False

        clock.tick(30)

    pygame.quit()
    if args.gif_name and frames:
        import imageio

        imageio.mimsave(args.gif_name, frames, duration=1 / 15)


if __name__ == "__main__":
    main()
