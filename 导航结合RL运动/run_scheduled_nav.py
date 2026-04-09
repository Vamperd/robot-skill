from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from low_level_policy_adapter import LowLevelPolicyAdapter
from scheduler_nav_runner import SchedulerNavRunner


CURRENT_DIR = Path(__file__).resolve().parent
SCHED_DIR = CURRENT_DIR.parent / "协同调度"
if str(SCHED_DIR) not in sys.path:
    sys.path.insert(0, str(SCHED_DIR))

from scenario_generator import load_scenarios  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行高层调度 + A* + 底层 PPO 的多机器人联调。")
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--split", default="stress")
    parser.add_argument("--family", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--scheduler-model", default=None, help="高层调度器 checkpoint；为空时使用 role_aware_greedy")
    parser.add_argument(
        "--scheduler-policy",
        default="role_aware_greedy",
        choices=["role_aware_greedy", "wait_aware_role_greedy", "upfront_wait_aware_greedy", "random"],
    )
    parser.add_argument(
        "--scheduler-policy-type",
        default="auto",
        choices=["auto", "legacy", "hetero_ppo", "hetero_actor_only", "hetero_ranker"],
        help="当提供 scheduler-model 时，指定或自动识别高层模型类型。",
    )
    parser.add_argument("--low-level-model", default="导航结合RL运动/results/local_eval_best/best_model.zip")
    parser.add_argument(
        "--scheduler-guard-mode",
        default="auto",
        choices=["auto", "off", "hard_only"],
        help="Conservative runtime guard mode for hetero_ranker deployment.",
    )
    parser.add_argument("--scheduler-min-margin", type=float, default=0.15)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gif-name", default=None)
    parser.add_argument("--max-frames", type=int, default=2500)
    return parser.parse_args()


def load_target_scenarios(args: argparse.Namespace):
    if args.scenario_file:
        with Path(args.scenario_file).open("rb") as handle:
            return [pickle.load(handle)]
    return load_scenarios(cache_dir=args.scenario_dir, split=args.split, family=args.family, limit=args.limit)


def main() -> None:
    args = parse_args()
    scenarios = load_target_scenarios(args)
    if not scenarios:
        raise ValueError("没有可运行的场景，请检查 scenario-file 或 scenario-dir/split/family。")

    low_level_adapter = LowLevelPolicyAdapter.from_model(args.low_level_model)
    scheduler_policy = (
        SchedulerNavRunner.load_scheduler(args.scheduler_model, policy_type=args.scheduler_policy_type)
        if args.scheduler_model
        else args.scheduler_policy
    )
    runner = SchedulerNavRunner(
        scheduler_policy=scheduler_policy,
        low_level_adapter=low_level_adapter,
        max_frames=args.max_frames,
        scheduler_guard_mode=args.scheduler_guard_mode,
        scheduler_min_margin=args.scheduler_min_margin,
    )

    for index, scenario in enumerate(scenarios):
        gif_path = args.gif_name
        if gif_path and len(scenarios) > 1:
            suffix = Path(gif_path).suffix or ".gif"
            stem = Path(gif_path).with_suffix("").name
            gif_path = str(Path(gif_path).with_name(f"{stem}_{index:02d}{suffix}"))

        info = runner.run_episode(
            scenario=scenario,
            render=args.render,
            gif_path=gif_path,
        )
        print(f"[{scenario['scenario_id']}] {info}")


if __name__ == "__main__":
    main()
