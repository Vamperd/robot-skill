from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attention_policy import load_scheduler_checkpoint
from baselines import evaluate_all
from scheduler_training_utils import evaluate_policy_on_scenarios
from scheduler_utils import load_split_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估高层 attention 调度器，并与现有基线比较。")
    parser.add_argument("--model", required=True, help="高层调度器 checkpoint 路径")
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--family", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_scheduler_checkpoint(args.model, device=device)
    scenarios = load_split_scenarios(
        scenario_dir=args.scenario_dir,
        split=args.split,
        families=[args.family] if args.family else None,
    )
    results = {
        "model_path": str(Path(args.model).resolve()),
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "scheduler": evaluate_policy_on_scenarios(
            scenarios=scenarios,
            policy=model,
            device=device,
            max_episodes=args.max_episodes,
            deterministic=True,
        ),
    }

    if args.include_baselines:
        results["baselines"] = evaluate_all(
            scenario_dir=args.scenario_dir,
            split=args.split,
            family=args.family,
            max_episodes=args.max_episodes,
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
