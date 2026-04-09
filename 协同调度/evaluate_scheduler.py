from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from attention_policy import load_scheduler_checkpoint
from baselines import evaluate_all
from hetero_attention_policy import load_hetero_policy_checkpoint
from hetero_training_utils import (
    evaluate_policy_family_breakdown as evaluate_hetero_policy_family_breakdown,
    evaluate_policy_on_scenarios as evaluate_hetero_policy_on_scenarios,
)
from scheduler_training_utils import evaluate_policy_on_scenarios as evaluate_legacy_policy_on_scenarios
from scheduler_utils import load_split_scenarios


BC_REFERENCE = {
    "success_rate": 1.0,
    "mean_makespan": 864.63,
    "mean_avoidable_wait_time": 9.20,
    "trap_avoidable_wait": 3.25,
}


def _deployment_failures(metadata: Dict[str, object]) -> List[str]:
    thresholds = metadata.get("deploy_thresholds", {}) or {}
    val_metrics = metadata.get("val_metrics", {}) or {}
    trap_metrics = metadata.get("trap_val_metrics", {}) or {}
    failures: List[str] = []
    if thresholds:
        if float(metadata.get("teacher_agreement", 0.0)) < float(thresholds.get("teacher_agreement", 0.0)):
            failures.append(
                f"teacher_agreement {float(metadata.get('teacher_agreement', 0.0)):.3f} < {float(thresholds['teacher_agreement']):.3f}"
            )
        if float(metadata.get("hard_teacher_agreement", 0.0)) < float(thresholds.get("hard_teacher_agreement", 0.0)):
            failures.append(
                "hard_teacher_agreement "
                f"{float(metadata.get('hard_teacher_agreement', 0.0)):.3f} < {float(thresholds['hard_teacher_agreement']):.3f}"
            )
        if float(metadata.get("single_vs_sync_conflict_agreement", 0.0)) < float(thresholds.get("single_vs_sync_conflict_agreement", 0.0)):
            failures.append(
                "single_vs_sync_conflict_agreement "
                f"{float(metadata.get('single_vs_sync_conflict_agreement', 0.0)):.3f} < "
                f"{float(thresholds['single_vs_sync_conflict_agreement']):.3f}"
            )
        if float(val_metrics.get("success_rate", 0.0)) < float(thresholds.get("success_rate", 0.0)):
            failures.append(
                f"success_rate {float(val_metrics.get('success_rate', 0.0)):.3f} < {float(thresholds['success_rate']):.3f}"
            )
        if float(val_metrics.get("mean_makespan", float('inf'))) > float(thresholds.get("mean_makespan", float('inf'))):
            failures.append(
                f"mean_makespan {float(val_metrics.get('mean_makespan', float('inf'))):.2f} > {float(thresholds['mean_makespan']):.2f}"
            )
        if float(val_metrics.get("mean_avoidable_wait_time", float('inf'))) > float(thresholds.get("mean_avoidable_wait_time", float('inf'))):
            failures.append(
                "mean_avoidable_wait_time "
                f"{float(val_metrics.get('mean_avoidable_wait_time', float('inf'))):.2f} > "
                f"{float(thresholds['mean_avoidable_wait_time']):.2f}"
            )
        if float(trap_metrics.get("mean_avoidable_wait_time", float('inf'))) > float(thresholds.get("trap_avoidable_wait", float('inf'))):
            failures.append(
                f"trap_avoidable_wait {float(trap_metrics.get('mean_avoidable_wait_time', float('inf'))):.2f} > {float(thresholds['trap_avoidable_wait']):.2f}"
            )
    return failures


def _recommend_deployment(
    *,
    policy_type: str,
    checkpoint_metadata: Dict[str, object],
    scheduler_metrics: Dict[str, float],
    trap_metrics: Dict[str, float] | None,
) -> str | None:
    if policy_type != "hetero_ranker":
        return None
    if bool(checkpoint_metadata.get("deploy_ready", False)):
        return "deploy_direct"
    if float(scheduler_metrics.get("success_rate", 0.0)) < 1.0:
        return "do_not_deploy"
    trap_avoidable_wait = float((trap_metrics or {}).get("mean_avoidable_wait_time", 0.0))
    stronger_than_bc = (
        float(scheduler_metrics.get("success_rate", 0.0)) >= BC_REFERENCE["success_rate"]
        and float(scheduler_metrics.get("mean_makespan", float("inf"))) <= BC_REFERENCE["mean_makespan"]
        and float(scheduler_metrics.get("mean_avoidable_wait_time", float("inf"))) <= BC_REFERENCE["mean_avoidable_wait_time"]
        and (not trap_metrics or trap_avoidable_wait <= BC_REFERENCE["trap_avoidable_wait"])
    )
    if stronger_than_bc:
        return "deploy_with_guard"
    return "do_not_deploy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估高层调度器，并与现有基线比较。")
    parser.add_argument("--model", default=None, help="高层调度器 checkpoint 路径")
    parser.add_argument(
        "--expert-policy",
        default=None,
        choices=["upfront_wait_aware_greedy", "rollout_upfront_teacher", "hybrid_upfront_teacher"],
        help="不加载 checkpoint，直接评估纯专家策略。",
    )
    parser.add_argument(
        "--policy-type",
        default="legacy",
        choices=["legacy", "hetero_ppo", "hetero_actor_only", "hetero_ranker"],
        help="checkpoint 对应的高层策略类型。",
    )
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--family", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument(
        "--trap-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否额外评估 partial_coalition_trap 子集。",
    )
    parser.add_argument(
        "--family-breakdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否额外导出每个 family 的独立评估结果。",
    )
    parser.add_argument("--teacher-rollout-depth", type=int, default=2)
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model and not args.expert_policy:
        raise ValueError("请在 --model 和 --expert-policy 之间至少提供一个评估对象。")

    device = torch.device(args.device)
    model_path = None
    checkpoint = {"metadata": {}}

    if args.expert_policy is not None:
        policy_under_test = args.expert_policy
        evaluate_fn = evaluate_hetero_policy_on_scenarios
        family_breakdown_fn = evaluate_hetero_policy_family_breakdown
        hetero_like = True
    elif args.policy_type in {"hetero_ppo", "hetero_actor_only", "hetero_ranker"}:
        model_path = str(Path(args.model).expanduser().resolve(strict=False))
        model, checkpoint = load_hetero_policy_checkpoint(model_path, device=device)
        policy_under_test = model
        evaluate_fn = evaluate_hetero_policy_on_scenarios
        family_breakdown_fn = evaluate_hetero_policy_family_breakdown
        hetero_like = True
    else:
        model_path = str(Path(args.model).expanduser().resolve(strict=False))
        model, checkpoint = load_scheduler_checkpoint(model_path, device=device)
        policy_under_test = model
        evaluate_fn = evaluate_legacy_policy_on_scenarios
        family_breakdown_fn = None
        hetero_like = False

    scenarios = load_split_scenarios(
        scenario_dir=args.scenario_dir,
        split=args.split,
        families=[args.family] if args.family else None,
    )
    if not scenarios:
        raise ValueError("评估场景为空，请检查 split/family 是否正确。")

    results = {
        "model_path": model_path,
        "expert_policy": args.expert_policy,
        "policy_type": args.policy_type,
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "scheduler": evaluate_fn(
            scenarios=scenarios,
            policy=policy_under_test,
            device=device,
            max_episodes=args.max_episodes,
            deterministic=True,
            **({"teacher_rollout_depth": args.teacher_rollout_depth} if hetero_like else {}),
        ),
    }

    trap_scenarios = []
    trap_family_requested = args.family == "partial_coalition_trap"
    if args.trap_eval and not trap_family_requested:
        trap_scenarios = load_split_scenarios(
            scenario_dir=args.scenario_dir,
            split=args.split,
            families=["partial_coalition_trap"],
            limit_per_family=args.max_episodes,
        )
        if trap_scenarios:
            results["trap_subset"] = evaluate_fn(
                scenarios=trap_scenarios,
                policy=policy_under_test,
                device=device,
                deterministic=True,
                **({"teacher_rollout_depth": args.teacher_rollout_depth} if hetero_like else {}),
            )

    if args.family_breakdown and family_breakdown_fn is not None and not args.family:
        results["family_breakdown"] = family_breakdown_fn(
            scenarios=scenarios,
            policy=policy_under_test,
            device=device,
            deterministic=True,
            teacher_rollout_depth=args.teacher_rollout_depth,
        )

    if args.include_baselines:
        baseline_results = evaluate_all(
            scenario_dir=args.scenario_dir,
            split=args.split,
            family=args.family,
            max_episodes=args.max_episodes,
        )
        if args.policy_type in {"hetero_ppo", "hetero_actor_only", "hetero_ranker"} or args.expert_policy is not None:
            baseline_results["upfront_wait_aware_greedy"] = evaluate_hetero_policy_on_scenarios(
                scenarios=scenarios,
                policy="upfront_wait_aware_greedy",
                device=device,
                max_episodes=args.max_episodes,
                deterministic=True,
                teacher_rollout_depth=args.teacher_rollout_depth,
            )
            baseline_results["rollout_upfront_teacher"] = evaluate_hetero_policy_on_scenarios(
                scenarios=scenarios,
                policy="rollout_upfront_teacher",
                device=device,
                max_episodes=args.max_episodes,
                deterministic=True,
                teacher_rollout_depth=args.teacher_rollout_depth,
            )
            baseline_results["hybrid_upfront_teacher"] = evaluate_hetero_policy_on_scenarios(
                scenarios=scenarios,
                policy="hybrid_upfront_teacher",
                device=device,
                max_episodes=args.max_episodes,
                deterministic=True,
                teacher_rollout_depth=args.teacher_rollout_depth,
            )
        results["baselines"] = baseline_results

        if args.trap_eval and not trap_family_requested and trap_scenarios:
            trap_baselines = evaluate_all(
                scenario_dir=args.scenario_dir,
                split=args.split,
                family="partial_coalition_trap",
                max_episodes=args.max_episodes,
            )
            if args.policy_type in {"hetero_ppo", "hetero_actor_only", "hetero_ranker"} or args.expert_policy is not None:
                trap_baselines["upfront_wait_aware_greedy"] = evaluate_hetero_policy_on_scenarios(
                    scenarios=trap_scenarios,
                    policy="upfront_wait_aware_greedy",
                    device=device,
                    deterministic=True,
                    teacher_rollout_depth=args.teacher_rollout_depth,
                )
                trap_baselines["rollout_upfront_teacher"] = evaluate_hetero_policy_on_scenarios(
                    scenarios=trap_scenarios,
                    policy="rollout_upfront_teacher",
                    device=device,
                    deterministic=True,
                    teacher_rollout_depth=args.teacher_rollout_depth,
                )
                trap_baselines["hybrid_upfront_teacher"] = evaluate_hetero_policy_on_scenarios(
                    scenarios=trap_scenarios,
                    policy="hybrid_upfront_teacher",
                    device=device,
                    deterministic=True,
                    teacher_rollout_depth=args.teacher_rollout_depth,
                )
            results["trap_baselines"] = trap_baselines

    checkpoint_metadata = checkpoint.get("metadata", {})
    if args.policy_type == "hetero_ranker" and args.expert_policy is None:
        deployment_failures = _deployment_failures(checkpoint_metadata)
        deployment_recommendation = _recommend_deployment(
            policy_type=args.policy_type,
            checkpoint_metadata=checkpoint_metadata,
            scheduler_metrics=results["scheduler"],
            trap_metrics=results.get("trap_subset"),
        )
        results["stage"] = checkpoint_metadata.get("stage")
        results["epoch"] = checkpoint_metadata.get("epoch")
        results["deploy_ready"] = bool(checkpoint_metadata.get("deploy_ready", False))
        results["deploy_thresholds"] = checkpoint_metadata.get("deploy_thresholds", {})
        results["deployment_failures"] = deployment_failures
        results["deployment_recommendation"] = deployment_recommendation

        deploy_ready_text = "yes" if results["deploy_ready"] else "no"
        print(f"[DEPLOY] hetero_ranker checkpoint deploy_ready={deploy_ready_text}")
        if deployment_failures:
            print("[DEPLOY] failed thresholds:")
            for item in deployment_failures:
                print(f"  - {item}")
        if deployment_recommendation is not None:
            print(f"[DEPLOY] recommendation={deployment_recommendation}")

    output = json.dumps(results, ensure_ascii=False, indent=2)
    print(output)
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
