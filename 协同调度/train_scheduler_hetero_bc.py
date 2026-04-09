from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from hetero_attention_policy import (
    HeteroAttentionSchedulerPolicy,
    load_hetero_scheduler_checkpoint,
    save_hetero_scheduler_checkpoint,
)
from hetero_training_utils import (
    HARD_SAMPLE_FAMILIES,
    HeteroSupervisedDataset,
    collect_expert_sample_records,
    collate_supervised_batch,
    evaluate_policy_family_breakdown,
    evaluate_policy_on_scenarios,
    is_hard_sample,
    mean_legal_action_count,
)
from scheduler_utils import FAMILY_NAMES, load_split_scenarios


TRAP_FAMILIES = {"far_near_trap", "partial_coalition_trap"}
GAP_FAMILIES = ["role_mismatch", "single_bottleneck", "double_bottleneck", "multi_sync_cluster", "partial_coalition_trap"]
FOCUS_FAMILIES = {"role_mismatch", "single_bottleneck", "double_bottleneck", "multi_sync_cluster"}
HARD_STAGE_BATCH_SIZE = 64
HARD_STAGE_DISAGREEMENT_MULTIPLIER = 8.0
HARD_STAGE_CONFLICT_DISAGREEMENT_MULTIPLIER = 10.0


def obs_batch_to_torch(obs_batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in obs_batch.items()
    }
    tensors["current_agent_index"] = tensors["current_agent_index"].long()
    return tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Hetero attention policy 做高层行为克隆预热。")
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=None, help="兼容旧入口；提供后将均分为 base/hard 两阶段。")
    parser.add_argument("--base-epochs", type=int, default=4)
    parser.add_argument("--hard-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--expert-policy",
        default="auto",
        choices=["auto", "upfront_wait_aware_greedy", "rollout_upfront_teacher", "hybrid_upfront_teacher"],
    )
    parser.add_argument("--teacher-rollout-depth", type=int, default=2)
    parser.add_argument("--limit-train-per-family", type=int, default=None)
    parser.add_argument("--limit-val-per-family", type=int, default=None)
    parser.add_argument("--save-dir", default="协同调度/checkpoints_hetero_bc")
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def metric_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["success_rate"] * 1200.0
        - metrics["mean_makespan"]
        - 0.4 * metrics["mean_wait_time"]
        - 0.8 * metrics["mean_avoidable_wait_time"]
        - 50.0 * metrics["mean_direct_sync_misassignment_rate"]
    )


def teacher_metric_score(
    val_metrics: Dict[str, float],
    trap_metrics: Dict[str, float] | None,
) -> float:
    score = metric_score(val_metrics)
    if trap_metrics is not None:
        score += 0.5 * metric_score(trap_metrics)
    return score


def _load_scenarios_by_family(
    scenario_dir: str,
    split: str,
    limit_per_family: int | None,
) -> Dict[str, List[Dict]]:
    return {
        family: load_split_scenarios(
            scenario_dir=scenario_dir,
            split=split,
            families=[family],
            limit_per_family=limit_per_family,
        )
        for family in FAMILY_NAMES
    }


def _flatten_family_scenarios(scenarios_by_family: Dict[str, Sequence[Dict]]) -> List[Dict]:
    flattened: List[Dict] = []
    for family in FAMILY_NAMES:
        flattened.extend(list(scenarios_by_family.get(family, [])))
    return flattened


def _record_to_sample(record: Dict[str, object]) -> tuple[Dict[str, np.ndarray], int]:
    return record["obs"], int(record["action"])


def _record_weight(
    record: Dict[str, object],
    family_count: int,
    disagreement_multiplier: float = 1.0,
    conflict_disagreement_multiplier: float = 1.0,
) -> float:
    family = str(record["family"])
    weight = 1.0
    if family in TRAP_FAMILIES:
        weight *= 3.0
    if family in FOCUS_FAMILIES:
        weight *= 4.0
    if int(record["legal_action_count"]) > 2:
        weight *= 2.0
    if bool(record["single_sync_conflict"]):
        weight *= 6.0
    if bool(record.get("disagreement", False)):
        weight *= disagreement_multiplier
        if bool(record["single_sync_conflict"]):
            weight *= conflict_disagreement_multiplier
    family_balance = 1.0 / max(float(family_count), 1.0)
    return weight * family_balance


def _build_record_pool(
    scenarios_by_family: Dict[str, Sequence[Dict]],
    expert_policy: str,
    teacher_rollout_depth: int,
) -> tuple[List[Dict[str, object]], Dict[str, int]]:
    records: List[Dict[str, object]] = []
    family_counts: Dict[str, int] = {}
    for family in FAMILY_NAMES:
        family_scenarios = list(scenarios_by_family.get(family, []))
        if not family_scenarios:
            continue
        family_records = collect_expert_sample_records(
            family_scenarios,
            expert_policy=expert_policy,
            teacher_rollout_depth=teacher_rollout_depth,
        )
        family_counts[family] = len(family_records)
        records.extend(family_records)
    return records, family_counts


def _build_dataset_and_sampler(
    records: Sequence[Dict[str, object]],
    family_counts: Dict[str, int],
    batch_size: int,
    *,
    disagreement_multiplier: float = 3.0,
    conflict_disagreement_multiplier: float = 1.0,
) -> tuple[HeteroSupervisedDataset, DataLoader]:
    samples = [_record_to_sample(record) for record in records]
    dataset = HeteroSupervisedDataset(samples)
    weights = [
        _record_weight(
            record,
            family_count=max(family_counts.get(str(record["family"]), 1), 1),
            disagreement_multiplier=disagreement_multiplier,
            conflict_disagreement_multiplier=conflict_disagreement_multiplier,
        )
        for record in records
    ]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_supervised_batch)
    return dataset, loader


def _per_sample_label_smoothing(global_mask: torch.Tensor, max_label_smoothing: float) -> torch.Tensor:
    legal_counts = (global_mask[:, 1:] < 0.5).sum(dim=-1)
    smoothing = torch.full_like(legal_counts, fill_value=max_label_smoothing, dtype=torch.float32)
    smoothing = torch.where(legal_counts <= 2, torch.zeros_like(smoothing), smoothing)
    smoothing = torch.where(legal_counts == 3, torch.full_like(smoothing, 0.02), smoothing)
    return smoothing


def _masked_label_smoothed_loss(
    logps: torch.Tensor,
    actions: torch.Tensor,
    global_mask: torch.Tensor,
    max_label_smoothing: float,
) -> torch.Tensor:
    nll = -logps.gather(1, actions.unsqueeze(1)).squeeze(1)
    if max_label_smoothing <= 0.0:
        return nll.mean()
    legal_mask = global_mask < 0.5
    legal_counts = legal_mask.sum(dim=-1).clamp_min(1)
    smoothed = -(logps.masked_fill(~legal_mask, 0.0).sum(dim=-1) / legal_counts)
    smoothing = _per_sample_label_smoothing(global_mask, max_label_smoothing)
    return ((1.0 - smoothing) * nll + smoothing * smoothed).mean()


def _predict_actions_for_records(
    model: HeteroAttentionSchedulerPolicy,
    records: Sequence[Dict[str, object]],
    device: torch.device,
    batch_size: int,
) -> List[int]:
    predictions: List[int] = []
    if not records:
        return predictions
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = list(records[start:start + batch_size])
            obs_batch_np, _ = collate_supervised_batch([_record_to_sample(record) for record in batch_records])
            obs_batch = obs_batch_to_torch(obs_batch_np, device)
            probs, _, _ = model(obs_batch)
            predictions.extend(torch.argmax(probs, dim=-1).cpu().tolist())
    model.train()
    return [int(item) for item in predictions]


def _agreement_on_records(
    model: HeteroAttentionSchedulerPolicy,
    records: Sequence[Dict[str, object]],
    device: torch.device,
    batch_size: int,
) -> float:
    if not records:
        return 0.0
    predictions = _predict_actions_for_records(model, records, device, batch_size)
    correct = 0
    for record, prediction in zip(records, predictions):
        correct += int(prediction == int(record["action"]))
    return correct / max(len(records), 1)


def _filter_records(
    records: Sequence[Dict[str, object]],
    predicate,
) -> List[Dict[str, object]]:
    return [record for record in records if predicate(record)]


def _select_hard_disagreement_records(
    records: Sequence[Dict[str, object]],
    predictions: Sequence[int],
) -> List[Dict[str, object]]:
    targeted: List[Dict[str, object]] = []
    fallback_hard: List[Dict[str, object]] = []
    for record, prediction in zip(records, predictions):
        if int(record["action"]) == int(prediction):
            continue
        disagreement_record = dict(record)
        disagreement_record["disagreement"] = True
        if bool(record["hard_state"]):
            fallback_hard.append(disagreement_record)
        family = str(record["family"])
        if bool(record["single_sync_conflict"]) or family in HARD_SAMPLE_FAMILIES:
            targeted.append(disagreement_record)
    if targeted:
        return targeted
    return fallback_hard


def _hard_family_gap_summary(
    current_breakdown: Dict[str, Dict[str, float]],
    teacher_breakdown: Dict[str, Dict[str, float]],
) -> str:
    parts: List[str] = []
    for family in GAP_FAMILIES:
        current = current_breakdown.get(family)
        teacher = teacher_breakdown.get(family)
        if not current or not teacher:
            continue
        aw_gap = current["mean_avoidable_wait_time"] - teacher["mean_avoidable_wait_time"]
        sm_gap = current["mean_direct_sync_misassignment_rate"] - teacher["mean_direct_sync_misassignment_rate"]
        parts.append(f"{family}:aw+{aw_gap:.2f}/sm+{sm_gap:.3f}")
    return " | ".join(parts)


def _trap_teacher_gap(
    trap_metrics: Dict[str, float] | None,
    teacher_trap_metrics: Dict[str, float] | None,
) -> str:
    if trap_metrics is None or teacher_trap_metrics is None:
        return "n/a"
    aw_gap = trap_metrics["mean_avoidable_wait_time"] - teacher_trap_metrics["mean_avoidable_wait_time"]
    sm_gap = (
        trap_metrics["mean_direct_sync_misassignment_rate"]
        - teacher_trap_metrics["mean_direct_sync_misassignment_rate"]
    )
    return f"aw+{aw_gap:.2f}/sm+{sm_gap:.3f}"


def _format_family_summary(metrics_by_family: Dict[str, Dict[str, float]]) -> str:
    parts = []
    for family in FAMILY_NAMES:
        metrics = metrics_by_family.get(family)
        if not metrics:
            continue
        parts.append(
            f"{family}:s={metrics['success_rate']:.3f}"
            f"/aw={metrics['mean_avoidable_wait_time']:.2f}"
            f"/sm={metrics['mean_direct_sync_misassignment_rate']:.3f}"
        )
    return " | ".join(parts)


def _run_training_epochs(
    *,
    stage_name: str,
    epoch_start: int,
    epoch_count: int,
    model: HeteroAttentionSchedulerPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_label_smoothing: float,
    val_scenarios: Sequence[Dict],
    trap_val_scenarios: Sequence[Dict],
    teacher_records: Sequence[Dict[str, object]],
    teacher_hard_records: Sequence[Dict[str, object]],
    teacher_conflict_records: Sequence[Dict[str, object]],
    teacher_reference: Dict,
    metadata: Dict,
    save_dir: Path,
    best_score: float,
) -> float:
    teacher_val_metrics = teacher_reference["val_metrics"]
    teacher_trap_metrics = teacher_reference["trap_val_metrics"]
    teacher_family_breakdown = teacher_reference["family_breakdown"]
    mean_legal = mean_legal_action_count([_record_to_sample(record) for record in teacher_records])

    for local_epoch in range(epoch_count):
        epoch = epoch_start + local_epoch
        running_loss = 0.0
        steps = 0

        for obs_batch_np, actions_np in loader:
            obs_batch = obs_batch_to_torch(obs_batch_np, device)
            actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)

            _, logps, _ = model(obs_batch)
            loss = _masked_label_smoothed_loss(
                logps=logps,
                actions=actions,
                global_mask=obs_batch["global_mask"],
                max_label_smoothing=max_label_smoothing,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            steps += 1

        train_loss = running_loss / max(steps, 1)
        teacher_agreement = _agreement_on_records(model, teacher_records, device=device, batch_size=loader.batch_size or 128)
        hard_teacher_agreement = _agreement_on_records(
            model,
            teacher_hard_records,
            device=device,
            batch_size=loader.batch_size or 128,
        )
        single_sync_conflict_agreement = _agreement_on_records(
            model,
            teacher_conflict_records,
            device=device,
            batch_size=loader.batch_size or 128,
        )
        val_metrics = evaluate_policy_on_scenarios(val_scenarios, model, device=device, deterministic=True)
        trap_metrics = None
        if trap_val_scenarios:
            trap_metrics = evaluate_policy_on_scenarios(trap_val_scenarios, model, device=device, deterministic=True)
        family_breakdown = evaluate_policy_family_breakdown(val_scenarios, model, device=device, deterministic=True)
        score = metric_score(val_metrics)
        bc_ready = (
            teacher_agreement >= 0.93
            and hard_teacher_agreement >= 0.90
            and single_sync_conflict_agreement >= 0.60
            and val_metrics["success_rate"] >= teacher_val_metrics["success_rate"]
            and val_metrics["mean_avoidable_wait_time"] <= 4.0
            and (trap_metrics is None or trap_metrics["mean_avoidable_wait_time"] <= 1.5)
        )

        checkpoint_metadata = {
            **metadata,
            "policy_type": "hetero_ppo",
            "stage": "behavior_cloning",
            "bc_stage_name": stage_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "teacher_reference": teacher_reference,
            "teacher_agreement": teacher_agreement,
            "hard_teacher_agreement": hard_teacher_agreement,
            "single_vs_sync_conflict_agreement": single_sync_conflict_agreement,
            "mean_legal_action_count": mean_legal,
            "val_metrics": val_metrics,
            "trap_val_metrics": trap_metrics,
            "family_breakdown": family_breakdown,
            "bc_ready_for_actor_only": bc_ready,
        }
        save_hetero_scheduler_checkpoint(
            save_dir / "latest_scheduler_bc.pt",
            model,
            optimizer=optimizer,
            metadata=checkpoint_metadata,
        )
        if score > best_score:
            best_score = score
            save_hetero_scheduler_checkpoint(
                save_dir / "best_scheduler_bc.pt",
                model,
                optimizer=optimizer,
                metadata=checkpoint_metadata,
            )

        line = (
            f"[HBC][epoch {epoch:02d}][{stage_name}] "
            f"loss={train_loss:.4f} "
            f"teacher_agreement={teacher_agreement:.3f} "
            f"hard_teacher_agreement={hard_teacher_agreement:.3f} "
            f"single_vs_sync_conflict_agreement={single_sync_conflict_agreement:.3f} "
            f"mean_legal_action_count={mean_legal:.2f} "
            f"success={val_metrics['success_rate']:.3f} "
            f"makespan={val_metrics['mean_makespan']:.2f} "
            f"wait={val_metrics['mean_wait_time']:.2f} "
            f"avoidable_wait={val_metrics['mean_avoidable_wait_time']:.2f} "
            f"sync_misassign={val_metrics['mean_direct_sync_misassignment_rate']:.3f} "
            f"wait_rate={val_metrics['mean_wait_action_rate']:.3f} "
            f"idle_wait={val_metrics['mean_idle_wait_rate']:.3f} "
            f"waiting_idle_wait={val_metrics['mean_waiting_idle_wait_rate']:.3f} "
            f"stalled_wait={val_metrics['mean_stalled_wait_rate']:.3f} "
            f"wait_flip={val_metrics['mean_wait_flip_rate']:.3f}"
        )
        if trap_metrics is not None:
            line += (
                f" trap_success={trap_metrics['success_rate']:.3f}"
                f" trap_avoidable_wait={trap_metrics['mean_avoidable_wait_time']:.2f}"
                f" trap_sync_misassign={trap_metrics['mean_direct_sync_misassignment_rate']:.3f}"
                f" trap_teacher_gap={_trap_teacher_gap(trap_metrics, teacher_trap_metrics)}"
            )
        print(line)
        print(f"[HBC][family {epoch:02d}] {_format_family_summary(family_breakdown)}")
        print(f"[HBC][gap {epoch:02d}] {_hard_family_gap_summary(family_breakdown, teacher_family_breakdown)}")
        if bc_ready:
            print(f"[HBC][ready {epoch:02d}] BC 已达到 actor-only 准入门槛。")
    return best_score


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.epochs is not None:
        base_epochs = max(1, int(args.epochs) // 2)
        hard_epochs = max(int(args.epochs) - base_epochs, 0)
    else:
        base_epochs = int(args.base_epochs)
        hard_epochs = int(args.hard_epochs)

    train_by_family = _load_scenarios_by_family(args.scenario_dir, args.train_split, args.limit_train_per_family)
    val_by_family = _load_scenarios_by_family(args.scenario_dir, args.val_split, args.limit_val_per_family)
    train_scenarios = _flatten_family_scenarios(train_by_family)
    val_scenarios = _flatten_family_scenarios(val_by_family)
    trap_val_scenarios = list(val_by_family.get("partial_coalition_trap", []))
    if not train_scenarios:
        raise ValueError("训练场景为空，请先生成 offline_maps_v2 数据集。")
    if not val_scenarios:
        raise ValueError("验证场景为空，请检查 val split。")

    print(f"收集 Hetero 专家轨迹: train={len(train_scenarios)} 个场景, val={len(val_scenarios)} 个场景")
    candidate_policies = (
        ["upfront_wait_aware_greedy", "rollout_upfront_teacher", "hybrid_upfront_teacher"]
        if args.expert_policy == "auto"
        else [args.expert_policy]
    )
    teacher_candidates: Dict[str, Dict] = {}
    chosen_policy = candidate_policies[0]
    chosen_score = float("-inf")
    for expert_policy in candidate_policies:
        candidate_val_metrics = evaluate_policy_on_scenarios(
            val_scenarios,
            policy=expert_policy,
            device=device,
            deterministic=True,
            teacher_rollout_depth=args.teacher_rollout_depth,
        )
        candidate_trap_metrics = None
        if trap_val_scenarios:
            candidate_trap_metrics = evaluate_policy_on_scenarios(
                trap_val_scenarios,
                policy=expert_policy,
                device=device,
                deterministic=True,
                teacher_rollout_depth=args.teacher_rollout_depth,
            )
        candidate_family_breakdown = evaluate_policy_family_breakdown(
            val_scenarios,
            policy=expert_policy,
            device=device,
            deterministic=True,
            teacher_rollout_depth=args.teacher_rollout_depth,
        )
        candidate_score = teacher_metric_score(candidate_val_metrics, candidate_trap_metrics)
        teacher_candidates[expert_policy] = {
            "score": candidate_score,
            "val_metrics": candidate_val_metrics,
            "trap_val_metrics": candidate_trap_metrics,
            "family_breakdown": candidate_family_breakdown,
        }
        if candidate_score > chosen_score:
            chosen_policy = expert_policy
            chosen_score = candidate_score

    train_records, family_counts = _build_record_pool(
        scenarios_by_family=train_by_family,
        expert_policy=chosen_policy,
        teacher_rollout_depth=args.teacher_rollout_depth,
    )
    dataset, loader = _build_dataset_and_sampler(
        train_records,
        family_counts,
        batch_size=args.batch_size,
        disagreement_multiplier=1.0,
    )

    val_teacher_records = collect_expert_sample_records(
        val_scenarios,
        expert_policy=chosen_policy,
        teacher_rollout_depth=args.teacher_rollout_depth,
    )
    hard_val_teacher_records = _filter_records(val_teacher_records, lambda record: bool(record["hard_state"]))
    conflict_val_teacher_records = _filter_records(
        val_teacher_records,
        lambda record: bool(record["single_sync_conflict"]),
    )
    teacher_val_metrics = teacher_candidates[chosen_policy]["val_metrics"]
    teacher_trap_metrics = teacher_candidates[chosen_policy]["trap_val_metrics"]
    teacher_family_breakdown = teacher_candidates[chosen_policy]["family_breakdown"]
    teacher_reference = {
        "expert_policy": args.expert_policy,
        "chosen_expert_policy": chosen_policy,
        "teacher_rollout_depth": args.teacher_rollout_depth,
        "teacher_candidates": teacher_candidates,
        "val_metrics": teacher_val_metrics,
        "trap_val_metrics": teacher_trap_metrics,
        "family_breakdown": teacher_family_breakdown,
        "hard_state_count": len(hard_val_teacher_records),
        "single_sync_conflict_count": len(conflict_val_teacher_records),
    }
    (save_dir / "teacher_val_reference.json").write_text(
        json.dumps(teacher_reference, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Hetero BC teacher: requested={args.expert_policy} chosen={chosen_policy} score={chosen_score:.2f}")

    sample_obs, _ = dataset[0]
    agent_dim = int(sample_obs["agent_inputs"].shape[-1])
    task_dim = int(sample_obs["task_inputs"].shape[-1])

    if args.init_model:
        model, checkpoint = load_hetero_scheduler_checkpoint(args.init_model, device=device)
        print(f"加载初始模型: {args.init_model}")
        metadata = checkpoint.get("metadata", {})
    else:
        model = HeteroAttentionSchedulerPolicy(agent_input_dim=agent_dim, task_input_dim=task_dim).to(device)
        metadata = {}
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = float("-inf")

    print(f"专家样本数: {len(dataset)}")
    best_score = _run_training_epochs(
        stage_name="base",
        epoch_start=1,
        epoch_count=base_epochs,
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        max_label_smoothing=args.max_label_smoothing,
        val_scenarios=val_scenarios,
        trap_val_scenarios=trap_val_scenarios,
        teacher_records=val_teacher_records,
        teacher_hard_records=hard_val_teacher_records,
        teacher_conflict_records=conflict_val_teacher_records,
        teacher_reference=teacher_reference,
        metadata=metadata,
        save_dir=save_dir,
        best_score=best_score,
    )

    disagreement_predictions = _predict_actions_for_records(model, train_records, device=device, batch_size=args.batch_size)
    disagreement_records = _select_hard_disagreement_records(train_records, disagreement_predictions)
    if not disagreement_records:
        for record in train_records:
            if bool(record["hard_state"]):
                disagreement_record = dict(record)
                disagreement_record["disagreement"] = True
                disagreement_records.append(disagreement_record)

    if disagreement_records and hard_epochs > 0:
        hard_family_counts: Dict[str, int] = {}
        for record in disagreement_records:
            family = str(record["family"])
            hard_family_counts[family] = hard_family_counts.get(family, 0) + 1
        hard_dataset, hard_loader = _build_dataset_and_sampler(
            disagreement_records,
            hard_family_counts,
            batch_size=HARD_STAGE_BATCH_SIZE,
            disagreement_multiplier=HARD_STAGE_DISAGREEMENT_MULTIPLIER,
            conflict_disagreement_multiplier=HARD_STAGE_CONFLICT_DISAGREEMENT_MULTIPLIER,
        )
        print(f"hard-state disagreement 样本数: {len(hard_dataset)}")
        best_score = _run_training_epochs(
            stage_name="hard",
            epoch_start=base_epochs + 1,
            epoch_count=hard_epochs,
            model=model,
            loader=hard_loader,
            optimizer=optimizer,
            device=device,
            max_label_smoothing=0.0,
            val_scenarios=val_scenarios,
            trap_val_scenarios=trap_val_scenarios,
            teacher_records=val_teacher_records,
            teacher_hard_records=hard_val_teacher_records,
            teacher_conflict_records=conflict_val_teacher_records,
            teacher_reference=teacher_reference,
            metadata=metadata,
            save_dir=save_dir,
            best_score=best_score,
        )


if __name__ == "__main__":
    main()
