from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from hetero_dispatch_env import TASK_IDX
from hetero_attention_policy import (
    HeteroRankerPolicy,
    load_hetero_ranker_checkpoint,
    save_hetero_ranker_checkpoint,
)
from hetero_training_utils import (
    HeteroSupervisedDataset,
    collect_expert_sample_records,
    collect_ranker_sample_records,
    collate_supervised_batch,
    evaluate_policy_family_breakdown,
    evaluate_policy_on_scenarios,
    policy_action,
    ranker_sample_metadata,
)
from scheduler_utils import FAMILY_NAMES, load_split_scenarios


TRAP_FAMILIES = {"far_near_trap", "partial_coalition_trap"}
FOCUS_FAMILIES = {"role_mismatch", "single_bottleneck", "double_bottleneck", "multi_sync_cluster"}
GAP_FAMILIES = ["role_mismatch", "single_bottleneck", "double_bottleneck", "multi_sync_cluster", "partial_coalition_trap"]
TEACHER_CANDIDATES = ["upfront_wait_aware_greedy", "rollout_upfront_teacher", "hybrid_upfront_teacher"]
DAGGER_FAMILIES = {"role_mismatch", "single_bottleneck", "double_bottleneck", "multi_sync_cluster", "partial_coalition_trap"}
RANK_FAMILIES = {"role_mismatch", "single_bottleneck", "double_bottleneck", "multi_sync_cluster", "partial_coalition_trap"}
DEPLOY_THRESHOLDS = {
    "teacher_agreement": 0.99,
    "hard_teacher_agreement": 0.99,
    "single_vs_sync_conflict_agreement": 0.93,
    "success_rate": 1.0,
    "mean_makespan": 800.0,
    "mean_avoidable_wait_time": 3.5,
    "trap_avoidable_wait": 1.2,
}
RANK_GUARDRAILS = {
    "teacher_agreement_drop": 0.01,
    "hard_teacher_agreement_drop": 0.01,
    "conflict_agreement_drop": 0.03,
    "avoidable_wait_increase": 0.5,
    "trap_avoidable_wait_increase": 0.3,
}


class RankerRecordDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, object]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.records[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hetero_ranker mainline.")
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--base-epochs", type=int, default=4)
    parser.add_argument("--rank-epochs", type=int, default=0)
    parser.add_argument("--dagger-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pairwise-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--ce-mix", type=float, default=0.5)
    parser.add_argument("--dagger-mode", choices=["off", "hard_only"], default="off")
    parser.add_argument("--dagger-max-records-per-family", type=int, default=512)
    parser.add_argument(
        "--expert-policy",
        default="auto",
        choices=["auto", "upfront_wait_aware_greedy", "rollout_upfront_teacher", "hybrid_upfront_teacher"],
    )
    parser.add_argument("--teacher-rollout-depth", type=int, default=2)
    parser.add_argument("--limit-train-per-family", type=int, default=None)
    parser.add_argument("--limit-val-per-family", type=int, default=None)
    parser.add_argument("--init-model", default="协同调度/checkpoints_hetero_bc/best_scheduler_bc.pt")
    parser.add_argument("--save-dir", default="协同调度/checkpoints_hetero_ranker")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def obs_batch_to_torch(obs_batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in obs_batch.items()
    }
    tensors["current_agent_index"] = tensors["current_agent_index"].long()
    return tensors


def metric_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["success_rate"] * 1200.0
        - metrics["mean_makespan"]
        - 0.4 * metrics["mean_wait_time"]
        - 0.8 * metrics["mean_avoidable_wait_time"]
        - 50.0 * metrics["mean_direct_sync_misassignment_rate"]
    )


def teacher_metric_score(val_metrics: Dict[str, float], trap_metrics: Dict[str, float] | None) -> float:
    score = metric_score(val_metrics)
    if trap_metrics is not None:
        score += 0.5 * metric_score(trap_metrics)
    return score


def ranker_selection_score(
    val_metrics: Dict[str, float],
    teacher_agreement: float,
    hard_teacher_agreement: float,
    conflict_teacher_agreement: float,
    trap_metrics: Dict[str, float] | None,
) -> float:
    score = metric_score(val_metrics)
    score += 120.0 * teacher_agreement
    score += 160.0 * hard_teacher_agreement
    score += 220.0 * conflict_teacher_agreement
    if trap_metrics is not None:
        score -= 4.0 * trap_metrics["mean_avoidable_wait_time"]
        score -= 40.0 * trap_metrics["mean_direct_sync_misassignment_rate"]
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


def _normalize_rank_candidates(record: Dict[str, object]) -> tuple[List[int], List[float]]:
    teacher_action = int(record["action"])
    candidate_actions = [int(action) for action in record.get("candidate_actions", [])]
    candidate_scores = [float(score) for score in record.get("candidate_scores", [])]
    if len(candidate_scores) < len(candidate_actions):
        candidate_scores.extend([0.0] * (len(candidate_actions) - len(candidate_scores)))
    ordered_actions: List[int] = []
    ordered_scores: List[float] = []
    seen: set[int] = set()
    for action, score in zip(candidate_actions, candidate_scores):
        if action in seen:
            continue
        seen.add(action)
        ordered_actions.append(action)
        ordered_scores.append(score)
    if teacher_action not in seen:
        fallback_score = min(ordered_scores) - 1.0 if ordered_scores else 0.0
        ordered_actions.append(teacher_action)
        ordered_scores.append(float(fallback_score))
    if not ordered_actions:
        ordered_actions = [teacher_action]
        ordered_scores = [0.0]
    return ordered_actions, ordered_scores


def _task_kind_from_record(record: Dict[str, object], action: int) -> str:
    if action == 0:
        return "wait"
    task_inputs = np.asarray(record["obs"]["task_inputs"], dtype=np.float32)
    if action < 0 or action >= task_inputs.shape[0]:
        return "other"
    row = task_inputs[action]
    if row[TASK_IDX["is_single"]] > 0.5:
        return "single"
    if row[TASK_IDX["is_sync"]] > 0.5:
        return "sync"
    if row[TASK_IDX["is_wait_node"]] > 0.5:
        return "wait"
    return "other"


def _is_rank_active_record(record: Dict[str, object]) -> bool:
    candidate_actions, _ = _normalize_rank_candidates(record)
    if len(candidate_actions) < 2:
        return False
    if bool(record.get("single_sync_conflict", False)):
        return True
    return str(record["family"]) in RANK_FAMILIES


def _candidate_top1_matches_teacher(record: Dict[str, object]) -> bool:
    candidate_actions = [int(action) for action in record.get("candidate_actions", [])]
    if not candidate_actions:
        return False
    return candidate_actions[0] == int(record["action"])


def _select_rank_negative_action(
    record: Dict[str, object],
    candidate_actions: Sequence[int],
    candidate_scores: Sequence[float],
    *,
    positive_action: int,
    logps_row: torch.Tensor,
    device: torch.device,
) -> int | None:
    negative_actions = [int(action) for action in candidate_actions if int(action) != positive_action]
    if not negative_actions:
        legal_actions = [int(action) for action in np.flatnonzero(record["obs"]["global_mask"] < 0.5)]
        negative_actions = [action for action in legal_actions if action != positive_action]
    if not negative_actions:
        return None

    positive_kind = _task_kind_from_record(record, positive_action)
    if bool(record.get("single_sync_conflict", False)) and positive_kind in {"single", "sync"}:
        opponent_kind = "sync" if positive_kind == "single" else "single"
        conflict_actions = [action for action in negative_actions if _task_kind_from_record(record, action) == opponent_kind]
        if conflict_actions:
            conflict_tensor = torch.as_tensor(conflict_actions, dtype=torch.long, device=device)
            conflict_logps = torch.gather(logps_row, 0, conflict_tensor)
            return int(conflict_actions[int(torch.argmax(conflict_logps).item())])

    negative_tensor = torch.as_tensor(negative_actions, dtype=torch.long, device=device)
    negative_logps = torch.gather(logps_row, 0, negative_tensor)
    best_model_negative = int(negative_actions[int(torch.argmax(negative_logps).item())])
    if best_model_negative != positive_action:
        return best_model_negative

    scored_negatives = [
        (int(action), float(score))
        for action, score in zip(candidate_actions, candidate_scores)
        if int(action) != positive_action
    ]
    if scored_negatives:
        scored_negatives.sort(key=lambda item: (-item[1], item[0]))
        return int(scored_negatives[0][0])
    return None


def _rank_stage_diagnostics(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    active_records = [record for record in records if _is_rank_active_record(record)]
    top1_matches = sum(int(_candidate_top1_matches_teacher(record)) for record in active_records)
    active_count = len(active_records)
    return {
        "teacher_action_top1_match_rate": top1_matches / max(active_count, 1),
        "pairwise_positive_is_teacher_rate": 1.0 if active_count else 0.0,
        "rank_active_record_count": float(active_count),
    }


def _record_weight(
    record: Dict[str, object],
    family_count: int,
    *,
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
    return weight / max(float(family_count), 1.0)


def _build_supervised_loader(
    records: Sequence[Dict[str, object]],
    family_counts: Dict[str, int],
    batch_size: int,
    *,
    disagreement_multiplier: float = 1.0,
    conflict_disagreement_multiplier: float = 1.0,
) -> tuple[HeteroSupervisedDataset, DataLoader]:
    samples = [_record_to_sample(record) for record in records]
    dataset = HeteroSupervisedDataset(samples)
    weights = [
        _record_weight(
            record,
            max(family_counts.get(str(record["family"]), 1), 1),
            disagreement_multiplier=disagreement_multiplier,
            conflict_disagreement_multiplier=conflict_disagreement_multiplier,
        )
        for record in records
    ]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_supervised_batch)
    return dataset, loader


def _build_record_loader(
    records: Sequence[Dict[str, object]],
    family_counts: Dict[str, int],
    batch_size: int,
    *,
    disagreement_multiplier: float = 1.0,
    conflict_disagreement_multiplier: float = 1.0,
) -> tuple[RankerRecordDataset, DataLoader]:
    dataset = RankerRecordDataset(records)
    weights = [
        _record_weight(
            record,
            max(family_counts.get(str(record["family"]), 1), 1),
            disagreement_multiplier=disagreement_multiplier,
            conflict_disagreement_multiplier=conflict_disagreement_multiplier,
        )
        for record in records
    ]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=list)
    return dataset, loader


def _per_sample_label_smoothing(global_mask: torch.Tensor) -> torch.Tensor:
    legal_counts = (global_mask[:, 1:] < 0.5).sum(dim=-1)
    smoothing = torch.full_like(legal_counts, fill_value=0.05, dtype=torch.float32)
    smoothing = torch.where(legal_counts <= 2, torch.zeros_like(smoothing), smoothing)
    smoothing = torch.where(legal_counts == 3, torch.full_like(smoothing, 0.02), smoothing)
    return smoothing


def _masked_label_smoothed_loss(logps: torch.Tensor, actions: torch.Tensor, global_mask: torch.Tensor) -> torch.Tensor:
    nll = -logps.gather(1, actions.unsqueeze(1)).squeeze(1)
    legal_mask = global_mask < 0.5
    legal_counts = legal_mask.sum(dim=-1).clamp_min(1)
    smoothed = -(logps.masked_fill(~legal_mask, 0.0).sum(dim=-1) / legal_counts)
    smoothing = _per_sample_label_smoothing(global_mask)
    return ((1.0 - smoothing) * nll + smoothing * smoothed).mean()


def _predict_actions_for_records(
    model: HeteroRankerPolicy,
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
            probs, _ = model(obs_batch)
            predictions.extend(torch.argmax(probs, dim=-1).cpu().tolist())
    model.train()
    return [int(item) for item in predictions]


def _agreement_on_records(
    model: HeteroRankerPolicy,
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


def _filter_records(records: Sequence[Dict[str, object]], predicate) -> List[Dict[str, object]]:
    return [record for record in records if predicate(record)]


def _select_dagger_records(
    model: HeteroRankerPolicy,
    scenarios: Sequence[Dict],
    expert_policy: str,
    teacher_rollout_depth: int,
    device: torch.device,
    dagger_mode: str,
    max_records_per_family: int,
) -> tuple[List[Dict[str, object]], Dict[str, int]]:
    from hetero_dispatch_env import HeteroDispatchEnv

    if dagger_mode == "off":
        return [], {}

    env = HeteroDispatchEnv(scenarios=scenarios)
    records: List[Dict[str, object]] = []
    family_counts: Dict[str, int] = {}
    teacher_rng = np.random.default_rng(0)

    for scenario in scenarios:
        family = str(scenario.get("family", "unknown"))
        obs, _ = env.reset(options={"scenario": scenario})
        done = False
        truncated = False
        while not (done or truncated):
            greedy_action = int(
                policy_action(
                    env,
                    obs,
                    policy=model,
                    rng=teacher_rng,
                    device=device,
                    deterministic=True,
                    teacher_rollout_depth=teacher_rollout_depth,
                )
            )
            teacher_action = int(
                policy_action(
                    env,
                    obs,
                    policy=expert_policy,
                    rng=teacher_rng,
                    deterministic=True,
                    teacher_rollout_depth=teacher_rollout_depth,
                )
            )
            legal_actions = [int(action) for action in np.flatnonzero(obs["global_mask"] < 0.5)]
            if teacher_action not in legal_actions:
                non_wait = [action for action in legal_actions if action != 0]
                teacher_action = non_wait[0] if non_wait else (legal_actions[0] if legal_actions else 0)
            cold_record = ranker_sample_metadata(family, obs, teacher_action, {teacher_action: 0.0})
            is_disagreement = greedy_action != teacher_action
            family_is_focus = family in DAGGER_FAMILIES
            is_conflict = bool(cold_record["single_sync_conflict"])
            needs_refresh = is_disagreement and (is_conflict or family_is_focus)
            if needs_refresh:
                if family_counts.get(family, 0) >= max(max_records_per_family, 0):
                    obs, _, done, truncated, _ = env.step(greedy_action)
                    continue
                candidate_scores = env.teacher_candidate_scores(rollout_depth=teacher_rollout_depth, top_k=2)
                if teacher_action not in candidate_scores:
                    candidate_scores[teacher_action] = float(max(candidate_scores.values()) if candidate_scores else 0.0)
                record = ranker_sample_metadata(family, obs, teacher_action, candidate_scores)
                record["disagreement"] = bool(is_disagreement)
                records.append(record)
                family_counts[family] = family_counts.get(family, 0) + 1
            obs, _, done, truncated, _ = env.step(greedy_action)
    return records, family_counts


def _pairwise_margin_loss(
    logps: torch.Tensor,
    batch_records: Sequence[Dict[str, object]],
    margin: float,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    losses: List[torch.Tensor] = []
    active_count = 0
    for index, record in enumerate(batch_records):
        if not _is_rank_active_record(record):
            continue
        candidate_actions, candidate_scores = _normalize_rank_candidates(record)
        positive_action = int(record["action"])
        hardest_negative = _select_rank_negative_action(
            record,
            candidate_actions,
            candidate_scores,
            positive_action=positive_action,
            logps_row=logps[index],
            device=device,
        )
        if hardest_negative is None:
            continue
        active_count += 1
        positive_logp = logps[index, positive_action]
        hardest_negative_logp = logps[index, int(hardest_negative)]
        losses.append(F.relu(margin - (positive_logp - hardest_negative_logp)))
    if not losses:
        return torch.zeros((), device=device), active_count
    return torch.stack(losses).mean(), active_count


def _rank_regression_reason(
    eval_result: Dict[str, object],
    reference_eval: Dict[str, object],
) -> str | None:
    reference_val = reference_eval["val_metrics"]
    current_val = eval_result["val_metrics"]
    reference_trap = reference_eval.get("trap_metrics") or {}
    current_trap = eval_result.get("trap_metrics") or {}
    if current_val["success_rate"] < 1.0:
        return f"success_rate dropped to {current_val['success_rate']:.3f}"
    if eval_result["teacher_agreement"] < reference_eval["teacher_agreement"] - RANK_GUARDRAILS["teacher_agreement_drop"]:
        return "teacher_agreement regressed beyond guardrail"
    if eval_result["hard_teacher_agreement"] < reference_eval["hard_teacher_agreement"] - RANK_GUARDRAILS["hard_teacher_agreement_drop"]:
        return "hard_teacher_agreement regressed beyond guardrail"
    if eval_result["conflict_teacher_agreement"] < reference_eval["conflict_teacher_agreement"] - RANK_GUARDRAILS["conflict_agreement_drop"]:
        return "single_vs_sync_conflict_agreement regressed beyond guardrail"
    if current_val["mean_avoidable_wait_time"] > reference_val["mean_avoidable_wait_time"] + RANK_GUARDRAILS["avoidable_wait_increase"]:
        return "mean_avoidable_wait_time regressed beyond guardrail"
    current_trap_aw = float(current_trap.get("mean_avoidable_wait_time", 0.0))
    reference_trap_aw = float(reference_trap.get("mean_avoidable_wait_time", 0.0))
    if current_trap_aw > reference_trap_aw + RANK_GUARDRAILS["trap_avoidable_wait_increase"]:
        return "trap_avoidable_wait regressed beyond guardrail"
    return None


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


def _format_family_summary(metrics_by_family: Dict[str, Dict[str, float]]) -> str:
    parts: List[str] = []
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


def _trap_teacher_gap(
    trap_metrics: Dict[str, float] | None,
    teacher_trap_metrics: Dict[str, float] | None,
) -> str:
    if trap_metrics is None or teacher_trap_metrics is None:
        return "n/a"
    aw_gap = trap_metrics["mean_avoidable_wait_time"] - teacher_trap_metrics["mean_avoidable_wait_time"]
    sm_gap = trap_metrics["mean_direct_sync_misassignment_rate"] - teacher_trap_metrics["mean_direct_sync_misassignment_rate"]
    return f"aw+{aw_gap:.2f}/sm+{sm_gap:.3f}"


def _evaluate_ranker(
    model: HeteroRankerPolicy,
    *,
    val_scenarios: Sequence[Dict],
    trap_val_scenarios: Sequence[Dict],
    teacher_records: Sequence[Dict[str, object]],
    hard_teacher_records: Sequence[Dict[str, object]],
    conflict_teacher_records: Sequence[Dict[str, object]],
    teacher_rollout_depth: int,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    val_metrics = evaluate_policy_on_scenarios(
        scenarios=val_scenarios,
        policy=model,
        device=device,
        deterministic=True,
        teacher_rollout_depth=teacher_rollout_depth,
    )
    trap_metrics = None
    if trap_val_scenarios:
        trap_metrics = evaluate_policy_on_scenarios(
            scenarios=trap_val_scenarios,
            policy=model,
            device=device,
            deterministic=True,
            teacher_rollout_depth=teacher_rollout_depth,
        )
    family_breakdown = evaluate_policy_family_breakdown(
        scenarios=val_scenarios,
        policy=model,
        device=device,
        deterministic=True,
        teacher_rollout_depth=teacher_rollout_depth,
    )
    return {
        "val_metrics": val_metrics,
        "trap_metrics": trap_metrics,
        "family_breakdown": family_breakdown,
        "teacher_agreement": _agreement_on_records(model, teacher_records, device, batch_size),
        "hard_teacher_agreement": _agreement_on_records(model, hard_teacher_records, device, batch_size),
        "conflict_teacher_agreement": _agreement_on_records(model, conflict_teacher_records, device, batch_size),
    }


def _log_stage(
    *,
    epoch: int,
    stage_name: str,
    loss_value: float,
    val_metrics: Dict[str, float],
    trap_metrics: Dict[str, float] | None,
    teacher_agreement: float,
    hard_teacher_agreement: float,
    conflict_teacher_agreement: float,
    teacher_reference: Dict[str, object],
    family_breakdown: Dict[str, Dict[str, float]],
    rank_diagnostics: Dict[str, float] | None = None,
) -> None:
    line = (
        f"[HRK][epoch {epoch:02d}][{stage_name}] "
        f"loss={loss_value:.4f} "
        f"teacher_agreement={teacher_agreement:.3f} "
        f"hard_teacher_agreement={hard_teacher_agreement:.3f} "
        f"single_vs_sync_conflict_agreement={conflict_teacher_agreement:.3f} "
        f"success={val_metrics['success_rate']:.3f} "
        f"makespan={val_metrics['mean_makespan']:.2f} "
        f"wait={val_metrics['mean_wait_time']:.2f} "
        f"avoidable_wait={val_metrics['mean_avoidable_wait_time']:.2f} "
        f"sync_misassign={val_metrics['mean_direct_sync_misassignment_rate']:.3f}"
    )
    if rank_diagnostics is not None:
        line += (
            f" teacher_action_top1_match_rate={rank_diagnostics['teacher_action_top1_match_rate']:.3f}"
            f" pairwise_positive_is_teacher_rate={rank_diagnostics['pairwise_positive_is_teacher_rate']:.3f}"
            f" rank_active_record_count={int(rank_diagnostics['rank_active_record_count'])}"
        )
    if trap_metrics is not None:
        line += (
            f" trap_success={trap_metrics['success_rate']:.3f}"
            f" trap_avoidable_wait={trap_metrics['mean_avoidable_wait_time']:.2f}"
            f" trap_sync_misassign={trap_metrics['mean_direct_sync_misassignment_rate']:.3f}"
            f" trap_teacher_gap={_trap_teacher_gap(trap_metrics, teacher_reference['trap_val_metrics'])}"
        )
    print(line)
    print(f"[HRK][family {epoch:02d}] {_format_family_summary(family_breakdown)}")
    print(f"[HRK][gap {epoch:02d}] {_hard_family_gap_summary(family_breakdown, teacher_reference['family_breakdown'])}")


def _checkpoint_metadata(
    base_metadata: Dict[str, object],
    *,
    epoch: int,
    stage_name: str,
    teacher_reference: Dict[str, object],
    eval_result: Dict[str, object],
) -> Dict[str, object]:
    trap_metrics = eval_result["trap_metrics"] or {}
    deploy_ready = (
        eval_result["teacher_agreement"] >= DEPLOY_THRESHOLDS["teacher_agreement"]
        and eval_result["hard_teacher_agreement"] >= DEPLOY_THRESHOLDS["hard_teacher_agreement"]
        and eval_result["conflict_teacher_agreement"] >= DEPLOY_THRESHOLDS["single_vs_sync_conflict_agreement"]
        and eval_result["val_metrics"]["success_rate"] >= DEPLOY_THRESHOLDS["success_rate"]
        and eval_result["val_metrics"]["mean_makespan"] <= DEPLOY_THRESHOLDS["mean_makespan"]
        and eval_result["val_metrics"]["mean_avoidable_wait_time"] <= DEPLOY_THRESHOLDS["mean_avoidable_wait_time"]
        and float(trap_metrics.get("mean_avoidable_wait_time", float("inf"))) <= DEPLOY_THRESHOLDS["trap_avoidable_wait"]
    )
    return {
        **base_metadata,
        "policy_type": "hetero_ranker",
        "stage": stage_name,
        "epoch": epoch,
        "teacher_reference": teacher_reference,
        "val_metrics": eval_result["val_metrics"],
        "trap_val_metrics": eval_result["trap_metrics"],
        "teacher_agreement": eval_result["teacher_agreement"],
        "hard_teacher_agreement": eval_result["hard_teacher_agreement"],
        "single_vs_sync_conflict_agreement": eval_result["conflict_teacher_agreement"],
        "deploy_ready": deploy_ready,
        "deploy_thresholds": DEPLOY_THRESHOLDS,
    }


def _deployment_failures(metadata: Dict[str, object]) -> List[str]:
    val_metrics = metadata.get("val_metrics", {}) or {}
    trap_metrics = metadata.get("trap_val_metrics", {}) or {}
    failures: List[str] = []
    if float(metadata.get("teacher_agreement", 0.0)) < DEPLOY_THRESHOLDS["teacher_agreement"]:
        failures.append(
            f"teacher_agreement {float(metadata.get('teacher_agreement', 0.0)):.3f} < {DEPLOY_THRESHOLDS['teacher_agreement']:.3f}"
        )
    if float(metadata.get("hard_teacher_agreement", 0.0)) < DEPLOY_THRESHOLDS["hard_teacher_agreement"]:
        failures.append(
            f"hard_teacher_agreement {float(metadata.get('hard_teacher_agreement', 0.0)):.3f} < {DEPLOY_THRESHOLDS['hard_teacher_agreement']:.3f}"
        )
    if float(metadata.get("single_vs_sync_conflict_agreement", 0.0)) < DEPLOY_THRESHOLDS["single_vs_sync_conflict_agreement"]:
        failures.append(
            "single_vs_sync_conflict_agreement "
            f"{float(metadata.get('single_vs_sync_conflict_agreement', 0.0)):.3f} < "
            f"{DEPLOY_THRESHOLDS['single_vs_sync_conflict_agreement']:.3f}"
        )
    if float(val_metrics.get("success_rate", 0.0)) < DEPLOY_THRESHOLDS["success_rate"]:
        failures.append(
            f"success_rate {float(val_metrics.get('success_rate', 0.0)):.3f} < {DEPLOY_THRESHOLDS['success_rate']:.3f}"
        )
    if float(val_metrics.get("mean_makespan", float('inf'))) > DEPLOY_THRESHOLDS["mean_makespan"]:
        failures.append(
            f"mean_makespan {float(val_metrics.get('mean_makespan', float('inf'))):.2f} > {DEPLOY_THRESHOLDS['mean_makespan']:.2f}"
        )
    if float(val_metrics.get("mean_avoidable_wait_time", float('inf'))) > DEPLOY_THRESHOLDS["mean_avoidable_wait_time"]:
        failures.append(
            "mean_avoidable_wait_time "
            f"{float(val_metrics.get('mean_avoidable_wait_time', float('inf'))):.2f} > "
            f"{DEPLOY_THRESHOLDS['mean_avoidable_wait_time']:.2f}"
        )
    if float(trap_metrics.get("mean_avoidable_wait_time", float('inf'))) > DEPLOY_THRESHOLDS["trap_avoidable_wait"]:
        failures.append(
            "trap_avoidable_wait "
            f"{float(trap_metrics.get('mean_avoidable_wait_time', float('inf'))):.2f} > "
            f"{DEPLOY_THRESHOLDS['trap_avoidable_wait']:.2f}"
        )
    return failures


def _maybe_save_deploy_checkpoint(
    *,
    save_dir: Path,
    model: HeteroRankerPolicy,
    optimizer: torch.optim.Optimizer,
    metadata_with_eval: Dict[str, object],
    score: float,
    best_deploy_score: float,
) -> tuple[float, bool]:
    if not bool(metadata_with_eval.get("deploy_ready", False)):
        return best_deploy_score, False
    if score <= best_deploy_score:
        return best_deploy_score, False
    deploy_metadata = {
        **metadata_with_eval,
        "deployment_role": "deploy_best",
    }
    save_hetero_ranker_checkpoint(
        save_dir / "best_scheduler_ranker_deploy.pt",
        model,
        optimizer=optimizer,
        metadata=deploy_metadata,
    )
    return score, True


def _checkpoint_summary(path: Path, metadata: Dict[str, object] | None) -> Dict[str, object] | None:
    if metadata is None:
        return None
    return {
        "path": str(path),
        "policy_type": metadata.get("policy_type"),
        "stage": metadata.get("stage"),
        "epoch": metadata.get("epoch"),
        "deploy_ready": bool(metadata.get("deploy_ready", False)),
        "teacher_agreement": float(metadata.get("teacher_agreement", 0.0)),
        "hard_teacher_agreement": float(metadata.get("hard_teacher_agreement", 0.0)),
        "single_vs_sync_conflict_agreement": float(metadata.get("single_vs_sync_conflict_agreement", 0.0)),
        "val_metrics": metadata.get("val_metrics"),
        "trap_val_metrics": metadata.get("trap_val_metrics"),
    }


def _write_deployment_report(
    *,
    save_dir: Path,
    teacher_reference: Dict[str, object],
    best_metadata: Dict[str, object] | None,
    deploy_metadata: Dict[str, object] | None,
    deploy_checkpoint_generated_this_run: bool,
    preexisting_deploy_checkpoint: bool,
) -> None:
    best_path = save_dir / "best_scheduler_ranker.pt"
    deploy_path = save_dir / "best_scheduler_ranker_deploy.pt"
    report = {
        "chosen_teacher_policy": teacher_reference.get("chosen_expert_policy"),
        "teacher_rollout_depth": teacher_reference.get("teacher_rollout_depth"),
        "teacher_baseline": {
            "val_metrics": teacher_reference.get("val_metrics"),
            "trap_val_metrics": teacher_reference.get("trap_val_metrics"),
        },
        "deploy_thresholds": DEPLOY_THRESHOLDS,
        "best_checkpoint": _checkpoint_summary(best_path, best_metadata),
        "deploy_checkpoint": _checkpoint_summary(deploy_path, deploy_metadata),
        "deploy_checkpoint_generated_this_run": deploy_checkpoint_generated_this_run,
        "preexisting_deploy_checkpoint": preexisting_deploy_checkpoint,
        "deploy_failures": [] if deploy_metadata is not None else _deployment_failures(best_metadata or {}),
    }
    (save_dir / "deployment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_ce_stage(
    *,
    stage_name: str,
    epoch_start: int,
    epoch_count: int,
    model: HeteroRankerPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    val_scenarios: Sequence[Dict],
    trap_val_scenarios: Sequence[Dict],
    teacher_records: Sequence[Dict[str, object]],
    hard_teacher_records: Sequence[Dict[str, object]],
    conflict_teacher_records: Sequence[Dict[str, object]],
    teacher_reference: Dict[str, object],
    metadata: Dict[str, object],
    save_dir: Path,
    best_score: float,
    best_deploy_score: float,
) -> tuple[float, float, bool]:
    deploy_checkpoint_updated = False
    for local_epoch in range(epoch_count):
        epoch = epoch_start + local_epoch
        running_loss = 0.0
        steps = 0
        model.train()
        for obs_batch_np, actions_np in loader:
            obs_batch = obs_batch_to_torch(obs_batch_np, device)
            actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
            _, logps = model(obs_batch)
            loss = _masked_label_smoothed_loss(logps, actions, obs_batch["global_mask"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            steps += 1

        eval_result = _evaluate_ranker(
            model,
            val_scenarios=val_scenarios,
            trap_val_scenarios=trap_val_scenarios,
            teacher_records=teacher_records,
            hard_teacher_records=hard_teacher_records,
            conflict_teacher_records=conflict_teacher_records,
            teacher_rollout_depth=int(teacher_reference["teacher_rollout_depth"]),
            device=device,
            batch_size=loader.batch_size or 128,
        )
        loss_value = running_loss / max(steps, 1)
        _log_stage(
            epoch=epoch,
            stage_name=stage_name,
            loss_value=loss_value,
            val_metrics=eval_result["val_metrics"],
            trap_metrics=eval_result["trap_metrics"],
            teacher_agreement=eval_result["teacher_agreement"],
            hard_teacher_agreement=eval_result["hard_teacher_agreement"],
            conflict_teacher_agreement=eval_result["conflict_teacher_agreement"],
            teacher_reference=teacher_reference,
            family_breakdown=eval_result["family_breakdown"],
        )
        metadata_with_eval = _checkpoint_metadata(
            metadata,
            epoch=epoch,
            stage_name=stage_name,
            teacher_reference=teacher_reference,
            eval_result=eval_result,
        )
        save_hetero_ranker_checkpoint(save_dir / "latest_scheduler_ranker.pt", model, optimizer=optimizer, metadata=metadata_with_eval)
        score = ranker_selection_score(
            eval_result["val_metrics"],
            eval_result["teacher_agreement"],
            eval_result["hard_teacher_agreement"],
            eval_result["conflict_teacher_agreement"],
            eval_result["trap_metrics"],
        )
        if score > best_score:
            best_score = score
            save_hetero_ranker_checkpoint(save_dir / "best_scheduler_ranker.pt", model, optimizer=optimizer, metadata=metadata_with_eval)
        best_deploy_score, deploy_saved = _maybe_save_deploy_checkpoint(
            save_dir=save_dir,
            model=model,
            optimizer=optimizer,
            metadata_with_eval=metadata_with_eval,
            score=score,
            best_deploy_score=best_deploy_score,
        )
        deploy_checkpoint_updated = deploy_checkpoint_updated or deploy_saved
    return best_score, best_deploy_score, deploy_checkpoint_updated


def _run_pairwise_stage(
    *,
    stage_name: str,
    epoch_start: int,
    epoch_count: int,
    model: HeteroRankerPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float,
    ce_mix: float,
    val_scenarios: Sequence[Dict],
    trap_val_scenarios: Sequence[Dict],
    teacher_records: Sequence[Dict[str, object]],
    hard_teacher_records: Sequence[Dict[str, object]],
    conflict_teacher_records: Sequence[Dict[str, object]],
    teacher_reference: Dict[str, object],
    metadata: Dict[str, object],
    save_dir: Path,
    best_score: float,
    best_deploy_score: float,
    reference_eval: Dict[str, object] | None = None,
    rank_diagnostics: Dict[str, float] | None = None,
) -> tuple[float, float, bool, bool]:
    stage_aborted = False
    deploy_checkpoint_updated = False
    for local_epoch in range(epoch_count):
        epoch = epoch_start + local_epoch
        running_loss = 0.0
        steps = 0
        active_records = 0
        model.train()
        for batch_records in loader:
            obs_batch_np, actions_np = collate_supervised_batch([_record_to_sample(record) for record in batch_records])
            obs_batch = obs_batch_to_torch(obs_batch_np, device)
            actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
            _, logps = model(obs_batch)
            pair_loss, batch_active = _pairwise_margin_loss(logps, batch_records, margin, device)
            ce_loss = _masked_label_smoothed_loss(logps, actions, obs_batch["global_mask"])
            loss = pair_loss + ce_mix * ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            steps += 1
            active_records += int(batch_active)

        eval_result = _evaluate_ranker(
            model,
            val_scenarios=val_scenarios,
            trap_val_scenarios=trap_val_scenarios,
            teacher_records=teacher_records,
            hard_teacher_records=hard_teacher_records,
            conflict_teacher_records=conflict_teacher_records,
            teacher_rollout_depth=int(teacher_reference["teacher_rollout_depth"]),
            device=device,
            batch_size=loader.batch_size or 64,
        )
        loss_value = running_loss / max(steps, 1)
        _log_stage(
            epoch=epoch,
            stage_name=stage_name,
            loss_value=loss_value,
            val_metrics=eval_result["val_metrics"],
            trap_metrics=eval_result["trap_metrics"],
            teacher_agreement=eval_result["teacher_agreement"],
            hard_teacher_agreement=eval_result["hard_teacher_agreement"],
            conflict_teacher_agreement=eval_result["conflict_teacher_agreement"],
            teacher_reference=teacher_reference,
            family_breakdown=eval_result["family_breakdown"],
            rank_diagnostics=rank_diagnostics,
        )
        metadata_with_eval = _checkpoint_metadata(
            metadata,
            epoch=epoch,
            stage_name=stage_name,
            teacher_reference=teacher_reference,
            eval_result=eval_result,
        )
        save_hetero_ranker_checkpoint(save_dir / "latest_scheduler_ranker.pt", model, optimizer=optimizer, metadata=metadata_with_eval)
        score = ranker_selection_score(
            eval_result["val_metrics"],
            eval_result["teacher_agreement"],
            eval_result["hard_teacher_agreement"],
            eval_result["conflict_teacher_agreement"],
            eval_result["trap_metrics"],
        )
        if score > best_score:
            best_score = score
            save_hetero_ranker_checkpoint(save_dir / "best_scheduler_ranker.pt", model, optimizer=optimizer, metadata=metadata_with_eval)
        best_deploy_score, deploy_saved = _maybe_save_deploy_checkpoint(
            save_dir=save_dir,
            model=model,
            optimizer=optimizer,
            metadata_with_eval=metadata_with_eval,
            score=score,
            best_deploy_score=best_deploy_score,
        )
        deploy_checkpoint_updated = deploy_checkpoint_updated or deploy_saved
        if reference_eval is not None:
            regression_reason = _rank_regression_reason(eval_result, reference_eval)
            if regression_reason is not None:
                print(
                    f"[HRK][{stage_name}-stop {epoch:02d}] "
                    f"Stopped further {stage_name} epochs: {regression_reason} "
                    f"(active_pair_records={active_records})"
                )
                stage_aborted = True
                break
    return best_score, best_deploy_score, stage_aborted, deploy_checkpoint_updated


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    preexisting_deploy_checkpoint = (save_dir / "best_scheduler_ranker_deploy.pt").exists()
    deploy_checkpoint_generated_this_run = False

    train_by_family = _load_scenarios_by_family(args.scenario_dir, args.train_split, args.limit_train_per_family)
    val_by_family = _load_scenarios_by_family(args.scenario_dir, args.val_split, args.limit_val_per_family)
    train_scenarios = _flatten_family_scenarios(train_by_family)
    val_scenarios = _flatten_family_scenarios(val_by_family)
    trap_val_scenarios = list(val_by_family.get("partial_coalition_trap", []))
    if not train_scenarios:
        raise ValueError("Training scenarios are empty. Generate offline_maps_v2 first.")
    if not val_scenarios:
        raise ValueError("Validation scenarios are empty.")

    teacher_policies = [args.expert_policy] if args.expert_policy != "auto" else TEACHER_CANDIDATES
    teacher_candidates: Dict[str, Dict[str, object]] = {}
    chosen_policy = teacher_policies[0]
    chosen_score = float("-inf")
    for expert_policy in teacher_policies:
        candidate_val_metrics = evaluate_policy_on_scenarios(
            scenarios=val_scenarios,
            policy=expert_policy,
            deterministic=True,
            teacher_rollout_depth=args.teacher_rollout_depth,
        )
        candidate_trap_metrics = None
        if trap_val_scenarios:
            candidate_trap_metrics = evaluate_policy_on_scenarios(
                scenarios=trap_val_scenarios,
                policy=expert_policy,
                deterministic=True,
                teacher_rollout_depth=args.teacher_rollout_depth,
            )
        candidate_family_breakdown = evaluate_policy_family_breakdown(
            scenarios=val_scenarios,
            policy=expert_policy,
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

    teacher_reference = {
        "expert_policy": args.expert_policy,
        "chosen_expert_policy": chosen_policy,
        "teacher_rollout_depth": args.teacher_rollout_depth,
        "teacher_candidates": teacher_candidates,
        "val_metrics": teacher_candidates[chosen_policy]["val_metrics"],
        "trap_val_metrics": teacher_candidates[chosen_policy]["trap_val_metrics"],
        "family_breakdown": teacher_candidates[chosen_policy]["family_breakdown"],
    }
    (save_dir / "teacher_val_reference.json").write_text(json.dumps(teacher_reference, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Hetero ranker teacher: requested={args.expert_policy} chosen={chosen_policy} score={chosen_score:.2f}")

    train_records: List[Dict[str, object]] = []
    family_counts: Dict[str, int] = {}
    for family in FAMILY_NAMES:
        family_scenarios = train_by_family.get(family, [])
        if not family_scenarios:
            continue
        family_records = collect_ranker_sample_records(
            family_scenarios,
            expert_policy=chosen_policy,
            teacher_rollout_depth=args.teacher_rollout_depth,
        )
        train_records.extend(family_records)
        family_counts[family] = len(family_records)
    if not train_records:
        raise ValueError("Teacher did not produce any ranker records.")

    val_teacher_records = collect_expert_sample_records(
        val_scenarios,
        expert_policy=chosen_policy,
        teacher_rollout_depth=args.teacher_rollout_depth,
    )
    hard_val_teacher_records = _filter_records(val_teacher_records, lambda record: bool(record["hard_state"]))
    conflict_val_teacher_records = _filter_records(val_teacher_records, lambda record: bool(record["single_sync_conflict"]))

    _, base_loader = _build_supervised_loader(train_records, family_counts, args.batch_size)
    pair_records = [record for record in train_records if _is_rank_active_record(record)]
    pair_loader: DataLoader | None = None
    rank_diagnostics: Dict[str, float] | None = None
    if pair_records:
        pair_family_counts: Dict[str, int] = {}
        for record in pair_records:
            family = str(record["family"])
            pair_family_counts[family] = pair_family_counts.get(family, 0) + 1
        _, pair_loader = _build_record_loader(pair_records, pair_family_counts, args.pairwise_batch_size)
        rank_diagnostics = _rank_stage_diagnostics(pair_records)

    sample_obs = train_records[0]["obs"]
    if args.init_model:
        model, checkpoint = load_hetero_ranker_checkpoint(args.init_model, device=device)
        print(f"Loaded initial model: {args.init_model}")
        metadata = checkpoint.get("metadata", {})
    else:
        model = HeteroRankerPolicy(
            agent_input_dim=int(sample_obs["agent_inputs"].shape[-1]),
            task_input_dim=int(sample_obs["task_inputs"].shape[-1]),
        ).to(device)
        metadata = {}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    base_metadata: Dict[str, object] = {
        **metadata,
        "train_mode": "hetero_ranker",
        "chosen_teacher_policy": chosen_policy,
        "teacher_rollout_depth": args.teacher_rollout_depth,
        "rank_epochs": args.rank_epochs,
        "ce_mix": args.ce_mix,
        "dagger_mode": args.dagger_mode,
        "dagger_max_records_per_family": args.dagger_max_records_per_family,
    }
    best_score = float("-inf")

    print(f"Ranker teacher records: {len(train_records)}")
    best_deploy_score = float("-inf")
    best_score, best_deploy_score, deploy_saved = _run_ce_stage(
        stage_name="ce",
        epoch_start=1,
        epoch_count=args.base_epochs,
        model=model,
        loader=base_loader,
        optimizer=optimizer,
        device=device,
        val_scenarios=val_scenarios,
        trap_val_scenarios=trap_val_scenarios,
        teacher_records=val_teacher_records,
        hard_teacher_records=hard_val_teacher_records,
        conflict_teacher_records=conflict_val_teacher_records,
        teacher_reference=teacher_reference,
        metadata=base_metadata,
        save_dir=save_dir,
        best_score=best_score,
        best_deploy_score=best_deploy_score,
    )
    deploy_checkpoint_generated_this_run = deploy_checkpoint_generated_this_run or deploy_saved

    rank_aborted = False
    if args.rank_epochs > 0 and pair_loader is not None:
        model, best_checkpoint = load_hetero_ranker_checkpoint(save_dir / "best_scheduler_ranker.pt", device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_meta = best_checkpoint.get("metadata", {})
        rank_reference_eval = {
            "val_metrics": best_meta.get("val_metrics", {}),
            "trap_metrics": best_meta.get("trap_val_metrics"),
            "teacher_agreement": float(best_meta.get("teacher_agreement", 0.0)),
            "hard_teacher_agreement": float(best_meta.get("hard_teacher_agreement", 0.0)),
            "conflict_teacher_agreement": float(best_meta.get("single_vs_sync_conflict_agreement", 0.0)),
        }
        best_score, best_deploy_score, rank_aborted, deploy_saved = _run_pairwise_stage(
            stage_name="rank",
            epoch_start=args.base_epochs + 1,
            epoch_count=args.rank_epochs,
            model=model,
            loader=pair_loader,
            optimizer=optimizer,
            device=device,
            margin=args.margin,
            ce_mix=args.ce_mix,
            val_scenarios=val_scenarios,
            trap_val_scenarios=trap_val_scenarios,
            teacher_records=val_teacher_records,
            hard_teacher_records=hard_val_teacher_records,
            conflict_teacher_records=conflict_val_teacher_records,
            teacher_reference=teacher_reference,
            metadata=base_metadata,
            save_dir=save_dir,
            best_score=best_score,
            best_deploy_score=best_deploy_score,
            reference_eval=rank_reference_eval,
            rank_diagnostics=rank_diagnostics,
        )
        deploy_checkpoint_generated_this_run = deploy_checkpoint_generated_this_run or deploy_saved
    elif args.rank_epochs > 0:
        print("Rank stage skipped: no hard-boundary records met the filter.")
    else:
        print("Rank stage skipped: default CE-only training (set --rank-epochs > 0 to enable experimental pairwise rank).")

    dagger_records, dagger_family_counts = _select_dagger_records(
        model,
        train_scenarios,
        expert_policy=chosen_policy,
        teacher_rollout_depth=args.teacher_rollout_depth,
        device=device,
        dagger_mode=args.dagger_mode,
        max_records_per_family=args.dagger_max_records_per_family,
    )
    if args.dagger_mode == "off":
        print("DAgger refresh skipped: --dagger-mode off")
    elif rank_aborted:
        print("DAgger refresh skipped: rank stage hit guardrails; keeping CE/best checkpoint as the deployment target.")
    elif dagger_records and args.dagger_epochs > 0:
        _, dagger_loader = _build_record_loader(
            dagger_records,
            dagger_family_counts,
            args.pairwise_batch_size,
            disagreement_multiplier=6.0,
            conflict_disagreement_multiplier=2.0,
        )
        print(f"DAgger refresh records: {len(dagger_records)}")
        model, best_checkpoint = load_hetero_ranker_checkpoint(save_dir / "best_scheduler_ranker.pt", device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_meta = best_checkpoint.get("metadata", {})
        dagger_reference_eval = {
            "val_metrics": best_meta.get("val_metrics", {}),
            "trap_metrics": best_meta.get("trap_val_metrics"),
            "teacher_agreement": float(best_meta.get("teacher_agreement", 0.0)),
            "hard_teacher_agreement": float(best_meta.get("hard_teacher_agreement", 0.0)),
            "conflict_teacher_agreement": float(best_meta.get("single_vs_sync_conflict_agreement", 0.0)),
        }
        dagger_diagnostics = _rank_stage_diagnostics(dagger_records)
        best_score, best_deploy_score, _, deploy_saved = _run_pairwise_stage(
            stage_name="dagger",
            epoch_start=args.base_epochs + args.rank_epochs + 1,
            epoch_count=args.dagger_epochs,
            model=model,
            loader=dagger_loader,
            optimizer=optimizer,
            device=device,
            margin=args.margin,
            ce_mix=args.ce_mix,
            val_scenarios=val_scenarios,
            trap_val_scenarios=trap_val_scenarios,
            teacher_records=val_teacher_records,
            hard_teacher_records=hard_val_teacher_records,
            conflict_teacher_records=conflict_val_teacher_records,
            teacher_reference=teacher_reference,
            metadata={**base_metadata, "dagger_records": len(dagger_records)},
            save_dir=save_dir,
            best_score=best_score,
            best_deploy_score=best_deploy_score,
            reference_eval=dagger_reference_eval,
            rank_diagnostics=dagger_diagnostics,
        )
        deploy_checkpoint_generated_this_run = deploy_checkpoint_generated_this_run or deploy_saved
    else:
        print("DAgger refresh skipped: no hard-only disagreement records met the filter.")

    best_metadata = None
    best_checkpoint_path = save_dir / "best_scheduler_ranker.pt"
    if best_checkpoint_path.exists():
        _, best_checkpoint = load_hetero_ranker_checkpoint(best_checkpoint_path, device=device)
        best_metadata = best_checkpoint.get("metadata", {})
    deploy_metadata = None
    deploy_checkpoint_path = save_dir / "best_scheduler_ranker_deploy.pt"
    if deploy_checkpoint_generated_this_run and deploy_checkpoint_path.exists():
        _, deploy_checkpoint = load_hetero_ranker_checkpoint(deploy_checkpoint_path, device=device)
        deploy_metadata = deploy_checkpoint.get("metadata", {})
    _write_deployment_report(
        save_dir=save_dir,
        teacher_reference=teacher_reference,
        best_metadata=best_metadata,
        deploy_metadata=deploy_metadata,
        deploy_checkpoint_generated_this_run=deploy_checkpoint_generated_this_run,
        preexisting_deploy_checkpoint=preexisting_deploy_checkpoint,
    )


if __name__ == "__main__":
    main()
