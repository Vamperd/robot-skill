from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_scheduler_hetero_ranker as tr
from hetero_dispatch_env import TASK_FEATURE_DIM, TASK_IDX


def _make_record(
    *,
    family: str,
    teacher_action: int,
    candidate_actions: list[int],
    candidate_scores: list[float],
    single_sync_conflict: bool,
    single_action: int | None = None,
    sync_action: int | None = None,
) -> dict[str, object]:
    task_inputs = np.zeros((4, TASK_FEATURE_DIM), dtype=np.float32)
    task_inputs[0, TASK_IDX["is_wait_node"]] = 1.0
    if single_action is not None:
        task_inputs[single_action, TASK_IDX["is_single"]] = 1.0
    if sync_action is not None:
        task_inputs[sync_action, TASK_IDX["is_sync"]] = 1.0
    global_mask = np.ones((4,), dtype=np.float32)
    global_mask[0] = 0.0
    for action in set(candidate_actions + [teacher_action]):
        global_mask[action] = 0.0
    return {
        "family": family,
        "action": teacher_action,
        "candidate_actions": candidate_actions,
        "candidate_scores": candidate_scores,
        "single_sync_conflict": single_sync_conflict,
        "obs": {
            "task_inputs": task_inputs,
            "global_mask": global_mask,
        },
    }


class RankerTrainingTests(unittest.TestCase):
    def test_pairwise_positive_stays_teacher_when_candidate_top1_differs(self) -> None:
        record = _make_record(
            family="role_mismatch",
            teacher_action=1,
            candidate_actions=[2, 1],
            candidate_scores=[1.0, 0.2],
            single_sync_conflict=True,
            single_action=1,
            sync_action=2,
        )
        candidate_actions, candidate_scores = tr._normalize_rank_candidates(record)
        self.assertEqual(candidate_actions[0], 2)
        self.assertEqual(record["action"], 1)
        negative = tr._select_rank_negative_action(
            record,
            candidate_actions,
            candidate_scores,
            positive_action=int(record["action"]),
            logps_row=torch.log_softmax(torch.tensor([0.0, 0.3, 0.9, -1.0]), dim=0),
            device=torch.device("cpu"),
        )
        self.assertEqual(negative, 2)

    def test_single_sync_conflict_prefers_opponent_action(self) -> None:
        record = _make_record(
            family="partial_coalition_trap",
            teacher_action=2,
            candidate_actions=[2, 1, 3],
            candidate_scores=[0.9, 0.4, 0.2],
            single_sync_conflict=True,
            single_action=1,
            sync_action=2,
        )
        negative = tr._select_rank_negative_action(
            record,
            [2, 1, 3],
            [0.9, 0.4, 0.2],
            positive_action=2,
            logps_row=torch.log_softmax(torch.tensor([0.0, 0.6, 0.8, 1.0]), dim=0),
            device=torch.device("cpu"),
        )
        self.assertEqual(negative, 1)

    def test_easy_state_does_not_enter_rank_stage(self) -> None:
        easy_record = _make_record(
            family="open_balance",
            teacher_action=1,
            candidate_actions=[1, 2],
            candidate_scores=[0.8, 0.7],
            single_sync_conflict=False,
            single_action=1,
            sync_action=2,
        )
        hard_record = _make_record(
            family="role_mismatch",
            teacher_action=1,
            candidate_actions=[1, 2],
            candidate_scores=[0.8, 0.7],
            single_sync_conflict=False,
            single_action=1,
            sync_action=2,
        )
        self.assertFalse(tr._is_rank_active_record(easy_record))
        self.assertTrue(tr._is_rank_active_record(hard_record))


if __name__ == "__main__":
    unittest.main()
