from __future__ import annotations

import numpy as np
import torch

from scheduler_nav_runner import LoadedSchedulerPolicy, SchedulerNavRunner
from hetero_dispatch_env import TASK_IDX


def _fake_obs(
    *,
    low_margin: bool,
    hard_family: str = "role_mismatch",
    deploy_ready: bool = False,
) -> tuple[SchedulerNavRunner, dict]:
    runner = SchedulerNavRunner(scheduler_guard_mode="auto", scheduler_min_margin=0.15)
    runner.scenario = {"family": hard_family}
    runner.robot_order = ["robot_0"]
    runner.task_order = ["task_single", "task_sync"]
    runner.metrics = {}
    task_inputs = np.zeros((3, max(TASK_IDX.values()) + 1), dtype=np.float32)
    task_inputs[1, TASK_IDX["is_single"]] = 1.0
    task_inputs[2, TASK_IDX["is_sync"]] = 1.0
    hetero_obs = {
        "agent_inputs": np.zeros((1, 4), dtype=np.float32),
        "task_inputs": task_inputs,
        "global_mask": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "current_agent_index": np.array([0], dtype=np.int64),
    }

    class FakeRanker:
        def eval(self) -> None:
            return None

        def act(self, obs, deterministic: bool = False):
            action = torch.tensor([2], dtype=torch.long)
            log_prob = torch.tensor([0.0], dtype=torch.float32)
            logps = torch.log(torch.tensor([[0.0, 0.51, 0.49]], dtype=torch.float32))
            return action, log_prob, logps

        def __call__(self, obs):
            probs = (
                torch.tensor([[0.0, 0.51, 0.49]], dtype=torch.float32)
                if low_margin
                else torch.tensor([[0.0, 0.90, 0.10]], dtype=torch.float32)
            )
            logps = torch.log(probs.clamp_min(1e-8))
            return probs, logps

    runner.scheduler_policy = LoadedSchedulerPolicy(
        policy_type="hetero_ranker",
        model=FakeRanker(),
        metadata={"deploy_ready": deploy_ready},
    )
    runner._build_hetero_scheduler_obs = lambda current_slot, pending_actions: hetero_obs  # type: ignore[method-assign]
    runner._upfront_wait_aware_action = lambda current_slot, pending_actions: 1  # type: ignore[method-assign]
    return runner, hetero_obs


def test_guard_triggers_on_low_margin_hard_state() -> None:
    runner, _ = _fake_obs(low_margin=True)
    action = runner._scheduler_action({}, current_slot=0, pending_actions=np.zeros(1, dtype=np.int64))
    assert action == 1
    assert runner.metrics["ranker_low_margin_count"] == 1
    assert runner.metrics["ranker_guard_trigger_count"] == 1
    assert runner.metrics["ranker_unsafe_checkpoint_count"] == 1


def test_guard_does_not_trigger_on_high_margin() -> None:
    runner, _ = _fake_obs(low_margin=False)
    action = runner._scheduler_action({}, current_slot=0, pending_actions=np.zeros(1, dtype=np.int64))
    assert action == 2
    assert runner.metrics["ranker_low_margin_count"] == 0
    assert runner.metrics["ranker_guard_trigger_count"] == 0
    assert runner.metrics["ranker_unsafe_checkpoint_count"] == 1


def test_guard_does_not_trigger_on_non_hard_state() -> None:
    runner, hetero_obs = _fake_obs(low_margin=True, hard_family="open_balance")
    hetero_obs["task_inputs"][2, TASK_IDX["is_sync"]] = 0.0
    action = runner._scheduler_action({}, current_slot=0, pending_actions=np.zeros(1, dtype=np.int64))
    assert action == 2
    assert runner.metrics["ranker_low_margin_count"] == 1
    assert runner.metrics["ranker_guard_trigger_count"] == 0
    assert runner.metrics["ranker_unsafe_checkpoint_count"] == 1


def test_guard_can_trigger_for_deploy_ready_checkpoint_on_low_margin_hard_state() -> None:
    runner, _ = _fake_obs(low_margin=True, deploy_ready=True)
    action = runner._scheduler_action({}, current_slot=0, pending_actions=np.zeros(1, dtype=np.int64))
    assert action == 1
    assert runner.metrics["ranker_low_margin_count"] == 1
    assert runner.metrics["ranker_guard_trigger_count"] == 1
    assert runner.metrics["ranker_unsafe_checkpoint_count"] == 0


if __name__ == "__main__":
    test_guard_triggers_on_low_margin_hard_state()
    test_guard_does_not_trigger_on_high_margin()
    test_guard_does_not_trigger_on_non_hard_state()
    test_guard_can_trigger_for_deploy_ready_checkpoint_on_low_margin_hard_state()
    print("ranker guard tests passed")
