from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from auction_mrta import AuctionMRTA
from ctas_d import CTASD
from milp_scheduler_small import MilpSchedulerSmall
from sas import SAS


class FakeHeteroEnv:
    def __init__(
        self,
        *,
        robot_order,
        task_order,
        robot_states,
        task_specs,
        legal_masks,
        eta_map,
        predict_map,
        event_index: int = 0,
        max_time: float = 2500.0,
    ) -> None:
        self.base_env = self
        self.current_scenario = {"scenario_id": "unit_test"}
        self.event_index = event_index
        self.robot_order = list(robot_order)
        self.task_order = list(task_order)
        self.robot_states = robot_states
        self.task_specs = task_specs
        self.task_states = {
            task_id: {
                "assigned_robot_ids": set(),
                "onsite_robot_ids": set(),
                "contributors": set(),
                "completed": False,
                "coalition_ready_eta": 0.0,
            }
            for task_id in task_order
        }
        self.pending_actions = np.zeros(len(self.robot_order), dtype=np.int64)
        self.max_time = float(max_time)
        self.wait_timeout = 60.0
        self._legal_masks = {robot_id: np.asarray(mask, dtype=np.float32) for robot_id, mask in legal_masks.items()}
        self._eta_map = {(robot_id, task_id): float(value) for (robot_id, task_id), value in eta_map.items()}
        self._predict_map = {
            (robot_id, task_id): tuple(float(v) for v in values)
            for (robot_id, task_id), values in predict_map.items()
        }

    def _legal_action_mask(self, robot_id: str):
        mask = np.array(self._legal_masks[robot_id], copy=True)
        for action in np.asarray(self.pending_actions, dtype=np.int64):
            if int(action) > 0 and int(action) < len(mask):
                mask[int(action)] = 0.0
        return mask

    def _robot_available_eta_for_task(self, robot_id: str, task_id: str) -> float:
        return float(self._eta_map[(robot_id, task_id)])

    def _predict_task_outcome(self, robot_id: str, task_id: str):
        return self._predict_map[(robot_id, task_id)]

    def _task_unlock_value(self, task_id: str) -> int:
        del task_id
        return 0


def test_auction_chooses_only_legal_single_task() -> None:
    env = FakeHeteroEnv(
        robot_order=["robot_a"],
        task_order=["task_1", "task_2"],
        robot_states={
            "robot_a": {
                "role": "carrier",
                "status": "idle",
                "speed_multiplier": 1.0,
                "service_multiplier": 1.0,
            }
        },
        task_specs={
            "task_1": {"kind": "single", "service_time": 30.0, "required_roles": {"carrier": 1}},
            "task_2": {"kind": "single", "service_time": 10.0, "required_roles": {"carrier": 1}},
        },
        legal_masks={"robot_a": [0.0, 0.0, 1.0]},
        eta_map={
            ("robot_a", "task_1"): 5.0,
            ("robot_a", "task_2"): 1.0,
        },
        predict_map={
            ("robot_a", "task_1"): (5.0, 5.0, 35.0),
            ("robot_a", "task_2"): (1.0, 1.0, 11.0),
        },
    )
    policy = AuctionMRTA()
    action = policy.select_action(env, None, 0, env._legal_action_mask("robot_a"), np.random.default_rng(0))
    assert action == 2


def test_auction_waits_when_sync_coalition_is_infeasible() -> None:
    env = FakeHeteroEnv(
        robot_order=["robot_a"],
        task_order=["task_sync"],
        robot_states={
            "robot_a": {
                "role": "carrier",
                "status": "idle",
                "speed_multiplier": 1.0,
                "service_multiplier": 1.0,
            }
        },
        task_specs={
            "task_sync": {
                "kind": "sync",
                "service_time": 20.0,
                "required_roles": {"carrier": 1, "scout": 1},
            }
        },
        legal_masks={"robot_a": [0.0, 1.0]},
        eta_map={("robot_a", "task_sync"): 4.0},
        predict_map={("robot_a", "task_sync"): (4.0, 4.0, 30.0)},
    )
    policy = AuctionMRTA()
    action = policy.select_action(env, None, 0, env._legal_action_mask("robot_a"), np.random.default_rng(0))
    assert action == 0


def test_milp_dependency_or_small_joint_assignment() -> None:
    try:
        policy = MilpSchedulerSmall()
    except RuntimeError as exc:
        assert "PuLP" in str(exc)
        return

    env = FakeHeteroEnv(
        robot_order=["robot_a", "robot_b"],
        task_order=["task_1", "task_2"],
        robot_states={
            "robot_a": {
                "role": "carrier",
                "status": "idle",
                "speed_multiplier": 1.0,
                "service_multiplier": 1.0,
            },
            "robot_b": {
                "role": "scout",
                "status": "idle",
                "speed_multiplier": 1.0,
                "service_multiplier": 1.0,
            },
        },
        task_specs={
            "task_1": {"kind": "single", "service_time": 15.0, "required_roles": {"carrier": 1}, "priority": 1.0},
            "task_2": {"kind": "single", "service_time": 12.0, "required_roles": {"scout": 1}, "priority": 1.0},
        },
        legal_masks={
            "robot_a": [0.0, 1.0, 0.0],
            "robot_b": [0.0, 0.0, 1.0],
        },
        eta_map={
            ("robot_a", "task_1"): 3.0,
            ("robot_a", "task_2"): 999.0,
            ("robot_b", "task_1"): 999.0,
            ("robot_b", "task_2"): 2.0,
        },
        predict_map={
            ("robot_a", "task_1"): (3.0, 3.0, 18.0),
            ("robot_b", "task_2"): (2.0, 2.0, 14.0),
        },
    )
    policy.reset_episode(env.current_scenario)
    action_a = policy.select_action(env, None, 0, env._legal_action_mask("robot_a"), np.random.default_rng(0))
    action_b = policy.select_action(env, None, 1, env._legal_action_mask("robot_b"), np.random.default_rng(0))
    assert action_a == 1
    assert action_b == 2
    diagnostics = policy.get_diagnostics()
    assert "mean_solve_time_ms" in diagnostics
    assert "timeout_rate" in diagnostics


def test_milp_oversize_falls_back_to_auction() -> None:
    try:
        policy = MilpSchedulerSmall(max_active_robots=4, max_candidate_tasks=6)
    except RuntimeError:
        return

    robot_order = [f"robot_{i}" for i in range(5)]
    task_order = [f"task_{i}" for i in range(5)]
    robot_states = {
        robot_id: {
            "role": "carrier",
            "status": "idle",
            "speed_multiplier": 1.0,
            "service_multiplier": 1.0,
        }
        for robot_id in robot_order
    }
    task_specs = {
        task_id: {"kind": "single", "service_time": 10.0, "required_roles": {"carrier": 1}, "priority": 1.0}
        for task_id in task_order
    }
    legal_masks = {
        robot_id: np.asarray([0.0] + [1.0] * len(task_order), dtype=np.float32)
        for robot_id in robot_order
    }
    eta_map = {
        (robot_id, task_id): float(index + task_slot + 1)
        for index, robot_id in enumerate(robot_order)
        for task_slot, task_id in enumerate(task_order)
    }
    predict_map = {
        (robot_id, task_id): (eta_map[(robot_id, task_id)], eta_map[(robot_id, task_id)], eta_map[(robot_id, task_id)] + 10.0)
        for robot_id in robot_order
        for task_id in task_order
    }
    env = FakeHeteroEnv(
        robot_order=robot_order,
        task_order=task_order,
        robot_states=robot_states,
        task_specs=task_specs,
        legal_masks=legal_masks,
        eta_map=eta_map,
        predict_map=predict_map,
    )
    policy.reset_episode(env.current_scenario)
    _ = policy.select_action(env, None, 0, env._legal_action_mask(robot_order[0]), np.random.default_rng(0))
    diagnostics = policy.get_diagnostics()
    assert diagnostics["oversize_fallback_rate"] > 0.0
    assert diagnostics["auction_fallback_rate"] > 0.0


def test_sas_sequential_assignment_respects_pending_actions() -> None:
    env = FakeHeteroEnv(
        robot_order=["robot_a", "robot_b"],
        task_order=["task_1", "task_2"],
        robot_states={
            "robot_a": {"role": "carrier", "status": "idle", "speed_multiplier": 1.0, "service_multiplier": 1.0},
            "robot_b": {"role": "carrier", "status": "idle", "speed_multiplier": 1.0, "service_multiplier": 1.0},
        },
        task_specs={
            "task_1": {"kind": "single", "service_time": 10.0, "required_roles": {"carrier": 1}},
            "task_2": {"kind": "single", "service_time": 12.0, "required_roles": {"carrier": 1}},
        },
        legal_masks={
            "robot_a": [0.0, 1.0, 0.0],
            "robot_b": [0.0, 1.0, 1.0],
        },
        eta_map={
            ("robot_a", "task_1"): 1.0,
            ("robot_a", "task_2"): 999.0,
            ("robot_b", "task_1"): 1.5,
            ("robot_b", "task_2"): 2.0,
        },
        predict_map={
            ("robot_a", "task_1"): (1.0, 1.0, 11.0),
            ("robot_b", "task_1"): (1.5, 1.5, 11.5),
            ("robot_b", "task_2"): (2.0, 2.0, 14.0),
        },
    )
    policy = SAS(use_local_repair=False)
    policy.reset_episode(env.current_scenario)
    action_a = policy.select_action(env, None, 0, env._legal_action_mask("robot_a"), np.random.default_rng(0))
    action_b = policy.select_action(env, None, 1, env._legal_action_mask("robot_b"), np.random.default_rng(0))
    assert action_a == 1
    assert action_b == 2
    diagnostics = policy.get_diagnostics()
    assert diagnostics["planner_policy"] == "sas"


def test_ctasd_dependency_or_small_joint_assignment() -> None:
    try:
        policy = CTASD(max_active_robots=4, max_candidate_tasks=4)
    except RuntimeError as exc:
        assert "gurobipy" in str(exc)
        return

    env = FakeHeteroEnv(
        robot_order=["robot_a", "robot_b"],
        task_order=["task_1", "task_2"],
        robot_states={
            "robot_a": {"role": "carrier", "status": "idle", "speed_multiplier": 1.0, "service_multiplier": 1.0},
            "robot_b": {"role": "scout", "status": "idle", "speed_multiplier": 1.0, "service_multiplier": 1.0},
        },
        task_specs={
            "task_1": {"kind": "single", "service_time": 15.0, "required_roles": {"carrier": 1}, "priority": 1.0},
            "task_2": {"kind": "single", "service_time": 12.0, "required_roles": {"scout": 1}, "priority": 1.0},
        },
        legal_masks={
            "robot_a": [0.0, 1.0, 0.0],
            "robot_b": [0.0, 0.0, 1.0],
        },
        eta_map={
            ("robot_a", "task_1"): 3.0,
            ("robot_a", "task_2"): 999.0,
            ("robot_b", "task_1"): 999.0,
            ("robot_b", "task_2"): 2.0,
        },
        predict_map={
            ("robot_a", "task_1"): (3.0, 3.0, 18.0),
            ("robot_b", "task_2"): (2.0, 2.0, 14.0),
        },
    )
    policy.reset_episode(env.current_scenario)
    action_a = policy.select_action(env, None, 0, env._legal_action_mask("robot_a"), np.random.default_rng(0))
    action_b = policy.select_action(env, None, 1, env._legal_action_mask("robot_b"), np.random.default_rng(0))
    assert action_a == 1
    assert action_b == 2
    diagnostics = policy.get_diagnostics()
    assert diagnostics["planner_policy"] == "ctas_d"


def test_ctasd_oversize_falls_back_to_sas() -> None:
    try:
        policy = CTASD(max_active_robots=1, max_candidate_tasks=1)
    except RuntimeError:
        return

    env = FakeHeteroEnv(
        robot_order=["robot_a", "robot_b"],
        task_order=["task_1", "task_2"],
        robot_states={
            "robot_a": {"role": "carrier", "status": "idle", "speed_multiplier": 1.0, "service_multiplier": 1.0},
            "robot_b": {"role": "carrier", "status": "idle", "speed_multiplier": 1.0, "service_multiplier": 1.0},
        },
        task_specs={
            "task_1": {"kind": "single", "service_time": 10.0, "required_roles": {"carrier": 1}},
            "task_2": {"kind": "single", "service_time": 10.0, "required_roles": {"carrier": 1}},
        },
        legal_masks={
            "robot_a": [0.0, 1.0, 1.0],
            "robot_b": [0.0, 1.0, 1.0],
        },
        eta_map={
            ("robot_a", "task_1"): 1.0,
            ("robot_a", "task_2"): 3.0,
            ("robot_b", "task_1"): 2.0,
            ("robot_b", "task_2"): 1.0,
        },
        predict_map={
            ("robot_a", "task_1"): (1.0, 1.0, 11.0),
            ("robot_a", "task_2"): (3.0, 3.0, 13.0),
            ("robot_b", "task_1"): (2.0, 2.0, 12.0),
            ("robot_b", "task_2"): (1.0, 1.0, 11.0),
        },
    )
    policy.reset_episode(env.current_scenario)
    _ = policy.select_action(env, None, 0, env._legal_action_mask("robot_a"), np.random.default_rng(0))
    diagnostics = policy.get_diagnostics()
    assert diagnostics["oversize_fallback_rate"] > 0.0


if __name__ == "__main__":
    test_auction_chooses_only_legal_single_task()
    test_auction_waits_when_sync_coalition_is_infeasible()
    test_milp_dependency_or_small_joint_assignment()
    test_milp_oversize_falls_back_to_auction()
    test_sas_sequential_assignment_respects_pending_actions()
    test_ctasd_dependency_or_small_joint_assignment()
    test_ctasd_oversize_falls_back_to_sas()
    print("planner baseline tests passed")
