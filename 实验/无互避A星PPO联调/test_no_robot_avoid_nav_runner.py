from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from no_robot_avoid_nav_runner import NoRobotAvoidNavRunner


class DummyLowLevelAdapter:
    def __init__(self) -> None:
        self.neighbors_seen = []

    def predict_action(self, robot_state, waypoint, obstacles, neighbors):
        self.neighbors_seen.append(list(neighbors))
        return np.asarray([1.0, 0.0], dtype=np.float32)


def _build_runner(low_level_adapter=None) -> NoRobotAvoidNavRunner:
    runner = NoRobotAvoidNavRunner(low_level_adapter=low_level_adapter)
    runner.scenario = {
        "scenario_id": "unit_test",
        "obstacles": [(120, 120, 60, 60)],
    }
    runner.task_specs = {
        "task_1": {
            "id": "task_1",
            "kind": "single",
            "pos": (220.0, 80.0),
            "service_time": 10.0,
            "required_roles": {},
        }
    }
    runner.task_states = {
        "task_1": {
            "onsite_robot_ids": set(),
        }
    }
    runner.robot_states = {
        "robot_1": {
            "id": "robot_1",
            "x": 40.0,
            "y": 80.0,
            "radius": 15.0,
            "status": "travel",
            "assigned_task": "task_1",
            "assigned_slot_pos": None,
            "position": (40.0, 80.0),
            "global_path": [],
            "lookahead_wp": (40.0, 80.0),
            "last_collision": 0.0,
            "position_history": deque([(40.0, 80.0)], maxlen=60),
            "task_progress": 0.0,
            "frames_since_replan": 999,
            "speed_multiplier": 1.0,
            "service_multiplier": 1.0,
            "travel_time": 0.0,
            "busy_time": 0.0,
            "idle_time": 0.0,
            "wait_time": 0.0,
            "wait_elapsed": 0.0,
            "blocked_count": 0,
            "location_type": "start",
            "location_id": None,
            "eta_remaining": 0.0,
            "last_action": np.zeros(2, dtype=np.float32),
        },
        "robot_2": {
            "id": "robot_2",
            "x": 55.0,
            "y": 80.0,
            "radius": 15.0,
            "status": "onsite",
            "assigned_task": "task_1",
            "assigned_slot_pos": None,
            "position": (55.0, 80.0),
            "global_path": [],
            "lookahead_wp": (55.0, 80.0),
            "last_collision": 0.0,
            "position_history": deque([(55.0, 80.0)], maxlen=60),
            "task_progress": 0.0,
            "frames_since_replan": 0,
            "speed_multiplier": 1.0,
            "service_multiplier": 1.0,
            "travel_time": 0.0,
            "busy_time": 0.0,
            "idle_time": 0.0,
            "wait_time": 0.0,
            "wait_elapsed": 0.0,
            "blocked_count": 0,
            "location_type": "task",
            "location_id": "task_1",
            "eta_remaining": 0.0,
            "last_action": np.zeros(2, dtype=np.float32),
        },
    }
    return runner


def test_neighbors_are_always_empty():
    runner = _build_runner()
    assert runner._neighbors("robot_1") == []


def test_collision_ignores_other_robots_but_hits_walls():
    runner = _build_runner()
    assert runner._check_collision(55.0, 80.0, "robot_1") is False
    assert runner._check_collision(130.0, 130.0, "robot_1") is True


def test_step_motion_passes_empty_neighbors_to_low_level_policy():
    adapter = DummyLowLevelAdapter()
    runner = _build_runner(low_level_adapter=adapter)
    runner._step_motion()
    assert adapter.neighbors_seen, "expected low-level policy to be queried"
    assert adapter.neighbors_seen[0] == []


if __name__ == "__main__":
    test_neighbors_are_always_empty()
    test_collision_ignores_other_robots_but_hits_walls()
    test_step_motion_passes_empty_neighbors_to_low_level_policy()
    print("no_robot_avoid_nav_runner tests passed")
