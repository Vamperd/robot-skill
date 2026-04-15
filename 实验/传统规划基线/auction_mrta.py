from __future__ import annotations

from typing import Dict

import numpy as np

from planner_baselines import (
    PlannerBaselinePolicy,
    action_from_task_id,
    coalition_info_for_task,
    current_robot_id,
    get_task_specs,
    mean_ms,
    now_ms,
    predict_task_outcome,
    single_task_runtime,
    task_unlock_value,
)


class AuctionMRTA(PlannerBaselinePolicy):
    name = "auction_mrta"

    def __init__(self) -> None:
        super().__init__()
        self._bid_eval_ms: list[float] = []

    def reset_episode(self, scenario: Dict) -> None:
        super().reset_episode(scenario)

    def select_action(
        self,
        env_like,
        obs: Dict[str, np.ndarray] | None,
        current_robot_index: int,
        legal_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        del obs, rng
        start_ms = now_ms()
        robot_id = current_robot_id(env_like, current_robot_index)
        task_specs = get_task_specs(env_like)
        best_action = 0
        best_cost = float("inf")

        for action in np.flatnonzero(np.asarray(legal_mask, dtype=np.float32) > 0.0):
            action = int(action)
            if action == 0:
                continue
            task_id = env_like.task_order[action - 1]
            task = task_specs[task_id]
            if task["kind"] == "single":
                eta, _, predicted_finish = predict_task_outcome(env_like, robot_id, task_id)
                if not np.isfinite(predicted_finish):
                    continue
                expected_runtime = single_task_runtime(env_like, robot_id, task_id)
                cost = float(eta) + 0.5 * float(expected_runtime) - 0.05 * task_unlock_value(env_like, task_id)
            else:
                coalition = coalition_info_for_task(env_like, task_id, forced_robot_id=robot_id)
                if coalition is None:
                    continue
                cost = (
                    float(coalition["coalition_eta"])
                    + 0.75 * float(coalition["coalition_wait"])
                    + 0.25 * float(coalition["coalition_size"])
                    - 0.05 * task_unlock_value(env_like, task_id)
                )
            if cost < best_cost - 1e-6 or (abs(cost - best_cost) <= 1e-6 and action < best_action):
                best_cost = cost
                best_action = action

        self._bid_eval_ms.append(now_ms() - start_ms)
        return int(best_action)

    def get_diagnostics(self) -> Dict[str, float]:
        diagnostics = super().get_diagnostics()
        diagnostics.update(
            {
                "planner_policy": self.name,
                "mean_bid_eval_time_ms": mean_ms(self._bid_eval_ms),
            }
        )
        return diagnostics
