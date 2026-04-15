from __future__ import annotations

from typing import Dict

import numpy as np

from planner_baselines import (
    ACTIVE_DISPATCH_STATUSES,
    PlannerBaselinePolicy,
    action_from_task_id,
    candidate_task_ids,
    current_robot_id,
    enumerate_sync_coalitions,
    get_base_env,
    get_round_key,
    get_robot_order,
    get_robot_states,
    get_task_specs,
    legal_mask_for_robot,
    mean_ms,
    now_ms,
    predict_task_outcome,
    task_unlock_value,
)
from sas import SAS
from solver_backend_gurobi import binary_value, build_model, has_incumbent, is_budget_truncated, is_optimal, require_gurobi


class CTASD(PlannerBaselinePolicy):
    name = "ctas_d"

    def __init__(
        self,
        *,
        planner_timeout_ms: int = 2500,
        mip_gap: float = 0.01,
        max_active_robots: int = 8,
        max_candidate_tasks: int = 12,
        max_role_candidates: int = 3,
        max_sync_primitives_per_task: int = 24,
    ) -> None:
        super().__init__()
        require_gurobi(self.name)
        self.planner_timeout_ms = int(planner_timeout_ms)
        self.mip_gap = float(mip_gap)
        self.max_active_robots = int(max_active_robots)
        self.max_candidate_tasks = int(max_candidate_tasks)
        self.max_role_candidates = int(max_role_candidates)
        self.max_sync_primitives_per_task = int(max_sync_primitives_per_task)
        self._search_ms: list[float] = []
        self._primitive_counts: list[float] = []
        self._sync_primitive_counts: list[float] = []
        self._round_count = 0
        self._oversize_fallbacks = 0
        self._budget_truncations = 0
        self._sas_fallbacks = 0
        self._cached_round_key: tuple[str, int] | None = None
        self._cached_actions: Dict[str, int] = {}
        self._fallback_policy = SAS(use_local_repair=False)

    def reset_episode(self, scenario: Dict) -> None:
        super().reset_episode(scenario)
        self._cached_round_key = None
        self._cached_actions = {}
        self._fallback_policy.reset_episode(scenario)

    def _active_robots(self, env_like) -> list[str]:
        robot_states = get_robot_states(env_like)
        return [
            robot_id
            for robot_id in get_robot_order(env_like)
            if robot_states[robot_id].get("status") in ACTIVE_DISPATCH_STATUSES
            and np.any(np.asarray(legal_mask_for_robot(env_like, robot_id)[1:], dtype=np.float32) > 0.0)
        ]

    def _single_primitive(self, env_like, robot_id: str, task_id: str) -> Dict[str, object] | None:
        eta, _, predicted_finish = predict_task_outcome(env_like, robot_id, task_id)
        if not np.isfinite(predicted_finish):
            return None
        return {
            "task_id": task_id,
            "robot_ids": (robot_id,),
            "cost": float(predicted_finish) - 0.05 * task_unlock_value(env_like, task_id),
            "kind": "single",
        }

    def _sync_primitives(self, env_like, task_id: str, active_robot_ids: list[str]) -> list[Dict[str, object]]:
        primitives: list[Dict[str, object]] = []
        for primitive in enumerate_sync_coalitions(
            env_like,
            task_id,
            active_robot_ids,
            max_role_candidates=self.max_role_candidates,
            max_primitives=self.max_sync_primitives_per_task,
        ):
            primitives.append(
                {
                    "task_id": task_id,
                    "robot_ids": tuple(str(robot_id) for robot_id in primitive["robot_ids"]),
                    "cost": (
                        float(primitive["predicted_finish"])
                        + 0.8 * float(primitive["coalition_wait"])
                        + 0.1 * float(primitive["coalition_size"])
                        - 0.05 * task_unlock_value(env_like, task_id)
                    ),
                    "kind": "sync",
                }
            )
        return primitives

    def _enumerate_primitives(self, env_like, active_robot_ids: list[str]) -> list[Dict[str, object]]:
        task_specs = get_task_specs(env_like)
        primitives: list[Dict[str, object]] = []
        for task_id in candidate_task_ids(env_like, active_robot_ids):
            task = task_specs[task_id]
            action = action_from_task_id(task_id, env_like)
            if task["kind"] == "single":
                for robot_id in active_robot_ids:
                    if legal_mask_for_robot(env_like, robot_id)[action] <= 0.0:
                        continue
                    primitive = self._single_primitive(env_like, robot_id, task_id)
                    if primitive is not None:
                        primitives.append(primitive)
                continue
            primitives.extend(self._sync_primitives(env_like, task_id, active_robot_ids))
        return primitives

    def _fallback_actions(self, env_like) -> Dict[str, int]:
        self._sas_fallbacks += 1
        return self._fallback_policy._solve_round(env_like)

    def _solve_round(self, env_like) -> Dict[str, int]:
        self._round_count += 1
        active_robot_ids = self._active_robots(env_like)
        candidate_tasks = candidate_task_ids(env_like, active_robot_ids)
        if len(active_robot_ids) > self.max_active_robots or len(candidate_tasks) > self.max_candidate_tasks:
            self._oversize_fallbacks += 1
            return self._fallback_actions(env_like)
        if not active_robot_ids:
            return {}

        primitives = self._enumerate_primitives(env_like, active_robot_ids)
        self._primitive_counts.append(float(len(primitives)))
        self._sync_primitive_counts.append(float(sum(1 for primitive in primitives if primitive["kind"] == "sync")))
        if not primitives:
            return {robot_id: 0 for robot_id in active_robot_ids}

        start_ms = now_ms()
        gp, GRB, model = build_model(
            name="ctas_d",
            timeout_ms=self.planner_timeout_ms,
            mip_gap=self.mip_gap,
            output_flag=0,
        )
        y = {index: model.addVar(vtype=GRB.BINARY, name=f"y_{index}") for index in range(len(primitives))}
        wait = {robot_id: model.addVar(vtype=GRB.BINARY, name=f"wait_{robot_id}") for robot_id in active_robot_ids}
        model.update()

        max_time = float(getattr(get_base_env(env_like), "max_time", 2500.0))
        model.setObjective(
            gp.quicksum(float(primitives[index]["cost"]) * y[index] for index in range(len(primitives)))
            + gp.quicksum(max_time * wait[robot_id] for robot_id in active_robot_ids),
            GRB.MINIMIZE,
        )
        for robot_id in active_robot_ids:
            model.addConstr(
                gp.quicksum(y[index] for index, primitive in enumerate(primitives) if robot_id in primitive["robot_ids"])
                + wait[robot_id]
                == 1,
                name=f"assign_{robot_id}",
            )
        for task_id in candidate_tasks:
            model.addConstr(
                gp.quicksum(y[index] for index, primitive in enumerate(primitives) if primitive["task_id"] == task_id) <= 1,
                name=f"task_{task_id}",
            )
        model.optimize()
        self._search_ms.append(now_ms() - start_ms)

        if not has_incumbent(model, GRB):
            return self._fallback_actions(env_like)
        if not is_optimal(model, GRB) and is_budget_truncated(model, GRB):
            self._budget_truncations += 1

        actions = {robot_id: 0 for robot_id in active_robot_ids}
        for index, primitive in enumerate(primitives):
            if binary_value(y[index]) < 0.5:
                continue
            action = action_from_task_id(str(primitive["task_id"]), env_like)
            for robot_id in primitive["robot_ids"]:
                actions[str(robot_id)] = action
        return actions

    def select_action(
        self,
        env_like,
        obs: Dict[str, np.ndarray] | None,
        current_robot_index: int,
        legal_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        del obs, legal_mask, rng
        round_key = get_round_key(env_like)
        if round_key != self._cached_round_key:
            self._cached_round_key = round_key
            self._cached_actions = self._solve_round(env_like)
        robot_id = current_robot_id(env_like, current_robot_index)
        return int(self._cached_actions.get(robot_id, 0))

    def get_diagnostics(self) -> Dict[str, float]:
        diagnostics = super().get_diagnostics()
        diagnostics.update(
            {
                "planner_policy": self.name,
                "mean_search_time_ms": mean_ms(self._search_ms),
                "oversize_fallback_rate": float(self._oversize_fallbacks) / float(max(self._round_count, 1)),
                "budget_truncation_rate": float(self._budget_truncations) / float(max(self._round_count, 1)),
                "mean_primitive_count": mean_ms(self._primitive_counts),
                "mean_sync_primitive_count": mean_ms(self._sync_primitive_counts),
            }
        )
        return diagnostics
