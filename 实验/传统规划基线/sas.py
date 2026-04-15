from __future__ import annotations

from typing import Dict

import numpy as np

from planner_baselines import (
    ACTIVE_DISPATCH_STATUSES,
    PlannerBaselinePolicy,
    action_from_task_id,
    current_robot_id,
    enumerate_sync_coalitions,
    get_base_env,
    get_round_key,
    get_task_specs,
    legal_mask_for_robot,
    mean_ms,
    now_ms,
    ordered_active_robot_ids,
    predict_task_outcome,
    single_task_runtime,
    task_unlock_value,
)
from solver_backend_gurobi import binary_value, build_model, gurobi_is_available, has_incumbent


class SAS(PlannerBaselinePolicy):
    name = "sas"

    def __init__(
        self,
        *,
        top_k_tasks_per_robot: int = 3,
        window_robots: int = 3,
        max_window_tasks: int = 6,
        local_timeout_ms: int = 150,
        mip_gap: float = 0.02,
        use_local_repair: bool = True,
    ) -> None:
        super().__init__()
        self.top_k_tasks_per_robot = int(top_k_tasks_per_robot)
        self.window_robots = int(window_robots)
        self.max_window_tasks = int(max_window_tasks)
        self.local_timeout_ms = int(local_timeout_ms)
        self.mip_gap = float(mip_gap)
        self.use_local_repair = bool(use_local_repair)
        self._round_eval_ms: list[float] = []
        self._decision_ms: list[float] = []
        self._local_repair_ms: list[float] = []
        self._round_count = 0
        self._local_repair_attempts = 0
        self._local_timeout_count = 0
        self._sync_candidate_total = 0
        self._decision_count = 0
        self._cached_round_key: tuple[str, int] | None = None
        self._cached_actions: Dict[str, int] = {}

    def reset_episode(self, scenario: Dict) -> None:
        super().reset_episode(scenario)
        self._cached_round_key = None
        self._cached_actions = {}

    def _score_single(self, env_like, robot_id: str, task_id: str) -> float:
        eta, _, predicted_finish = predict_task_outcome(env_like, robot_id, task_id)
        if not np.isfinite(predicted_finish):
            return float("inf")
        return float(eta) + 0.5 * float(single_task_runtime(env_like, robot_id, task_id)) - 0.05 * task_unlock_value(env_like, task_id)

    def _score_sync(self, env_like, task_id: str, primitive: Dict[str, object]) -> float:
        return (
            float(primitive["coalition_eta"])
            + 0.9 * float(primitive["coalition_wait"])
            + 0.2 * float(primitive["coalition_size"])
            - 0.05 * task_unlock_value(env_like, task_id)
        )

    def _candidate_actions_for_robot(self, env_like, robot_id: str, pending_actions: np.ndarray) -> list[Dict[str, object]]:
        legal_mask = np.asarray(legal_mask_for_robot(env_like, robot_id, pending_actions), dtype=np.float32)
        task_specs = get_task_specs(env_like)
        candidates: list[Dict[str, object]] = []
        for action in np.flatnonzero(legal_mask > 0.0):
            action = int(action)
            if action == 0:
                continue
            task_id = env_like.task_order[action - 1]
            task = task_specs[task_id]
            if task["kind"] == "single":
                score = self._score_single(env_like, robot_id, task_id)
                if not np.isfinite(score):
                    continue
                candidates.append(
                    {
                        "robot_ids": (robot_id,),
                        "task_id": task_id,
                        "action": action,
                        "kind": "single",
                        "score": float(score),
                    }
                )
                continue

            sync_primitives = [
                primitive
                for primitive in enumerate_sync_coalitions(
                    env_like,
                    task_id,
                    ordered_active_robot_ids(env_like, pending_actions),
                    pending_actions,
                    max_role_candidates=3,
                    max_primitives=24,
                )
                if robot_id in primitive["robot_ids"]
            ]
            self._sync_candidate_total += len(sync_primitives)
            for primitive in sync_primitives:
                candidates.append(
                    {
                        "robot_ids": tuple(str(item) for item in primitive["robot_ids"]),
                        "task_id": task_id,
                        "action": action,
                        "kind": "sync",
                        "score": self._score_sync(env_like, task_id, primitive),
                    }
                )

        candidates.sort(key=lambda item: (float(item["score"]), int(item["action"])))
        return candidates[: self.top_k_tasks_per_robot]

    def _local_repair(
        self,
        env_like,
        robot_order: list[str],
        start_index: int,
        pending_actions: np.ndarray,
    ) -> Dict[str, int] | None:
        if not self.use_local_repair or not gurobi_is_available():
            return None
        window_robot_ids = robot_order[start_index : start_index + self.window_robots]
        if not window_robot_ids:
            return None

        candidate_map = {
            robot_id: self._candidate_actions_for_robot(env_like, robot_id, pending_actions)
            for robot_id in window_robot_ids
        }
        candidate_tasks = sorted(
            {
                str(candidate["task_id"])
                for candidates in candidate_map.values()
                for candidate in candidates
            }
        )
        if not candidate_tasks:
            return None
        if len(candidate_tasks) > self.max_window_tasks:
            candidate_tasks = candidate_tasks[: self.max_window_tasks]
            for robot_id in window_robot_ids:
                candidate_map[robot_id] = [
                    candidate for candidate in candidate_map[robot_id] if str(candidate["task_id"]) in candidate_tasks
                ]

        primitives: list[Dict[str, object]] = []
        seen = set()
        for candidates in candidate_map.values():
            for candidate in candidates:
                task_id = str(candidate["task_id"])
                if task_id not in candidate_tasks:
                    continue
                robot_ids = tuple(str(robot_id) for robot_id in candidate["robot_ids"])
                if any(robot_id not in window_robot_ids for robot_id in robot_ids):
                    continue
                key = (task_id, robot_ids)
                if key in seen:
                    continue
                seen.add(key)
                primitives.append(
                    {
                        "task_id": task_id,
                        "robot_ids": robot_ids,
                        "cost": float(candidate["score"]),
                    }
                )

        if not primitives:
            return None

        self._local_repair_attempts += 1
        start_ms = now_ms()
        gp, GRB, model = build_model(
            name="sas_local_repair",
            timeout_ms=self.local_timeout_ms,
            mip_gap=self.mip_gap,
            output_flag=0,
        )
        y = {index: model.addVar(vtype=GRB.BINARY, name=f"y_{index}") for index in range(len(primitives))}
        wait = {robot_id: model.addVar(vtype=GRB.BINARY, name=f"wait_{robot_id}") for robot_id in window_robot_ids}
        model.update()

        max_time = float(getattr(get_base_env(env_like), "max_time", 2500.0))
        model.setObjective(
            gp.quicksum(float(primitives[index]["cost"]) * y[index] for index in range(len(primitives)))
            + gp.quicksum(max_time * wait[robot_id] for robot_id in window_robot_ids),
            GRB.MINIMIZE,
        )
        for robot_id in window_robot_ids:
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
        elapsed_ms = now_ms() - start_ms
        self._local_repair_ms.append(elapsed_ms)

        if not has_incumbent(model, GRB):
            self._local_timeout_count += 1
            return None

        actions = {robot_id: 0 for robot_id in window_robot_ids}
        for index, primitive in enumerate(primitives):
            if binary_value(y[index]) < 0.5:
                continue
            action = action_from_task_id(str(primitive["task_id"]), env_like)
            for robot_id in primitive["robot_ids"]:
                actions[str(robot_id)] = action
        return actions

    def _solve_round(self, env_like) -> Dict[str, int]:
        self._round_count += 1
        start_ms = now_ms()
        robot_order = ordered_active_robot_ids(env_like)
        actions = {robot_id: 0 for robot_id in robot_order}
        original_pending = np.array(
            getattr(env_like, "pending_actions", np.zeros(len(env_like.robot_order), dtype=np.int64)),
            copy=True,
        )
        pending_actions = np.array(original_pending, copy=True)

        try:
            env_like.pending_actions = pending_actions
            for index, robot_id in enumerate(robot_order):
                decision_start = now_ms()
                env_like.pending_actions = pending_actions
                suggested = self._local_repair(env_like, robot_order, index, pending_actions)
                action = None if suggested is None else int(suggested.get(robot_id, 0))
                if action is None or action <= 0:
                    candidates = self._candidate_actions_for_robot(env_like, robot_id, pending_actions)
                    action = int(candidates[0]["action"]) if candidates else 0
                actions[robot_id] = int(action)
                robot_slot = env_like.robot_order.index(robot_id)
                pending_actions[robot_slot] = int(action)
                env_like.pending_actions = pending_actions
                self._decision_ms.append(now_ms() - decision_start)
                self._decision_count += 1
        finally:
            env_like.pending_actions = original_pending

        self._round_eval_ms.append(now_ms() - start_ms)
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
                "mean_round_eval_time_ms": mean_ms(self._round_eval_ms),
                "mean_robot_decision_time_ms": mean_ms(self._decision_ms),
                "mean_local_repair_time_ms": mean_ms(self._local_repair_ms),
                "local_timeout_rate": float(self._local_timeout_count) / float(max(self._local_repair_attempts, 1)),
                "mean_sync_candidate_count": float(self._sync_candidate_total) / float(max(self._decision_count, 1)),
            }
        )
        return diagnostics
