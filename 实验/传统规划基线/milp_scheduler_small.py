from __future__ import annotations

from itertools import combinations
from typing import Dict

import numpy as np

from auction_mrta import AuctionMRTA
from planner_baselines import (
    ACTIVE_DISPATCH_STATUSES,
    PlannerBaselinePolicy,
    action_from_task_id,
    candidate_task_ids,
    coalition_info_for_task,
    current_robot_id,
    get_base_env,
    get_robot_order,
    get_robot_states,
    get_round_key,
    get_task_order,
    get_task_specs,
    legal_mask_for_robot,
    mean_ms,
    now_ms,
    predict_task_outcome,
    role_deficit_for_task,
    task_unlock_value,
)


class MilpSchedulerSmall(PlannerBaselinePolicy):
    name = "milp_scheduler_small"

    def __init__(
        self,
        *,
        planner_timeout_ms: int = 2000,
        max_active_robots: int = 4,
        max_candidate_tasks: int = 6,
    ) -> None:
        super().__init__()
        try:
            import pulp  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency controlled by user env
            raise RuntimeError(
                "milp_scheduler_small requires optional dependency 'PuLP + CBC'. "
                "Please install PuLP in the active environment before selecting this planner."
            ) from exc

        self.pulp = pulp
        self.planner_timeout_ms = int(planner_timeout_ms)
        self.max_active_robots = int(max_active_robots)
        self.max_candidate_tasks = int(max_candidate_tasks)
        self._solve_ms: list[float] = []
        self._round_count = 0
        self._timeout_fallbacks = 0
        self._oversize_fallbacks = 0
        self._auction_fallbacks = 0
        self._cached_round_key: tuple[str, int] | None = None
        self._cached_actions: Dict[str, int] = {}
        self._auction_policy = AuctionMRTA()

    def reset_episode(self, scenario: Dict) -> None:
        super().reset_episode(scenario)
        self._cached_round_key = None
        self._cached_actions = {}
        self._auction_policy.reset_episode(scenario)

    def _round_assignments_with_auction(self, env_like, rng: np.random.Generator) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        active_robot_ids = [
            robot_id
            for robot_id in get_robot_order(env_like)
            if get_robot_states(env_like)[robot_id].get("status") in ACTIVE_DISPATCH_STATUSES
        ]
        for robot_id in active_robot_ids:
            slot = get_robot_order(env_like).index(robot_id)
            legal_mask = legal_mask_for_robot(env_like, robot_id)
            if not np.any(np.asarray(legal_mask[1:], dtype=np.float32) > 0.0):
                actions[robot_id] = 0
                continue
            actions[robot_id] = int(self._auction_policy.select_action(env_like, None, slot, legal_mask, rng))
        return actions

    def _enumerate_choices(self, env_like, active_robot_ids: list[str]) -> list[Dict[str, object]]:
        task_specs = get_task_specs(env_like)
        task_order = get_task_order(env_like)
        choices: list[Dict[str, object]] = []
        legal_by_robot = {robot_id: legal_mask_for_robot(env_like, robot_id) for robot_id in active_robot_ids}

        for task_id in candidate_task_ids(env_like, active_robot_ids):
            task = task_specs[task_id]
            task_action = action_from_task_id(task_id, env_like)
            if task["kind"] == "single":
                for robot_id in active_robot_ids:
                    if legal_by_robot[robot_id][task_action] <= 0.0:
                        continue
                    _, _, predicted_finish = predict_task_outcome(env_like, robot_id, task_id)
                    if not np.isfinite(predicted_finish):
                        continue
                    cost = float(predicted_finish) - 1.5 * task_unlock_value(env_like, task_id)
                    choices.append(
                        {
                            "task_id": task_id,
                            "robot_ids": (robot_id,),
                            "cost": cost,
                        }
                    )
                continue

            deficit = role_deficit_for_task(env_like, task_id)
            deficit_size = int(sum(deficit.values()))
            if deficit_size <= 0:
                continue
            feasible_robot_ids = [
                robot_id for robot_id in active_robot_ids if legal_by_robot[robot_id][task_action] > 0.0
            ]
            if len(feasible_robot_ids) < deficit_size:
                continue
            for coalition in combinations(feasible_robot_ids, deficit_size):
                coalition_info = coalition_info_for_task(
                    env_like,
                    task_id,
                    forced_robot_id=coalition[0],
                    selected_new_robot_ids=coalition,
                )
                if coalition_info is None:
                    continue
                cost = (
                    float(coalition_info["predicted_finish"])
                    + 0.6 * float(coalition_info["coalition_wait"])
                    + 0.05 * float(coalition_info["coalition_size"])
                    - 1.5 * task_unlock_value(env_like, task_id)
                )
                choices.append(
                    {
                        "task_id": task_id,
                        "robot_ids": tuple(str(robot_id) for robot_id in coalition),
                        "cost": cost,
                    }
                )
        return choices

    def _solve_round(self, env_like, rng: np.random.Generator) -> Dict[str, int]:
        self._round_count += 1
        active_robot_ids = [
            robot_id
            for robot_id in get_robot_order(env_like)
            if get_robot_states(env_like)[robot_id].get("status") in ACTIVE_DISPATCH_STATUSES
            and np.any(np.asarray(legal_mask_for_robot(env_like, robot_id)[1:], dtype=np.float32) > 0.0)
        ]
        candidate_tasks = candidate_task_ids(env_like, active_robot_ids)

        if len(active_robot_ids) > self.max_active_robots or len(candidate_tasks) > self.max_candidate_tasks:
            self._oversize_fallbacks += 1
            self._auction_fallbacks += 1
            return self._round_assignments_with_auction(env_like, rng)

        choices = self._enumerate_choices(env_like, active_robot_ids)
        if not active_robot_ids or not choices:
            return {robot_id: 0 for robot_id in active_robot_ids}

        start_ms = now_ms()
        pulp = self.pulp
        problem = pulp.LpProblem("milp_scheduler_small", pulp.LpMinimize)
        choice_vars = {
            index: pulp.LpVariable(f"y_{index}", lowBound=0, upBound=1, cat="Binary")
            for index in range(len(choices))
        }
        wait_vars = {
            robot_id: pulp.LpVariable(f"wait_{robot_id}", lowBound=0, upBound=1, cat="Binary")
            for robot_id in active_robot_ids
        }

        max_time = float(getattr(get_base_env(env_like), "max_time", 2500.0))
        problem += (
            pulp.lpSum(float(choice["cost"]) * choice_vars[index] for index, choice in enumerate(choices))
            + pulp.lpSum(max_time * wait_vars[robot_id] for robot_id in active_robot_ids)
        )

        for robot_id in active_robot_ids:
            covering = [
                choice_vars[index]
                for index, choice in enumerate(choices)
                if robot_id in choice["robot_ids"]
            ]
            problem += pulp.lpSum(covering) + wait_vars[robot_id] == 1

        for task_id in candidate_tasks:
            related = [
                choice_vars[index]
                for index, choice in enumerate(choices)
                if choice["task_id"] == task_id
            ]
            if related:
                problem += pulp.lpSum(related) <= 1

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=max(self.planner_timeout_ms / 1000.0, 0.1))
        problem.solve(solver)
        self._solve_ms.append(now_ms() - start_ms)

        status = pulp.LpStatus.get(problem.status, "Unknown")
        if status != "Optimal":
            self._timeout_fallbacks += 1
            self._auction_fallbacks += 1
            return self._round_assignments_with_auction(env_like, rng)

        actions = {robot_id: 0 for robot_id in active_robot_ids}
        for index, choice in enumerate(choices):
            if choice_vars[index].value() is None or float(choice_vars[index].value()) < 0.5:
                continue
            action = action_from_task_id(str(choice["task_id"]), env_like)
            for robot_id in choice["robot_ids"]:
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
        del obs, legal_mask
        round_key = get_round_key(env_like)
        if round_key != self._cached_round_key:
            self._cached_round_key = round_key
            self._cached_actions = self._solve_round(env_like, rng)
        robot_id = current_robot_id(env_like, current_robot_index)
        return int(self._cached_actions.get(robot_id, 0))

    def get_diagnostics(self) -> Dict[str, float]:
        diagnostics = super().get_diagnostics()
        round_count = max(self._round_count, 1)
        diagnostics.update(
            {
                "planner_policy": self.name,
                "mean_solve_time_ms": mean_ms(self._solve_ms),
                "timeout_rate": float(self._timeout_fallbacks) / float(round_count),
                "oversize_fallback_rate": float(self._oversize_fallbacks) / float(round_count),
                "auction_fallback_rate": float(self._auction_fallbacks) / float(round_count),
            }
        )
        return diagnostics
