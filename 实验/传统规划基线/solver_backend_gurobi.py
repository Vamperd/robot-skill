from __future__ import annotations

from typing import Any, Tuple


def gurobi_is_available() -> bool:
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        return False
    return True


def require_gurobi(policy_name: str) -> Tuple[Any, Any]:
    try:
        import gurobipy as gp  # type: ignore
        from gurobipy import GRB  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            f"{policy_name} requires optional dependency 'gurobipy'. "
            "Please run this planner on the Linux host with a working Gurobi installation."
        ) from exc
    return gp, GRB


def build_model(
    *,
    name: str,
    timeout_ms: int,
    mip_gap: float,
    output_flag: int = 0,
    threads: int | None = None,
):
    gp, GRB = require_gurobi(name)
    model = gp.Model(name)
    model.Params.OutputFlag = int(output_flag)
    model.Params.TimeLimit = max(float(timeout_ms) / 1000.0, 0.01)
    model.Params.MIPGap = max(float(mip_gap), 0.0)
    if threads is not None:
        model.Params.Threads = int(threads)
    return gp, GRB, model


def has_incumbent(model, GRB) -> bool:
    del GRB
    return int(getattr(model, "SolCount", 0)) > 0


def is_optimal(model, GRB) -> bool:
    return int(getattr(model, "Status", -1)) == int(GRB.OPTIMAL)


def is_budget_truncated(model, GRB) -> bool:
    return int(getattr(model, "Status", -1)) in {
        int(GRB.TIME_LIMIT),
        int(getattr(GRB, "ITERATION_LIMIT", -999999)),
        int(getattr(GRB, "NODE_LIMIT", -999998)),
    }


def binary_value(var) -> float:
    if var is None:
        return 0.0
    value = getattr(var, "X", None)
    if value is None:
        return 0.0
    return float(value)
