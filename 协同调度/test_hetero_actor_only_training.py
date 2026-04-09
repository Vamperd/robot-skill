from __future__ import annotations

import argparse

import numpy as np
import torch

from hetero_attention_policy import HeteroActorOnlyPolicy
from train_scheduler_hetero_actor_only import (
    FULL_FIDELITY_DEFAULTS,
    batched_log_prob_sums_for_rollouts,
    apply_profile_defaults,
    log_prob_sum_for_rollout,
    _should_run_agreement_eval,
    _should_run_full_eval,
)


def _make_obs(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    legal_positions = [
        np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    ]
    return {
        "agent_inputs": rng.normal(size=(3, 4)).astype(np.float32),
        "task_inputs": rng.normal(size=(5, 5)).astype(np.float32),
        "global_mask": legal_positions[seed % len(legal_positions)],
        "current_agent_index": np.asarray(seed % 3, dtype=np.int64),
    }


def _make_rollout(seed: int, actions: list[int]) -> dict:
    observations = [_make_obs(seed + step) for step in range(len(actions))]
    return {
        "observations": observations,
        "actions": actions,
    }


def test_batched_log_prob_matches_scalar() -> None:
    torch.manual_seed(0)
    model = HeteroActorOnlyPolicy(
        agent_input_dim=4,
        task_input_dim=5,
        embedding_dim=32,
        n_head=4,
        encoder_layers=1,
        decoder_layers=1,
    )
    device = torch.device("cpu")
    model.to(device)

    rollouts = [
        _make_rollout(0, [0, 1, 0]),
        _make_rollout(10, [2, 0]),
        {"observations": [], "actions": []},
        _make_rollout(20, [1]),
    ]

    scalar_results = [log_prob_sum_for_rollout(model, rollout, device) for rollout in rollouts]
    batched_results = batched_log_prob_sums_for_rollouts(model, rollouts, device)

    assert len(scalar_results) == len(batched_results)
    for (scalar_log_prob, scalar_entropy), (batched_log_prob, batched_entropy) in zip(scalar_results, batched_results):
        assert abs(float((scalar_log_prob - batched_log_prob).item())) <= 1e-5
        assert abs(float((scalar_entropy - batched_entropy).item())) <= 1e-5


def test_quick_trend_profile_preserves_explicit_overrides() -> None:
    args = argparse.Namespace(
        profile="quick_trend",
        total_updates=12,
        scenarios_per_update=FULL_FIDELITY_DEFAULTS["scenarios_per_update"],
        rollouts_per_scenario=2,
        eval_every=FULL_FIDELITY_DEFAULTS["eval_every"],
        min_updates_before_stop=FULL_FIDELITY_DEFAULTS["min_updates_before_stop"],
        early_stop_patience=FULL_FIDELITY_DEFAULTS["early_stop_patience"],
        lr=FULL_FIDELITY_DEFAULTS["lr"],
        weight_decay=FULL_FIDELITY_DEFAULTS["weight_decay"],
        max_grad_norm=FULL_FIDELITY_DEFAULTS["max_grad_norm"],
        sample_temperature=FULL_FIDELITY_DEFAULTS["sample_temperature"],
        small_margin=FULL_FIDELITY_DEFAULTS["small_margin"],
        bc_anchor_updates=FULL_FIDELITY_DEFAULTS["bc_anchor_updates"],
        bc_anchor_coef=FULL_FIDELITY_DEFAULTS["bc_anchor_coef"],
        anchor_batch_size=FULL_FIDELITY_DEFAULTS["anchor_batch_size"],
        expert_episodes_per_family=FULL_FIDELITY_DEFAULTS["expert_episodes_per_family"],
        trap_eval=True,
        allow_weak_init=False,
        quick_val_max_episodes=FULL_FIDELITY_DEFAULTS["quick_val_max_episodes"],
        quick_trap_max_episodes=FULL_FIDELITY_DEFAULTS["quick_trap_max_episodes"],
        full_eval_every=FULL_FIDELITY_DEFAULTS["full_eval_every"],
        agreement_eval_every=FULL_FIDELITY_DEFAULTS["agreement_eval_every"],
        full_eval_on_improve=FULL_FIDELITY_DEFAULTS["full_eval_on_improve"],
    )

    args = apply_profile_defaults(args)

    assert args.total_updates == 12
    assert args.rollouts_per_scenario == 2
    assert args.scenarios_per_update == 24
    assert args.eval_every == 8
    assert args.quick_val_max_episodes == 56
    assert args.full_eval_every == 16


def test_quick_and_full_eval_rules() -> None:
    args = argparse.Namespace(
        total_updates=96,
        full_eval_every=16,
        agreement_eval_every=16,
        full_eval_on_improve=True,
    )

    assert _should_run_full_eval(8, args, quick_improved=True)
    assert not _should_run_full_eval(8, args, quick_improved=False)
    assert _should_run_full_eval(16, args, quick_improved=False)
    assert _should_run_full_eval(96, args, quick_improved=False)

    assert not _should_run_agreement_eval(8, args, quick_improved=False, run_full_eval=False)
    assert _should_run_agreement_eval(8, args, quick_improved=True, run_full_eval=True)
    assert _should_run_agreement_eval(16, args, quick_improved=False, run_full_eval=True)
    assert _should_run_agreement_eval(96, args, quick_improved=False, run_full_eval=True)


if __name__ == "__main__":
    test_batched_log_prob_matches_scalar()
    test_quick_trend_profile_preserves_explicit_overrides()
    test_quick_and_full_eval_rules()
    print("All hetero actor-only training tests passed.")
