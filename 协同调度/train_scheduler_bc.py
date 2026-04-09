from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attention_policy import AttentionSchedulerPolicy, load_scheduler_checkpoint, masked_logits, save_scheduler_checkpoint
from scheduler_training_utils import (
    SchedulerSupervisedDataset,
    collect_expert_samples,
    collate_supervised_batch,
    evaluate_policy_on_scenarios,
)
from scheduler_utils import load_split_scenarios


def obs_batch_to_torch(obs_batch, device: torch.device):
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in obs_batch.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用等待感知专家进行高层调度行为克隆预热。")
    parser.add_argument("--scenario-dir", default="offline_maps_v2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit-train-per-family", type=int, default=None)
    parser.add_argument("--limit-val-per-family", type=int, default=None)
    parser.add_argument("--save-dir", default="协同调度/checkpoints_improve_bc")
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_scenarios = load_split_scenarios(
        scenario_dir=args.scenario_dir,
        split=args.train_split,
        limit_per_family=args.limit_train_per_family,
    )
    val_scenarios = load_split_scenarios(
        scenario_dir=args.scenario_dir,
        split=args.val_split,
        limit_per_family=args.limit_val_per_family,
    )
    if not train_scenarios:
        raise ValueError("训练场景为空，请先生成 offline_maps_v2 数据集。")
    if not val_scenarios:
        raise ValueError("验证场景为空，请检查 val split。")

    print(f"收集专家轨迹: train={len(train_scenarios)} 个场景, val={len(val_scenarios)} 个场景")
    samples = collect_expert_samples(train_scenarios, expert_policy="wait_aware_role_greedy")
    dataset = SchedulerSupervisedDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_supervised_batch,
    )

    if args.init_model:
        model, checkpoint = load_scheduler_checkpoint(args.init_model, device=device)
        print(f"加载初始模型: {args.init_model}")
        metadata = checkpoint.get("metadata", {})
    else:
        model = AttentionSchedulerPolicy().to(device)
        metadata = {}
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = float("-inf")

    print(f"专家样本数: {len(dataset)}")
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        steps = 0

        for obs_batch_np, actions_np in loader:
            obs_batch = obs_batch_to_torch(obs_batch_np, device)
            actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)

            logits, _ = model(obs_batch)
            masked = masked_logits(logits, obs_batch["current_action_mask"])
            loss = F.cross_entropy(masked, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            steps += 1

        train_loss = running_loss / max(steps, 1)
        val_metrics = evaluate_policy_on_scenarios(val_scenarios, model, device=device, deterministic=True)
        score = (
            val_metrics["success_rate"] * 1000.0
            - val_metrics["mean_makespan"]
            - 0.5 * val_metrics["mean_wait_time"]
            - 0.75 * val_metrics["mean_avoidable_wait_time"]
        )

        checkpoint_metadata = {
            **metadata,
            "stage": "behavior_cloning",
            "epoch": epoch,
            "train_loss": train_loss,
            "expert_policy": "wait_aware_role_greedy",
            "val_metrics": val_metrics,
        }
        save_scheduler_checkpoint(
            save_dir / "latest_scheduler_bc.pt",
            model,
            optimizer=optimizer,
            metadata=checkpoint_metadata,
        )

        if score > best_score:
            best_score = score
            save_scheduler_checkpoint(
                save_dir / "best_scheduler_bc.pt",
                model,
                optimizer=optimizer,
                metadata=checkpoint_metadata,
            )

        print(
            f"[BC][epoch {epoch:02d}] "
            f"loss={train_loss:.4f} "
            f"success={val_metrics['success_rate']:.3f} "
            f"makespan={val_metrics['mean_makespan']:.2f} "
            f"wait={val_metrics['mean_wait_time']:.2f} "
            f"avoidable_wait={val_metrics['mean_avoidable_wait_time']:.2f} "
            f"reassign_rate={val_metrics['mean_productive_reassign_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
