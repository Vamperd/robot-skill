from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from scheduler_utils import ROLE_DIM, ROBOT_FEATURE_DIM, TASK_FEATURE_DIM


def _masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    invalid = action_mask <= 0.0
    return logits.masked_fill(invalid, -1e9)


def obs_to_torch(obs: Dict[str, np.ndarray], device: torch.device | str) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 2 and key not in {"robot_task_eta", "task_task_eta"}:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 2 and key in {"robot_task_eta", "task_task_eta"}:
            tensor = tensor.unsqueeze(0)
        tensors[key] = tensor
    return tensors


class AttentionSchedulerPolicy(nn.Module):
    def __init__(
        self,
        robot_dim: int = ROBOT_FEATURE_DIM,
        task_dim: int = TASK_FEATURE_DIM + ROLE_DIM + 1,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_tasks: int = 10,
    ):
        super().__init__()
        self.robot_dim = robot_dim
        self.task_dim = task_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_tasks = max_tasks

        self.robot_embed = nn.Linear(robot_dim, embed_dim)
        self.task_embed = nn.Linear(task_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
        )
        self.robot_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.task_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.task_to_robot = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.robot_to_task = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        self.current_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.pointer_query = nn.Linear(embed_dim, embed_dim)
        self.pointer_key = nn.Linear(embed_dim, embed_dim)
        self.wait_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.wait_head = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        nn.init.normal_(self.wait_token, std=0.02)

    def task_inputs_from_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        current_robot = obs["current_robot"]
        current_robot_eta = torch.einsum("br,brt->bt", current_robot, obs["robot_task_eta"]).unsqueeze(-1)
        return torch.cat(
            [
                obs["tasks"],
                obs["remaining_role_deficit"],
                current_robot_eta,
            ],
            dim=-1,
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        robot_valid = obs["robots"][..., 0] > 0.0
        task_valid = obs["tasks"][..., 0] > 0.0
        robot_pad = ~robot_valid
        task_pad = ~task_valid

        robot_tokens = self.robot_embed(obs["robots"])
        task_tokens = self.task_embed(self.task_inputs_from_obs(obs))
        robot_tokens = self.robot_encoder(robot_tokens, src_key_padding_mask=robot_pad)
        task_tokens = self.task_encoder(task_tokens, src_key_padding_mask=task_pad)

        task_context, _ = self.task_to_robot(
            query=task_tokens,
            key=robot_tokens,
            value=robot_tokens,
            key_padding_mask=robot_pad,
            need_weights=False,
        )
        robot_context, _ = self.robot_to_task(
            query=robot_tokens,
            key=task_tokens,
            value=task_tokens,
            key_padding_mask=task_pad,
            need_weights=False,
        )

        current_selector = obs["current_robot"].unsqueeze(-1)
        current_robot_token = (robot_context * current_selector).sum(dim=1)
        pooled_robot = _masked_mean(robot_context, robot_valid)
        pooled_task = _masked_mean(task_context, task_valid)
        fused = self.current_fusion(torch.cat([current_robot_token, pooled_robot, pooled_task], dim=-1))

        query = self.pointer_query(fused).unsqueeze(1)
        keys = self.pointer_key(task_context)
        task_logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.embed_dim)
        task_logits = task_logits.masked_fill(task_pad, -1e9)

        wait_context = self.wait_token.expand(fused.size(0), -1, -1).squeeze(1)
        wait_logits = self.wait_head(torch.cat([fused, pooled_robot, wait_context], dim=-1))
        logits = torch.cat([wait_logits, task_logits], dim=-1)

        value = self.value_head(torch.cat([fused, pooled_robot, pooled_task], dim=-1)).squeeze(-1)
        return logits, value

    def action_distribution(self, obs: Dict[str, torch.Tensor]) -> Tuple[Categorical, torch.Tensor]:
        logits, value = self.forward(obs)
        masked = masked_logits(logits, obs["current_action_mask"])
        return Categorical(logits=masked), value

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.action_distribution(obs)
        if deterministic:
            logits = dist.logits
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def get_config(self) -> Dict[str, int | float]:
        return {
            "robot_dim": self.robot_dim,
            "task_dim": self.task_dim,
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "max_tasks": self.max_tasks,
        }


def save_scheduler_checkpoint(
    path: str | Path,
    model: AttentionSchedulerPolicy,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: Dict | None = None,
) -> None:
    checkpoint = {
        "config": model.get_config(),
        "model_state": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    torch.save(checkpoint, Path(path))


def load_scheduler_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[AttentionSchedulerPolicy, Dict]:
    checkpoint = torch.load(Path(path), map_location=device)
    config = checkpoint["config"]
    model = AttentionSchedulerPolicy(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
