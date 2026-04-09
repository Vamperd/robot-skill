from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def get_attn_pad_mask(seq_q: torch.Tensor, seq_k: torch.Tensor) -> torch.Tensor:
    batch_size, len_q = seq_q.sum(dim=2).size()
    _, len_k = seq_k.sum(dim=2).size()
    pad_attn_mask_k = seq_q.eq(0).all(2).unsqueeze(1)
    pad_attn_mask_q = seq_k.eq(0).all(2).unsqueeze(1)
    pad_attn_mask_k = pad_attn_mask_k.expand(batch_size, len_k, len_q).permute(0, 2, 1)
    pad_attn_mask_q = pad_attn_mask_q.expand(batch_size, len_q, len_k)
    return ~torch.logical_and(~pad_attn_mask_k, ~pad_attn_mask_q)


def obs_to_torch(obs: Dict[str, np.ndarray], device: torch.device | str) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        tensors[key] = tensor
    if "current_agent_index" in tensors:
        tensors["current_agent_index"] = tensors["current_agent_index"].long()
    return tensors


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.input_dim = embedding_dim
        self.key_dim = embedding_dim
        self.tanh_clipping = 10.0
        self.norm_factor = 1.0 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.init_parameters()

    def init_parameters(self) -> None:
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(
        self,
        q: torch.Tensor,
        h: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)

        h_flat = h.reshape(-1, input_dim)
        q_flat = q.reshape(-1, input_dim)

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        query = torch.matmul(q_flat, self.w_query).view(shape_q)
        key = torch.matmul(h_flat, self.w_key).view(shape_k)

        logits = self.norm_factor * torch.matmul(query, key.transpose(1, 2))
        logits = self.tanh_clipping * torch.tanh(logits)

        if mask is not None:
            mask = mask.view(batch_size, -1, target_size).expand_as(logits)
            logits = logits.masked_fill(mask.bool(), -1e8)

        probs = torch.softmax(logits, dim=-1)
        logps = torch.log_softmax(logits, dim=-1)
        return probs, logps


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim // n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1.0 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(n_heads, self.value_dim, self.embedding_dim))
        self.init_parameters()

    def init_parameters(self) -> None:
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(
        self,
        q: torch.Tensor,
        h: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)

        h_flat = h.contiguous().view(-1, input_dim)
        q_flat = q.contiguous().view(-1, input_dim)

        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        query = torch.matmul(q_flat, self.w_query).view(shape_q)
        key = torch.matmul(h_flat, self.w_key).view(shape_k)
        value = torch.matmul(h_flat, self.w_value).view(shape_v)

        logits = self.norm_factor * torch.matmul(query, key.transpose(2, 3))
        if mask is not None:
            expanded_mask = mask.view(1, batch_size, -1, target_size).expand_as(logits)
            logits = logits.masked_fill(expanded_mask.bool(), -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        if mask is not None:
            attention = attention.masked_fill(expanded_mask.bool(), 0.0)

        heads = torch.matmul(attention, value)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            self.w_out.view(-1, self.embedding_dim),
        ).view(batch_size, n_query, self.embedding_dim)
        return out


class GateFFNDense(nn.Module):
    def __init__(self, model_dim: int, hidden_unit: int = 512):
        super().__init__()
        self.w = nn.Linear(model_dim, hidden_unit, bias=False)
        self.v = nn.Linear(model_dim, hidden_unit, bias=False)
        self.w2 = nn.Linear(hidden_unit, model_dim, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_act = self.act(self.w(hidden_states))
        hidden_linear = self.v(hidden_states)
        hidden_states = hidden_act * hidden_linear
        return self.w2(hidden_states)


class Normalization(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.normalizer(input_tensor.view(-1, input_tensor.size(-1))).view(*input_tensor.size())


class GateFFNLayer(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.dense = GateFFNDense(model_dim)
        self.layer_norm = Normalization(model_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(self.layer_norm(hidden_states))


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, n_head: int):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization = Normalization(embedding_dim)
        self.feed_forward = GateFFNLayer(embedding_dim)

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = src
        hidden = self.normalization(src)
        hidden = self.multi_head_attention(q=hidden, mask=mask)
        hidden = hidden + residual
        residual = hidden
        hidden = self.feed_forward(hidden)
        return hidden + residual


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, n_head: int):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, n_head)
        self.feed_forward = GateFFNLayer(embedding_dim)
        self.normalization1 = Normalization(embedding_dim)
        self.normalization2 = Normalization(embedding_dim)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        dec_self_attn_mask: torch.Tensor | None,
        dec_enc_attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        del dec_self_attn_mask
        residual = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization2(memory)
        hidden = self.multi_head_attention(q=tgt, h=memory, mask=dec_enc_attn_mask)
        hidden = hidden + residual
        residual = hidden
        hidden = self.feed_forward(hidden)
        return hidden + residual


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, n_head: int = 8, n_layer: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for _ in range(n_layer))

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, n_head: int = 8, n_layer: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(embedding_dim, n_head) for _ in range(n_layer))

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        dec_self_attn_mask: torch.Tensor | None = None,
        dec_enc_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory, dec_self_attn_mask, dec_enc_attn_mask)
        return tgt


class HeteroAttentionActor(nn.Module):
    def __init__(
        self,
        agent_input_dim: int,
        task_input_dim: int,
        embedding_dim: int = 128,
        n_head: int = 8,
        encoder_layers: int = 1,
        decoder_layers: int = 2,
    ):
        super().__init__()
        self.agent_input_dim = agent_input_dim
        self.task_input_dim = task_input_dim
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.agent_embedding = nn.Linear(agent_input_dim, embedding_dim)
        self.task_embedding = nn.Linear(task_input_dim, embedding_dim)
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

        self.task_encoder = Encoder(embedding_dim=embedding_dim, n_head=n_head, n_layer=encoder_layers)
        self.agent_encoder = Encoder(embedding_dim=embedding_dim, n_head=n_head, n_layer=encoder_layers)
        self.cross_decoder_1 = Decoder(embedding_dim=embedding_dim, n_head=n_head, n_layer=decoder_layers)
        self.cross_decoder_2 = Decoder(embedding_dim=embedding_dim, n_head=n_head, n_layer=decoder_layers)
        self.global_decoder = Decoder(embedding_dim=embedding_dim, n_head=n_head, n_layer=decoder_layers)
        self.pointer = SingleHeadAttention(embedding_dim)

    def encoding_tasks(self, task_inputs: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        task_embedding = self.task_embedding(task_inputs)
        task_encoding = self.task_encoder(task_embedding, mask)
        embedding_dim = task_encoding.size(-1)
        mean_mask = mask[:, 0, :].unsqueeze(2).repeat(1, 1, embedding_dim)
        compressed_task = torch.where(mean_mask, torch.nan, task_embedding)
        aggregated_tasks = torch.nanmean(compressed_task, dim=1).unsqueeze(1)
        aggregated_tasks = torch.nan_to_num(aggregated_tasks, nan=0.0)
        return aggregated_tasks, task_encoding

    def encoding_agents(self, agent_inputs: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        agent_embedding = self.agent_embedding(agent_inputs)
        agent_encoding = self.agent_encoder(agent_embedding, mask)
        embedding_dim = agent_encoding.size(-1)
        mean_mask = mask[:, 0, :].unsqueeze(2).repeat(1, 1, embedding_dim)
        compressed_agents = torch.where(mean_mask, torch.nan, agent_embedding)
        aggregated_agents = torch.nanmean(compressed_agents, dim=1).unsqueeze(1)
        aggregated_agents = torch.nan_to_num(aggregated_agents, nan=0.0)
        return aggregated_agents, agent_encoding

    def encode_context(
        self,
        tasks: torch.Tensor,
        agents: torch.Tensor,
        global_mask: torch.Tensor,
        current_agent_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if global_mask.ndim == 2:
            global_mask = global_mask.unsqueeze(1)
        if current_agent_index.ndim == 1:
            current_agent_index = current_agent_index.view(-1, 1, 1)
        elif current_agent_index.ndim == 2:
            current_agent_index = current_agent_index.unsqueeze(-1)

        task_mask = get_attn_pad_mask(tasks, tasks)
        agent_mask = get_attn_pad_mask(agents, agents)
        task_agent_mask = get_attn_pad_mask(tasks, agents)
        agent_task_mask = get_attn_pad_mask(agents, tasks)

        aggregated_task, task_encoding = self.encoding_tasks(tasks, mask=task_mask)
        aggregated_agents, agents_encoding = self.encoding_agents(agents, mask=agent_mask)
        task_agent_feature = self.cross_decoder_1(task_encoding, agents_encoding, None, task_agent_mask)
        agent_task_feature = self.cross_decoder_2(agents_encoding, task_encoding, None, agent_task_mask)
        current_state = torch.gather(
            agent_task_feature,
            1,
            current_agent_index.repeat(1, 1, agent_task_feature.size(2)),
        )
        current_state = self.fusion(torch.cat((current_state, aggregated_task, aggregated_agents), dim=-1))
        current_state_prime = self.global_decoder(current_state, task_agent_feature, None, global_mask)
        return {
            "task_agent_feature": task_agent_feature,
            "current_state_prime": current_state_prime.squeeze(1),
            "aggregated_task": aggregated_task.squeeze(1),
            "aggregated_agents": aggregated_agents.squeeze(1),
        }

    def forward(
        self,
        tasks: torch.Tensor,
        agents: torch.Tensor,
        global_mask: torch.Tensor,
        current_agent_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_context(
            tasks=tasks,
            agents=agents,
            global_mask=global_mask,
            current_agent_index=current_agent_index,
        )
        task_agent_feature = context["task_agent_feature"]
        current_state_prime = context["current_state_prime"].unsqueeze(1)
        probs, logps = self.pointer(current_state_prime, task_agent_feature, mask=global_mask)
        return probs.squeeze(1), logps.squeeze(1), context["current_state_prime"]


class HeteroAttentionSchedulerPolicy(nn.Module):
    def __init__(
        self,
        agent_input_dim: int,
        task_input_dim: int,
        embedding_dim: int = 128,
        n_head: int = 8,
        encoder_layers: int = 1,
        decoder_layers: int = 2,
    ):
        super().__init__()
        self.agent_input_dim = agent_input_dim
        self.task_input_dim = task_input_dim
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.actor = HeteroAttentionActor(
            agent_input_dim=agent_input_dim,
            task_input_dim=task_input_dim,
            embedding_dim=embedding_dim,
            n_head=n_head,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, logps, state = self.actor(
            tasks=obs["task_inputs"],
            agents=obs["agent_inputs"],
            global_mask=obs["global_mask"],
            current_agent_index=obs["current_agent_index"],
        )
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        value = self.value_head(state).squeeze(-1)
        return probs, logps, value

    def action_distribution(self, obs: Dict[str, torch.Tensor]) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        probs, logps, value = self.forward(obs)
        dist = Categorical(logits=logps)
        return dist, value, logps

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, _ = self.action_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, _ = self.action_distribution(obs)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def get_config(self) -> Dict[str, int]:
        return {
            "agent_input_dim": self.agent_input_dim,
            "task_input_dim": self.task_input_dim,
            "embedding_dim": self.embedding_dim,
            "n_head": self.n_head,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
        }


class HeteroActorOnlyPolicy(nn.Module):
    def __init__(
        self,
        agent_input_dim: int,
        task_input_dim: int,
        embedding_dim: int = 128,
        n_head: int = 8,
        encoder_layers: int = 1,
        decoder_layers: int = 2,
    ):
        super().__init__()
        self.agent_input_dim = agent_input_dim
        self.task_input_dim = task_input_dim
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.actor = HeteroAttentionActor(
            agent_input_dim=agent_input_dim,
            task_input_dim=task_input_dim,
            embedding_dim=embedding_dim,
            n_head=n_head,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        probs, logps, _ = self.actor(
            tasks=obs["task_inputs"],
            agents=obs["agent_inputs"],
            global_mask=obs["global_mask"],
            current_agent_index=obs["current_agent_index"],
        )
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs, logps

    def action_distribution(self, obs: Dict[str, torch.Tensor]) -> tuple[Categorical, torch.Tensor]:
        probs, logps = self.forward(obs)
        dist = Categorical(logits=logps)
        return dist, logps

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist, _ = self.action_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, logps = self.action_distribution(obs)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, logps

    def get_config(self) -> Dict[str, int]:
        return {
            "agent_input_dim": self.agent_input_dim,
            "task_input_dim": self.task_input_dim,
            "embedding_dim": self.embedding_dim,
            "n_head": self.n_head,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
        }


class HeteroRankerPolicy(nn.Module):
    def __init__(
        self,
        agent_input_dim: int,
        task_input_dim: int,
        embedding_dim: int = 128,
        n_head: int = 8,
        encoder_layers: int = 1,
        decoder_layers: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.agent_input_dim = agent_input_dim
        self.task_input_dim = task_input_dim
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.hidden_dim = hidden_dim

        self.encoder = HeteroAttentionActor(
            agent_input_dim=agent_input_dim,
            task_input_dim=task_input_dim,
            embedding_dim=embedding_dim,
            n_head=n_head,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )
        score_input_dim = embedding_dim * 6 + agent_input_dim + task_input_dim
        self.action_scorer = nn.Sequential(
            nn.Linear(score_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        tasks = obs["task_inputs"]
        agents = obs["agent_inputs"]
        global_mask = obs["global_mask"]
        current_agent_index = obs["current_agent_index"]
        context = self.encoder.encode_context(
            tasks=tasks,
            agents=agents,
            global_mask=global_mask,
            current_agent_index=current_agent_index,
        )
        current_state = context["current_state_prime"]
        task_feature = context["task_agent_feature"]
        aggregated_task = context["aggregated_task"]
        aggregated_agents = context["aggregated_agents"]

        if current_agent_index.ndim == 1:
            current_agent_index = current_agent_index.unsqueeze(-1)
        current_agent_index = current_agent_index.long().clamp_min(0)
        gathered_index = current_agent_index.unsqueeze(-1).expand(-1, 1, agents.size(-1))
        current_agent_inputs = torch.gather(agents, 1, gathered_index).squeeze(1)

        expanded_current_state = current_state.unsqueeze(1).expand(-1, task_feature.size(1), -1)
        expanded_aggregated_task = aggregated_task.unsqueeze(1).expand_as(task_feature)
        expanded_aggregated_agents = aggregated_agents.unsqueeze(1).expand_as(task_feature)
        expanded_current_agent_inputs = current_agent_inputs.unsqueeze(1).expand(-1, task_feature.size(1), -1)

        score_inputs = torch.cat(
            [
                expanded_current_state,
                task_feature,
                expanded_aggregated_task,
                expanded_aggregated_agents,
                expanded_current_state * task_feature,
                expanded_current_state - task_feature,
                expanded_current_agent_inputs,
                tasks,
            ],
            dim=-1,
        )
        logits = self.action_scorer(score_inputs).squeeze(-1)
        logits = logits.masked_fill(global_mask >= 0.5, -1e8)
        probs = torch.softmax(logits, dim=-1)
        logps = torch.log_softmax(logits, dim=-1)
        return probs, logps

    def action_distribution(self, obs: Dict[str, torch.Tensor]) -> tuple[Categorical, torch.Tensor]:
        probs, logps = self.forward(obs)
        dist = Categorical(logits=logps)
        return dist, logps

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist, _ = self.action_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, logps = self.action_distribution(obs)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, logps

    def get_config(self) -> Dict[str, int]:
        return {
            "agent_input_dim": self.agent_input_dim,
            "task_input_dim": self.task_input_dim,
            "embedding_dim": self.embedding_dim,
            "n_head": self.n_head,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "hidden_dim": self.hidden_dim,
        }


def save_hetero_scheduler_checkpoint(
    path: str | Path,
    model: HeteroAttentionSchedulerPolicy,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: Dict | None = None,
) -> None:
    output_path = Path(path).expanduser().resolve(strict=False)
    checkpoint = {
        "policy_type": "hetero_ppo",
        "config": model.get_config(),
        "model_state": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(output_path))


def load_hetero_scheduler_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[HeteroAttentionSchedulerPolicy, Dict]:
    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    config = _common_hetero_config(checkpoint["config"])
    model = HeteroAttentionSchedulerPolicy(**config)
    policy_type = checkpoint.get("policy_type") or checkpoint.get("metadata", {}).get("policy_type")
    if policy_type in {None, "hetero_ppo"}:
        model.load_state_dict(checkpoint["model_state"])
    else:
        actor_state = _extract_actor_like_state(checkpoint["model_state"], policy_type)
        missing, unexpected = model.load_state_dict(actor_state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys when warm-starting hetero scheduler: {unexpected}")
        non_value_missing = [key for key in missing if not key.startswith("value_head.")]
        if non_value_missing:
            raise ValueError(f"Missing non-value weights when warm-starting hetero scheduler: {non_value_missing}")
    model.to(device)
    model.eval()
    return model, checkpoint


def save_hetero_actor_only_checkpoint(
    path: str | Path,
    model: HeteroActorOnlyPolicy,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: Dict | None = None,
) -> None:
    output_path = Path(path).expanduser().resolve(strict=False)
    checkpoint = {
        "policy_type": "hetero_actor_only",
        "config": model.get_config(),
        "model_state": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(output_path))


def save_hetero_ranker_checkpoint(
    path: str | Path,
    model: HeteroRankerPolicy,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: Dict | None = None,
) -> None:
    output_path = Path(path).expanduser().resolve(strict=False)
    checkpoint = {
        "policy_type": "hetero_ranker",
        "config": model.get_config(),
        "model_state": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(output_path))


def _extract_actor_state(model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    actor_state = {key: value for key, value in model_state.items() if key.startswith("actor.")}
    if not actor_state:
        raise ValueError("Checkpoint does not contain actor weights.")
    return actor_state


def _common_hetero_config(config: Dict[str, int]) -> Dict[str, int]:
    return {
        "agent_input_dim": config["agent_input_dim"],
        "task_input_dim": config["task_input_dim"],
        "embedding_dim": config.get("embedding_dim", 128),
        "n_head": config.get("n_head", 8),
        "encoder_layers": config.get("encoder_layers", 1),
        "decoder_layers": config.get("decoder_layers", 2),
    }


def _extract_actor_like_state(
    model_state: Dict[str, torch.Tensor],
    policy_type: str | None,
) -> Dict[str, torch.Tensor]:
    if policy_type == "hetero_ranker":
        encoder_state = {key: value for key, value in model_state.items() if key.startswith("encoder.")}
        if not encoder_state:
            raise ValueError("Checkpoint does not contain encoder weights for hetero_ranker warm-start.")
        return {key.replace("encoder.", "actor.", 1): value for key, value in encoder_state.items()}
    return _extract_actor_state(model_state)


def load_hetero_actor_only_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[HeteroActorOnlyPolicy, Dict]:
    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    config = _common_hetero_config(checkpoint["config"])
    model = HeteroActorOnlyPolicy(**config)
    policy_type = checkpoint.get("policy_type") or checkpoint.get("metadata", {}).get("policy_type")
    if policy_type == "hetero_actor_only":
        model.load_state_dict(checkpoint["model_state"])
    else:
        actor_state = _extract_actor_like_state(checkpoint["model_state"], policy_type)
        model.load_state_dict(actor_state, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def load_hetero_ranker_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[HeteroRankerPolicy, Dict]:
    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    config = checkpoint["config"]
    model = HeteroRankerPolicy(**config)
    policy_type = checkpoint.get("policy_type") or checkpoint.get("metadata", {}).get("policy_type")
    if policy_type == "hetero_ranker":
        model.load_state_dict(checkpoint["model_state"])
    else:
        actor_state = _extract_actor_state(checkpoint["model_state"])
        encoder_state = {key.replace("actor.", "encoder.", 1): value for key, value in actor_state.items()}
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys when warm-starting hetero_ranker: {unexpected}")
        if not any(key.startswith("encoder.") for key in encoder_state):
            raise ValueError("Checkpoint does not contain compatible encoder weights for hetero_ranker.")
    model.to(device)
    model.eval()
    return model, checkpoint


def load_hetero_policy_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, Dict]:
    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    policy_type = checkpoint.get("policy_type") or checkpoint.get("metadata", {}).get("policy_type") or "hetero_ppo"
    if policy_type == "hetero_actor_only":
        return load_hetero_actor_only_checkpoint(checkpoint_path, device=device)
    if policy_type == "hetero_ranker":
        return load_hetero_ranker_checkpoint(checkpoint_path, device=device)
    return load_hetero_scheduler_checkpoint(checkpoint_path, device=device)
