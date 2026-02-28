"""PPO (Proximal Policy Optimization) -- included for comparison, not recommended.

PPO solves a real problem: generating rollouts is expensive (you have to run
inference on the full model), so you want to squeeze multiple gradient updates
out of each batch of rollouts instead of throwing them away after one step.

But naive reuse is dangerous -- after one gradient update, the model has changed,
so the old rollouts no longer reflect the current policy. PPO adds two mechanisms
to make reuse safe:

1. **Clipping (trust region):** The ratio new_prob / old_prob measures how much
   the policy has changed for each token. PPO clips this ratio to [1-eps, 1+eps]
   so no single update can move the policy too far. Think of it as a speed limit
   on learning.

2. **Value head (learned baseline):** Instead of using a simple average reward as
   the baseline (like REINFORCE) or a per-group average (like GRPO), PPO trains
   a separate neural network head to *predict* the expected reward at each token
   position. This gives per-token advantages via GAE (Generalized Advantage
   Estimation), which can be more precise but adds a lot of complexity.

Why the LLM community moved to GRPO:

    PPO was designed for game-playing agents that take thousands of small actions.
    LLMs take one "action" (generate a full response) and get one reward. The
    per-token value head and GAE machinery add complexity without much benefit
    in this setting. GRPO gets comparable results with a fraction of the code by
    using the much simpler per-group baseline.

This file has three main functions:
    - prepare_ppo_rollouts: snapshot old log-probs + value estimates (frozen)
    - compute_ppo_policy_loss: clipped surrogate objective
    - compute_ppo_value_loss: value head regression loss
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from algorithms.common import encode_chat_messages, get_per_token_logprobs
from core.types import Rollout


@dataclass(slots=True)
class PPORolloutBatch:
    """Frozen snapshot of one rollout for PPO training.

    Created by prepare_ppo_rollouts, consumed by compute_ppo_policy_loss
    and compute_ppo_value_loss. Stores everything needed so the model
    can change between prepare and compute steps.
    """

    input_ids: torch.Tensor  # [T]
    assistant_mask: torch.Tensor  # [T] bool
    old_logprobs: torch.Tensor  # [num_assistant_tokens] (shifted)
    old_values: torch.Tensor  # [num_assistant_tokens] (shifted)
    advantages: torch.Tensor  # [num_assistant_tokens]
    returns: torch.Tensor  # [num_assistant_tokens]
    reward: float  # sum of episode rewards


def _infer_hidden_size(model: Any) -> int:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embed", "d_model", "model_dim"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    weight = getattr(embeddings, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[1])
    raise ValueError("Cannot infer hidden size from model")


def ensure_value_head(model):
    head = getattr(model, "value_head", None)
    if isinstance(head, torch.nn.Module):
        return head

    hidden_size = _infer_hidden_size(model)
    head = torch.nn.Linear(hidden_size, 1)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)

    # match the model's dtype/device so value head params train alongside the model
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), None)
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    head = head.to(device=device, dtype=dtype) if dtype is not None else head.to(device=device)

    model.value_head = head
    return head


def tokenize_with_assistant_ranges(
    tokenizer: Any, messages: list[dict[str, str]], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    input_ids = encode_chat_messages(tokenizer, messages).to(device)
    mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    assistant_ranges: list[tuple[int, int]] = []
    prev_len = 0
    for idx, message in enumerate(messages):
        ids = encode_chat_messages(tokenizer, messages[: idx + 1]).to(device)
        current_len = ids.shape[0]

        if message["role"] == "assistant":
            mask[prev_len:current_len] = True
            assistant_ranges.append((prev_len, current_len))
        prev_len = current_len

    return input_ids, mask, assistant_ranges


def _compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, *, gamma: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, device=rewards.device)
    next_value = torch.tensor(0.0, device=rewards.device)

    for t in range(
        rewards.shape[0] - 1, -1, -1
    ):  # we traverse in reverse order because the final token/reward helps in calculating the gae
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def prepare_ppo_rollouts(
    rollouts: list[Rollout],
    model: Any,
    tokenizer: Any,
    *,
    device: torch.device,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    normalize_advantages: bool = True,
) -> tuple[list[PPORolloutBatch], dict[str, float]]:
    value_head = ensure_value_head(model)
    batches: list[PPORolloutBatch] = []
    total_reward = 0.0
    total_logprob = 0.0
    total_tokens = 0
    used_rollouts = 0

    for rollout in rollouts:
        input_ids, mask, assistant_ranges = tokenize_with_assistant_ranges(
            tokenizer, rollout.messages, device=device
        )
        if not mask.any():
            continue

        with torch.no_grad():
            outputs = model(
                input_ids.unsqueeze(0), output_hidden_states=True, use_cache=False
            )
            logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1)
            hidden_states = outputs.hidden_states[-1]
            values = value_head(hidden_states).squeeze(-1)[
                0
            ]  # one scalar value estimate per token position

            target_ids = input_ids[1:]
            token_logprobs = (
                logprobs[0, :-1, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)
            )

        token_mask = mask[1:]
        if not token_mask.any():
            continue

        old_logprobs = token_logprobs[token_mask].float()
        old_values = values[:-1][token_mask].float()

        token_rewards = torch.zeros_like(token_logprobs)
        for reward, (_start, end) in zip(
            rollout.rewards, assistant_ranges, strict=False
        ):
            last_token = end - 1
            if last_token <= 0:
                continue

            token_rewards[last_token - 1] += float(reward)
        rewards_seq = token_rewards[token_mask].float()

        advantages, returns = _compute_gae(
            rewards_seq,
            old_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        batches.append(
            PPORolloutBatch(
                input_ids=input_ids.cpu(),
                assistant_mask=mask.cpu(),
                old_logprobs=old_logprobs.cpu(),
                old_values=old_values.cpu(),
                advantages=advantages.cpu(),
                returns=returns.cpu(),
                reward=float(sum(rollout.rewards)),
            )
        )
        used_rollouts += 1
        total_reward += float(sum(rollout.rewards))
        total_logprob += float(old_logprobs.sum().item())
        total_tokens += int(old_logprobs.numel())

    adv_mean = 0.0
    adv_std = 0.0

    if batches:
        all_advantages = torch.cat([batch.advantages for batch in batches], dim=0)
        adv_mean = float(all_advantages.mean().item())
        adv_std = float(all_advantages.std(unbiased=False).item())

        if normalize_advantages and all_advantages.numel() > 1 and adv_std > 1e-8:
            for batch in batches:
                batch.advantages = (batch.advantages - adv_mean) / (adv_std + 1e-8)

    metrics = {
        "avg_reward": total_reward / max(used_rollouts, 1),
        "avg_logprob": total_logprob / max(total_tokens, 1),
        "assistant_tokens": float(total_tokens),
        "rollouts_used": float(used_rollouts),
        "rollouts_skipped": float(len(rollouts) - used_rollouts),
        "adv_mean": adv_mean,
        "adv_std": adv_std,
    }
    return batches, metrics


def compute_ppo_policy_loss(
    batches: list[PPORolloutBatch],
    model: Any,
    *,
    device: torch.device,
    clip_range: float,
    entropy_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    total_loss = torch.tensor(0.0, device=device)
    entropy_term = torch.tensor(0.0, device=device) if entropy_coef != 0.0 else None
    total_entropy = 0.0
    total_kl = 0.0
    total_clip = 0.0
    total_logprob = 0.0
    total_tokens = 0

    for batch in batches:
        input_ids = batch.input_ids.to(device)
        mask = batch.assistant_mask.to(device)
        old_logprobs = batch.old_logprobs.to(device)
        advantages = batch.advantages.to(device)

        outputs = model(input_ids.unsqueeze(0), use_cache=False)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        target_ids = input_ids[1:]
        token_logprobs = (
            log_probs[0, :-1, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)
        )
        token_mask = mask[1:]
        new_logprobs = token_logprobs[token_mask]

        # This below should never really happen
        # if new_logprobs.numel() == 0:
        #     continue
        # if new_logprobs.numel() != old_logprobs.numel():
        #     raise ValueError(
        #         f"Mismatch between old and new logprobs (old={old_logprobs.numel()}, new={new_logprobs.numel()})."
        #     )
        # if new_logprobs.numel() != advantages.numel():
        #     raise ValueError(
        #         f"Mismatch between advantages and logprobs (adv={advantages.numel()}, logprobs={new_logprobs.numel()})."
        #     )

        # PPO loss calc
        logratio = new_logprobs - old_logprobs
        ratio = logratio.exp()
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        loss = -torch.min(unclipped, clipped)
        total_loss = total_loss + loss.sum()

        token_entropy = -(log_probs[0, :-1, :] * log_probs[0, :-1, :].exp()).sum(-1)
        entropy_tokens = token_entropy[token_mask]
        # .item() extracts scalar from tensor, float() ensures python float
        total_entropy += float(entropy_tokens.sum().item())

        if entropy_term is not None:
            entropy_term = entropy_term + entropy_tokens.sum()

        total_kl += float((old_logprobs - new_logprobs).sum().item())
        total_clip += float((ratio - 1.0).abs().gt(clip_range).sum().item())
        total_logprob += float(new_logprobs.sum().item())
        total_tokens += int(new_logprobs.numel())

    if total_tokens == 0:
        return torch.tensor(0.0, device=device), {
            "policy_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "avg_logprob": 0.0,
            "assistant_tokens": 0.0,
        }

    policy_loss = total_loss / total_tokens
    entropy = total_entropy / total_tokens
    if entropy_term is not None:
        entropy_tensor = entropy_term / total_tokens
        policy_loss = policy_loss - entropy_coef * entropy_tensor

    metrics = {
        "policy_loss": float(policy_loss.detach().item()),
        "entropy": float(entropy),
        "approx_kl": total_kl / total_tokens,
        "clip_fraction": total_clip / total_tokens,
        "avg_logprob": total_logprob / total_tokens,
        "assistant_tokens": total_tokens,
    }
    return policy_loss, metrics


def compute_ppo_value_loss(
    batches: list[PPORolloutBatch],
    model: Any,
    *,
    device: torch.device,
    clip_range_vf: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    value_head = ensure_value_head(model)
    total_loss = torch.tensor(0.0, device=device)
    total_value = 0.0
    total_clip = 0.0
    total_tokens = 0

    for batch in batches:
        input_ids = batch.input_ids.to(device)
        mask = batch.assistant_mask.to(device)
        old_values = batch.old_values.to(device)
        returns = batch.returns.to(device)

        outputs = model(
            input_ids.unsqueeze(0), output_hidden_states=True, use_cache=False
        )
        hidden_states = outputs.hidden_states[-1]
        values = value_head(hidden_states).squeeze(-1)[0]
        token_values = values[:-1][mask[1:]]
        # token_values is a flat tensor containing only the value predictions at assistant token positions

        if clip_range_vf is None:
            value_loss = 0.5 * (token_values - returns).pow(2)
        else:
            value_pred_clipped = old_values + (token_values - old_values).clamp(
                -clip_range_vf, clip_range_vf
            )
            value_loss = 0.5 * torch.max(
                (token_values - returns).pow(2), (value_pred_clipped - returns).pow(2)
            )
            total_clip += float(
                (token_values - old_values).abs().gt(clip_range_vf).sum().item()
            )

        total_loss = total_loss + value_loss.sum()
        total_value += float(token_values.sum().item())
        total_tokens += int(token_values.numel())

    if total_tokens == 0:
        return torch.tensor(0.0, device=device), {
            "value_loss": 0.0,
            "value_mean": 0.0,
            "value_clip_fraction": 0.0,
            "assistant_tokens": 0.0,
        }

    value_loss = total_loss / total_tokens
    metrics = {
        "value_loss": float(value_loss.detach().item()),
        "value_mean": total_value / total_tokens,
        "value_clip_fraction": total_clip / total_tokens
        if clip_range_vf is not None
        else 0.0,
        "assistant_tokens": total_tokens,
    }

    return value_loss, metrics
