from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol

import torch
from torch.nn import functional as F
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from picorl.algorithms.utils import get_per_token_logprobs, tokenize_with_assistant_mask
from picorl.types import Rollout


def compute_reinforce_loss(
    rollouts: list[Rollout],
    model,
    tokenizer,
    *,
    device: torch.device,
    normalize_by_tokens: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = torch.tensor([sum(r.rewards) for r in rollouts], device=device)
    baseline = rewards.mean()

    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0
    total_logprob = 0.0

    # Loop one rollout at a time (batch size 1). Rollouts have variable lengths
    # so real batching would require padding. Not worth it at this scale —
    # gradient accumulation across the loop gives the same result.
    for rollout, reward in zip(rollouts, rewards, strict=False):
        input_ids, mask = tokenize_with_assistant_mask(
            tokenizer, rollout.messages, device=device
        )

        # unsqueeze to [1, T] for model, [0] to unwrap back to [T-1]
        token_logprobs = get_per_token_logprobs(model, input_ids.unsqueeze(0))[0]
        token_mask = mask[1:]  # shift mask to align with causal logprobs

        masked_logprobs = token_logprobs[token_mask]
        if masked_logprobs.numel() == 0:
            continue

        logprob_term = (
            masked_logprobs.mean() if normalize_by_tokens else masked_logprobs.sum()
        )

        advantage = reward - baseline
        total_loss = total_loss - advantage * logprob_term

        total_logprob += float(masked_logprobs.sum().item())
        total_tokens += int(token_mask.sum().item())

    loss = total_loss / max(len(rollouts), 1)

    advantages = rewards - baseline
    used_rollouts = len(rollouts)
    adv_mean = float(advantages.mean().item()) if used_rollouts else 0.0
    adv_std = float(advantages.std().item()) if used_rollouts > 1 else 0.0

    metrics = {
        "avg_reward": float(rewards.mean().item()) if used_rollouts else 0.0,
        "avg_logprob": total_logprob / max(total_tokens, 1),
        "assistant_tokens": float(total_tokens),
        "adv_mean": adv_mean,
        "adv_std": adv_std,
    }
    return loss, metrics
