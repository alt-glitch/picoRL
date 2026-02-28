"""GRPO (Group Relative Policy Optimization) -- smarter baselines, same simplicity.

GRPO uses the exact same loss as REINFORCE:

    loss = -mean(advantage_i * logprob_i)

The only difference is *how advantage is computed*. Instead of comparing each
response to the batch average (which mixes easy and hard prompts), GRPO groups
multiple completions for the *same* prompt and compares them to each other:

    advantage_i = reward_i - mean(rewards in my group)

This is the algorithm behind DeepSeek-R1 and most recent LLM RL work. It is
simpler than PPO (no value network, no clipping, no GAE) and works better for
LLMs because the per-group baseline gives a cleaner gradient signal.

Concrete example with k=4 completions per prompt:

    Prompt: "How many times does 'r' appear in 'strawberry'?"
    Group rewards: [0, 1, 0, 0]   (only completion #2 got it right)
    Group mean:    0.25
    Advantages:    [-0.25, +0.75, -0.25, -0.25]

    The correct completion gets a strong positive advantage (+0.75) and its
    tokens are reinforced. The wrong completions get mild negative advantages
    (-0.25) and their tokens are slightly suppressed. Compare this to a
    batch-mean REINFORCE baseline that might be 0.05 if most other prompts
    in the batch also scored low -- the signal would be much weaker.

Why "group" matters for sparse rewards:

    With binary reward (0 or 1) and ~5% accuracy, most k=8 groups will have
    all-zero rewards and therefore zero advantage. These groups contribute
    nothing to learning. The metric `groups_with_variance` tracks how many
    groups actually have a useful training signal. If this number is very low,
    you need more completions per prompt (larger k) or easier tasks.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from algorithms.common import get_per_token_logprobs, tokenize_with_assistant_mask
from core.types import Rollout


def compute_grpo_loss(
    rollouts: list[Rollout],
    model,
    tokenizer,
    *,
    device: torch.device,
    normalize_advantages: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GRPO loss with per-group baselines.

    Same as REINFORCE but advantage_i = reward_i - mean(group_rewards) instead
    of reward_i - mean(batch_rewards). No ratio clipping (we're on-policy with
    a single gradient step per rollout batch).
    """
    # Group rollouts by group_id
    groups: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rollouts):
        gid = r.group_id if r.group_id is not None else id(r)
        groups[gid].append(i)

    # Compute per-rollout advantages using group baselines
    rewards = torch.tensor([sum(r.rewards) for r in rollouts], device=device)
    advantages = torch.zeros_like(rewards)

    groups_total = len(groups)
    # groups_with_variance: groups where at least one completion scored
    # differently from the others. These are the groups that actually produce
    # a gradient signal. If all completions in a group got reward=0, the
    # advantage for every completion is 0 and the group contributes nothing.
    groups_with_variance = 0
    # groups_skipped: groups where all completions got the same reward (often
    # all zeros). High values here mean the model rarely gets the answer right,
    # so most groups are "wasted." Consider increasing k or lowering difficulty.
    groups_skipped = 0

    for indices in groups.values():
        group_rewards = rewards[indices]
        group_mean = group_rewards.mean()
        group_adv = group_rewards - group_mean

        if group_rewards.max() != group_rewards.min():
            groups_with_variance += 1
        else:
            groups_skipped += 1

        for idx, adv in zip(indices, group_adv):
            advantages[idx] = adv

    # Optional: normalize advantages across the full batch
    if normalize_advantages and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / advantages.std()

    # Compute policy gradient loss
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0
    total_logprob = 0.0

    for rollout, advantage in zip(rollouts, advantages):
        input_ids, mask = tokenize_with_assistant_mask(
            tokenizer, rollout.messages, device=device
        )

        token_logprobs = get_per_token_logprobs(model, input_ids.unsqueeze(0))[0]
        token_mask = mask[1:]  # shift mask to align with causal logprobs

        masked_logprobs = token_logprobs[token_mask]
        if masked_logprobs.numel() == 0:
            continue

        logprob_term = masked_logprobs.mean()
        total_loss = total_loss - advantage * logprob_term

        total_logprob += float(masked_logprobs.sum().item())
        total_tokens += int(token_mask.sum().item())

    loss = total_loss / max(len(rollouts), 1)

    metrics = {
        "avg_reward": float(rewards.mean().item()),
        "avg_logprob": total_logprob / max(total_tokens, 1),
        "assistant_tokens": float(total_tokens),
        "adv_mean": float(advantages.mean().item()),
        "adv_std": float(advantages.std().item()) if len(rollouts) > 1 else 0.0,
        "groups_total": float(groups_total),
        "groups_with_variance": float(groups_with_variance),
        "groups_skipped": float(groups_skipped),
    }
    return loss, metrics
