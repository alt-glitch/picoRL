"""REINFORCE with a batch-mean baseline -- the simplest policy gradient for LLMs.

Core idea (if you've built agents but never done RL training):

    When your agent generates a response, some responses get high reward and some
    get low reward. REINFORCE says: "look at the tokens in each response, and
    nudge the model to make high-reward tokens more likely and low-reward tokens
    less likely." That nudge is proportional to the *advantage* -- how much better
    (or worse) this response was compared to a baseline.

The math (one line):

    loss = -mean(advantage_i * logprob_i)

    where advantage_i = reward_i - mean(all rewards in the batch)

The problem with this approach:

    The baseline is the average reward across the *entire batch*. But the batch
    mixes prompts of different difficulty. A mediocre answer to an easy question
    might score higher than a great answer to a hard question. Subtracting one
    global mean from all of them adds noise to the gradient signal.

    GRPO (see grpo.py) fixes this by grouping multiple completions *per prompt*
    and using per-group baselines.
"""
from __future__ import annotations

import torch

from algorithms.common import get_per_token_logprobs, tokenize_with_assistant_mask
from core.types import Rollout


def compute_reinforce_loss(
    rollouts: list[Rollout],
    model,
    tokenizer,
    *,
    device: torch.device,
    normalize_by_tokens: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    # Step 1: Compute the baseline -- the average reward across the whole batch.
    # Every rollout's advantage is measured relative to this single number.
    rewards = torch.tensor([sum(r.rewards) for r in rollouts], device=device)
    baseline = rewards.mean()

    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0
    total_logprob = 0.0

    # Step 2: Accumulate the policy gradient loss across all rollouts.
    # Loop one rollout at a time (batch size 1). Rollouts have variable lengths
    # so real batching would require padding. Not worth it at this scale --
    # gradient accumulation across the loop gives the same result.
    for rollout, reward in zip(rollouts, rewards, strict=False):
        # Step 2a: Tokenize and figure out which tokens the model wrote.
        input_ids, mask = tokenize_with_assistant_mask(
            tokenizer, rollout.messages, device=device
        )

        # Step 2b: Get the model's log-probability for each token it generated.
        # unsqueeze to [1, T] for model, [0] to unwrap back to [T-1]
        token_logprobs = get_per_token_logprobs(model, input_ids.unsqueeze(0))[0]
        # Shift the mask by one to match logprobs (see common.py for why).
        token_mask = mask[1:]

        # Keep only assistant tokens -- we don't train on the prompt.
        masked_logprobs = token_logprobs[token_mask]
        if masked_logprobs.numel() == 0:
            continue

        logprob_term = (
            masked_logprobs.mean() if normalize_by_tokens else masked_logprobs.sum()
        )

        # Step 2c: The REINFORCE gradient. If advantage > 0, this response was
        # better than average, so we *increase* the log-probs of its tokens
        # (the negative sign flips the gradient direction for the optimizer).
        advantage = reward - baseline
        total_loss = total_loss - advantage * logprob_term

        total_logprob += float(masked_logprobs.sum().item())
        total_tokens += int(token_mask.sum().item())

    # Average over rollouts so the loss scale doesn't depend on batch size.
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
