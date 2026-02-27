from __future__ import annotations

from dataclasses import dataclass

# Standard chat message format (same as OpenAI/HuggingFace)
Message = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": "..."}


@dataclass(frozen=True)
class Rollout:
    """A single completed episode (conversation + rewards).

    One rollout = one completion for one prompt. GRPO generates k rollouts
    per prompt, grouped by group_id.

    rewards is per-step: one float per assistant turn. Single-turn tasks
    just have [final_reward]. PPO needs per-turn rewards to place them
    at the right token positions; REINFORCE/GRPO just sum them.
    """

    messages: list[Message]  # full conversation history
    rewards: list[float]  # per-step rewards (one per assistant turn)
    group_id: int | None = None  # which prompt this came from (for GRPO grouping)
