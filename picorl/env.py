from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Callable

from picorl.types import Message, Rollout

# generate_fn: takes rendered prompts, returns completions
GenerateFn = Callable[[list[str]], list[str]]


class Env(ABC):
    """Minimal environment interface for RL training.

    Two methods:
    - reset()  -> start a new episode, return initial messages
    - step()   -> given model response, return (next_msg, reward, done)
    """

    @abstractmethod
    def reset(self) -> list[Message]:
        """Start a new episode.

        Returns the opening messages, typically [system_msg, first_user_msg].
        """
        ...

    @abstractmethod
    def step(self, response: str) -> tuple[Message | None, float, bool]:
        """Process the model's response.

        Args:
            response: The model's generated text.

        Returns:
            (next_message, reward, done) where next_message is the next user/env
            message (or None if done), reward is the reward for this step, and
            done indicates if the episode has ended.
        """
        ...


class BatchedEnv:
    """Manages multiple Env instances for parallel rollout collection.

    Constructor takes env_fns (one factory per task). For GRPO, call the same
    factory k times to produce k envs for the same task sharing a group_id.
    """

    def __init__(self, env_fns: Sequence[Callable[[], Env]]) -> None:
        self.env_fns = env_fns

    def collect_rollouts(
        self,
        generate_fn: GenerateFn,
        tokenizer,
        *,
        k: int = 1,
    ) -> list[Rollout]:
        """Run all envs to completion and return rollouts.

        Args:
            generate_fn: Takes list of rendered prompt strings, returns list of completions.
            tokenizer: HF tokenizer (used for apply_chat_template to render prompts).
            k: Number of rollouts per env factory (for GRPO grouping).

        Returns:
            List of completed Rollouts with group_ids set.
        """
        # 1. Create len(env_fns) * k env instances
        envs: list[Env] = []
        group_ids: list[int] = []
        for group_id, env_fn in enumerate(self.env_fns):
            for _ in range(k):
                envs.append(env_fn())
                group_ids.append(group_id)

        n = len(envs)

        # 2. Reset all envs
        conversations: list[list[Message]] = [env.reset() for env in envs]
        rewards: list[list[float]] = [[] for _ in range(n)]
        done: list[bool] = [False] * n

        # 3. Loop until all done
        while not all(done):
            # Find active slots
            active = [i for i in range(n) if not done[i]]

            # Render prompts for active slots
            prompts = [
                tokenizer.apply_chat_template(
                    conversations[i],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for i in active
            ]

            # Generate completions
            completions = generate_fn(prompts)

            # Step each active env
            for idx, completion in zip(active, completions):
                conversations[idx].append({"role": "assistant", "content": completion})
                next_msg, reward, is_done = envs[idx].step(completion)
                rewards[idx].append(reward)
                done[idx] = is_done
                if next_msg is not None:
                    conversations[idx].append(next_msg)

        # 4. Package into Rollouts
        return [
            Rollout(
                messages=conversations[i],
                rewards=rewards[i],
                group_id=group_ids[i],
            )
            for i in range(n)
        ]
