from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Message


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
