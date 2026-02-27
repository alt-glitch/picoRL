from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def encode_chat_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> torch.Tensor:
    """Tokenize a chat message list. Returns 1D input_ids [T]."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    return encoded[0]  # [1, T] -> [T]


def tokenize_with_assistant_mask(
    tokenizer: Any, messages: list[dict[str, str]], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (input_ids [T], mask [T]) — 1D, on device. mask=True for assistant tokens."""
    input_ids = encode_chat_messages(tokenizer, messages).to(device)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    prev_len = 0
    for idx, message in enumerate(messages):
        ids = encode_chat_messages(tokenizer, messages[: idx + 1]).to(device)
        current_len = ids.shape[0]
        if message["role"] == "assistant":
            mask[prev_len:current_len] = True
        prev_len = current_len
    return input_ids, mask


def get_per_token_logprobs(
    model,
    input_ids: torch.Tensor,  # [B, T]
) -> torch.Tensor:  # [B, T-1]
    """
    Compute per-token log-probabilities for a batch of sequences.
    Causal LM shift: logits[:, t, :] predicts input_ids[:, t+1],
    so we align logits[:, :-1] with input_ids[:, 1:].
    """
    logits = model(input_ids).logits
    logprobs = F.log_softmax(logits, dim=-1)

    shift_logprobs = logprobs[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    token_logprobs = shift_logprobs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_logprobs
