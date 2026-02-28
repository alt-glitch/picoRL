"""Shared utilities for all RL algorithms: tokenization and log-probability computation.

If you build LLM agents, you already know how to turn text into tokens and feed
them to a model. These two functions do exactly that, plus one extra piece that
RL training needs: *per-token log-probabilities* -- the model's confidence in
each token it generated. The RL loss uses these to nudge the model toward tokens
that led to high rewards and away from tokens that led to low rewards.

The trickiest part of this file is the **Causal LM Shift**. It trips up almost
everyone the first time, so here is an analogy:

    Think of a weather forecaster. Monday's forecast predicts Tuesday's weather.
    In the same way, logits[:, t, :] is the model's prediction of input_ids[:, t+1].
    The model at position t has seen tokens 0..t and is predicting what comes next.

The shift in practice -- a 5-token sequence "A B C D E":

    Position index:    0    1    2    3    4
    input_ids:        [A]  [B]  [C]  [D]  [E]
                       |    |    |    |
                       v    v    v    v
    logits predict:   [B]  [C]  [D]  [E]         <-- logits[:, :-1, :]
    actual labels:    [B]  [C]  [D]  [E]         <-- input_ids[:, 1:]
                       ^    ^    ^    ^
                       |    |    |    |
    logprobs index:    0    1    2    3            <-- length T-1

So `get_per_token_logprobs` returns a tensor of length T-1, not T. Any mask
built for the original input_ids (length T) must also be shifted: mask[1:].
"""
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
    """Returns (input_ids [T], mask [T]) -- 1D, on device. mask=True for assistant tokens.

    We only want gradients on the tokens the *model* wrote (assistant turns),
    not on the prompt or tool outputs (user/system turns). This function figures
    out which tokens belong to which role.
    """
    input_ids = encode_chat_messages(tokenizer, messages).to(device)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    # Incremental encoding trick: we can't just tokenize each message separately
    # and concatenate, because chat templates insert special tokens *between*
    # messages (e.g. <|im_start|>assistant\n). The boundary between message N
    # and message N+1 only exists when you tokenize messages[:N+1] as a whole.
    #
    # So we encode progressively longer prefixes:
    #   messages[:1]  -> length 20 tokens   (system)
    #   messages[:2]  -> length 45 tokens   (system + user)
    #   messages[:3]  -> length 80 tokens   (system + user + assistant)
    # The assistant's tokens are positions 45..80. We mark those True in the mask.
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
    logits = model(input_ids).logits  # [B, T, vocab_size]
    logprobs = F.log_softmax(logits, dim=-1)  # normalize to log-probabilities

    # Apply the causal LM shift (see module docstring for the ASCII diagram):
    # - logprobs[:, t, :] is the prediction *after* seeing tokens 0..t
    # - It predicts the *next* token, which is input_ids[:, t+1]
    # - So we drop the last position from logprobs (nothing to predict after it)
    #   and drop the first position from input_ids (nothing predicted it).
    shift_logprobs = logprobs[:, :-1, :]  # [B, T-1, vocab_size]
    shift_labels = input_ids[:, 1:]  # [B, T-1]

    # gather picks out the log-prob of the token that was actually generated,
    # giving us one scalar per position: "how confident was the model in this
    # specific token?"
    token_logprobs = shift_logprobs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [B, T-1]

    return token_logprobs
