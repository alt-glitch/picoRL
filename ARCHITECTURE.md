# picoRL Architecture

## What We're Building

A minimal GRPO (Group Relative Policy Optimization) implementation for training LLMs with RL.
No clipping, no importance sampling, no KL penalty. Pure on-policy REINFORCE with group-relative baselines.

**Target**: Qwen 0.5B-3B on a single A100. Workshop participants write ~500 lines.

## The Four Layers

```
┌─────────────────────────────────────────────────────────┐
│  WORKSHOP API (what participants see)                   │
│  train.py, workshop/stubs/, workshop/solutions/         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  FRAMEWORK LAYER                                        │
│  NanoLLM, Env, BatchedEnv, collect_rollouts, types      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  CORE LAYER (extracted from scratch.py)                 │
│  get_logprobs(), grpo_loss(), compute_advantages()      │
│  Raw PyTorch, no abstractions, every line explicit      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  SCRATCH.PY (Sid's learning sandbox)                    │
│  Implement each fn, test inline, checkpoint questions   │
└─────────────────────────────────────────────────────────┘
```

---

## Reference Libraries: What We Stole

### From GyLLM/NanoRL

GyLLM is a minimal single-GPU RL training setup. ~340-line scripts, no Trainer class.

**Stolen: NanoLLM weight sharing**

The key insight is `p.data = hf_p.data` -- rebind vLLM parameter storage to point at HF model parameters.
After this, gradient updates to HF weights are immediately visible to vLLM without any copy.

```python
# The money line from gyllm/packages/nanorl/src/nanorl/rollout/nanollm.py
for name, p in vllm_model.named_parameters():
    hf_p = _lookup_hf_param(name, hf_params, prefixes)
    p.data = hf_p.data  # rebind storage -- zero-copy weight sharing
```

Critical vLLM settings for this to work:
- `distributed_executor_backend="uni"` -- single-process (same address space)
- `model_impl="transformers"` -- matching module names with HF
- `load_format="dummy"` -- skip loading weights from disk
- `enforce_eager=True` -- disable CUDA graphs

Sleep/wake cycle: `llm.sleep(level=1)` frees KV cache for training memory. `llm.wake_up()` re-allocates.
Weights stay shared; only inference buffers are toggled. This is what enables train+infer on one GPU.

**Stolen: Standalone loss functions**

`compute_grpo_advantages()` (~50 lines): groups by group_id, `advantage_i = reward_i - mean(group_rewards)`.
`compute_grpo_loss()` (~50 lines): per-rollout forward pass, shift logits by 1, gather token logprobs,
mask to assistant tokens, policy gradient `loss -= advantage * logprob_term`.

No class hierarchies. Just functions that take tensors and return tensors.

**Stolen: Incremental tokenization for assistant mask**

`tokenize_with_assistant_mask()` (~30 lines): encode `messages[:1]`, `messages[:2]`, ..., diff lengths
to find where each message's tokens start/end. Mark assistant spans as 1. Handles arbitrary chat templates.

```python
# From gyllm/packages/nanorl/src/nanorl/rl/reinforce.py
prev_len = 0
for idx, message in enumerate(messages):
    prefix_ids = _encode_chat_messages(tokenizer, messages[:idx + 1])
    curr_len = prefix_ids.shape[0]
    if message["role"] == "assistant":
        mask[prev_len:curr_len] = True
    prev_len = curr_len
```

**Stolen: Explicit training loop structure**

```
for update in range(N):
    model.train(False); llm.wake_up()
    rollouts = collect(env, agent)
    llm.sleep(1); model.train(True)
    advantages = compute_grpo_advantages(rollouts)
    for minibatch in chunks(rollouts):
        loss = compute_grpo_loss(minibatch, advantages, model, tokenizer)
        (loss * scale).backward()
    clip_grad_norm_(); optimizer.step(); optimizer.zero_grad()
```

**Stolen: EpisodeRollout dataclass**

```python
@dataclass(frozen=True)
class EpisodeRollout:
    messages: list[Message]
    rewards: list[float]
    actions: list[str]
    group_id: str | int | None = None
```

**Skipped**: Request TypedDict, ActorId encoding, BatchedEnv auto-reset, Agent state machine.

---

### From Ludic

Ludic is a production RL library. We steal mental models, not code.

**Stolen concept: "Algorithm = CreditAssigner + Loss"**

This decomposition is gold for teaching:

```
RLAlgorithm = CreditAssigner + Loss
  CreditAssigner: Rollouts -> per-rollout scalar weights (advantages)
  Loss:           (logits, batch) -> scalar loss
```

Composing different CreditAssigners with different Losses gives you:
- GroupNormalizedReturn + ReinforceLoss = GRPO
- MonteCarloReturn + ReinforceLoss = REINFORCE
- ConstantCredit + CrossEntropyLoss = SFT

For picoRL we hardcode GRPO, but participants should understand this decomposition.

**Stolen concept: GroupNormalizedReturn**

From `ludic/src/ludic/training/credit_assignment.py`:

```python
rewards = [r.total_reward for r in group_rollouts]
baseline = mean(rewards)
advantages = [r - baseline for r in rewards]
# Same advantage assigned to EVERY step in a rollout (episodic, not per-step)
```

Key options we expose: `normalize_adv` (divide by std+eps), `positive_only` (clamp negatives to 0).

**Stolen: `selective_log_softmax`**

Fused log_softmax + gather avoids materializing the full `[B, T, V]` probability tensor.
For a 7B model with V=32K, B=8, T=4096: saves ~4GB VRAM.

```python
# From ludic/src/ludic/training/loss.py
def selective_log_softmax(logits, index):
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
```

Looks naive, but `torch.compile` fuses it into a single kernel. We teach the naive version
in scratch.py, then show the compiled version as an optimization.

**Stolen concept: Parser composition**

```python
def compose_parsers(*parsers):
    def _p(raw):
        current = ParseResult(action=raw, reward=0.0, obs=None)
        for parser in parsers:
            result = parser(current.action)
            if result.action is None:  # parse failed
                return ParseResult(action=None, reward=current.reward + result.reward, obs=result.obs)
            current = ParseResult(action=result.action, reward=current.reward + result.reward, obs=None)
        return current
    return _p
```

Short-circuit on failure, reward accumulation. Clean pattern for format verification rewards.
We don't implement this in picoRL core, but it's the right pattern for env reward functions.

**Stolen concept: InteractionProtocol**

The agent-env loop is explicit, not hidden inside the agent:
- Protocol owns the loop
- Agent owns inference + parsing
- Env owns state

For picoRL: `collect_rollouts()` is our protocol. It owns the multi-turn loop.

**Skipped**: Multi-agent kernel (LudicEnv with AgentID dicts), TokenClippedSurrogateLoss,
SharedContext lazy caching, CompositeLoss, FSDP2/distributed, all exotic algorithms
(CISPO, SAPO, GMPO, ScaleRL).

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Reference model | No separate ref model | On-policy GRPO doesn't need one. No KL penalty. Saves GPU memory. |
| Importance sampling | No IS ratios | On-policy = no old policy to correct for. Clearest formulation. |
| Loss formulation | `-(advantage * sum_logprobs).mean()` | Pure policy gradient. No clipping, no ratios. Add later as extensions. |
| Environment API | 3 methods: reset/step/reward | Minimum viable. reward() separate from step() for clarity. |
| Multi-turn | BatchedEnv manages parallel episodes | Each env steps independently. Rollout loop generates for all non-done. |
| Weight sharing | vLLM native sleep/wake + data rebinding | Single GPU train+infer. `p.data = hf_p.data` for zero-copy. |
| Training | No Trainer class | Explicit loop. Every line visible. |
| Batching | Simple gradient accumulation | Chunk into mini-batches, accumulate gradients, single optimizer step. |

---

## GRPO in One Paragraph

Sample k completions per prompt. Score each with a reward function. For each group of k completions,
compute advantage as reward minus group mean. Multiply advantage by the sum of log-probabilities of
the assistant tokens. Negate (because we minimize). Average over all completions. Backprop. Step.
That's it. The "group-relative" part replaces the value network (PPO) or batch baseline (REINFORCE)
with a per-prompt baseline computed from the k completions themselves.

## The Causal LM Shift

This is the single most confusing thing in the codebase if you haven't seen it before.

A causal LM predicts the NEXT token. `model(input_ids)` returns logits where:
- `logits[:, t, :]` is the prediction for position `t+1`
- `logits[:, 0, :]` predicts `input_ids[:, 1]`
- `logits[:, -1, :]` predicts the token AFTER the sequence (not in input_ids)

So to get the log-probability of each actual token:
```python
logits = model(input_ids).logits          # [B, T, V]
shifted_logits = logits[:, :-1, :]        # [B, T-1, V] -- predictions
shifted_labels = input_ids[:, 1:]         # [B, T-1]    -- targets
logprobs = log_softmax(shifted_logits, dim=-1)
token_logprobs = gather(logprobs, dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
# token_logprobs[b, t] = log P(input_ids[b, t+1] | input_ids[b, :t+1])
```

The assistant mask must also be shifted by 1 to align: `mask[1:]` not `mask[:-1]`.
