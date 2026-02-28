# War Journal: Letter Counting with Qwen3-4B + LoRA

Training a 4B parameter model to count letters in random strings using GRPO.
64 updates completed before we stopped. Here's every footgun we stepped on.

wandb: https://wandb.ai/sidbin/picorl/runs/i9cxyhh6

---

## The Footgun Parade

### 1. OOM During wake_up(), Not During Training

You'd think the GPU would run out of memory during the backward pass. Nope.
It crashed when vLLM tried to re-allocate its KV cache after training.

**What happens:** PyTorch's CUDA allocator holds onto freed memory as a cache.
When vLLM calls `wake_up()`, it asks CUDA for a big contiguous block for the
KV cache. PyTorch is sitting on all the fragments. Boom.

**Fix:** `torch.cuda.empty_cache()` after `optimizer.step()`, before `wake_up()`.
One line. Two hours of debugging.

### 2. PEFT Renames Your Parameters (Silently)

We wrapped the model with LoRA, then created NanoLLM for weight sharing.
NanoLLM linked 146 out of 398 parameters. The other 252? Gone.

**What happens:** PEFT replaces `nn.Linear` modules with `LoraLayer` wrappers.
The parameter name `model.layers.0.self_attn.q_proj.weight` becomes
`base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight`.
NanoLLM tries to match names between vLLM and HF. Doesn't find them.

**Fix:** Create NanoLLM FIRST (binds weights by original names), THEN apply PEFT.
PEFT preserves the original weight tensors inside `base_layer` -- the pointer
sharing still works, the names just don't match anymore.

Order matters: NanoLLM, then PEFT. Not the other way around.

### 3. Binary Reward = Zero Gradient

With binary reward (0 or 1) and ~5% accuracy on difficulty 6, here's what
happens in a k=8 group:

    Group rewards: [0, 0, 0, 0, 0, 0, 0, 0]
    Group mean: 0.0
    Advantages: [0, 0, 0, 0, 0, 0, 0, 0]

Zero advantage = zero gradient = zero learning. About 66% of groups looked
like this. Two-thirds of our compute was doing nothing.

**Fix:** Partial credit. If the model gets the format right but the wrong
answer, give it 0.1 instead of 0. For multi-letter counting, give
`0.1 + 0.9 * (fraction of letters correct)`. Now most groups have variance.

### 4. Partial Credit Was Too Generous

First attempt at partial credit used distance-based scoring: how close was
each letter count to the correct answer? Sounds reasonable. In practice,
rewards compressed into the 0.7-0.9 range.

    Group rewards: [0.82, 0.85, 0.79, 0.83, 0.81, 0.84, 0.80, 0.82]
    Group mean: 0.82
    Advantages: [-0.00, +0.03, -0.03, +0.01, -0.01, +0.02, -0.02, +0.00]

Grad norms: 0.001. Effectively zero again, just for a different reason.

**Fix:** Switched to fraction-of-exact-matches: `0.1 + 0.9 * (n_correct / n_letters)`.
Either you got a letter count exactly right or you didn't. Wider spread.

### 5. Eval Accuracy Was a Lie

We had `sum(1 for r in rewards if r > 0)` to count correct answers.
With partial credit, a wrong answer that had correct format scored 0.1.
`0.1 > 0` is `True`. Every formatted response was "correct."

Eval accuracy: 95%! We're geniuses! ...wait.

**Fix:** `r >= 1.0` instead of `r > 0`. Exact match only.

### 6. The Goldilocks Problem

| Model | What Happened |
|-------|--------------|
| Qwen2.5-0.5B | Learned the format (88% compliance). Could not learn to count. Reward flat at 5%. Too dumb. |
| Qwen2.5-3B | Already solves difficulties 1-7 out of the box (~90%+ accuracy). Nothing for RL to improve. Too smart. |
| Qwen3-4B (full fine-tune) | Model (8GB) + optimizer states (24GB) + activations = OOM on A100 80GB. Too big. |
| **Qwen3-4B + LoRA r=32** | 66M trainable params (1.6%). Fits. Has room to improve on difficulty 6. Just right. |

### 7. vLLM's 4GB Serialization Wall

vLLM v0.15 uses msgspec for inter-process communication. msgspec has a hard
limit: objects larger than 2^32 bytes can't be serialized. A 4B model's
weights are ~8GB. Crash on startup with a cryptic `EncodeError`.

**Fix:** `VLLM_ENABLE_V1_MULTIPROCESSING=0` -- run vLLM in-process instead
of spawning a worker. No serialization needed. But now both models live
on the same GPU, so memory management matters even more.

### 8. RL Can't Teach Algorithms

This is the big one. After all the fixes above, training worked mechanically:
gradients flowed, advantages were nonzero, loss decreased. But eval accuracy
on the training difficulty plateaued at ~45%.

The model counts letters by *intuition* -- pattern matching from pretraining.
It doesn't spell out "s-t-r-a-w-b-e-r-r-y, that's 3 r's." Without that
chain-of-thought reasoning, there's a hard ceiling on accuracy.

RL can reinforce strategies the model already uses. It cannot invent new ones.
To teach counting, you'd need CoT-SFT first (teach the model to reason
step-by-step), then use RL to refine that reasoning.

Qwen3-4B has native `<think>...</think>` mode. Enabling it would let RL
discover and reinforce CoT strategies. That's the obvious next experiment.

---

## Final Numbers

| Metric | Start (update 0) | Best | End (update 63) |
|--------|-------------------|------|-----------------|
| Train reward | 0.551 | 0.750 | 0.615 |
| Frac correct (exact match) | 38% | 55% | 34% |
| 3-letter eval accuracy | 90.6% | 100% | 96.9% |
| 5-letter eval accuracy | 87.5% | 90.6% | 75.0% |
| 8-letter eval accuracy (train diff) | 40.6% | 56.2% | 43.8% |

See `training_curves.png` for the full picture.

---

## Config

```
model: Qwen/Qwen3-4B-Instruct-2507
algo: GRPO
lora_rank: 32, lora_alpha: 64
difficulty: 6 (40-80 char random strings, 8-12 letters to count)
k: 8 rollouts per task, 8 tasks per batch (64 rollouts/update)
lr: 5e-6
sft_steps: 50
gpu: A100 80GB PCIe
time: ~2 min/update, 64 updates completed
```
