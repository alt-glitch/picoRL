# picoRL

Minimal RL for LLMs. Three algorithms, one environment, one GPU.

Workshop companion code for [Fifth Elephant](https://fiftheelephant.talkfunnel.com/).
The audience builds LLM agents but has never touched training -- this repo
makes the jump from *using* models to *training* them.

## The 30-second version

RL for LLMs in one paragraph: sample k completions per prompt, score each
with a reward function, compute advantage = reward - group_mean, multiply
by log-probability of the completion, negate, backprop. That's GRPO.

## Reading order

**If you want to understand the algorithms** (the star of the show):

1. `algorithms/common.py` -- log-probs and the causal LM shift
2. `algorithms/reinforce.py` -- simplest policy gradient (~30 lines of core)
3. `algorithms/grpo.py` -- group-relative baselines, the DeepSeek-R1 algorithm (~35 lines)
4. `algorithms/ppo.py` -- clipping + value head, for comparison

**If you want to understand the full training loop:**

1. `environments/base.py` -- the env interface (reset/step)
2. `environments/letter_counting.py` -- a concrete env
3. `core/types.py` -- data structures (Message, Rollout)
4. `core/batched_env.py` -- parallel rollout collection
5. `train.py` -- everything wired together

**If you want to see what actually happens when you train:**

1. `runs/letter_counting_qwen3_4b/README.md` -- the war journal

## Architecture

```
algorithms/        Policy gradient loss functions (pure PyTorch, no abstractions)
  common.py          Shared: tokenization, log-probs, causal shift
  reinforce.py       REINFORCE with batch-mean baseline
  grpo.py            GRPO with per-group baselines
  ppo.py             PPO with clipping + value head

environments/      Task definitions
  base.py            Env ABC (reset/step)
  letter_counting.py 10 difficulty tiers, partial credit scoring

core/              Infrastructure
  types.py           Message, Rollout dataclasses
  batched_env.py     Parallel env management + rollout collection
  nanollm.py         NanoLLM: vLLM weight sharing with HF model

train.py           Monolithic training script -- explicit loop, no Trainer class
scratch.py         Workshop sandbox -- implement core functions from scratch
```

## Quick start

```bash
# Install
git clone https://github.com/alt-glitch/picoRL.git && cd picoRL
uv sync

# Train (on a GPU server with vLLM)
uv run python train.py \
    --algo grpo \
    --model-id Qwen/Qwen3-4B-Instruct-2507 \
    --lora-rank 32 \
    --sft-steps 50 \
    --train-difficulty 6

# Workshop mode: implement functions in scratch.py, test inline
uv run python scratch.py
```

## Key concepts

- **The Causal LM Shift**: `logits[:,t,:]` predicts `input_ids[:,t+1]`. See `algorithms/common.py`.
- **Weight Sharing**: vLLM and HF model share GPU tensors via `p.data = hf_p.data`. See `core/nanollm.py`.
- **Group Baselines**: Compare completions to their group, not the batch. See `algorithms/grpo.py`.
- **LoRA**: Train 1.6% of parameters. Create NanoLLM first, THEN wrap with PEFT. See `train.py`.

## References

- [GyLLM](https://github.com/RedTachyon/gyllm/) -- NanoLLM weight-sharing pattern, loss functions
- [Ludic](https://github.com/hallerite/ludic) -- Algorithm = CreditAssigner x Loss decomposition
- [Atropos](https://github.com/NousResearch/Atropos) -- Multi-environment RL training
- [Unsloth](https://github.com/unslothai/unsloth) -- Fast LoRA + GRPO training
