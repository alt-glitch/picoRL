# picoRL Training Report

## Run: divine-thunder-25 (wandb `i9cxyhh6`)

**Date:** 2026-02-27
**Model:** Qwen/Qwen3-4B-Instruct-2507
**Method:** GRPO + LoRA (rank 32, alpha 64)
**Task:** Letter counting (difficulty 6: 40-80 char random strings, 8-12 letters)
**Hardware:** NVIDIA A100 80GB PCIe (single GPU)
**Updates completed:** 64 / 200 (stopped early — plateaued)

## Config

| Parameter | Value |
|-----------|-------|
| algo | grpo |
| model | Qwen3-4B-Instruct-2507 |
| lora_rank | 32 |
| lora_alpha | 64 |
| lr | 5e-6 |
| k (rollouts/task) | 8 |
| num_tasks | 8 |
| train_difficulty | 6 |
| max_tokens | 2048 |
| sft_steps | 50 |
| temperature | 1.0 |
| gpu_memory_utilization | 0.3 |
| trainable params | 66M / 4.1B (1.6%) |

## SFT Warmup

| Metric | Value |
|--------|-------|
| SFT loss | 0.67 → 0.11 (50 steps) |
| Post-SFT format compliance | 84% |
| Post-SFT mean reward | 0.751 |

## Training Results

### Reward Trajectory

| Updates | Mean Reward | Frac Correct | Notes |
|---------|-------------|--------------|-------|
| 0 | 0.551 | 0.38 | Baseline after SFT |
| 1-3 | 0.695-0.713 | 0.48-0.50 | Quick initial jump |
| 4-63 | 0.54-0.75 | 0.31-0.55 | Plateau, high variance |

**5-update rolling average reward:** 0.55 → 0.67 (initial gain) → flat at ~0.66

### Eval Accuracy (greedy, every 10 updates)

| Update | 3-letter | 5-letter | 8-letter (train) |
|--------|----------|----------|------------------|
| 0 | 90.6% | 87.5% | 40.6% |
| 10 | 93.8% | 81.2% | 34.4% |
| 20 | 93.8% | 84.4% | **53.1%** |
| 30 | **96.9%** | 78.1% | **56.2%** |
| 40 | **100%** | 84.4% | 50.0% |
| 50 | 96.9% | **90.6%** | 43.8% |
| 60 | 96.9% | 75.0% | 43.8% |

### Key Observations

1. **3-letter counting (easy):** Improved from 90.6% → 100%. Model mastered this.
2. **8-letter counting (train difficulty):** Peaked at 56.2% (update 30), then regressed to ~44%. No sustained improvement.
3. **Training reward plateau:** Flat at ~0.66 after the first 3 updates. The initial jump represents GRPO quickly exploiting the model's existing capabilities.
4. **Tiny gradients:** grad_norm consistently 0.02-0.04. Within-group reward variance is low because partial credit compresses rewards into 0.5-0.8 range.

## Why It Plateaued

**RL can't teach algorithms.** The letter counting task requires a systematic strategy:
1. Spell out each character in the string
2. Track counts for each target letter
3. Report the tallied counts

The model doesn't do this — it outputs a JSON answer directly without chain-of-thought reasoning. RL can only reinforce existing reasoning patterns; it can't invent new ones.

**No CoT = no learning signal.** Qwen3-4B has native `<think>...</think>` mode, but we didn't enable it. Without CoT, the model's "counting" is pattern matching / intuition, which has a hard ceiling.

**Partial credit compresses variance.** Most rollouts in a group score 0.5-0.8 (getting 5-7 of 10 letters right). GRPO needs within-group variance to produce gradients. When all rollouts score similarly, advantages are near-zero.

## What Would Improve Results

1. **Enable Qwen3 thinking mode** (`<think>...</think>`) — let the model develop CoT through RL
2. **CoT SFT** — teach the model to spell out the string and count before answering
3. **Curriculum learning** — start at easier difficulty, gradually increase (like Atropos)
4. **Binary reward** — remove partial credit to maximize within-group variance
5. **Larger k** — more rollouts per task increases chance of finding winning strategies

## Previous Runs Summary

| Model | Method | Result |
|-------|--------|--------|
| Qwen2.5-0.5B | GRPO full-ft | SFT works, RL flat (model too dumb to count) |
| Qwen2.5-3B | GRPO full-ft | Already solves diff 1-7 (too smart), diff 8+ too hard |
| Qwen3-4B | GRPO full-ft | OOM (can't fit on 80GB) |
| **Qwen3-4B** | **GRPO + LoRA** | **Small gains on easy tasks, plateau on hard (this run)** |

## Files

- `results/training_curves.png` — reward, accuracy, loss curves
- `results/wandb_metrics_i9cxyhh6.json` — raw wandb metrics
- `results/train_qwen3_4b_lora_diff6.log` — full training log
- wandb: https://wandb.ai/sidbin/picorl/runs/i9cxyhh6
