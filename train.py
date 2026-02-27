"""
picoRL training script.

Supports REINFORCE and PPO on LetterCountingEnv with vLLM inference.

Usage:
    uv run python train.py --algo reinforce
    uv run python train.py --algo ppo
    uv run python train.py --algo ppo --model-id Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from time import perf_counter

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import torch
import torch.nn.functional as F
import wandb
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from envs.letter_counting import LetterCountingEnv
from picorl.algorithms.ppo import (
    compute_ppo_policy_loss,
    compute_ppo_value_loss,
    prepare_ppo_rollouts,
)
from picorl.algorithms.reinforce import compute_reinforce_loss
from picorl.algorithms.utils import tokenize_with_assistant_mask
from picorl.env import BatchedEnv
from picorl.rollout import NanoLLM, make_generate_fn
from picorl.types import Message, Rollout


# Difficulty tier -> approximate letter count label for wandb keys
DIFFICULTY_LABELS = {3: "3_letter", 5: "5_letter", 6: "8_letter"}


@dataclass
class Config:
    # Model
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"

    # Algorithm
    algo: str = "reinforce"  # reinforce | ppo
    k: int = 1  # rollouts per task

    # Environment
    train_difficulty: int = 1  # tier 1 -> single letter in short words
    num_tasks: int = 16
    eval_every: int = 10
    eval_difficulties: tuple[int, ...] = (3, 5, 6)  # train difficulty + harder
    eval_num_tasks: int = 32  # more tasks for stable eval
    eval_examples: int = 20  # completions to log in wandb table

    # Training
    num_updates: int = 100
    lr: float = 1e-5
    max_grad_norm: float = 1.0

    # PPO-specific
    ppo_clip_range: float = 0.2
    ppo_value_coef: float = 0.5

    # SFT warmup
    sft_steps: int = 0  # 0 to skip warmup
    sft_num_examples: int = 100  # size of synthetic dataset

    # Sampling
    temperature: float = 1.0
    max_tokens: int = 512

    # NanoLLM
    gpu_memory_utilization: float = 0.4
    enable_sleep_mode: bool = True

    # Logging
    wandb_project: str = "picorl"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="picoRL training")
    parser.add_argument("--algo", default="reinforce", choices=["reinforce", "ppo"])
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--train-difficulty", type=int, default=3)
    parser.add_argument("--num-tasks", type=int, default=16)
    parser.add_argument("--num-updates", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-num-tasks", type=int, default=32)
    parser.add_argument("--sft-steps", type=int, default=0)
    parser.add_argument("--sft-num-examples", type=int, default=100)
    parser.add_argument("--wandb-project", default="picorl")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        model_id=args.model_id,
        algo=args.algo,
        k=args.k,
        train_difficulty=args.train_difficulty,
        num_tasks=args.num_tasks,
        num_updates=args.num_updates,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        eval_every=args.eval_every,
        eval_num_tasks=args.eval_num_tasks,
        sft_steps=args.sft_steps,
        sft_num_examples=args.sft_num_examples,
        wandb_project=args.wandb_project,
    )
    cfg._no_wandb = args.no_wandb  # type: ignore[attr-defined]
    return cfg


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_reward_stats(rollouts: list[Rollout]) -> dict[str, float]:
    """Compute reward distribution stats from rollouts."""
    rewards = [sum(r.rewards) for r in rollouts]
    t = torch.tensor(rewards)
    correct = sum(1 for r in rewards if r > 0)
    return {
        "reward/mean": float(t.mean()),
        "reward/std": float(t.std()) if len(rewards) > 1 else 0.0,
        "reward/max": float(t.max()),
        "reward/min": float(t.min()),
        "reward/fraction_correct": correct / max(len(rewards), 1),
    }


def compute_advantage_stats(rollouts: list[Rollout]) -> dict[str, float]:
    """Compute advantage stats using batch-mean baseline (REINFORCE-style)."""
    rewards = torch.tensor([sum(r.rewards) for r in rollouts])
    advantages = rewards - rewards.mean()
    positive = int((advantages > 0).sum())
    return {
        "advantages/mean": float(advantages.mean()),
        "advantages/std": float(advantages.std()) if len(rollouts) > 1 else 0.0,
        "advantages/fraction_positive": positive / max(len(rollouts), 1),
    }


def compute_format_compliance(rollouts: list[Rollout]) -> float:
    """Fraction of responses with parseable <answer>...</answer> tags."""
    compliant = 0
    for r in rollouts:
        for msg in r.messages:
            if msg["role"] == "assistant":
                if re.search(r"<answer>.+?</answer>", msg["content"], re.IGNORECASE):
                    compliant += 1
                    break
    return compliant / max(len(rollouts), 1)


def make_example_table(
    rollouts: list[Rollout],
    update: int,
    max_examples: int = 20,
) -> wandb.Table:
    """Build a wandb table of example completions."""
    table = wandb.Table(columns=["step", "prompt", "completion", "reward"])
    for r in rollouts[:max_examples]:
        prompt = r.messages[0]["content"] if r.messages else ""
        completion = ""
        for msg in r.messages:
            if msg["role"] == "assistant":
                completion = msg["content"]
                break
        reward = sum(r.rewards)
        table.add_data(update, prompt[:200], completion[:200], reward)
    return table


def clip_and_report_grad(
    model: torch.nn.Module,
    max_grad_norm: float,
) -> dict[str, float]:
    """Clip gradients and return stats. grad/norm is the norm BEFORE clipping."""
    grad_norm = float(clip_grad_norm_(model.parameters(), max_grad_norm).item())
    return {
        "grad/norm": grad_norm,
        "grad/clipped": 1.0 if grad_norm > max_grad_norm else 0.0,
    }


def compute_ppo_explained_variance(batches) -> float:
    """Explained variance of the value head: 1 - Var(returns - values) / Var(returns)."""
    all_returns = []
    all_values = []
    for b in batches:
        all_returns.append(b.returns)
        all_values.append(b.old_values)
    if not all_returns:
        return 0.0
    returns = torch.cat(all_returns)
    values = torch.cat(all_values)
    var_returns = returns.var()
    if var_returns < 1e-8:
        return 0.0
    return float(1.0 - (returns - values).var() / var_returns)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    generate_fn,
    tokenizer,
    difficulties: tuple[int, ...],
    num_tasks: int,
    update: int,
    max_examples: int = 20,
) -> tuple[dict[str, float], wandb.Table | None]:
    """Run greedy evaluation on multiple difficulties.

    Returns (metrics_dict, example_table).
    """
    metrics: dict[str, float] = {}
    all_rollouts: list[Rollout] = []

    for diff in difficulties:
        label = DIFFICULTY_LABELS.get(diff, f"diff{diff}")
        env_fns = [lambda d=diff: LetterCountingEnv(difficulty=d) for _ in range(num_tasks)]
        rollouts = BatchedEnv(env_fns).collect_rollouts(generate_fn, tokenizer, k=1)

        rewards = [sum(r.rewards) for r in rollouts]
        correct = sum(1 for r in rewards if r > 0)
        accuracy = correct / max(len(rollouts), 1)
        compliance = compute_format_compliance(rollouts)

        metrics[f"eval/{label}_accuracy"] = accuracy
        metrics[f"eval/{label}_format_compliance"] = compliance
        all_rollouts.extend(rollouts)

    table = make_example_table(all_rollouts, update, max_examples) if all_rollouts else None
    return metrics, table


# ---------------------------------------------------------------------------
# SFT warmup
# ---------------------------------------------------------------------------

def generate_sft_example(difficulty: int) -> list[Message]:
    """Generate a synthetic letter-counting example with a reasoning trace."""
    env = LetterCountingEnv(difficulty=difficulty)
    msgs = env.reset()  # [system, user]
    question = msgs[1]["content"]

    if len(env._target_letters) == 1:
        letter = env._target_letters[0]
        count = env._expected_counts[letter]
        match = re.search(r"in the string (.+?)(?:\?|\n)", question)
        text = match.group(1) if match else ""

        spelled = ", ".join(text)
        reasoning = f'Let me count each "{letter}" in "{text}":\n{spelled}\n'
        reasoning += f'The letter "{letter}" appears {count} time{"s" if count != 1 else ""}.'
        answer = f"\n<answer>{count}</answer>"
    else:
        match = re.search(r"in the string (.+?)(?:\n|$)", question)
        text = match.group(1) if match else ""

        spelled = ", ".join(text)
        reasoning = f'Let me count each target letter in "{text}":\n{spelled}\n\n'
        for letter in env._target_letters:
            count = env._expected_counts[letter]
            reasoning += f'"{letter}": {count} time{"s" if count != 1 else ""}\n'

        answer_dict = json.dumps(env._expected_counts)
        answer = f"\n<answer>{answer_dict}</answer>"

    msgs.append({"role": "assistant", "content": reasoning + answer})
    return msgs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    print(f"Config: algo={cfg.algo} model={cfg.model_id} k={cfg.k} "
          f"difficulty={cfg.train_difficulty} tasks={cfg.num_tasks}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=getattr(torch, cfg.dtype),
        device_map="cuda",
    )

    llm = NanoLLM(
        model,
        tokenizer=tokenizer,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enable_sleep_mode=cfg.enable_sleep_mode,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    device = next(model.parameters()).device

    # Training sampling (stochastic)
    train_sampling = SamplingParams(
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    train_generate_fn = make_generate_fn(llm, train_sampling)

    # Eval sampling (greedy)
    eval_sampling = SamplingParams(temperature=0.0, max_tokens=cfg.max_tokens)
    eval_generate_fn = make_generate_fn(llm, eval_sampling)

    use_wandb = not getattr(cfg, "_no_wandb", False)
    if use_wandb:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    # --- SFT WARMUP ---
    if cfg.sft_steps > 0:
        print(f"SFT warmup: {cfg.sft_steps} steps on {cfg.sft_num_examples} synthetic examples")
        sft_data = [generate_sft_example(cfg.train_difficulty) for _ in range(cfg.sft_num_examples)]

        model.train(True)
        for step in range(cfg.sft_steps):
            optimizer.zero_grad(set_to_none=True)
            batch = random.sample(sft_data, min(cfg.num_tasks, len(sft_data)))
            total_loss = 0.0
            for msgs in batch:
                input_ids, mask = tokenize_with_assistant_mask(tokenizer, msgs, device=device)
                logits = model(input_ids.unsqueeze(0)).logits
                shift_logits = logits[0, :-1, :]
                shift_labels = input_ids[1:]
                shift_mask = mask[1:]
                ce = F.cross_entropy(shift_logits, shift_labels, reduction="none")
                masked_ce = ce[shift_mask]
                if masked_ce.numel() > 0:
                    total_loss = total_loss + masked_ce.mean()
            loss = total_loss / len(batch)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            if step % 10 == 0:
                print(f"  sft step={step:3d} loss={loss.item():.4f} grad_norm={grad_norm:.2f}")

        if use_wandb:
            wandb.log({"sft/final_loss": loss.item()}, step=-1)
        print("SFT warmup complete.")

    # --- TRAINING LOOP ---
    for update in range(cfg.num_updates):
        iter_start = perf_counter()

        # 1. ROLLOUT (no grad)
        model.train(False)
        with torch.no_grad():
            llm.wake_up()
            rollout_start = perf_counter()
            env_fns = [
                lambda: LetterCountingEnv(difficulty=cfg.train_difficulty)
                for _ in range(cfg.num_tasks)
            ]
            rollouts = BatchedEnv(env_fns).collect_rollouts(
                train_generate_fn, tokenizer, k=cfg.k,
            )
            rollout_time = perf_counter() - rollout_start
        llm.sleep(1)

        if not rollouts:
            print(f"update={update} skipped (no rollouts)")
            continue

        # Compute rollout stats (shared across all algos)
        reward_stats = compute_reward_stats(rollouts)
        advantage_stats = compute_advantage_stats(rollouts)

        # 2. TRAIN
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        grad_start = perf_counter()
        algo_metrics: dict[str, float] = {}

        if cfg.algo == "reinforce":
            loss, reinforce_metrics = compute_reinforce_loss(
                rollouts, model, tokenizer, device=device,
            )
            loss.backward()
            algo_metrics = {
                "loss/policy": float(loss.item()),
                "logprobs/mean_assistant": reinforce_metrics["avg_logprob"],
            }

        elif cfg.algo == "ppo":
            batches, prep_metrics = prepare_ppo_rollouts(
                rollouts, model, tokenizer, device=device,
            )
            if not batches:
                print(f"update={update} skipped (no PPO batches)")
                continue

            policy_loss, policy_metrics = compute_ppo_policy_loss(
                batches, model, device=device, clip_range=cfg.ppo_clip_range,
            )
            value_loss, value_metrics = compute_ppo_value_loss(
                batches, model, device=device,
            )
            loss = policy_loss + cfg.ppo_value_coef * value_loss
            loss.backward()

            # PPO advantage stats (from actual GAE, overrides batch-mean stats)
            all_adv = torch.cat([b.advantages for b in batches])
            advantage_stats = {
                "advantages/mean": float(all_adv.mean()),
                "advantages/std": float(all_adv.std()) if all_adv.numel() > 1 else 0.0,
                "advantages/fraction_positive": float((all_adv > 0).float().mean()),
            }

            algo_metrics = {
                "loss/policy": float(policy_loss.item()),
                "loss/value": float(value_loss.item()),
                "ppo/clip_fraction": policy_metrics["clip_fraction"],
                "ppo/approx_kl": policy_metrics["approx_kl"],
                "ppo/value_loss": value_metrics["value_loss"],
                "ppo/explained_variance": compute_ppo_explained_variance(batches),
                "logprobs/mean_assistant": prep_metrics["avg_logprob"],
            }

        else:
            raise ValueError(f"Unknown algo: {cfg.algo}")

        # Gradient stats (clip + capture norm)
        grad_stats = clip_and_report_grad(model, cfg.max_grad_norm)
        optimizer.step()
        grad_time = perf_counter() - grad_start
        iter_time = perf_counter() - iter_start

        # GPU memory
        gpu_stats: dict[str, float] = {}
        if torch.cuda.is_available():
            gpu_stats["gpu/memory_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

        # 3. EVALUATION (periodic, greedy)
        eval_metrics: dict[str, float] = {}
        eval_table = None
        if cfg.eval_every > 0 and update % cfg.eval_every == 0:
            model.train(False)
            with torch.no_grad():
                llm.wake_up()
                eval_metrics, eval_table = run_evaluation(
                    eval_generate_fn, tokenizer, cfg.eval_difficulties,
                    cfg.eval_num_tasks, update, cfg.eval_examples,
                )
            llm.sleep(1)

        # 4. LOG
        log_data = {
            **reward_stats,
            **advantage_stats,
            **algo_metrics,
            **grad_stats,
            **gpu_stats,
            **eval_metrics,
            "time/rollout_seconds": rollout_time,
            "time/train_seconds": grad_time,
            "time/total_step_seconds": iter_time,
            "rollouts": len(rollouts),
        }

        if use_wandb:
            wandb_data = dict(log_data)
            if eval_table is not None:
                wandb_data["examples/letter_counting"] = eval_table
            wandb.log(wandb_data, step=update)

        # Console output
        eval_str = ""
        if eval_metrics:
            eval_str = " | " + " ".join(
                f"{k.split('/')[-1]}={v:.3f}" for k, v in eval_metrics.items()
                if "accuracy" in k
            )

        print(
            f"update={update:3d} "
            f"reward={reward_stats['reward/mean']:.3f} "
            f"frac_correct={reward_stats['reward/fraction_correct']:.2f} "
            f"loss={algo_metrics.get('loss/policy', 0):.4f} "
            f"grad_norm={grad_stats['grad/norm']:.2f} "
            f"rollout_s={rollout_time:.1f} grad_s={grad_time:.1f}"
            f"{eval_str}"
        )

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
