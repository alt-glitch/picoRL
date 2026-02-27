"""
picoRL scratch pad -- Sid's learning sandbox.

Implement 5 functions, test each one inline, answer checkpoint questions.
Work through READING.md alongside this file.

Run with: uv run python scratch.py
Each function has a test block that runs when you uncomment it at the bottom.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# =============================================================================
# FUNCTION 1: get_per_token_logprobs
# =============================================================================
#
# GOAL: Given a model and input_ids, compute the log-probability of each token
#       given all preceding tokens.
#
# READING: See READING.md "Function 1" section. Key references:
#   - HuggingFace Qwen2 source: search for "shift_logits"
#   - PyTorch torch.gather docs
#   - GyLLM: gyllm/packages/nanorl/src/nanorl/rl/grpo.py lines 93-100
#   - Ludic: ludic/src/ludic/training/loss.py selective_log_softmax
#
# STEPS:
#   1. Forward pass: logits = model(input_ids).logits          -> [B, T, V]
#   2. Shift: logits[:, :-1, :] predicts input_ids[:, 1:]      -> [B, T-1, V] and [B, T-1]
#   3. Log-softmax: log_softmax(shifted_logits, dim=-1)         -> [B, T-1, V]
#   4. Gather: pick the logprob of the actual next token        -> [B, T-1]
#
# RETURNS: Tensor of shape [B, T-1] -- per-token log-probabilities.
#          token_logprobs[b, t] = log P(input_ids[b, t+1] | input_ids[b, :t+1])
#


def get_per_token_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,  # [B, T]
) -> torch.Tensor:  # [B, T-1]
    """Compute per-token log-probabilities for a batch of sequences.

    The causal LM shift: logit at position t predicts token at position t+1.
    So we align logits[:, :-1] with input_ids[:, 1:].
    """
    model.forward(input_ids)
    # TODO: Implement this function
    # Hint: 4 lines of code. See STEPS above.
    raise NotImplementedError("Implement get_per_token_logprobs")


def test_get_per_token_logprobs():
    """Test: logprob(" 4" | "2+2=") should be higher than logprob(" 7" | "2+2=")."""
    print("=" * 60)
    print("TEST 1: get_per_token_logprobs")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    # Encode two sequences: "2+2= 4" and "2+2= 7"
    correct = tokenizer("2+2= 4", return_tensors="pt")["input_ids"]
    wrong = tokenizer("2+2= 7", return_tensors="pt")["input_ids"]

    with torch.no_grad():
        logprobs_correct = get_per_token_logprobs(model, correct)
        logprobs_wrong = get_per_token_logprobs(model, wrong)

    # The last logprob is for the final token (" 4" vs " 7")
    lp_4 = logprobs_correct[0, -1].item()
    lp_7 = logprobs_wrong[0, -1].item()

    print(f"  log P(' 4' | '2+2=') = {lp_4:.4f}")
    print(f"  log P(' 7' | '2+2=') = {lp_7:.4f}")
    print(
        f"  P(' 4') / P(' 7') = {torch.exp(torch.tensor(lp_4 - lp_7)).item():.2f}x more likely"
    )
    assert lp_4 > lp_7, (
        f"Expected ' 4' to be more likely than ' 7', got {lp_4:.4f} vs {lp_7:.4f}"
    )
    print("  PASSED\n")

    # Also verify shapes
    batch = torch.stack([correct[0], wrong[0]])  # [2, T]
    with torch.no_grad():
        logprobs = get_per_token_logprobs(model, batch)
    assert logprobs.shape == (2, batch.shape[1] - 1), (
        f"Shape mismatch: {logprobs.shape}"
    )
    print(f"  Shape check: input {batch.shape} -> logprobs {logprobs.shape}  PASSED\n")

    # CHECKPOINT QUESTIONS (answer these before moving on):
    # 1. Why do we shift logits by 1? What would happen if we didn't?
    # 2. What is the shape of model(input_ids).logits and why?
    # 3. Why log_softmax instead of just softmax? (Hint: numerical stability + gradient)
    # 4. What does torch.gather do? Draw the index selection for a 2x3 example.
    # 5. Why is the output shape [B, T-1] and not [B, T]?


# =============================================================================
# FUNCTION 2: tokenize_with_assistant_mask
# =============================================================================
#
# GOAL: Given a tokenizer and a multi-turn conversation, produce input_ids and
#       a binary mask that is 1 for assistant tokens and 0 for everything else.
#
# READING: See READING.md "Function 2" section. Key references:
#   - HuggingFace chat templates docs
#   - GyLLM: gyllm/packages/nanorl/src/nanorl/rl/reinforce.py lines 9-40
#
# WHY INCREMENTAL: Chat templates add special tokens (role markers, separators)
#   that make it impossible to know token boundaries from the text alone. By
#   tokenizing progressively longer prefixes and diffing lengths, we reliably
#   find each message's span regardless of template format.
#
# STEPS:
#   1. Tokenize full conversation: apply_chat_template(messages) -> input_ids
#   2. For each message at index i:
#      a. Tokenize prefix messages[:i+1] -> get length
#      b. Span of message i = [prev_len:curr_len]
#      c. If role == "assistant", set mask[prev_len:curr_len] = 1
#      d. Update prev_len = curr_len
#
# RETURNS: (input_ids: Tensor[T], assistant_mask: Tensor[T])
#          Both 1-D tensors. mask[t] = 1 iff token t belongs to an assistant turn.
#


def tokenize_with_assistant_mask(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a conversation and create a mask for assistant tokens.

    Uses incremental tokenization: encode messages[:1], messages[:2], etc.
    and diff lengths to find where each message's tokens start and end.
    """
    # TODO: Implement this function
    # Hint: ~15 lines. See STEPS above.
    # Helper for encoding a message list:
    #   tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_tensors="pt")
    raise NotImplementedError("Implement tokenize_with_assistant_mask")


def test_tokenize_with_assistant_mask():
    """Test: verify mask aligns with assistant turns in a multi-turn conversation."""
    print("=" * 60)
    print("TEST 2: tokenize_with_assistant_mask")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]

    input_ids, mask = tokenize_with_assistant_mask(tokenizer, messages)

    # Print tokens aligned with mask for visual inspection
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    print("  Token alignment:")
    for i, (tok, m) in enumerate(zip(tokens, mask.tolist())):
        marker = "<<<" if m == 1 else ""
        print(f"    [{i:3d}] mask={m} | {tok} {marker}")

    # Verify: mask should have some 1s (assistant tokens exist)
    assert mask.sum() > 0, "Mask has no assistant tokens!"

    # Verify: mask should have some 0s (non-assistant tokens exist)
    assert (mask == 0).sum() > 0, "Mask has no non-assistant tokens!"

    # Verify: shapes match
    assert input_ids.shape == mask.shape, (
        f"Shape mismatch: {input_ids.shape} vs {mask.shape}"
    )

    print(f"\n  Total tokens: {len(input_ids)}")
    print(f"  Assistant tokens: {mask.sum().item()}")
    print(f"  Non-assistant tokens: {(mask == 0).sum().item()}")
    print("  PASSED\n")

    # CHECKPOINT QUESTIONS:
    # 1. Why incremental tokenization instead of regex or sentinel tokens?
    # 2. What happens with BOS/EOS tokens? Are they part of any message's span?
    # 3. Does the mask align with the shifted logprobs from Function 1?
    #    (Hint: if mask[t]=1, the logprob for token t is at logprobs[t-1].
    #     So for the loss, we need mask[1:] to align with logprobs.)
    # 4. What's the time complexity of this approach? Is it a problem?


# =============================================================================
# FUNCTION 3: compute_grpo_advantages
# =============================================================================
#
# GOAL: Given a flat list of rewards and the group size k, compute per-rollout
#       advantages using the GRPO formula: advantage_i = reward_i - mean(group).
#
# READING: See READING.md "Function 3" section. Key references:
#   - GRPO paper (DeepSeekMath) Section 3.1
#   - GyLLM: gyllm/packages/nanorl/src/nanorl/rl/grpo.py lines 21-70
#   - Ludic: ludic/src/ludic/training/credit_assignment.py lines 14-97
#
# STEPS:
#   1. Reshape rewards into groups: [r1, r2, r3, r4] with k=2 -> [[r1,r2], [r3,r4]]
#   2. Per group: mean_j = mean(group_j)
#   3. Per rollout: advantage_i = reward_i - mean_j (where j is i's group)
#   4. Optional: normalize by std(all_advantages) + eps
#
# RETURNS: Tensor of advantages, same length as input rewards.
#


def compute_grpo_advantages(
    rewards: torch.Tensor,  # [N] flat reward tensor
    group_size: int,  # k completions per prompt
    normalize: bool = False,  # divide by std + eps
) -> torch.Tensor:  # [N] advantages
    """Compute GRPO advantages: advantage_i = reward_i - mean(group_i).

    The "group-relative" part: each prompt's completions are compared only
    to each other, not to the whole batch. This means a hard prompt where
    all completions fail (reward=0) gets advantage=0 (no gradient), rather
    than negative advantage (which would penalize for prompt difficulty).
    """
    # TODO: Implement this function
    # Hint: 5-8 lines. reshape -> mean -> subtract -> optional normalize.
    raise NotImplementedError("Implement compute_grpo_advantages")


def test_compute_grpo_advantages():
    """Test: known input/output pairs for advantage computation."""
    print("=" * 60)
    print("TEST 3: compute_grpo_advantages")
    print("=" * 60)

    # Test 1: Simple case
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    advantages = compute_grpo_advantages(rewards, group_size=2)
    expected = torch.tensor([0.5, -0.5, 0.5, -0.5])
    assert torch.allclose(advantages, expected), (
        f"Expected {expected}, got {advantages}"
    )
    print(
        f"  Test 1: rewards={rewards.tolist()} -> advantages={advantages.tolist()}  PASSED"
    )

    # Test 2: All same reward in a group -> advantage = 0
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    advantages = compute_grpo_advantages(rewards, group_size=2)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(advantages, expected), (
        f"Expected {expected}, got {advantages}"
    )
    print(
        f"  Test 2: rewards={rewards.tolist()} -> advantages={advantages.tolist()}  PASSED"
    )

    # Test 3: Group size = 4
    rewards = torch.tensor([4.0, 2.0, 1.0, 1.0])
    advantages = compute_grpo_advantages(rewards, group_size=4)
    mean_r = 2.0  # (4+2+1+1)/4
    expected = torch.tensor([2.0, 0.0, -1.0, -1.0])
    assert torch.allclose(advantages, expected), (
        f"Expected {expected}, got {advantages}"
    )
    print(
        f"  Test 3: rewards={rewards.tolist()} -> advantages={advantages.tolist()}  PASSED"
    )

    # Test 4: With normalization
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    advantages = compute_grpo_advantages(rewards, group_size=2, normalize=True)
    # After normalization: mean=0, std=0.5 -> advantages / 0.5 = [1, -1, 1, -1]
    expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
    assert torch.allclose(advantages, expected, atol=1e-5), (
        f"Expected {expected}, got {advantages}"
    )
    print(f"  Test 4 (normalized): advantages={advantages.tolist()}  PASSED")

    print()

    # CHECKPOINT QUESTIONS:
    # 1. Why group-relative instead of batch-mean baseline?
    #    (Hint: what happens to a hard prompt where all k completions score 0?)
    # 2. What happens when all rewards in a group are equal? Is that correct behavior?
    # 3. How does group_size (k) affect the variance of the gradient estimate?
    #    (Hint: larger k = better baseline estimate, but more compute per prompt)
    # 4. What's the connection between this and a value function baseline in PPO?


# =============================================================================
# FUNCTION 4: grpo_loss
# =============================================================================
#
# GOAL: Compute the GRPO policy gradient loss:
#       loss = -(advantages * per_sequence_logprob_sum).mean()
#
# READING: See READING.md "Function 4" section. Key references:
#   - Sutton & Barto Ch 13.3 (REINFORCE)
#   - Spinning Up: Policy Gradient
#   - GyLLM: gyllm/packages/nanorl/src/nanorl/rl/grpo.py lines 73-121
#   - Ludic: ludic/src/ludic/training/loss.py ReinforceLoss
#
# STEPS:
#   1. Mask logprobs to assistant tokens: masked = logprobs * mask
#   2. Sum over token dimension: seq_logprob = masked.sum(dim=-1)   -> [B]
#   3. Weight by advantages: weighted = advantages * seq_logprob     -> [B]
#   4. Negate and average: loss = -weighted.mean()                   -> scalar
#
# WHY NEGATIVE: We MINIMIZE the loss. Positive advantage should INCREASE
#   logprob (make the action more likely). increase logprob -> negative
#   contribution to loss -> minimizing drives logprob up. Hence the negation.
#
# RETURNS: Scalar loss tensor (with grad).
#


def grpo_loss(
    per_token_logprobs: torch.Tensor,  # [B, T-1] from get_per_token_logprobs
    advantages: torch.Tensor,  # [B] from compute_grpo_advantages
    assistant_mask: torch.Tensor,  # [B, T-1] shifted mask (1 for assistant tokens)
) -> torch.Tensor:  # scalar
    """Compute GRPO policy gradient loss.

    loss = -(advantages * sum_of_masked_logprobs).mean()

    This is REINFORCE with group-relative advantages. The "group" part is in
    how advantages are computed (Function 3), not in the loss itself.
    """
    # TODO: Implement this function
    # Hint: 3 lines. mask -> sum -> negate & average.
    raise NotImplementedError("Implement grpo_loss")


def test_grpo_loss():
    """Test: gradient properties of the loss function."""
    print("=" * 60)
    print("TEST 4: grpo_loss")
    print("=" * 60)

    # Create fake logprobs (requires grad so we can check gradients)
    logprobs = torch.tensor(
        [[-1.0, -2.0, -0.5], [-1.5, -1.0, -2.0]], requires_grad=True
    )
    advantages = torch.tensor([1.0, -0.5])
    mask = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.float32)

    loss = grpo_loss(logprobs, advantages, mask)

    # Check it's a scalar
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"  Loss value: {loss.item():.4f}")

    # Check backward works
    loss.backward()
    assert logprobs.grad is not None, "No gradients computed!"
    print(f"  Gradients: {logprobs.grad}")

    # With positive advantage, gradient should push masked logprobs UP
    # (gradient of logprobs should be negative, because optimizer does param -= lr * grad)
    # Actually: d(loss)/d(logprob) = -advantage * mask / B
    # For positive advantage: gradient is negative -> optimizer increases logprob. Correct!
    grad_row0 = logprobs.grad[0]
    print(f"  Row 0 (adv=+1.0): grad = {grad_row0.tolist()}")
    assert grad_row0[0] == 0.0, "Unmasked token should have zero gradient"
    assert grad_row0[1] < 0.0, (
        "Positive advantage should give negative gradient (increases logprob)"
    )

    # With negative advantage, gradient should push masked logprobs DOWN
    grad_row1 = logprobs.grad[1]
    print(f"  Row 1 (adv=-0.5): grad = {grad_row1.tolist()}")
    assert grad_row1[0] == 0.0, "Unmasked token should have zero gradient"
    assert grad_row1[2] > 0.0, (
        "Negative advantage should give positive gradient (decreases logprob)"
    )

    # Test: zero advantage -> zero gradient
    logprobs2 = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    advantages2 = torch.tensor([0.0])
    mask2 = torch.tensor([[1, 1]], dtype=torch.float32)
    loss2 = grpo_loss(logprobs2, advantages2, mask2)
    loss2.backward()
    assert torch.allclose(logprobs2.grad, torch.zeros_like(logprobs2.grad)), (
        "Zero advantage should give zero gradient"
    )
    print("  Zero advantage -> zero gradient  PASSED")

    print("  PASSED\n")

    # CHECKPOINT QUESTIONS:
    # 1. Why the negative sign? Trace through: positive advantage, what happens to loss,
    #    what happens to gradient, what does the optimizer do?
    # 2. Why sum over tokens instead of mean? What would change with mean?
    # 3. What happens if advantage=0 for all rollouts? (Hint: no learning. Is that right?)
    # 4. What's the connection to REINFORCE? Write out the REINFORCE gradient formula
    #    and show how this loss gives the same gradient.
    # 5. Where does the "group" part of GRPO come in? (Hint: it's in Function 3, not here.)


# =============================================================================
# FUNCTION 5: Mini training loop
# =============================================================================
#
# GOAL: Wire functions 1-4 together into a tiny training loop.
#       Use raw HF model (no vLLM), a dummy "repeat after me" env,
#       and verify that reward goes up over 10 gradient steps.
#
# READING: See READING.md "Function 5" section. Key references:
#   - GyLLM training script: gyllm/scripts/train_grpo_agent.py
#   - PyTorch autograd tutorial
#
# STEPS:
#   1. Load model + tokenizer
#   2. For each iteration:
#      a. Generate k completions per prompt (model.generate, no grad)
#      b. Score each completion (reward = overlap with target)
#      c. Compute advantages (Function 3)
#      d. For each completion:
#         - Tokenize the full conversation (Function 2)
#         - Get per-token logprobs (Function 1)
#         - Accumulate loss (Function 4)
#      e. loss.backward()
#      f. optimizer.step(), optimizer.zero_grad()
#   3. Print reward per iteration, verify upward trend
#
# NOTE: This uses model.generate() which is SLOW. The real pipeline uses vLLM.
#       The point is to understand the data flow, not to train efficiently.
#


def mini_training_loop():
    """Tiny training loop: generate -> score -> advantages -> loss -> backward -> step.

    Uses a "repeat after me" task: the model gets a word and must repeat it.
    Reward = 1.0 if the response contains the target word, else 0.0.
    """
    print("=" * 60)
    print("TEST 5: Mini training loop")
    print("=" * 60)

    # TODO: Implement this function
    # This is the most open-ended one. Here's the skeleton:
    #
    # 1. Load model and tokenizer
    #    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    #    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    #    tokenizer.pad_token = tokenizer.eos_token
    #    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    #
    # 2. Define the task
    #    target_words = ["banana", "elephant", "rainbow"]
    #    group_size = 2  # k=2 completions per prompt
    #
    # 3. Training loop (10 iterations):
    #    for iteration in range(10):
    #        all_rewards = []
    #        all_conversations = []  # list of message lists
    #
    #        # --- ROLLOUT PHASE (no grad) ---
    #        with torch.no_grad():
    #            for word in target_words:
    #                messages_prefix = [
    #                    {"role": "system", "content": "Repeat the word the user says. Say ONLY that word."},
    #                    {"role": "user", "content": word},
    #                ]
    #                prompt = tokenizer.apply_chat_template(
    #                    messages_prefix, tokenize=False, add_generation_prompt=True
    #                )
    #                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    #
    #                for _ in range(group_size):
    #                    output_ids = model.generate(
    #                        input_ids,
    #                        max_new_tokens=20,
    #                        temperature=0.7,
    #                        do_sample=True,
    #                        pad_token_id=tokenizer.pad_token_id,
    #                    )
    #                    response = tokenizer.decode(
    #                        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    #                    )
    #                    reward = 1.0 if word.lower() in response.lower() else 0.0
    #
    #                    full_messages = messages_prefix + [
    #                        {"role": "assistant", "content": response}
    #                    ]
    #                    all_conversations.append(full_messages)
    #                    all_rewards.append(reward)
    #
    #        # --- ADVANTAGE COMPUTATION ---
    #        rewards_tensor = torch.tensor(all_rewards)
    #        advantages = compute_grpo_advantages(rewards_tensor, group_size=group_size)
    #
    #        # --- TRAINING PHASE ---
    #        optimizer.zero_grad()
    #        total_loss = torch.tensor(0.0)
    #
    #        for conv, adv in zip(all_conversations, advantages):
    #            input_ids, mask = tokenize_with_assistant_mask(tokenizer, conv)
    #            logprobs = get_per_token_logprobs(model, input_ids.unsqueeze(0))
    #            shifted_mask = mask[1:].unsqueeze(0).float()
    #
    #            loss = grpo_loss(logprobs, adv.unsqueeze(0), shifted_mask)
    #            total_loss = total_loss + loss
    #
    #        avg_loss = total_loss / len(all_conversations)
    #        avg_loss.backward()
    #        optimizer.step()
    #
    #        avg_reward = rewards_tensor.mean().item()
    #        print(f"  Iter {iteration}: reward={avg_reward:.2f}, loss={avg_loss.item():.4f}")
    #
    # 4. Check that reward trended upward
    #    print("  (Check: did reward improve over iterations?)")
    #    print("  DONE\n")

    raise NotImplementedError("Implement mini_training_loop")

    # CHECKPOINT QUESTIONS:
    # 1. What's the full data flow from prompt to gradient update? Trace a single rollout.
    # 2. Where do gradients flow and where are they stopped?
    #    (Hint: generation is no_grad. Only the training forward pass has grad.)
    # 3. Why do we need torch.no_grad() during generation?
    #    (Hint: we don't want to build a compute graph for the rollout.)
    # 4. Why is the generation forward pass different from the training forward pass?
    #    (Generation: autoregressive, one token at a time. Training: full sequence, teacher-forced.)
    # 5. What would happen if we used the same forward pass for both?


# =============================================================================
# RUN TESTS
# =============================================================================
# Uncomment each test as you implement the corresponding function.
# They're sequential -- each builds on the previous one.

if __name__ == "__main__":
    # test_get_per_token_logprobs()
    # test_tokenize_with_assistant_mask()
    # test_compute_grpo_advantages()
    # test_grpo_loss()
    # mini_training_loop()
    print(
        "All tests are commented out. Uncomment them one at a time as you implement each function."
    )
    print("Start with test_get_per_token_logprobs() -- see Function 1 above.")
