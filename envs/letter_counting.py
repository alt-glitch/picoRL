from __future__ import annotations

import json
import random
import re
import string

from picorl.env import Env
from picorl.types import Message

# =============================================================================
# Difficulty tiers (ported from Atropos)
# Tiers 1-5: real English words, tiers 6-10: random letter strings
# =============================================================================
DIFFICULTY_TIERS = {
    1: {
        "min_word_length": 3,
        "max_word_length": 8,
        "multi_letter_probability": 0.0,
        "min_letters_to_count": 1,
        "max_letters_to_count": 1,
        "use_random_string": False,
    },
    2: {
        "min_word_length": 5,
        "max_word_length": 12,
        "multi_letter_probability": 0.5,
        "min_letters_to_count": 1,
        "max_letters_to_count": 2,
        "use_random_string": False,
    },
    3: {
        "min_word_length": 8,
        "max_word_length": 18,
        "multi_letter_probability": 0.6,
        "min_letters_to_count": 2,
        "max_letters_to_count": 3,
        "use_random_string": False,
    },
    4: {
        "min_word_length": 10,
        "max_word_length": 25,
        "multi_letter_probability": 0.8,
        "min_letters_to_count": 3,
        "max_letters_to_count": 5,
        "use_random_string": False,
    },
    5: {
        "min_word_length": 15,
        "max_word_length": 35,
        "multi_letter_probability": 0.9,
        "min_letters_to_count": 5,
        "max_letters_to_count": 8,
        "use_random_string": False,
    },
    6: {
        "min_word_length": 40,
        "max_word_length": 80,
        "multi_letter_probability": 1.0,
        "min_letters_to_count": 8,
        "max_letters_to_count": 12,
        "use_random_string": True,
    },
    7: {
        "min_word_length": 80,
        "max_word_length": 150,
        "multi_letter_probability": 1.0,
        "min_letters_to_count": 12,
        "max_letters_to_count": 20,
        "use_random_string": True,
    },
    8: {
        "min_word_length": 150,
        "max_word_length": 250,
        "multi_letter_probability": 1.0,
        "min_letters_to_count": 18,
        "max_letters_to_count": 30,
        "use_random_string": True,
    },
    9: {
        "min_word_length": 250,
        "max_word_length": 400,
        "multi_letter_probability": 1.0,
        "min_letters_to_count": 25,
        "max_letters_to_count": 40,
        "use_random_string": True,
    },
    10: {
        "min_word_length": 400,
        "max_word_length": 500,
        "multi_letter_probability": 1.0,
        "min_letters_to_count": 35,
        "max_letters_to_count": 50,
        "use_random_string": True,
    },
}

# Module-level word cache (loaded once from NLTK)
_WORDS_CACHE: list[str] | None = None


def _load_words() -> list[str]:
    """Load English words from NLTK, filtered to alphabetic-only, 3-35 chars."""
    global _WORDS_CACHE
    if _WORDS_CACHE is not None:
        return _WORDS_CACHE

    import nltk
    from nltk.corpus import words

    try:
        word_list = words.words()
    except LookupError:
        nltk.download("words", quiet=True)
        word_list = words.words()

    _WORDS_CACHE = [
        w.lower()
        for w in word_list
        if w.isalpha() and 3 <= len(w) <= 35
    ]
    random.shuffle(_WORDS_CACHE)
    return _WORDS_CACHE


def _generate_random_string(min_length: int, max_length: int) -> str:
    """Generate a random lowercase letter string."""
    length = random.randint(min_length, max_length)
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def _select_target_letters(text: str, num_letters: int) -> list[str]:
    """Select target letters with randomized presence bias (30-90%)."""
    all_letters = list(string.ascii_lowercase)
    text_lower = text.lower()
    present = [ch for ch in all_letters if ch in text_lower]
    absent = [ch for ch in all_letters if ch not in text_lower]

    present_bias = random.uniform(0.3, 0.9)
    selected: list[str] = []
    present_pool = present.copy()
    absent_pool = absent.copy()

    for _ in range(num_letters):
        if not present_pool and not absent_pool:
            break
        elif not present_pool:
            effective_bias = 0.0
        elif not absent_pool:
            effective_bias = 1.0
        else:
            effective_bias = present_bias

        if random.random() < effective_bias and present_pool:
            chosen = random.choice(present_pool)
            present_pool.remove(chosen)
        elif absent_pool:
            chosen = random.choice(absent_pool)
            absent_pool.remove(chosen)
        elif present_pool:
            chosen = random.choice(present_pool)
            present_pool.remove(chosen)
        else:
            break

        selected.append(chosen)

    return selected


def _extract_answer(text: str, expected_format: str) -> int | dict | None:
    """Extract answer from <answer>...</answer> tags.

    Args:
        text: Model response text.
        expected_format: "single" for int, "multi" for JSON dict.

    Returns:
        Parsed answer or None if format is invalid.
    """
    if expected_format == "single":
        matches = re.findall(r"<answer>\s*(\d+)\s*</answer>", text, re.IGNORECASE)
        if len(matches) != 1:
            return None
        return int(matches[0])
    else:
        matches = re.findall(r"<answer>\s*(\{[^}]+\})\s*</answer>", text, re.IGNORECASE)
        if len(matches) != 1:
            return None
        try:
            answer_dict = json.loads(matches[0])
            if not isinstance(answer_dict, dict):
                return None
            for key, value in answer_dict.items():
                if not isinstance(key, str) or not isinstance(value, int):
                    return None
            return answer_dict
        except (json.JSONDecodeError, ValueError):
            return None


class LetterCountingEnv(Env):
    """Count letter occurrences in words/strings. Single-turn, binary reward.

    Uses 10 difficulty tiers from Atropos:
    - Tiers 1-5: Real English words (NLTK), increasing length + multi-letter probability
    - Tiers 6-10: Random letter strings, always multi-letter, extreme lengths
    """

    def __init__(self, difficulty: int = 1) -> None:
        self.difficulty = max(1, min(10, difficulty))
        self._expected_counts: dict[str, int] = {}
        self._target_letters: list[str] = []

    def reset(self) -> list[Message]:
        tier = DIFFICULTY_TIERS[self.difficulty]
        min_len = tier["min_word_length"]
        max_len = tier["max_word_length"]

        # Pick text
        if tier["use_random_string"]:
            text = _generate_random_string(min_len, max_len)
        else:
            words = _load_words()
            candidates = [w for w in words if min_len <= len(w) <= max_len]
            text = random.choice(candidates) if candidates else _generate_random_string(min_len, max_len)

        # Decide number of letters
        if random.random() < tier["multi_letter_probability"]:
            num_letters = random.randint(
                max(2, tier["min_letters_to_count"]),
                tier["max_letters_to_count"],
            )
        else:
            num_letters = 1

        self._target_letters = _select_target_letters(text, num_letters)
        self._expected_counts = {
            letter: text.lower().count(letter)
            for letter in self._target_letters
        }

        # Build question
        if len(self._target_letters) == 1:
            letter = self._target_letters[0]
            question = (
                f"How many {letter}s are in the string {text}?\n\n"
                f"Provide your answer in the format: <answer>{{number}}</answer>"
            )
        else:
            letters_str = (
                ", ".join(f"'{l}'" for l in self._target_letters[:-1])
                + f", and '{self._target_letters[-1]}'"
            )
            example_json = (
                "{" + ", ".join(f'"{l}": 0' for l in self._target_letters) + "}"
            )
            question = (
                f"Count the occurrences of the letters {letters_str} in the string {text}\n\n"
                f"Provide your answer as JSON in the format: <answer>{example_json}</answer>"
            )

        return [{"role": "user", "content": question}]

    def step(self, response: str) -> tuple[Message | None, float, bool]:
        expected_format = "single" if len(self._target_letters) == 1 else "multi"
        answer = _extract_answer(response, expected_format)

        if answer is None:
            return None, 0.0, True

        if isinstance(answer, int):
            expected = self._expected_counts[self._target_letters[0]]
            reward = 1.0 if answer == expected else 0.0
        else:
            reward = (
                1.0
                if set(answer.keys()) == set(self._target_letters)
                and all(answer[k] == self._expected_counts[k] for k in self._target_letters)
                else 0.0
            )

        return None, reward, True
