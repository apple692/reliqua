"""Tests for generated_solution length filtering during prompt selection."""

import random

import pytest

from reliquary.miner.engine import pick_prompt_idx
from reliquary.miner.prompt_selection import (
    DEFAULT_MAX_GENERATED_SOLUTION_TOKENS_FOR_PICK,
    passes_generated_solution_length_filter,
)


class _FakeTokenizer:
    name_or_path = "fake"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text)))


class _SolutionLengthEnv:
    def __init__(self, lengths: dict[int, int]) -> None:
        self._lengths = lengths

    def __len__(self) -> int:
        return max(self._lengths) + 1 if self._lengths else 0

    def get_problem(self, index: int) -> dict:
        return {
            "prompt": f"question {index}",
            "ground_truth": "1",
            "generated_solution": "x" * self._lengths[index],
            "id": f"id{index}",
        }

    def get_generated_solution_token_length(self, index: int, tokenizer) -> int:
        return self._lengths[index]


def test_passes_filter_when_solution_below_limit():
    env = _SolutionLengthEnv({0: 100, 1: 499})
    tok = _FakeTokenizer()
    assert passes_generated_solution_length_filter(env, 0, tok, max_tokens=500)
    assert passes_generated_solution_length_filter(env, 1, tok, max_tokens=500)


def test_rejects_when_solution_at_or_above_limit():
    env = _SolutionLengthEnv({0: 500, 1: 800})
    tok = _FakeTokenizer()
    assert not passes_generated_solution_length_filter(env, 0, tok, max_tokens=500)
    assert not passes_generated_solution_length_filter(env, 1, tok, max_tokens=500)


def test_filter_disabled_without_tokenizer():
    env = _SolutionLengthEnv({0: 10_000})
    assert passes_generated_solution_length_filter(env, 0, None, max_tokens=500)


def test_pick_prompt_idx_skips_long_generated_solution():
    env = _SolutionLengthEnv({0: 900, 1: 200, 2: 100})
    rng = random.Random(0)
    idx = pick_prompt_idx(
        env,
        cooldown_prompts=set(),
        rng=rng,
        tokenizer=_FakeTokenizer(),
        max_generated_solution_tokens=500,
    )
    assert idx in {1, 2}


def test_pick_prompt_idx_raises_when_all_solutions_too_long():
    env = _SolutionLengthEnv({0: 600, 1: 700})
    rng = random.Random(0)
    with pytest.raises(RuntimeError, match="no eligible prompt"):
        pick_prompt_idx(
            env,
            cooldown_prompts=set(),
            rng=rng,
            tokenizer=_FakeTokenizer(),
            max_generated_solution_tokens=500,
            max_attempts=20,
        )


def test_default_limit_is_500():
    assert DEFAULT_MAX_GENERATED_SOLUTION_TOKENS_FOR_PICK == 500
