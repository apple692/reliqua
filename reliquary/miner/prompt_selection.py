"""Miner-side prompt eligibility filters (dataset metadata, not protocol rules)."""

from __future__ import annotations

import os
from typing import Any

# Skip OpenMathInstruct rows whose reference ``generated_solution`` exceeds
# this many tokens when picking prompts. Set to 0 to disable.
DEFAULT_MAX_GENERATED_SOLUTION_TOKENS_FOR_PICK = 150


def max_generated_solution_tokens_for_pick() -> int:
    return int(
        os.environ.get(
            "RELIQUARY_MAX_GENERATED_SOLUTION_TOKENS",
            str(DEFAULT_MAX_GENERATED_SOLUTION_TOKENS_FOR_PICK),
        )
    )


def generated_solution_token_length(
    env: Any,
    index: int,
    tokenizer: Any,
) -> int | None:
    """Return token length of the dataset reference solution, if available."""
    if hasattr(env, "get_generated_solution_token_length"):
        return env.get_generated_solution_token_length(index, tokenizer)
    problem = env.get_problem(index)
    sol = problem.get("generated_solution")
    if not sol:
        return None
    return len(tokenizer.encode(str(sol), add_special_tokens=False))


def passes_generated_solution_length_filter(
    env: Any,
    index: int,
    tokenizer: Any | None,
    *,
    max_tokens: int | None = None,
) -> bool:
    """True when the row has no reference solution or its length is below *max_tokens*."""
    limit = (
        max_generated_solution_tokens_for_pick()
        if max_tokens is None
        else max_tokens
    )
    if limit <= 0 or tokenizer is None:
        return True
    n = generated_solution_token_length(env, index, tokenizer)
    if n is None:
        return True
    return n < limit
