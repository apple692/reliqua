"""Per-rollout INFO logging for staged miner bundles (validator/canonical tokens)."""

from __future__ import annotations

import logging
import os
from typing import Any

from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

# Truncation limits for miner logs (override with env vars if needed).
_LOG_PROMPT_CHARS = int(os.getenv("MINER_ROLLOUT_LOG_PROMPT_CHARS", "800"))
_LOG_RESPONSE_CHARS = int(os.getenv("MINER_ROLLOUT_LOG_RESPONSE_CHARS", "1200"))
_LOG_TAIL_CHARS = int(os.getenv("MINER_ROLLOUT_LOG_TAIL_CHARS", "160"))


def _preview(text: str, limit: int) -> str:
    text = (text or "").replace("\n", "\\n")
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def log_staged_rollouts(
    logger: logging.Logger,
    *,
    prompt_idx: int,
    problem: dict[str, Any],
    generations: list[dict],
    rewards: list[float],
    sigma: float,
    tokenizer,
    eos_set: set[int],
    source: str,
    outcome: str = "",
) -> None:
    """Log canonical prompt + each rollout response/lengths (submitted format)."""
    prompt_text = str(problem.get("prompt", ""))
    problem_id = problem.get("id", "?")
    ground_truth = problem.get("ground_truth")
    outcome_label = outcome or "eval"

    logger.info(
        "rollout_detail begin prompt=%d source=%s outcome=%s σ=%.3f "
        "problem_id=%s ground_truth=%r n_rollouts=%d",
        prompt_idx, source, outcome_label, sigma, problem_id, ground_truth,
        len(generations),
    )
    plen_ref = int(generations[0].get("prompt_length", 0)) if generations else 0
    logger.info(
        "rollout_detail prompt=%d canonical_prompt_len=%d prompt_text=%s",
        prompt_idx,
        plen_ref,
        _preview(prompt_text, _LOG_PROMPT_CHARS),
    )

    for i, gen in enumerate(generations):
        tokens = list(gen.get("tokens") or [])
        plen = int(gen.get("prompt_length", 0))
        completion_tokens = tokens[plen:]
        clen = len(completion_tokens)
        total = len(tokens)
        last = int(tokens[-1]) if tokens else -1
        in_eos = last in eos_set if eos_set else False
        reward = float(rewards[i]) if i < len(rewards) else 0.0

        try:
            response_text = tokenizer.decode(
                completion_tokens, skip_special_tokens=False,
            )
        except Exception as exc:
            response_text = f"<decode error: {exc}>"

        try:
            tail_slice = (
                completion_tokens[-20:]
                if len(completion_tokens) > 20
                else completion_tokens
            )
            tail = _preview(
                tokenizer.decode(tail_slice, skip_special_tokens=False),
                _LOG_TAIL_CHARS,
            )
        except Exception:
            tail = "?"

        logger.info(
            "rollout_detail prompt=%d #%d reward=%.1f "
            "prompt_len=%d completion_len=%d total_len=%d "
            "last_tok=%d in_eos=%s truncated=%s "
            "at_protocol_cap=%s tokens_to_cap=%d tail=%r",
            prompt_idx,
            i,
            reward,
            plen,
            clen,
            total,
            last,
            in_eos,
            not in_eos if eos_set else False,
            total >= MAX_NEW_TOKENS_PROTOCOL_CAP,
            MAX_NEW_TOKENS_PROTOCOL_CAP - total,
            tail,
        )
        logger.info(
            "rollout_detail prompt=%d #%d response=%s",
            prompt_idx,
            i,
            _preview(response_text, _LOG_RESPONSE_CHARS),
        )

    logger.info(
        "rollout_detail end prompt=%d source=%s outcome=%s rewards=%s",
        prompt_idx,
        source,
        outcome_label,
        [int(r) for r in rewards],
    )
