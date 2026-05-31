"""Miner-only generation prompt augmentation.

Generation may use a chat template + instruction suffix. Submitted rollouts
always use the bare env prompt tokens (validator prompt binding unchanged).

``max_new_tokens`` is keyed to the *canonical* prefix length so cap-hit
completions reach ``MAX_NEW_TOKENS_PROTOCOL_CAP`` after rebind. vLLM
``max_model_len`` must include the generation-prefix overhead — see
:func:`default_max_model_len`.
"""

from __future__ import annotations

import logging

from reliquary.constants import (
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MINER_GENERATION_MAX_MODEL_LEN_OVERHEAD,
    MINER_GENERATION_PROMPT_SUFFIX_TEMPLATE,
    MINER_GENERATION_SOFT_TARGET_TOKENS,
    MINER_USE_CHAT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def canonical_prompt_tokens(original_prompt: str, tokenizer) -> list[int]:
    """Tokenize the env prompt exactly (matches validator canonical binding)."""
    return list(tokenizer.encode(original_prompt, add_special_tokens=False))


def completion_token_budget(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> int:
    """Max completion tokens on the submitted (canonical) sequence at protocol cap."""
    canonical_len = len(canonical_prompt_tokens(original_prompt, tokenizer))
    return max(1, max_total - canonical_len)


def soft_target_tokens(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> int:
    """Concise-response goal shown in the generation instruction block."""
    protocol_room = completion_token_budget(
        original_prompt, tokenizer, max_total=max_total,
    )
    return min(
        MINER_GENERATION_SOFT_TARGET_TOKENS,
        max(1, protocol_room - 64),
    )


def generation_instruction_text(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> str:
    """Instruction suffix (question + this = generation user content)."""
    target = soft_target_tokens(
        original_prompt, tokenizer, max_total=max_total,
    )
    return MINER_GENERATION_PROMPT_SUFFIX_TEMPLATE.format(soft_target=target)


def generation_user_content(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> str:
    """Full user-turn text for generation (question + instruction block)."""
    return original_prompt + generation_instruction_text(
        original_prompt, tokenizer, max_total=max_total,
    )


def generation_prompt_text(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> str:
    """Human-readable generation prefix (for diagnostics)."""
    if MINER_USE_CHAT_TEMPLATE and _supports_chat_template(tokenizer):
        return f"[chat user turn]\n{generation_user_content(original_prompt, tokenizer, max_total=max_total)}"
    return generation_user_content(original_prompt, tokenizer, max_total=max_total)


def _supports_chat_template(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return callable(getattr(tokenizer, "apply_chat_template", None)) and bool(template)


def _encode_plain_generation_prefix(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> list[int]:
    return list(
        tokenizer.encode(
            generation_user_content(
                original_prompt, tokenizer, max_total=max_total,
            ),
            add_special_tokens=False,
        )
    )


def _encode_chat_generation_prefix(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> list[int]:
    messages = [
        {
            "role": "user",
            "content": generation_user_content(
                original_prompt, tokenizer, max_total=max_total,
            ),
        },
    ]
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        add_special_tokens=False,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return list(token_ids)


def generation_prompt_tokens(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> list[int]:
    """Tokenize generation prefix (chat template when enabled, else plain)."""
    if MINER_USE_CHAT_TEMPLATE and _supports_chat_template(tokenizer):
        try:
            return _encode_chat_generation_prefix(
                original_prompt, tokenizer, max_total=max_total,
            )
        except Exception:
            logger.warning(
                "chat template encode failed; falling back to plain prefix",
                exc_info=True,
            )
    return _encode_plain_generation_prefix(
        original_prompt, tokenizer, max_total=max_total,
    )


def generation_prefix_overhead(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> int:
    """Extra tokens in the generation prefix vs the canonical env prompt."""
    canonical_len = len(canonical_prompt_tokens(original_prompt, tokenizer))
    augmented_len = len(
        generation_prompt_tokens(
            original_prompt, tokenizer, max_total=max_total,
        )
    )
    return max(0, augmented_len - canonical_len)


def max_new_tokens_for_generation(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> int:
    """vLLM/HF ``max_new_tokens`` — room after *canonical* prefix (post-rebind cap).

    Uses canonical length so a cap-hit completion reaches ``max_total`` on the
    submitted rollout: ``canonical_len + max_new == max_total``.
    """
    canonical_len = len(canonical_prompt_tokens(original_prompt, tokenizer))
    return max(1, max_total - canonical_len)


def required_max_model_len(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> int:
    """Minimum vLLM ``max_model_len`` for this prompt's generation prefix."""
    return max_total + generation_prefix_overhead(
        original_prompt, tokenizer, max_total=max_total,
    )


def default_max_model_len(
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    overhead: int = MINER_GENERATION_MAX_MODEL_LEN_OVERHEAD,
) -> int:
    """Default vLLM ``max_model_len`` at engine startup (worst-case overhead budget)."""
    return max_total + overhead


def generation_limits(
    original_prompt: str,
    tokenizer,
    *,
    max_total: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
) -> dict[str, int | bool]:
    """Return canonical/augmented lengths and sampling limits for diagnostics."""
    canonical_len = len(canonical_prompt_tokens(original_prompt, tokenizer))
    augmented_len = len(
        generation_prompt_tokens(
            original_prompt, tokenizer, max_total=max_total,
        )
    )
    overhead = max(0, augmented_len - canonical_len)
    max_new = max_new_tokens_for_generation(
        original_prompt, tokenizer, max_total=max_total,
    )
    return {
        "canonical_len": canonical_len,
        "augmented_len": augmented_len,
        "prefix_overhead": overhead,
        "max_new_tokens": max_new,
        "required_max_model_len": max_total + overhead,
        "soft_target_tokens": soft_target_tokens(
            original_prompt, tokenizer, max_total=max_total,
        ),
        "protocol_completion_ceiling": completion_token_budget(
            original_prompt, tokenizer, max_total=max_total,
        ),
        "uses_chat_template": (
            MINER_USE_CHAT_TEMPLATE and _supports_chat_template(tokenizer)
        ),
    }


def rebind_rollouts_to_canonical_prompt(
    rollouts: list[dict],
    canonical_prompt_tokens: list[int],
    augmented_prompt_length: int,
) -> list[dict]:
    """Swap augmented generation prefix for canonical env prompt tokens."""
    canonical = list(canonical_prompt_tokens)
    canon_len = len(canonical)
    rebound: list[dict] = []
    for rollout in rollouts:
        tokens = rollout.get("tokens") or []
        completion = list(tokens[augmented_prompt_length:])
        rebound.append({
            "tokens": canonical + completion,
            "prompt_length": canon_len,
        })
    return rebound
