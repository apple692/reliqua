"""Tests for miner generation prompt augmentation (canonical rebind)."""

from reliquary.shared.prompt_augment import (
    completion_token_budget,
    default_max_model_len,
    generation_limits,
    generation_prompt_text,
    generation_prompt_tokens,
    max_new_tokens_for_generation,
    rebind_rollouts_to_canonical_prompt,
    soft_target_tokens,
)


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


class _ChatTokenizer(_FakeTokenizer):
    chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        add_special_tokens: bool = False,
    ):
        parts = [m["content"] for m in messages]
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        text = " ".join(parts)
        return self.encode(text, add_special_tokens=add_special_tokens)


def test_generation_prompt_text_includes_soft_target():
    original = "Solve 2+2."
    tok = _FakeTokenizer()
    text = generation_prompt_text(original, tok, max_total=100)
    target = soft_target_tokens(original, tok, max_total=100)
    assert str(target) in text
    assert "\\boxed{your answer}" in text
    assert "end your message immediately" in text


def test_max_new_tokens_uses_canonical_protocol_room():
    original = "word " * 10
    tok = _FakeTokenizer()
    limits = generation_limits(original, tok, max_total=100)
    assert limits["max_new_tokens"] == 100 - limits["canonical_len"]
    assert limits["max_new_tokens"] == max_new_tokens_for_generation(
        original, tok, max_total=100,
    )
    # Cap-hit submitted total reaches protocol cap after rebind.
    assert (
        limits["canonical_len"] + limits["max_new_tokens"]
        == 100
    )


def test_prefix_overhead_and_default_max_model_len():
    original = "Solve 2+2."
    tok = _FakeTokenizer()
    limits = generation_limits(original, tok, max_total=8192)
    assert limits["prefix_overhead"] == (
        limits["augmented_len"] - limits["canonical_len"]
    )
    assert limits["required_max_model_len"] == 8192 + limits["prefix_overhead"]
    assert default_max_model_len() >= 8192


def test_chat_template_increases_augmented_prefix():
    original = "Solve 2+2."
    plain = generation_prompt_tokens(original, _FakeTokenizer(), max_total=200)
    chat = generation_prompt_tokens(original, _ChatTokenizer(), max_total=200)
    assert len(chat) >= len(plain)


def test_soft_target_capped_by_protocol_room():
    original = "x " * 80
    tok = _FakeTokenizer()
    target = soft_target_tokens(original, tok, max_total=100)
    ceiling = completion_token_budget(original, tok, max_total=100)
    assert target <= ceiling
    assert target >= 1


def test_rebind_swaps_prefix_keeps_completion():
    canonical = [10, 11, 12]
    augmented = [10, 11, 12, 20, 21]
    rollouts = [
        {"tokens": augmented + [100, 101, 102], "prompt_length": len(augmented)},
        {"tokens": augmented + [200], "prompt_length": len(augmented)},
    ]
    rebound = rebind_rollouts_to_canonical_prompt(
        rollouts, canonical, len(augmented),
    )
    assert rebound[0]["tokens"] == canonical + [100, 101, 102]
    assert rebound[0]["prompt_length"] == len(canonical)
    assert rebound[1]["tokens"] == canonical + [200]
    assert rebound[1]["prompt_length"] == len(canonical)
