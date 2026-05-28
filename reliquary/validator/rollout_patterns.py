"""Cross-rollout pattern checks for manufactured GRPO submissions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence


_EOS_MARKERS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
)
_MAX_COMPARE_CHARS = 6000
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class OppositeRewardCloneMetrics:
    suspicious: bool
    reward_vector: str
    matched_pairs: int
    mirror_pairs: int
    max_similarity: float
    mean_similarity: float
    min_length_ratio: float

    def to_log_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "suspicious": self.suspicious,
            "reward_vector": self.reward_vector,
            "matched_pairs": self.matched_pairs,
            "mirror_pairs": self.mirror_pairs,
            "max_similarity": round(self.max_similarity, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "min_length_ratio": round(self.min_length_ratio, 4),
        }


def _normalise_completion(text: str) -> str:
    text = text or ""
    for marker in _EOS_MARKERS:
        text = text.replace(marker, "")
    text = _SPACE_RE.sub(" ", text).strip().lower()
    if len(text) <= _MAX_COMPARE_CHARS:
        return text
    half = _MAX_COMPARE_CHARS // 2
    return f"{text[:half]} {text[-half:]}"


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _length_ratio(left: str, right: str) -> float:
    longest = max(len(left), len(right))
    if longest == 0:
        return 0.0
    return min(len(left), len(right)) / longest


def detect_opposite_reward_clones(
    completion_texts: Sequence[str],
    rewards: Sequence[float],
    *,
    min_pairs: int = 3,
    min_similarity: float = 0.965,
    min_length_ratio: float = 0.94,
    mirror_similarity: float = 0.985,
) -> OppositeRewardCloneMetrics:
    """Detect manufactured groups that clone reasoning across reward classes.

    The target exploit submits paired rollouts where the first copy is scored
    correct and the second copy is nearly identical except for a wrong final
    answer. This is visible only at group level: each individual rollout can
    have valid tokens, logprobs, termination and GRAIL proof.
    """

    reward_bits = ["1" if float(reward) >= 0.5 else "0" for reward in rewards]
    reward_vector = "".join(reward_bits)
    empty = OppositeRewardCloneMetrics(
        suspicious=False,
        reward_vector=reward_vector,
        matched_pairs=0,
        mirror_pairs=0,
        max_similarity=0.0,
        mean_similarity=0.0,
        min_length_ratio=0.0,
    )

    n = min(len(completion_texts), len(reward_bits))
    if n < 4 or len(set(reward_bits[:n])) < 2:
        return empty

    texts = [_normalise_completion(text) for text in completion_texts[:n]]
    candidates: list[tuple[float, float, int, int]] = []
    for left in range(n):
        for right in range(left + 1, n):
            if reward_bits[left] == reward_bits[right]:
                continue
            ratio = _length_ratio(texts[left], texts[right])
            if ratio < min_length_ratio:
                continue
            sim = _similarity(texts[left], texts[right])
            if sim >= min_similarity:
                candidates.append((sim, ratio, left, right))

    used: set[int] = set()
    matched: list[tuple[float, float, int, int]] = []
    for candidate in sorted(candidates, reverse=True):
        _, _, left, right = candidate
        if left in used or right in used:
            continue
        used.add(left)
        used.add(right)
        matched.append(candidate)

    mirror_pairs = 0
    if n % 2 == 0:
        half = n // 2
        for left in range(half):
            right = left + half
            if reward_bits[left] == reward_bits[right]:
                continue
            ratio = _length_ratio(texts[left], texts[right])
            if ratio < min_length_ratio:
                continue
            if _similarity(texts[left], texts[right]) >= mirror_similarity:
                mirror_pairs += 1

    similarities = [pair[0] for pair in matched]
    ratios = [pair[1] for pair in matched]
    return OppositeRewardCloneMetrics(
        suspicious=len(matched) >= min_pairs,
        reward_vector=reward_vector,
        matched_pairs=len(matched),
        mirror_pairs=mirror_pairs,
        max_similarity=max(similarities, default=0.0),
        mean_similarity=(
            sum(similarities) / len(similarities) if similarities else 0.0
        ),
        min_length_ratio=min(ratios, default=0.0),
    )
