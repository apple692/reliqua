"""Online σ-zone yield predictor — bias prompt selection toward learning-frontier buckets.

The protocol rewards in-zone bundles (σ ≥ SIGMA_MIN). With binary rewards over
M=8 rollouts in {0,1}, "in-zone" requires 2 ≤ k ≤ 6 successes. The probability
of this depends on the underlying success rate p, which varies by prompt
difficulty. This predictor maintains a Beta(α, β) posterior over P(in_zone)
per cheap-feature bucket, then uses Thompson sampling so ``pick_prompt_idx``
biases toward buckets with high posterior yield while preserving exploration.

The 1M-window cooldown makes per-prompt history useless (every prompt is
one-shot in steady state), so all signal must come from *bucket* features
that generalise across the 14M-prompt corpus.

Reset on checkpoint update: bucket statistics from the prior model are stale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random as _random
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# 8 length × 4 digit-density × 2 structure = 64 buckets.
# Small enough each bucket fills in O(100) samples; large enough to capture
# meaningful problem-difficulty variation across OpenMathInstruct-2.
_LENGTH_BUCKETS = 8
_DIGIT_BUCKETS = 4
_STRUCTURE_BUCKETS = 2
_N_BUCKETS = _LENGTH_BUCKETS * _DIGIT_BUCKETS * _STRUCTURE_BUCKETS

# Beta(1,1) = uniform [0,1] prior. Before any updates Thompson sampling
# equals uniform candidate pick — i.e. the cold-start path matches baseline.
_ALPHA_PRIOR = 1.0
_BETA_PRIOR = 1.0


def _bucket_features(prompt: str) -> tuple[int, int, int]:
    """Three cheap features: log-length, digit density, structure hash."""
    n = len(prompt)
    length_b = max(0, min(_LENGTH_BUCKETS - 1, int(math.log1p(n) / math.log(1.7))))
    n_digits = sum(c.isdigit() for c in prompt)
    digit_ratio = n_digits / max(n, 1)
    digit_b = max(0, min(_DIGIT_BUCKETS - 1, int(digit_ratio * (_DIGIT_BUCKETS * 2))))
    # Use a hash of head+tail to discriminate problem families that share
    # length/digit ratios but differ in structure (e.g. word problem vs eq).
    digest = hashlib.sha256((prompt[:16] + prompt[-16:]).encode()).hexdigest()
    structure_b = int(digest[:2], 16) % _STRUCTURE_BUCKETS
    return length_b, digit_b, structure_b


def bucket_of(prompt: str) -> int:
    """Map *prompt* to a fixed bucket index in ``[0, N_BUCKETS)``."""
    length_b, digit_b, structure_b = _bucket_features(prompt)
    return (length_b * _DIGIT_BUCKETS * _STRUCTURE_BUCKETS
            + digit_b * _STRUCTURE_BUCKETS
            + structure_b)


@dataclass
class BetaBucketPredictor:
    """Per-bucket Beta(α, β) posterior over in-zone yield.

    Thread-safe: ``_lock`` serialises mutation across worker-thread updates
    from ``_prepare_bundle`` and main-thread save/reset calls.
    """

    alpha: list[float] = field(
        default_factory=lambda: [_ALPHA_PRIOR] * _N_BUCKETS,
    )
    beta: list[float] = field(
        default_factory=lambda: [_BETA_PRIOR] * _N_BUCKETS,
    )
    n_observations: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def score(self, prompt: str, rng: _random.Random | None = None) -> float:
        """Thompson sample — draw from Beta(α_b, β_b) for the prompt's bucket."""
        rng = rng if rng is not None else _random
        b = bucket_of(prompt)
        with self._lock:
            a = self.alpha[b]
            be = self.beta[b]
        return rng.betavariate(a, be)

    def update(self, prompt: str, in_zone: bool) -> None:
        """Bayesian update — ``in_zone`` increments α, else β."""
        b = bucket_of(prompt)
        with self._lock:
            if in_zone:
                self.alpha[b] += 1.0
            else:
                self.beta[b] += 1.0
            self.n_observations += 1

    def reset(self) -> None:
        """Clear all evidence. Call on checkpoint pull (stale model → stale stats)."""
        with self._lock:
            self.alpha = [_ALPHA_PRIOR] * _N_BUCKETS
            self.beta = [_BETA_PRIOR] * _N_BUCKETS
            self.n_observations = 0

    def stats_line(self) -> str:
        with self._lock:
            yields = [
                self.alpha[i] / (self.alpha[i] + self.beta[i])
                for i in range(_N_BUCKETS)
            ]
            total_a = sum(self.alpha)
            total_b = sum(self.beta)
            n = self.n_observations
        yields_sorted = sorted(yields, reverse=True)
        top3 = ",".join(f"{y:.2f}" for y in yields_sorted[:3])
        bot3 = ",".join(f"{y:.2f}" for y in yields_sorted[-3:])
        global_yield = total_a / max(total_a + total_b, 1e-9)
        return (
            f"sigma_predictor obs={n} global_yield={global_yield:.3f} "
            f"top3=[{top3}] bot3=[{bot3}]"
        )

    def save(self, path: str) -> None:
        with self._lock:
            payload = {
                "alpha": list(self.alpha),
                "beta": list(self.beta),
                "n_observations": self.n_observations,
                "n_buckets": _N_BUCKETS,
            }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "BetaBucketPredictor":
        with open(path) as f:
            data = json.load(f)
        if data.get("n_buckets") != _N_BUCKETS:
            logger.warning(
                "sigma_predictor: bucket-count mismatch in %s (have %d, file %d); "
                "starting fresh",
                path, _N_BUCKETS, data.get("n_buckets"),
            )
            return cls()
        return cls(
            alpha=list(data["alpha"]),
            beta=list(data["beta"]),
            n_observations=int(data.get("n_observations", 0)),
        )


def pick_with_predictor(
    env,
    cooldown_prompts: set[int],
    *,
    excluded_prompts: set[int] | None = None,
    rng: _random.Random | None = None,
    predictor: BetaBucketPredictor | None = None,
    term_predictor: BetaBucketPredictor | None = None,
    n_candidates: int = 8,
    uniform_fallback,
    min_prompt_chars: int = 30,
    max_prompt_chars: int = 1500,
    exclude_low_mean_threshold: float = 0.05,
    tokenizer=None,
    max_generated_solution_tokens: int | None = None,
) -> int:
    """Score *n_candidates* eligible prompts via Thompson sampling, return best.

    Improvements over the original:
    - ``term_predictor`` (optional): separate Beta-Bernoulli predictor for
      termination quality. Combined score = zone × term, so a prompt needs
      BOTH high zone yield AND good termination to win.
    - Length filter: skip prompts shorter than ``min_prompt_chars`` (one-word
      answers tend not to terminate) or longer than ``max_prompt_chars``
      (confuse the model into rambling).
    - Reference-solution filter: when *tokenizer* is set, skip prompts whose
      dataset ``generated_solution`` token length is >=
      *max_generated_solution_tokens* (default 500).
    - Hard exclusion: candidate skipped when posterior mean (alpha/(alpha+beta))
      from either predictor is below ``exclude_low_mean_threshold``. Once a
      bucket has accumulated strong negative evidence we stop wasting compute
      sampling from it. Thompson exploration still works on unseen buckets.

    Falls back to ``uniform_fallback`` if no candidates pass filters.
    """
    from reliquary.miner.prompt_selection import (
        passes_generated_solution_length_filter,
    )

    rng = rng if rng is not None else _random
    blocked = cooldown_prompts | (excluded_prompts or set())
    n = len(env)
    pick_kwargs = {
        "tokenizer": tokenizer,
        "max_generated_solution_tokens": max_generated_solution_tokens,
    }

    if predictor is None or n_candidates <= 1:
        return uniform_fallback(
            env, cooldown_prompts,
            excluded_prompts=excluded_prompts, rng=rng,
            **pick_kwargs,
        )

    candidates: list[tuple[int, str]] = []
    seen: set[int] = set()
    # 8x oversample to account for length filter + hard exclusion rejections.
    for _ in range(n_candidates * 8):
        idx = rng.randrange(n)
        if idx in blocked or idx in seen:
            continue
        seen.add(idx)
        if not passes_generated_solution_length_filter(
            env, idx, tokenizer,
            max_tokens=max_generated_solution_tokens,
        ):
            continue
        prompt = env.get_problem(idx).get("prompt", "")
        plen = len(prompt)
        if plen < min_prompt_chars or plen > max_prompt_chars:
            continue
        # Hard exclusion based on bucket posterior means.
        b = bucket_of(prompt)
        zone_mean = predictor.alpha[b] / (predictor.alpha[b] + predictor.beta[b])
        term_mean = (
            term_predictor.alpha[b] / (term_predictor.alpha[b] + term_predictor.beta[b])
            if term_predictor is not None
            else 1.0
        )
        # Only exclude if the bucket has > 30 observations AND posterior mean
        # is below threshold. Avoids over-aggressive cold-start exclusions.
        zone_obs = predictor.alpha[b] + predictor.beta[b] - 2  # subtract prior
        if zone_obs > 30 and zone_mean < exclude_low_mean_threshold:
            continue
        if term_predictor is not None:
            term_obs = term_predictor.alpha[b] + term_predictor.beta[b] - 2
            if term_obs > 30 and term_mean < exclude_low_mean_threshold:
                continue
        candidates.append((idx, prompt))
        if len(candidates) >= n_candidates:
            break

    if not candidates:
        return uniform_fallback(
            env, cooldown_prompts,
            excluded_prompts=excluded_prompts, rng=rng,
            **pick_kwargs,
        )

    best_idx = candidates[0][0]
    best_score = -1.0
    for idx, prompt in candidates:
        zone_score = predictor.score(prompt, rng=rng)
        term_score = (
            term_predictor.score(prompt, rng=rng)
            if term_predictor is not None
            else 1.0
        )
        # Product favours prompts with BOTH good zone yield and good
        # termination — multiplication penalises any weakness.
        combined = zone_score * term_score
        if combined > best_score:
            best_score = combined
            best_idx = idx
    return best_idx
