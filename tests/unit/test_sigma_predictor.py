"""Unit tests for the σ-predictor."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest

from reliquary.miner.sigma_predictor import (
    BetaBucketPredictor,
    bucket_of,
    pick_with_predictor,
)


class _FakeEnv:
    def __init__(self, prompts: list[str]) -> None:
        self._prompts = prompts

    def __len__(self) -> int:
        return len(self._prompts)

    def get_problem(self, idx: int) -> dict:
        return {"prompt": self._prompts[idx % len(self._prompts)]}


def _uniform_fallback(env, cooldown, *, excluded_prompts, rng):
    blocked = cooldown | (excluded_prompts or set())
    for _ in range(1000):
        idx = rng.randrange(len(env))
        if idx not in blocked:
            return idx
    raise RuntimeError("no eligible prompt")


def test_bucket_of_is_deterministic():
    assert bucket_of("foo bar") == bucket_of("foo bar")


def test_bucket_of_in_range():
    for s in ["", "short", "a" * 1000, "12345" * 100, "What is 2+2?"]:
        b = bucket_of(s)
        assert 0 <= b < 64


def test_update_shifts_posterior():
    p = BetaBucketPredictor()
    prompt = "test prompt"
    b = bucket_of(prompt)
    base_alpha = p.alpha[b]
    base_beta = p.beta[b]
    p.update(prompt, in_zone=True)
    assert p.alpha[b] == base_alpha + 1
    assert p.beta[b] == base_beta
    p.update(prompt, in_zone=False)
    assert p.beta[b] == base_beta + 1


def test_reset_clears_state():
    p = BetaBucketPredictor()
    for _ in range(50):
        p.update("foo", in_zone=True)
    assert p.n_observations == 50
    p.reset()
    assert p.n_observations == 0
    assert all(a == 1.0 for a in p.alpha)
    assert all(b == 1.0 for b in p.beta)


def test_save_load_roundtrip():
    p = BetaBucketPredictor()
    for i in range(30):
        p.update(f"prompt-{i}", in_zone=(i % 3 == 0))
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "pred.json")
        p.save(path)
        q = BetaBucketPredictor.load(path)
        assert q.n_observations == p.n_observations
        assert q.alpha == p.alpha
        assert q.beta == p.beta


def test_score_is_in_unit_interval():
    p = BetaBucketPredictor()
    rng = random.Random(42)
    for _ in range(100):
        s = p.score("test", rng=rng)
        assert 0.0 <= s <= 1.0


def test_pick_with_predictor_falls_back_when_predictor_none():
    env = _FakeEnv([f"p{i}" for i in range(20)])
    rng = random.Random(0)
    idx = pick_with_predictor(
        env, set(), excluded_prompts=set(),
        rng=rng, predictor=None,
        uniform_fallback=_uniform_fallback,
    )
    assert 0 <= idx < 20


def test_pick_with_predictor_skips_blocked():
    env = _FakeEnv([f"p{i}" for i in range(20)])
    rng = random.Random(0)
    blocked = set(range(0, 18))  # only 18, 19 eligible
    for _ in range(20):
        idx = pick_with_predictor(
            env, blocked, excluded_prompts=set(),
            rng=rng,
            predictor=BetaBucketPredictor(),
            uniform_fallback=_uniform_fallback,
            n_candidates=4,
        )
        assert idx in {18, 19}


def test_pick_with_predictor_biases_toward_high_yield_bucket():
    # Two distinct buckets, one trained to high-yield, one to low-yield.
    # Verify Thompson sampling biases picks toward the high-yield bucket.
    high_yield_prompts = [f"H{i:03d}_short" for i in range(10)]
    low_yield_prompts = [f"L{i:03d}_some_much_longer_prompt_text_here" * 3 for i in range(10)]
    env = _FakeEnv(high_yield_prompts + low_yield_prompts)

    p = BetaBucketPredictor()
    # Strong evidence: high-yield bucket gets 100 successes
    for hp in high_yield_prompts:
        for _ in range(100):
            p.update(hp, in_zone=True)
    # Low-yield bucket gets 100 failures
    for lp in low_yield_prompts:
        for _ in range(100):
            p.update(lp, in_zone=False)

    rng = random.Random(0)
    high_picked = 0
    n_trials = 200
    for _ in range(n_trials):
        idx = pick_with_predictor(
            env, set(),
            rng=rng, predictor=p,
            uniform_fallback=_uniform_fallback,
            n_candidates=8,
        )
        if idx < 10:
            high_picked += 1
    # Expect overwhelmingly biased toward high-yield (well above uniform 50%)
    assert high_picked / n_trials > 0.85, (
        f"high-yield share={high_picked}/{n_trials}; expected >85%"
    )
