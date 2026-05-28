"""Staged prep queue, zone filter, and pre-GRAIL validation."""

import random

import pytest

from reliquary.miner.engine import (
    StagedBundle,
    StagedQueue,
    _PREP_PROMPT_ATTEMPTS,
    _passes_zone_filter,
    _rewards_from_generations,
    bundle_discard_reason,
    pick_prompt_idx,
)
from reliquary.validator.verifier import is_in_zone, rewards_std


class FakeEnv:
    def __len__(self):
        return 100

    def get_problem(self, index: int) -> dict:
        return {"prompt": f"problem {index}", "ground_truth": "1", "id": str(index)}

    def compute_reward(self, problem, completion: str) -> float:
        return 1.0 if "good" in completion else 0.0


class FakeTokenizer:
    def decode(self, tokens) -> str:
        return "good" if tokens and tokens[0] == 1 else "bad"


def test_staged_queue_push_pop_limit():
    q = StagedQueue(max_size=2)
    b = lambda i: StagedBundle(
        prompt_idx=i, problem={}, generations=[], rewards=[],
        sigma=0.5, checkpoint_n=1, checkpoint_hash="abc",
    )
    assert q.push(b(1))
    assert q.push(b(2))
    assert not q.push(b(3))
    assert len(q) == 2
    assert q.pop().prompt_idx == 1
    assert q.pop().prompt_idx == 2
    assert q.pop() is None


def test_purge_removes_cooldown_and_checkpoint_mismatch():
    q = StagedQueue(max_size=8)
    q.push(StagedBundle(1, {}, [], [], 0.5, 1, "revA"))
    q.push(StagedBundle(2, {}, [], [], 0.5, 1, "revA"))
    q.push(StagedBundle(3, {}, [], [], 0.5, 1, "revB"))
    dropped = q.purge(cooldown_set={2}, checkpoint_hash="revA")
    assert dropped == 2
    assert len(q) == 1
    assert q.pop().prompt_idx == 1


def test_bundle_discard_reason():
    b = StagedBundle(42, {}, [], [], 0.5, 1, "revA")
    assert bundle_discard_reason(b, local_hash="revA", cooldown_set=set()) is None
    assert bundle_discard_reason(b, local_hash="revB", cooldown_set=set()) == "checkpoint_mismatch"
    assert bundle_discard_reason(b, local_hash="revA", cooldown_set={42}) == "cooldown"


def test_passes_zone_binary_rewards():
    # k=4 ones → σ=0.5
    assert _passes_zone_filter([1.0] * 4 + [0.0] * 4, bootstrap=False)
    # k=8 ones → σ=0
    assert not _passes_zone_filter([1.0] * 8, bootstrap=False)
    # k=1 one → σ≈0.33 — bootstrap only
    rewards = [1.0] + [0.0] * 7
    assert not _passes_zone_filter(rewards, bootstrap=False)
    assert _passes_zone_filter(rewards, bootstrap=True)


def test_pick_prompt_idx_excludes_staged():
    env = FakeEnv()
    rng = random.Random(0)
    cooldown = set(range(0, 98))
    excluded = {98}
    idx = pick_prompt_idx(env, cooldown, excluded_prompts=excluded, rng=rng)
    assert idx == 99


def test_staged_queue_push_front():
    q = StagedQueue(max_size=2)
    b1 = StagedBundle(1, {}, [], [], 0.5, 1, "abc")
    b2 = StagedBundle(2, {}, [], [], 0.5, 1, "abc")
    assert q.push(b1)
    assert q.push_front(b2)
    assert q.pop().prompt_idx == 2
    assert q.pop().prompt_idx == 1


def test_prepare_bundle_retries_until_in_zone():
    from reliquary.miner.engine import MiningEngine

    eng = object.__new__(MiningEngine)
    eng.env = FakeEnv()
    eng.tokenizer = FakeTokenizer()
    eng._rng = random.Random(0)

    call_count = 0

    def fake_generate(problem):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return [{"tokens": [0, 0], "prompt_length": 1}] * 8
        return (
            [{"tokens": [0, 1], "prompt_length": 1}] * 4
            + [{"tokens": [0, 0], "prompt_length": 1}] * 4
        )

    eng._generate_m_rollouts = fake_generate
    bundle = eng._prepare_bundle(set(), set(), 1, "rev", bootstrap=False)
    assert bundle is not None
    assert call_count == 3
    assert is_in_zone(bundle.sigma, bootstrap=False)


def test_prepare_bundle_exhausts_attempts():
    from reliquary.miner.engine import MiningEngine

    eng = object.__new__(MiningEngine)
    eng.env = FakeEnv()
    eng.tokenizer = FakeTokenizer()
    eng._rng = random.Random(1)
    eng._generate_m_rollouts = lambda problem: [
        {"tokens": [0, 0], "prompt_length": 1},
    ] * 8

    bundle = eng._prepare_bundle(
        set(), set(), 1, "rev", bootstrap=False,
        max_prompt_attempts=_PREP_PROMPT_ATTEMPTS,
    )
    assert bundle is None


def test_rewards_from_generations():
    env = FakeEnv()
    gens = [
        {"tokens": [0, 1], "prompt_length": 1},
        {"tokens": [0, 0], "prompt_length": 1},
    ]
    rewards = _rewards_from_generations(env, env.get_problem(0), gens, FakeTokenizer())
    assert rewards == [1.0, 0.0]
    assert is_in_zone(rewards_std(rewards), bootstrap=False)
