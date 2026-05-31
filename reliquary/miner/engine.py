"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.

Staged prep: generate + zone-filter (σ) during downtime; GRAIL + submit on OPEN.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import random as _random

from reliquary.constants import (
    BOOTSTRAP_WINDOWS,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    MAX_TRUNCATED_PER_SUBMISSION,
    MIN_EOS_PROBABILITY,
    M_ROLLOUTS,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)
from reliquary.infrastructure import chain
from reliquary.miner.generation import HFGenerator, VLLMGenerator
from reliquary.miner.sigma_predictor import (
    BetaBucketPredictor,
    pick_with_predictor,
)
from reliquary.shared.prompt_augment import (
    canonical_prompt_tokens,
    generation_prompt_tokens,
    max_new_tokens_for_generation,
    rebind_rollouts_to_canonical_prompt,
)
from reliquary.protocol.signatures import sign_envelope
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)
from reliquary.validator.verifier import is_in_zone, rewards_std

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)

# Stop starting new prep when the batch is nearly full (B_BATCH = 8).
_BATCH_NEARLY_FULL = 7
# Prompts to try per prep cycle before giving up (σ filter rejects most).
_PREP_PROMPT_ATTEMPTS = 8
# Fast-reject: sample 2 rollouts first; skip full 8 when both rewards match (σ=0).
_ZONE_SCREEN_ROLLOUTS = 2
# Tiebreaker: when first 2 binary rewards agree, generate this many more before
# declaring σ=0. At T_PROTO=0.9 a p≈0.4 borderline prompt has ~52% chance of
# n=2 collision; one extra rollout drops that false-negative rate to ~28%.
_ZONE_SCREEN_TIEBREAKER_ROLLOUTS = 1
# Local truncation cap at PREP TIME. Matches validator's
# MAX_TRUNCATED_PER_SUBMISSION (=1) — the deployed validator allows at most
# ONE cap-truncated rollout per 8-rollout submission. Anything stricter
# wastes scanning effort; anything looser stages bundles the validator
# guaranteed to reject.
_LOCAL_MAX_TRUNCATED = 1
# Batched vLLM prep: how many prompts per vLLM call. Each prompt produces
# M_ROLLOUTS=8 sequences, so K=8 → 64 concurrent sequences. B200's KV cache
# advertises 126x concurrency for 8192-token sequences; 64 is well inside.
# K is the ~Kx throughput multiplier on the σ-predictor's warmup rate;
# cycles at max_new=8192 are slow, so we want more prompts per cycle.
_BATCHED_PREP_K = 8

# Generation max-new-tokens for batched prep. MUST equal the protocol cap
# (MAX_NEW_TOKENS_PROTOCOL_CAP=8192). Any rollout that stops below this cap
# without a real EOS is "broken" from the validator's view:
#   - in_eos=False  → fails verify_termination Path 2 (EOS+p_stop)
#   - total<8192    → fails verify_termination Path 1 (cap reached)
#   - total<8192    → is_cap_truncation=False → NOT absorbed by the
#                     MAX_TRUNCATED_PER_SUBMISSION=1 budget either
# So a local cap below 8192 manufactures rejections out of nothing — the
# rollout becomes neither valid nor accountable. Wall-clock cost is real
# (~3-4× slower per cycle vs 2048) but we have no choice: cap=1 leaves
# zero budget for self-inflicted cap truncations.
_VLLM_BATCHED_MAX_NEW = MAX_NEW_TOKENS_PROTOCOL_CAP
# vLLM batched prep: generate this many head rollouts first; skip the tail
# when every head rollout hits max_tokens without EOS (saves ~75% gen on
# non-terminating prompts). HF _try_prompt_bundle already does an equivalent
# 2-rollout screen; batched prep skipped it for throughput — re-enabled here.
_BATCHED_HEAD_SCREEN_ROLLOUTS = 2
# Rejection-sampling rescue: when an initial M=8 prep cycle misses on
# truncation but a meaningful fraction of rollouts naturally terminated, the
# prompt has demonstrated the model CAN terminate on it — keep generating
# until we accumulate 7+ terminated rollouts (validator cap=1 allows one
# truncated). Pool size capped to bound compute per prompt; per-cycle attempt
# cap bounds total wall-clock the rescue path can add.
_RESCUE_MIN_INITIAL_TERMINATED = 2
# Required count of VALIDLY terminated initial rollouts (last_token in eos_set
# AND p_stop ≥ MIN_EOS_PROBABILITY) before we commit to a rescue. This is the
# stricter gate behind the cheap EOS-last gate above. Reason: ~24-rollout
# pool with cap=1 truncated needs ≥7 valid terminators in the bundle, so the
# per-rollout valid rate must be ≥ 7/24 ≈ 0.29 for rescue to plausibly succeed.
# 3/8 = 0.375 sample rate gives reasonable confidence we're above that.
# Skipping below this threshold trades ~1-2s of pre-check (8 HF forward
# passes) against avoiding the full ~90-100s of a doomed rescue.
_RESCUE_MIN_INITIAL_VALID_TERMINATED = 1
# Local p_stop floor for "valid termination" — currently matches the public
# validator threshold MIN_EOS_PROBABILITY=0.01 exactly.
#
# Earlier we saw a dashboard BAD_TERMINATION rejection on a bundle whose
# lowest local p_stop was ~0.25, suggesting the deployed validator might
# use a stricter threshold than 0.01. Setting this above 0.01 (e.g. 0.05
# or 0.30) adds a safety margin against that, but starves the miner of
# rollouts when the checkpoint has low natural EOS probability (most
# rollouts get classified as "border" and excluded from selection).
#
# Recommended: keep at 0.01 to match the public validator; if acceptance
# rate stays at 0% with measurable cross-GPU drift evidence, bump to 0.05
# or 0.10 to add margin.
_LOCAL_MIN_EOS_PROB = 0.01
_RESCUE_MAX_POOL_SIZE = 24
_RESCUE_BATCH_SIZE = M_ROLLOUTS
_RESCUE_MAX_PER_CYCLE = 2
# Poll interval when staging can immediately start another prep cycle.
_PREP_POLL_SECONDS = 0.2
# Main-loop /state poll pacing (seconds).
_STATE_POLL_BUSY = 0.25
_STATE_POLL_OPEN = 0.5
_STATE_POLL_CAP = 1.0
_STATE_POLL_DOWNTIME_BUSY = 0.25
_STATE_POLL_DOWNTIME_IDLE = 1.0
_HEARTBEAT_SECONDS = 60.0


def eos_set_from_model(model, tokenizer) -> set[int]:
    """EOS token IDs — EXACTLY match validator's ``_eos_set_from_model``.

    Validator uses ``model.generation_config.eos_token_id`` first, falling
    back to ``tokenizer.eos_token_id``. For Qwen3-4B-Instruct-2507 the
    loaded ``model.generation_config.eos_token_id`` resolves to
    ``[151645, 151643]`` (both <|im_end|> AND <|endoftext|>), even though
    ``tokenizer.eos_token_id`` is just 151645.

    Previous fix attempt used tokenizer-only (set={151645}) on the theory
    that 151643 was a pad token only. That was wrong:
    - vLLM only stopped on 151645
    - Model could emit 151643 mid-completion
    - Validator (with broader set) saw 151643 mid-completion AS EOS PADDING
      → immediate bad_termination reject (no budget, no truncation count)

    The right fix is to mirror the validator exactly: use the broader set
    in BOTH our stop_token_ids (so vLLM stops on either) AND our
    has_eos_padding check (so we agree with the validator).
    """
    eos_ids: set[int] = set()

    gen_cfg = getattr(model, "generation_config", None) if model is not None else None
    if gen_cfg is not None:
        cfg_ids = getattr(gen_cfg, "eos_token_id", None)
        if cfg_ids is not None:
            if isinstance(cfg_ids, int):
                cfg_ids = [cfg_ids]
            eos_ids.update(int(e) for e in cfg_ids if e is not None)

    if not eos_ids:
        tok_eos = getattr(tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            eos_ids.add(int(tok_eos))

    return eos_ids


def truncate_completion_at_eos(
    completion_tokens: list[int],
    eos_set: set[int],
) -> list[int]:
    """Keep tokens through the first EOS in *eos_set* (validator-aligned)."""
    if not eos_set:
        return completion_tokens
    for idx, token in enumerate(completion_tokens):
        if int(token) in eos_set:
            return completion_tokens[: idx + 1]
    return completion_tokens


@dataclass
class StagedBundle:
    """Pre-validated rollout group — GRAIL commits built only at submit time."""

    prompt_idx: int
    problem: dict
    generations: list[dict]
    rewards: list[float]
    sigma: float
    checkpoint_n: int
    checkpoint_hash: str


class StagedQueue:
    """FIFO queue of in-zone bundles awaiting GRAIL + submit."""

    def __init__(self, max_size: int = MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW) -> None:
        self.max_size = max_size
        self._items: deque[StagedBundle] = deque()

    def __len__(self) -> int:
        return len(self._items)

    def staged_prompt_indices(self) -> set[int]:
        return {b.prompt_idx for b in self._items}

    def push(self, bundle: StagedBundle) -> bool:
        if len(self._items) >= self.max_size:
            return False
        self._items.append(bundle)
        return True

    def push_front(self, bundle: StagedBundle) -> bool:
        """Re-queue a bundle skipped at submit time (e.g. batch nearly full)."""
        if len(self._items) >= self.max_size:
            return False
        self._items.appendleft(bundle)
        return True

    def pop(self) -> StagedBundle | None:
        if not self._items:
            return None
        return self._items.popleft()

    def clear(self) -> None:
        self._items.clear()

    def purge(self, cooldown_set: set[int], checkpoint_hash: str) -> int:
        """Drop bundles on cooldown or wrong checkpoint. Returns drop count."""
        before = len(self._items)
        self._items = deque(
            b for b in self._items
            if b.checkpoint_hash == checkpoint_hash
            and b.prompt_idx not in cooldown_set
        )
        return before - len(self._items)


def bundle_discard_reason(
    bundle: StagedBundle,
    *,
    local_hash: str,
    cooldown_set: set[int],
) -> str | None:
    """Return a discard reason, or None if the bundle may proceed to GRAIL."""
    if bundle.checkpoint_hash != local_hash:
        return "checkpoint_mismatch"
    if bundle.prompt_idx in cooldown_set:
        return "cooldown"
    return None


async def maybe_pull_checkpoint(
    state,
    local_n: int,
    local_hash: str,
    local_model,
    *,
    download_fn,
    load_fn,
):
    """If remote checkpoint_n > local, download via HF and load.

    state.checkpoint_repo_id + state.checkpoint_revision identify the
    HF snapshot. download_fn/load_fn still injected for testability.

    Returns ``(new_local_n, new_local_hash, new_model)``. If no update is
    needed (remote ≤ local, or remote has no repo/revision yet), returns
    inputs unchanged.
    """
    if state.checkpoint_n <= local_n:
        return local_n, local_hash, local_model
    if state.checkpoint_repo_id is None or state.checkpoint_revision is None:
        return local_n, local_hash, local_model
    local_path = await download_fn(state.checkpoint_repo_id, state.checkpoint_revision)
    new_model = load_fn(local_path)
    return state.checkpoint_n, state.checkpoint_revision, new_model


async def _hf_download(repo_id: str, revision: str) -> str:
    """Download a snapshot into the local HF cache and return the model folder path."""
    from huggingface_hub import snapshot_download

    return await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["model.safetensors", "config.json", "tokenizer*"],
    )


async def _prune_old_checkpoint_revisions(
    repo_id: str, keep_revision: str,
) -> None:
    """Delete every cached revision of *repo_id* except *keep_revision*.

    HuggingFace's snapshot cache writes a new commit-hash directory on every
    pull but never garbage-collects. Each R0mAI/reliquary-sn-v23 revision is
    ~8 GB; over a day of mining the cache balloons to 400+ GB. This prunes
    after each successful checkpoint transition so disk usage stays bounded
    to ~one revision.

    Uses huggingface_hub's cache-aware delete API so blobs are properly
    garbage-collected (not just symlinks). Best-effort: failures are logged
    but never raise. Runs in a thread to avoid blocking the event loop on
    large caches.
    """
    def _do_prune() -> tuple[int, int]:
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            return -1, 0
        try:
            info = scan_cache_dir()
            repo = next(
                (r for r in info.repos if r.repo_id == repo_id),
                None,
            )
            if repo is None:
                return 0, 0
            to_delete = [
                rev.commit_hash for rev in repo.revisions
                if rev.commit_hash != keep_revision
            ]
            if not to_delete:
                return 0, 0
            strategy = info.delete_revisions(*to_delete)
            freed = strategy.expected_freed_size
            strategy.execute()
            return len(to_delete), freed
        except Exception:
            logger.exception(
                "checkpoint cache prune failed for repo=%s", repo_id,
            )
            return -1, 0

    n_deleted, freed = await asyncio.to_thread(_do_prune)
    if n_deleted > 0:
        logger.info(
            "checkpoint cache: pruned %d stale revision(s) of %s, "
            "freed %.1f GiB (kept %s)",
            n_deleted, repo_id, freed / (1024 ** 3), keep_revision[:12],
        )


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    excluded_prompts: set[int] | None = None,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
    tokenizer=None,
    max_generated_solution_tokens: int | None = None,
) -> int:
    """Pick a random prompt index that isn't in cooldown or *excluded_prompts*.

    The reference miner uses uniform-random selection with rejection
    sampling against the blocked set. More sophisticated strategies
    (pre-screening zone probability, etc.) are left to miner operators.

    When *tokenizer* is set, skips prompts whose dataset
    ``generated_solution`` token length is >= *max_generated_solution_tokens*
    (default from ``RELIQUARY_MAX_GENERATED_SOLUTION_TOKENS``, 500).

    Raises ``RuntimeError`` if no eligible prompt can be found — typically
    because the env is fully in cooldown.
    """
    from reliquary.miner.prompt_selection import (
        max_generated_solution_tokens_for_pick,
        passes_generated_solution_length_filter,
    )

    rng = rng or _random
    blocked = cooldown_prompts | (excluded_prompts or set())
    n = len(env)
    solution_limit = (
        max_generated_solution_tokens_for_pick()
        if max_generated_solution_tokens is None
        else max_generated_solution_tokens
    )

    def _eligible(idx: int) -> bool:
        if idx in blocked:
            return False
        return passes_generated_solution_length_filter(
            env, idx, tokenizer, max_tokens=solution_limit,
        )

    if len(blocked) < n / 2:
        for _ in range(max_attempts):
            idx = rng.randrange(n)
            if _eligible(idx):
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(n) if _eligible(i)]
    if not eligible:
        raise RuntimeError("no eligible prompt — env fully in cooldown")
    return rng.choice(eligible)


def _rewards_from_generations(
    env,
    problem: dict,
    generations: list[dict],
    tokenizer,
) -> list[float]:
    rewards: list[float] = []
    for gen in generations:
        prompt_length = gen["prompt_length"]
        completion_tokens = gen["tokens"][prompt_length:]
        completion_text = tokenizer.decode(completion_tokens)
        rewards.append(float(env.compute_reward(problem, completion_text)))
    return rewards


def _passes_zone_filter(rewards: list[float], *, bootstrap: bool) -> bool:
    return is_in_zone(rewards_std(rewards), bootstrap=bootstrap)


def _compute_malformed_flags(
    generations: list[dict],
    rewards: list[float],
    tokenizer,
) -> list[bool]:
    """For each rollout, True if it would trigger MALFORMED_FINAL_ANSWER.

    The validator's ``has_malformed_final_answer`` ([boxed_integrity.py:80])
    is a proof-free structural check that fires when a reward<0.5 rollout's
    final ``\\boxed{...}`` / ``\\fbox{...}`` is empty, contains a special
    token, or is unclosed. Most-common honest trigger: cap-truncated rollouts
    that opened a fresh box near token 8190 and never reached the closing
    ``}`` — the last box in the completion is unclosed, validator rejects the
    entire bundle.

    Mirrors the validator's logic so we can exclude poison rollouts from
    selection before submission. Pure text decode + scan; cheap (~1ms per
    rollout).
    """
    from reliquary.validator.boxed_integrity import has_malformed_final_answer

    flags: list[bool] = []
    for gen, reward in zip(generations, rewards):
        if reward >= 0.5:
            # Validator only inspects reward<0.5 rollouts; correct rollouts
            # with weird-looking boxes are exempt by construction.
            flags.append(False)
            continue
        prompt_length = gen.get("prompt_length", 0)
        completion_tokens = gen["tokens"][prompt_length:]
        completion_text = tokenizer.decode(completion_tokens)
        is_bad, _ = has_malformed_final_answer(reward, completion_text)
        flags.append(bool(is_bad))
    return flags


def _count_truncated(generations: list[dict], eos_set: set[int]) -> int:
    """Number of rollouts whose last token is NOT in ``eos_set``.

    Mirrors the validator's truncation check — any rollout that hit
    ``max_new_tokens`` without producing an EOS token counts. The protocol
    rejects submissions with more than ``MAX_TRUNCATED_PER_SUBMISSION``
    truncated rollouts (``reason=bad_termination``).
    """
    if not eos_set:
        return 0
    n_trunc = 0
    for gen in generations:
        tokens = gen.get("tokens", [])
        if not tokens:
            n_trunc += 1
            continue
        if int(tokens[-1]) not in eos_set:
            n_trunc += 1
    return n_trunc


def _select_8_passing(
    generations: list[dict],
    rewards: list[float],
    eos_set: set[int],
    *,
    bootstrap: bool,
    rng,
    p_stops: list[float] | None = None,
    malformed_flags: list[bool] | None = None,
) -> tuple[list[dict], list[float], float, int] | None:
    """Pick 8 rollouts from a larger pool so the bundle would pass the validator.

    Constraints mirrored from the validator: ≤ MAX_TRUNCATED_PER_SUBMISSION
    truncated rollouts AND ``σ >= SIGMA_MIN`` (or BOOTSTRAP_SIGMA_MIN).

    Selection prefers 0 truncated rollouts when feasible (more headroom for
    validator-side numerical drift in p_stop). Among feasible (n_pos, n_neg)
    reward splits, picks the closest-to-4/4 split that satisfies the σ floor.

    When ``p_stops`` is provided, a rollout is considered VALIDLY terminated
    only when (last_token ∈ eos_set) AND (p_stop ≥ MIN_EOS_PROBABILITY). This
    matches the validator's ``verify_termination`` Path 2 — without it, the
    selector would treat sampling-fluke EOS rollouts (where the model emitted
    a low-probability ``<|endoftext|>`` mid-ramble) as valid terminators, and
    the downstream submission would be rejected on bad_termination.

    When ``malformed_flags`` is provided, rollouts flagged True are EXCLUDED
    from candidate buckets entirely — they would trigger the validator's
    ``MALFORMED_FINAL_ANSWER`` reject regardless of how the rest of the
    bundle looks. Typically these are cap-truncated rollouts with an unclosed
    final ``\\boxed{`` (model opened a fresh box near 8190 and ran out of
    tokens before closing it).

    Returns ``(selected_gens, selected_rewards, σ, n_trunc)`` or ``None`` if
    no valid 8-rollout bundle can be assembled from the pool.
    """
    sigma_threshold = 0.33 if bootstrap else 0.43

    term_pos: list[int] = []
    term_neg: list[int] = []
    trunc_pos: list[int] = []
    trunc_neg: list[int] = []
    for i, gen in enumerate(generations):
        # Drop malformed rollouts from any candidate bucket — they're poison.
        if malformed_flags is not None and malformed_flags[i]:
            continue
        tokens = gen.get("tokens", [])
        is_eos = bool(tokens) and int(tokens[-1]) in eos_set
        if p_stops is not None:
            is_valid_term = is_eos and p_stops[i] >= _LOCAL_MIN_EOS_PROB
        else:
            is_valid_term = is_eos
        is_pos = rewards[i] >= 0.5
        if is_valid_term:
            (term_pos if is_pos else term_neg).append(i)
        else:
            (trunc_pos if is_pos else trunc_neg).append(i)

    # σ peak at (4,4); equal-quality (3,5) / (5,3) → 0.484; (2,6) / (6,2) → 0.433.
    # bootstrap also accepts (1,7) / (7,1) → 0.331. Try in σ-descending order.
    splits = [(4, 4), (3, 5), (5, 3), (2, 6), (6, 2)]
    if bootstrap:
        splits.extend([(1, 7), (7, 1)])

    def _try_compose(n_pos: int, n_neg: int, n_trunc_budget: int):
        """Pick n_pos positive + n_neg negative rollouts using ≤ n_trunc_budget truncated."""
        # Prefer terminated first; spend truncated budget only when needed.
        for trunc_pos_used in range(min(n_pos, n_trunc_budget) + 1):
            term_pos_used = n_pos - trunc_pos_used
            if term_pos_used > len(term_pos) or trunc_pos_used > len(trunc_pos):
                continue
            for trunc_neg_used in range(min(n_neg, n_trunc_budget - trunc_pos_used) + 1):
                term_neg_used = n_neg - trunc_neg_used
                if term_neg_used > len(term_neg) or trunc_neg_used > len(trunc_neg):
                    continue
                indices = (
                    rng.sample(term_pos, term_pos_used)
                    + rng.sample(trunc_pos, trunc_pos_used)
                    + rng.sample(term_neg, term_neg_used)
                    + rng.sample(trunc_neg, trunc_neg_used)
                )
                rng.shuffle(indices)
                return indices, trunc_pos_used + trunc_neg_used
        return None

    # Phase 1: try splits with 0 truncated (safest).
    for n_pos, n_neg in splits:
        sigma = rewards_std([1.0] * n_pos + [0.0] * n_neg)
        if sigma < sigma_threshold:
            continue
        result = _try_compose(n_pos, n_neg, n_trunc_budget=0)
        if result is None:
            continue
        indices, used = result
        return (
            [generations[i] for i in indices],
            [rewards[i] for i in indices],
            sigma,
            used,
        )

    # Phase 2: allow up to MAX_TRUNCATED_PER_SUBMISSION truncated.
    for n_pos, n_neg in splits:
        sigma = rewards_std([1.0] * n_pos + [0.0] * n_neg)
        if sigma < sigma_threshold:
            continue
        result = _try_compose(
            n_pos, n_neg, n_trunc_budget=MAX_TRUNCATED_PER_SUBMISSION,
        )
        if result is None:
            continue
        indices, used = result
        return (
            [generations[i] for i in indices],
            [rewards[i] for i in indices],
            sigma,
            used,
        )

    return None


def _count_eos_padding(generations: list[dict], eos_set: set[int]) -> int:
    """Rollouts with EOS padding (multiple EOS or EOS not at end of completion).

    Mirrors validator's ``has_eos_padding`` ([verifier.py:208]). ANY rollout
    with EOS padding triggers an immediate bad_termination reject, so we want
    zero of these.
    """
    if not eos_set:
        return 0
    n = 0
    for gen in generations:
        tokens = gen.get("tokens", [])
        prompt_length = gen.get("prompt_length", 0)
        completion = tokens[prompt_length:]
        eos_positions = [i for i, t in enumerate(completion) if int(t) in eos_set]
        if not eos_positions:
            continue
        if len(eos_positions) > 1 or eos_positions[0] != len(completion) - 1:
            n += 1
    return n


def _predict_validator_bad_termination(
    commit: dict,
    eos_set: set[int],
    min_eos_logp: float,
    cap_total_tokens: int,
) -> bool:
    """Predict whether the validator will mark this rollout as truncated.

    Replicates the validator's logic in ``verify_termination`` +
    ``is_cap_truncation`` using the ``token_logprobs`` already in the GRAIL
    commit. Returns True if the rollout would count toward the validator's
    truncation budget.
    """
    rollout_meta = commit.get("rollout", {}) or {}
    prompt_length = int(rollout_meta.get("prompt_length", 0))
    completion_length = int(rollout_meta.get("completion_length", 0))
    token_logprobs = rollout_meta.get("token_logprobs") or []
    tokens = commit.get("tokens") or []
    total_length = prompt_length + completion_length

    if not tokens or not token_logprobs:
        return True
    if not eos_set:
        return False  # can't check; assume good

    last_tok = int(tokens[-1])
    last_logp = float(token_logprobs[-1])
    last_in_eos = last_tok in eos_set
    p_stop_ok = last_in_eos and last_logp >= min_eos_logp

    if total_length >= cap_total_tokens:
        # Path 1: cap-truncation. Validator's is_cap_truncation: bad unless
        # last is EOS AND p_stop_ok.
        return not p_stop_ok
    # Path 2: natural-EOS. Validator's verify_termination requires
    # last_in_eos AND p_stop_ok.
    return not p_stop_ok


def _zone_screen_passes(partial_rewards: list[float], *, bootstrap: bool) -> bool:
    """Whether partial rollouts justify generating the remaining M−2 samples."""
    if bootstrap or len(partial_rewards) < _ZONE_SCREEN_ROLLOUTS:
        return True
    # Steady state: two identical binary rewards → σ=0 at n=2; skip the other 6.
    return partial_rewards[0] != partial_rewards[1]


def _compute_merkle_root(rollouts) -> str:
    """Compute Merkle root over rollout leaves — returns 64-char hex.

    Uses canonical JSON (sort_keys=True, compact separators) for dict/list
    serialisation so the root is deterministic across Python
    implementations and refactor-stable against dict-construction-order
    changes.
    """
    import hashlib
    import json

    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(json.dumps(r.tokens, separators=(",", ":")).encode())
        h.update(json.dumps(r.reward).encode())
        h.update(json.dumps(r.commit, sort_keys=True, separators=(",", ":")).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        new = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            new.append(hashlib.sha256(left + right).digest())
        leaves = new
    return leaves[0].hex()


def _current_drand_round_at_send() -> int:
    """Drand quicknet round currently in progress at wall-clock now.

    Called just before POSTing /submit so the attached round matches what
    the validator sees at receipt (modulo the 1-round tolerance). Uses
    chain params cached at process start; one drand period of skew is
    tolerated by the validator.
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    return compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])


class MiningEngine:
    """Two-GPU mining: generation on GPU 0, GRAIL proofs on GPU 1.

    When two GPUs are present:
    - cuda:0 never runs GRAIL — prep continues during submit while cuda:1 proofs.
    - Two concurrent prep jobs (one per GPU) plus dual-GPU prompt racing inline.
    - Steady-state 2-rollout zone screen before full 8-sample generation.
    """

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment",
        *,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override: str | None = None,
        gen_backend: HFGenerator | VLLMGenerator | None = None,
        predictor: BetaBucketPredictor | None = None,
        predictor_save_path: str | None = None,
        predictor_save_every: int = 50,
        attn_implementation: str | None = None,
    ) -> None:
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.env = env
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self.validator_url_override = validator_url_override
        self._dual_gpu = vllm_gpu != proof_gpu

        # One writer per GPU — generation and GRAIL may run concurrently on
        # separate devices (e.g. B300×2) but never two jobs on the same GPU.
        self._gen_lock = threading.Lock()
        self._proof_lock = threading.Lock()
        self._prep_backend_lock = threading.Lock()
        self._prep_backend_counter = 0
        self._grail_in_flight = False

        # Lazy imports for heavy deps — keep module import cheap.
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)
        self._rng = _random.Random()

        # Optional pluggable generation backend (vLLM). When set, all rollouts
        # route through it and the HF-based dual-race is bypassed (vLLM is
        # ~3-5× faster, so the parallelism over two HF copies is moot).
        self.gen_backend = gen_backend
        self._using_vllm = gen_backend is not None and getattr(
            gen_backend, "is_vllm", False,
        )

        # Optional σ-predictor for prompt selection. None ⇒ uniform random.
        self._predictor = predictor
        # Paired termination predictor tracks per-bucket termination quality
        # (well-terminated = ≤ _LOCAL_MAX_TRUNCATED cap-hits) separately from
        # zone yield. Combined Thompson product steers prompt picks toward
        # buckets that produce BOTH varied AND terminating completions.
        self._term_predictor = BetaBucketPredictor() if predictor is not None else None
        self._predictor_save_path = predictor_save_path
        self._predictor_save_every = max(1, predictor_save_every)
        self._predictor_updates_since_save = 0

        from reliquary.shared.hf_compat import resolve_attn_implementation

        self._attn_implementation = (
            attn_implementation or resolve_attn_implementation()
        )

        if self._using_vllm:
            logger.info(
                "vLLM generation backend on cuda:%d; GRAIL on cuda:%d (HF)",
                vllm_gpu, proof_gpu,
            )
        elif self._dual_gpu:
            logger.info(
                "Dual-GPU pipeline: generation=cuda:%d (always), "
                "GRAIL=cuda:%d only, 2× parallel prep + zone screen",
                vllm_gpu, proof_gpu,
            )
        else:
            logger.info(
                "Single-GPU mode: generation and GRAIL share cuda:%d",
                vllm_gpu,
            )

        if self._predictor is not None:
            logger.info("σ-predictor enabled: %s", self._predictor.stats_line())

    def _maybe_update_predictor(
        self,
        problem: dict,
        *,
        in_zone: bool,
        terminated_well: bool | None = None,
    ) -> None:
        """Update σ-predictor (zone yield) AND termination predictor.

        ``in_zone`` = bundle had σ ≥ threshold AND passed local truncation cap.
        ``terminated_well`` = the prompt produced ≤ _LOCAL_MAX_TRUNCATED
        cap-hits. Defaults to ``in_zone`` if not provided (back-compat).
        """
        if getattr(self, "_predictor", None) is None:
            return
        prompt = problem.get("prompt", "") if isinstance(problem, dict) else ""
        self._predictor.update(prompt, in_zone=in_zone)
        if self._term_predictor is not None:
            term_ok = in_zone if terminated_well is None else terminated_well
            self._term_predictor.update(prompt, in_zone=term_ok)

        self._predictor_updates_since_save += 1
        if (
            self._predictor_save_path
            and self._predictor_updates_since_save >= self._predictor_save_every
        ):
            try:
                self._predictor.save(self._predictor_save_path)
            except Exception:
                logger.exception(
                    "σ-predictor save to %s failed; continuing",
                    self._predictor_save_path,
                )
            self._predictor_updates_since_save = 0

    def _max_parallel_bundle_prep(
        self,
        state,
        *,
        staged_len: int,
        nearly_full: bool,
    ) -> int:
        """Concurrent ``_prepare_bundle`` jobs on dual-GPU rigs."""
        # vLLM mode: ALL generation goes through vLLM on cuda:0. The HF proof
        # model on cuda:1 is reserved exclusively for GRAIL. Attempts to also
        # use HF for parallel prep cause two problems:
        #   1. Autoregressive HF generation at 8192 max-tokens is genuinely
        #      slow (~100-300s for 8 rollouts), so it dominates the prep mix
        #      and blocks the vLLM slot from rotating in.
        #   2. HF prep holds ``_proof_lock``, blocking GRAIL behind 100s+ of
        #      generation. GRAIL latency spikes (191s observed) push bundles
        #      past the submission window.
        # The cuda:1 GPU appearing "idle" outside GRAIL bursts is the correct
        # state — reserving it for low-latency GRAIL is more valuable than
        # filling it with slow auxiliary work.
        if self._using_vllm:
            return 1
        if not self._dual_gpu or self.vllm_model is None or self.hf_model is None:
            return 1
        return 2

    def _alloc_prep_backend(self, *, grail_active: bool = False) -> tuple[object, threading.Lock, int]:
        """Allocate a prep backend. In vLLM mode, always vLLM (cuda:0)."""
        if self._using_vllm:
            return (None, None, self.vllm_gpu)

        # Non-vLLM (HF dual-race) path — unchanged.
        if grail_active:
            return self.vllm_model, self._gen_lock, self.vllm_gpu
        with self._prep_backend_lock:
            slot = self._prep_backend_counter % 2
            self._prep_backend_counter += 1
        if slot == 0:
            return self.vllm_model, self._gen_lock, self.vllm_gpu
        return self.hf_model, self._proof_lock, self.proof_gpu

    def _poll_delay_seconds(
        self,
        state,
        *,
        staged_len: int,
        prep_in_flight: int,
        submissions_this_window: int,
        nearly_full: bool,
    ) -> float:
        """Seconds to sleep before the next ``/state`` poll."""
        from reliquary.protocol.submission import WindowState

        if state.state != WindowState.OPEN:
            if (
                staged_len < MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
                and prep_in_flight > 0
            ):
                return _STATE_POLL_DOWNTIME_BUSY
            return _STATE_POLL_DOWNTIME_IDLE
        if submissions_this_window >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
            return _STATE_POLL_CAP
        if staged_len > 0 or prep_in_flight > 0 or self._grail_in_flight:
            return _STATE_POLL_BUSY
        if nearly_full:
            return _STATE_POLL_BUSY
        if staged_len == 0:
            return 0.15
        return _STATE_POLL_OPEN

    def _maybe_log_heartbeat(
        self,
        state,
        *,
        staged_len: int,
        prep_in_flight: int,
        submissions_this_window: int,
        last_heartbeat: list[float],
    ) -> None:
        now = time.monotonic()
        if now - last_heartbeat[0] < _HEARTBEAT_SECONDS:
            return
        last_heartbeat[0] = now
        st = state.state.value if hasattr(state.state, "value") else state.state
        logger.info(
            "status window=%d state=%s queue=%d/%d prep=%d subs=%d/8 "
            "valid=%d grail=%s",
            state.window_n,
            st,
            staged_len,
            MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
            prep_in_flight,
            submissions_this_window,
            getattr(state, "valid_submissions", -1),
            "busy" if self._grail_in_flight else "idle",
        )
        if self._predictor is not None:
            logger.info("zone-predictor %s", self._predictor.stats_line())
            if self._term_predictor is not None:
                logger.info("term-predictor %s", self._term_predictor.stats_line())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,  # v2.0 param kept for CLI compat; ignored
        use_drand: bool = True,
    ) -> list:
        """Poll state, stage in-zone rollouts during downtime, submit when OPEN.

        Returns the list of BatchSubmissionResponse objects collected
        across the loop. The loop exits only on external cancellation
        (asyncio.CancelledError) or if env becomes fully cooldown'd.
        """
        import httpx

        from reliquary.constants import POLL_INTERVAL_SECONDS
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
            get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import WindowState

        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        staged = StagedQueue()
        results: list = []
        local_n = 0
        local_hash = ""
        tracked_window_n: int | None = None
        submissions_this_window = 0
        prep_in_flight = 0
        last_heartbeat: list[float] = [0.0]

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                try:
                    state = await get_window_state_v2(url, client=client)
                except SubmissionError:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                except Exception as e:
                    logger.debug("state fetch failed: %s", e)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                prev_n = local_n
                try:
                    local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
                        state=state, local_n=local_n, local_hash=local_hash,
                        local_model=self.hf_model,
                        download_fn=_hf_download,
                        load_fn=self._load_checkpoint,
                    )
                except Exception:
                    logger.exception("checkpoint pull failed; keeping local")

                if local_n > prev_n:
                    n_cleared = len(staged)
                    staged.clear()
                    logger.info(
                        "checkpoint %d→%d: cleared %d staged bundles",
                        prev_n, local_n, n_cleared,
                    )
                    if self._predictor is not None:
                        logger.info(
                            "checkpoint %d→%d: resetting σ-predictor (%s)",
                            prev_n, local_n, self._predictor.stats_line(),
                        )
                        self._predictor.reset()
                        self._predictor_updates_since_save = 0
                    if self._term_predictor is not None:
                        logger.info(
                            "checkpoint %d→%d: resetting term-predictor (%s)",
                            prev_n, local_n, self._term_predictor.stats_line(),
                        )
                        self._term_predictor.reset()
                    # Prune stale revisions from the HF cache. Without this,
                    # each checkpoint pull (~8 GB) accumulates indefinitely
                    # and fills the disk in ~24h.
                    if (
                        state.checkpoint_repo_id is not None
                        and state.checkpoint_revision is not None
                    ):
                        await _prune_old_checkpoint_revisions(
                            state.checkpoint_repo_id,
                            state.checkpoint_revision,
                        )

                cooldown_set = set(state.cooldown_prompts)
                bootstrap = state.window_n < BOOTSTRAP_WINDOWS

                if state.window_n != tracked_window_n:
                    tracked_window_n = state.window_n
                    submissions_this_window = 0
                    dropped = staged.purge(cooldown_set, local_hash)
                    if dropped:
                        logger.info(
                            "window %d: purged %d stale staged bundles",
                            state.window_n, dropped,
                        )

                nearly_full = (
                    state.state == WindowState.OPEN
                    and state.valid_submissions >= _BATCH_NEARLY_FULL
                )
                max_prep = self._max_parallel_bundle_prep(
                    state,
                    staged_len=len(staged),
                    nearly_full=nearly_full,
                )
                if (
                    len(staged) < staged.max_size
                    and prep_in_flight < max_prep
                    and (
                        self.vllm_model is not None
                        or getattr(self, "gen_backend", None) is not None
                    )
                ):
                    prep_cooldown = set(cooldown_set)
                    prep_excluded = staged.staged_prompt_indices()
                    prep_local_n = local_n
                    prep_local_hash = local_hash
                    prep_bootstrap = bootstrap
                    prep_window_n = state.window_n
                    if max_prep >= 2:
                        prep_model, prep_lock, prep_gpu = self._alloc_prep_backend(
                            grail_active=self._grail_in_flight,
                        )
                    elif self._using_vllm:
                        # Single-slot vLLM: model=None routes through gen_backend
                        prep_model, prep_lock, prep_gpu = (
                            None, None, self.vllm_gpu,
                        )
                    else:
                        prep_model, prep_lock, prep_gpu = (
                            self.vllm_model, self._gen_lock, self.vllm_gpu,
                        )
                    prep_in_flight += 1

                    use_batched = getattr(self, "_using_vllm", False)

                    async def _run_prep(
                        _model=prep_model,
                        _lock=prep_lock,
                        _gpu=prep_gpu,
                        _use_batched=use_batched,
                    ) -> None:
                        nonlocal prep_in_flight
                        try:
                            if _use_batched:
                                # vLLM: batched multi-prompt prep — one call
                                # produces 0..K bundles in the time of one
                                # sequence. Stage them all.
                                bundles = await asyncio.to_thread(
                                    self._prepare_bundles_batched,
                                    prep_cooldown,
                                    prep_excluded,
                                    prep_local_n,
                                    prep_local_hash,
                                    prep_bootstrap,
                                )
                                for bundle in bundles:
                                    if staged.push(bundle):
                                        logger.info(
                                            "staged prompt=%d σ=%.3f queue=%d/%d "
                                            "checkpoint=%d@%s gpu=prep:cuda:%d",
                                            bundle.prompt_idx,
                                            bundle.sigma,
                                            len(staged),
                                            staged.max_size,
                                            bundle.checkpoint_n,
                                            bundle.checkpoint_hash[:12],
                                            _gpu,
                                        )
                                    else:
                                        logger.debug(
                                            "staged queue full; dropped prompt=%d",
                                            bundle.prompt_idx,
                                        )
                                        break
                            else:
                                bundle = await asyncio.to_thread(
                                    self._prepare_bundle,
                                    prep_cooldown,
                                    prep_excluded,
                                    prep_local_n,
                                    prep_local_hash,
                                    prep_bootstrap,
                                    gen_model=_model,
                                    gen_lock=_lock,
                                )
                                if bundle is not None:
                                    if staged.push(bundle):
                                        logger.info(
                                            "staged prompt=%d σ=%.3f queue=%d/%d "
                                            "checkpoint=%d@%s gpu=prep:cuda:%d",
                                            bundle.prompt_idx,
                                            bundle.sigma,
                                            len(staged),
                                            staged.max_size,
                                            bundle.checkpoint_n,
                                            bundle.checkpoint_hash[:12],
                                            _gpu,
                                        )
                                    else:
                                        logger.debug(
                                            "staged queue full; dropped prompt=%d",
                                            bundle.prompt_idx,
                                        )
                        except RuntimeError:
                            logger.debug(
                                "prep: no eligible prompt (window=%d queue=%d/%d)",
                                prep_window_n, len(staged), staged.max_size,
                            )
                        except Exception:
                            logger.exception("prep failed for window=%d", prep_window_n)
                        finally:
                            prep_in_flight -= 1

                    asyncio.create_task(_run_prep())

                if state.state != WindowState.OPEN:
                    self._maybe_log_heartbeat(
                        state,
                        staged_len=len(staged),
                        prep_in_flight=prep_in_flight,
                        submissions_this_window=submissions_this_window,
                        last_heartbeat=last_heartbeat,
                    )
                    await asyncio.sleep(
                        self._poll_delay_seconds(
                            state,
                            staged_len=len(staged),
                            prep_in_flight=prep_in_flight,
                            submissions_this_window=submissions_this_window,
                            nearly_full=nearly_full,
                        )
                    )
                    continue

                randomness = state.randomness
                if not randomness:
                    await asyncio.sleep(_STATE_POLL_BUSY)
                    continue

                if submissions_this_window >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
                    self._maybe_log_heartbeat(
                        state,
                        staged_len=len(staged),
                        prep_in_flight=prep_in_flight,
                        submissions_this_window=submissions_this_window,
                        last_heartbeat=last_heartbeat,
                    )
                    await asyncio.sleep(_STATE_POLL_CAP)
                    continue

                # If the validator's batch is already full and we have a bundle,
                # don't enter the drain loop — _submit_bundle would re-queue on
                # every iteration and spam the log. Wait for the window to seal.
                if nearly_full and len(staged) > 0:
                    await asyncio.sleep(_STATE_POLL_CAP)
                    continue

                # Drain the staged queue — up to the per-window submission cap.
                while (
                    len(staged) > 0
                    and submissions_this_window < MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
                ):
                    bundle = staged.pop()
                    discard = bundle_discard_reason(
                        bundle,
                        local_hash=local_hash,
                        cooldown_set=cooldown_set,
                    )
                    if discard:
                        logger.debug(
                            "discard staged prompt=%d reason=%s",
                            bundle.prompt_idx, discard,
                        )
                        continue

                    try:
                        resp = await self._submit_bundle(
                            bundle,
                            state,
                            randomness,
                            local_hash,
                            url,
                            client,
                            submit_batch_v2,
                            source="queue",
                        )
                        if resp is None:
                            if not staged.push_front(bundle):
                                logger.warning(
                                    "re-queue failed (full) prompt=%d — bundle dropped",
                                    bundle.prompt_idx,
                                )
                            break
                        results.append(resp)
                        submissions_this_window += 1
                    except SubmissionError as exc:
                        logger.error("submit failed (queue): %s", exc)
                        break

                if submissions_this_window >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
                    await asyncio.sleep(_STATE_POLL_CAP)
                    continue

                if len(staged) > 0:
                    await asyncio.sleep(_STATE_POLL_BUSY)
                    continue

                if prep_in_flight > 0:
                    await asyncio.sleep(_STATE_POLL_BUSY)
                    continue

                if nearly_full:
                    await asyncio.sleep(_STATE_POLL_BUSY)
                    continue

                try:
                    bundle = await asyncio.to_thread(
                        self._prepare_bundle,
                        cooldown_set,
                        staged.staged_prompt_indices(),
                        local_n,
                        local_hash,
                        bootstrap,
                    )
                except RuntimeError:
                    logger.info("env fully in cooldown; sleeping")
                    await asyncio.sleep(5)
                    continue

                if bundle is None:
                    await asyncio.sleep(_STATE_POLL_OPEN)
                    continue

                discard = bundle_discard_reason(
                    bundle,
                    local_hash=local_hash,
                    cooldown_set=cooldown_set,
                )
                if discard:
                    logger.debug(
                        "discard inline prompt=%d reason=%s",
                        bundle.prompt_idx, discard,
                    )
                    await asyncio.sleep(_STATE_POLL_OPEN)
                    continue

                try:
                    resp = await self._submit_bundle(
                        bundle,
                        state,
                        randomness,
                        local_hash,
                        url,
                        client,
                        submit_batch_v2,
                        source="inline",
                    )
                    if resp is None:
                        if not staged.push(bundle):
                            logger.warning(
                                "re-queue failed (full) inline prompt=%d — dropped",
                                bundle.prompt_idx,
                            )
                        await asyncio.sleep(_STATE_POLL_BUSY)
                        continue
                    results.append(resp)
                    submissions_this_window += 1
                except SubmissionError as exc:
                    logger.error("submit failed (inline): %s", exc)

                self._maybe_log_heartbeat(
                    state,
                    staged_len=len(staged),
                    prep_in_flight=prep_in_flight,
                    submissions_this_window=submissions_this_window,
                    last_heartbeat=last_heartbeat,
                )
                await asyncio.sleep(
                    self._poll_delay_seconds(
                        state,
                        staged_len=len(staged),
                        prep_in_flight=prep_in_flight,
                        submissions_this_window=submissions_this_window,
                        nearly_full=nearly_full,
                    )
                )

        return results

    def _prepare_bundle(
        self,
        cooldown_set: set[int],
        excluded_prompts: set[int],
        local_n: int,
        local_hash: str,
        bootstrap: bool,
        *,
        max_prompt_attempts: int = _PREP_PROMPT_ATTEMPTS,
        gen_model=None,
        gen_lock: threading.Lock | None = None,
    ) -> StagedBundle | None:
        """Generate rollouts; retry prompts until one passes the σ zone filter."""
        # vLLM mode: route through sequential. The caller passes either
        # (gen_model=None) → vLLM via gen_backend, or (gen_model=hf_model,
        # gen_lock=proof_lock) → HF on cuda:1 for the parallel prep slot.
        if getattr(self, "_using_vllm", False):
            return self._prepare_bundle_sequential(
                cooldown_set,
                excluded_prompts,
                local_n,
                local_hash,
                bootstrap,
                max_prompt_attempts=max_prompt_attempts,
                gen_model=gen_model,
                gen_lock=gen_lock,
            )
        if getattr(self, "_dual_gpu", False) and gen_model is None and not self._grail_in_flight:
            return self._prepare_bundle_dual_race(
                cooldown_set,
                excluded_prompts,
                local_n,
                local_hash,
                bootstrap,
                max_prompt_attempts=max_prompt_attempts,
            )
        if getattr(self, "_dual_gpu", False) and gen_model is None and self._grail_in_flight:
            gen_model = self.vllm_model
            gen_lock = self._gen_lock
        return self._prepare_bundle_sequential(
            cooldown_set,
            excluded_prompts,
            local_n,
            local_hash,
            bootstrap,
            max_prompt_attempts=max_prompt_attempts,
            gen_model=gen_model,
            gen_lock=gen_lock,
        )

    def _prepare_bundle_dual_race(
        self,
        cooldown_set: set[int],
        excluded_prompts: set[int],
        local_n: int,
        local_hash: str,
        bootstrap: bool,
        *,
        max_prompt_attempts: int,
    ) -> StagedBundle | None:
        """Try two prompts in parallel (one per GPU); first in-zone bundle wins."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tried: set[int] = set()
        blocked = set(excluded_prompts)
        backends = (
            (self.vllm_model, self._gen_lock, self.vllm_gpu),
            (self.hf_model, self._proof_lock, self.proof_gpu),
        )
        rounds = max(1, (max_prompt_attempts + 1) // 2)

        for _round in range(rounds):
            jobs: list[tuple[int, dict, object, threading.Lock]] = []
            for model, lock, _gpu in backends:
                if len(jobs) >= 2:
                    break
                try:
                    prompt_idx = pick_with_predictor(
                        self.env,
                        cooldown_set,
                        excluded_prompts=blocked | tried,
                        rng=self._rng,
                        predictor=self._predictor,
                        uniform_fallback=pick_prompt_idx,
                        tokenizer=self.tokenizer,
                    )
                except RuntimeError:
                    break
                tried.add(prompt_idx)
                jobs.append(
                    (prompt_idx, self.env.get_problem(prompt_idx), model, lock)
                )

            if not jobs:
                break

            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                futures = {
                    pool.submit(
                        self._try_prompt_bundle,
                        prompt_idx,
                        problem,
                        local_n,
                        local_hash,
                        bootstrap,
                        gen_model=model,
                        gen_lock=lock,
                    ): prompt_idx
                    for prompt_idx, problem, model, lock in jobs
                }
                for future in as_completed(futures):
                    bundle = future.result()
                    if bundle is not None:
                        return bundle

        logger.debug(
            "dual-race exhausted %d round(s) — no in-zone bundle",
            rounds,
        )
        return None

    def _try_prompt_bundle(
        self,
        prompt_idx: int,
        problem: dict,
        local_n: int,
        local_hash: str,
        bootstrap: bool,
        *,
        gen_model,
        gen_lock: threading.Lock,
    ) -> StagedBundle | None:
        logger.info(
            "prep start prompt=%d (checkpoint=%d@%s bootstrap=%s)",
            prompt_idx, local_n, local_hash[:12] if local_hash else "?",
            bootstrap,
        )
        t0 = time.monotonic()
        generations = self._generate_m_rollouts(
            problem,
            gen_model=gen_model,
            gen_lock=gen_lock,
            bootstrap=bootstrap,
        )
        gen_s = time.monotonic() - t0
        if len(generations) < M_ROLLOUTS:
            if len(generations) == 0:
                logger.info(
                    "prep miss prompt=%d gen=%.1fs reason=screen_or_head_trunc "
                    "(σ-screen failed OR all head rollouts hit max_tokens, "
                    "tail skipped — not queued)",
                    prompt_idx, gen_s,
                )
                self._log_prep_rollouts(
                    prompt_idx=prompt_idx,
                    problem=problem,
                    generations=[],
                    rewards=[],
                    sigma=0.0,
                    source="prep",
                    outcome="miss_screen_or_head_trunc",
                )
                # Either rejection ≈ confident "not useful" — treat as σ=0 bucket.
                self._maybe_update_predictor(problem, in_zone=False)
            else:
                logger.info(
                    "prep miss prompt=%d gen=%.1fs reason=gen_short (%d/%d, not queued)",
                    prompt_idx, gen_s, len(generations), M_ROLLOUTS,
                )
                partial_rewards = _rewards_from_generations(
                    self.env, problem, generations, self.tokenizer,
                )
                partial_sigma = (
                    rewards_std(partial_rewards)
                    if len(partial_rewards) >= 2
                    else 0.0
                )
                self._log_prep_rollouts(
                    prompt_idx=prompt_idx,
                    problem=problem,
                    generations=generations,
                    rewards=partial_rewards,
                    sigma=partial_sigma,
                    source="prep",
                    outcome=f"miss_gen_short_{len(generations)}/{M_ROLLOUTS}",
                )
            return None

        rewards = _rewards_from_generations(
            self.env, problem, generations, self.tokenizer,
        )
        sigma = rewards_std(rewards)

        # Count rollouts that didn't reach EOS — these would be marked
        # ``truncated`` by the validator and trigger ``bad_termination`` if
        # > MAX_TRUNCATED_PER_SUBMISSION. Use the proof model's EOS set since
        # it's what the validator will use to validate.
        eos_set = eos_set_from_model(self.hf_model, self.tokenizer)
        n_trunc = _count_truncated(generations, eos_set)

        if n_trunc > _LOCAL_MAX_TRUNCATED:
            logger.info(
                "prep miss prompt=%d σ=%.3f gen=%.1fs reason=too_truncated "
                "(%d/%d rollouts hit max_tokens without EOS — local cap=%d, "
                "validator-side cap=%d, not queued)",
                prompt_idx, sigma, gen_s, n_trunc, M_ROLLOUTS,
                _LOCAL_MAX_TRUNCATED, MAX_TRUNCATED_PER_SUBMISSION,
            )
            self._log_prep_rollouts(
                prompt_idx=prompt_idx,
                problem=problem,
                generations=generations,
                rewards=rewards,
                sigma=sigma,
                source="prep",
                outcome="miss_too_truncated",
            )
            self._maybe_update_predictor(problem, in_zone=False)
            return None

        if not _passes_zone_filter(rewards, bootstrap=bootstrap):
            logger.info(
                "prep miss prompt=%d σ=%.3f gen=%.1fs reason=out_of_zone "
                "(σ < %.2f threshold, not queued) rewards=%s",
                prompt_idx, sigma, gen_s,
                0.33 if bootstrap else 0.43,
                [int(r) for r in rewards],
            )
            self._log_prep_rollouts(
                prompt_idx=prompt_idx,
                problem=problem,
                generations=generations,
                rewards=rewards,
                sigma=sigma,
                source="prep",
                outcome="miss_out_of_zone",
            )
            self._maybe_update_predictor(problem, in_zone=False)
            return None

        # Mirror validator's MALFORMED_FINAL_ANSWER gate: skip bundles whose
        # reward<0.5 rollouts have an unclosed/empty/special-token final box.
        init_malformed = _compute_malformed_flags(
            generations, rewards, self.tokenizer,
        )
        n_malformed = sum(init_malformed)
        if n_malformed > 0:
            logger.info(
                "prep miss prompt=%d σ=%.3f gen=%.1fs reason=malformed_final_answer "
                "(%d/%d rollouts have malformed final \\boxed{})",
                prompt_idx, sigma, gen_s, n_malformed, M_ROLLOUTS,
            )
            self._log_prep_rollouts(
                prompt_idx=prompt_idx,
                problem=problem,
                generations=generations,
                rewards=rewards,
                sigma=sigma,
                source="prep",
                outcome="miss_malformed_final_answer",
            )
            self._maybe_update_predictor(problem, in_zone=False)
            return None

        logger.info(
            "prep hit prompt=%d σ=%.3f gen=%.1fs truncated=%d/%d rewards=%s → queued",
            prompt_idx, sigma, gen_s, n_trunc, M_ROLLOUTS,
            [int(r) for r in rewards],
        )
        self._log_prep_rollouts(
            prompt_idx=prompt_idx,
            problem=problem,
            generations=generations,
            rewards=rewards,
            sigma=sigma,
            source="prep",
            outcome="hit_queued",
        )
        self._maybe_update_predictor(problem, in_zone=True)
        return StagedBundle(
            prompt_idx=prompt_idx,
            problem=problem,
            generations=generations,
            rewards=rewards,
            sigma=sigma,
            checkpoint_n=local_n,
            checkpoint_hash=local_hash,
        )

    def _prepare_bundle_sequential(
        self,
        cooldown_set: set[int],
        excluded_prompts: set[int],
        local_n: int,
        local_hash: str,
        bootstrap: bool,
        *,
        max_prompt_attempts: int = _PREP_PROMPT_ATTEMPTS,
        gen_model=None,
        gen_lock: threading.Lock | None = None,
    ) -> StagedBundle | None:
        """Single-GPU sequential prompt attempts."""
        tried: set[int] = set()
        blocked = set(excluded_prompts)

        for attempt in range(1, max_prompt_attempts + 1):
            try:
                prompt_idx = pick_with_predictor(
                    self.env,
                    cooldown_set,
                    excluded_prompts=blocked | tried,
                    rng=self._rng,
                    predictor=getattr(self, "_predictor", None),
                    uniform_fallback=pick_prompt_idx,
                    tokenizer=self.tokenizer,
                )
            except RuntimeError:
                if attempt == 1:
                    raise
                logger.info(
                    "prep: no eligible prompts after %d attempt(s)",
                    attempt - 1,
                )
                return None

            # In vLLM mode, gen_model=None is the explicit "use vLLM" sentinel —
            # don't fall back to self.vllm_model (which is None there).
            using_vllm = getattr(self, "_using_vllm", False)
            effective_gen_model = (
                gen_model
                if gen_model is not None or using_vllm
                else self.vllm_model
            )
            effective_gen_lock = (
                gen_lock
                if gen_lock is not None or using_vllm
                else self._gen_lock
            )
            bundle = self._try_prompt_bundle(
                prompt_idx,
                self.env.get_problem(prompt_idx),
                local_n,
                local_hash,
                bootstrap,
                gen_model=effective_gen_model,
                gen_lock=effective_gen_lock,
            )
            if bundle is not None:
                logger.debug(
                    "prep in-zone attempt %d/%d prompt=%d σ=%.3f",
                    attempt, max_prompt_attempts, bundle.prompt_idx, bundle.sigma,
                )
                return bundle

            tried.add(prompt_idx)

        logger.debug(
            "prep exhausted %d attempts — no in-zone bundle",
            max_prompt_attempts,
        )
        return None

    def _log_prep_rollouts(
        self,
        *,
        prompt_idx: int,
        problem: dict,
        generations: list[dict],
        rewards: list[float] | None,
        sigma: float,
        source: str,
        outcome: str,
        canonical_tokens: list[int] | None = None,
        augmented_length: int | None = None,
    ) -> None:
        """Log every rollout for a prep attempt (queued or rejected)."""
        from reliquary.miner.rollout_log import log_staged_rollouts

        gens = list(generations)
        if gens and canonical_tokens is not None and augmented_length is not None:
            gens = rebind_rollouts_to_canonical_prompt(
                gens, canonical_tokens, augmented_length,
            )
        if not gens:
            logger.info(
                "rollout_detail begin prompt=%d source=%s outcome=%s "
                "n_rollouts=0 (no generations)",
                prompt_idx, source, outcome,
            )
            logger.info(
                "rollout_detail end prompt=%d source=%s outcome=%s",
                prompt_idx, source, outcome,
            )
            return

        rews = list(rewards) if rewards is not None else []
        if len(rews) < len(gens):
            rews.extend([0.0] * (len(gens) - len(rews)))

        log_staged_rollouts(
            logger,
            prompt_idx=prompt_idx,
            problem=problem,
            generations=gens,
            rewards=rews[: len(gens)],
            sigma=sigma,
            tokenizer=self.tokenizer,
            eos_set=eos_set_from_model(self.hf_model, self.tokenizer),
            source=source,
            outcome=outcome,
        )

    def _compute_p_stops(
        self,
        generations: list[dict],
        eos_set: set[int],
    ) -> list[float]:
        """Compute validator's p_stop metric for each rollout in the list.

        For rollouts whose last token is NOT in ``eos_set`` (or which are too
        short), returns 0.0 — they can never satisfy verify_termination Path 2.
        For EOS-terminated rollouts, returns
        ``sum(softmax(logits[seq_len-2])[eos_set])`` — the same quantity the
        validator computes via ``_gpu_p_stop``.

        Uses ``self.hf_model`` on ``self.proof_gpu`` (same loader path as the
        validator). Sequential per-rollout forward passes; pool-of-24 with
        mixed lengths typically runs in 3-6s on a B200.
        """
        import torch

        if not eos_set or not generations:
            return [0.0] * len(generations)

        device = self.proof_gpu
        eos_tensor = torch.tensor(sorted(eos_set), device=device)
        p_stops: list[float] = []

        for gen in generations:
            tokens = gen.get("tokens", [])
            if not tokens or int(tokens[-1]) not in eos_set or len(tokens) < 2:
                p_stops.append(0.0)
                continue
            input_ids = torch.tensor(
                [tokens], dtype=torch.long, device=device,
            )
            with torch.no_grad():
                outputs = self.hf_model(input_ids)
                logits = outputs.logits[0]
            probs = torch.softmax(logits[len(tokens) - 2], dim=-1)
            p_stops.append(float(probs[eos_tensor].sum().item()))
            del input_ids, outputs, logits, probs

        torch.cuda.empty_cache()
        return p_stops

    def _rescue_prompt(
        self,
        prompt_idx: int,
        problem: dict,
        prompt_tokens: list[int],
        max_new: int,
        eos_set: set[int],
        initial_generations: list[dict],
        initial_rewards: list[float],
        bootstrap: bool,
        *,
        initial_p_stops: list[float] | None = None,
    ) -> tuple[list[dict], list[float], float, int] | None:
        """Rejection-sampling rescue: pool more rollouts until we can pick 8
        that pass the validator's termination + zone gates.

        Triggered when an initial M=8 prep cycle missed only on truncation
        but had enough naturally terminating rollouts to suggest the prompt
        IS terminatable. Each additional vLLM call produces ``_RESCUE_BATCH_SIZE``
        rollouts; we stop when the pool has 7+ terminated rollouts or the
        pool size cap is reached.

        ``initial_p_stops`` may be passed in pre-computed (e.g., by the rescue
        gate in ``_prepare_bundles_batched``) to avoid recomputing the initial
        forward passes here.

        Returns ``(generations, rewards, σ, n_trunc_in_selection)`` for the
        chosen 8-rollout bundle, or ``None`` if rescue failed.
        """
        pool_gens: list[dict] = list(initial_generations)
        pool_rewards: list[float] = list(initial_rewards)
        # Validator p_stop per rollout — needed because `last_token in eos_set`
        # is necessary but NOT sufficient for valid termination. The validator
        # also checks p_stop ≥ MIN_EOS_PROBABILITY. Without this, sampling-fluke
        # EOS rollouts (where the model emitted a low-probability <|endoftext|>
        # mid-ramble) get counted as valid terminators by selection and the
        # downstream submission is rejected on bad_termination.
        if initial_p_stops is not None:
            assert len(initial_p_stops) == len(pool_gens), (
                "initial_p_stops length mismatch"
            )
            pool_p_stops: list[float] = list(initial_p_stops)
        else:
            pool_p_stops = self._compute_p_stops(pool_gens, eos_set)
        # Validator's MALFORMED_FINAL_ANSWER gate fires on any reward<0.5
        # rollout whose final \boxed{}/\fbox{} is empty, contains a special
        # token, or is unclosed. Cap-truncated rollouts that opened a fresh
        # box near token 8190 hit "unclosed" routinely. Flag now and exclude
        # from selection — they are poison and would reject the whole bundle.
        pool_malformed: list[bool] = _compute_malformed_flags(
            pool_gens, pool_rewards, self.tokenizer,
        )

        rescue_t0 = time.monotonic()
        rescue_calls = 0
        last_selection: tuple[list[dict], list[float], float, int] | None = None

        # Loop is feasibility-driven: keep generating until we can actually
        # assemble a σ-passing 8-rollout bundle (where "valid termination"
        # means EOS AND p_stop ≥ MIN_EOS_PROBABILITY), not just until we have
        # ≥7 EOS-last-token rollouts.
        while True:
            last_selection = _select_8_passing(
                pool_gens, pool_rewards, eos_set,
                bootstrap=bootstrap, rng=self._rng,
                p_stops=pool_p_stops,
                malformed_flags=pool_malformed,
            )
            if last_selection is not None:
                break  # found a passing bundle
            if len(pool_gens) >= _RESCUE_MAX_POOL_SIZE:
                break  # pool capped; give up
            batch_size = min(
                _RESCUE_BATCH_SIZE, _RESCUE_MAX_POOL_SIZE - len(pool_gens),
            )
            try:
                new_gens = self.gen_backend.generate(
                    prompt_tokens, batch_size,
                    max_new_tokens=max_new,
                    eos_set=eos_set,
                    tokenizer=self.tokenizer,
                )
            except Exception:
                logger.exception(
                    "rescue prompt=%d: generate failed mid-pool", prompt_idx,
                )
                return None
            rescue_calls += 1
            if not new_gens:
                break
            new_rewards = _rewards_from_generations(
                self.env, problem, new_gens, self.tokenizer,
            )
            new_p_stops = self._compute_p_stops(new_gens, eos_set)
            new_malformed = _compute_malformed_flags(
                new_gens, new_rewards, self.tokenizer,
            )
            pool_gens.extend(new_gens)
            pool_rewards.extend(new_rewards)
            pool_p_stops.extend(new_p_stops)
            pool_malformed.extend(new_malformed)

        rescue_s = time.monotonic() - rescue_t0

        if last_selection is None:
            # Diagnostic: bucket the final pool by (validly-terminated, reward).
            # The "border_*" buckets are rollouts that END in EOS but have
            # p_stop < MIN_EOS_PROBABILITY — these LOOK terminated but the
            # validator would reject them. Splitting them out makes it clear
            # whether the bottleneck is "model can't terminate" vs "model is
            # picking low-probability EOS as a sampling fluke".
            term_pos = term_neg = bord_pos = bord_neg = trunc_pos = trunc_neg = 0
            n_malformed = 0
            for i, gen in enumerate(pool_gens):
                if pool_malformed[i]:
                    n_malformed += 1
                    continue
                tokens = gen.get("tokens", [])
                is_eos = bool(tokens) and int(tokens[-1]) in eos_set
                is_valid_term = is_eos and pool_p_stops[i] >= _LOCAL_MIN_EOS_PROB
                is_pos = pool_rewards[i] >= 0.5
                if is_valid_term:
                    if is_pos:
                        term_pos += 1
                    else:
                        term_neg += 1
                elif is_eos:
                    # EOS but low p_stop — sampling-fluke EOS
                    if is_pos:
                        bord_pos += 1
                    else:
                        bord_neg += 1
                else:
                    if is_pos:
                        trunc_pos += 1
                    else:
                        trunc_neg += 1
            logger.info(
                "rescue prompt=%d FAIL: pool=%d "
                "term=%d+/%d- border=%d+/%d- (EOS p_stop<%.2f) "
                "trunc=%d+/%d- malformed=%d "
                "(need ≥7 valid-term AND σ-feasible w/ ≤1 invalid; "
                "extra_calls=%d, %.1fs)",
                prompt_idx, len(pool_gens),
                term_pos, term_neg,
                bord_pos, bord_neg, _LOCAL_MIN_EOS_PROB,
                trunc_pos, trunc_neg, n_malformed,
                rescue_calls, rescue_s,
            )
            return None

        gens, rewards, sigma, n_trunc_sel = last_selection
        logger.info(
            "rescue prompt=%d HIT: pool=%d → 8 picked σ=%.3f trunc=%d "
            "(extra_calls=%d, %.1fs)",
            prompt_idx, len(pool_gens), sigma, n_trunc_sel,
            rescue_calls, rescue_s,
        )
        return last_selection

    def _prepare_bundles_batched(
        self,
        cooldown_set: set[int],
        excluded_prompts: set[int],
        local_n: int,
        local_hash: str,
        bootstrap: bool,
        *,
        n_prompts: int = _BATCHED_PREP_K,
    ) -> list[StagedBundle]:
        """Batched multi-prompt prep — picks K prompts, generates M_ROLLOUTS
        each in ONE vLLM call, returns all in-zone bundles.

        Wall-clock per call ≈ time for the slowest 8K-token sequence.
        Returns 0..K StagedBundles. Replaces ``_prepare_bundle`` in vLLM mode.

        Skips the 2-rollout σ-screen (parallel generation means no time
        savings from aborting individual prompts early). Applies a
        ``_BATCHED_HEAD_SCREEN_ROLLOUTS`` trunc-screen before tail generation.
        Still applies the same downstream zone-filter + truncation +
        termination checks per prompt.
        """
        if getattr(self, "gen_backend", None) is None:
            # No vLLM — fall back to the single-prompt path so callers can
            # use this method uniformly.
            single = self._prepare_bundle(
                cooldown_set, excluded_prompts, local_n, local_hash, bootstrap,
            )
            return [single] if single is not None else []

        # 1. Pick K prompts via dual predictors (zone × termination), with
        # hard exclusion of well-known-bad buckets and length filter.
        tried: set[int] = set(excluded_prompts)
        picks: list[tuple[int, dict]] = []
        for _ in range(n_prompts):
            try:
                prompt_idx = pick_with_predictor(
                    self.env,
                    cooldown_set,
                    excluded_prompts=tried,
                    rng=self._rng,
                    predictor=getattr(self, "_predictor", None),
                    term_predictor=getattr(self, "_term_predictor", None),
                    uniform_fallback=pick_prompt_idx,
                    tokenizer=self.tokenizer,
                )
            except RuntimeError:
                break
            tried.add(prompt_idx)
            picks.append((prompt_idx, self.env.get_problem(prompt_idx)))

        if not picks:
            logger.info("batched prep: no eligible prompts")
            return []

        # 2. Tokenize: augmented for generation, canonical for submitted rollouts.
        prompts_tokens: list[list[int]] = []
        canonical_tokens_list: list[list[int]] = []
        augmented_lengths: list[int] = []
        for _prompt_idx, problem in picks:
            canonical = canonical_prompt_tokens(problem["prompt"], self.tokenizer)
            augmented = generation_prompt_tokens(problem["prompt"], self.tokenizer)
            canonical_tokens_list.append(canonical)
            prompts_tokens.append(augmented)
            augmented_lengths.append(len(augmented))

        max_new_per_prompt = [
            min(
                self.max_new_tokens,
                _VLLM_BATCHED_MAX_NEW,
                max_new_tokens_for_generation(
                    problem["prompt"], self.tokenizer,
                ),
            )
            for _prompt_idx, problem in picks
        ]

        eos_set = eos_set_from_model(self.hf_model, self.tokenizer)

        logger.info(
            "batched prep start: %d prompts × M=%d rollouts "
            "(max_new range=[%d, %d], head_screen=%d, eos=%s)",
            len(picks), M_ROLLOUTS,
            min(max_new_per_prompt), max(max_new_per_prompt),
            _BATCHED_HEAD_SCREEN_ROLLOUTS,
            sorted(eos_set),
        )

        # 3. Head trunc-screen then tail generation (or one-shot M=8 if disabled).
        t0 = time.monotonic()
        all_rollouts: list[list[dict]] = [[] for _ in picks]
        try:
            head_n = min(_BATCHED_HEAD_SCREEN_ROLLOUTS, M_ROLLOUTS)
            if head_n <= 0 or head_n >= M_ROLLOUTS:
                all_rollouts = self.gen_backend.generate_batch(
                    prompts_tokens, M_ROLLOUTS,
                    max_new_tokens=max_new_per_prompt,
                    eos_set=eos_set,
                    tokenizer=self.tokenizer,
                )
            else:
                head_all = self.gen_backend.generate_batch(
                    prompts_tokens, head_n,
                    max_new_tokens=max_new_per_prompt,
                    eos_set=eos_set,
                    tokenizer=self.tokenizer,
                )
                active_indices: list[int] = []
                for i, (prompt_idx, problem) in enumerate(picks):
                    head_gens = head_all[i] if i < len(head_all) else []
                    if len(head_gens) < head_n:
                        logger.info(
                            "batched prep miss prompt=%d reason=head_gen_short "
                            "(%d/%d)",
                            prompt_idx, len(head_gens), head_n,
                        )
                        head_rewards = (
                            _rewards_from_generations(
                                self.env, problem, head_gens, self.tokenizer,
                            )
                            if head_gens
                            else []
                        )
                        head_sigma = (
                            rewards_std(head_rewards)
                            if len(head_rewards) >= 2
                            else 0.0
                        )
                        self._log_prep_rollouts(
                            prompt_idx=prompt_idx,
                            problem=problem,
                            generations=head_gens,
                            rewards=head_rewards,
                            sigma=head_sigma,
                            source="batched_prep",
                            outcome=f"miss_head_gen_short_{len(head_gens)}/{head_n}",
                            canonical_tokens=canonical_tokens_list[i],
                            augmented_length=augmented_lengths[i],
                        )
                        self._maybe_update_predictor(
                            problem, in_zone=False, terminated_well=False,
                        )
                        continue
                    n_head_trunc = _count_truncated(head_gens, eos_set)
                    if n_head_trunc >= len(head_gens) and eos_set:
                        logger.info(
                            "batched prep miss prompt=%d reason=head_trunc_screen "
                            "(%d/%d head rollouts hit max_tokens without EOS)",
                            prompt_idx, n_head_trunc, len(head_gens),
                        )
                        head_rewards = _rewards_from_generations(
                            self.env, problem, head_gens, self.tokenizer,
                        )
                        head_sigma = (
                            rewards_std(head_rewards)
                            if len(head_rewards) >= 2
                            else 0.0
                        )
                        self._log_prep_rollouts(
                            prompt_idx=prompt_idx,
                            problem=problem,
                            generations=head_gens,
                            rewards=head_rewards,
                            sigma=head_sigma,
                            source="batched_prep",
                            outcome="miss_head_trunc_screen",
                            canonical_tokens=canonical_tokens_list[i],
                            augmented_length=augmented_lengths[i],
                        )
                        self._maybe_update_predictor(
                            problem, in_zone=False, terminated_well=False,
                        )
                        continue
                    active_indices.append(i)
                    all_rollouts[i] = list(head_gens)

                if not active_indices:
                    gen_s = time.monotonic() - t0
                    logger.info(
                        "batched prep result: 0/%d bundles in-zone "
                        "(gen=%.1fs, all failed head trunc screen)",
                        len(picks), gen_s,
                    )
                    return []

                tail_n = M_ROLLOUTS - head_n
                active_prompts = [prompts_tokens[i] for i in active_indices]
                active_max_new = [max_new_per_prompt[i] for i in active_indices]
                tail_all = self.gen_backend.generate_batch(
                    active_prompts, tail_n,
                    max_new_tokens=active_max_new,
                    eos_set=eos_set,
                    tokenizer=self.tokenizer,
                )
                for j, i in enumerate(active_indices):
                    tail_gens = tail_all[j] if j < len(tail_all) else []
                    all_rollouts[i] = all_rollouts[i] + list(tail_gens)
        except Exception:
            logger.exception("batched generate_batch failed")
            return []
        gen_s = time.monotonic() - t0
        logger.info(
            "batched prep gen done in %.1fs for %d prompts → "
            "evaluating zone + termination per prompt",
            gen_s, len(picks),
        )

        # 4. Two-pass per-prompt evaluation:
        #   Pass A: classify each prompt as hit / clear_miss / rescue_candidate.
        #   Pass B: rank rescue_candidates by (n_init_term, σ) descending and
        #           attempt rescue on the top _RESCUE_MAX_PER_CYCLE. Single-pass
        #           was iteration-order-priority: a marginally-promising prompt
        #           early in the batch consumed the rescue slot, blocking a
        #           strictly-better prompt later in the batch from being tried.
        bundles: list[StagedBundle] = []
        rescue_candidates: list[dict] = []
        for i, (prompt_idx, problem) in enumerate(picks):
            generations = all_rollouts[i] if i < len(all_rollouts) else []
            if len(generations) < M_ROLLOUTS:
                logger.info(
                    "batched prep miss prompt=%d reason=gen_short (%d/%d)",
                    prompt_idx, len(generations), M_ROLLOUTS,
                )
                partial_rewards = (
                    _rewards_from_generations(
                        self.env, problem, generations, self.tokenizer,
                    )
                    if generations
                    else []
                )
                partial_sigma = (
                    rewards_std(partial_rewards)
                    if len(partial_rewards) >= 2
                    else 0.0
                )
                self._log_prep_rollouts(
                    prompt_idx=prompt_idx,
                    problem=problem,
                    generations=generations,
                    rewards=partial_rewards,
                    sigma=partial_sigma,
                    source="batched_prep",
                    outcome=f"miss_gen_short_{len(generations)}/{M_ROLLOUTS}",
                    canonical_tokens=canonical_tokens_list[i],
                    augmented_length=augmented_lengths[i],
                )
                continue

            rewards = _rewards_from_generations(
                self.env, problem, generations, self.tokenizer,
            )
            sigma = rewards_std(rewards)
            n_trunc = _count_truncated(generations, eos_set)
            n_init_term = M_ROLLOUTS - n_trunc
            term_ok = n_trunc <= _LOCAL_MAX_TRUNCATED

            if term_ok:
                if not _passes_zone_filter(rewards, bootstrap=bootstrap):
                    logger.info(
                        "batched prep miss prompt=%d σ=%.3f reason=out_of_zone "
                        "rewards=%s",
                        prompt_idx, sigma,
                        [int(r) for r in rewards],
                    )
                    self._log_prep_rollouts(
                        prompt_idx=prompt_idx,
                        problem=problem,
                        generations=generations,
                        rewards=rewards,
                        sigma=sigma,
                        source="batched_prep",
                        outcome="miss_out_of_zone",
                        canonical_tokens=canonical_tokens_list[i],
                        augmented_length=augmented_lengths[i],
                    )
                    self._maybe_update_predictor(
                        problem, in_zone=False, terminated_well=True,
                    )
                    continue
                # Last gate before queueing: any reward<0.5 rollout with a
                # malformed final \boxed{} would trigger MALFORMED_FINAL_ANSWER
                # on the validator side. Cap-truncated rollouts that ran out
                # of tokens mid-box are the common case. Check now and skip
                # rather than waste the GRAIL forward + HTTP roundtrip.
                init_malformed = _compute_malformed_flags(
                    generations, rewards, self.tokenizer,
                )
                n_malformed = sum(init_malformed)
                if n_malformed > 0:
                    logger.info(
                        "batched prep miss prompt=%d σ=%.3f reason=malformed_final_answer "
                        "(%d/%d rollouts have malformed final \\boxed{} — would "
                        "trigger MALFORMED_FINAL_ANSWER reject)",
                        prompt_idx, sigma, n_malformed, M_ROLLOUTS,
                    )
                    self._log_prep_rollouts(
                        prompt_idx=prompt_idx,
                        problem=problem,
                        generations=generations,
                        rewards=rewards,
                        sigma=sigma,
                        source="batched_prep",
                        outcome="miss_malformed_final_answer",
                        canonical_tokens=canonical_tokens_list[i],
                        augmented_length=augmented_lengths[i],
                    )
                    self._maybe_update_predictor(
                        problem, in_zone=False, terminated_well=True,
                    )
                    continue
                logger.info(
                    "batched prep hit prompt=%d σ=%.3f truncated=%d/%d "
                    "rewards=%s → queued",
                    prompt_idx, sigma, n_trunc, M_ROLLOUTS,
                    [int(r) for r in rewards],
                )
                self._log_prep_rollouts(
                    prompt_idx=prompt_idx,
                    problem=problem,
                    generations=generations,
                    rewards=rewards,
                    sigma=sigma,
                    source="batched_prep",
                    outcome="hit_queued",
                    canonical_tokens=canonical_tokens_list[i],
                    augmented_length=augmented_lengths[i],
                )
                rebound_gens = rebind_rollouts_to_canonical_prompt(
                    generations,
                    canonical_tokens_list[i],
                    augmented_lengths[i],
                )
                self._maybe_update_predictor(
                    problem, in_zone=True, terminated_well=True,
                )
                bundles.append(StagedBundle(
                    prompt_idx=prompt_idx,
                    problem=problem,
                    generations=rebound_gens,
                    rewards=rewards,
                    sigma=sigma,
                    checkpoint_n=local_n,
                    checkpoint_hash=local_hash,
                ))
                continue

            # Truncation failed. Two-stage gate before becoming a rescue
            # candidate:
            #   (a) Cheap: ≥ _RESCUE_MIN_INITIAL_TERMINATED EOS-last rollouts.
            #   (b) Strict: of those EOS-last rollouts, ≥
            #       _RESCUE_MIN_INITIAL_VALID_TERMINATED have p_stop ≥
            #       MIN_EOS_PROBABILITY (matches validator's verify_termination).
            # Without gate (b), "5 EOS-last" can include sampling-flukes that
            # vanish during rescue. Computing p_stops on the initial 8 is
            # ~1-2s, much cheaper than wasting a ~90-100s rescue on a doomed
            # bucket. The computed p_stops also flow into rescue so its first
            # iteration doesn't recompute them.
            if n_init_term >= _RESCUE_MIN_INITIAL_TERMINATED:
                init_p_stops = self._compute_p_stops(generations, eos_set)
                n_init_valid = sum(
                    1 for p in init_p_stops if p >= _LOCAL_MIN_EOS_PROB
                )
                if n_init_valid >= _RESCUE_MIN_INITIAL_VALID_TERMINATED:
                    rescue_candidates.append({
                        "idx": i,
                        "prompt_idx": prompt_idx,
                        "problem": problem,
                        "generations": generations,
                        "rewards": rewards,
                        "sigma": sigma,
                        "n_trunc": n_trunc,
                        "n_init_term": n_init_term,
                        "n_init_valid": n_init_valid,
                        "init_p_stops": init_p_stops,
                    })
                    self._log_prep_rollouts(
                        prompt_idx=prompt_idx,
                        problem=problem,
                        generations=generations,
                        rewards=rewards,
                        sigma=sigma,
                        source="batched_prep",
                        outcome="miss_too_truncated_rescue_eligible",
                        canonical_tokens=canonical_tokens_list[i],
                        augmented_length=augmented_lengths[i],
                    )
                    continue
                # EOS-last looked promising but valid-term is too low — the
                # initial 5 were mostly sampling-flukes; rescue would burn
                # ~90s and fail. Skip with a clear diagnostic.
                logger.info(
                    "batched prep miss prompt=%d σ=%.3f reason=too_truncated "
                    "(%d/%d EOS-last but only %d/%d valid p_stop≥%.2f — "
                    "rescue gate %d not met, likely fluke EOS)",
                    prompt_idx, sigma, n_trunc, M_ROLLOUTS,
                    n_init_valid, n_init_term, _LOCAL_MIN_EOS_PROB,
                    _RESCUE_MIN_INITIAL_VALID_TERMINATED,
                )
                self._log_prep_rollouts(
                    prompt_idx=prompt_idx,
                    problem=problem,
                    generations=generations,
                    rewards=rewards,
                    sigma=sigma,
                    source="batched_prep",
                    outcome="miss_too_truncated_rescue_gate",
                    canonical_tokens=canonical_tokens_list[i],
                    augmented_length=augmented_lengths[i],
                )
                self._maybe_update_predictor(
                    problem, in_zone=False, terminated_well=False,
                )
                continue

            # Clear miss: not enough initial terminators to bother rescuing.
            logger.info(
                "batched prep miss prompt=%d σ=%.3f reason=too_truncated "
                "(%d/%d, local cap=%d)",
                prompt_idx, sigma, n_trunc, M_ROLLOUTS,
                _LOCAL_MAX_TRUNCATED,
            )
            self._log_prep_rollouts(
                prompt_idx=prompt_idx,
                problem=problem,
                generations=generations,
                rewards=rewards,
                sigma=sigma,
                source="batched_prep",
                outcome="miss_too_truncated",
                canonical_tokens=canonical_tokens_list[i],
                augmented_length=augmented_lengths[i],
            )
            self._maybe_update_predictor(
                problem, in_zone=False, terminated_well=False,
            )

        # Pass B: rescue top-N candidates by (n_init_term, σ) descending.
        # More terminators = fewer additional rollouts needed; higher σ =
        # more balanced reward distribution → easier to find a σ-passing
        # subset. Both matter; lexicographic prioritisation favours the
        # quantity that most-directly drives rescue success (terminators).
        # Priority key uses n_init_valid (validator-realistic terminator count)
        # FIRST — it's a much stronger predictor of rescue success than the
        # raw EOS-last count. σ is the tiebreaker.
        rescue_candidates.sort(
            key=lambda c: (c["n_init_valid"], c["sigma"]),
            reverse=True,
        )
        for rank, cand in enumerate(rescue_candidates):
            prompt_idx = cand["prompt_idx"]
            problem = cand["problem"]
            sigma = cand["sigma"]
            n_trunc = cand["n_trunc"]
            n_init_term = cand["n_init_term"]
            n_init_valid = cand["n_init_valid"]
            generations = cand["generations"]
            rewards = cand["rewards"]
            init_p_stops = cand["init_p_stops"]
            i = cand["idx"]

            if rank >= _RESCUE_MAX_PER_CYCLE:
                # Out of rescue budget for this cycle.
                logger.info(
                    "batched prep miss prompt=%d σ=%.3f reason=too_truncated "
                    "(%d/%d; rescue-eligible valid_term=%d but cycle budget "
                    "%d/%d spent on higher-ranked candidates)",
                    prompt_idx, sigma, n_trunc, M_ROLLOUTS,
                    n_init_valid,
                    _RESCUE_MAX_PER_CYCLE, _RESCUE_MAX_PER_CYCLE,
                )
                self._maybe_update_predictor(
                    problem, in_zone=False, terminated_well=False,
                )
                continue

            logger.info(
                "batched prep promising prompt=%d σ=%.3f trunc=%d/%d "
                "(init EOS-last=%d valid_term=%d ≥ %d) → rescue attempt %d/%d "
                "[%d candidates ranked]",
                prompt_idx, sigma, n_trunc, M_ROLLOUTS,
                n_init_term, n_init_valid,
                _RESCUE_MIN_INITIAL_VALID_TERMINATED,
                rank + 1, _RESCUE_MAX_PER_CYCLE,
                len(rescue_candidates),
            )
            rescue = self._rescue_prompt(
                prompt_idx, problem, prompts_tokens[i],
                max_new_per_prompt[i],
                eos_set, generations, rewards, bootstrap,
                initial_p_stops=init_p_stops,
            )
            if rescue is not None:
                r_gens, r_rewards, r_sigma, r_trunc = rescue
                r_gens = rebind_rollouts_to_canonical_prompt(
                    r_gens,
                    canonical_tokens_list[i],
                    augmented_lengths[i],
                )
                logger.info(
                    "batched prep hit prompt=%d σ=%.3f truncated=%d/%d "
                    "rewards=%s → queued (via rescue)",
                    prompt_idx, r_sigma, r_trunc, M_ROLLOUTS,
                    [int(r) for r in r_rewards],
                )
                self._log_prep_rollouts(
                    prompt_idx=prompt_idx,
                    problem=problem,
                    generations=r_gens,
                    rewards=r_rewards,
                    sigma=r_sigma,
                    source="batched_prep_rescue",
                    outcome="hit_via_rescue",
                )
                self._maybe_update_predictor(
                    problem, in_zone=True, terminated_well=True,
                )
                bundles.append(StagedBundle(
                    prompt_idx=prompt_idx,
                    problem=problem,
                    generations=r_gens,
                    rewards=r_rewards,
                    sigma=r_sigma,
                    checkpoint_n=local_n,
                    checkpoint_hash=local_hash,
                ))
                continue

            # Rescue failed (initial rollouts already logged as rescue_eligible).
            logger.info(
                "batched prep miss prompt=%d σ=%.3f reason=too_truncated "
                "(%d/%d, after failed rescue)",
                prompt_idx, sigma, n_trunc, M_ROLLOUTS,
            )
            self._maybe_update_predictor(
                problem, in_zone=False, terminated_well=False,
            )

        logger.info(
            "batched prep result: %d/%d bundles in-zone (gen=%.1fs)",
            len(bundles), len(picks), gen_s,
        )
        return bundles

    async def _submit_bundle(
        self,
        bundle: StagedBundle,
        state,
        randomness: str,
        local_hash: str,
        url: str,
        client,
        submit_batch_v2,
        *,
        source: str,
    ):
        """GRAIL + sign + POST. Caller must run bundle_discard_reason first.

        Returns ``None`` when the window is no longer worth submitting to
        (not OPEN or batch nearly full) so the caller can re-queue the bundle.
        """
        from reliquary.miner.submitter import get_window_state_v2
        from reliquary.protocol.submission import WindowState

        try:
            fresh = await get_window_state_v2(url, client=client)
        except Exception as exc:
            logger.warning(
                "pre-GRAIL state refresh failed (%s); using cached state",
                exc,
            )
            fresh = state

        if fresh.state != WindowState.OPEN:
            logger.info(
                "skip submit prompt=%d: window state=%s (was OPEN)",
                bundle.prompt_idx,
                fresh.state.value if hasattr(fresh.state, "value") else fresh.state,
            )
            return None

        if fresh.window_n != state.window_n:
            logger.info(
                "skip submit prompt=%d: window advanced %d→%d",
                bundle.prompt_idx, state.window_n, fresh.window_n,
            )
            return None

        if fresh.valid_submissions >= _BATCH_NEARLY_FULL:
            # Throttle to one log per 10s for the same (prompt, window).
            last = getattr(self, "_last_nearly_full_log", (None, None, 0.0))
            now = time.monotonic()
            if (last[0], last[1]) != (bundle.prompt_idx, fresh.window_n) or (now - last[2]) > 10.0:
                logger.info(
                    "skip submit prompt=%d: batch nearly full "
                    "valid_submissions=%d/%d — re-queue for next window",
                    bundle.prompt_idx,
                    fresh.valid_submissions,
                    _BATCH_NEARLY_FULL + 1,
                )
                self._last_nearly_full_log = (bundle.prompt_idx, fresh.window_n, now)
            return None

        randomness = fresh.randomness or randomness
        if not randomness:
            logger.info(
                "skip submit prompt=%d: no randomness on fresh state",
                bundle.prompt_idx,
            )
            return None

        # GRAIL on the proof GPU runs in a worker thread so the asyncio loop
        # can keep scheduling generation prep on the other GPU in parallel.
        self._grail_in_flight = True
        t_grail = time.monotonic()
        logger.info(
            "grail start prompt=%d σ=%.3f rollouts=%d source=%s",
            bundle.prompt_idx, bundle.sigma, len(bundle.generations), source,
        )
        try:
            rollout_submissions = await asyncio.to_thread(
                self._build_rollout_submissions_from_bundle,
                bundle,
                randomness,
            )
        finally:
            self._grail_in_flight = False

        if rollout_submissions is None:
            # Predicted validator bad_termination — skip the submit.
            logger.info(
                "skip submit prompt=%d σ=%.3f source=%s reason=predicted_bad_termination "
                "(validator would reject; not wasting HTTP roundtrip)",
                bundle.prompt_idx, bundle.sigma, source,
            )
            self._maybe_update_predictor(bundle.problem, in_zone=False)
            # Returning a synthetic "rejected" response keeps the caller's
            # accounting consistent without a real submission.
            from reliquary.protocol.submission import (
                BatchSubmissionResponse, RejectReason,
            )
            return BatchSubmissionResponse(
                accepted=False, reason=RejectReason.BAD_TERMINATION,
            )

        logger.info(
            "grail done prompt=%d rollouts=%d elapsed=%.1fs → RolloutSubmission×%d",
            bundle.prompt_idx,
            len(rollout_submissions),
            time.monotonic() - t_grail,
            len(rollout_submissions),
        )
        merkle_root = _compute_merkle_root(rollout_submissions)
        current_round = _current_drand_round_at_send()

        import os as _os
        _nonce = _os.urandom(16).hex()
        _envelope_sig = sign_envelope(
            wallet=self.wallet,
            miner_hotkey=self.wallet.hotkey.ss58_address,
            window_start=fresh.window_n,
            prompt_idx=bundle.prompt_idx,
            merkle_root=merkle_root,
            checkpoint_hash=local_hash,
            drand_round=current_round,
            randomness=randomness,
            nonce=_nonce,
        ).hex()
        request = BatchSubmissionRequest(
            miner_hotkey=self.wallet.hotkey.ss58_address,
            prompt_idx=bundle.prompt_idx,
            window_start=fresh.window_n,
            merkle_root=merkle_root,
            rollouts=rollout_submissions,
            checkpoint_hash=local_hash,
            drand_round=current_round,
            nonce=_nonce,
            envelope_signature=_envelope_sig,
        )
        grail_mode = "proof-gpu"
        resp = await submit_batch_v2(url, request, client=client)
        # Track running accept/reject rate so we can see if changes help.
        if not hasattr(self, "_submit_accepted"):
            self._submit_accepted = 0
            self._submit_rejected = 0
        if resp.accepted:
            self._submit_accepted += 1
        else:
            self._submit_rejected += 1
        total = self._submit_accepted + self._submit_rejected
        logger.info(
            "submitted window=%d prompt=%d σ=%.3f source=%s "
            "valid_submissions=%d grail=%s accepted=%s reason=%s "
            "[lifetime: %d/%d = %.0f%% accept]",
            fresh.window_n,
            bundle.prompt_idx,
            bundle.sigma,
            source,
            fresh.valid_submissions,
            grail_mode,
            resp.accepted,
            resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
            self._submit_accepted, total,
            100.0 * self._submit_accepted / max(total, 1),
        )
        return resp

    def _build_rollout_submissions_from_bundle(
        self,
        bundle: StagedBundle,
        randomness: str,
    ) -> list[RolloutSubmission] | None:
        """Build eight rollout submissions with GRAIL commits on the proof GPU.

        Returns ``None`` when the predicted validator verdict is
        ``bad_termination`` — caller should skip the submit.
        """
        with self._proof_lock:
            return self._build_rollout_submissions_on_model(
                bundle.generations,
                bundle.rewards,
                randomness,
                model=self.hf_model,
                gpu=self.proof_gpu,
            )

    def _build_rollout_submissions_on_model(
        self,
        generations: list[dict],
        rewards: list[float],
        randomness: str,
        *,
        model,
        gpu: int,
    ) -> list[RolloutSubmission] | None:
        """Build GRAIL commits + predict validator's bad_termination verdict.

        Returns the submissions, OR ``None`` if the predicted bad-termination
        count would exceed the validator's budget (avoids wasting the HTTP
        round-trip on a submission we know will be rejected).
        """
        import math
        from reliquary.constants import MIN_EOS_PROBABILITY

        eos_set = eos_set_from_model(self.hf_model, self.tokenizer)
        # Use the local-stricter floor (deployed validator appears to use a
        # higher threshold than public 0.01; see _LOCAL_MIN_EOS_PROB docstring).
        _local_min_eos_prob = _LOCAL_MIN_EOS_PROB
        logger.info(
            "termination_check: eos_set=%s local_min_eos_prob=%.4f "
            "(public validator=%.4f) cap_tokens=%d",
            sorted(eos_set), _local_min_eos_prob, MIN_EOS_PROBABILITY,
            MAX_NEW_TOKENS_PROTOCOL_CAP,
        )

        # Early reject if ANY generation has EOS padding — validator's
        # has_eos_padding triggers immediate bad_termination (no budget).
        n_padding = _count_eos_padding(generations, eos_set)
        if n_padding > 0:
            # Log the offending rollouts so we can see which one(s) had padding
            for i, gen in enumerate(generations):
                tokens = gen.get("tokens", [])
                prompt_length = gen.get("prompt_length", 0)
                completion = tokens[prompt_length:]
                eos_positions = [
                    j for j, t in enumerate(completion) if int(t) in eos_set
                ]
                if eos_positions and (
                    len(eos_positions) > 1
                    or eos_positions[0] != len(completion) - 1
                ):
                    logger.info(
                        "  padded rollout %d: completion_len=%d "
                        "eos_positions=%s last_token=%d",
                        i, len(completion), eos_positions,
                        int(completion[-1]) if completion else -1,
                    )
            logger.info(
                "abort submission build: %d/%d rollouts have EOS padding "
                "(validator would reject immediately on has_eos_padding)",
                n_padding, len(generations),
            )
            return None

        min_eos_logp = math.log(max(_local_min_eos_prob, 1e-12))

        submissions: list[RolloutSubmission] = []
        predicted_bad = 0
        per_rollout_diag: list[str] = []
        for gen_idx, (gen, reward) in enumerate(zip(generations, rewards)):
            commit = self._build_grail_commit(
                gen, randomness, model=model, gpu=gpu,
            )

            # Diagnostic per-rollout — show what our predictor sees.
            tokens = commit.get("tokens") or []
            rollout_meta = commit.get("rollout", {}) or {}
            token_logprobs = rollout_meta.get("token_logprobs") or []
            last_tok = int(tokens[-1]) if tokens else -1
            last_logp = float(token_logprobs[-1]) if token_logprobs else float("nan")
            last_in_eos = last_tok in eos_set
            completion_length = int(rollout_meta.get("completion_length", 0))
            prompt_length = int(rollout_meta.get("prompt_length", 0))
            total_len = prompt_length + completion_length
            is_bad = _predict_validator_bad_termination(
                commit, eos_set, min_eos_logp, MAX_NEW_TOKENS_PROTOCOL_CAP,
            )
            if is_bad:
                predicted_bad += 1

            # Check explicit eos_padding for this rollout — same code path the
            # validator uses, so disagreement here would be a real bug.
            completion = tokens[prompt_length:prompt_length + completion_length]
            eos_positions = [
                j for j, t in enumerate(completion) if int(t) in eos_set
            ]
            has_padding = (
                bool(eos_positions)
                and (
                    len(eos_positions) > 1
                    or eos_positions[0] != len(completion) - 1
                )
            )

            # Decode last 5 tokens for visual sanity-check.
            tail_tokens = (
                tokens[-5:] if len(tokens) >= 5 else tokens
            )
            try:
                tail_text = self.tokenizer.decode(
                    tail_tokens, skip_special_tokens=False,
                )[-60:]
                tail_text = tail_text.replace("\n", "\\n")
            except Exception:
                tail_text = "?"
            per_rollout_diag.append(
                f"#{gen_idx}: last_tok={last_tok} in_eos={last_in_eos} "
                f"last_logp={last_logp:.3f} (p={math.exp(min(0, last_logp)):.5f}) "
                f"total_len={total_len} eos_pos={eos_positions[:3]}"
                f"{'...' if len(eos_positions) > 3 else ''} "
                f"has_padding={has_padding} predicted_bad={is_bad} "
                f"tail_text={tail_text!r}"
            )

            # If has_padding fires here, the bundle is doomed — abort now and
            # save the remaining GRAIL forward passes.
            if has_padding:
                for line in per_rollout_diag:
                    logger.info("  %s", line)
                logger.info(
                    "abort submission build: rollout #%d has eos_padding "
                    "(positions=%s, last_pos=%d) — validator would immediately "
                    "reject with bad_termination",
                    gen_idx, eos_positions, len(completion) - 1,
                )
                return None

            submissions.append(
                RolloutSubmission(
                    tokens=gen["tokens"],
                    reward=reward,
                    commit=commit,
                )
            )

            # Early abort: stop building when predicted_bad exceeds the
            # VALIDATOR's exact cap. With the eos_set fix our prediction now
            # aligns with the validator's count to within numerical-precision
            # noise (0-1 rollout), so the previous heavy safety margins are
            # no longer needed.
            if predicted_bad > MAX_TRUNCATED_PER_SUBMISSION:
                for line in per_rollout_diag:
                    logger.info("  %s", line)
                logger.info(
                    "abort submission build: predicted_bad_termination=%d > "
                    "validator cap=%d (saved remaining %d GRAIL commits)",
                    predicted_bad, MAX_TRUNCATED_PER_SUBMISSION,
                    len(generations) - len(submissions),
                )
                return None

        # Show all per-rollout diagnostics whenever we submit, so we can
        # cross-reference against any validator rejection.
        for line in per_rollout_diag:
            logger.info("  %s", line)
        logger.info(
            "termination_check result: predicted_bad=%d/%d "
            "(validator cap=%d) → %s",
            predicted_bad, len(submissions),
            MAX_TRUNCATED_PER_SUBMISSION,
            "submit" if predicted_bad <= MAX_TRUNCATED_PER_SUBMISSION else "abort",
        )
        if predicted_bad > MAX_TRUNCATED_PER_SUBMISSION:
            return None

        return submissions

    def _load_checkpoint(self, local_path: str):
        """Reload both hf_model and vllm_model from *local_path*.

        Both attributes are ``AutoModelForCausalLM`` instances despite the
        historical ``vllm_model`` naming — vllm_model is the fast-generation
        copy on ``self.vllm_gpu``, hf_model is the GRAIL-proof copy on
        ``self.proof_gpu``.
        """
        import torch
        from transformers import AutoModelForCausalLM

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        try:
            new_hf = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=self._attn_implementation,
            ).to(f"cuda:{self.proof_gpu}").eval()
        except Exception:
            logger.exception(
                "Failed to reload hf_model from %s; keeping old model",
                local_path,
            )
            return self.hf_model

        old_hf = self.hf_model
        self.hf_model = new_hf
        del old_hf
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # When vLLM owns cuda:0, the HF generation copy was never loaded; skip
        # its reload and rebuild the vLLM engine instead. vLLM can't hot-swap
        # weights, so we tear down and rebuild — slow (~40-60s) but only runs
        # every CHECKPOINT_PUBLISH_INTERVAL_WINDOWS windows.
        if getattr(self, "gen_backend", None) is not None and getattr(
            self.gen_backend, "is_vllm", False,
        ):
            import time as _time
            from reliquary.miner.generation import (
                build_vllm_generator,
                shutdown_vllm_generator,
            )
            logger.info(
                "vLLM rebuild: shutting down old engine on cuda:%d",
                self.vllm_gpu,
            )
            shutdown_t0 = _time.monotonic()
            old_backend = self.gen_backend
            self.gen_backend = None
            shutdown_vllm_generator(old_backend, gpu=self.vllm_gpu)
            del old_backend
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info(
                "vLLM rebuild: shutdown took %.1fs",
                _time.monotonic() - shutdown_t0,
            )

            try:
                logger.info(
                    "vLLM rebuild: building new engine for %s on cuda:%d",
                    local_path, self.vllm_gpu,
                )
                build_t0 = _time.monotonic()
                self.gen_backend = build_vllm_generator(
                    local_path, gpu=self.vllm_gpu,
                )
                self._loaded_checkpoint_path = local_path
                logger.info(
                    "vLLM rebuild: complete in %.1fs; checkpoint=%s",
                    _time.monotonic() - build_t0, local_path,
                )
                return self.hf_model
            except Exception:
                logger.exception(
                    "vLLM rebuild from %s FAILED — falling back to HF generation on cuda:%d. "
                    "Throughput will drop ~5-10× until next checkpoint pull recovers vLLM.",
                    local_path, self.vllm_gpu,
                )
                # Fallback path: load HF model on cuda:0 for generation. Sets
                # ``_using_vllm`` False so the rest of the engine routes through
                # the HF path. Future checkpoint pulls will retry the vLLM build.
                try:
                    new_gen = AutoModelForCausalLM.from_pretrained(
                        local_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=ATTN_IMPLEMENTATION,
                    ).to(f"cuda:{self.vllm_gpu}").eval()
                    self.vllm_model = new_gen
                    self._using_vllm = False
                    self._loaded_checkpoint_path = local_path
                    logger.info(
                        "HF fallback generation model loaded on cuda:%d",
                        self.vllm_gpu,
                    )
                    return self.hf_model
                except Exception:
                    logger.exception(
                        "HF fallback also failed; generation is BROKEN until next pull.",
                    )
                    self._loaded_checkpoint_path = None
                    return self.hf_model

        try:
            new_gen = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=self._attn_implementation,
            ).to(f"cuda:{self.vllm_gpu}").eval()
        except Exception:
            logger.exception(
                "Failed to reload vllm_model from %s; miner generation is "
                "BROKEN until the next successful pull. hf_model was swapped "
                "so GRAIL proofs will be inconsistent.",
                local_path,
            )
            self.vllm_model = None
            self._loaded_checkpoint_path = None
            return self.hf_model

        old_gen = self.vllm_model
        self.vllm_model = new_gen
        del old_gen
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self._loaded_checkpoint_path = local_path
        logger.info("Checkpoint %s loaded into both models", local_path)
        return self.hf_model

    def _generate_m_rollouts(
        self,
        problem,
        *,
        gen_model=None,
        gen_lock: threading.Lock | None = None,
        bootstrap: bool = False,
    ) -> list[dict]:
        """Generate M_ROLLOUTS completions at T_PROTO (with optional 2+1 zone screen)."""
        model = gen_model if gen_model is not None else self.vllm_model
        lock = gen_lock if gen_lock is not None else self._gen_lock

        if bootstrap or M_ROLLOUTS <= _ZONE_SCREEN_ROLLOUTS:
            return self._generate_rollouts_on_model(problem, model, lock, M_ROLLOUTS)

        head = self._generate_rollouts_on_model(
            problem, model, lock, _ZONE_SCREEN_ROLLOUTS,
        )
        if len(head) < _ZONE_SCREEN_ROLLOUTS:
            return head
        head_rewards = _rewards_from_generations(
            self.env, problem, head, self.tokenizer,
        )

        if not _zone_screen_passes(head_rewards, bootstrap=bootstrap):
            if _ZONE_SCREEN_TIEBREAKER_ROLLOUTS <= 0:
                return []
            tie = self._generate_rollouts_on_model(
                problem, model, lock, _ZONE_SCREEN_TIEBREAKER_ROLLOUTS,
            )
            if len(tie) < _ZONE_SCREEN_TIEBREAKER_ROLLOUTS:
                return []
            tie_rewards = _rewards_from_generations(
                self.env, problem, tie, self.tokenizer,
            )
            if len(set(head_rewards + tie_rewards)) == 1:
                return []
            head = head + tie
            head_rewards = head_rewards + tie_rewards

        # Truncation pre-check (Option 3): if every head rollout already hit
        # max_tokens without EOS, the prompt is overwhelmingly likely to
        # produce a too-truncated tail that the validator will reject as
        # ``bad_termination``. Skip the expensive tail generation. Saves ~T
        # per prompt where the model is non-terminating. False positives are
        # rare: it would require head 100% truncated AND tail mostly
        # terminating, which contradicts the model's observed behavior.
        eos_set = eos_set_from_model(self.hf_model, self.tokenizer)
        n_head_trunc = _count_truncated(head, eos_set)
        if n_head_trunc == len(head) and eos_set:
            logger.debug(
                "head trunc-screen abort: %d/%d head rollouts hit max_tokens; "
                "skipping %d-rollout tail",
                n_head_trunc, len(head), M_ROLLOUTS - len(head),
            )
            return []

        remaining = M_ROLLOUTS - len(head)
        if remaining <= 0:
            return head[:M_ROLLOUTS]
        tail = self._generate_rollouts_on_model(problem, model, lock, remaining)
        if len(tail) < remaining:
            return []
        return head + tail

    def _generate_rollouts_on_model(
        self,
        problem,
        model,
        lock: threading.Lock,
        n_rollouts: int,
    ) -> list[dict]:
        """Run *n_rollouts* sampled completions on *model* (validator-aligned EOS).

        When ``self.gen_backend`` is set (vLLM), route through it ONLY when
        ``model`` is None — explicit ``model=<hf>`` forces the HF path so the
        parallel prep slot on cuda:1 (the proof model) can also generate
        alongside vLLM on cuda:0. GRAIL commits are always computed against
        ``hf_model``, keeping bit-identicality with the validator.
        """
        gen_backend = getattr(self, "gen_backend", None)
        use_vllm_path = gen_backend is not None and model is None

        eos_set = eos_set_from_model(
            self.hf_model if use_vllm_path else (model or self.hf_model),
            self.tokenizer,
        )

        prompt_tokens = generation_prompt_tokens(
            problem["prompt"], self.tokenizer,
        )
        augmented_prompt_length = len(prompt_tokens)
        canonical_tokens = canonical_prompt_tokens(
            problem["prompt"], self.tokenizer,
        )
        max_new = min(
            self.max_new_tokens,
            max_new_tokens_for_generation(
                problem["prompt"], self.tokenizer,
            ),
        )

        if use_vllm_path:
            rollouts = gen_backend.generate(
                prompt_tokens, n_rollouts,
                max_new_tokens=max_new,
                eos_set=eos_set,
                tokenizer=self.tokenizer,
            )
            return rebind_rollouts_to_canonical_prompt(
                rollouts, canonical_tokens, augmented_prompt_length,
            )

        import torch
        eos_for_generate = (
            sorted(eos_set) if len(eos_set) > 1 else (next(iter(eos_set)) if eos_set else None)
        )

        with lock:
            gen_kwargs: dict = {
                "max_new_tokens": max_new,
                "do_sample": True,
                "temperature": T_PROTO,
                "top_p": TOP_P_PROTO,
                "top_k": TOP_K_PROTO,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if eos_for_generate is not None:
                gen_kwargs["eos_token_id"] = eos_for_generate

            with torch.no_grad():
                input_tensor = torch.tensor(
                    [prompt_tokens] * n_rollouts,
                    device=getattr(model, "device", "cpu"),
                )
                outputs = model.generate(input_tensor, **gen_kwargs)

            rollouts = []
            for i in range(n_rollouts):
                seq = outputs[i].tolist()
                gen = truncate_completion_at_eos(
                    seq[augmented_prompt_length:], eos_set,
                )
                rollouts.append({
                    "tokens": prompt_tokens + gen,
                    "prompt_length": augmented_prompt_length,
                })
            return rebind_rollouts_to_canonical_prompt(
                rollouts, canonical_tokens, augmented_prompt_length,
            )

    def _build_rollout_submission(self, generation, problem, randomness):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit."""
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        reward = self.env.compute_reward(problem, completion_text)

        commit = self._build_grail_commit(
            generation, randomness, model=self.hf_model, gpu=self.proof_gpu,
        )
        return RolloutSubmission(
            tokens=all_tokens,
            reward=reward,
            commit=commit,
        )

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool
    ) -> str:
        """Derive window randomness from the drand beacon (v2.3+: drand-only)."""
        if use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                None, beacon["randomness"], drand_round=beacon["round"]
            )
        block_hash = await chain.get_block_hash(subtensor, window_start)
        return chain.compute_window_randomness(block_hash)

    def _build_grail_commit(
        self,
        generation: dict,
        randomness: str,
        *,
        model=None,
        gpu: int | None = None,
    ) -> dict:
        """Construct a GRAIL proof commit dict from a generation dict."""
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        model = model if model is not None else self.hf_model
        gpu = self.proof_gpu if gpu is None else gpu

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]

        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs: list[float] = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        model_name: str = getattr(model, "name_or_path", "unknown")
        signature = sign_commit_binding(
            all_tokens, randomness, model_name, LAYER_INDEX,
            commitments, self.wallet,
        )

        return {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": GRAIL_PROOF_VERSION,
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": {
                "prompt_length": prompt_length,
                "completion_length": len(all_tokens) - prompt_length,
                "success": True,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": token_logprobs,
            },
        }
