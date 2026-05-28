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
    M_ROLLOUTS,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)
from reliquary.infrastructure import chain
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
_PREP_PROMPT_ATTEMPTS = 5
# Poll interval when staging can immediately start another prep cycle.
_PREP_POLL_SECONDS = 0.05


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


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    excluded_prompts: set[int] | None = None,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
) -> int:
    """Pick a random prompt index that isn't in cooldown or *excluded_prompts*.

    The reference miner uses uniform-random selection with rejection
    sampling against the blocked set. More sophisticated strategies
    (pre-screening zone probability, etc.) are left to miner operators.

    Raises ``RuntimeError`` if no eligible prompt can be found — typically
    because the env is fully in cooldown.
    """
    rng = rng or _random
    blocked = cooldown_prompts | (excluded_prompts or set())
    n = len(env)
    if len(blocked) < n / 2:
        for _ in range(max_attempts):
            idx = rng.randrange(n)
            if idx not in blocked:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(n) if i not in blocked]
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

    When two GPUs are present, generation and proof work can overlap:
    prep/staging uses the generation GPU while submit/GRAIL uses the proof
    GPU, keeping both devices busy instead of serialising all work on one card.
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
        # separate devices (e.g. H200×2) but never two jobs on the same GPU.
        self._gen_lock = threading.Lock()
        self._proof_lock = threading.Lock()

        # Lazy imports for heavy deps — keep module import cheap.
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)
        self._rng = _random.Random()

        if self._dual_gpu:
            logger.info(
                "Dual-GPU pipeline: generation=cuda:%d, GRAIL proofs=cuda:%d",
                vllm_gpu, proof_gpu,
            )
        else:
            logger.info(
                "Single-GPU mode: generation and GRAIL share cuda:%d",
                vllm_gpu,
            )

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
        gen_prep_in_flight = False

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
                if (
                    len(staged) < staged.max_size
                    and not gen_prep_in_flight
                    and self.vllm_model is not None
                    and not nearly_full
                ):
                    gen_prep_in_flight = True
                    prep_cooldown = set(cooldown_set)
                    prep_excluded = staged.staged_prompt_indices()
                    prep_local_n = local_n
                    prep_local_hash = local_hash
                    prep_bootstrap = bootstrap
                    prep_window_n = state.window_n

                    async def _run_prep() -> None:
                        nonlocal gen_prep_in_flight
                        try:
                            bundle = await asyncio.to_thread(
                                self._prepare_bundle,
                                prep_cooldown,
                                prep_excluded,
                                prep_local_n,
                                prep_local_hash,
                                prep_bootstrap,
                            )
                            if bundle is not None:
                                if staged.push(bundle):
                                    logger.info(
                                        "staged prompt=%d σ=%.3f queue=%d/%d "
                                        "checkpoint=%d@%s gpu=gen:cuda:%d",
                                        bundle.prompt_idx,
                                        bundle.sigma,
                                        len(staged),
                                        staged.max_size,
                                        bundle.checkpoint_n,
                                        bundle.checkpoint_hash[:12],
                                        self.vllm_gpu,
                                    )
                                else:
                                    logger.debug(
                                        "staged queue full; dropped prompt=%d",
                                        bundle.prompt_idx,
                                    )
                        except RuntimeError:
                            logger.info(
                                "prep: no eligible prompt (window=%d queue=%d/%d)",
                                prep_window_n, len(staged), staged.max_size,
                            )
                        except Exception:
                            logger.exception("prep failed for window=%d", prep_window_n)
                        finally:
                            gen_prep_in_flight = False

                    asyncio.create_task(_run_prep())

                if state.state != WindowState.OPEN:
                    if (
                        len(staged) < staged.max_size
                        and not gen_prep_in_flight
                        and self.vllm_model is not None
                    ):
                        await asyncio.sleep(_PREP_POLL_SECONDS)
                    else:
                        await asyncio.sleep(1)
                    continue

                randomness = state.randomness
                if not randomness:
                    await asyncio.sleep(0.1)
                    continue

                if submissions_this_window >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
                    await asyncio.sleep(1)
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
                    continue

                if len(staged) > 0:
                    # More staged bundles remain; re-poll quickly.
                    continue

                if gen_prep_in_flight:
                    await asyncio.sleep(_PREP_POLL_SECONDS)
                    continue

                if nearly_full:
                    await asyncio.sleep(0.5)
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
                    await asyncio.sleep(_PREP_POLL_SECONDS)
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
                        continue
                    results.append(resp)
                    submissions_this_window += 1
                except SubmissionError as exc:
                    logger.error("submit failed (inline): %s", exc)

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
    ) -> StagedBundle | None:
        """Generate rollouts; retry prompts until one passes the σ zone filter."""
        tried: set[int] = set()
        blocked = set(excluded_prompts)

        for attempt in range(1, max_prompt_attempts + 1):
            try:
                prompt_idx = pick_prompt_idx(
                    self.env,
                    cooldown_set,
                    excluded_prompts=blocked | tried,
                    rng=self._rng,
                )
            except RuntimeError:
                if attempt == 1:
                    raise
                logger.info(
                    "prep: no eligible prompts after %d attempt(s)",
                    attempt - 1,
                )
                return None

            problem = self.env.get_problem(prompt_idx)
            generations = self._generate_m_rollouts(problem)
            if len(generations) < M_ROLLOUTS:
                logger.warning(
                    "prep attempt %d/%d prompt=%d generated %d/%d rollouts",
                    attempt, max_prompt_attempts, prompt_idx,
                    len(generations), M_ROLLOUTS,
                )
                tried.add(prompt_idx)
                continue

            rewards = _rewards_from_generations(
                self.env, problem, generations, self.tokenizer,
            )
            sigma = rewards_std(rewards)
            if _passes_zone_filter(rewards, bootstrap=bootstrap):
                logger.info(
                    "prep in-zone attempt %d/%d prompt=%d σ=%.3f",
                    attempt, max_prompt_attempts, prompt_idx, sigma,
                )
                return StagedBundle(
                    prompt_idx=prompt_idx,
                    problem=problem,
                    generations=generations,
                    rewards=rewards,
                    sigma=sigma,
                    checkpoint_n=local_n,
                    checkpoint_hash=local_hash,
                )

            logger.info(
                "prep out-of-zone attempt %d/%d prompt=%d σ=%.3f (bootstrap=%s)",
                attempt, max_prompt_attempts, prompt_idx, sigma, bootstrap,
            )
            tried.add(prompt_idx)

        logger.info(
            "prep exhausted %d attempts — no in-zone bundle",
            max_prompt_attempts,
        )
        return None

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
            logger.info(
                "skip submit prompt=%d: batch nearly full "
                "valid_submissions=%d/%d — re-queue for next window",
                bundle.prompt_idx,
                fresh.valid_submissions,
                _BATCH_NEARLY_FULL + 1,
            )
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
        rollout_submissions = await asyncio.to_thread(
            self._build_rollout_submissions_from_bundle,
            bundle,
            randomness,
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
        resp = await submit_batch_v2(url, request, client=client)
        logger.info(
            "submitted window=%d prompt=%d σ=%.3f source=%s "
            "valid_submissions=%d gpu=proof:cuda:%d accepted=%s reason=%s",
            fresh.window_n,
            bundle.prompt_idx,
            bundle.sigma,
            source,
            fresh.valid_submissions,
            self.proof_gpu,
            resp.accepted,
            resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
        )
        return resp

    def _build_rollout_submissions_from_bundle(
        self,
        bundle: StagedBundle,
        randomness: str,
    ) -> list[RolloutSubmission]:
        with self._proof_lock:
            submissions: list[RolloutSubmission] = []
            for gen, reward in zip(bundle.generations, bundle.rewards):
                commit = self._build_grail_commit(gen, randomness)
                submissions.append(
                    RolloutSubmission(
                        tokens=gen["tokens"],
                        reward=reward,
                        commit=commit,
                    )
                )
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

        from reliquary.constants import ATTN_IMPLEMENTATION

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        try:
            new_hf = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
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

        try:
            new_gen = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
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

    def _generate_m_rollouts(self, problem) -> list[dict]:
        """Generate M_ROLLOUTS completions at T_PROTO in one batched call."""
        import torch

        with self._gen_lock:
            prompt_tokens = self.tokenizer.encode(
                problem["prompt"], add_special_tokens=False
            )
            prompt_length = len(prompt_tokens)

            with torch.no_grad():
                input_tensor = torch.tensor(
                    [prompt_tokens] * M_ROLLOUTS,
                    device=getattr(self.vllm_model, "device", "cpu"),
                )
                outputs = self.vllm_model.generate(
                    input_tensor,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=T_PROTO,
                    top_p=TOP_P_PROTO,
                    top_k=TOP_K_PROTO,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            eos = self.tokenizer.eos_token_id
            rollouts = []
            for i in range(M_ROLLOUTS):
                seq = outputs[i].tolist()
                gen = seq[prompt_length:]
                try:
                    first_eos = gen.index(eos)
                    gen = gen[: first_eos + 1]
                except ValueError:
                    pass
                rollouts.append({
                    "tokens": prompt_tokens + gen,
                    "prompt_length": prompt_length,
                })
            return rollouts

    def _build_rollout_submission(self, generation, problem, randomness):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit."""
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        reward = self.env.compute_reward(problem, completion_text)

        commit = self._build_grail_commit(generation, randomness)
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

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Construct a GRAIL proof commit dict from a generation dict."""
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]

        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs: list[float] = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")
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
