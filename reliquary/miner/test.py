"""Debug a single prompt through the miner prep pipeline.

Runs the same generation + validation checks as production prep, with
step-by-step logs so you can verify prompt augmentation, rollouts, and gates.

Usage::

    python reliquary/miner/test.py --prompt-idx 433264 \\
        --checkpoint /path/to/model --use-vllm

    # Match your running miner (validator checkpoint + vLLM):
    python reliquary/miner/test.py --prompt-idx 433264 \\
        --validator-url http://86.38.238.30:8080 --use-vllm
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reliquary.miner.engine import MiningEngine

log = logging.getLogger("reliquary.miner.test")

_LOCAL_MAX_TRUNCATED = 1  # mirrors engine._LOCAL_MAX_TRUNCATED
_DEFAULT_ENV = os.getenv("RELIQUARY_ENVIRONMENT_NAME", "openmathinstruct")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub", "vllm", "bittensor"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _out(msg: str = "") -> None:
    """Print to stdout — survives bittensor logging overrides."""
    print(msg, flush=True)


def _step(n: int, title: str) -> None:
    _out("")
    _out("=" * 72)
    _out(f"STEP {n} — {title}")
    _out("=" * 72)
    log.info("STEP %d — %s", n, title)


def _preview(text: str, limit: int = 400) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass
class RolloutReport:
    idx: int
    reward: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    last_token: int
    ends_with_eos: bool
    truncated: bool
    malformed: bool
    tail_text: str


def _analyze_rollouts(
    generations: list[dict],
    rewards: list[float],
    malformed_flags: list[bool],
    eos_set: set[int],
    tokenizer: Any,
) -> list[RolloutReport]:
    reports: list[RolloutReport] = []
    for i, gen in enumerate(generations):
        tokens = gen.get("tokens") or []
        plen = int(gen.get("prompt_length", 0))
        clen = max(0, len(tokens) - plen)
        last = int(tokens[-1]) if tokens else -1
        in_eos = last in eos_set if eos_set else False
        completion = tokens[plen:]
        try:
            tail = tokenizer.decode(
                completion[-20:] if len(completion) > 20 else completion,
                skip_special_tokens=False,
            )
            tail = _preview(tail, 120)
        except Exception:
            tail = "?"
        reports.append(
            RolloutReport(
                idx=i,
                reward=rewards[i] if i < len(rewards) else 0.0,
                total_tokens=len(tokens),
                prompt_tokens=plen,
                completion_tokens=clen,
                last_token=last,
                ends_with_eos=in_eos,
                truncated=not in_eos if eos_set else False,
                malformed=malformed_flags[i] if i < len(malformed_flags) else False,
                tail_text=tail,
            )
        )
    return reports


def _log_rollout_table(reports: list[RolloutReport]) -> None:
    header = (
        f"{'#':<4} {'reward':<6} {'total':<6} {'prompt':<6} {'compl':<6} "
        f"{'last_tok':<8} {'eos':<5} {'trunc':<5} {'malf':<5}  tail"
    )
    _out(header)
    _out("-" * 72)
    log.info(header)
    log.info("-" * 72)
    for r in reports:
        line = (
            f"{r.idx:<4} {r.reward:<6.1f} {r.total_tokens:<6} {r.prompt_tokens:<6} "
            f"{r.completion_tokens:<6} {r.last_token:<8} {str(r.ends_with_eos):<5} "
            f"{str(r.truncated):<5} {str(r.malformed):<5}  {r.tail_text!r}"
        )
        _out(line)
        log.info(line)


def _log_full_prompts(
    original: str,
    augmented_text: str,
    *,
    show_full: bool,
) -> None:
    limit = None if show_full else 2000
    _out("")
    _out("--- CANONICAL PROMPT (env, submitted to validator) ---")
    text = original if show_full or len(original) <= (limit or 0) else original[: limit - 3] + "..."
    _out(text)
    _out("")
    _out("--- AUGMENTED PROMPT (generation only = canonical + suffix) ---")
    text = augmented_text if show_full or len(augmented_text) <= (limit or 0) else augmented_text[: limit - 3] + "..."
    _out(text)
    _out("")
    _out(f"--- SUFFIX ONLY ({len(augmented_text) - len(original)} chars) ---")
    _out(augmented_text[len(original):])


def _log_completion_samples(
    rollouts: list[dict],
    rewards: list[float],
    tokenizer: Any,
    *,
    n_samples: int = 3,
    chars: int = 600,
) -> None:
    _out("")
    _out(f"--- completion text samples (first {n_samples} rollouts) ---")
    for i, gen in enumerate(rollouts[:n_samples]):
        plen = int(gen.get("prompt_length", 0))
        completion = gen.get("tokens", [])[plen:]
        try:
            text = tokenizer.decode(completion, skip_special_tokens=False)
        except Exception as exc:
            text = f"<decode error: {exc}>"
        if len(text) > chars:
            text = text[: chars - 3] + "..."
        reward = rewards[i] if i < len(rewards) else 0.0
        _out(f"[rollout {i}] reward={reward:.1f} completion_tokens={len(completion)}")
        _out(text)
        _out("")


def _diagnose_head_trunc_screen(
    raw_rollouts: list[dict],
    raw_rewards: list[float],
    eos_set: set[int],
    *,
    bootstrap: bool,
) -> None:
    from reliquary.miner.engine import _count_truncated, _zone_screen_passes

    _out("")
    _out("--- zone screen / head-trunc diagnostic (HF _generate_m_rollouts path) ---")
    _out("NOTE: vLLM batched prep in production SKIPS this screen; step 4 does not.")
    head_n = 2
    if len(raw_rollouts) < head_n:
        _out("  fewer than 2 rollouts — screen would abort")
        return
    head = raw_rollouts[:head_n]
    head_rewards = raw_rewards[:head_n]
    zone_ok = _zone_screen_passes(head_rewards, bootstrap=bootstrap)
    n_head_trunc = _count_truncated(head, eos_set)
    _out(f"  head rewards (first 2): {[int(r) for r in head_rewards]}")
    _out(f"  zone screen pass (rewards differ or bootstrap): {zone_ok}")
    _out(f"  head truncations: {n_head_trunc}/{head_n}")
    if n_head_trunc == head_n and eos_set:
        _out("  → head-trunc abort: ALL head rollouts hit max_tokens without EOS")
        _out("    (_try_prompt_bundle would return gen=0 / screen_or_head_trunc)")
    if not zone_ok:
        _out("  → zone screen abort: first 2 rewards identical (σ=0 at n=2)")


def _prompt_binding_ok(
    generations: list[dict],
    canonical: list[int],
) -> tuple[bool, str]:
    for i, gen in enumerate(generations):
        plen = int(gen.get("prompt_length", 0))
        prefix = list((gen.get("tokens") or [])[:plen])
        if prefix != list(canonical):
            return False, f"rollout {i}: prefix mismatch (len {len(prefix)} vs canonical {len(canonical)})"
    return True, "all rollouts have canonical prompt prefix"


def _generate_raw_augmented(
    engine: MiningEngine,
    problem: dict,
    *,
    n_rollouts: int,
) -> list[dict]:
    from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
    from reliquary.miner.engine import eos_set_from_model, truncate_completion_at_eos
    from reliquary.shared.prompt_augment import (
        generation_prompt_tokens,
        max_new_tokens_for_generation,
    )

    augmented = generation_prompt_tokens(problem["prompt"], engine.tokenizer)
    augmented_len = len(augmented)
    max_new = max_new_tokens_for_generation(problem["prompt"], engine.tokenizer)
    eos_set = eos_set_from_model(engine.hf_model, engine.tokenizer)

    if engine.gen_backend is not None:
        return engine.gen_backend.generate(
            augmented,
            n_rollouts,
            max_new_tokens=max_new,
            eos_set=eos_set,
            tokenizer=engine.tokenizer,
        )

    import torch

    model = engine.vllm_model
    eos_for_generate = (
        sorted(eos_set) if len(eos_set) > 1
        else (next(iter(eos_set)) if eos_set else None)
    )
    gen_kwargs: dict = {
        "max_new_tokens": max_new,
        "do_sample": True,
        "temperature": T_PROTO,
        "top_p": TOP_P_PROTO,
        "top_k": TOP_K_PROTO,
        "pad_token_id": engine.tokenizer.pad_token_id,
    }
    if eos_for_generate is not None:
        gen_kwargs["eos_token_id"] = eos_for_generate

    with torch.no_grad():
        input_tensor = torch.tensor(
            [augmented] * n_rollouts,
            device=getattr(model, "device", "cpu"),
        )
        outputs = model.generate(input_tensor, **gen_kwargs)

    rollouts = []
    for i in range(n_rollouts):
        seq = outputs[i].tolist()
        comp = truncate_completion_at_eos(seq[augmented_len:], eos_set)
        rollouts.append({
            "tokens": augmented + comp,
            "prompt_length": augmented_len,
        })
    return rollouts


def _resolve_checkpoint(checkpoint: str, validator_url: str | None) -> str:
    if not validator_url:
        return checkpoint

    import asyncio

    import httpx
    from huggingface_hub import snapshot_download

    from reliquary.miner.submitter import get_window_state_v2

    async def _fetch() -> str:
        async with httpx.AsyncClient(timeout=30) as client:
            state = await get_window_state_v2(validator_url, client=client)
        if state.checkpoint_repo_id and state.checkpoint_revision:
            log.info(
                "Validator checkpoint n=%d repo=%s rev=%s",
                state.checkpoint_n,
                state.checkpoint_repo_id,
                state.checkpoint_revision[:12],
            )
            return snapshot_download(
                repo_id=state.checkpoint_repo_id,
                revision=state.checkpoint_revision,
            )
        log.warning("Validator has no checkpoint; using --checkpoint")
        return checkpoint

    return asyncio.run(_fetch())


def _load_engine(checkpoint: str, environment: str, use_vllm: bool) -> MiningEngine:
    import bittensor as bt
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reliquary.environment import load_environment
    from reliquary.miner.engine import MiningEngine
    from reliquary.miner.generation import apply_transformers_compat_patches
    from reliquary.shared.hf_compat import resolve_attn_implementation

    apply_transformers_compat_patches()
    attn_impl = resolve_attn_implementation()

    log.info("Loading tokenizer + models from %s", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        gen_gpu, proof_gpu = 0, 1
        gen_device, proof_device = "cuda:0", "cuda:1"
    else:
        gen_gpu = proof_gpu = 0
        gen_device = proof_device = "cuda:0"
    log.info("GPUs visible=%d generation=%s GRAIL=%s", n_gpu, gen_device, proof_device)

    hf_model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(proof_device).eval()

    gen_backend = None
    vllm_model = None
    if use_vllm:
        from reliquary.miner.generation import build_vllm_generator

        gen_backend = build_vllm_generator(checkpoint, gpu=gen_gpu)
        log.info("vLLM backend ready on cuda:%d", gen_gpu)
    else:
        vllm_model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to(gen_device).eval()
        log.info("HF generation model on %s", gen_device)

    env = load_environment(environment)
    wallet = bt.Wallet(name="default", hotkey="default")

    engine = MiningEngine(
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env,
        vllm_gpu=gen_gpu,
        proof_gpu=proof_gpu,
        gen_backend=gen_backend,
        predictor=None,
        attn_implementation=attn_impl,
    )
    engine._loaded_checkpoint_path = checkpoint
    return engine


def run_prompt_test(
    prompt_idx: int,
    *,
    checkpoint: str,
    environment: str = _DEFAULT_ENV,
    use_vllm: bool = False,
    validator_url: str | None = None,
    bootstrap: bool | None = None,
    window_n: int = 101,
    show_full_prompt: bool = False,
    run_miner_path: bool = False,
    log_level: str = "INFO",
) -> int:
    from reliquary.constants import (
        BOOTSTRAP_SIGMA_MIN,
        BOOTSTRAP_WINDOWS,
        MAX_NEW_TOKENS_PROTOCOL_CAP,
        MAX_TRUNCATED_PER_SUBMISSION,
        M_ROLLOUTS,
        SIGMA_MIN,
    )
    from reliquary.miner.engine import (
        _compute_malformed_flags,
        _count_truncated,
        _passes_zone_filter,
        _rewards_from_generations,
        eos_set_from_model,
    )
    from reliquary.shared.prompt_augment import (
        canonical_prompt_tokens,
        generation_limits,
        generation_prompt_text,
        generation_prompt_tokens,
        max_new_tokens_for_generation,
        rebind_rollouts_to_canonical_prompt,
    )
    from reliquary.validator.verifier import rewards_std

    if bootstrap is None:
        bootstrap = window_n < BOOTSTRAP_WINDOWS
    sigma_threshold = BOOTSTRAP_SIGMA_MIN if bootstrap else SIGMA_MIN

    _out(f"\n>>> Miner prompt test — prompt_idx={prompt_idx} <<<")
    _out("Loading checkpoint and models (this may take ~1 min)...")

    resolved = _resolve_checkpoint(checkpoint, validator_url)
    engine = _load_engine(resolved, environment, use_vllm)
    _setup_logging(log_level)  # bittensor import resets logging; restore our config

    env = engine.env
    tokenizer = engine.tokenizer

    if prompt_idx < 0 or prompt_idx >= len(env):
        log.error("prompt_idx=%d out of range [0, %d)", prompt_idx, len(env))
        return 2

    problem = env.get_problem(prompt_idx)
    original = problem["prompt"]

    _step(1, "Load problem from environment")
    _out(f"prompt_idx={prompt_idx} env={env.name} bootstrap={bootstrap}")
    _out(f"problem id={problem.get('id', '?')} ground_truth={problem.get('ground_truth')!r}")
    log.info("prompt_idx=%d env=%s bootstrap=%s", prompt_idx, env.name, bootstrap)

    _step(2, "Prompt augmentation (generation-only suffix + token budget)")
    canonical = canonical_prompt_tokens(original, tokenizer)
    augmented_text = generation_prompt_text(original, tokenizer)
    augmented = generation_prompt_tokens(original, tokenizer)
    limits = generation_limits(original, tokenizer)
    max_new = limits["max_new_tokens"]

    _log_full_prompts(original, augmented_text, show_full=show_full_prompt)
    _out(f"canonical prompt tokens: {limits['canonical_len']}")
    _out(
        f"augmented prompt tokens:  {limits['augmented_len']} "
        f"(+{limits['prefix_overhead']} vs canonical)"
    )
    _out(f"generation uses chat template: {limits['uses_chat_template']}")
    _out(f"soft target (instruction text, advisory): {limits['soft_target_tokens']}")
    _out(f"vLLM/HF max_new_tokens (hard cap): {max_new}")
    _out(
        f"required vLLM max_model_len (this prompt): "
        f"{limits['required_max_model_len']}"
    )
    _out(
        f"max submitted total if completion fills max_new: "
        f"{limits['canonical_len'] + max_new} (protocol cap={MAX_NEW_TOKENS_PROTOCOL_CAP})"
    )

    _step(3, "EOS set (validator-aligned)")
    eos_set = eos_set_from_model(engine.hf_model, tokenizer)
    _out(f"eos_set={sorted(eos_set)}")

    _step(4, f"Generate M={M_ROLLOUTS} rollouts on AUGMENTED prefix (raw, pre-rebind)")
    _out(f"Generating {M_ROLLOUTS} rollouts (max_new={max_new}) — may take 1-3 min...")
    t0 = time.monotonic()
    raw_rollouts = _generate_raw_augmented(engine, problem, n_rollouts=M_ROLLOUTS)
    gen_s = time.monotonic() - t0
    _out(f"generation done in {gen_s:.1f}s → {len(raw_rollouts)} rollouts")

    if not raw_rollouts:
        _out("ERROR: No rollouts returned from generator")
        return 1

    raw_rewards = _rewards_from_generations(env, problem, raw_rollouts, tokenizer)
    raw_malformed = _compute_malformed_flags(raw_rollouts, raw_rewards, tokenizer)
    raw_reports = _analyze_rollouts(raw_rollouts, raw_rewards, raw_malformed, eos_set, tokenizer)
    _out("--- raw augmented rollouts ---")
    _log_rollout_table(raw_reports)
    raw_n_trunc = _count_truncated(raw_rollouts, eos_set)
    _out(f"raw truncation: {raw_n_trunc}/{M_ROLLOUTS} (local cap={_LOCAL_MAX_TRUNCATED})")
    _log_completion_samples(raw_rollouts, raw_rewards, tokenizer)
    _diagnose_head_trunc_screen(raw_rollouts, raw_rewards, eos_set, bootstrap=bootstrap)

    _step(5, "Rebind to canonical prompt (submitted rollout format)")
    rebound = rebind_rollouts_to_canonical_prompt(raw_rollouts, canonical, len(augmented))
    ok_bind, bind_msg = _prompt_binding_ok(rebound, canonical)
    _out(f"prompt binding check: {bind_msg}")
    for i, gen in enumerate(rebound):
        plen = gen["prompt_length"]
        prefix = gen["tokens"][:plen]
        _out(f"  rollout {i}: prompt_len={plen} prefix_ok={prefix == canonical} total={len(gen['tokens'])}")

    rebound_rewards = _rewards_from_generations(env, problem, rebound, tokenizer)
    rebound_malformed = _compute_malformed_flags(rebound, rebound_rewards, tokenizer)
    rebound_reports = _analyze_rollouts(
        rebound, rebound_rewards, rebound_malformed, eos_set, tokenizer,
    )
    _out("--- rebound (canonical prefix) rollouts ---")
    _log_rollout_table(rebound_reports)

    _step(6, "Validation gates (same as miner batched prep / _try_prompt_bundle)")
    sigma = rewards_std(rebound_rewards)
    n_trunc = _count_truncated(rebound, eos_set)
    n_malformed = sum(rebound_malformed)
    in_zone = _passes_zone_filter(rebound_rewards, bootstrap=bootstrap)
    term_ok = n_trunc <= _LOCAL_MAX_TRUNCATED

    _out(f"rewards (binary view): {[int(r) for r in rebound_rewards]}")
    _out(f"σ={sigma:.3f} (threshold={sigma_threshold:.2f}) → {'PASS' if in_zone else 'FAIL'}")
    _out(
        f"truncation {n_trunc}/{M_ROLLOUTS} (local cap={_LOCAL_MAX_TRUNCATED}, "
        f"validator cap={MAX_TRUNCATED_PER_SUBMISSION}) → {'PASS' if term_ok else 'FAIL'}"
    )
    _out(f"malformed \\boxed{{}}: {n_malformed}/{M_ROLLOUTS} → {'PASS' if n_malformed == 0 else 'FAIL'}")
    _out(f"prompt binding → {'PASS' if ok_bind else 'FAIL'}")

    bundle = None
    if run_miner_path:
        _step(7, "Production miner path (_try_prompt_bundle — includes zone screen)")
        _out("Re-running via _try_prompt_bundle (2-rollout zone screen + up to 8 total)...")
        gen_model = None if engine.gen_backend else engine.vllm_model
        gen_lock = threading.Lock()
        t1 = time.monotonic()
        bundle = engine._try_prompt_bundle(
            prompt_idx,
            problem,
            local_n=1,
            local_hash="test",
            bootstrap=bootstrap,
            gen_model=gen_model,
            gen_lock=gen_lock,
        )
        miner_s = time.monotonic() - t1
        if bundle is not None:
            _out(f"miner _try_prompt_bundle: HIT in {miner_s:.1f}s σ={bundle.sigma:.3f}")
        else:
            _out(f"miner _try_prompt_bundle: MISS in {miner_s:.1f}s")
    else:
        _out("")
        _out("(skipped STEP 7 — pass --run-miner-path to also run _try_prompt_bundle)")

    _step(8 if run_miner_path else 7, "VERDICT")
    would_stage = (
        ok_bind
        and len(rebound) >= M_ROLLOUTS
        and term_ok
        and in_zone
        and n_malformed == 0
    )
    if would_stage and (bundle is not None or not run_miner_path):
        _out("RESULT: Would be STAGED for submit (all gates pass)")
        return 0
    if would_stage and bundle is None and run_miner_path:
        _out("RESULT: gates pass but _try_prompt_bundle returned None — see zone screen diagnostic")
        return 1

    reasons = []
    if not ok_bind:
        reasons.append("prompt_binding")
    if len(rebound) < M_ROLLOUTS:
        reasons.append("gen_short")
    if not term_ok:
        reasons.append(f"too_truncated ({n_trunc}/{M_ROLLOUTS})")
    if not in_zone:
        reasons.append(f"out_of_zone (σ={sigma:.3f})")
    if n_malformed:
        reasons.append(f"malformed_final_answer ({n_malformed}/{M_ROLLOUTS})")
    _out(f"RESULT: Would NOT be staged — failed: {', '.join(reasons) or 'unknown'}")
    return 1


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run one prompt through the miner prep pipeline with step-by-step logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompt-idx", type=int, required=True, help="Environment prompt index")
    p.add_argument("--checkpoint", default="Qwen/Qwen3-4B-Instruct-2507", help="HF model path")
    p.add_argument("--environment", default=_DEFAULT_ENV, help="Environment name")
    p.add_argument("--use-vllm", action="store_true", help="Use vLLM generation backend")
    p.add_argument("--validator-url", default="", help="Fetch checkpoint from validator /state")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--bootstrap", action="store_true", default=None, help="Bootstrap σ threshold")
    g.add_argument("--no-bootstrap", action="store_true", default=None, help="Steady-state σ threshold")
    p.add_argument("--window-n", type=int, default=101, help="Window n for auto bootstrap mode")
    p.add_argument("--show-full-prompt", action="store_true", help="Print full prompt text (no truncation)")
    p.add_argument(
        "--run-miner-path", action="store_true",
        help="Also run _try_prompt_bundle (slow; uses 2-rollout zone screen)",
    )
    p.add_argument("--log-level", default="INFO", help="Log level")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.log_level)
    bootstrap: bool | None
    if args.bootstrap:
        bootstrap = True
    elif args.no_bootstrap:
        bootstrap = False
    else:
        bootstrap = None
    return run_prompt_test(
        args.prompt_idx,
        checkpoint=args.checkpoint,
        environment=args.environment,
        use_vllm=args.use_vllm,
        validator_url=args.validator_url or None,
        bootstrap=bootstrap,
        window_n=args.window_n,
        show_full_prompt=args.show_full_prompt,
        run_miner_path=args.run_miner_path,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    sys.exit(main())
