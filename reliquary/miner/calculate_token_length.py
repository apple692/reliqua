"""Compute validator-aligned token lengths and log per-rollout generation details.

Canonical prompt binding (validator)::

    tokenizer.encode(problem["prompt"], add_special_tokens=False)

Generate rollouts from a checkpoint (same path as ``test.py``), rebind to
canonical prefix, and log prompt_idx, prompt, response, and token lengths.

Usage::

    # Generate M rollouts and log each rollout in detail:
    python -m reliquary.miner.calculate_token_length \\
        --prompt-idx 433264 \\
        --checkpoint Qwen/Qwen3-4B-Instruct-2507 \\
        --use-vllm

    # Match running miner checkpoint:
    python -m reliquary.miner.calculate_token_length \\
        --prompt-idx 433264 \\
        --validator-url http://86.38.238.30:8080 \\
        --use-vllm

    # Manual prompt + completion text (no generation):
    python -m reliquary.miner.calculate_token_length \\
        --checkpoint /path/to/model \\
        --no-generate --prompt-text "..." --completion-text "..."
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from reliquary.constants import (
    ENVIRONMENT_NAME,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
)
from reliquary.shared.prompt_augment import (
    canonical_prompt_tokens,
    completion_token_budget,
    generation_prompt_tokens,
    max_new_tokens_for_generation,
    rebind_rollouts_to_canonical_prompt,
)

log = logging.getLogger("reliquary.miner.calculate_token_length")

_DEFAULT_ENV = os.getenv("RELIQUARY_ENVIRONMENT_NAME", ENVIRONMENT_NAME)


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
    print(msg, flush=True)


def tokenize_canonical_prompt(prompt_text: str, tokenizer) -> list[int]:
    """Tokenize env prompt exactly like the validator prompt-binding check."""
    return canonical_prompt_tokens(prompt_text, tokenizer)


def tokenize_submitted_rollout(
    prompt_text: str,
    completion_text: str,
    tokenizer,
) -> tuple[list[int], int, int]:
    """Return ``(tokens, prompt_length, completion_length)`` for submitted format."""
    prompt_tokens = tokenize_canonical_prompt(prompt_text, tokenizer)
    full_tokens = list(
        tokenizer.encode(prompt_text + completion_text, add_special_tokens=False)
    )
    plen = len(prompt_tokens)
    if full_tokens[:plen] != prompt_tokens:
        raise ValueError(
            "Prompt is not a token prefix of prompt+completion — "
            "use raw generated token ids instead of re-encoding completion alone."
        )
    return full_tokens, plen, len(full_tokens) - plen


def lengths_from_texts(
    prompt_text: str,
    completion_text: str | None,
    tokenizer,
) -> dict[str, Any]:
    """Return validator-aligned length fields for prompt-only or full rollout."""
    prompt_length = len(tokenize_canonical_prompt(prompt_text, tokenizer))
    max_new = max_new_tokens_for_generation(prompt_text, tokenizer)
    protocol_room = completion_token_budget(prompt_text, tokenizer)
    out: dict[str, Any] = {
        "prompt_length": prompt_length,
        "completion_length": 0,
        "total_length": prompt_length,
        "max_new_tokens_protocol": max_new,
        "protocol_completion_ceiling": protocol_room,
        "protocol_cap_total": MAX_NEW_TOKENS_PROTOCOL_CAP,
        "tokens_remaining_to_cap": MAX_NEW_TOKENS_PROTOCOL_CAP - prompt_length,
    }
    if completion_text is None:
        return out

    tokens, plen, completion_length = tokenize_submitted_rollout(
        prompt_text, completion_text, tokenizer,
    )
    total_length = plen + completion_length
    out.update({
        "prompt_length": plen,
        "completion_length": completion_length,
        "total_length": total_length,
        "tokens_to_protocol_cap": MAX_NEW_TOKENS_PROTOCOL_CAP - total_length,
        "at_protocol_cap": total_length >= MAX_NEW_TOKENS_PROTOCOL_CAP,
    })
    return out


def load_tokenizer(checkpoint: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def run_generate(
    prompt_idx: int,
    *,
    checkpoint: str,
    environment: str,
    use_vllm: bool,
    validator_url: str | None,
    n_rollouts: int,
    show_full_prompt: bool,
    show_full_response: bool,
    log_level: str,
    as_json: bool,
) -> int:
    from reliquary.environment import load_environment
    from reliquary.miner.engine import _rewards_from_generations, eos_set_from_model
    from reliquary.miner.rollout_log import log_staged_rollouts
    from reliquary.miner.test import (
        _generate_raw_augmented,
        _load_engine,
        _resolve_checkpoint,
    )
    from reliquary.validator.verifier import rewards_std

    if show_full_prompt:
        os.environ["MINER_ROLLOUT_LOG_PROMPT_CHARS"] = "0"
    if show_full_response:
        os.environ["MINER_ROLLOUT_LOG_RESPONSE_CHARS"] = "0"

    _setup_logging(log_level)
    resolved = _resolve_checkpoint(checkpoint, validator_url)
    log.info("resolved checkpoint=%s", resolved)

    engine = _load_engine(resolved, environment, use_vllm)
    env = engine.env

    if prompt_idx < 0 or prompt_idx >= len(env):
        log.error("prompt_idx=%d out of range [0, %d)", prompt_idx, len(env))
        return 2

    problem = env.get_problem(prompt_idx)
    prompt_text = problem["prompt"]
    canonical = canonical_prompt_tokens(prompt_text, engine.tokenizer)
    augmented = generation_prompt_tokens(prompt_text, engine.tokenizer)
    max_new = max_new_tokens_for_generation(prompt_text, engine.tokenizer)
    eos_set = eos_set_from_model(engine.hf_model, engine.tokenizer)

    _out("")
    _out(f">>> generate + token lengths  prompt_idx={prompt_idx} <<<")
    log.info(
        "start prompt_idx=%d env=%s canonical_len=%d augmented_len=%d max_new=%d",
        prompt_idx, environment, len(canonical), len(augmented), max_new,
    )

    t0 = time.monotonic()
    raw_rollouts = _generate_raw_augmented(engine, problem, n_rollouts=n_rollouts)
    gen_s = time.monotonic() - t0
    log.info("generation done in %.1fs rollouts=%d", gen_s, len(raw_rollouts))

    if not raw_rollouts:
        log.error("generation returned 0 rollouts")
        return 1

    rebound = rebind_rollouts_to_canonical_prompt(
        raw_rollouts, canonical, len(augmented),
    )
    rewards = _rewards_from_generations(env, problem, rebound, engine.tokenizer)
    sigma = rewards_std(rewards)

    _out("")
    _out("--- SUMMARY ---")
    summary = (
        f"prompt_idx={prompt_idx}  environment={environment}  "
        f"problem_id={problem.get('id', '?')}  ground_truth={problem.get('ground_truth')!r}\n"
        f"checkpoint={resolved}\n"
        f"canonical_prompt_tokens={len(canonical)}  "
        f"augmented_prompt_tokens={len(augmented)}  max_new={max_new}\n"
        f"generation_time={gen_s:.1f}s  rollouts={len(rebound)}  "
        f"σ={sigma:.3f}  rewards={[int(r) for r in rewards]}"
    )
    _out(summary)
    log.info(summary.replace("\n", " | "))

    log_staged_rollouts(
        log,
        prompt_idx=prompt_idx,
        problem=problem,
        generations=rebound,
        rewards=rewards,
        sigma=sigma,
        tokenizer=engine.tokenizer,
        eos_set=eos_set,
        source="calculate_token_length",
    )

    if as_json:
        payload = {
            "prompt_idx": prompt_idx,
            "environment": environment,
            "checkpoint": resolved,
            "problem_id": problem.get("id"),
            "ground_truth": problem.get("ground_truth"),
            "canonical_prompt_length": len(canonical),
            "augmented_prompt_length": len(augmented),
            "max_new_tokens": max_new,
            "generation_seconds": round(gen_s, 2),
            "sigma": sigma,
            "rewards": [int(r) for r in rewards],
        }
        _out("")
        _out(json.dumps(payload, indent=2))
    return 0


def _read_text(path: str | None, inline: str | None) -> str | None:
    if inline is not None:
        return inline
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate rollouts from a checkpoint (or measure manual text) and "
            "log validator-aligned token lengths per rollout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="HF checkpoint path or hub id")
    p.add_argument("--environment", default=_DEFAULT_ENV, help="Environment name")
    p.add_argument(
        "--prompt-idx",
        type=int,
        default=None,
        help="Env prompt index (enables generation unless --no-generate)",
    )
    p.add_argument("--validator-url", default="", help="Fetch checkpoint from /state")
    p.add_argument("--use-vllm", action="store_true", help="Use vLLM generation backend")
    p.add_argument(
        "--n-rollouts",
        type=int,
        default=M_ROLLOUTS,
        help="Rollouts to generate (protocol M=8 for submissions)",
    )
    p.add_argument(
        "--generate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate from checkpoint (default: on when --prompt-idx and no manual completion)",
    )
    p.add_argument("--prompt-text", default=None, help="Manual canonical prompt text")
    p.add_argument("--prompt-file", default=None, help="File with prompt text")
    p.add_argument("--completion-text", default=None, help="Manual completion text")
    p.add_argument("--completion-file", default=None, help="File with completion text")
    p.add_argument(
        "--text",
        default=None,
        help="Tokenize one string only (add_special_tokens=False)",
    )
    p.add_argument("--text-file", default=None, help="File for --text mode")
    p.add_argument(
        "--show-full-prompt",
        action="store_true",
        help="Do not truncate logged prompt text",
    )
    p.add_argument(
        "--show-full-response",
        action="store_true",
        help="Do not truncate logged response text",
    )
    p.add_argument("--log-level", default="INFO", help="Logging level")
    p.add_argument("--json", action="store_true", help="Also emit JSON summary on stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.log_level)

    if args.text is not None or args.text_file is not None:
        text = _read_text(args.text_file, args.text)
        if text is None:
            log.error("--text or --text-file required")
            return 2
        tokenizer = load_tokenizer(args.checkpoint)
        token_length = len(tokenizer.encode(text, add_special_tokens=False))
        result = {"mode": "single_string", "token_length": token_length}
        if args.json:
            _out(json.dumps(result, indent=2))
        else:
            msg = f"token_length={token_length} (add_special_tokens=False)"
            _out(msg)
            log.info(msg)
        return 0

    manual_completion = _read_text(args.completion_file, args.completion_text) is not None
    want_generate = args.generate
    if want_generate is None:
        want_generate = args.prompt_idx is not None and not manual_completion

    if want_generate:
        if args.prompt_idx is None:
            log.error("--prompt-idx required for --generate")
            return 2
        return run_generate(
            args.prompt_idx,
            checkpoint=args.checkpoint,
            environment=args.environment,
            use_vllm=args.use_vllm,
            validator_url=args.validator_url or None,
            n_rollouts=args.n_rollouts,
            show_full_prompt=args.show_full_prompt,
            show_full_response=args.show_full_response,
            log_level=args.log_level,
            as_json=args.json,
        )

    prompt_text: str | None = None
    if args.prompt_idx is not None:
        from reliquary.environment import load_environment

        env = load_environment(args.environment)
        if args.prompt_idx < 0 or args.prompt_idx >= len(env):
            log.error("prompt_idx=%d out of range [0, %d)", args.prompt_idx, len(env))
            return 2
        prompt_text = env.get_problem(args.prompt_idx)["prompt"]
    else:
        prompt_text = _read_text(args.prompt_file, args.prompt_text)

    if prompt_text is None:
        log.error("provide --prompt-idx (with --generate) or --prompt-text/--prompt-file")
        return 2

    tokenizer = load_tokenizer(args.checkpoint)
    completion_text = _read_text(args.completion_file, args.completion_text)
    result = lengths_from_texts(prompt_text, completion_text, tokenizer)
    if args.prompt_idx is not None:
        result["prompt_idx"] = args.prompt_idx
        result["environment"] = args.environment

    if args.json:
        _out(json.dumps(result, indent=2))
        return 0

    lines = [
        f"prompt_length={result['prompt_length']}",
        f"max_new_tokens_protocol={result['max_new_tokens_protocol']}",
        f"protocol_completion_ceiling={result['protocol_completion_ceiling']}",
        f"protocol_cap_total={result['protocol_cap_total']}",
    ]
    if completion_text is not None:
        lines.extend([
            f"completion_length={result['completion_length']}",
            f"total_length={result['total_length']}",
            f"tokens_to_protocol_cap={result['tokens_to_protocol_cap']}",
            f"at_protocol_cap={result['at_protocol_cap']}",
        ])
    for line in lines:
        _out(line)
        log.info(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
