#!/usr/bin/env python3
"""check_prompt_acceptance.py — single-prompt validator-acceptance diagnostic.

Mirrors the live miner pipeline in ``reliquary/miner/engine.py`` for ONE prompt
(an integer index into the configured environment) and predicts whether the
on-chain validator would accept the resulting submission bundle.

Pipeline (kept tight against engine.py / verifier.py):
  1. transformers↔vLLM compat patches (from miner.generation).
  2. Resolve checkpoint path (default: latest snapshot under
     ``models--R0mAI--reliquary-sn-v23/snapshots``).
  3. Load tokenizer + HF proof model on cuda:<proof_gpu> (bf16).
  4. Build vLLM generator on cuda:<vllm_gpu> via build_vllm_generator
     (do NOT reimplement — the FLASH_ATTN env var and patch order are
     load-bearing on Blackwell).
  5. Load env via reliquary.environment.load_environment("openmathinstruct"),
     fetch problem at idx, tokenize raw prompt (NO chat template,
     add_special_tokens=False — engine.py:2095).
  6. Compute eos_set from hf_model.generation_config (NOT vLLM/tokenizer-only).
  7. Generate M_ROLLOUTS=8 completions via VLLMGenerator.generate.
  8. Local reward via env.compute_reward (engine._rewards_from_generations).
  9. Per-rollout p_stop AND completion_chosen_probs on the HF model from a
     single forward pass — replicates verifier._gpu_p_stop:344-356 and
     verifier._gpu_completion_chosen_probs:402-437 exactly.
 10. Predict per-rollout bad_termination using the validator's THRESHOLD
     (MIN_EOS_PROBABILITY = 0.01 on the SUMMED-eos-set p_stop), NOT the
     engine's single-token-logp shortcut.
 11. Run validator integrity gates that operate on the precomputed streams:
     - has_malformed_final_answer (per-rollout, reward<0.5)
     - detect_opposite_reward_clones (cross-rollout)
     - evaluate_boxed_answer_probability (per-rollout, hard, ≥ 0.001)
     - evaluate_token_distribution (per-rollout, hard tri-state — None passes)
 12. Print one line per rollout, then VERDICT block.

NOTE: gate order in the VERDICT mirrors batcher.py:642-835 exactly —
zone → malformed → clones → padding → too_truncated → bad_termination
budget → distribution → boxed_answer. The bad-termination budget uses
BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION when --bootstrap is set, matching
the validator's per-window cap selection (batcher.py:684-688).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

# Make sure /root/reliqua is importable when this is run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# -------- defaults --------
DEFAULT_SNAPSHOT_ROOT = (
    "/root/.cache/huggingface/hub/"
    "models--R0mAI--reliquary-sn-v23/snapshots"
)


def _resolve_default_checkpoint() -> str:
    """Pick the most-recently-modified snapshot directory under the HF cache."""
    root = Path(DEFAULT_SNAPSHOT_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(
            f"No snapshots directory at {root}; pass --checkpoint explicitly."
        )
    snaps = [p for p in root.iterdir() if p.is_dir()]
    if not snaps:
        raise FileNotFoundError(
            f"No snapshot subdirs under {root}; pass --checkpoint explicitly."
        )
    snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(snaps[0])


def _decode_tail(tokenizer, tokens: list[int], n_chars: int = 30) -> str:
    """Decode last n_chars of completion (sanitised for single-line print)."""
    if not tokens:
        return ""
    txt = tokenizer.decode(tokens[-32:])  # ~32 tokens is plenty for 30 chars
    txt = txt.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    if len(txt) > n_chars:
        txt = txt[-n_chars:]
    return txt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Predict whether the validator would accept a submission for a "
            "given prompt index."
        ),
    )
    ap.add_argument("prompt_id", type=int, help="Integer index into the env")
    ap.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint path (default: latest snapshot under HF cache).",
    )
    ap.add_argument("--gpu-vllm", type=int, default=0, help="vLLM GPU index.")
    ap.add_argument("--gpu-proof", type=int, default=1, help="HF proof GPU.")
    ap.add_argument(
        "--bootstrap", action="store_true",
        help="Use bootstrap σ threshold (0.33) instead of steady-state (0.43).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="(vLLM SamplingParams does not use seed in the live miner; "
             "kept for visibility — generation is non-deterministic.)",
    )
    args = ap.parse_args(argv)

    # -- Step 1: compat patches BEFORE anything imports vllm/transformers heavy.
    from reliquary.miner.generation import apply_transformers_compat_patches
    apply_transformers_compat_patches()

    # Heavy imports only after patches.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reliquary.constants import (
        BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION,
        BOXED_ANSWER_MIN_PROB,
        LAYER_INDEX,
        MAX_NEW_TOKENS_PROTOCOL_CAP,
        MAX_TRUNCATED_PER_SUBMISSION,
        MIN_EOS_PROBABILITY,
        M_ROLLOUTS,
        SIGMA_MIN,
        BOOTSTRAP_SIGMA_MIN,
        T_PROTO,
        TOP_P_PROTO,
    )
    from reliquary.environment import load_environment
    from reliquary.miner.engine import (
        _LOCAL_MAX_TRUNCATED,
        _count_eos_padding,
        _count_truncated,
        _passes_zone_filter,
        _rewards_from_generations,
        eos_set_from_model,
    )
    from reliquary.shared.forward import forward_single_layer
    from reliquary.shared.hf_compat import resolve_attn_implementation
    from reliquary.validator.boxed_integrity import has_malformed_final_answer
    from reliquary.validator.rollout_patterns import detect_opposite_reward_clones
    from reliquary.validator.verifier import (
        ProofResult,
        _gpu_completion_chosen_probs,
        evaluate_boxed_answer_probability,
        evaluate_token_distribution,
        rewards_std,
    )

    # -- Step 2: resolve checkpoint.
    checkpoint = args.checkpoint or _resolve_default_checkpoint()
    print(f"[diag] checkpoint = {checkpoint}")
    print(f"[diag] prompt_id  = {args.prompt_id}")
    print(f"[diag] gpu_vllm   = cuda:{args.gpu_vllm}")
    print(f"[diag] gpu_proof  = cuda:{args.gpu_proof}")
    print(f"[diag] bootstrap  = {args.bootstrap}")

    # -- Step 3: tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # -- Step 5 (env first — pure CPU, cheap, lets us fail fast on bad idx).
    env = load_environment("openmathinstruct")
    problem = env.get_problem(args.prompt_id)
    prompt_text = problem["prompt"]
    print(f"[diag] prompt (first 300 chars):\n{prompt_text[:300]!r}")
    if "ground_truth" in problem:
        print(f"[diag] ground_truth = {problem['ground_truth']!r}")

    # -- Step 4: HF proof model on cuda:<proof_gpu> BEFORE build_vllm_generator,
    #   because build_vllm_generator mutates CUDA_VISIBLE_DEVICES; once that's
    #   done vLLM only sees one device, but pre-existing tensors on cuda:1 keep
    #   working through the established CUDA context.
    proof_device = f"cuda:{args.gpu_proof}"
    attn_impl = resolve_attn_implementation()
    print(f"[diag] loading HF proof model on {proof_device} (attn={attn_impl})...")
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        .to(proof_device)
        .eval()
    )

    # -- Step 6: eos_set from the HF proof model (validator-aligned).
    eos_set = eos_set_from_model(hf_model, tokenizer)
    print(f"[diag] eos_set = {sorted(eos_set)}")

    # -- Step 4 cont: vLLM on cuda:<vllm_gpu>.
    from reliquary.miner.generation import build_vllm_generator
    print(f"[diag] building vLLM on cuda:{args.gpu_vllm}...")
    gen_backend = build_vllm_generator(checkpoint, gpu=args.gpu_vllm)

    # -- Step 7: tokenize prompt (NO chat template, add_special_tokens=False).
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_length = len(prompt_tokens)
    max_new = min(
        MAX_NEW_TOKENS_PROTOCOL_CAP,
        max(1, MAX_NEW_TOKENS_PROTOCOL_CAP - prompt_length),
    )
    print(
        f"[diag] prompt_length = {prompt_length} tokens, "
        f"max_new_tokens = {max_new}, T={T_PROTO}, top_p={TOP_P_PROTO}"
    )

    print(f"[diag] generating {M_ROLLOUTS} rollouts via vLLM...")
    generations = gen_backend.generate(
        prompt_tokens, M_ROLLOUTS,
        max_new_tokens=max_new,
        eos_set=eos_set,
        tokenizer=tokenizer,
    )
    assert len(generations) == M_ROLLOUTS, (
        f"expected {M_ROLLOUTS} rollouts, got {len(generations)}"
    )

    # -- Step 8: rewards via the same path engine uses.
    rewards = _rewards_from_generations(env, problem, generations, tokenizer)
    sigma = rewards_std(rewards)
    print(f"[diag] rewards = {rewards}")
    print(f"[diag] σ = {sigma:.4f}")

    # -- Step 9: truncation count and EOS-padding count, validator-aligned.
    n_trunc = _count_truncated(generations, eos_set)
    n_padding = _count_eos_padding(generations, eos_set)
    in_zone = _passes_zone_filter(rewards, bootstrap=args.bootstrap)
    sigma_threshold = BOOTSTRAP_SIGMA_MIN if args.bootstrap else SIGMA_MIN
    # Validator picks the bootstrap budget when bootstrap is set (see
    # batcher.py:684-688). Mirror that selection here so the truncation gate
    # uses the same cap the deployed validator applies to this submission.
    validator_max_truncated = (
        BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION
        if args.bootstrap
        else MAX_TRUNCATED_PER_SUBMISSION
    )

    # -- Step 9b: per-rollout completion text (decoded completion-only).
    # The malformed-final-answer gate and the opposite-reward-clone gate both
    # operate on the completion-only decoded string (validator strips the
    # prompt via _completion_text → tokenizer.decode(tokens[prompt_length:])).
    completion_texts: list[str] = []
    for gen in generations:
        tokens_g: list[int] = gen["tokens"]
        pl_g: int = gen["prompt_length"]
        completion_texts.append(tokenizer.decode(tokens_g[pl_g:]))

    # -- Step 10: per-rollout HF forward pass → p_stop + completion_chosen_probs
    # (validator math, mirrors verifier._gpu_p_stop and
    # _gpu_completion_chosen_probs from the same forward pass).
    print("[diag] running per-rollout HF forward passes for p_stop + probs...")
    eos_sorted = sorted(eos_set)
    eos_idx_tensor = (
        torch.tensor(eos_sorted, device=proof_device, dtype=torch.long)
        if eos_sorted else None
    )
    min_eos_logp = math.log(max(MIN_EOS_PROBABILITY, 1e-12))

    per_rollout_lines: list[str] = []
    predicted_bad_count = 0
    # Per-rollout outputs reused by the malformed/clone/boxed gates after the
    # loop closes; keeping them parallel to ``generations`` lets the gate code
    # read by index without re-deriving anything from the commit dict.
    completion_chosen_probs_list: list[list[float]] = []
    for i, gen in enumerate(generations):
        tokens: list[int] = gen["tokens"]
        pl: int = gen["prompt_length"]
        completion = tokens[pl:]
        total_len = len(tokens)
        completion_len = len(completion)

        # has_eos_padding (per-rollout, mirrors verifier.has_eos_padding).
        eos_positions = [
            j for j, t in enumerate(completion) if int(t) in eos_set
        ]
        has_padding = bool(eos_positions) and (
            len(eos_positions) > 1
            or eos_positions[0] != completion_len - 1
        )

        last_tok = int(tokens[-1]) if tokens else -1
        in_eos = last_tok in eos_set

        # p_stop = sum_{e in eos_set} softmax(logits[seq_len-2])[e]
        # — replicates verifier._gpu_p_stop EXACTLY.
        # completion_chosen_probs[j] = softmax(logits[t-1] / T_PROTO)[tokens[t]]
        # for valid completion-producing steps t — exactly what
        # verifier._gpu_completion_chosen_probs returns from the same forward.
        p_stop: float | None = None
        last_token_logp: float | None = None
        completion_chosen_probs: list[float] = []
        with torch.no_grad():
            proof_input = torch.tensor([tokens], device=proof_device)
            _hidden, logits = forward_single_layer(
                hf_model, proof_input, None, LAYER_INDEX,
            )
            # logits shape: [1, seq_len, vocab]
            seq_len = logits.size(1)
            if seq_len >= 2 and eos_idx_tensor is not None:
                probs_last = torch.softmax(
                    logits[0, seq_len - 2].float(), dim=-1,
                )
                p_stop = float(probs_last[eos_idx_tensor].sum().item())
                last_token_logp = float(
                    torch.log_softmax(
                        logits[0, seq_len - 2].float(), dim=-1,
                    )[last_tok].item()
                )
            # Reuse the same logits tensor for the chosen-probability stream.
            # logits_gpu is [seq_len, vocab]; helper handles all boundary/skip
            # logic that the boxed-answer + distribution gates require.
            completion_chosen_probs = _gpu_completion_chosen_probs(
                logits[0],
                tokens,
                pl,
                completion_len,
                seq_len,
                proof_device,
            )
        completion_chosen_probs_list.append(completion_chosen_probs)

        # Validator's bad_termination logic (verifier.verify_termination +
        # is_cap_truncation), reduced to per-rollout boolean.
        cap_hit = total_len >= MAX_NEW_TOKENS_PROTOCOL_CAP
        # natural-EOS gate: in_eos AND p_stop >= MIN_EOS_PROBABILITY
        eos_gate_ok = (
            in_eos
            and p_stop is not None
            and p_stop >= MIN_EOS_PROBABILITY
        )
        if cap_hit:
            # Path 1 termination_ok is True; cap_truncated is NOT eos_gate_ok.
            termination_ok = True
            cap_truncated = not eos_gate_ok
        else:
            # Path 2: needs eos_gate_ok.
            termination_ok = eos_gate_ok
            cap_truncated = False
        rollout_bad = (not termination_ok) or cap_truncated
        if rollout_bad:
            predicted_bad_count += 1

        # Cleanup before next forward to keep peak VRAM bounded.
        del logits, _hidden, proof_input
        torch.cuda.empty_cache()

        tail = _decode_tail(tokenizer, completion, n_chars=30)
        p_stop_str = (
            f"{p_stop:.4f}" if p_stop is not None else "N/A"
        )
        last_logp_str = (
            f"{last_token_logp:.3f}" if last_token_logp is not None else "N/A"
        )
        per_rollout_lines.append(
            f"#{i}: last_tok={last_tok} in_eos={'T' if in_eos else 'F'} "
            f"p_stop={p_stop_str} last_token_logp={last_logp_str} "
            f"total_len={total_len} eos_pos={eos_positions} "
            f"has_padding={'T' if has_padding else 'F'} "
            f"predicted_bad={'T' if rollout_bad else 'F'} "
            f"reward={rewards[i]:.1f} "
            f"tail={tail!r}"
        )

    # -- Step 10b: validator's cross-rollout + per-rollout integrity gates,
    # using the completion texts and chosen-prob streams gathered above.

    # Malformed-final-answer (per-rollout, batcher.py:650-664). A reward<0.5
    # rollout whose LAST \boxed{}/\fbox{} is unclosed, empty, or contains a
    # SPECIAL_TOKEN is rejected as a fake negative.
    n_malformed = 0
    malformed_first_idx: int | None = None
    for i, text in enumerate(completion_texts):
        bad, _reason = has_malformed_final_answer(
            float(rewards[i]),
            text,
            completion_length=len(generations[i]["tokens"]) - generations[i]["prompt_length"],
            cap=MAX_NEW_TOKENS_PROTOCOL_CAP,
        )
        if bad:
            n_malformed += 1
            if malformed_first_idx is None:
                malformed_first_idx = i

    # Opposite-reward-clone detection (cross-rollout, batcher.py:666-674).
    clone_metrics = detect_opposite_reward_clones(
        completion_texts, [float(r) for r in rewards]
    )
    clones_suspicious = bool(clone_metrics.suspicious)

    # Per-rollout boxed-answer min-probability + soft token-distribution gates.
    # Both read completion_chosen_probs[i] computed above. Distribution is a
    # tri-state SOFT filter (None == accept); boxed is a HARD reject.
    n_boxed_low_prob = 0
    boxed_first_idx: int | None = None
    boxed_min_prob_overall: float | None = None
    n_dist_suspicious = 0
    dist_first_idx: int | None = None
    dist_q10_min: float | None = None
    for i, gen in enumerate(generations):
        tokens_g: list[int] = gen["tokens"]
        pl_g: int = gen["prompt_length"]
        completion_len_g = len(tokens_g) - pl_g
        # Build a ProofResult shim carrying the chosen-prob stream. The two
        # evaluate_* functions only read .completion_chosen_probs from proof,
        # so the other fields are inert defaults.
        proof_shim = ProofResult(
            all_passed=True,
            passed=0,
            checked=0,
            has_sparse_outputs=True,
            completion_chosen_probs=completion_chosen_probs_list[i],
        )
        boxed_ok, boxed_metrics = evaluate_boxed_answer_probability(
            tokens=tokens_g,
            prompt_length=pl_g,
            completion_length=completion_len_g,
            proof=proof_shim,
            tokenizer=tokenizer,
        )
        if boxed_metrics and "min_prob" in boxed_metrics:
            mp = float(boxed_metrics["min_prob"])
            if boxed_min_prob_overall is None or mp < boxed_min_prob_overall:
                boxed_min_prob_overall = mp
        if not boxed_ok:
            n_boxed_low_prob += 1
            if boxed_first_idx is None:
                boxed_first_idx = i

        dist_ok, dist_metrics = evaluate_token_distribution(
            tokens=tokens_g,
            prompt_length=pl_g,
            completion_length=completion_len_g,
            proof=proof_shim,
        )
        if dist_metrics and "q10" in dist_metrics:
            q10 = float(dist_metrics["q10"])
            if dist_q10_min is None or q10 < dist_q10_min:
                dist_q10_min = q10
        if dist_ok is False:
            n_dist_suspicious += 1
            if dist_first_idx is None:
                dist_first_idx = i

    print("\n=== PER-ROLLOUT DIAGNOSTICS ===")
    for line in per_rollout_lines:
        print(line)

    # -- Final verdict.
    print("\n=== VERDICT ===")
    print(
        f"σ = {sigma:.4f} "
        f"(threshold = {sigma_threshold:.2f}; "
        f"in_zone={'T' if in_zone else 'F'})"
    )
    print(
        f"truncated = {n_trunc}/{M_ROLLOUTS} "
        f"(local cap = {_LOCAL_MAX_TRUNCATED}, "
        f"validator cap = {validator_max_truncated})"
    )
    print(
        f"eos_padding = {n_padding} (any > 0 → immediate reject)"
    )
    print(
        f"predicted_bad = {predicted_bad_count}/{M_ROLLOUTS} "
        f"(validator cap = {validator_max_truncated})"
    )
    print(
        f"malformed_final = {n_malformed}/{M_ROLLOUTS}"
        + (
            f" (first reject idx = {malformed_first_idx})"
            if malformed_first_idx is not None
            else ""
        )
    )
    print(
        f"opposite_reward_clones: suspicious={clones_suspicious} "
        f"reward_vec={clone_metrics.reward_vector} "
        f"matched_pairs={clone_metrics.matched_pairs} "
        f"mirror_pairs={clone_metrics.mirror_pairs} "
        f"max_sim={clone_metrics.max_similarity:.4f}"
    )
    print(
        f"boxed_low_prob = {n_boxed_low_prob}/{M_ROLLOUTS}"
        f" (threshold = {BOXED_ANSWER_MIN_PROB:.3g}, "
        f"min_prob_seen = "
        f"{'N/A' if boxed_min_prob_overall is None else f'{boxed_min_prob_overall:.4g}'})"
        + (
            f" (first reject idx = {boxed_first_idx})"
            if boxed_first_idx is not None
            else ""
        )
    )
    # Soft filter — surfaced as a warning so a tripped distribution check is
    # visible even though it is not in the reject chain when paired with a
    # hard gate that fired first. The validator treats False as a HARD reject;
    # we still apply that below.
    print(
        f"dist_suspicious(soft) = {n_dist_suspicious}/{M_ROLLOUTS} "
        f"dist_q10_min = "
        f"{'N/A' if dist_q10_min is None else f'{dist_q10_min:.4g}'}"
    )

    # Apply gates in the validator's ACTUAL batcher order
    # (reliquary/validator/batcher.py:642-835), reduced to the checks we
    # can simulate locally:
    #
    #   1. zone           — OUT_OF_ZONE  (sigma)
    #   2. malformed      — MALFORMED_FINAL_ANSWER (per-rollout, reward<0.5)
    #   3. clones         — DISTRIBUTION_SUSPICIOUS (cross-rollout)
    #   4. padding        — BAD_TERMINATION (per-rollout, instant)
    #   5. too_truncated  — local miner gate (engine.py:_LOCAL_MAX_TRUNCATED)
    #   6. bad_termination_budget — BAD_TERMINATION (per-rollout cap)
    #   7. dist_distribution — DISTRIBUTION_SUSPICIOUS (per-rollout, hard)
    #   8. boxed_answer   — BOXED_ANSWER_TAMPERED (per-rollout, hard)
    #
    # Signature / randomness / GRAIL / logprob run in between but cannot be
    # simulated without the miner's commit payload; they are silently
    # assumed-pass here.
    if not in_zone:
        reason = "out_of_zone"
        accept = False
    elif n_malformed > 0:
        reason = "malformed_final_answer"
        accept = False
    elif clones_suspicious:
        reason = "opposite_reward_clones"
        accept = False
    elif n_padding > 0:
        reason = "eos_padding"
        accept = False
    elif n_trunc > _LOCAL_MAX_TRUNCATED:
        reason = "too_truncated"
        accept = False
    elif predicted_bad_count > validator_max_truncated:
        reason = "bad_termination_budget"
        accept = False
    elif n_dist_suspicious > 0:
        reason = "distribution_suspicious"
        accept = False
    elif n_boxed_low_prob > 0:
        reason = "boxed_answer_tampered"
        accept = False
    else:
        reason = "ACCEPTED"
        accept = True

    print(f"WOULD VALIDATOR ACCEPT? {'YES' if accept else 'NO'}")
    print(f"reason: {reason}")

    return 0 if accept else 1


if __name__ == "__main__":
    raise SystemExit(main())
