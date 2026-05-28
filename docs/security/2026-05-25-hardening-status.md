# 2026-05-25 Validator Hardening Status

Status: shipped and deployed

This note summarizes the live security posture after the 2026-05-25 exploit
response cycle. Historical incident write-ups remain in this directory; this
file is the current-state pointer for operators and future audits.

## Shipped Fixes

| Area | PR | Current behavior |
| --- | --- | --- |
| Canonical rollout tokens | #41 | `rollout.tokens` must match `commit["tokens"]`; reward, archive, and training consume committed tokens |
| Clean training reset | #42 | Validator reset to a clean checkpoint after token-split poisoning risk |
| Cap-path termination | #43, #47 | Steady state rejects any cap/non-EOS truncated rollout as `bad_termination` |
| EOS padding | #44 | Repeated EOS / tokens after first EOS reject as `bad_termination` |
| Reward distribution | #46 | OpenMath steady-state binary groups outside k=3..5 reject as `reward_distribution` |
| Cheap reject ordering | #47 | Proof-free schema/token/prompt/reward-distribution checks run before expensive GRAIL work |
| Training quarantine | #48 | High-confidence suspicious selected windows archive/credit but skip GRPO and checkpoint publish |
| Reward-claim verification restored | #49 reverted | Miners submit local `env.compute_reward` values; validator recomputes and rejects mismatches before sigma/archive/training |

## Live Design Invariants

- The canonical token source is `rollout.commit["tokens"]`.
- Miners should submit honest local reward claims; the validator verifies them.
- Steady-state OpenMath binary rewards must have 3, 4, or 5 correct rollouts
  out of 8.
- Steady-state training does not ingest cap/non-EOS truncated completions.
- EOS ends a completion; repeated EOS padding is invalid.
- Suspicious windows can still count for emission accounting, but not for model
  updates.
- R2 selected batch entries include the verifier-checked reward claims used for
  sigma and training.

## Remaining Design Risk

Reward-claim verification is not reward secrecy. OpenMath labels are
public/reconstructable, so a sophisticated miner may still infer labels outside
the stock miner and optimize candidate-pool selection or formatting behavior.

PR #49's validator-owned reward experiment was reverted because it removed too
much miner-side frontier signal before private/generated tasks were ready.

The next structural fix should not be another narrow heuristic unless live data
forces it. The durable direction is:

1. private/generated validator-side math tasks;
2. public prompt view without stable public label mapping;
3. archive redaction or delay for label-bearing data;
4. environment-level `RewardPolicy` for scoring, group acceptance, quarantine,
   and archive behavior;
5. commit-first sampling if miners continue to shape candidate pools after
   reward secrecy improves.

## Data To Watch

- `training_quarantine.quarantined` rate and reasons;
- `reward_distribution` rejects after k=3..5;
- `bad_termination` rejects after cap/EOS hardening;
- accepted reward-vector repetition or monotonicity by hotkey;
- per-hotkey batch dominance and prompt sharing;
- completion length distribution and repeated-prefix patterns;
- W&B `train/kl`, `train/ppo_loss`, `train/valid_rollout_ratio`, and
  `train/rollouts_processed`;
- whether previously watched hotkeys adapt into new patterns.

## Operator Rule

Preserve checkpoint lineage on deploy. Unless deliberately resetting, restart
the trainer with:

```text
RELIQUARY_RESUME_FROM=sha:<current /health checkpoint_revision>
```
