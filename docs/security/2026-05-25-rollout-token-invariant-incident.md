# Rollout token invariant incident

Date: 2026-05-25

Status: patched in this worktree, pending deploy.

## Executive summary

The report from Gilles is substantially correct: the validator accepted a
`RolloutSubmission` with two independent token sequences:

- `rollout.commit["tokens"]`, used by GRAIL, signature verification,
  termination, logprob verification, and token-distribution checks.
- `rollout.tokens`, formerly used by reward decoding, R2 archive output, and
  GRPO training input.

Before this patch, no invariant required those two fields to be equal. A miner
could prove one sequence cryptographically and ask the validator to score and
archive a different sequence.

The exploiter evidence is also strongly supported for hotkey
`5DAvrWnM8MygaYq5dyC1bs8tP71Pd8d5QF5yRBN36ggkEzPe`. Treat the hotkey as the
identity. UID labels are not stable across snapshots; in the live dashboard
snapshot I checked, that hotkey was UID 149, not UID 1.

## What was fixed

The patch makes `commit["tokens"]` the canonical token source and rejects
split-token payloads:

- `RolloutSubmission` now validates `tokens == commit["tokens"]`.
- `GrpoWindowBatcher` has a runtime guard returning `tokens_mismatch` if a
  constructed or mutated object bypasses Pydantic.
- Reward decoding uses `rollout.commit["tokens"]`.
- R2 archiving uses `r.commit["tokens"]`.
- GRPO training uses `rollout.commit["tokens"]`.
- Regression tests cover schema rejection, runtime rejection, archive behavior,
  and training's canonical token source.

Focused verification:

```text
python3 -m pytest \
  tests/unit/test_batch_submission_schema.py \
  tests/unit/test_grpo_window_batcher.py \
  tests/unit/test_archive_window_content.py \
  tests/unit/test_training_rollout_loss.py -q
```

Result after the `tokens_mismatch` enum addition: `90 passed, 1 warning`.

## Exploit mechanics

The attack relies on validator split-brain:

1. Miner generates or obtains an honest-looking long sequence.
2. Miner builds `commit["tokens"]` from that sequence and signs the GRAIL
   commit. This is the sequence the validator proves against the model.
3. Miner places a different short sequence in outer `rollout.tokens`.
4. The short sequence decodes to text like `Answer: \boxed{240}.`
5. Reward verification used to decode the outer tokens, so it compared the fake
   answer against `ground_truth` and accepted the claimed reward.
6. GRAIL/logprob/distribution checks used the honest inner tokens, so they
   passed cleanly.
7. The accepted submission entered the batch and the public archive showed only
   the fake outer tokens.

The sigma values in the evidence are exactly what an attacker would choose for
`M_ROLLOUTS = 8` and binary rewards:

```text
2 correct, 6 wrong: sqrt(0.25 * 0.75)  = 0.4330127
3 correct, 5 wrong: sqrt(0.375 * 0.625) = 0.4841229
4 correct, 4 wrong: sqrt(0.5 * 0.5)     = 0.5
```

Those values clear steady-state `SIGMA_MIN = 0.43` while minimizing the amount
of answer guessing needed.

## Evidence verified

I verified the code-level claim directly in the validator repo. The vulnerable
shape existed in `reliquary/protocol/submission.py`, `validator/service.py`, and
`validator/training.py` before this patch.

I also checked public R2 archives through:

- `https://www.reliqua.ai/api/r2/window/5710`
- `https://www.reliqua.ai/api/r2/window/<window_id>`

For hotkey `5DAvrWnM8MygaYq5dyC1bs8tP71Pd8d5QF5yRBN36ggkEzPe`, accessible
windows in `5710-5769` gave:

| Metric | Observed |
| --- | ---: |
| Accepted batch entries | 130 |
| Rollouts | 1040 |
| EOS last-token count `{151645, 151643}` | 0 |
| Last token `7810` count | 935 |
| Last token `7810` share | 89.904% |
| Reward-1 count | 510 |
| Reward-1 share | 49.038% |
| `len(tokens) < completion_length` | 1037 |
| Sigma modes | 0.433013: 70, 0.484123: 38, 0.5: 22 |

The reported total `147` entries / `1176` rollouts likely includes windows
`5770-5780`, but those endpoints returned `403`/`429` during my check. I did
not independently verify the final 17 entries because of that API limitation.
The accessible subset strongly supports the same conclusion.

Concrete example verified from public archive window 5710:

```text
prompt_idx: 40765
ground_truth: 240
sigma: 0.4330127018922193

rollouts:
  reward 0: \boxed{243}
  reward 0: \boxed{243}
  reward 1: \boxed{240}
  reward 0: \boxed{237}
  reward 0: \boxed{242}
  reward 1: \boxed{240}
  reward 0: \boxed{238}
  reward 0: \boxed{242}
```

That is exactly the `2/8` correct construction that lands at sigma `0.4330`.

## Important nuance

The report says the trainer was being trained on the fake completions. That is
possible for this class of bug, and the patch closes it, but the specific
observed payloads often had `len(outer tokens) < commit completion_length`.

Before this patch, `training.py` used outer tokens but old logprobs from the
inner commit. For the observed short-token variant, `_rollout_loss` would likely
hit a logprob length mismatch and skip those rollouts. So the strongest
confirmed impact is:

- scoring/batch-slot manipulation,
- public archive contamination,
- potential training contamination if a variant pads outer tokens to compatible
  lengths,
- trainer instability or skipped rollouts for the short-token variant.

Do not overclaim model degradation from the observed short-token payload until
W&B/training logs confirm nonzero processed rollouts for those windows.

## Current top-miner claim

The same fingerprint did not match the current rank-1 hotkey in the latest
snapshot I checked (`5Hga6BeZyjvswXar7aCKNGTPrWVgY8G1LzdFyEUSiGi4pnR6`).
Across windows `6204-6275`, that hotkey had:

- 1136 rollouts,
- 0% ending in token `7810`,
- 98.7% ending in token `151643`,
- no `len(tokens) < completion_length` symptom.

Recommendation: communicate this as a validator bug plus a hotkey-specific
forensic finding, not as "the current top miner cheated" unless later evidence
supports that exact identity.
