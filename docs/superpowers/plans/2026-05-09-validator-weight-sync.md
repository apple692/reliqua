# Validator Weight Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align all validators of netuid 81 on the subnet epoch boundary so they submit weights inside the same ~20-block window each epoch and converge on identical weights.

**Architecture:** Replace `WeightOnlyValidator`'s per-validator block counter (`_last_submit_block` + `WEIGHT_SUBMISSION_INTERVAL`) with an epoch-anchored gate driven by `subtensor.blocks_until_next_epoch(netuid)`. One submit per epoch, fired when the chain reports we're inside the lead window. Bootstrap path covers fresh starts.

**Tech Stack:** bittensor SDK (`AsyncSubtensor.blocks_until_next_epoch`), Python asyncio, pytest.

**Spec:** `docs/superpowers/specs/2026-05-09-validator-weight-sync-design.md`

---

## File Structure

- **`reliquary/infrastructure/chain.py`** — add async wrapper `blocks_until_next_epoch(subtensor, netuid)` mirroring the existing wrappers (timeout + asyncio.wait_for).
- **`reliquary/constants.py`** — remove `WEIGHT_SUBMISSION_INTERVAL = 360`; add `EPOCH_SUBMIT_LEAD_BLOCKS = 20`. Replace the comment block above the EMA constants accordingly.
- **`reliquary/validator/weight_only.py`** — replace block-interval gate in `run()` with epoch-anchored gate. State changes from `_last_submit_block: int` to `_last_submit_epoch: int | None`. Hardcode `ROLLING_WINDOWS_HISTORY = 72` locally (no longer derived from `WEIGHT_SUBMISSION_INTERVAL`).
- **`tests/unit/test_weight_only_validator.py`** — rewrite the loop test to mock `blocks_until_next_epoch`. Add: bootstrap submit, in-window submit, out-of-window skip, repeat-epoch skip, next-epoch submit.
- **`tests/unit/test_state_machine.py`** — delete `test_weight_submission_gate_is_block_based` (asserts arithmetic on `_last_weight_block` which no longer exists in `service.py`; pure dead code).

---

## Task 1: Chain wrapper for blocks_until_next_epoch

**Files:**
- Modify: `reliquary/infrastructure/chain.py` (add function after `get_current_block`)
- Test: `tests/unit/test_chain_wrappers.py` (new file — none exists today)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_chain_wrappers.py`:

```python
"""Smoke tests for reliquary.infrastructure.chain async wrappers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_blocks_until_next_epoch_wraps_subtensor():
    """Wrapper delegates to subtensor.blocks_until_next_epoch under wait_for."""
    from reliquary.infrastructure import chain

    fake_sub = MagicMock()
    fake_sub.blocks_until_next_epoch = MagicMock(return_value=42)

    result = await chain.blocks_until_next_epoch(fake_sub, netuid=81)
    assert result == 42
    fake_sub.blocks_until_next_epoch.assert_called_once_with(81)


@pytest.mark.asyncio
async def test_blocks_until_next_epoch_timeout():
    """A hanging subtensor call surfaces as TimeoutError, not silent hang."""
    from reliquary.infrastructure import chain

    fake_sub = MagicMock()

    def _hang(*_a, **_kw):
        import time
        time.sleep(60)  # would block forever without wait_for
    fake_sub.blocks_until_next_epoch = _hang

    # Patch the timeout constant down to a tenth of a second for the test.
    original = chain.CHAIN_READ_TIMEOUT
    chain.CHAIN_READ_TIMEOUT = 0.1
    try:
        with pytest.raises(asyncio.TimeoutError):
            await chain.blocks_until_next_epoch(fake_sub, netuid=81)
    finally:
        chain.CHAIN_READ_TIMEOUT = original
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_chain_wrappers.py -v
```

Expected: FAIL with `AttributeError: module 'reliquary.infrastructure.chain' has no attribute 'blocks_until_next_epoch'`.

- [ ] **Step 3: Implement the wrapper**

Add to `reliquary/infrastructure/chain.py` immediately after `get_current_block`:

```python
async def blocks_until_next_epoch(subtensor, netuid: int) -> int | None:
    """Blocks remaining in the current epoch for ``netuid``.

    All validators of the same netuid see the same boundary because the
    underlying SDK formula is purely a function of (netuid, current_block,
    tempo). Used by ``WeightOnlyValidator.run()`` to sync weight submissions.
    """
    return await asyncio.wait_for(
        asyncio.to_thread(subtensor.blocks_until_next_epoch, netuid),
        timeout=CHAIN_READ_TIMEOUT,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_chain_wrappers.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/infrastructure/chain.py tests/unit/test_chain_wrappers.py
git commit -m "feat(chain): add blocks_until_next_epoch async wrapper

Wraps subtensor.blocks_until_next_epoch under the standard CHAIN_READ_TIMEOUT.
Used by WeightOnlyValidator to anchor weight submissions to the subnet epoch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Constants — replace WEIGHT_SUBMISSION_INTERVAL with EPOCH_SUBMIT_LEAD_BLOCKS

**Files:**
- Modify: `reliquary/constants.py` (line 120 + comment block at line 60-62)

- [ ] **Step 1: Read current constants context**

```bash
sed -n '55,65p;115,125p' reliquary/constants.py
```

Note the comment at line 60-62 references `WEIGHT_SUBMISSION_INTERVAL=360` for explaining `ROLLING_WINDOWS=72`. Both will be reworded.

- [ ] **Step 2: Apply the replacement**

In `reliquary/constants.py`:

Replace the line:
```python
WEIGHT_SUBMISSION_INTERVAL = 360  # Blocks between weight submissions
```

with:
```python
# Submit weights when blocks_until_next_epoch <= this value. Tuned so all
# validators of a netuid land in the same ~20-block window (≈4 min on
# 12s/block) and read near-identical R2 archive snapshots, then converge
# to identical weights via the deterministic EMA replay.
EPOCH_SUBMIT_LEAD_BLOCKS = 20
```

The reference comment at line 60-62 ("Given a typical tempo of 360 blocks and `WEIGHT_SUBMISSION_INTERVAL=360`, that yields `ROLLING_WINDOWS=72`...") — open the file and shorten the comment to drop the `WEIGHT_SUBMISSION_INTERVAL` reference. The exact rewording: change "and `WEIGHT_SUBMISSION_INTERVAL=360`" to nothing (the tempo alone suffices for the 72-window math).

The reference at line 219 (EMA_ALPHA comment "with N=ROLLING_WINDOWS=72") stays unchanged.

- [ ] **Step 3: Verify no orphan imports**

```bash
grep -rn "WEIGHT_SUBMISSION_INTERVAL" reliquary tests --include='*.py'
```

Expected output: matches only in `tests/unit/test_weight_only_validator.py` and `tests/unit/test_state_machine.py` and `reliquary/validator/weight_only.py` — these are rewritten / deleted in later tasks. Do NOT yet remove the import in `weight_only.py`; that happens in Task 3.

- [ ] **Step 4: Run constants smoke**

```bash
python -c "from reliquary.constants import EPOCH_SUBMIT_LEAD_BLOCKS; assert EPOCH_SUBMIT_LEAD_BLOCKS == 20"
python -c "from reliquary.constants import WEIGHT_SUBMISSION_INTERVAL" 2>&1 | grep -q "ImportError" && echo OK
```

Expected: first command silent (no error), second prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add reliquary/constants.py
git commit -m "refactor(constants): replace WEIGHT_SUBMISSION_INTERVAL with EPOCH_SUBMIT_LEAD_BLOCKS

Weight submissions move from per-validator block counter to epoch-anchored
gate (next commit). The new constant defines the lead window before the
epoch boundary in which all validators submit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: WeightOnlyValidator — epoch-anchored submit loop

**Files:**
- Modify: `reliquary/validator/weight_only.py` (imports, class state, `run()`)

- [ ] **Step 1: Read the current weight_only.py to understand the loop**

```bash
sed -n '1,80p' reliquary/validator/weight_only.py
```

You're replacing the block-interval gate in `run()` (lines 53-61 area) and the `_last_submit_block` state with an epoch-anchored gate.

- [ ] **Step 2: Rewrite the imports + module-level constants**

Replace lines 15-27 of `reliquary/validator/weight_only.py` with:

```python
from reliquary.constants import (
    B_BATCH,
    EMA_ALPHA,
    EPOCH_SUBMIT_LEAD_BLOCKS,
    POLL_INTERVAL_SECONDS,
    UID_BURN,
    WINDOW_LENGTH,
)
from reliquary.infrastructure import chain, storage

# EMA history depth — number of past windows replayed to compute miner
# scores. Independent of the on-chain tempo: 72 windows ≈ ~6 hours on a
# typical cadence, enough to smooth out per-window noise.
ROLLING_WINDOWS_HISTORY = 72
```

(Keep the rest of the imports above this block — `asyncio`, `logging`, `defaultdict`, etc.)

- [ ] **Step 3: Update class state initialiser**

Replace `self._last_submit_block: int = 0` in `__init__` with:

```python
self._last_submit_epoch: int | None = None
```

- [ ] **Step 4: Rewrite the run() loop body**

Replace the existing `while True:` body (the part with the block-interval gate) so the structure becomes:

```python
async def run(self, subtensor) -> None:
    logger.info(
        "Weight-only validator started (netuid=%d, hotkey=%s)",
        self.netuid, self.wallet.hotkey.ss58_address,
    )
    while True:
        try:
            blocks_until = await chain.blocks_until_next_epoch(
                subtensor, self.netuid,
            )
            if blocks_until is None:
                logger.warning("blocks_until_next_epoch returned None — retrying")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue
            current_block = await chain.get_current_block(subtensor)
            # Stable per-epoch identifier: the absolute block number of the
            # next epoch boundary stays constant for every poll inside the
            # current epoch (current_block + blocks_until is invariant).
            current_epoch_id = current_block + blocks_until

            if self._last_submit_epoch == current_epoch_id:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            bootstrap = self._last_submit_epoch is None
            in_lead_window = blocks_until <= EPOCH_SUBMIT_LEAD_BLOCKS
            if not bootstrap and not in_lead_window:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            if await self.submit_once(subtensor):
                self._last_submit_epoch = current_epoch_id
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            logger.warning(
                "substrate call timed out — recreating subtensor connection",
            )
            try:
                subtensor = await chain.get_subtensor()
                logger.info("substrate reconnected")
            except Exception:
                logger.exception("substrate reconnect failed; will retry next poll")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except Exception:
            logger.exception("weight-only loop iteration failed")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
```

(Keep all the rest of `run()` outside the loop unchanged — the existing reconnect block in the original file was inside `except asyncio.TimeoutError`; preserve that structure.)

- [ ] **Step 5: Update submit_once to use new constant**

Find the line `n=ROLLING_WINDOWS * 3,` in `submit_once()` and replace with:

```python
n=ROLLING_WINDOWS_HISTORY * 3,
```

- [ ] **Step 6: Update the docstring of WeightOnlyValidator**

Replace the docstring (currently mentions "Every `WEIGHT_SUBMISSION_INTERVAL` blocks") with:

```python
"""Lightweight validator that only sets weights.

Each subnet epoch (anchored on subtensor.blocks_until_next_epoch):
  1. Read last K archives from R2
  2. Replay EMA update
  3. Submit weights on-chain via chain.set_weights

All validators of a netuid hit the same epoch boundary, so they submit
inside a shared ~EPOCH_SUBMIT_LEAD_BLOCKS-block window and converge to
identical weights from the deterministic EMA replay.

A freshly-booted validator submits immediately on its first poll, then
joins the synced cadence from the next epoch onward.

No local state: every submit recomputes from scratch.
"""
```

- [ ] **Step 7: Sanity-check the file imports + parses**

```bash
python -c "from reliquary.validator import weight_only; print('imports OK')"
```

Expected: `imports OK`. If you see `ImportError`, the constants or chain wrapper from Tasks 1-2 isn't reachable — fix before proceeding.

- [ ] **Step 8: Commit**

```bash
git add reliquary/validator/weight_only.py
git commit -m "fix(validator): anchor weight submissions to subnet epoch

WeightOnlyValidator.run() now reads blocks_until_next_epoch(netuid) every
poll and submits when blocks_until <= EPOCH_SUBMIT_LEAD_BLOCKS, at most
once per epoch (tracked by _last_submit_epoch). Fresh starts submit on
the first poll regardless of position in the epoch, then converge.

All validators of a netuid see the same boundary → submit inside the
same ~20-block window → read near-identical R2 archives → converge to
identical weights via the deterministic EMA replay. Fixes the divergent-
weights symptom observed in production.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Tests — rewrite weight-only loop tests + delete dead state-machine test

**Files:**
- Modify: `tests/unit/test_weight_only_validator.py` (rewrite the `test_run_loop_submits_after_interval` test, add 4 more)
- Modify: `tests/unit/test_state_machine.py` (delete `test_weight_submission_gate_is_block_based`)

- [ ] **Step 1: Delete the dead state-machine test**

Open `tests/unit/test_state_machine.py`, locate `test_weight_submission_gate_is_block_based` (around line 268-290), and delete the entire function (including its docstring). The test asserts arithmetic on `svc._last_weight_block`, an attribute that no longer exists in `service.py` — it has zero coverage value today.

Verify it's gone:

```bash
grep -n "test_weight_submission_gate_is_block_based\|_last_weight_block" tests/unit/test_state_machine.py
```

Expected: no output.

- [ ] **Step 2: Replace the obsolete loop test**

In `tests/unit/test_weight_only_validator.py`, locate `test_run_loop_submits_after_interval` (lines 99-147). Replace it entirely with the five new tests below (replacing both the old test and its imports of `WEIGHT_SUBMISSION_INTERVAL`):

```python
async def _run_one_iteration(wov, subtensor):
    """Helper: run wov.run() until it completes one poll, then cancel."""
    import asyncio
    task = asyncio.create_task(wov.run(subtensor))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _patch_chain_and_storage(blocks_until: int, current_block: int = 1_000_000):
    """Patch the four module-level chain/storage entry points
    weight_only.run() touches. Returns (patches, captured_calls)."""
    from unittest.mock import AsyncMock
    import reliquary.validator.weight_only as wov_mod

    captured = {"submit_calls": 0}
    originals = {
        "blocks_until_next_epoch": wov_mod.chain.blocks_until_next_epoch,
        "get_current_block": wov_mod.chain.get_current_block,
        "list_all_window_keys": wov_mod.storage.list_all_window_keys,
        "list_recent_datasets": wov_mod.storage.list_recent_datasets,
    }
    wov_mod.chain.blocks_until_next_epoch = AsyncMock(return_value=blocks_until)
    wov_mod.chain.get_current_block = AsyncMock(return_value=current_block)
    wov_mod.storage.list_all_window_keys = AsyncMock(return_value=[1, 2, 3])
    wov_mod.storage.list_recent_datasets = AsyncMock(return_value=[
        _archive(1, ["alice"]),
        _archive(2, ["alice"]),
        _archive(3, ["bob"]),
    ])
    return originals, captured


def _restore(originals):
    import reliquary.validator.weight_only as wov_mod
    wov_mod.chain.blocks_until_next_epoch = originals["blocks_until_next_epoch"]
    wov_mod.chain.get_current_block = originals["get_current_block"]
    wov_mod.storage.list_all_window_keys = originals["list_all_window_keys"]
    wov_mod.storage.list_recent_datasets = originals["list_recent_datasets"]


async def _wire_submit_counter(wov, captured):
    async def _fake_submit(subtensor, miner_weights, burn_weight):
        captured["submit_calls"] += 1
        return True
    wov._submit_weights = _fake_submit


@pytest.mark.asyncio
async def test_bootstrap_submits_immediately_regardless_of_lead_window():
    """A freshly-booted validator submits on its first poll even if we're
    far from the epoch boundary."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    assert wov._last_submit_epoch is None
    # Far from boundary (200 blocks remain → outside any reasonable lead).
    originals, captured = _patch_chain_and_storage(blocks_until=200)
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov, MagicMock())
    finally:
        _restore(originals)
    assert captured["submit_calls"] == 1
    assert wov._last_submit_epoch == 1_000_200


@pytest.mark.asyncio
async def test_in_lead_window_submits():
    """Inside [tempo - EPOCH_SUBMIT_LEAD_BLOCKS, tempo - 1] → submit."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EPOCH_SUBMIT_LEAD_BLOCKS
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    wov._last_submit_epoch = 999_000  # earlier epoch — not the current one
    originals, captured = _patch_chain_and_storage(
        blocks_until=EPOCH_SUBMIT_LEAD_BLOCKS,
    )
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov, MagicMock())
    finally:
        _restore(originals)
    assert captured["submit_calls"] == 1


@pytest.mark.asyncio
async def test_outside_lead_window_skips():
    """Far from boundary AND already submitted before → no submit."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EPOCH_SUBMIT_LEAD_BLOCKS
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    wov._last_submit_epoch = 999_000  # earlier epoch
    originals, captured = _patch_chain_and_storage(
        blocks_until=EPOCH_SUBMIT_LEAD_BLOCKS + 50,  # well outside
    )
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov, MagicMock())
    finally:
        _restore(originals)
    assert captured["submit_calls"] == 0


@pytest.mark.asyncio
async def test_repeat_poll_in_same_epoch_skips():
    """Once submitted in epoch E, a second poll in E does not re-submit."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    wov._last_submit_epoch = 1_000_005  # = current_block + blocks_until
    originals, captured = _patch_chain_and_storage(
        blocks_until=5, current_block=1_000_000,
    )
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov, MagicMock())
    finally:
        _restore(originals)
    assert captured["submit_calls"] == 0


@pytest.mark.asyncio
async def test_next_epoch_submits_again():
    """After crossing the boundary, _last_submit_epoch != current_epoch_id
    and we're inside the lead window → submit fires again."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import EPOCH_SUBMIT_LEAD_BLOCKS
    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    wov._last_submit_epoch = 1_000_000  # the prior epoch end
    originals, captured = _patch_chain_and_storage(
        blocks_until=EPOCH_SUBMIT_LEAD_BLOCKS,
        current_block=1_000_001,  # one block past the prior boundary
    )
    await _wire_submit_counter(wov, captured)
    try:
        await _run_one_iteration(wov, MagicMock())
    finally:
        _restore(originals)
    assert captured["submit_calls"] == 1
    assert wov._last_submit_epoch == 1_000_001 + EPOCH_SUBMIT_LEAD_BLOCKS
```

- [ ] **Step 3: Run weight-only validator tests**

```bash
pytest tests/unit/test_weight_only_validator.py -v
```

Expected: 4 existing EMA tests + 5 new loop tests = 9 PASS.

- [ ] **Step 4: Run state-machine tests (verify deletion didn't break anything)**

```bash
pytest tests/unit/test_state_machine.py -v
```

Expected: all remaining tests PASS, no `test_weight_submission_gate_is_block_based` in output.

- [ ] **Step 5: Run full unit test suite**

```bash
pytest tests/unit/ -v 2>&1 | tail -20
```

Expected: all PASS, no test imports `WEIGHT_SUBMISSION_INTERVAL` (would crash on `ImportError`).

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_weight_only_validator.py tests/unit/test_state_machine.py
git commit -m "test(validator): cover epoch-anchored weight submission

5 new tests for the epoch-aligned WeightOnlyValidator loop: bootstrap
submit, in-lead-window submit, out-of-window skip, repeat-epoch skip,
next-epoch submit.

Drop test_weight_submission_gate_is_block_based — the asserted attribute
(_last_weight_block) no longer exists in service.py; the test was dead.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Final verification

**Files:** none modified — verification only.

- [ ] **Step 1: Full test suite**

```bash
pytest tests/ -q 2>&1 | tail -10
```

Expected: 0 failures.

- [ ] **Step 2: Confirm WEIGHT_SUBMISSION_INTERVAL is fully removed**

```bash
grep -rn "WEIGHT_SUBMISSION_INTERVAL" reliquary tests --include='*.py'
```

Expected: no output.

- [ ] **Step 3: Confirm new constant is wired**

```bash
python -c "
from reliquary.constants import EPOCH_SUBMIT_LEAD_BLOCKS
from reliquary.infrastructure.chain import blocks_until_next_epoch
from reliquary.validator.weight_only import WeightOnlyValidator
assert EPOCH_SUBMIT_LEAD_BLOCKS == 20
assert callable(blocks_until_next_epoch)
print('wiring OK')
"
```

Expected: `wiring OK`.

- [ ] **Step 4: Visually inspect git log on branch**

```bash
git log --oneline origin/main..HEAD
```

Expected: 4 new commits on top of the existing branch state — chain wrapper, constants, validator refactor, tests.
