"""Generation backends — HF transformers (default) and vLLM (fast).

Both expose :meth:`generate(prompt_tokens, n_rollouts, *, max_new_tokens,
eos_set, tokenizer)` returning a list of ``{"tokens": [...], "prompt_length": N}``
dicts matching the miner's existing rollout format.

The protocol's GRAIL proof is computed by feeding the generated tokens through
the miner's *proof* model (``hf_model`` on cuda:1), not the generation engine.
So switching generation to vLLM does not affect bit-identicality with the
validator — only the token sequence is observed downstream, and vLLM/HF both
draw from valid model samples.

The :class:`HFGenerator` mirrors the original ``model.generate`` behaviour
exactly. The :class:`VLLMGenerator` plugs in vLLM's PagedAttention engine
for 3–5× higher throughput.

vLLM is an optional dependency: ``pip install vllm``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from reliquary.constants import (
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)

logger = logging.getLogger(__name__)


def _truncate_at_eos(tokens: list[int], eos_set: set[int]) -> list[int]:
    if not eos_set:
        return tokens
    for idx, tok in enumerate(tokens):
        if int(tok) in eos_set:
            return tokens[: idx + 1]
    return tokens


class HFGenerator:
    """Wraps a HF ``AutoModelForCausalLM`` to emit rollouts in the miner's format."""

    is_vllm: bool = False

    def __init__(self, model: Any, lock: threading.Lock, gpu: int) -> None:
        self.model = model
        self.lock = lock
        self.gpu = gpu

    def generate(
        self,
        prompt_tokens: list[int],
        n_rollouts: int,
        *,
        max_new_tokens: int,
        eos_set: set[int],
        tokenizer,
    ) -> list[dict]:
        import torch

        eos_for_generate = (
            sorted(eos_set) if len(eos_set) > 1
            else (next(iter(eos_set)) if eos_set else None)
        )

        with self.lock:
            gen_kwargs: dict = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": T_PROTO,
                "top_p": TOP_P_PROTO,
                "top_k": TOP_K_PROTO,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if eos_for_generate is not None:
                gen_kwargs["eos_token_id"] = eos_for_generate

            with torch.no_grad():
                input_tensor = torch.tensor(
                    [prompt_tokens] * n_rollouts,
                    device=getattr(self.model, "device", "cpu"),
                )
                outputs = self.model.generate(input_tensor, **gen_kwargs)

            prompt_length = len(prompt_tokens)
            rollouts = []
            for i in range(n_rollouts):
                seq = outputs[i].tolist()
                gen = _truncate_at_eos(seq[prompt_length:], eos_set)
                rollouts.append({
                    "tokens": prompt_tokens + gen,
                    "prompt_length": prompt_length,
                })
            return rollouts


class VLLMGenerator:
    """Generation via vLLM. ``llm`` is a ``vllm.LLM`` instance pinned to one GPU.

    vLLM internally batches and schedules; the engine's own lock is unneeded.
    A best-effort lock is still held for symmetry with HF (prevents weird
    interleaving if two prep threads call into it concurrently — vLLM handles
    it, but the lock keeps log ordering sane).
    """

    is_vllm: bool = True

    def __init__(self, llm: Any, gpu: int) -> None:
        self.llm = llm
        self.gpu = gpu
        self.lock = threading.Lock()

    def generate(
        self,
        prompt_tokens: list[int],
        n_rollouts: int,
        *,
        max_new_tokens: int,
        eos_set: set[int],
        tokenizer,
    ) -> list[dict]:
        """Single-prompt path — generate ``n_rollouts`` completions."""
        return self.generate_batch(
            [prompt_tokens], n_rollouts,
            max_new_tokens=max_new_tokens,
            eos_set=eos_set,
            tokenizer=tokenizer,
        )[0]

    def generate_batch(
        self,
        prompts_tokens: list[list[int]],
        n_per_prompt: int,
        *,
        max_new_tokens: int | list[int],
        eos_set: set[int],
        tokenizer,
    ) -> list[list[dict]]:
        """Batched multi-prompt generation. Returns ``len(prompts_tokens)``
        lists of rollouts, each with ``n_per_prompt`` entries (or fewer if
        a sequence finished short).

        Critical for throughput: vLLM's continuous batching schedules all
        ``len(prompts) * n_per_prompt`` sequences in parallel — total
        wall-clock is set by the slowest sequence, not the sum. Going from
        K=1 → K=4 prompts is ~8× the bundle yield for ~1.3× the time.

        ``max_new_tokens`` may be a single int (applied uniformly) OR a list
        with one entry per prompt. **Per-prompt is required for batches with
        mixed prompt lengths** — the validator's ``is_cap_truncation`` triggers
        only when ``prompt_length + completion_length >= MAX_NEW_TOKENS_PROTOCOL_CAP``;
        using one uniform value lets shorter-prompt rollouts stop below the
        protocol cap, making them "broken" (neither validly terminated nor
        cap-truncated) instead of cap-truncated. That manufactures rejections
        on bundles the validator would otherwise accept.
        """
        from vllm import SamplingParams, TokensPrompt

        stop_token_ids = list(eos_set) if eos_set else None

        if isinstance(max_new_tokens, list):
            if len(max_new_tokens) != len(prompts_tokens):
                raise ValueError(
                    f"max_new_tokens list length ({len(max_new_tokens)}) does "
                    f"not match prompts ({len(prompts_tokens)})"
                )
            sampling: SamplingParams | list[SamplingParams] = [
                SamplingParams(
                    n=n_per_prompt,
                    temperature=T_PROTO,
                    top_p=TOP_P_PROTO,
                    top_k=TOP_K_PROTO if TOP_K_PROTO > 0 else -1,
                    max_tokens=mn,
                    stop_token_ids=stop_token_ids,
                    include_stop_str_in_output=True,
                )
                for mn in max_new_tokens
            ]
        else:
            sampling = SamplingParams(
                n=n_per_prompt,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=TOP_K_PROTO if TOP_K_PROTO > 0 else -1,
                max_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
                include_stop_str_in_output=True,
            )

        prompts = [
            TokensPrompt(prompt_token_ids=list(pt)) for pt in prompts_tokens
        ]

        with self.lock:
            req_outputs = self.llm.generate(
                prompts, sampling, use_tqdm=False,
            )

        # vLLM may reorder outputs; req_output.request_id has an integer
        # parsable from the position, but the safer path is to match by
        # iteration order — vLLM preserves input order in the V1 engine.
        result: list[list[dict]] = []
        for prompt_idx, req_output in enumerate(req_outputs):
            prompt_tokens = prompts_tokens[prompt_idx]
            prompt_length = len(prompt_tokens)
            rollouts: list[dict] = []
            for sample in req_output.outputs:
                comp_tokens = list(sample.token_ids)
                comp_tokens = _truncate_at_eos(comp_tokens, eos_set)
                rollouts.append({
                    "tokens": prompt_tokens + comp_tokens,
                    "prompt_length": prompt_length,
                })
            result.append(rollouts)
        return result


def apply_transformers_compat_patches() -> None:
    """Idempotent compatibility patches for transformers ↔ vLLM/checkpoint issues.

    Safe to call multiple times and from any code path that loads transformers
    or vLLM. Currently applies:

    - ``_patch_extra_special_tokens_for_vllm`` — handles checkpoints where
      ``tokenizer_config.json`` stores ``extra_special_tokens`` as a list.
    - ``_patch_autoconfig_register_for_vllm`` — makes ``AutoConfig.register``
      idempotent so vLLM-side re-registrations are no-ops.
    """
    _patch_extra_special_tokens_for_vllm()
    _patch_autoconfig_register_for_vllm()


def _patch_extra_special_tokens_for_vllm() -> None:
    """Tolerate ``extra_special_tokens`` being a list in tokenizer_config.json.

    Some checkpoints (e.g. R0mAI's reliquary-sn-v23) store ``extra_special_tokens``
    as a JSON list (legacy format). transformers ≥ 4.51 assumes it's a dict and
    calls ``.keys()`` on it inside ``_set_model_specific_special_tokens``,
    raising ``AttributeError: 'list' object has no attribute 'keys'`` during
    ``Qwen2TokenizerFast.__init__``. vLLM's get_tokenizer hits this path; the
    HF miner's ``AutoTokenizer.from_pretrained`` happens to avoid it.

    Patch: when ``special_tokens`` arrives as a list, treat it as an empty dict.
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return
    original = PreTrainedTokenizerBase._set_model_specific_special_tokens
    if getattr(original, "_reliquary_patched", False):
        return

    def patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {}
        return original(self, special_tokens)

    patched._reliquary_patched = True  # type: ignore[attr-defined]
    PreTrainedTokenizerBase._set_model_specific_special_tokens = patched  # type: ignore[assignment]


def _patch_autoconfig_register_for_vllm() -> None:
    """Make ``AutoConfig.register`` idempotent so vLLM can coexist with newer transformers.

    vLLM ≤ ~0.10 registers configs like ``aimv2`` with ``exist_ok=False``.
    transformers ≥ 4.51 ships those types natively, so vLLM's import-time
    registration raises ``ValueError: 'aimv2' is already used by a
    Transformers config``. We swap the underlying mapping's ``register`` for
    one that forces ``exist_ok=True`` — newer transformers wins on tie,
    which is what we want anyway.
    """
    try:
        from transformers.models.auto.configuration_auto import _LazyConfigMapping
    except Exception:
        return
    original = _LazyConfigMapping.register
    if getattr(original, "_reliquary_patched", False):
        return

    def patched(self, key, value, exist_ok=False):
        return original(self, key, value, exist_ok=True)

    patched._reliquary_patched = True  # type: ignore[attr-defined]
    _LazyConfigMapping.register = patched  # type: ignore[assignment]


def _list_engine_core_pids() -> list[int]:
    """Find all live ``EngineCore_0`` subprocesses spawned by vLLM.

    vLLM V1's engine runs in a child process named ``EngineCore_<rank>``. The
    parent's API calls (``llm.shutdown()`` etc.) send a soft shutdown signal
    but some versions don't actually wait for the child to exit. We use this
    list to SIGTERM/SIGKILL stragglers directly.
    """
    import os
    pids: list[int] = []
    parent = os.getpid()
    try:
        import psutil  # type: ignore
    except ImportError:
        return pids
    try:
        for proc in psutil.process_iter(["pid", "name", "ppid"]):
            try:
                name = proc.info.get("name") or ""
                if "EngineCore" not in name:
                    continue
                # Only kill descendants of OUR process, not someone else's
                # vLLM elsewhere on the box.
                ppid = proc.info.get("ppid") or 0
                if ppid != parent and not _is_descendant(proc, parent):
                    continue
                pids.append(int(proc.info["pid"]))
            except Exception:
                continue
    except Exception:
        return pids
    return pids


def _is_descendant(proc, ancestor_pid: int) -> bool:
    try:
        for p in proc.parents():
            if p.pid == ancestor_pid:
                return True
    except Exception:
        return False
    return False


def _gpu_free_bytes(gpu_idx: int) -> int:
    try:
        import torch
        free, _total = torch.cuda.mem_get_info(gpu_idx)
        return int(free)
    except Exception:
        return -1


def shutdown_vllm_generator(
    gen, *, gpu: int | None = None, inflight_timeout: float = 180.0,
) -> None:
    """Release GPU memory held by a vLLM generator before rebuild.

    vLLM ≥ 0.10 spawns an ``EngineCore_<rank>`` subprocess that owns the KV
    cache + weights. Cleaning up requires four orthogonal actions, all of
    which we now do:

      0. **Wait for any in-flight ``generate()`` call to complete** — the
         ``VLLMGenerator.lock`` is held during generate, so we acquire it
         (with timeout) before killing the subprocess. Without this step,
         a prep cycle running on a worker thread can be killed mid-call and
         remain permanently blocked inside the dead subprocess, freezing the
         miner even though the new vLLM rebuild succeeded.
      1. **Invoke every soft shutdown path vLLM exposes** (not break on first)
         — different versions wire the shutdown to different attribute paths
         and some paths return without actually killing the worker.
      2. **Find the ``EngineCore`` subprocess directly and SIGTERM it**, then
         SIGKILL if it hasn't exited within 10s. Skipping this is the most
         common cause of "rebuild hangs at 0% GPU util" — the parent thinks
         the child is dead, the child is still holding the CUDA context.
      3. **Actively poll GPU free memory** until it crosses a threshold that
         indicates the worker has fully released its allocations, with a
         60s cap. We log free memory at each poll so a hang is visible.
    """
    import gc
    import os
    import signal
    import time

    try:
        import torch  # noqa: F401
    except Exception:
        torch = None  # type: ignore[assignment]

    if gen is None:
        return

    # Step 0: drain in-flight generate() calls. A prep cycle started just
    # before checkpoint pull may still be running on a worker thread; tearing
    # down the subprocess underneath it leaves that thread permanently stuck
    # waiting for IPC results that will never arrive.
    gen_lock = getattr(gen, "lock", None)
    if gen_lock is not None:
        logger.info(
            "vLLM shutdown: waiting up to %.0fs for in-flight generate() "
            "to complete",
            inflight_timeout,
        )
        wait_t0 = time.monotonic()
        acquired = gen_lock.acquire(timeout=inflight_timeout)
        wait_dt = time.monotonic() - wait_t0
        if acquired:
            logger.info(
                "vLLM shutdown: in-flight drained after %.1fs", wait_dt,
            )
            # Release so the soon-to-be-destroyed lock doesn't deadlock
            # any other thread that still tries to take it.
            try:
                gen_lock.release()
            except Exception:
                pass
        else:
            logger.warning(
                "vLLM shutdown: in-flight generate did NOT complete in %.0fs "
                "— forcing shutdown anyway. Worker thread holding the call "
                "will hang until the next miner restart.",
                inflight_timeout,
            )

    pids_before = _list_engine_core_pids()
    if gpu is not None and torch is not None:
        free_before = _gpu_free_bytes(gpu)
        logger.info(
            "vLLM shutdown: %d EngineCore subprocess(es) running, "
            "cuda:%d free=%.1f GiB",
            len(pids_before), gpu, free_before / (1024 ** 3),
        )
    else:
        logger.info(
            "vLLM shutdown: %d EngineCore subprocess(es) running",
            len(pids_before),
        )

    llm = getattr(gen, "llm", None)
    if llm is not None:
        # Run ALL shutdown paths in sequence — different vLLM versions
        # implement different combinations. Don't break on the first success
        # because a "success" might be a no-op stub.
        for label, fn in (
            ("llm.shutdown", lambda l: l.shutdown()),
            ("llm.llm_engine.shutdown", lambda l: l.llm_engine.shutdown()),
            ("engine_core.shutdown",
                lambda l: l.llm_engine.engine_core.shutdown()),
            ("engine_core.close",
                lambda l: l.llm_engine.engine_core.close()),
        ):
            try:
                fn(llm)
                logger.debug("vLLM shutdown: %s succeeded", label)
            except Exception as e:
                logger.debug("vLLM shutdown: %s failed (%s)", label, type(e).__name__)
        try:
            gen.llm = None
        except Exception:
            pass

    # Forcibly reap any EngineCore subprocess that survived the soft shutdown.
    if pids_before:
        for pid in pids_before:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except Exception:
                pass
        # Give SIGTERM a chance to land cleanly.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            alive = _list_engine_core_pids()
            if not alive:
                break
            time.sleep(0.5)
        # SIGKILL any holdouts.
        for pid in _list_engine_core_pids():
            logger.warning(
                "vLLM shutdown: EngineCore pid=%d did not exit on SIGTERM, "
                "sending SIGKILL", pid,
            )
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        # Wait briefly for kernel-level cleanup.
        time.sleep(1.0)

    gc.collect()
    if torch is not None:
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Poll GPU memory to confirm release. The CUDA context teardown is
    # asynchronous and can lag the subprocess exit by several seconds.
    if gpu is not None and torch is not None:
        baseline = _gpu_free_bytes(gpu)
        deadline = time.monotonic() + 60.0
        last_log = time.monotonic()
        while time.monotonic() < deadline:
            free = _gpu_free_bytes(gpu)
            # 90% of total is the threshold for "essentially empty" on a
            # B200 (most of the 141 GiB freed). Below that we keep waiting.
            try:
                _free_now, total = torch.cuda.mem_get_info(gpu)
                if free / total >= 0.90:
                    break
            except Exception:
                break
            if time.monotonic() - last_log > 5.0:
                logger.info(
                    "vLLM shutdown: cuda:%d free=%.1f GiB / %.1f GiB "
                    "(%.0f%%) — waiting for full release",
                    gpu, free / (1024 ** 3), total / (1024 ** 3),
                    100 * free / total,
                )
                last_log = time.monotonic()
            time.sleep(0.5)
        free_after = _gpu_free_bytes(gpu)
        logger.info(
            "vLLM shutdown complete: cuda:%d free=%.1f GiB (was %.1f GiB)",
            gpu, free_after / (1024 ** 3), baseline / (1024 ** 3),
        )
    else:
        # No GPU index given — fall back to a fixed wait.
        time.sleep(5)

    gc.collect()
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def build_vllm_generator(
    checkpoint_path: str,
    *,
    gpu: int = 0,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
) -> VLLMGenerator:
    """Construct a vLLM engine pinned to ``cuda:gpu``.

    Set ``CUDA_VISIBLE_DEVICES`` so vLLM only sees the target GPU — vLLM's
    own ``device`` argument is unreliable across versions.

    ``max_model_len`` defaults to protocol cap + generation-prefix overhead
    so augmented/chat prompts can still emit cap-length canonical completions.
    """
    if max_model_len is None:
        from reliquary.shared.prompt_augment import default_max_model_len
        max_model_len = default_max_model_len()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # On Blackwell (B300, SM_100) vLLM 0.10.1.1 auto-selects flashinfer's
    # TRTLLM attention kernels, which have a Python↔C++ type-binding bug:
    # ``trtllm_paged_attention_context`` expects ``int`` for arg #9 but the
    # binding passes an ``ffi.Tensor``. Force vLLM to use its FlashAttention
    # backend instead — bypasses flashinfer/TRTLLM entirely.
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    # On Blackwell Ultra (B300 Ultra, sm_103a) and any future GPU whose arch
    # the installed CUDA toolkit's nvcc doesn't yet support, the FlashInfer
    # top-k/top-p sampler tries to JIT-compile a CUDA kernel via
    # ``-gencode=arch=compute_103a,code=sm_103a`` and fails with
    # ``nvcc fatal: Unsupported gpu architecture``. Disabling the FlashInfer
    # sampler falls back to vLLM's PyTorch-native top-k/top-p path which has
    # no JIT step. Override with VLLM_USE_FLASHINFER_SAMPLER=1 on hardware
    # where the JIT works (B200/sm_100 with a recent CUDA toolkit) to get
    # the faster path.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _patch_autoconfig_register_for_vllm()
    _patch_extra_special_tokens_for_vllm()
    from vllm import LLM

    # torch.compile / Inductor uses Triton+LLVM to JIT CUDA kernels. On
    # bleeding-edge SMs (sm_103a / Blackwell Ultra etc.) the LLVM ships
    # without a code generator for the target, causing a hard
    # ``LLVM ERROR: Cannot select intrinsic`` crash during vLLM init —
    # which kills the process before Python's except handler can catch it.
    # Default to enforce_eager=True so the engine actually starts; override
    # with VLLM_ENFORCE_EAGER=0 on known-good hardware (B200/SM_100) where
    # torch.compile gives ~30-50% throughput.
    env_eager = os.environ.get("VLLM_ENFORCE_EAGER", "1").strip().lower()
    enforce_eager = env_eager not in ("0", "false", "no", "off", "")

    logger.info(
        "vLLM: loading %s on cuda:%d (max_model_len=%d, util=%.2f, dtype=%s, "
        "enforce_eager=%s)",
        checkpoint_path, gpu, max_model_len, gpu_memory_utilization, dtype,
        enforce_eager,
    )
    llm = LLM(
        model=checkpoint_path,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    return VLLMGenerator(llm, gpu=gpu)
