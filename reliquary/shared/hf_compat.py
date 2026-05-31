"""HuggingFace model compatibility helpers."""

from typing import Any

from transformers import PretrainedConfig, PreTrainedModel


def resolve_hidden_size(model: PreTrainedModel) -> int:
    """Return model hidden size across HF variants (Gemma/Qwen/etc.)."""
    cfg = getattr(model, "config", None)

    for attr in ("hidden_size", "d_model", "n_embd", "embed_dim", "hidden_dim"):
        try:
            if cfg is not None and hasattr(cfg, attr):
                val = getattr(cfg, attr)
                if isinstance(val, int) and val > 0:
                    return val
        except Exception:
            pass

    try:
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            val = text_cfg.hidden_size
            if isinstance(val, int) and val > 0:
                return val
    except Exception:
        pass

    try:
        if hasattr(model, "get_input_embeddings") and callable(model.get_input_embeddings):
            emb = model.get_input_embeddings()
            weight = getattr(emb, "weight", None)
            if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
                return int(weight.shape[1])
    except Exception:
        pass

    raise AttributeError("Could not determine model hidden size from config or embeddings")


def resolve_vocab_size(model_config: PretrainedConfig | Any) -> int | None:
    """Return vocab size if present in config (direct or nested)."""
    try:
        for attr in ("vocab_size", "n_vocab", "vocabulary_size"):
            if hasattr(model_config, attr):
                val = getattr(model_config, attr)
                if isinstance(val, int) and val > 0:
                    return val

        text_cfg = getattr(model_config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "vocab_size"):
            val = text_cfg.vocab_size
            if isinstance(val, int) and val > 0:
                return val
    except Exception:
        pass

    return None


def resolve_max_context_length(model_config: PretrainedConfig | Any) -> int:
    """Return best-effort max context length with common fallbacks."""
    candidates = (
        "max_position_embeddings",
        "max_seq_len",
        "n_positions",
        "seq_length",
        "max_sequence_length",
        "model_max_length",
    )

    for name in candidates:
        try:
            if hasattr(model_config, name):
                val = getattr(model_config, name)
                if isinstance(val, int) and val > 0:
                    return val
        except Exception:
            pass

    return 16384


def resolve_attn_implementation() -> str:
    """Return an attention backend that can load in the current environment.

    Honors ``GRAIL_ATTN_IMPL`` (default ``flash_attention_2``). When flash-attn
    is requested but not importable, falls back to ``sdpa`` so the miner can
    boot — mainnet GRAIL still requires ``flash_attention_2`` matching the
    validator stack.
    """
    import logging
    import os

    requested = os.environ.get("GRAIL_ATTN_IMPL", "flash_attention_2")
    if requested.startswith("flash_attention"):
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logging.getLogger(__name__).warning(
                "flash-attn is not installed but GRAIL_ATTN_IMPL=%r — "
                "falling back to sdpa. Install flash-attn for mainnet mining "
                "(GRAIL proofs are bit-sensitive to the attention kernel).",
                requested,
            )
            return "sdpa"
    return requested
