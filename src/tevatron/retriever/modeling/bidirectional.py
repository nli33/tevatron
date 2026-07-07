"""Turn a causal (decoder-only) backbone into a bidirectional encoder.

This is the LLM2Vec ("LLMs Are Secretly Powerful Text Encoders") trick: the
pretrained weights are left untouched, but the causal attention mask is removed
so every token attends to the whole sequence — which is what you want when the
model is used to produce a single text embedding rather than to generate.

Two changes are both required, because modern transformers optimizes the mask
away when there is no padding and then relies on the attention module's
``is_causal`` flag:

  1. ``is_causal = False`` on every attention submodule, so the SDPA / flash
     paths do not re-impose an implicit causal mask when the explicit mask is
     ``None``.
  2. Swap the modeling module's ``create_causal_mask`` for
     ``create_bidirectional_mask``, so the 4D mask handed to the layers only
     blocks padding, never future positions.

This works for any modern-transformers decoder that builds its mask via
``transformers.masking_utils.create_causal_mask`` (Llama, Qwen, Mistral,
NanoChat, ...). Backbones that do not follow that convention raise a clear error
rather than silently staying causal.
"""

import sys
import logging

logger = logging.getLogger(__name__)


def _bidirectional_mask_compat(create_bidirectional_mask):
    """Adapt create_bidirectional_mask to the create_causal_mask call sites.

    ``create_causal_mask`` accepts ``position_ids`` / ``cache_position`` kwargs
    that ``create_bidirectional_mask`` does not; drop them so an in-place swap
    inside a model's ``forward`` keeps working.
    """

    def _mask(*args, **kwargs):
        kwargs.pop("position_ids", None)
        kwargs.pop("cache_position", None)
        return create_bidirectional_mask(*args, **kwargs)

    return _mask


def enable_bidirectional_attention(model):
    """Convert ``model`` (a causal decoder backbone) to bidirectional attention.

    Returns the same model. Raises if the backbone does not expose a patchable
    ``create_causal_mask`` (so we never quietly leave it causal).
    """
    try:
        from transformers.masking_utils import create_bidirectional_mask
    except ImportError as e:  # pragma: no cover - version guard
        raise ImportError(
            "bidirectional attention requires a transformers version that exposes "
            "transformers.masking_utils.create_bidirectional_mask"
        ) from e

    n_attn = 0
    for module in model.modules():
        if getattr(module, "is_causal", False) is True:
            module.is_causal = False
            n_attn += 1

    mod = sys.modules.get(type(model).__module__)
    if mod is None or not hasattr(mod, "create_causal_mask"):
        raise RuntimeError(
            f"Cannot enable bidirectional attention: module '{type(model).__module__}' for "
            f"backbone '{type(model).__name__}' has no patchable create_causal_mask. This "
            "backbone does not follow the transformers masking_utils convention."
        )
    mod.create_causal_mask = _bidirectional_mask_compat(create_bidirectional_mask)

    logger.info(
        "Enabled bidirectional attention on %s: is_causal=False on %d attention module(s); "
        "create_causal_mask -> create_bidirectional_mask in %s",
        type(model).__name__, n_attn, type(model).__module__,
    )
    return model
