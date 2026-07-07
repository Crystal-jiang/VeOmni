"""Generic word-token embedding (``wte``) + LM head as a graph node.

``TextEncoder(TextEncoderModuleMixin, PreTrainedModel)`` — ``encode`` /
``decode`` call-sites mirror a VQ codec pre/post stage so the backbone stays
vocab-agnostic.  Family-specific chat template / sampling live in
``modules/<family>/text_encoder/``.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ......distributed.parallel_state import get_parallel_state
from ......ops.kernels.embed import AllToAllEmbedding, VocabParallelLinear
from .configuration import TextEncoderConfig
from .modulemixin import TextEncoderMetricMeterMixin, TextEncoderModuleMixin


class TextEncoder(TextEncoderModuleMixin, TextEncoderMetricMeterMixin, PreTrainedModel):
    """Word-token embedding + LM head."""

    config_class = TextEncoderConfig
    base_model_prefix = ""
    _no_split_modules: list = ["Embedding", "Linear"]
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(self, config: TextEncoderConfig):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.post_init()

    # ── Embedding accessors ────────────────────────────────────────────────────

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def get_output_embeddings(self) -> Optional[nn.Module]:
        # When tied there is no separate ``lm_head`` — ``_project`` reuses
        # ``embed_tokens.weight`` directly. Returning ``embed_tokens`` makes the
        # generic load-time weight-tie a harmless self-assignment instead of
        # crashing on a ``None`` output module.
        if self.config.tie_word_embeddings:
            return self.embed_tokens
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if not self.config.tie_word_embeddings:
            self.lm_head = new_embeddings

    # ── Gradient checkpointing (no-op — nothing to recompute) ──────────────────

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """No-op: an embedding/head has no activations worth checkpointing.

        Overridden so a uniform ``enable`` call from the trainer is accepted
        silently instead of raising in ``PreTrainedModel`` (see class note).
        """
        return

    def gradient_checkpointing_disable(self) -> None:
        """No-op counterpart to :meth:`gradient_checkpointing_enable`."""
        return

    def forward(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        return self.encode(**kwargs)

    def encode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        embeds = self._embed_tokens(input_ids)
        return {
            "inputs_embeds": embeds.squeeze(0) if embeds.size(0) == 1 else embeds,
        }

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding lookup, vocab-parallel-aware when ``emb`` extra-parallel is on.
        Use AllToAllEmbedding to handle the vocab-parallel embedding.
        """
        ps = get_parallel_state()
        if "emb" not in ps.extra_parallel_sizes or not ps.extra_parallel_enabled("emb"):
            return self.embed_tokens(input_ids)

        # Under ``emb`` extra-parallel the weight is sharded on dim-0 (vocab) by
        # the parallel plan AND on dim-1 (hidden) by FSDP over the ``emb_fsdp``
        # mesh. AllToAllEmbedding needs this emb-rank's FULL [vocab/emb_size,
        # hidden] slice, so ``full_tensor()`` all-gathers the hidden shards over
        # emb_fsdp (reconstructing the same emb-chunk — not mixing emb ranks).
        weight = self.embed_tokens.weight
        if isinstance(weight, DTensor):
            # Inference (no grad): ``detach()`` first — full_tensor()'s redistribute
            # hits an unsupported in-place ``detach_`` on a grad-requiring DTensor.
            # Training: keep grad so AllToAllEmbedding's backward reaches the param.
            weight = weight.full_tensor() if torch.is_grad_enabled() else weight.detach().full_tensor()
        return AllToAllEmbedding.apply(ps.extra_parallel_group("emb"), input_ids, weight)

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> dict:
        from ......ops.kernels.cross_entropy import chunk_loss_function

        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None

        if shift_labels is not None:
            weight = self._get_lm_head_weight()
            hs_3d = hidden_states.unsqueeze(0)
            labels_2d = shift_labels.unsqueeze(0)
            loss, _ = chunk_loss_function(
                hidden_states=hs_3d,
                weights=weight,
                labels=labels_2d,
                chunk_size=1024,
                ignore_index=-100,
            )
        elif labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            hidden_states = hidden_states[..., :-1, :].contiguous()
            weight = self._get_lm_head_weight()
            hs_3d = hidden_states.unsqueeze(0)
            labels_2d = shift_labels.unsqueeze(0)
            loss, _ = chunk_loss_function(
                hidden_states=hs_3d,
                weights=weight,
                labels=labels_2d,
                chunk_size=1024,
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
        }

    def _get_lm_head_weight(self) -> torch.Tensor:
        if not self.config.tie_word_embeddings:
            return self.lm_head.weight

        weight = self.embed_tokens.weight
        if isinstance(weight, DTensor):
            weight = weight.full_tensor() if torch.is_grad_enabled() else weight.detach().full_tensor()
        weight = weight.to(dtype=self.dtype)
        return weight

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.config.tie_word_embeddings:
            return self.lm_head(hidden_states)

        # Tied head reuses ``embed_tokens.weight`` directly, bypassing
        # ``embed_tokens.__call__`` — and ``embed_tokens`` is its own FSDP2 unit
        # (``_no_split_modules``), so the pre-forward all-gather that would
        # materialize + cast the weight never fires here. Reconstruct it
        # ourselves, mirroring ``_embed_tokens``.
        weight = self.embed_tokens.weight
        if isinstance(weight, DTensor):
            # ``full_tensor()`` all-gathers the FSDP (``emb_fsdp``) hidden shards
            # back to this rank's slice — under ``emb`` that is the vocab-sharded
            # ``[vocab/emb_size, hidden]`` chunk; without ``emb`` it is the full
            # ``[vocab, hidden]`` weight. Keep grad in training so the tied param
            # still receives gradient; detach under inference (full_tensor()'s
            # redistribute trips on an in-place ``detach_`` of a grad DTensor).
            weight = weight.full_tensor() if torch.is_grad_enabled() else weight.detach().full_tensor()
        # ``full_tensor()`` returns the fp32 master param (no mixed-precision
        # cast), so align with the activation dtype before the matmul.
        weight = weight.to(hidden_states.dtype)

        ps = get_parallel_state()
        if "emb" in ps.extra_parallel_sizes and ps.extra_parallel_enabled("emb"):
            # Vocab-parallel tied head: ``weight`` above is only this rank's vocab
            # chunk, so all-gather the vocab shards over ``emb`` and project to
            # full-vocab logits (dual of ``AllToAllEmbedding``).
            return VocabParallelLinear.apply(ps.extra_parallel_group("emb"), hidden_states, weight)
        return F.linear(hidden_states, weight)
