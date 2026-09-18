# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Patch configuration for the Qwen4-Exp NPU correctness path.

Regen command:
patchgen veomni.models.transformers.qwen4_exp.qwen4_exp_npu_patch_gen_config -o veomni/models/transformers/qwen4_exp/generated --diff

Qwen4-Exp uses partial interleaved mRoPE and experimental QSA/PLE code paths.
This NPU build inherits the device-agnostic GPU patch set and adds fused
RMSNorm/RoPE dispatch while retaining eager fallbacks for unsupported layouts.
"""

from copy import deepcopy

import torch

from veomni.models.transformers.qwen4_exp.qwen4_exp_gpu_patch_gen_config import config as gpu_config


config = deepcopy(gpu_config)
config.target_file = "patched_modeling_qwen4_exp_npu.py"
config.description = "Qwen4-Exp NPU VLM-SFT correctness integration with PLE sharding"

config.add_post_import_block(
    """
    # NPU-only OpSlots. Qwen4-Exp shares Qwen3.5's zero-centered
    # ``(1 + weight)`` RMSNorm contract.
    veomni_rms_norm = OpSlot("rms_norm", "qwen3_5")
    veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "partial")
    veomni_apply_rotary_pos_emb_vision = OpSlot("rotary_pos_emb_vision", "full")
    """
)

# OpSlots are declared in the generated module's post-import blocks.
veomni_rms_norm = None
veomni_apply_rotary_pos_emb = None
veomni_apply_rotary_pos_emb_vision = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions for the eager RoPE fallbacks."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@config.override_method(
    "Qwen4ExpTextRMSNorm.forward",
    description="Use NPU fused zero-centered RMSNorm when the layout is compatible",
)
def qwen4_exp_text_rms_norm_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    # Grouped RMSNorm normalizes each group independently. The NPU fused op
    # normalizes the complete last dimension, so only the ungrouped layout is
    # safe to replace.
    if veomni_rms_norm.use_non_eager_impl and self.group_size is None:
        return veomni_rms_norm(x, self.weight, self.eps)

    output = self._norm(x.float())
    # Llama does x.to(float16) * w whilst Qwen4ExpText is (x * w).to(float16)
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)


@config.replace_function(
    "apply_rotary_pos_emb",
    description="Use NPU fused partial rotary position embedding",
)
def apply_rotary_pos_emb(q, k=None, cos=None, sin=None, unsqueeze_dim=1):
    # The NPU partial-RoPE kernel returns both q and k. Keep the query-only
    # calls on the exact eager path to preserve the upstream return contract.
    if veomni_apply_rotary_pos_emb.use_non_eager_impl and k is not None:
        return veomni_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rope, q_nope = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rope = (q_rope * cos) + (rotate_half(q_rope) * sin)
    q_rotated = torch.cat([q_rope, q_nope], dim=-1)

    if k is None:
        return q_rotated

    k_rope, k_nope = k[..., :rotary_dim], k[..., rotary_dim:]
    k_rope = (k_rope * cos) + (rotate_half(k_rope) * sin)
    k_rotated = torch.cat([k_rope, k_nope], dim=-1)
    return q_rotated, k_rotated


@config.replace_function(
    "apply_rotary_pos_emb_vision",
    description="Use NPU fused rotary position embedding in the vision tower",
)
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if veomni_apply_rotary_pos_emb_vision.use_non_eager_impl:
        return veomni_apply_rotary_pos_emb_vision(q, k, cos, sin)

    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)
