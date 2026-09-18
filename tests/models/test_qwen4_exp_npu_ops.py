"""CPU tests for Qwen4-Exp NPU op dispatch semantics."""

from __future__ import annotations

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

from veomni.models.transformers.qwen4_exp import qwen4_exp_npu_patch_gen_config as npu_config
from veomni.ops.kernel_registry import KERNEL_REGISTRY


class _EagerSlot:
    use_non_eager_impl = False

    def __call__(self, *args, **kwargs):
        raise AssertionError("Eager slot must not be called")


class _RecordingSlot:
    use_non_eager_impl = True

    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _expected_rms_norm(norm: Qwen4ExpTextRMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    normalized = norm._norm(hidden_states.float())
    return (normalized * (1.0 + norm.weight.float())).to(hidden_states.dtype)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    half = tensor.shape[-1] // 2
    return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


def _expected_partial_rope(
    query: torch.Tensor,
    key: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    query_rope = query[..., :rotary_dim]
    query_rotated = query_rope * cos + _rotate_half(query_rope) * sin
    query_rotated = torch.cat((query_rotated, query[..., rotary_dim:]), dim=-1)
    if key is None:
        return query_rotated

    key_rope = key[..., :rotary_dim]
    key_rotated = key_rope * cos + _rotate_half(key_rope) * sin
    key_rotated = torch.cat((key_rotated, key[..., rotary_dim:]), dim=-1)
    return query_rotated, key_rotated


def test_qwen4_exp_npu_rms_norm_dispatches_centered_kernel(monkeypatch) -> None:
    torch.manual_seed(0)
    norm = Qwen4ExpTextRMSNorm(8, eps=1e-6)
    hidden_states = torch.randn(2, 3, 8)
    fused_result = torch.full_like(hidden_states, 7.0)
    slot = _RecordingSlot(fused_result)
    monkeypatch.setattr(npu_config, "veomni_rms_norm", slot)

    result = npu_config.qwen4_exp_text_rms_norm_forward_patched(norm, hidden_states)

    assert torch.equal(result, fused_result)
    assert slot.calls == [((hidden_states, norm.weight, norm.eps), {})]


def test_qwen4_exp_grouped_rms_norm_stays_eager(monkeypatch) -> None:
    torch.manual_seed(0)
    norm = Qwen4ExpTextRMSNorm(8, group_size=4, eps=1e-6)
    hidden_states = torch.randn(2, 3, 8)
    monkeypatch.setattr(npu_config, "veomni_rms_norm", _EagerSlot())
    expected = npu_config.qwen4_exp_text_rms_norm_forward_patched(norm, hidden_states)

    slot = _RecordingSlot(torch.full_like(hidden_states, 7.0))
    monkeypatch.setattr(npu_config, "veomni_rms_norm", slot)
    result = npu_config.qwen4_exp_text_rms_norm_forward_patched(norm, hidden_states)

    assert slot.calls == []
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result, _expected_rms_norm(norm, hidden_states))


def test_qwen4_exp_partial_rope_eager_and_dispatch(monkeypatch) -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn_like(query)
    cos = torch.randn(2, 4, 4)
    sin = torch.randn_like(cos)
    monkeypatch.setattr(npu_config, "veomni_apply_rotary_pos_emb", _EagerSlot())

    expected_pair = npu_config.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
    expected_query_only = npu_config.apply_rotary_pos_emb(query, cos=cos, sin=sin, unsqueeze_dim=1)
    torch.testing.assert_close(expected_pair, _expected_partial_rope(query, key, cos, sin, 1))
    torch.testing.assert_close(expected_query_only, _expected_partial_rope(query, None, cos, sin, 1))

    fused_query = torch.full_like(query, 11.0)
    fused_key = torch.full_like(key, 13.0)
    slot = _RecordingSlot((fused_query, fused_key))
    monkeypatch.setattr(npu_config, "veomni_apply_rotary_pos_emb", slot)

    result = npu_config.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
    assert torch.equal(result[0], fused_query)
    assert torch.equal(result[1], fused_key)
    assert slot.calls == [((query, key, cos, sin), {"unsqueeze_dim": 1})]

    query_only = npu_config.apply_rotary_pos_emb(query, cos=cos, sin=sin, unsqueeze_dim=1)
    assert len(slot.calls) == 1
    torch.testing.assert_close(query_only, expected_query_only)


def test_qwen4_exp_vision_rope_eager_and_dispatch(monkeypatch) -> None:
    torch.manual_seed(0)
    query = torch.randn(5, 2, 8)
    key = torch.randn_like(query)
    cos = torch.randn(5, 1, 8)
    sin = torch.randn_like(cos)
    monkeypatch.setattr(npu_config, "veomni_apply_rotary_pos_emb_vision", _EagerSlot())
    expected = npu_config.apply_rotary_pos_emb_vision(query, key, cos, sin)

    expanded_cos = cos.unsqueeze(-2).float()
    expanded_sin = sin.unsqueeze(-2).float()
    expected_query = (query.float() * expanded_cos + _rotate_half(query.float()) * expanded_sin).to(query.dtype)
    expected_key = (key.float() * expanded_cos + _rotate_half(key.float()) * expanded_sin).to(key.dtype)
    torch.testing.assert_close(expected, (expected_query, expected_key))

    fused_query = torch.full_like(query, 17.0)
    fused_key = torch.full_like(key, 19.0)
    slot = _RecordingSlot((fused_query, fused_key))
    monkeypatch.setattr(npu_config, "veomni_apply_rotary_pos_emb_vision", slot)
    result = npu_config.apply_rotary_pos_emb_vision(query, key, cos, sin)

    assert torch.equal(result[0], fused_query)
    assert torch.equal(result[1], fused_key)
    assert slot.calls == [((query, key, cos, sin), {})]


def test_qwen4_exp_npu_op_registry_backends_exist() -> None:
    assert "npu" in KERNEL_REGISTRY.list_available("rms_norm", "qwen3_5")
    assert "npu" in KERNEL_REGISTRY.list_available("rotary_pos_emb", "partial")
    assert "npu" in KERNEL_REGISTRY.list_available("rotary_pos_emb_vision", "full")
