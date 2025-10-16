import unittest
import torch

from torch import tensor
from veomni.models.transformers.qwen2_5vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision, rotate_half

class TestModelingQwen25VL(unittest.TestCase):
    """modeling_qwen_2_5_vl单元测试"""

    def test_apply_rotary_pos_emb_vision(self):
        q = tensor([[[ 4.6094e-01,  -5.7031e-01],
            [-2.2559e-01,  5.8984e-01]],
            [[ 4.5508e-01,  -5.7422e-01],
            [-2.1680e-01,  5.7812e-01]]], 
            dtype=torch.bfloat16)
        k = tensor([[[ 0.2520, 0.6484],
            [-0.1855, 1.0781]],
            [[ 0.2422, 0.6641],
            [-0.2109, 1.0703]]],
            dtype=torch.bfloat16)
        cos = tensor([[ 1.0000,  1.0000],
            [ 1.0000,  1.0000]],
            dtype=torch.bfloat16)
        sin = tensor([[ 0.0000e+00, 0.0000e+00],
            [ 0.0000e+00, 1.5831e-04],], 
            dtype=torch.bfloat16)
        return_q_embed, return_k_embed = apply_rotary_pos_emb_vision(q, k, cos, sin)
        
        expected_q_embed = tensor(
            [[[ 0.4609, -0.5703],
            [-0.2256,  0.5898]],
            [[ 0.4551, -0.5742],
            [-0.2168,  0.5781]]], 
            dtype=torch.bfloat16)
        expected_k_embed = tensor(
            [[[ 0.2520,  0.6484],
            [-0.1855,  1.0781]],
            [[ 0.2422,  0.6641],
            [-0.2109,  1.0703]]], 
            dtype=torch.bfloat16)
        torch.testing.assert_close(return_q_embed, expected_q_embed, atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(return_k_embed, expected_k_embed, atol=1e-6, rtol=1e-4)
    
    def test_rotate_half(self):
        # 测试普通张量
        x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
        expected = torch.tensor([-5, -6, -7, -8, 1, 2, 3, 4], dtype=torch.float32)
        result = rotate_half(x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-4)

        # 测试奇数长度张量
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        # 对于奇数长度，最后一部分会多一个元素
        expected = torch.tensor([-3, -4, -5, 1, 2], dtype=torch.float32)
        result = rotate_half(x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-4)

        # 测试空张量
        x = torch.tensor([], dtype=torch.float32)
        expected = torch.tensor([], dtype=torch.float32)
        result = rotate_half(x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-4)

if __name__ == "__main__":
    unittest.main()