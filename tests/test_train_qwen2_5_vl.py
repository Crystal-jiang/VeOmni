import os
import unittest

from torch import nn
from tasks.omni.train_qwen2_5_vl import get_param_groups

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.visual_layer = nn.Linear(10, 5)
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc(self.visual_layer(x))

class TestTrainQwen25VL(unittest.TestCase):
    """train_qwen_2_5_vl单元测试"""

    def test_get_param_groups(self):
        model = nn.Module()
        default_lr = 1e-05
        vit_lr = 1e-06
        ret = get_param_groups(model, default_lr, vit_lr)
        self.assertEqual(len(ret), 2)
        self.assertEqual(ret[0]['lr'], vit_lr)
        self.assertEqual(ret[1]['lr'], default_lr)
    
    def test(self):
        # 创建一个SimpleModel实例
        model = SimpleModel()

        # 获取模型参数组
        param_groups = get_param_groups(model, default_lr=0.001, vit_lr=0.01)

        # 验证返回结果
        self.assertEqual(len(param_groups), 2)

        vit_params = param_groups[0]['params']
        vit_lr = param_groups[0]['lr']
        self.assertEqual(vit_lr, 0.01)  # 验证学习率是否正确
        self.assertEqual(len(vit_params), 2)  # visual_layer参数应该在vit_params中
        
        other_params = param_groups[1]['params']
        other_lr = param_groups[1]['lr']
        self.assertEqual(other_lr, 0.001)  # 验证学习率是否正确
        self.assertEqual(len(other_params), 2)  # fc层参数应该在other_params中
        
if __name__ == "__main__":
    unittest.main()