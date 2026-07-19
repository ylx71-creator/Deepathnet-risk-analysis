import sys
import unittest
from pathlib import Path

import torch
from torch import nn


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from model_transformer_lrp import LRP  # noqa: E402


class CpuOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.seen_input_device = None
        self.seen_cam_device = None

    def forward(self, x):
        self.seen_input_device = x.device
        return self.linear(x)

    def relprop(self, cam, **kwargs):
        self.seen_cam_device = cam.device
        return cam


class LRPDeviceTest(unittest.TestCase):
    def test_generate_lrp_uses_model_device_on_cpu(self):
        model = CpuOnlyModel()
        lrp = LRP(model)
        data = (torch.ones(1, 3), torch.tensor([[1]]))

        result = lrp.generate_LRP(data, index=1)

        self.assertEqual(model.seen_input_device.type, "cpu")
        self.assertEqual(model.seen_cam_device.type, "cpu")
        self.assertEqual(result.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
