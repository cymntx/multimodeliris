import sys
import os
import torch
from model import MultiModalNet
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
)



def test_model_initialization():
    model = MultiModalNet(num_classes=10, dropout=0.5)
    assert model is not None


def test_model_forward():
    model = MultiModalNet(num_classes=10, dropout=0.5)
    fp = torch.randn(4, 3, 128, 128)  # Batch size of 4
    left = torch.randn(4, 1, 64, 64)
    right = torch.randn(4, 1, 64, 64)
    output = model(fp, left, right)
    assert output.shape == (4, 10)
