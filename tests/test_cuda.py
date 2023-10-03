# test_inference.py
import torch


class TestCuda:
    def test_cuda(self):
        f = torch.nn.Conv2d(3, 8, 3, device="cuda")
        X = torch.randn(2, 3, 4, 4, device="cuda")

        Y = X @ X
        assert Y.shape == torch.Size([2, 3, 4, 4])
        Y = f(X)
        assert Y.shape == torch.Size([2, 8, 2, 2])
