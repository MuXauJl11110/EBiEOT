import torch

from src.utils.evaluation.metrics import mixture_kernel, rbf_kernel


def test_rbf_kernel_is_symmetric_psd_diagonal():
    x = torch.randn(8, 3)
    k = rbf_kernel(x, x, sigma=1.0)
    assert torch.allclose(k, k.T)
    assert (torch.diag(k) > 0).all()


def test_mixture_kernel_averages_psd_kernels():
    x = torch.randn(6, 2)
    y = torch.randn(5, 2)
    kernels = [rbf_kernel, lambda a, b: rbf_kernel(a, b, sigma=0.5)]
    k = mixture_kernel(x, y, kernels)
    expected = sum(f(x, y) for f in kernels) / len(kernels)
    assert torch.allclose(k, expected)
