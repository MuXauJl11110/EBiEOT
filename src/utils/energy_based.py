from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import autograd


def computePotGrad(input: torch.Tensor, output: torch.Tensor, create_graph: bool = True, retain_graph: bool = True):
    """
    :Parameters:
    input : tensor (bs, *shape)
    output: tensor (bs, 1), i.e. NN(input)
    :Returns:
    gradient of output w.r.t. input (in batch manner), shape (bs, *shape)
    """
    grad = autograd.grad(
        outputs=output,
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )  # (bs, *shape)
    return grad[0]


# taken from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998
@contextmanager
def evaluating(net: torch.nn.Module):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


class SpectralNorm:
    def __init__(self, name: str, bound: bool = False, n_iters: int = 1):
        self.name = name
        self.bound = bound
        self.n_iters = n_iters

    def compute_weight(self, module: nn.Module):
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            for _ in range(self.n_iters):
                v = weight_mat.t() @ u
                v = v / v.norm()
                u = weight_mat @ v
                u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)

        else:
            weight_sn = weight / sigma

        return weight_sn, u

    @staticmethod
    def apply(module: nn.Module, name: str, bound, n_iters: int):
        fn = SpectralNorm(name, bound, n_iters)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + "_orig", weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + "_u", u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module: nn.Module):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + "_u", u)


def spectral_norm(
    module: nn.Module, init: bool = True, zero_bias: bool = True, std: float = 1, bound: bool = False, n_iters: int = 1
):
    if init:
        nn.init.normal_(module.weight, 0, std)

    if zero_bias:
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()

    SpectralNorm.apply(module, "weight", bound=bound, n_iters=n_iters)

    return module
