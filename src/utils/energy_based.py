from contextlib import contextmanager

import torch
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
