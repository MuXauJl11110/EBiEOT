import torch

from src.ebieot.costs.cnn import NonlocalCost, SquareCost, UNetCost, VanillaCost
from src.ebieot.potentials.cnn import VanillaPotential


def test_cnn_cost_forward_grad_shapes():
    device = torch.device("cpu")
    b, c, h, w = 4, 3, 32, 32
    flat = c * h * w
    x = torch.randn(b, c, h, w, device=device)
    y = torch.randn(b, c, h, w, device=device)

    cases = [
        (SquareCost, {"x_dim": flat, "y_dim": flat}),
        (VanillaCost, {"n_c": c, "x_dim": flat, "y_dim": flat}),
        (NonlocalCost, {"n_c": c, "x_dim": flat, "y_dim": flat}),
        (UNetCost, {"n_c": c, "x_dim": flat, "y_dim": flat}),
    ]
    for cost_cls, kwargs in cases:
        cost = cost_cls(**kwargs).to(device)
        out = cost(x, y)
        assert out.shape == (b,)
        grad = cost.grad_y(x, y)
        assert grad.shape == (b, c, h, w)


def test_cnn_potential_forward():
    device = torch.device("cpu")
    b, c, h, w = 2, 3, 32, 32
    y = torch.randn(b, c, h, w, device=device)
    pot = VanillaPotential(n_c=c, y_dim=c * h * w).to(device)
    out = pot(y)
    assert out.shape == (b,)
