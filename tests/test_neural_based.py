import torch
from torch.distributions import Independent, Normal

from src.ebieot.costs.base import BaseCost
from src.ebieot.ebieot_nn import EbieotNn
from src.ebieot.potentials.base import BasePotential
from src.ebieot.sampling.sample_buffer import SampleBuffer


class ToyPotential(BasePotential):
    def func(self, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * (y**2).sum()


class ToyCost(BaseCost):
    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((x - y) ** 2).sum()


def make_neural_model(
    *,
    sampler_type: str = "langevin",
    num_iterations: int = 100,
    step_size: float = 0.05,
    noise: float = 0.02,
    decay: float = 1.0,
    thresh: float | None = None,
    grad_proj_type: str = "none",
) -> EbieotNn:
    y_dim = 2
    noise_gen = Independent(Normal(torch.zeros(y_dim), torch.ones(y_dim)), 1)
    buffer = SampleBuffer(noise_gen, p=0.0, max_samples=1000)
    return EbieotNn(
        potential=ToyPotential(y_dim=y_dim),
        cost=ToyCost(x_dim=y_dim, y_dim=y_dim),
        sample_buffer=buffer,
        epsilon=1.0,
        alpha=0.01,
        reference_data_noise_sigma=0.0,
        sampler_type=sampler_type,
        num_iterations=num_iterations,
        step_size=step_size,
        noise=noise,
        decay=decay,
        thresh=thresh,
        grad_proj_type=grad_proj_type,
    )


def test_get_samples_energy_langevin_shape_and_stats():
    torch.manual_seed(0)
    model = make_neural_model(
        sampler_type="langevin",
        num_iterations=2,
        step_size=0.05,
        noise=0.02,
        decay=1.0,
        thresh=None,
    )
    x = torch.randn(4, 2)
    y0 = torch.randn(4, 2)
    with torch.no_grad():
        y_out, stats = model.get_samples_energy(x, y0, compute_stats=True)
    assert y_out.shape == y0.shape
    assert set(stats) == {"neg_energy_t", "cost_t", "potential_t", "noise"}
    assert all(isinstance(v, float) for v in stats.values())


def test_get_samples_energy_pseudo_runs_with_grad_proj():
    torch.manual_seed(1)
    model = make_neural_model(
        sampler_type="pseudo",
        num_iterations=2,
        step_size=0.1,
        noise=0.01,
        decay=1.0,
        grad_proj_type="none",
        thresh=None,
    )
    x = torch.randn(3, 2)
    y0 = torch.randn(3, 2)
    with torch.no_grad():
        y_out, stats = model.get_samples_energy(x, y0, compute_stats=False)
    assert y_out.shape == y0.shape
    assert stats == {}


def test_get_samples_energy_empty_stats_dict_when_disabled():
    torch.manual_seed(2)
    model = make_neural_model(
        sampler_type="langevin", num_iterations=1, step_size=0.01, noise=0.01
    )
    x = torch.randn(2, 2)
    y0 = torch.randn(2, 2)
    with torch.no_grad():
        _, stats = model.get_samples_energy(x, y0, compute_stats=False)
    assert stats == {}


def test_compute_paired_loss_returns_dict_with_loss():
    torch.manual_seed(3)
    model = make_neural_model(sampler_type="langevin")
    x = torch.randn(5, 2)
    y = torch.randn(5, 2)
    out = model.compute_paired_loss(x, y)
    assert isinstance(out, dict)
    assert "loss" in out
    assert out["loss"].ndim == 0


def test_forward_returns_samples_same_batch_dim():
    torch.manual_seed(4)
    model = make_neural_model(
        sampler_type="langevin", num_iterations=1, step_size=0.01, noise=0.01
    )
    x = torch.randn(6, 2)
    with torch.no_grad():
        y = model.forward(x)
    assert y.shape == x.shape
