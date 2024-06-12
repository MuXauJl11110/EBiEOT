import pytest
import torch

from src.light_gcot import LightGCOT

x_dim_list = [5]
y_dim_list = [5]
n_potentials_list = [200]
m_potentials_list = [300]
A_diagonal_init_list = [0.1]
is_B_diagonal_list = [True]
batch_size_list = [128]


@pytest.fixture(params=x_dim_list)
def x_dim(request):
    return request.param


@pytest.fixture(params=y_dim_list)
def y_dim(request):
    return request.param


@pytest.fixture(params=n_potentials_list)
def n_potentials(request):
    return request.param


@pytest.fixture(params=m_potentials_list)
def m_potentials(request):
    return request.param


@pytest.fixture(params=A_diagonal_init_list)
def A_diagonal_init(request):
    return request.param


@pytest.fixture(params=is_B_diagonal_list)
def is_B_diagonal(request):
    return request.param


@pytest.fixture(params=batch_size_list)
def batch_size(request):
    return request.param


@pytest.fixture
def b_m(batch_size: int, m_potentials: int, y_dim: int):
    return torch.randn(batch_size, m_potentials, y_dim)


@pytest.fixture
def B_m(batch_size: int, m_potentials: int, y_dim: int):
    return torch.rand(batch_size, m_potentials, y_dim)


@pytest.fixture
def log_v_m(batch_size: int, m_potentials: int, y_dim: int):
    return torch.rand(batch_size, m_potentials)


@pytest.fixture
def D(x_dim: int, y_dim: int, n_potentials: int, m_potentials: int, A_diagonal_init: float, is_B_diagonal: bool):
    return LightGCOT(
        x_dim=x_dim,
        y_dim=y_dim,
        n_potentials=n_potentials,
        m_potentials=m_potentials,
        A_diagonal_init=A_diagonal_init,
        is_B_diagonal=is_B_diagonal,
    )
