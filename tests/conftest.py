import pytest

from src.light_gcot import LightGCOT

n_potentials_list = [100, 200]
m_potentials_list = [300, 400]
A_diagonal_init_list = [0.1]
B_diagonal_init_list = [0.3]
batch_size_list = [128]


@pytest.fixture(params=n_potentials_list)
def n_potentials(request):
    return request.param


@pytest.fixture(params=m_potentials_list)
def m_potentials(request):
    return request.param


@pytest.fixture(params=A_diagonal_init_list)
def A_diagonal_init(request):
    return request.param


@pytest.fixture(params=B_diagonal_init_list)
def B_diagonal_init(request):
    return request.param


@pytest.fixture(params=batch_size_list)
def batch_size(request):
    return request.param


@pytest.fixture
def D(n_potentials: int, m_potentials: int, A_diagonal_init: float, B_diagonal_init: float):
    return LightGCOT(
        n_potentials=n_potentials,
        m_potentials=m_potentials,
        A_diagonal_init=A_diagonal_init,
        B_diagonal_init=B_diagonal_init,
    )
