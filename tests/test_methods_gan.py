import torch

from src.networks.gan.descriminator import CGANDiscriminator, MLPDiscriminator
from src.networks.gan.generator import CGANGenerator, MLPGenerator


def test_mlp_generator_forward_shape():
    batch_size = 4
    x_dim, z_dim, out_dim = 2, 1, 2
    gen = MLPGenerator(x_dim=x_dim, z_dim=z_dim, out_dim=out_dim, layers=[32, 32])
    x = torch.randn(batch_size, x_dim)
    z = torch.randn(batch_size, z_dim)
    out = gen(x, z)
    assert out.shape == (batch_size, out_dim)


def test_mlp_discriminator_forward_shape():
    batch_size = 4
    y_dim = 2
    disc = MLPDiscriminator(x_dim=y_dim, layers=[32, 32])
    y = torch.randn(batch_size, y_dim)
    out = disc(y)
    assert out.shape == (batch_size,)


def test_cgan_generator_forward_shape():
    batch_size = 4
    x_dim, y_dim, out_dim, n_y, z_dim = 2, 2, 2, 3, 1
    gen = CGANGenerator(
        x_dim=x_dim,
        y_dim=y_dim,
        n_y=n_y,
        out_dim=out_dim,
        z_dim=z_dim,
        layers=[32, 32],
    )
    x = torch.randn(batch_size, x_dim)
    t = torch.randint(0, n_y, (batch_size,))
    z = torch.randn(batch_size, z_dim)
    out = gen(x, t, z)
    assert out.shape == (batch_size, out_dim)


def test_cgan_discriminator_forward_shape():
    batch_size = 4
    x_dim, y_dim, n_y = 2, 2, 3
    disc = CGANDiscriminator(x_dim=x_dim, y_dim=y_dim, n_y=n_y, layers=[32, 32])
    x_y = torch.randn(batch_size, x_dim)
    t = torch.randint(0, n_y, (batch_size,))
    x_y_next = torch.randn(batch_size, x_dim)
    out = disc(x_y, t, x_y_next)
    assert out.shape == (batch_size,)
