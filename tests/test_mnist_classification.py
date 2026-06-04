"""Tests for MNIST energy-based classification utilities."""


import random

import torch

from src.ebieot.classification_based import ClassificationBasedEBiEOT
from src.utils.datasets.mnist_classification import (
    build_paired_marginal_val_indices,
    normalize_arch,
)


def test_normalize_arch_aliases():
    assert normalize_arch("cnn_embed") == "cnn"
    assert normalize_arch("mlp_embed") == "mlp"
    assert normalize_arch("CNN") == "cnn"


def test_build_paired_marginal_val_indices_no_overlap():
    indices_by_class = {
        0: list(range(0, 100)),
        1: list(range(100, 200)),
    }
    rng = random.Random(0)
    paired, marginal, val = build_paired_marginal_val_indices(
        indices_by_class,
        paired_per_class=5,
        marginal_per_class=10,
        val_per_class=3,
        rng=rng,
    )
    paired_set = set(paired)
    marginal_set = set(marginal)
    val_set = set(val)
    assert paired_set.isdisjoint(marginal_set)
    assert paired_set.isdisjoint(val_set)
    assert marginal_set.isdisjoint(val_set)
    assert len(paired) == 2 * 5
    assert len(marginal) == 2 * 10
    assert len(val) == 2 * 3


def test_loss_with_parts_marginal_mean_vs_labeled():
    torch.manual_seed(0)
    model = ClassificationBasedEBiEOT(
        num_classes=10,
        in_channels=1,
        image_size=28,
        epsilon=0.5,
        arch="cnn",
    )
    with torch.no_grad():
        model.class_potential.logits.copy_(torch.linspace(-1.0, 1.0, 10))
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    x_m = torch.randn(8, 1, 28, 28)
    y_m = torch.randint(0, 10, (8,))
    ref = model.loss_with_parts(x, y, x_m)
    labeled = model.loss_with_parts(x, y, x_m, y_m)
    assert not torch.allclose(ref["margYTerm"], labeled["margYTerm"])

    eps = model.epsilon
    joint = model.supervised_cost(x, y).mean() / eps
    marg = model.class_potential().mean() / eps
    logz = model.log_z(x_m).mean()
    expected = joint - marg + logz
    assert torch.allclose(ref["loss"], expected, atol=1e-5)
