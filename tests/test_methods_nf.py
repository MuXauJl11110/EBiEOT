import torch

from src.networks.nf.real_nvp import ConditionalRealNVP


def test_conditional_real_nvp_log_prob_smoke():
    torch.manual_seed(0)
    batch_size = 4
    features = 2
    context_features = 2
    model = ConditionalRealNVP(
        features=features,
        hidden_features=16,
        context_features=context_features,
        hidden_context_features=32,
        num_layers=2,
        num_blocks_per_layer=1,
    )
    x = torch.randn(batch_size, context_features)
    y = torch.randn(batch_size, features)
    log_prob = model.log_prob(inputs=y, context=x)
    assert log_prob.shape == (batch_size,)
    assert torch.isfinite(log_prob).all()
