import torch

from src.models.base import BaseGenerativeModel


def update_average(model_tgt: torch.nn.Module, model_src: torch.nn.Module, beta: float) -> None:
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert p_src is not p_tgt
            p_tgt.data.copy_(beta * p_tgt.data + (1.0 - beta) * p_src.data)


def compute_loss(
    model: BaseGenerativeModel,
    X_unpaired: torch.Tensor,
    Y_unpaired: torch.Tensor,
    X_paired: torch.Tensor,
    Y_paired: torch.Tensor,
) -> float:
    output = model.compute_unpaired_loss(X_unpaired, Y_unpaired)
    unpaired_loss = output["loss"]

    paired_loss = model.compute_paired_loss(X_paired, Y_paired)
    return (paired_loss + unpaired_loss).item()
