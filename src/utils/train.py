import torch

from src.models.gmm_based import LightGCOT


def update_average(model_tgt: torch.nn.Module, model_src: torch.nn.Module, beta: float) -> None:
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert p_src is not p_tgt
            p_tgt.data.copy_(beta * p_tgt.data + (1.0 - beta) * p_src.data)


def compute_loss(
    model: LightGCOT,
    X_unpaired: torch.Tensor,
    Y_unpaired: torch.Tensor,
    X_paired: torch.Tensor,
    Y_paired: torch.Tensor,
) -> float:
    log_w_n = model.compute_log_w_n()  # [N]
    a_n = model.compute_a_n()  # [N x y_dim]
    A_n = model.compute_A_n()  # [N x y_dim]

    log_v_m_unpaired = model.compute_log_v_m(X_unpaired)  # [bs x M]
    b_m_unpaired = model.compute_b_m(X_unpaired)  # [bs x M x y_dim]
    f_c = model.compute_dual_potential(log_w_n, a_n, A_n, log_v_m_unpaired, b_m_unpaired)
    f = model.compute_primal_potential(Y_unpaired, log_w_n, a_n, A_n)

    unpaired_loss = -(f + f_c).mean()

    log_v_m_paired = model.compute_log_v_m(X_paired)  # [bs x M]
    b_m_paired = model.compute_b_m(X_paired)  # [bs x M x y_dim]
    c = model.compute_cost(Y_paired, log_v_m_paired, b_m_paired)

    paired_loss = c.mean()
    return (paired_loss + unpaired_loss).item()
