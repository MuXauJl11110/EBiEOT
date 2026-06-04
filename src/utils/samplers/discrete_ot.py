from functools import partial

import ot as pot
import torch


class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean
    cost) with different implementations of the plan calculation.

    Based on https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py
    """

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        cost_function: str = "l2",
        normalize_cost: bool = False,
        **kwargs,
    ):
        r"""Initialize the OTPlanSampler class.

        Parameters
        ----------
        method : str
            The method used to compute the OT plan. Can be one of "exact", "sinkhorn",
            "unbalanced", or "partial".
        reg : float (default : 0.05)
            Entropic regularization coefficients.
        reg_m : float (default : 1.0)
            Marginal relaxation term for unbalanced OT (`method='unbalanced'`).
        cost_function : str (default : "l2")
            Which cost should be used. Can be one of "l2", "anti-l2", "rotation", "rotation-v2".
        normalize_cost : bool (default : False)
            Whether to normalize the cost matrix by its maximum value.
            It should be set to `False` when using minibatches.
        """
        # ot_fn should take (a, b, M) as arguments where a, b are marginals and
        # M is a cost matrix
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg, method="sinkhorn_log")
        elif method == "unbalanced":
            self.ot_fn = partial(
                pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m
            )
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.cost_function = cost_function
        self.normalize_cost = normalize_cost
        self.kwargs = kwargs

    def get_map(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)

        x0 = x0.double()
        x1 = x1.double()
        a = torch.full(
            (x0.shape[0],), 1.0 / x0.shape[0], dtype=torch.float64, device=x0.device
        )
        b = torch.full(
            (x1.shape[0],), 1.0 / x1.shape[0], dtype=torch.float64, device=x1.device
        )

        if self.cost_function == "l2":
            M = torch.cdist(x0, x1) ** 2
        elif self.cost_function == "anti-l2":
            M = torch.cdist(x0, x1) ** 2
        elif self.cost_function == "rotation":
            rotation_angle = torch.tensor(torch.pi / 2, dtype=x0.dtype, device=x0.device)
            rotation_matrix = torch.tensor(
                [
                    [torch.cos(rotation_angle), -torch.sin(rotation_angle)],
                    [torch.sin(rotation_angle), torch.cos(rotation_angle)],
                ],
                dtype=x0.dtype,
                device=x0.device,
            )
            M = torch.cdist(x0, -x1 @ rotation_matrix) ** 2
        elif self.cost_function == "rotation-v2":
            rotation_angle = torch.tensor(torch.pi / 2, dtype=x0.dtype, device=x0.device)
            rotation_matrix_A = torch.tensor(
                [
                    [torch.cos(rotation_angle), -torch.sin(rotation_angle)],
                    [torch.sin(rotation_angle), torch.cos(rotation_angle)],
                ],
                dtype=x0.dtype,
                device=x0.device,
            )
            rotation_matrix_B = torch.tensor(
                [
                    [torch.cos(-rotation_angle), -torch.sin(-rotation_angle)],
                    [torch.sin(-rotation_angle), torch.cos(-rotation_angle)],
                ],
                dtype=x0.dtype,
                device=x0.device,
            )
            M = torch.min(
                torch.cdist(x0, -x1 @ rotation_matrix_A),
                torch.cdist(x0, -x1 @ rotation_matrix_B),
            )
        else:
            raise ValueError(f"Unkown cost function: {self.cost_function}!")
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M)
        p = torch.as_tensor(p, dtype=torch.float64, device=M.device)
        p = torch.clamp(p, min=0.0)
        if not torch.isfinite(p).all():
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        return p

    def sample_map(
        self, pi: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        p = pi.flatten().double()
        p = torch.clamp(p, min=0.0)
        p_sum = p.sum()
        if not torch.isfinite(p_sum) or p_sum <= 0:
            p = torch.full_like(p, 1.0 / p.numel())
        else:
            p = p / p_sum
        choices = torch.multinomial(p, batch_size, replacement=True)
        i = torch.div(choices, pi.shape[1], rounding_mode="floor")
        j = torch.remainder(choices, pi.shape[1])
        return i, j

    def sample_plan(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0])
        return x0[i], x1[j]
