import torch
import torch.nn.functional as F

from src.ebieot.base import BaseEBiEOT
from src.ebieot.costs.nn import ClassificationCnnEnergy, ClassificationMlpEnergy
from src.ebieot.potentials.nn import ClassificationClassPotential


class ClassificationBasedEBiEOT(BaseEBiEOT):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        image_size: int = 28,
        epsilon: float = 0.5,
        arch: str = "cnn",
        mlp_embed_dim: int = 32,
        cnn_embed_dim: int = 64,
        hidden_dim: int = 128,
        mlp_hidden_dim: int | None = None,
        cnn_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.register_buffer("epsilon", torch.tensor(float(epsilon)))
        self.class_potential = ClassificationClassPotential(self.num_classes)

        arch_key = arch.lower().replace("_embed", "")
        mlp_hidden = int(mlp_hidden_dim if mlp_hidden_dim is not None else hidden_dim)
        cnn_hidden = int(cnn_hidden_dim if cnn_hidden_dim is not None else hidden_dim)
        if arch_key == "mlp":
            self.energy_model = ClassificationMlpEnergy(
                num_classes=self.num_classes,
                in_channels=int(in_channels),
                image_size=int(image_size),
                y_embed_dim=int(mlp_embed_dim),
                hidden_dim=mlp_hidden,
            )
        elif arch_key == "cnn":
            self.energy_model = ClassificationCnnEnergy(
                num_classes=self.num_classes,
                in_channels=int(in_channels),
                image_size=int(image_size),
                y_embed_dim=int(cnn_embed_dim),
                hidden_dim=cnn_hidden,
            )
        else:
            raise ValueError(f"Unknown arch value: {arch}. Expected 'mlp' or 'cnn'.")

    def compute_energies(self, x_batch: torch.Tensor) -> torch.Tensor:
        return self.energy_model(x_batch)

    def posterior(self, x_batch: torch.Tensor) -> torch.Tensor:
        energy = self.compute_energies(x_batch)
        scaled = (self.class_potential().unsqueeze(0) - energy) / self.epsilon
        return F.softmax(scaled, dim=1)

    def log_z(self, x_batch: torch.Tensor) -> torch.Tensor:
        energy = self.compute_energies(x_batch)
        scaled = (self.class_potential().unsqueeze(0) - energy) / self.epsilon
        return torch.logsumexp(scaled, dim=1)

    def supervised_cost(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> torch.Tensor:
        energy = self.compute_energies(x_batch)
        return energy.gather(1, y_batch.unsqueeze(1)).squeeze(1)

    def loss_with_parts(
        self,
        x_paired: torch.Tensor,
        y_paired: torch.Tensor,
        x_marginal: torch.Tensor,
        y_marginal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if y_paired.dim() == 1:
            paired_label = y_paired.long()
        elif y_paired.dim() == 2:
            paired_label = y_paired.argmax(dim=1).long()
        else:
            raise ValueError(
                "Expected label tensor shape [bs] or one hot shape [bs, num_classes]."
            )

        joint_term = self.supervised_cost(x_paired, paired_label).mean() / self.epsilon

        if y_marginal is None:
            marg_y_term = self.class_potential().mean() / self.epsilon
        else:
            if y_marginal.dim() == 1:
                marginal_label = y_marginal.long()
            elif y_marginal.dim() == 2:
                marginal_label = y_marginal.argmax(dim=1).long()
            else:
                raise ValueError(
                    "Expected label tensor shape [bs] or one hot shape [bs, num_classes]."
                )
            marg_y_term = (
                self.class_potential().gather(0, marginal_label).mean() / self.epsilon
            )

        log_z_term = self.log_z(x_marginal).mean()
        loss_value = joint_term - marg_y_term + log_z_term
        return {
            "loss": loss_value,
            "jointTerm": joint_term,
            "margYTerm": marg_y_term,
            "logzTerm": log_z_term,
        }

    def forward(self, batched_x: torch.Tensor) -> torch.Tensor:
        return self.posterior(batched_x)

    def compute_paired_loss(
        self, X_paired: torch.Tensor, Y_paired: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if Y_paired.dim() == 1:
            y_label = Y_paired.long()
        elif Y_paired.dim() == 2:
            y_label = Y_paired.argmax(dim=1).long()
        else:
            raise ValueError(
                "Expected label tensor shape [bs] or one hot shape [bs, num_classes]."
            )
        joint_term = self.supervised_cost(X_paired, y_label).mean() / self.epsilon
        return {"loss": joint_term, "jointTerm": joint_term}

    def compute_unpaired_loss(
        self, X_unpaired: torch.Tensor, Y_unpaired: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if Y_unpaired is None:
            marg_y_term = self.class_potential().mean() / self.epsilon
        else:
            if Y_unpaired.dim() == 1:
                y_label = Y_unpaired.long()
            elif Y_unpaired.dim() == 2:
                y_label = Y_unpaired.argmax(dim=1).long()
            else:
                raise ValueError(
                    "Expected label tensor shape [bs] or one hot shape [bs, num_classes]."
                )
            marg_y_term = (
                self.class_potential().gather(0, y_label).mean() / self.epsilon
            )

        log_z_term = self.log_z(X_unpaired).mean()
        loss_value = -marg_y_term + log_z_term
        return {"loss": loss_value, "margYTerm": marg_y_term, "logzTerm": log_z_term}
