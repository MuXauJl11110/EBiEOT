import random
from typing import Generator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.samplers.base import Sampler
from src.utils.discrete_ot import OTPlanSampler


class SubsetGuidedDataset(Dataset):
    def __init__(
        self,
        dataset_in: Dataset,
        dataset_out: Dataset,
        num_labeled: str | int = "all",
        in_indicies: list[list[int]] | None = None,
        out_indicies: list[list[int]] | None = None,
    ):
        super(SubsetGuidedDataset, self).__init__()
        self.dataset_in = dataset_in
        self.dataset_out = dataset_out
        assert len(in_indicies) == len(out_indicies)
        self.num_classes = len(in_indicies)
        self.subsets_in = in_indicies
        self.subsets_out = out_indicies
        if num_labeled != "all":
            assert type(num_labeled) == int
            self.subsets_out = [np.random.choice(subset, num_labeled) for subset in self.subsets_out]

    def get(self, class_idx: int, subset_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_subset, y_subset = [], []
        in_indexis = random.sample(list(self.subsets_in[class_idx]), subset_size)
        out_indexis = random.sample(list(self.subsets_out[class_idx]), subset_size)
        for x_i, y_i in zip(in_indexis, out_indexis):
            x, c1 = self.dataset_in[x_i]
            y, c2 = self.dataset_out[y_i]
            assert c1 == c2
            x_subset.append(x)
            y_subset.append(y)
        return torch.stack(x_subset), torch.stack(y_subset)

    def __len__(self) -> int:
        return len(self.dataset_in)


class PairedSubsetSampler(Sampler):
    def __init__(
        self, dataset: SubsetGuidedDataset, subset_size: int, weight: float | None = None, device: str = "cuda"
    ):
        super(PairedSubsetSampler, self).__init__(device)
        self.dataset = dataset
        self.subset_size = subset_size
        if weight is None:
            weight = [1 / self.dataset.num_classes for _ in range(self.dataset.num_classes)]
        self.weight = weight

    def sample(self, batch_size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        classes = np.random.choice(self.dataset.num_classes, batch_size, p=self.weight)
        batch_X = []
        batch_Y = []
        with torch.no_grad():
            for class_ in classes:
                X, Y = self.dataset.get(class_, self.subset_size)
                batch_X.append(X.clone().to(self.device))
                batch_Y.append(Y.clone().to(self.device))

        return torch.stack(batch_X).to(self.device), torch.stack(batch_Y).to(self.device)


def get_indicies_subset(
    dataset: Dataset,
    subset_classes: np.ndarray | None = None,
    new_labels: dict[int, int] = {},
) -> tuple[list[torch.Tensor], list[int], list[list[int]]]:
    labels_subset: list[int] = []
    dataset_subset: list[torch.Tensor] = []
    class_indicies: list[list[int]] = [[] for _ in range(len(subset_classes))]
    i = 0
    for x, y in dataset:
        y_int = y.item()
        if y_int in subset_classes:
            class_indicies[new_labels[y_int]].append(i)
            labels_subset.append(new_labels[y_int])
            dataset_subset.append(x)
            i += 1
    return dataset_subset, labels_subset, class_indicies


class PairedSampler(Sampler):
    def __init__(
        self,
        X_sampler: Sampler,
        Y_sampler: Sampler,
        batch_size: int = 128,
        n_paired_samples: int | None = None,
        m_unpaired_samples: int | None = None,
        mini_batch_size: int | None = None,
        otp_sampler: OTPlanSampler | None = None,
        device: str = "cuda",
    ):
        super(PairedSampler, self).__init__(device=device)
        self.batch_size = batch_size
        self.device = device

        if mini_batch_size is not None and otp_sampler is None:
            raise ValueError("OTPlanSampler must initialized during mini-batch sampling! But is None.")
        self.paired_loader = self._init_paired_loader(
            X_sampler, Y_sampler, n_paired_samples, mini_batch_size, otp_sampler
        )
        self.paired_generator = iter(self.paired_loader)

        self.unpaired_loader = self._init_unpaired_loader(X_sampler, Y_sampler, m_unpaired_samples)
        self.unpaired_generator = iter(self.unpaired_loader)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sample_from_generator(self.unpaired_generator, self.unpaired_loader)

    def sample_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sample_from_generator(self.paired_generator, self.paired_loader)

    def sample_from_generator(
        self, generator: Generator[tuple[torch.Tensor, torch.Tensor], None, None], loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            try:
                X, Y = next(generator)
            except StopIteration:
                generator = iter(loader)
                X, Y = next(generator)
        return X, Y

    def _init_paired_loader(
        self,
        X_sampler: Sampler,
        Y_sampler: Sampler,
        n_paired_samples: int,
        mini_batch_size: int | None,
        otp_sampler: OTPlanSampler | None,
    ) -> DataLoader:
        X_paired, Y_paired = torch.empty((0, X_sampler.dim)), torch.empty((0, Y_sampler.dim))
        if mini_batch_size is not None:
            num_sampling_iterations = n_paired_samples // mini_batch_size
            num_remaining_samples = n_paired_samples % mini_batch_size
            for _ in range(num_sampling_iterations):
                _X, _Y = X_sampler.sample(mini_batch_size), Y_sampler.sample(mini_batch_size)
                X, Y = otp_sampler.sample_plan(_X, _Y)
                X_paired, Y_paired = torch.cat((X_paired, X), 0), torch.cat((Y_paired, Y), 0)
            if num_remaining_samples > 0:
                _X, _Y = X_sampler.sample(num_remaining_samples), Y_sampler.sample(num_remaining_samples)
                X, Y = otp_sampler.sample_plan(_X, _Y)
                X_paired, Y_paired = torch.cat((X_paired, X), 0), torch.cat((Y_paired, Y), 0)
        else:
            num_gaussians = len(X_sampler.mu)
            X, Y = X_sampler.sample(n_paired_samples), Y_sampler.sample(n_paired_samples)
            for i in range(num_gaussians):
                indices = np.arange(i, len(X), num_gaussians)
                for x in X[indices]:
                    for y in Y[indices]:
                        X_paired, Y_paired = torch.cat((X_paired, x[None, :]), 0), torch.cat((Y_paired, y[None, :]), 0)

        dataset = TensorDataset(X_paired, Y_paired)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.device)
        )

    def _init_unpaired_loader(
        self,
        X_sampler: Sampler,
        Y_sampler: Sampler,
        m_unpaired_samples: int,
    ) -> DataLoader:
        X, Y = X_sampler.sample(m_unpaired_samples), Y_sampler.sample(m_unpaired_samples)
        dataset = TensorDataset(X, Y)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.device)
        )
