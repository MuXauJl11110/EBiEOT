import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.samplers.base import Sampler


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
