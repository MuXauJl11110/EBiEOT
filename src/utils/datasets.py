from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.samplers.guided import (
    PairedSubsetSampler,
    SubsetGuidedDataset,
    get_indicies_subset,
)


def get_Splatter_dataset(dataset_path: str = "../datasets/Splatter/") -> dict[str, np.ndarray]:
    dataset_path = Path(dataset_path)
    print(f"Dataset path: {dataset_path}")
    norm_counts = pd.read_csv(dataset_path.joinpath("combine_expression.csv"))
    labels = pd.read_csv(dataset_path.joinpath("combine_labels.csv"))
    domain_labels = pd.read_csv(dataset_path.joinpath("domain_labels.csv"))
    data_set = {
        "features": norm_counts.T.values,
        "labels": labels.iloc[:, 0].values,
        "accessions": domain_labels.iloc[:, 0].values,
    }
    return data_set


def get_Splatter_data(
    data_set: dict[str, np.ndarray],
    source_name: str = "TM_baron_mouse_for_segerstolpe",
    target_name: str = "segerstolpe_human",
) -> tuple[TensorDataset, np.ndarray, TensorDataset, np.ndarray]:
    """
    :param dict[str, np.ndarray] data_set: Dataset.
    :param str source_name: Source name "TM_baron_mouse_for_segerstolpe" or "segerstolpe_human", defaults to "TM_baron_mouse_for_segerstolpe"
    :param str target_name: Target name "TM_baron_mouse_for_segerstolpe" or "segerstolpe_human", defaults to "segerstolpe_human"
    """

    def extract_data_by_name(name: str) -> dict[str, np.ndarray]:
        domain_to_indices = np.where(data_set["accessions"] == name)[0]
        return {
            "features": data_set["features"][domain_to_indices],
            "labels": data_set["labels"][domain_to_indices],
            "accessions": data_set["accessions"][domain_to_indices],
        }

    source_set = extract_data_by_name(source_name)
    source_labels = np.unique(source_set["labels"])
    print(f"Initial source labels: {source_labels}")

    target_set = extract_data_by_name(target_name)
    target_labels = np.unique(target_set["labels"])
    print(f"Initial target labels: {target_labels}")

    common_labels = np.intersect1d(np.unique(source_set["labels"]), np.unique(target_set["labels"]))
    print(f"Common labels: {common_labels}")

    source_set_filtered_mask = np.isin(source_set["labels"], common_labels)
    source_set_filtered = {
        "features": source_set["features"][source_set_filtered_mask],
        "labels": source_set["labels"][source_set_filtered_mask],
        "accessions": source_set["accessions"][source_set_filtered_mask],
    }

    label_mapping = {label: index for index, label in enumerate(np.unique(source_set_filtered["labels"]))}
    print(f"Label mapping: {label_mapping}")

    source_set_filtered["labels"] = np.array([label_mapping[label] for label in source_set_filtered["labels"]])
    # This works, because target and common labels are the same
    target_set["labels"] = np.array([label_mapping[label] for label in target_set["labels"]])
    source_labels = np.unique(source_set_filtered["labels"])
    print(f"Processed source labels: {source_labels}")
    print(f"Processed target labels: {target_labels}")

    print(f"Source dataset shape: {source_set_filtered['features'].shape}")
    print(f"Target dataset shape: {target_set['features'].shape}")

    source_data = TensorDataset(
        torch.FloatTensor(source_set_filtered["features"]), torch.LongTensor(source_set_filtered["labels"])
    )
    target_data = TensorDataset(torch.FloatTensor(target_set["features"]), torch.LongTensor(target_set["labels"]))

    return source_data, source_labels, target_data, source_labels  # target_labels := source_labels


def get_Splatter_samplers(
    source_data: TensorDataset,
    source_labels: np.ndarray,
    target_data: TensorDataset,
    target_labels: np.ndarray,
    num_labeled: int,
    train_subset_size: int,
    device: str = "cuda",
) -> tuple[PairedSubsetSampler, PairedSubsetSampler]:
    source_subset_samples, source_subset_labels, source_class_indicies = get_indicies_subset(
        source_data, subset_classes=np.arange(len(source_labels)), new_labels=source_labels
    )
    source_train = TensorDataset(torch.stack(source_subset_samples), torch.LongTensor(source_subset_labels))

    target_subset_samples, target_subset_labels, target_class_indicies = get_indicies_subset(
        target_data, subset_classes=np.arange(len(target_labels)), new_labels=target_labels
    )
    target_train = TensorDataset(torch.stack(target_subset_samples), torch.LongTensor(target_subset_labels))

    train_set = SubsetGuidedDataset(
        source_train,
        target_train,
        num_labeled=num_labeled,
        in_indicies=source_class_indicies,
        out_indicies=target_class_indicies,
    )

    full_set = SubsetGuidedDataset(
        source_train,
        target_train,
        num_labeled="all",
        in_indicies=source_class_indicies,
        out_indicies=target_class_indicies,
    )
    train_XY_sampler = PairedSubsetSampler(train_set, subset_size=train_subset_size, device=device)
    full_XY_sampler = PairedSubsetSampler(full_set, subset_size=1, device=device)

    return train_XY_sampler, full_XY_sampler
