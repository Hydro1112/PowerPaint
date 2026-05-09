import torch
import numpy as np
import random


class ProbPickingDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_list):
        self.datasets_list = datasets_list

        self.probs = np.array([
            d["prob"] for d in datasets_list
        ])

        self.probs = self.probs / self.probs.sum()

        self.indices = []

        for idx, d in enumerate(datasets_list):
            self.indices.extend([
                (idx, i)
                for i in range(len(d["dataset"]))
            ])

        self.total_length = len(self.indices)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.indices[idx]

        return self.datasets_list[
            dataset_idx
        ]["dataset"][sample_idx]

    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        random.shuffle(self.indices)

        return self

    def select(self, indices):
        new_dataset = ProbPickingDataset.__new__(
            ProbPickingDataset
        )

        new_dataset.datasets_list = self.datasets_list
        new_dataset.probs = self.probs

        available_indices = [
            i for i in indices
            if i < len(self.indices)
        ]

        new_dataset.indices = [
            self.indices[i]
            for i in available_indices
        ]

        new_dataset.total_length = len(
            new_dataset.indices
        )

        return new_dataset