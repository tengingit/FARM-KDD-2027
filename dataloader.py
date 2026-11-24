from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from config import DATA_DIR, FEATURE_FIELD


class MDCDataset(Dataset):
    def __init__(self, name, data_dir=DATA_DIR):
        data_file = Path(data_dir) / f"{name}.mat"
        self.data = scipy.io.loadmat(data_file)
        field = FEATURE_FIELD.get(name, 1)
        self.features = torch.from_numpy(self.data["data"][0][0][field]).float()
        self.labels = torch.from_numpy(self.data["target"]).long() - 1

    def idx_cv(self, fold):
        split = self.data["idx_folds"][fold][0]
        train = split["train"][0][0].reshape(-1).astype(np.int32) - 1
        test = split["test"][0][0].reshape(-1).astype(np.int32) - 1
        return train, test

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def data_loader(dataset, fold, batch_size):
    train, test = dataset.idx_cv(fold)
    return (
        DataLoader(Subset(dataset, train), batch_size=batch_size, shuffle=False),
        DataLoader(Subset(dataset, test), batch_size=batch_size, shuffle=False),
    )
