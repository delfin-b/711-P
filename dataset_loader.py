# dataset_loader.py
import pandas as pd
import torch
from torch.utils.data import Dataset

class LatentToProfileDataset(Dataset):
    def __init__(self, pickle_path):
        df = pd.read_pickle(pickle_path)
        self.latents = torch.tensor(df[[f"z{i}" for i in range(64)]].values, dtype=torch.float32)
        self.profiles = torch.tensor(df["vs_profile"].values.tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.profiles[idx]
