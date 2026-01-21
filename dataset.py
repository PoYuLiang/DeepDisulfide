from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch

from typing import Optional, Literal
import numpy as np
import random
import os

NOISE_HINT_TEXT = "NOISE"
# MAX_LEN = 8886
MAX_LEN = 1024


class DisulfidEmbeddingDataset(Dataset):
    def __init__(
        self,
        folder_path,
        split: Optional[Literal["train", "val"]] = None,
        noise_ration=0,
    ):
        self.file_path = sorted(
            [
                os.path.join(folder_path, i)
                for i in os.listdir(folder_path)
                if i.endswith(".pt")
            ]
        )
        if split == "train":
            self.file_path = self.file_path[: int(len(self.file_path) * 0.8)]
        elif split == "val":
            self.file_path = self.file_path[int(len(self.file_path) * 0.8) :]
        if noise_ration != 0:
            noise_count = int(len(self.file_path) * noise_ration)
            self.file_path += [NOISE_HINT_TEXT] * noise_count

    def __len__(self):
        return len(self.file_path)

    def get_noise_item(self):
        # Create a random length embedding with pure noise
        # Not res. should be disulfide.
        seq_len = random.randint(5,MAX_LEN)
        mask = torch.zeros(MAX_LEN, dtype=torch.float32)
        mask[:seq_len] = 1
        cys_mask = torch.zeros(MAX_LEN, dtype=torch.bool)
        emb = torch.randn(size=(MAX_LEN,1024))
        disulfid_res = torch.zeros(MAX_LEN, dtype=torch.float32)
        return {
            "id": "NOISE",
            "seq": "NOISE",
            "emb": emb,
            "mask": mask,
            "cys_mask": cys_mask,
            "disulfid_res": disulfid_res,
        }

    def __getitem__(self, idx):
        if self.file_path[idx] == NOISE_HINT_TEXT:
            return self.get_noise_item()
        data = torch.load(self.file_path[idx], weights_only=False)
        seq_len = data["embedding"].shape[0]
        mask = torch.zeros(MAX_LEN, dtype=torch.float32)
        mask[:seq_len] = 1
        cys_mask = torch.tensor(np.array(list(data["seq"])) == "C")
        cys_mask = torch.nn.functional.pad(
            cys_mask, (0, MAX_LEN - seq_len), "constant", False
        )
        emb = F.pad(
            torch.tensor(data["embedding"], dtype=torch.float32),
            (0, 0, 0, MAX_LEN - seq_len),
        )
        disulfid_res = torch.zeros(MAX_LEN, dtype=torch.float32)
        disulfid_res[data["disulfid_res"]] = 1
        return {
            "id": data["id"],
            "seq": data["seq"],
            "emb": emb,
            "mask": mask,
            "cys_mask": cys_mask,
            "disulfid_res": disulfid_res,
        }


"""
from dataset import *
from tqdm import tqdm
dataset = DisulfidEmbeddingDataset("./data/testing")
dataset = DisulfidEmbeddingDataset("./data/training")
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16, prefetch_factor=4)
for i in tqdm(loader):
    break
"""
