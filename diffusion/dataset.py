
from diffusion import load_env_variables
from torch.utils.data import Dataset, DataLoader

import os
import torch



class ADE20kDataset(Dataset):
    def __init__(self,):
        var = load_env_variables()

        self.annotations = None
        self.root_dir = var["data_path"]

    def __getitem__(self, index):
        pass