from torchvision import transforms
import torchvision
import numpy as np
from diffusion import load_env_variables
from torch.utils.data import Dataset, DataLoader

import os
import torch


class ADE20kDataset(Dataset):
    def __init__(self, ):
        var = load_env_variables()

        self.annotations = None
        self.root_dir = var["data_path"]

    def __getitem__(self, index):
        pass

# def load_transformed_dataset(IMG_SIZE):
#     data_transforms = [
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),  # Scales data into [0,1]
#         transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
#     ]
#     data_transform = transforms.Compose(data_transforms)
#
#     train = torchvision.datasets.StanfordCars(root=".", download=True,
#                                               transform=data_transform)
#
#     test = torchvision.datasets.StanfordCars(root=".", download=True,
#                                              transform=data_transform, split='test')
#     return torch.utils.data.ConcatDataset([train, test])
#
#
# def show_tensor_image(image, plt):
#     reverse_transforms = transforms.Compose([
#         transforms.Lambda(lambda t: (t + 1) / 2),
#         transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
#         transforms.Lambda(lambda t: t * 255.),
#         transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
#         transforms.ToPILImage(),
#     ])
#
#     # Take first image of batch
#     if len(image.shape) == 4:
#         image = image[0, :, :, :]
#     plt.imshow(reverse_transforms(image))
