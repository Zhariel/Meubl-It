import torchvision
import numpy as np
from diffusion import load_env_variables
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import os
import torch


class ADE20kDataset(Dataset):
    def __init__(self, ):
        var = load_env_variables()

        self.annotations = None
        self.root_dir = var["data_path"]

    def __getitem__(self, index):
        pass


def custom_dataset(IMG_SIZE=64, train_split=80, paths_annot=[(os.path.join('assets', 'image.jpg'), 0)]):

    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ])

    dataset = DatasetFolder(root='', loader=lambda x: x, transform=data_transforms)
    dataset.samples = paths_annot
    dataset.classes = {label: label for _, label in paths_annot}

    test_split = 100 - train_split

    train_set, val_set = torch.utils.data.random_split(dataset, [train_split, test_split])
    return train_set, val_set

def load_transformed_dataset(IMG_SIZE):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root=".", download=True,
                                              transform=data_transform)

    test = torchvision.datasets.StanfordCars(root=".", download=True,
                                             transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])
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
