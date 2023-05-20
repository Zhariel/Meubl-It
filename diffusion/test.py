import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from diffusion.unet import UNet

IMAGE_PATH = os.path.join('assets', 'image.jpg')
IMG_SIZE = 64
BATCH_SIZE = 128

# def load_transformed_dataset():
#     data_transforms = [
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(), # Scales data into [0,1]
#         transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
#     ]
#     data_transform = transforms.Compose(data_transforms)
#
#     train = torchvision.datasets.StanfordCars(root=".", download=True,
#                                          transform=data_transform)
#
#     test = torchvision.datasets.StanfordCars(root=".", download=True,
#                                          transform=data_transform, split='test')
#     return torch.utils.data.ConcatDataset([train, test])
# def show_tensor_image(image):
#     reverse_transforms = transforms.Compose([
#         transforms.Lambda(lambda t: (t + 1) / 2),
#         transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#         transforms.Lambda(lambda t: t * 255.),
#         transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
#         transforms.ToPILImage(),
#     ])
#
#     # Take first image of batch
#     if len(image.shape) == 4:
#         image = image[0, :, :, :]
#     plt.imshow(reverse_transforms(image))
#
# data = load_transformed_dataset()
# dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = UNet()

input_image = Image.open(IMAGE_PATH).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_tensor = preprocess(input_image).unsqueeze(0)



unet = UNet()
























