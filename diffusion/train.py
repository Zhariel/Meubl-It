import os
import numpy as np

from diffusion.dataset import custom_dataset
from torch.utils.data import DataLoader
from diffusion.model.unet import SimpleUnet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from numba import jit, njit
from torch.optim import Adam

IMAGE_PATH = os.path.join('..', 'assets', 'image.jpg')
IMG_SIZE = 64
BATCH_SIZE = 128
device = "cpu" if torch.cuda.is_available() else "cpu"

T = 10
res = 8


class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_image, target_image = self.data[index]
        return input_image, target_image


# @jit
def prepare_training_sample(img, steps, labels, x1, y1, x2, y2):
    lis = []
    mask = np.zeros_like(img)
    mask[y1:y2 + 1, x1:x2 + 1, :] = 1

    for i in range(steps - 1, -1, -1):
        clone = np.copy(img)
        noise_mask = np.random.uniform(-1, 1, img.shape) * mask
        clone[y1:y2, x1:x2, :] = 0
        x_flat = np.reshape(clone + noise_mask, (-1))
        flat_mask = np.reshape(mask[:, :, 0], (-1))
        lis.append((clone + noise_mask, mask[:, :, 0:1], labels))
        # lis.append((np.concatenate((x_flat, flat_mask, labels)), img))
        break
    return lis


# @jit
def one_hot_labels(label_list, selected):
    return np.array([1 if selected == elt else 0 for elt in label_list])


box_examples = [
    [2, 2, 4, 4],
    [0, 0, 4, 4],
    [1, 1, 3, 3],
    [4, 4, 8, 8],
    [4, 4, 6, 6],
]

x_labels = [
    'chair',
    'bookshelf',
    'dresser',
    'sofa',
    'table',
]

sample = Image.open(os.path.join('assets', 'sample.png')).resize((res, res))

arr = np.array(sample)
sample_labels = one_hot_labels(x_labels, np.random.choice(x_labels))

# x = [None] * len(box_examples)
# for i, b in enumerate(box_examples):
#     label = one_hot_labels(x_labels, np.random.choice(x_labels))
#     x[i] = prepare_training_sample(arr, T, label, *b)

x = []
for b in box_examples:
    label = one_hot_labels(x_labels, np.random.choice(x_labels))
    x.append(*prepare_training_sample(arr, T, label, *b))

im = torch.from_numpy(x[0][0])
mask = torch.from_numpy(x[0][1])
label = torch.tensor(x[0][2], dtype=torch.float32)


lbltensor = torch.randn(64)
lbltensor = lbltensor.view(8, 8, 1)
concatenated_tensor = torch.cat((im, mask, lbltensor), dim=2)

print(concatenated_tensor.shape)

model = SimpleUnet(label.shape[0], res)
