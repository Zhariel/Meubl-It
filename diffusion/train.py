import os
import numpy as np
import torch.nn.functional as F

from diffusion.model.unet import Unet
from dataset import ListDataset, gather_links, one_hot_labels, prepare_training_sample, load_images_and_labels
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import random

res = 64
device = "cuda:0" if torch.cuda.is_available() else "cpu"
T = 3

x_labels = ['chair', 'bookshelf', 'dresser', 'sofa', 'table']

LABEL_SHAPE = len(x_labels)
links = gather_links()

print("Extracting images")
# images, box_coords, labels = test_load_images_and_labels([], x_labels, res)
images, box_coords, labels = load_images_and_labels(links, x_labels, res)
tensor = transforms.ToTensor()
normalize = transforms.Lambda(lambda t: ((t / 255) * 2) - 1)
# normalize = transforms.Lambda(lambda t: t / 255)

print("Preparing samples")
x_list, y_list, mask_list, label_list = [], [], [], []
for sample in tqdm(zip(images, box_coords, labels)):
    image, box, labels = sample
    label = one_hot_labels(labels, np.random.choice(labels))
    prepare_training_sample(normalize(image), T, label, x_list, y_list, mask_list, label_list, *box)

learning_rate = 0.001
batch_size = 16
num_epochs = 10

print("Creating dataset")
dataset = ListDataset(x_list, mask_list, label_list, y_list, device)
dataset.shuffle()
dataloader = DataLoader(dataset, batch_size=batch_size)
model = Unet(LABEL_SHAPE, res).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()

print("Starting training")
for epoch in range(num_epochs):
    for i, (x, m, l, target) in enumerate(dataloader):
        outputs = model(x, m, l)
        # loss = criterion(outputs, target.permute(0, 3, 1, 2))
        loss = F.l1_loss(outputs, target.permute(0, 3, 1, 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

model = model.to("cpu")
torch.save(model.state_dict(), os.path.join('model', 'model2.pkl'))
