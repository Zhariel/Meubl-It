import os
import numpy as np
import torch.nn.functional as F

from diffusion.model.unet import Unet
from dataset import ListDataset, gather_links, one_hot_labels, prepare_training_sample, load_images_and_labels
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

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
links += gather_links(folder="home_or_hotel")[:250]

print("Extracting images")
# images, box_coords, labels = test_load_images_and_labels([], x_labels, res)
images, box_coords, labels = load_images_and_labels(links, x_labels, res)
tensor = transforms.ToTensor()
# normalize = transforms.Lambda(lambda t: ((t / 255) * 2) - 1)
normalize = transforms.Lambda(lambda t: t / 255)

print("Preparing samples")
x_list, y_list, mask_list, label_list, time_list = [], [], [], [], []
for sample in tqdm(zip(images, box_coords, labels)):
    image, box, labels = sample
    label = one_hot_labels(labels, np.random.choice(labels))
    prepare_training_sample(image, T, label, normalize, x_list, y_list, mask_list, label_list, time_list, *box)

learning_rate = 0.001
batch_size = 128
num_epochs = 100

print("Creating dataset")
dataset = ListDataset(x_list, mask_list, label_list, y_list, time_list, device)
dataset.shuffle()
dataloader = DataLoader(dataset, batch_size=batch_size)
model = Unet(LABEL_SHAPE, res).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter()

print("Starting training")
for epoch in range(num_epochs):
    for i, (x, m, l, target, time) in enumerate(dataloader):
        outputs = model(x, m, l, time)
        loss = F.l1_loss(outputs, target.permute(0, 3, 1, 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss', loss.item(), epoch)

        if i % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

writer.close()
model = model.to("cpu")
name = writer.get_logdir()
name = name.replace('runs\\', '')
name = name.rsplit('_', 1)[0]

torch.save(model.state_dict(), os.path.join('model', '0-1' + name + '.pkl'))
