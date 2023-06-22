from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import os


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super().__init__()
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Down or Upsample
        return self.transform(h)


class Unet(nn.Module):

    def __init__(self, labels_len, in_len):
        super().__init__()
        image_channels = 5  # RBG, mask, labels
        down_channels = (32, 64, 128, 256, 512)
        up_channels = (512, 256, 128, 64, 32)
        out_dim = 3
        self.in_len = in_len

        self.linear = nn.Linear(labels_len, in_len ** 2)

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1]) \
                                    for i in range(len(down_channels) - 1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], up=True) \
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, mask, labels):
        labels = self.linear(labels).view(self.in_len, self.in_len, 1).unsqueeze(0)
        x = torch.cat((x, mask, labels), dim=3)
        x = x.permute(0, 3, 1, 2)
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x)
        return self.output(x)


def prepare_training_sample(img, labels_list, steps, x1, y1, x2, y2):
    step = random.randint(0, steps)
    mask = np.zeros_like(img)
    mask[y1:y2 + 1, x1:x2 + 1, :] = 1
    noise = np.random.uniform(-1, 1, img.shape) / (steps + 1)

    clone = np.copy(img)
    clone[y1:y2, x1:x2, :] = 0

    return torch.from_numpy(np.copy(clone) + (noise * (step + 1) * mask)).float().unsqueeze(0), \
        torch.from_numpy(np.copy(clone) + (noise * step * mask)).float().unsqueeze(0), \
        torch.from_numpy(mask[:, :, 0:1]).float().unsqueeze(0), \
        torch.from_numpy(np.array(labels_list)).float()


def crop_largest_square_around_point(width, height, box, IMG_SIZE):
    box_side = abs(box[0] - box[2])
    point = (box[0] + box_side // 2, box[1] + box_side // 2)
    square_size = min(width, height)

    left = max(0, point[0] - square_size // 2)
    top = max(0, point[1] - square_size // 2)
    right = min(width, left + square_size)
    bottom = min(height, top + square_size)

    if right - left < square_size:
        left = max(0, right - square_size)
    if bottom - top < square_size:
        top = max(0, bottom - square_size)

    scale_factor = IMG_SIZE / square_size

    nleft = int((box[0] - left) * scale_factor)
    ntop = int((box[1] - top) * scale_factor)
    nright = int((box[2] - left) * scale_factor)
    nbottom = int((box[3] - top) * scale_factor)

    # returns coord after crop, coords after crop and resize
    return (left, top, right, bottom), (nleft, ntop, nright, nbottom)


steps = 3
resolution = 64
BATCH_SIZE = 4
MODEL_PATH = os.path.join('model', 'model.pt')
x_labels = ['chair', 'bookshelf', 'dresser', 'sofa', 'table']
resize = transforms.Resize((resolution, resolution))

from PIL import Image  # Remove this

img = Image.open(os.path.join('assets', 'adesample', 'a.jpg'))
box = (50, 50, 250, 250)
coords, newcoords = crop_largest_square_around_point(*img.size, box, resolution)

img = np.array(resize(img.crop(coords)))

label = "chair"
labels = [1 if label == elt else 0 for elt in x_labels]

# samples = [prepare_training_sample(img, label, steps, *coords[1]) for img, label, coords in zip(imges, labels, coordinates)]
normalize = transforms.Lambda(lambda t: (t * 2) - 1)
x, y, m, l = prepare_training_sample(normalize(img), labels, steps, *coords)

model = Unet(len(labels), resolution)
model.load_state_dict(torch.load(MODEL_PATH))

output = model(x, m, l)

print(output.shape)
import matplotlib.pyplot as plt

output_tensor = output.detach().cpu().numpy()

output_tensor = (output_tensor * 255).astype(np.uint8)

output_image = Image.fromarray(output_tensor)
output_image.show()
print()
