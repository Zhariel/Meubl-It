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
        image_channels = 6  # RBG, mask, labels, time
        down_channels = (32, 64, 128, 256, 512)
        up_channels = (512, 256, 128, 64, 32)
        out_dim = 3
        self.in_len = in_len

        self.linear = nn.Linear(labels_len, in_len ** 2)
        self.time = nn.Linear(1, in_len ** 2)

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1]) \
                                    for i in range(len(down_channels) - 1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], up=True) \
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, mask, labels, time):
        labels = self.linear(labels).view(x.shape[0], self.in_len, self.in_len, 1)
        time = self.time(time).view(x.shape[0], self.in_len, self.in_len, 1)
        x = torch.cat((x, mask, labels, time), dim=3)
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
    step = random.randint(0, steps - 1)
    mask = np.zeros_like(img)
    mask[y1:y2 + 1, x1:x2 + 1, :] = 1
    noise = np.random.randint(0, 256, size=img.shape)
    inter = np.linspace(img, noise, steps + 1)
    clone_x = np.copy(img)
    clone_y = np.copy(img)

    clone_x[y1:y2, x1:x2, :] = inter[step + 1, y1:y2, x1:x2, :]
    clone_y[y1:y2, x1:x2, :] = inter[step, y1:y2, x1:x2, :]

    return torch.from_numpy(clone_x).float().unsqueeze(0), \
        torch.from_numpy(clone_y).float().unsqueeze(0), \
        torch.from_numpy(mask[:, :, 0:1]).float().unsqueeze(0), \
        torch.from_numpy(np.array(labels_list)).float(), \
        torch.tensor([step]).float()


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

imges = [Image.open(os.path.join('assets', 'adesample', 'a.jpg'))] * BATCH_SIZE  ######################################
boxes = [(50, 50, 250, 250)] * BATCH_SIZE  ############################################################################
coordinates = [crop_largest_square_around_point(*i.size, b, resolution) for i, b in zip(imges, boxes)]

imges = [np.array(resize(i.crop(coords[0]))) for i, coords in zip(imges, coordinates)]

labels = ["chair"] * BATCH_SIZE
labels = [[1 if l == elt else 0 for elt in x_labels] for l in labels]

normalize = transforms.Lambda(lambda t: ((t / 255) * 2) - 1)
samples = [prepare_training_sample(normalize(img), label, steps, *coords[1]) for img, label, coords in
           zip(imges, labels, coordinates)]

learning_rate = 0.001
batch_size = 1
num_epochs = 10

model = Unet(len(labels[0]), resolution)
model.load_state_dict(torch.load(MODEL_PATH))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

x = torch.cat([s[0] for s in samples])
y = torch.cat([s[1] for s in samples])
m = torch.cat([s[2] for s in samples])
l = torch.stack([s[3] for s in samples])
t = torch.stack([s[4] for s in samples])

outputs = model(x, m, l, t)
loss = criterion(outputs, y.permute(0, 3, 1, 2))

optimizer.zero_grad()
loss.backward()
optimizer.step()

torch.save(model.state_dict(), MODEL_PATH)
