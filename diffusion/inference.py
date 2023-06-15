import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
import os

box = (0, 285, 440, 725)  # x1, y1, x2, y2
BATCH_SIZE = 1
T = 5
MODEL_DIVIDER = 1
device = "cuda"


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
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

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (32, 64, 128, 256, 512)
        up_channels = (512, 256, 128, 64, 32)
        out_dim = 3
        time_emb_dim = 32

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], \
                                          time_emb_dim) \
                                    for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], \
                                        time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x):
        # Initial conv
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


def mask_image_tensor(t1, t2, x1, y1, x2, y2):
    # Merges the boxed in t2 defined by the coordinates onto t1
    # Therefore t1 should be the original image and t2 the transformed image
    # Tensor(1, 3, X, X)
    mask = torch.zeros_like(t1, device=device)

    mask[:, :, y1 - 1:y2, x1 - 1:x2] = 1
    masked_t2 = t2 * mask
    t1[:, :, y1 - 1:y2, x1 - 1:x2] = 0

    return t1 + masked_t2


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

    return (left, top, right, bottom), (nleft, ntop, nright, nbottom)

IMG_SIZE = 64

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

# -------------------------------------------------------------------- #

MODEL_PATH = os.path.join('assets', 'model.pkl')
INPUT_IMAGE_PATH = os.path.join('assets', 'sample.png')
OUTPUT_IMAGE_PATH = os.path.join('assets', 'test4.png')

image = Image.open(INPUT_IMAGE_PATH).convert('RGB')
coord, newbox = crop_largest_square_around_point(*image.size, box, IMG_SIZE)
image = image.crop(coord)
image.show()
input_tensor = preprocess(image).unsqueeze(0)
input_tensor = input_tensor.to(device)

model = SimpleUnet().to(device)
torch.save(model.state_dict(), MODEL_PATH)

if os.path.isfile(MODEL_PATH):
    print('loading model')
    model.load_state_dict(torch.load(MODEL_PATH))


output_tensor = input_tensor.squeeze(0)
output_tensor = output_tensor.permute(1, 2, 0)
output_tensor = output_tensor.detach().cpu().numpy()

output_tensor = (output_tensor * 255).astype(np.uint8)

output_image = Image.fromarray(output_tensor)
output_image.show()

while T > 0:
    print(f"T = {T}")
    T -= 1
    t -= 1
    x_noisy = model(x_noisy)

output_tensor = input_tensor.squeeze(0)
output_tensor = output_tensor.permute(1, 2, 0)
output_tensor = output_tensor.detach().cpu().numpy()

output_tensor = (output_tensor * 255).astype(np.uint8)

output_image = Image.fromarray(output_tensor)
output_image.save(OUTPUT_IMAGE_PATH)
