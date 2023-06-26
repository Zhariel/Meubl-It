from io import BytesIO
import zlib
import datetime
import random
import base64

from utils import load_from_s3

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import boto3
from PIL import Image

STEPS = 3
RESOLUTION_IMG = 64
x_labels = ['chair', 'bookshelf', 'dresser', 'sofa', 'table']


def load_model_from_s3(LOGGER, bucket_name, model_key, label_size):
    model_file = load_from_s3(LOGGER, bucket_name, model_key)
    LOGGER.info("Load model from s3 bucket")
    model = Unet(label_size, RESOLUTION_IMG)
    LOGGER.info("Load model state dict")
    model.load_state_dict(torch.load(model_file))
    LOGGER.info("Model eval")
    model.eval()
    return model


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
        labels = self.linear(labels).view(self.in_len, self.in_len, 1).unsqueeze(0)
        time = self.time(time).view(x.shape[0], self.in_len, self.in_len, 1)
        x = torch.cat((x, mask, labels, time), dim=3)
        x = x.permute(0, 3, 1, 2)
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        # with torch.no_grad:
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

    clone_x[y1:y2, x1:x2, :] = inter[step + 1, y1:y2, x1:x2, :]

    return torch.from_numpy(clone_x).float().unsqueeze(0), \
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


def inference(LOGGER, image, box, label, bucket_name, model_key):
    resize = transforms.Resize((RESOLUTION_IMG, RESOLUTION_IMG))
    coords, newcoords = crop_largest_square_around_point(*image.size, box, RESOLUTION_IMG)

    img = np.array(resize(image.crop(coords)))
    labels = [1 if label == elt else 0 for elt in x_labels]
    normalize = transforms.Lambda(lambda t: (t / 255))

    x, m, l = prepare_training_sample(normalize(img), labels, STEPS, *newcoords)

    LOGGER.info("Read model in s3 bucket")
    model = load_model_from_s3(LOGGER, bucket_name, model_key, len(labels))

    for i in range(STEPS, -1, -1):
        time = torch.tensor([i]).float()
        x = model(x, m, l, time).permute(0, 3, 2, 1)

    x = x.permute(0, 3, 2, 1)

    denormalize = transforms.Lambda(lambda t: (t * 255))

    pred = x[:, :, newcoords[1]:newcoords[3], newcoords[0]:newcoords[2]]

    w = abs(box[0] - box[2])
    h = abs(box[1] - box[3])

    LOGGER.info(f"denormalize(pred) shape : {denormalize(pred).shape}")
    LOGGER.info(f"w : {w}")
    LOGGER.info(f"h : {h}")

    upscale = nn.functional.interpolate(denormalize(pred), size=(h, w), mode="bilinear").squeeze().clone().detach().numpy()

    LOGGER.info(f"upscale shape : {upscale.shape}")
    output = np.array(image)
    LOGGER.info(f"output image shape : {output.shape}")
    LOGGER.info(f"transpose upscale shape : {np.transpose(upscale, (1, 2, 0)).shape}")
    output[box[1]:box[3], box[0]:box[2], :] = np.transpose(upscale, (1, 2, 0))
    output_image = Image.fromarray(output)

    return output_image


def inference_model_pipeline(LOGGER, encoded_img, selected_furniture, start_x_axis, start_y_axis, end_x_axis,
                             end_y_axis, bucket_name, model_key, unannotated_data_folder_key):
    LOGGER.info("Selected furniture: " + selected_furniture)
    LOGGER.info("start_x_axis: " + str(start_x_axis))
    LOGGER.info("start_y_axis: " + str(start_y_axis))
    LOGGER.info("end_x_axis: " + str(end_x_axis))
    LOGGER.info("end_y_axis: " + str(end_y_axis))

    LOGGER.info("Decode the image from base64")
    decoded_image = base64.b64decode(encoded_img)

    LOGGER.info("Decompress the image")
    decoded_image = zlib.decompress(decoded_image)

    LOGGER.info("Save encoded image in /unannotated S3 bucket")
    s3 = boto3.resource('s3')
    s3.Object(bucket_name, unannotated_data_folder_key + datetime.datetime.now().strftime("%Y%m%d") + "_" + str(
        np.random.randint(100000)) + '.png').put(Body=decoded_image)

    LOGGER.info("Convert the image in data stream")
    image_stream = BytesIO(decoded_image)
    image = Image.open(image_stream).convert('RGB')

    LOGGER.info("Inference")
    box = (int(float(start_x_axis)), int(float(start_y_axis)), int(float(end_x_axis)), int(float(end_y_axis)))
    output_image = inference(LOGGER=LOGGER, image=image, box=box, label=selected_furniture, bucket_name=bucket_name,
                             model_key=model_key)

    output_image_bytes = BytesIO()
    output_image.save(output_image_bytes, format='PNG')
    LOGGER.info("Compress the image from the inference")
    output_image_compress = zlib.compress(output_image_bytes.getvalue())
    LOGGER.info("Encode the image from the inference in base64")
    output_image_base64 = base64.b64encode(output_image_compress).decode('utf-8')

    return output_image_base64
