import logging
from random import random

import numpy as np
import boto3
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

from torch import optim

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

bucket_name = os.environ['BUCKET_NAME']
model_key = os.environ['MODEL_KEY']
annotated_data_folder_key = os.environ['ANNOTATED_DATA_FOLDER_KEY']
trained_data_folder_key = os.environ['TRAINED_DATA_FOLDER_KEY']
steps = 3
resolution = 64
learning_rate = 0.001
BATCH_SIZE = 4
resize = transforms.Resize((resolution, resolution))

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
        labels = self.linear(labels).view(x.shape[0], self.in_len, self.in_len, 1)
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
def load_from_s3(LOGGER, bucket_name, key_file):
    s3 = boto3.client('s3')
    LOGGER.info(f"Get {key_file} from {bucket_name} S3 bucket")
    response = s3.get_object(Bucket=bucket_name, Key=key_file)
    file_bytes = response['Body'].read()
    return file_bytes
def load_model_from_s3(bucket_name, model_key, labels_len, resolution):
    s3 = boto3.client('s3')
    LOGGER.info(f"Get {model_key} from {bucket_name} S3 bucket")
    response = s3.get_object(Bucket=bucket_name, Key=model_key)
    model_bytes = response['Body'].read()
    model_file = BytesIO(model_bytes)
    LOGGER.info("Load model from s3 bucket")
    model = Unet(labels_len, resolution)
    LOGGER.info("Load model state dict")
    model.load_state_dict(torch.load(model_file))
    # LOGGER.info("Model eval")
    # model.eval()
    return model
def save_model_from_s3(bucket_name, model_key, model):
    s3 = boto3.client('s3')
    LOGGER.info(f"Save {model_key} to {bucket_name} S3 bucket")
    model_file = BytesIO()
    torch.save(model.state_dict(), model_file)
    model_file.seek(0)
    s3.put_object(Bucket=bucket_name, Key=model_key, Body=model_file)
    return "Success - Model saved to S3 bucket"
def preprocess(box_array, label_array, images_array, x_labels):
    coordinates = [crop_largest_square_around_point(*i.size, b, resolution) for i, b in zip(images_array, box_array)]
    images = [np.array(resize(i.crop(coords[0]))) for i, coords in zip(images_array, coordinates)]

    labels = [[1 if l == elt else 0 for elt in x_labels] for l in label_array]
    samples = [prepare_training_sample(img, label, steps, *coords[1]) for img, label, coords in
               zip(images, labels, coordinates)]

    return samples, len(labels[0])

def retrain_model(model, samples, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    x = torch.cat([s[0] for s in samples])
    y = torch.cat([s[1] for s in samples])
    m = torch.cat([s[2] for s in samples])
    l = torch.stack([s[3] for s in samples])
    outputs = model(x, m, l)
    loss = criterion(outputs, y.permute(0, 3, 1, 2))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return save_model_from_s3(bucket_name, model_key, model)
def fake_retrain():
    LOGGER.info("Fake retrain")
    s3 = boto3.resource('s3')
    s3.Object(bucket_name, annotated_data_folder_key).copy_from(CopySource=f"{bucket_name}/{trained_data_folder_key}")
    s3.Object(bucket_name, annotated_data_folder_key).delete()
    return True


def handler(event, context):
    LOGGER.info("Event: " + str(event))

    s3 = boto3.client('s3')
    LOGGER.info(f"Get list of images from {bucket_name}/{annotated_data_folder_key} S3 bucket")
    response = s3.list_objects(Bucket=bucket_name, Prefix=annotated_data_folder_key)
    images = [obj['Key'] for obj in response['Contents']]
    LOGGER.info("List images: " + str(images))
    nb_data = (len(images) - 1) / 2
    if nb_data < BATCH_SIZE:
        LOGGER.info(f"Number of images in {bucket_name}/{annotated_data_folder_key} S3 bucket is less than batch size")
        return {"statusCode": 200, "body": "Number of images in S3 bucket is less than batch size"}
    LOGGER.info(f"Number of images in {bucket_name}/{annotated_data_folder_key} S3 bucket is {nb_data}")

    box_array = []
    label_array = []
    images_array = []

    for i in range(1, len(images), 2):
        img_file = load_from_s3(LOGGER, bucket_name, images[i])
        image_stream = BytesIO(img_file)
        image = Image.open(image_stream).convert('RGB')

        metadata_file = load_from_s3(LOGGER, bucket_name, images[i + 1]).decode('utf-8')
        metadata = json.loads(metadata_file)
        LOGGER.info("metadata json: " + str(metadata))
        box = (
            int(float(metadata["start_x_axis"])),
            int(float(metadata["end_x_axis"])),
            int(float(metadata["start_y_axis"])),
            int(float(metadata["end_y_axis"]))
        )

        images_array.append(image)
        label_array.append(metadata["selected_furniture"])
        box_array.append(box)

    LOGGER.info("Box array: " + str(box_array))
    LOGGER.info("Label array: " + str(label_array))
    LOGGER.info("Images array: " + str(images_array))

    x_labels = ['chair', 'bookshelf', 'dresser', 'sofa', 'table']

    samples, len_label = preprocess(box_array, label_array, images_array, x_labels)
    model = load_model_from_s3(bucket_name, model_key, len_label, resolution)

    if retrain_model(model, samples, learning_rate) == "Success - Model saved to S3 bucket":
        LOGGER.info("Retrained")

    return {"statusCode": 200, "body": "fake retrained"}

