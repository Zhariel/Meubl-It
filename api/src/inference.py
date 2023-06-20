import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import base64
import math
import boto3
from PIL import Image
from io import BytesIO
import zlib
import datetime

from utils import load_from_s3

BATCH_SIZE = 1
T = 300
MODEL_DIVIDER = 1
device = "cpu"


def load_model_from_s3(LOGGER, bucket_name, model_key):
    model_file = load_from_s3(LOGGER, bucket_name, model_key)
    LOGGER.info("Load model from s3 bucket")
    model = SimpleUnet(DIVIDER=MODEL_DIVIDER)
    LOGGER.info("Load model state dict")
    model.load_state_dict(torch.load(model_file))
    LOGGER.info("Model eval")
    model.eval()
    return model


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

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self, DIVIDER=1):
        super().__init__()
        image_channels = 3
        down_channels = tuple(value // DIVIDER for value in (32, 64, 128, 256, 512))
        up_channels = tuple(value // DIVIDER for value in (512, 256, 128, 64, 32))
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

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

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        with torch.no_grad():
            for down in self.downs:
                x = down(x, t)
                residual_inputs.append(x)
            for up in self.ups:
                residual_x = residual_inputs.pop()
                # Add residual x as additional channels
                x = torch.cat((x, residual_x), dim=1)
                x = up(x, t)
        return self.output(x)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, xy1, xy2, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )

    noisy_image = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
                  + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

    masked_noisy_image = mask_image_tensor(x_0, noisy_image, *xy1, *xy2)

    return masked_noisy_image.to(device)


def mask_image_tensor(t1, t2, x1, y1, x2, y2):
    mask = torch.zeros_like(t1)

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


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def code_romain(T, model, img, x1, y1, x2, y2):
    box = (int(x1), int(y1), int(x2), int(y2))
    betas = linear_beta_schedule(timesteps=T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    IMG_SIZE = 64

    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    coord, newbox = crop_largest_square_around_point(*img.size, box, IMG_SIZE)
    image = img.crop(coord)

    input_tensor = preprocess(image).unsqueeze(0)

    t = torch.linspace(T - 1, T - 1, BATCH_SIZE, device=device).long()

    x_noisy = forward_diffusion_sample(input_tensor, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                       (box[0], box[1]), (box[2], box[3]), device="cpu")

    while T > 0:
        print(f"T = {T}")
        T -= 1
        t -= 1
        x_noisy = model(x_noisy, t)

    output_tensor = x_noisy.squeeze(0)
    output_tensor = output_tensor.permute(1, 2, 0)
    output_tensor = output_tensor.detach().cpu().numpy()

    output_tensor = (output_tensor * 255).astype(np.uint8)

    output_image = Image.fromarray(output_tensor)

    return output_image


def inference_model(LOGGER, encoded_img, selected_furniture, start_x_axis, end_x_axis, start_y_axis, end_y_axis,
                    bucket_name, model_key, unannotated_data_folder_key):
    LOGGER.info("Selected furniture: " + selected_furniture)
    LOGGER.info("start_x_axis: " + str(start_x_axis))
    LOGGER.info("end_x_axis: " + str(end_x_axis))
    LOGGER.info("start_y_axis: " + str(start_y_axis))
    LOGGER.info("end_y_axis: " + str(end_y_axis))

    LOGGER.info("Read model in s3 bucket")
    model = load_model_from_s3(LOGGER, bucket_name, model_key)

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
    output_image = code_romain(T, model, image, start_x_axis, start_y_axis, end_x_axis, end_y_axis)

    output_image_bytes = BytesIO()
    output_image.save(output_image_bytes, format='PNG')
    LOGGER.info("Compress the image from the inference")
    output_image_compress = zlib.compress(output_image_bytes.getvalue())
    LOGGER.info("Encode the image from the inference in base64")
    output_image_base64 = base64.b64encode(output_image_compress).decode('utf-8')

    return output_image_base64
